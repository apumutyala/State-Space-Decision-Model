"""CPU hill-climb over cached frozen-Q9 features: train proxy heads, score them with the REAL harness, apply the
pre-registered rule (train/proxy_rule.py). Deterministic and seeded; no GPU, no backbone.

    python -X utf8 -m train.proxy_loop recipes --cache <dir> --dev-rows <dev-rows.jsonl.gz> --out <dir> \
        --clef <clef-flash results.jsonl> --stock <stock Q9 results.jsonl (SELECT)> [--only R01,R11] [--B 2000]
    python -X utf8 -m train.proxy_loop holdout --cache <dir> --dev-rows ... --out <dir> --clef ... [--recipe Rxx]
    python -X utf8 -m train.proxy_loop stock-check --cache <dir> --dev-rows ... --stock <job-5cmat results>

``<cache>/<arm>/<split>/`` are the job's shard directories (``jobs.q9_feature_cache.extract.load_shards``; every shard
re-hashed against MANIFEST.jsonl on load). Scoring = ``jobs.common.dev_subset`` (kit decomposition, cluster bootstrap,
chance-corrected skills, unanswered = wrong): a request with any unanswerable question is written with status
"error" and scores zero, as the stock readout's errored rows do.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

from . import proxy_rule as R
from .losses import LossConfig, decision_loss
from .proxy_heads import ProxyRequest, describe, from_record, make_proxy_head

ROOT = Path(__file__).resolve().parents[1]
#: training budget for EVERY recipe (pre-registered alongside the recipes; no early stopping on DEV)
TRAIN = {"epochs": 3, "grad_accum": 8, "warmup_steps": 50, "weight_decay": 0.01, "max_grad_norm": 1.0,
         "lr": {"fork": 1e-3, "mixer": 1e-3, "attn_ref": 3e-4}}
LOSSES = {"ce": LossConfig(arm="ce"), "clef": LossConfig(arm="clef"), "clef_norps": LossConfig(arm="clef", rps_weight=0.0)}


# ----------------------------------------------------------------------------------------------- data
def load_split(cache: str | Path, arm: str, split: str, verify: bool = True) -> list[ProxyRequest]:
    from jobs.q9_feature_cache.extract import load_shards

    return [from_record(r) for r in load_shards(Path(cache) / arm / split, verify)]


def d_model_of(reqs: list[ProxyRequest]) -> int:
    for r in reqs:
        for q in r.questions:
            return int(q.fork.shape[-1])
    raise ValueError("no cached questions")


# ----------------------------------------------------------------------------------------------- training
def _order(seed: int, epoch: int, n: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)  # == train.proxy._order / Trainer._order
    return torch.randperm(n, generator=g).tolist()


def train_head(recipe: dict, train: list[ProxyRequest], seed: int, budget: dict = TRAIN) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    head = make_proxy_head(recipe, d_model_of(train), seed)
    loss_cfg = LOSSES[recipe["loss"]]
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=budget["lr"][recipe["family"]], weight_decay=budget["weight_decay"])
    warm = budget["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
    head.train()
    hist, steps, t0 = [], 0, time.perf_counter()
    acc, total, nq = budget["grad_accum"], 0.0, 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(budget["epochs"]):
        for j, i in enumerate(_order(seed, epoch, len(train))):
            req = train[i]
            lps = head(req)
            loss = sum(decision_loss(lp.unsqueeze(0).float(), torch.tensor([q.label]), [q.qtype], loss_cfg)
                       for lp, q in zip(lps, req.questions)) / max(len(lps), 1)
            (loss / acc).backward()
            total += float(loss.detach())
            nq += 1
            if nq % acc == 0:
                torch.nn.utils.clip_grad_norm_(params, budget["max_grad_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                hist.append(total / acc)
                total, steps = 0.0, steps + 1
    head.eval()
    return head, {"steps": steps, "loss_first": hist[:5], "loss_last": hist[-5:], "seconds": time.perf_counter() - t0,
                  "train_requests": len(train), "train_questions": sum(len(r.questions) for r in train)}


@torch.no_grad()
def nll(head, reqs: list[ProxyRequest]) -> dict:
    s, n, acc = 0.0, 0, 0
    for r in reqs:
        for lp, q in zip(head(r), r.questions):
            if q.label < 0:
                continue
            s -= float(lp[q.label])
            acc += int(int(lp.argmax()) == q.label)
            n += 1
    return {"nll": s / max(n, 1), "acc": acc / max(n, 1), "n": n}


@torch.no_grad()
def predict(head, reqs: list[ProxyRequest]) -> dict[str, dict[str, list[float]]]:
    """rid -> qid -> log-probabilities in the cached (canonical, original-id) option order."""
    return {r.rid: {q.qid: lp.float().tolist() for q, lp in zip(r.questions, head(r))} for r in reqs}


def stock_predictions(reqs: list[ProxyRequest]) -> dict[str, dict[str, list[float]]]:
    """The stock Q9 readout recomputed from the cache: softmax over options of the summed continuation log-likelihood
    (identity option-count temperature, as jobs.common.readout_engine serves it)."""
    return {r.rid: {q.qid: torch.log_softmax(q.lm_lp.float(), -1).tolist() for q in r.questions} for r in reqs}


# ----------------------------------------------------------------------------------------------- results + scoring
def _dev_rows(dev_rows: str, ids: set[str]) -> dict[str, dict]:
    out = {}
    with gzip.open(dev_rows, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["_evaluation"]["run_id"] in ids:
                out[r["_evaluation"]["run_id"]] = r
    return out


def write_results(path: str | Path, dev_rows: dict[str, dict], reqs: list[ProxyRequest],
                  preds: dict[str, dict[str, list[float]]], model: str) -> dict:
    """Kit results rows. A request whose questions are not ALL predicted (unanswerable / skipped) -> status "error"."""
    from decide.schema import wire_answer

    by = {r.rid: r for r in reqs}
    n_ok = n_err = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rid, row in dev_rows.items():
            ev = row["_evaluation"]
            rec = {"run_id": rid, "catalog_id": ev["catalog_id"]}
            r, p = by.get(rid), preds.get(rid, {})
            if r is None or r.skipped or set(p) != set(row["questions"]):
                rec.update(status="error", error={"code": "proxy_unanswerable",
                                                  "message": json.dumps((r.skipped if r else {"request": "missing"}))[:300]})
                n_err += 1
            else:
                answers = {}
                for q in r.questions:
                    probs = {o: float(torch.tensor(v).exp()) for o, v in zip(q.option_ids, p[q.qid])}
                    answers[q.qid] = wire_answer(row["questions"][q.qid], probs)
                rec.update(status="ok", response={"model": model, "answers": answers})
                n_ok += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": n_ok, "error": n_err}


def subset_file(base: str | Path, keep, out: str | Path) -> str:
    """A dev_subset-format file holding the base subset's run ids filtered by ``keep(run_id, catalog_id)``."""
    from jobs.common import dev_subset as D

    d = json.loads(Path(base).read_text(encoding="utf-8"))
    cat = {e["run_id"]: e["catalog_id"] for e in D.read_manifest()}
    ids = sorted(i for i in d["run_ids"] if keep(i, cat[i]))
    d.update(run_ids=ids, ids_sha256=D.ids_sha256(ids))
    Path(out).write_text(json.dumps(d), encoding="utf-8")
    return str(out)


def score(results: str, subset: str, B: int = R.BOOTSTRAP_B, seed: int = R.BOOTSTRAP_SEED) -> dict:
    from jobs.common import dev_subset as D

    return D.cmd_score(Namespace(results=results, B=B, seed=seed), D.load_subset(subset))


def compare(cand: str, base: str, subset: str, B: int = R.BOOTSTRAP_B, seed: int = R.BOOTSTRAP_SEED) -> dict:
    from jobs.common import dev_subset as D

    out = D.cmd_compare(Namespace(candidate=cand, baseline=base, B=B, seed=seed), D.load_subset(subset))
    return {"diff": out["diff"]["index"], "candidate": out.get("candidate", {}).get("index"),
            "baseline": out.get("baseline", {}).get("index")}


def report(results: str, split_subset: str, out_dir: Path, clef: str | None, stock: str | None, B: int) -> dict:
    """Index point + CI, paired diffs vs Clef-flash / stock Q9, and the IN/NEW-domain breakdown."""
    rep = {"all": score(results, split_subset, B)["slice_index"]["index"]}
    for name, keep in (("in_domain", lambda i, c: R.in_domain(c)), ("new_domain", lambda i, c: not R.in_domain(c))):
        sf = subset_file(split_subset, keep, out_dir / f"subset_{name}.json")
        rep[name] = score(results, sf, B)["slice_index"]["index"]
        if clef:
            rep[f"{name}_vs_clef"] = compare(results, clef, sf, B)["diff"]
    if clef:
        rep["vs_clef_flash"] = compare(results, clef, split_subset, B)["diff"]
    if stock:
        rep["vs_stock_q9"] = compare(results, stock, split_subset, B)["diff"]
    return rep


# ----------------------------------------------------------------------------------------------- runs
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_recipe(recipe: dict, cache: str, dev_rows: str, out: Path, split: str, subset: str, seed: int,
               clef: str | None, stock: str | None, B: int, budget: dict = TRAIN) -> dict:
    arm = recipe["arm"]
    train = load_split(cache, arm, "train")
    head, tr = train_head(recipe, train, seed, budget)
    reqs = load_split(cache, arm, split)
    ids = set(json.loads((ROOT / subset).read_text(encoding="utf-8"))["run_ids"]) if not Path(subset).is_absolute() else \
        set(json.loads(Path(subset).read_text(encoding="utf-8"))["run_ids"])
    rows = _dev_rows(dev_rows, ids)
    d = out / f"{recipe['id']}_s{seed}_{split}"
    d.mkdir(parents=True, exist_ok=True)
    res = d / "results.jsonl"
    counts = write_results(res, rows, reqs, predict(head, reqs), f"proxy-{recipe['id']}")
    counts.update(rows=len(rows), cached=len({r.rid for r in reqs} & set(rows)))
    if stock == "from-cache":  # the stock Q9 readout recomputed from this split's cached option log-likelihoods
        stock = str(d / "stock_from_cache.jsonl")
        write_results(stock, rows, reqs, stock_predictions(reqs), "stock-from-cache")
    rec = {"recipe": recipe, "seed": seed, "split": split, "head": describe(head), "train": tr,
           "train_fit": nll(head, train[: min(len(train), 2000)]), "results": str(res), "results_sha256": _sha(res),
           "counts": counts, "rule": R.RULE_VERSION, "budget": budget,
           "score": report(str(res), str(ROOT / subset) if not Path(subset).is_absolute() else subset, d, clef, stock, B)}
    (d / "receipt.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train.proxy_loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("recipes", "holdout", "stock-check"):
        s = sub.add_parser(name)
        s.add_argument("--cache", required=True)
        s.add_argument("--dev-rows", required=True)
        s.add_argument("--out", default="proxy-out")
        s.add_argument("--clef", default=None)
        s.add_argument("--stock", default=None)
        s.add_argument("--B", type=int, default=R.BOOTSTRAP_B)
        s.add_argument("--only", default=None)
        s.add_argument("--recipe", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recipes = {r["id"]: dict(r) for r in R.RECIPES}
    if a.cmd == "stock-check":
        reqs = load_split(a.cache, "stock", "select")
        ids = set(json.loads((ROOT / R.SELECT_SUBSET).read_text(encoding="utf-8"))["run_ids"])
        res = out / "stock_from_cache_select.jsonl"
        write_results(res, _dev_rows(a.dev_rows, ids), reqs, stock_predictions(reqs), "stock-from-cache")
        rep = {"from_cache": score(str(res), str(ROOT / R.SELECT_SUBSET), a.B)["slice_index"]["index"],
               "vs_job_5cmat": compare(str(res), a.stock, str(ROOT / R.SELECT_SUBSET), a.B) if a.stock else None}
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "recipes":
        ids = a.only.split(",") if a.only else list(recipes)
        rows = []
        for rid in ids:
            rec = run_recipe(recipes[rid], a.cache, a.dev_rows, out, "select", R.SELECT_SUBSET, R.SELECT_SEEDS[0],
                             a.clef, a.stock, a.B)
            rows.append({"id": rid, "family": recipes[rid]["family"], "select": rec["score"]["all"]["point"],
                         "ci": [rec["score"]["all"]["lo"], rec["score"]["all"]["hi"]]})
            print(json.dumps(rows[-1]), flush=True)
        chosen = R.choose({r["id"]: r["select"] for r in rows})
        (out / "select_summary.json").write_text(json.dumps({"rows": rows, "chosen": chosen, "rule": R.RULE_VERSION},
                                                            indent=1), encoding="utf-8")
        return 0
    chosen = a.recipe or json.loads((out / "select_summary.json").read_text(encoding="utf-8"))["chosen"]
    recs = [run_recipe(recipes[chosen], a.cache, a.dev_rows, out, "holdout", R.HOLDOUT_SUBSET, s, a.clef,
                       "from-cache", a.B) for s in R.HOLDOUT_SEEDS]
    c = recs[0]["counts"]
    v = R.verdict(chosen, recs[0]["score"]["all"]["point"], holdout_complete=c["cached"] == c["rows"])
    v["holdout_seeds"] = {r["seed"]: r["score"]["all"]["point"] for r in recs}
    (out / "verdict.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CPU hill-climb over cached frozen-Q9 features: train proxy heads, score them with the REAL harness, apply the
pre-registered rule (train/proxy_rule.py, proxy-rule-v1 + A1). Deterministic and seeded; no GPU, no backbone.

Order (each step refuses to run out of order):
    python -X utf8 -m train.proxy_loop stock-check --cache <dir> --dev-rows <dev-rows.jsonl.gz> --out <dir> --stock <job-5cmat results>
    python -X utf8 -m train.proxy_loop recipes --cache ... --dev-rows ... --out ... --clef <clef-flash results> --stock ...
        [--only R01,R11]       # --only: exploratory receipts; NEVER writes select_summary.json
    python -X utf8 -m train.proxy_loop select  --cache ... --out ...     # select_summary.json from ALL candidate receipts
    python -X utf8 -m train.proxy_loop holdout --cache ... --dev-rows ... --out ... --clef ...   # SINGLE SHOT

``<cache>/<arm>/<split>/`` are the job's shard directories (every shard re-hashed against MANIFEST.jsonl on load) and
``<cache>/<arm>/lex_vocab_<part>.pt`` the part's lexical table (sha256 checked against summary_<part>.json). Several
job parts (e.g. stock select+holdout and stock train) unpack into one cache directory. Scoring =
``jobs.common.dev_subset`` (kit decomposition, cluster bootstrap, chance-corrected skills, unanswered = wrong): a request
with any unanswerable question is written with status "error" and scores zero, as the stock readout's errored rows do.
Verdict edition 0.2.1; every score is also reported under 0.3 (A1(c)).

Single-shot HOLDOUT (review B2, re-review N1 / N2):
  * the choice is RE-DERIVED: ``chosen == proxy_rule.choose(rows)`` over select_summary's rows, and every row's
    ``results_sha256`` / ``head_sha256`` must equal its SELECT receipt AND the files on disk; select_summary records a
    code identity (sha256 of the proxy loop's own modules) that must still match the code that runs the holdout;
  * select_summary must cover EVERY candidate (R01-R15) trained with one budget and one cache identity (which must still
    match the cache on disk); ``--recipe`` other than the choice is refused;
  * every input (``--dev-rows`` and ``--clef`` are REQUIRED) is validated BEFORE any training or directory creation: the
    dev-rows file holds every HOLDOUT id, the Clef-flash results cover every HOLDOUT id, the cached HOLDOUT split exists;
  * the attempt is then bound atomically in ``holdout_attempt.json`` (choice, select_summary sha256, input sha256s,
    code identity). Each seed's receipt directory is built under a temporary name and renamed into place only when its
    receipt is complete, so a crash never leaves a half receipt. Re-running ``holdout`` after a crash RESUMES the same
    attempt (same choice, same inputs; completed seeds are kept, missing ones are run) and cannot re-choose; a
    ``*_holdout`` directory without an attempt file, a changed input, or an existing verdict.json is refused;
  * the split counts complete only if cached == len(ids) == rows over the HOLDOUT id list.
Procedure (once-only record): commit select_summary.json, holdout_attempt.json and verdict.json to the results branch
``stack/q9/proxy-head`` (``receipts/<date>-D-q9-proxy-holdout/``) as soon as each is written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

from . import proxy_rule as R
from .losses import LossConfig, decision_loss
from .proxy_heads import ProxyRequest, describe, from_record, make_proxy_head

ROOT = Path(__file__).resolve().parents[1]
#: training budget for EVERY recipe (A1(b): fixed optimiser steps, every arm; no early stopping on DEV)
TRAIN = {"steps": R.TRAIN_STEPS, "grad_accum": R.GRAD_ACCUM, "warmup_steps": 50, "weight_decay": 0.01,
         "max_grad_norm": 1.0, "lr": {"fork": 1e-3, "mixer": 1e-3, "attn_ref": 3e-4}}
LOSSES = {"ce": LossConfig(arm="ce"), "clef": LossConfig(arm="clef"), "clef_norps": LossConfig(arm="clef", rps_weight=0.0)}
#: A1(b): conv1 / e5 recipes are compared only against these matched-data, matched-steps stock controls
MATCHED_CONTROLS = {"R09": "R14", "R10": "R14", "R13": "R15"}


class OutOfOrder(RuntimeError):
    """A step of the pre-registered sequence was attempted out of order (or a second time)."""


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_sha(budget: dict) -> str:
    return hashlib.sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()


#: the modules whose code decides a proxy score (re-review N1: the holdout must run the code SELECT ran)
CODE_FILES = ("train/proxy_loop.py", "train/proxy_heads.py", "train/proxy_rule.py", "train/losses.py",
              "heads/heads.py", "heads/typed.py", "jobs/q9_feature_cache/extract.py")


def code_identity() -> dict:
    """sha256 of each file in CODE_FILES, with CRLF normalised to LF (a Windows checkout scores the same code)."""
    return {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in CODE_FILES}


# ----------------------------------------------------------------------------------------------- data
def lex_table(cache: str | Path, arm: str, split: str) -> dict | None:
    """The lexical table of the part that holds ``split`` (named in that part's summary_<part>.json), sha256-checked."""
    for s in sorted((Path(cache) / arm).glob("summary_*.json")):
        lv = json.loads(s.read_text(encoding="utf-8")).get("lex_vocab") or {}
        if split in (lv.get("splits") or []):
            p = Path(cache) / arm / lv["file"]
            if _sha(p) != lv["sha256"]:
                raise ValueError(f"{p}: sha256 != {s.name} lex_vocab.sha256")
            return torch.load(p, map_location="cpu", weights_only=True)
    return None


def load_split(cache: str | Path, arm: str, split: str, verify: bool = True) -> list[ProxyRequest]:
    from jobs.q9_feature_cache.extract import load_shards

    table = lex_table(cache, arm, split)
    return [from_record(r, table) for r in load_shards(Path(cache) / arm / split, verify)]


def cache_identity(cache: str | Path, arms) -> dict:
    """sha256 of every split MANIFEST.jsonl (which hashes every shard) and of the lex table, per arm."""
    out = {}
    for arm in sorted(set(arms)):
        d = Path(cache) / arm
        out[arm] = {}
        for sp in ("select", "holdout", "train"):
            if (d / sp / "MANIFEST.jsonl").exists():
                out[arm][sp] = _sha(d / sp / "MANIFEST.jsonl")
        for lv in sorted(d.glob("lex_vocab_*.pt")):
            out[arm][lv.name] = _sha(lv)
    return out


def first_rows(train: list[ProxyRequest], n: int) -> list[ProxyRequest]:
    """The first ``n`` training ROWS (questions) in cache (= draw) order; the last request is cut to fit."""
    out, k = [], 0
    for r in train:
        if k >= n:
            break
        take = r.questions[: n - k]
        out.append(r if len(take) == len(r.questions) else ProxyRequest(r.rid, r.split, take, r.banks, r.catalog_id,
                                                                         r.skipped))
        k += len(take)
    if k < n:
        raise ValueError(f"only {k} training rows cached, recipe needs {n}")
    return out


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
    """Exactly budget["steps"] optimiser steps of budget["grad_accum"] requests (A1(b)); the data cycles in seeded
    per-epoch permutations. ``recipe["train_rows"]`` restricts training to the first N rows (matched-data controls)."""
    if recipe.get("train_rows"):
        train = first_rows(train, int(recipe["train_rows"]))
    torch.manual_seed(seed)
    head = make_proxy_head(recipe, d_model_of(train), seed)
    loss_cfg = LOSSES[recipe["loss"]]
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=budget["lr"][recipe["family"]], weight_decay=budget["weight_decay"])
    warm = budget["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
    head.train()
    hist, t0 = [], time.perf_counter()
    acc, epoch, cursor = budget["grad_accum"], 0, 0
    order = _order(seed, epoch, len(train))
    for _ in range(budget["steps"]):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(acc):
            if cursor >= len(train):
                epoch, cursor = epoch + 1, 0
                order = _order(seed, epoch, len(train))
            req = train[order[cursor]]
            cursor += 1
            lps = head(req)
            loss = sum(decision_loss(lp.unsqueeze(0).float(), torch.tensor([q.label]), [q.qtype], loss_cfg)
                       for lp, q in zip(lps, req.questions)) / max(len(lps), 1)
            (loss / acc).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, budget["max_grad_norm"])
        opt.step()
        sched.step()
        hist.append(total / acc)
    head.eval()
    return head, {"steps": len(hist), "epochs_seen": epoch + cursor / max(len(train), 1), "loss_first": hist[:5],
                  "loss_last": hist[-5:], "seconds": time.perf_counter() - t0, "train_requests": len(train),
                  "train_questions": sum(len(r.questions) for r in train)}


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


def _subset_path(subset: str) -> str:
    return subset if Path(subset).is_absolute() else str(ROOT / subset)


def subset_ids(subset: str) -> list[str]:
    return list(json.loads(Path(_subset_path(subset)).read_text(encoding="utf-8"))["run_ids"])


def subset_file(base: str | Path, keep, out: str | Path) -> str:
    """A dev_subset-format file holding the base subset's run ids filtered by ``keep(run_id, catalog_id)``."""
    from jobs.common import dev_subset as D

    d = json.loads(Path(base).read_text(encoding="utf-8"))
    cat = {e["run_id"]: e["catalog_id"] for e in D.read_manifest()}
    ids = sorted(i for i in d["run_ids"] if keep(i, cat[i]))
    d.update(run_ids=ids, ids_sha256=D.ids_sha256(ids))
    Path(out).write_text(json.dumps(d), encoding="utf-8")
    return str(out)


def _ns(edition: str, **kw) -> Namespace:
    return Namespace(edition=edition, omit_gsm8k03=edition != "0.2.1", **kw)


def score(results: str, subset: str, B: int = R.BOOTSTRAP_B, seed: int = R.BOOTSTRAP_SEED,
          edition: str = R.VERDICT_EDITION) -> dict:
    """0.3: the 0.3 public index over the same rows, without the 0.3 GSM8K twins (the cache holds 0.2.1 rows only)."""
    from jobs.common import dev_subset as D

    return D.cmd_score(_ns(edition, results=results, B=B, seed=seed), D.load_subset(subset))


def compare(cand: str, base: str, subset: str, B: int = R.BOOTSTRAP_B, seed: int = R.BOOTSTRAP_SEED,
            edition: str = R.VERDICT_EDITION) -> dict:
    from jobs.common import dev_subset as D

    out = D.cmd_compare(_ns(edition, candidate=cand, baseline=base, B=B, seed=seed), D.load_subset(subset))
    return {"diff": out["diff"]["index"], "candidate": out.get("candidate", {}).get("index"),
            "baseline": out.get("baseline", {}).get("index")}


def report(results: str, split_subset: str, out_dir: Path, clef: str | None, stock: str | None, B: int) -> dict:
    """Per edition (0.2.1 = verdict, 0.3 = reported): index point + CI, the families-v1 columns (A1(a)), paired diffs
    vs Clef-flash (overall and per column) and vs stock Q9."""
    out = {}
    cols = {c: subset_file(split_subset, lambda i, cat, c=c: c in R.domain_of(cat), out_dir / f"subset_{c}.json")
            for c in R.DOMAIN_COLUMNS}
    for ed in R.REPORT_EDITIONS:
        rep = {"all": score(results, split_subset, B, edition=ed)["slice_index"]["index"]}
        for c, sf in cols.items():
            rep[c] = score(results, sf, B, edition=ed)["slice_index"]["index"]
            if clef:
                rep[f"{c}_vs_clef"] = compare(results, clef, sf, B, edition=ed)["diff"]
        if clef:
            rep["vs_clef_flash"] = compare(results, clef, split_subset, B, edition=ed)["diff"]
        if stock:
            rep["vs_stock_q9"] = compare(results, stock, split_subset, B, edition=ed)["diff"]
        out[ed] = rep
    out["all"] = out[R.VERDICT_EDITION]["all"]  # the verdict edition's headline
    return out


# ----------------------------------------------------------------------------------------------- runs
def run_recipe(recipe: dict, cache: str, dev_rows: str, out: Path, split: str, subset: str, seed: int,
               clef: str | None, stock: str | None, B: int, budget: dict = TRAIN) -> dict:
    arm = recipe["arm"]
    train = load_split(cache, arm, "train")
    head, tr = train_head(recipe, train, seed, budget)
    reqs = load_split(cache, arm, split)
    ids = subset_ids(subset)
    rows = _dev_rows(dev_rows, set(ids))
    final = out / f"{recipe['id']}_s{seed}_{split}"
    # built under a temporary name, renamed into place only when the receipt is complete (re-review N2)
    d = out / f"{final.name}.partial-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    res = d / "results.jsonl"
    counts = write_results(res, rows, reqs, predict(head, reqs), f"proxy-{recipe['id']}")
    counts.update(ids=len(ids), rows=len(rows), cached=len({r.rid for r in reqs} & set(ids)))
    if stock == "from-cache":  # the stock Q9 readout recomputed from this split's cached option log-likelihoods
        stock = str(d / "stock_from_cache.jsonl")
        write_results(stock, rows, reqs, stock_predictions(reqs), "stock-from-cache")
    torch.save(head.state_dict(), d / "head.pt")  # the trained head itself (Track L review: a pinned artifact)
    rec = {"recipe": recipe, "seed": seed, "split": split, "head": describe(head), "head_sha256": _sha(d / "head.pt"),
           "train": tr, "train_fit": nll(head, train[: min(len(train), 2000)]), "results": str(final / "results.jsonl"),
           "results_sha256": _sha(res), "counts": counts, "rule": R.RULE_VERSION, "amendments": list(R.AMENDMENTS),
           "budget": budget, "budget_sha256": budget_sha(budget), "cache_identity": cache_identity(cache, [arm]),
           "code_identity": code_identity(), "score": report(str(res), _subset_path(subset), d, clef, stock, B)}
    (d / "receipt.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    if final.exists():
        raise OutOfOrder(f"{final} appeared while this run was writing {d.name}: refusing to overwrite a receipt")
    os.replace(d, final)
    return rec


def select_summary(out: Path, cache: str, budget: dict = TRAIN, clef: str | None = None, B: int = R.BOOTSTRAP_B) -> dict:
    """select_summary.json from the SELECT receipts of EVERY candidate recipe (refused if any is missing, if budgets or
    cache identities differ, or if a summary already exists). The only writer of select_summary.json."""
    p = out / "select_summary.json"
    if p.exists():
        raise OutOfOrder(f"{p} exists: the SELECT choice is made once")
    recs = {}
    for r in R.RECIPES:
        f = out / f"{r['id']}_s{R.SELECT_SEEDS[0]}_select" / "receipt.json"
        if f.exists():
            recs[r["id"]] = json.loads(f.read_text(encoding="utf-8"))
    missing = [i for i in R.candidate_ids() if i not in recs]
    if missing:
        raise OutOfOrder(f"SELECT receipts missing for candidate recipes {missing}: every candidate must be scored")
    bs = {x["budget_sha256"] for x in recs.values()}
    if bs != {budget_sha(budget)}:
        raise OutOfOrder(f"receipts were trained with budgets {sorted(bs)} != the pre-registered {budget_sha(budget)}")
    ident = cache_identity(cache, [r["arm"] for r in R.RECIPES])
    for i, x in recs.items():
        arm = x["recipe"]["arm"]
        if x["cache_identity"].get(arm) != ident.get(arm):
            raise OutOfOrder(f"{i}: receipt cache identity != the cache on disk ({arm})")
    rows = [{"id": i, "family": x["recipe"]["family"], "select": x["score"]["all"]["point"],
             "ci": [x["score"]["all"]["lo"], x["score"]["all"]["hi"]], "results_sha256": x["results_sha256"],
             "head_sha256": x.get("head_sha256")} for i, x in sorted(recs.items())]
    chosen = R.choose({r["id"]: r["select"] for r in rows})
    controls = {}
    for i, ctl in MATCHED_CONTROLS.items():
        if i in recs and ctl in recs:
            controls[f"{i}-{ctl}"] = compare(recs[i]["results"], recs[ctl]["results"], _subset_path(R.SELECT_SUBSET), B)["diff"]
    doc = {"chosen": chosen, "rows": rows, "candidates": list(R.candidate_ids()), "budget_sha256": budget_sha(budget),
           "code_identity": code_identity(),
           "cache_identity": ident, "matched_controls": controls, "rule": R.RULE_VERSION,
           "amendments": list(R.AMENDMENTS), "edition": R.VERDICT_EDITION}
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def _validate_selection(out: Path, summ: dict, recipe: str | None, budget: dict) -> str | None:
    """Re-derive and authenticate the SELECT choice (re-review N1). -> the chosen recipe id."""
    if set(summ.get("candidates") or []) != set(R.candidate_ids()):
        raise OutOfOrder("select_summary does not list every candidate recipe")
    by = {r["id"]: r for r in summ.get("rows") or []}
    missing = [i for i in R.candidate_ids() if i not in by or by[i].get("select") is None]
    if missing:
        raise OutOfOrder(f"select_summary does not cover every candidate recipe: no SELECT score for {missing}")
    if summ.get("budget_sha256") != budget_sha(budget):
        raise OutOfOrder("select_summary budget != the pre-registered budget")
    if summ.get("code_identity") != code_identity():
        raise OutOfOrder("the proxy-loop code differs from the code that made the SELECT choice (code_identity)")
    chosen = R.choose({i: r["select"] for i, r in by.items()})
    if summ.get("chosen") != chosen:
        raise OutOfOrder(f"select_summary says chosen={summ.get('chosen')!r} but its own rows choose {chosen!r}")
    if recipe is not None and recipe != chosen:
        raise OutOfOrder(f"--recipe {recipe} != the SELECT choice {chosen}: only the chosen recipe is scored on HOLDOUT")
    for i, row in by.items():
        f = out / f"{i}_s{R.SELECT_SEEDS[0]}_select" / "receipt.json"
        if not f.exists():
            raise OutOfOrder(f"{i}: no SELECT receipt in {out}")
        x = json.loads(f.read_text(encoding="utf-8"))
        for k in ("results_sha256", "head_sha256"):
            if x.get(k) != row.get(k):
                raise OutOfOrder(f"{i}: select_summary {k} != the SELECT receipt's")
        if x["score"]["all"]["point"] != row["select"]:
            raise OutOfOrder(f"{i}: select_summary SELECT score {row['select']} != the receipt's {x['score']['all']['point']}")
        if _sha(f.parent / "results.jsonl") != row["results_sha256"] or _sha(f.parent / "head.pt") != row["head_sha256"]:
            raise OutOfOrder(f"{i}: the SELECT results / head files on disk do not match select_summary")
    return chosen


def _validate_inputs(cache: str, dev_rows: str, clef: str, arm: str) -> dict:
    """Every HOLDOUT input, checked before any training or directory creation (re-review N2). -> input identity."""
    from harness.jsonl import read_jsonl_strict

    ids = set(subset_ids(R.HOLDOUT_SUBSET))
    for name, p in (("--dev-rows", dev_rows), ("--clef", clef)):
        if not p or not Path(p).is_file():
            raise OutOfOrder(f"{name} {p!r} is not a file")
    got = set(_dev_rows(dev_rows, ids))
    if got != ids:
        raise OutOfOrder(f"--dev-rows holds {len(got)} of the {len(ids)} HOLDOUT ids")
    cl = {r.get("run_id") for r in read_jsonl_strict(clef)}
    if not ids <= cl:
        raise OutOfOrder(f"--clef results miss {len(ids - cl)} HOLDOUT ids")
    man = Path(cache) / arm / "holdout" / "MANIFEST.jsonl"
    if not man.is_file() or not (Path(cache) / arm / "train" / "MANIFEST.jsonl").is_file():
        raise OutOfOrder(f"the cache has no {arm} holdout / train split")
    return {"dev_rows_sha256": _sha(dev_rows), "clef_sha256": _sha(clef), "holdout_ids_sha256": _sha(_subset_path(R.HOLDOUT_SUBSET)),
            "cache_identity": cache_identity(cache, [arm])[arm]}


def _write_atomic(p: Path, obj: dict) -> None:
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def holdout(out: Path, cache: str, dev_rows: str, clef: str, B: int, recipe: str | None = None,
            budget: dict = TRAIN) -> dict:
    """The single-shot HOLDOUT run (review B2, re-review N1 / N2; see the module docstring)."""
    if (out / "verdict.json").exists():
        raise OutOfOrder("verdict.json exists: HOLDOUT is scored once")
    sp = out / "select_summary.json"
    if not sp.exists():
        raise OutOfOrder("no select_summary.json: run every candidate on SELECT, then `select`")
    summ = json.loads(sp.read_text(encoding="utf-8"))
    chosen = _validate_selection(out, summ, recipe, budget)
    recipes = {r["id"]: dict(r) for r in R.RECIPES}
    attempt_p = out / "holdout_attempt.json"
    if not chosen:
        if any(out.glob("*_holdout")) or attempt_p.exists():
            raise OutOfOrder("holdout receipts exist although no candidate was chosen")
        v = R.verdict(None, None)
        v["select_summary_sha256"] = _sha(sp)
        _write_atomic(out / "verdict.json", v)
        return v
    arm = recipes[chosen]["arm"]
    if cache_identity(cache, [arm])[arm] != summ["cache_identity"].get(arm):
        raise OutOfOrder(f"the cache on disk ({arm}) differs from the one SELECT was run on")
    inputs = _validate_inputs(cache, dev_rows, clef, arm)
    attempt = {"chosen": chosen, "select_summary_sha256": _sha(sp), "inputs": inputs, "code_identity": code_identity(),
               "seeds": list(R.HOLDOUT_SEEDS), "budget_sha256": budget_sha(budget), "rule": R.RULE_VERSION}
    if attempt_p.exists():  # resume after a crash: the SAME attempt only (no re-choice, no changed input)
        prev = json.loads(attempt_p.read_text(encoding="utf-8"))
        if {k: prev.get(k) for k in attempt} != attempt:
            raise OutOfOrder("holdout_attempt.json binds a different choice / select_summary / input / code: refused")
    else:
        if any(out.glob("*_holdout")):
            raise OutOfOrder(f"holdout receipts exist in {out} without a holdout_attempt.json: HOLDOUT is scored once")
        _write_atomic(attempt_p, attempt)
    stray = [p.name for p in out.glob("*_holdout") if not p.name.startswith(f"{chosen}_s")]
    if stray:
        raise OutOfOrder(f"holdout receipts of a non-chosen recipe exist: {stray}")
    recs = []
    for s in R.HOLDOUT_SEEDS:
        f = out / f"{chosen}_s{s}_holdout" / "receipt.json"
        recs.append(json.loads(f.read_text(encoding="utf-8")) if f.exists() else
                    run_recipe(recipes[chosen], cache, dev_rows, out, "holdout", R.HOLDOUT_SUBSET, s, clef, "from-cache", B,
                               budget))
    c = recs[0]["counts"]
    complete = c["cached"] == c["ids"] == c["rows"]
    v = R.verdict(chosen, recs[0]["score"]["all"]["point"], holdout_complete=complete)
    v["holdout_seeds"] = {r["seed"]: r["score"]["all"]["point"] for r in recs}
    v["holdout_0.3"] = recs[0]["score"]["0.3"]["all"]
    v["counts"] = c
    v["vs_clef_flash"] = recs[0]["score"][R.VERDICT_EDITION].get("vs_clef_flash")
    v["domain_columns"] = {col: recs[0]["score"][R.VERDICT_EDITION][col] for col in R.DOMAIN_COLUMNS}
    v["select_summary_sha256"] = _sha(sp)
    v["holdout_attempt_sha256"] = _sha(attempt_p)
    _write_atomic(out / "verdict.json", v)
    return v


def stock_check(out: Path, cache: str, dev_rows: str, stock: str | None, B: int) -> dict:
    """A1(d): the stock readout recomputed from the cached option log-likelihoods on SELECT vs job-5cmat."""
    reqs = load_split(cache, "stock", "select")
    ids = set(subset_ids(R.SELECT_SUBSET))
    res = out / "stock_from_cache_select.jsonl"
    write_results(res, _dev_rows(dev_rows, ids), reqs, stock_predictions(reqs), "stock-from-cache")
    point = score(str(res), _subset_path(R.SELECT_SUBSET), B)["slice_index"]["index"]["point"]
    rep = {**R.stock_check(point), "vs_job_5cmat": compare(str(res), stock, _subset_path(R.SELECT_SUBSET), B) if stock else None,
           "results_sha256": _sha(res)}
    (out / "stock_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def require_stock_check(out: Path) -> None:
    p = out / "stock_check.json"
    if not p.exists():
        raise OutOfOrder("run `stock-check` first (A1(d))")
    if not json.loads(p.read_text(encoding="utf-8"))["ok"]:
        raise OutOfOrder("stock-check failed (|delta| > STOCK_CHECK_TOL): the loop is no-measurement (A1(d))")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train.proxy_loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("recipes", "select", "holdout", "stock-check"):
        s = sub.add_parser(name)
        s.add_argument("--cache", required=True)
        s.add_argument("--dev-rows", default=None, required=name == "holdout")
        s.add_argument("--out", default="proxy-out")
        s.add_argument("--clef", default=None, required=name == "holdout")
        s.add_argument("--stock", default=None)
        s.add_argument("--B", type=int, default=R.BOOTSTRAP_B)
        if name == "recipes":
            s.add_argument("--only", default=None, help="exploratory: receipts only, never select_summary.json")
        if name == "holdout":
            s.add_argument("--recipe", default=None, help="must equal select_summary's choice (checked)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "holdout" and not out.is_dir():  # holdout never creates its output directory (re-review N2)
        print(f"refused: --out {out} does not exist (holdout runs in the directory holding the SELECT receipts)",
              file=sys.stderr)
        return 4
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.cmd == "stock-check":
            rep = stock_check(out, a.cache, a.dev_rows, a.stock, a.B)
            print(json.dumps(rep, indent=1))
            return 0 if rep["ok"] else 3
        if a.cmd == "select":
            print(json.dumps(select_summary(out, a.cache, clef=a.clef, B=a.B), indent=1))
            return 0
        if a.cmd == "recipes":
            require_stock_check(out)
            recipes = {r["id"]: dict(r) for r in R.RECIPES}
            ids = a.only.split(",") if a.only else list(recipes)
            for rid in ids:
                rec = run_recipe(recipes[rid], a.cache, a.dev_rows, out, "select", R.SELECT_SUBSET, R.SELECT_SEEDS[0],
                                 a.clef, a.stock, a.B)
                print(json.dumps({"id": rid, "select": rec["score"]["all"]}), flush=True)
            if not a.only:
                print(json.dumps(select_summary(out, a.cache, clef=a.clef, B=a.B), indent=1))
            return 0
        require_stock_check(out)
        print(json.dumps(holdout(out, a.cache, a.dev_rows, a.clef, a.B, a.recipe), indent=1))
        return 0
    except OutOfOrder as e:
        print(f"refused: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

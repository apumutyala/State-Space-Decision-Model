"""JEV1-DIAG (A-PREREG-JEV1-DIAG): why r2 (JEV1-4B) is flat, from its SAVED checkpoint; no training.

Eval-only, one H100 process, the r2 checkpoint loaded exactly as the screen loads it (fulldev_eval.load_child,
temperature False, the cell's own shim template + layout). Arms on the 231 pinned public JevBench items:

  native       the trained fork head (= r2's PRIMARY arm; identity check: its preds must equal r2's native preds)
  lora_lmhead  the TRAINED backbone (LoRA on) read the shim's way: shim prompt, tied LM-head label log-probs, shim
               format_answer (T 0.8, yes/no floor) = screen.StockShimEngine without lora_off
  stock_shim   LoRA off, the same engine (= r2's baseline; identity check vs r2's stock_shim preds)

and on an IN-DISTRIBUTION probe: ``indist_per_family`` jev1 rows per JevBench family (sha256 order of
"jevdiag:<id>"; jev1 rows, never JevBench items) answered by the same three arms, scored for accuracy + log-loss.

Diagnosis (pre-registered in A-PREREG-JEV1-DIAG; computed by ``report``):
  READOUT   lora_lmhead - native, paired on the jevkit items (screen.paired_vs: B 1000, seed 16): delta I CI > 0 ->
            the fork head / readout loses what the backbone learned.
  TRANSFER  native in-dist accuracy - stock_shim in-dist accuracy >= +5 points (family-cluster bootstrap CI > 0) while
            r2's jevkit delta I CI includes 0 -> learned the training distribution, does not transfer to the items.
  UNDERFIT  native in-dist gain < +2 points -> under-trained / LR too low (not a format problem).
  COVERAGE  every arm's paired deltas on the jev1-covered vs uncovered JevBench families.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ARMS = ("native", "lora_lmhead", "stock_shim")
SCHEMA = "ssdm-jevdiag-v1"


def slice_ids_sha256(record_ids) -> str:
    """Calibration-slice identity (EV-PREREG-AREA-TEMP addendum): sha256 of the sorted record ids, newline-joined."""
    return hashlib.sha256("\n".join(sorted(record_ids)).encode()).hexdigest()


def indist_records(path: Path, per_family: int, exclude: set[str] = frozenset()) -> list[dict]:
    """The probe = calibration slice: per JevBench family (aug.jev1.family8) the ``per_family`` smallest
    sha256("jevdiag:<id>") among the records NOT in ``exclude`` (the record ids r2's checkpoint trained on)."""
    import heapq

    heaps: dict[str, list] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in exclude:
                continue
            fam = ((r.get("aug") or {}).get("jev1") or {}).get("family8") or r.get("family", "?")
            key = -int(hashlib.sha256(f"jevdiag:{r['id']}".encode()).hexdigest()[:16], 16)
            h = heaps.setdefault(fam, [])
            if len(h) < per_family:
                heapq.heappush(h, (key, r["id"], line))
            elif key > h[0][0]:
                heapq.heapreplace(h, (key, r["id"], line))
    out = []
    for fam in sorted(heaps):
        for _, _, line in sorted(heaps[fam], key=lambda x: -x[0]):
            r = json.loads(line)
            r["_fam"] = fam
            out.append(r)
    return out


def correct_and_logloss(q: dict, gold: str, ans: Any) -> tuple[int, float] | None:
    """(correct, -log p(gold)) of one SystemOne answer; None if the answer is invalid."""
    if not isinstance(ans, dict):
        return None
    t = q.get("type", "choice")
    try:
        if t == "noul":
            py = float(ans["noul"])
            g = str(gold).lower() in ("true", "yes", "1")
            p = py if g else 1.0 - py
            return int((py >= 0.5) == g), -math.log(max(p, 1e-12))
        probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
        g = str(gold)
        pred = max(probs, key=probs.get)
        return int(pred == g), -math.log(max(probs.get(g, 0.0), 1e-12))
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------- GPU: run
def run(a) -> int:
    from jevkit.items import load_public
    from jevkit.systemone import SystemOneAdapter, run_items

    from jobs.m2_first_cell.fulldev_eval import load_child
    from jobs.m2_first_cell.screen import JEVKIT_WARMUP, StockShimEngine

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    dg = params["jevdiag"]
    a.want_visited = True  # fulldev_eval.load_child: the item ids this checkpoint's optimiser steps consumed
    try:
        items = load_public()
        decider, _, ident = load_child(a, temperature=False)
    except Exception as e:  # noqa: BLE001 - identity / lineage failures
        print(f"identity: {type(e).__name__}: {e}", file=sys.stderr)
        return 10
    visited_items = ident.pop("visited_item_ids")
    visited = {i.rsplit("/", 1)[0] for i in visited_items}
    with gzip.open(out / "r2_visited_record_ids.txt.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(sorted(visited)) + "\n")
    eng = StockShimEngine(decider.backend, a.model_dir)
    targets = {"native": SystemOneAdapter(decider, kind="decider", model_name="child-native"),
               "lora_lmhead": SystemOneAdapter(eng, kind="request", model_name="lora-lmhead-shim"),
               "stock_shim": SystemOneAdapter(eng, kind="request", model_name="stock-shim")}
    probe = indist_records(Path(a.data) / dg["indist_records"], int(dg["indist_per_family"]), visited)
    sha = slice_ids_sha256(r["id"] for r in probe)
    pin = dg.get("calib_ids_sha256")
    if pin and len(pin) == 64 and pin != sha:
        print(f"identity: calib slice ids sha256 {sha} != pinned {pin}", file=sys.stderr)
        return 10
    meta = {"identity": {k: v for k, v in ident.items() if k != "cell_config"}, "arms": {}, "indist": {
        "n": len(probe), "ids_sha256": sha, "per_family": int(dg["indist_per_family"]),
        "excluded_r2_visited_records": len(visited), "visited": ident.get("visited"),
        "rule": "per family, ascending sha256('jevdiag:<id>') among jev1 records r2's checkpoint never stepped on"}}
    export_head_probs(decider.backend, probe, out / "calib_probs.jsonl")
    for name, ad in targets.items():
        ctx = eng.lora_off() if name == "stock_shim" else _null()
        with ctx:
            for it in items[:JEVKIT_WARMUP]:
                ad.answer_item(it)
            t0 = time.perf_counter()
            recs = run_items(ad, items)
            with open(out / f"{name}.preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
            t1 = time.perf_counter()
            with open(out / f"{name}.indist.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in probe:
                    q = r["questions"]["q"]
                    resp = ad({"model": "jevdiag", "state": r["state"], "questions": {"q": q}})
                    f.write(json.dumps({"id": r["id"], "family": r["_fam"], "type": q.get("type", "choice"),
                                        "gold": r["labels"]["q"], "status": resp["status"],
                                        "answer": resp["answers"].get("q")}, separators=(",", ":"), default=str) + "\n")
        meta["arms"][name] = {"jevkit_s": round(t1 - t0, 1), "indist_s": round(time.perf_counter() - t1, 1)}
    (out / "identity.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return 0


def export_head_probs(backend, probe: list[dict], path: Path) -> None:
    """The native fork head's per-option probabilities (untempered) for every slice decision: id, family, type,
    option ids, probs, gold index. T_J / per-family T_f are fitted offline from this file (never on public-231)."""
    import torch

    from train.data import items_from_records
    from train.run_cell import to_train_record

    fam = {r["id"]: r["_fam"] for r in probe}
    recs = [to_train_record({k: v for k, v in r.items() if k != "_fam"}) for r in probe]
    its = items_from_records(recs, backend.encode, layout=backend.layout, template=backend.template)
    m = backend.model
    was = m.training
    m.eval()
    try:
        with torch.no_grad(), open(path, "w", encoding="utf-8", newline="\n") as g:
            for it in its:
                lp = torch.log_softmax(m([it.row]).log_probs[0].float().cpu()[: it.row.n_options], -1)
                g.write(json.dumps({"id": it.id, "family": fam.get(it.id.rsplit("/", 1)[0]), "type": it.row.qtype,
                                    "option_ids": list(it.row.option_ids), "probs": lp.exp().tolist(),
                                    "gold": int(it.label)}, separators=(",", ":")) + "\n")
    finally:
        m.train(was)


def _null():
    import contextlib
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------------------- CPU: report
def family_cluster_ci(rows_a: list[dict], rows_b: list[dict], B: int = 2000, seed: int = 0) -> dict:
    """Paired accuracy difference a - b on the same probe rows; bootstrap over families (clusters), then rows."""
    import random

    by = {}
    for x, y in zip(rows_a, rows_b):
        if x["id"] != y["id"]:
            raise ValueError("in-dist rows out of order")
        by.setdefault(x["family"], []).append((x["_c"], y["_c"]))
    fams = sorted(by)
    pt = sum(c1 - c2 for v in by.values() for c1, c2 in v) / max(1, sum(len(v) for v in by.values()))
    rng, draws = random.Random(seed), []
    for _ in range(B):
        s = n = 0
        for f in (rng.choice(fams) for _ in fams):
            v = by[f]
            for _ in v:
                c1, c2 = v[rng.randrange(len(v))]
                s += c1 - c2
                n += 1
        draws.append(s / n)
    draws.sort()
    return {"point": pt, "ci95": [draws[int(0.025 * B)], draws[int(0.975 * B) - 1]], "B": B, "seed": seed}


def report(a) -> int:
    from jevkit import cli as JC
    from jevkit import scoring as JS
    from jevkit.items import load_public
    from jevkit.systemone import preds_to_answers

    from jobs.m2_first_cell.screen import paired_vs

    d = Path(a.out_dir)
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    dg, price = params["jevdiag"], params["jevkit"]["price"]
    items = load_public()
    tables, axes, answers, summ = {}, {}, {}, {}
    for name in ARMS:
        pf, rp = d / f"{name}.preds.jsonl", d / f"{name}.jevkit.json"
        if JC.main(["score", "--public", "--pred", str(pf), "--price", price, "--out", str(rp)]) != 0:
            print(f"jevkit score {name} failed", file=sys.stderr)
            return 9
        rep = json.loads(rp.read_text(encoding="utf-8"))
        preds = [json.loads(x) for x in pf.read_text(encoding="utf-8").splitlines() if x.strip()]
        answers[name] = preds_to_answers(preds)
        tables[name] = JS.build_table(items, answers[name])
        axes[name] = (rep["axes"]["speed"], rep["axes"]["cost"])
        summ[name] = {"I": rep["axes"]["intelligence"], "C": rep["axes"]["calibration"], "capability": rep["capability"],
                      "composite": rep["scores"]}
    pairs = {"native_vs_stock": paired_vs(tables, axes, "native", "stock_shim"),
             "lora_lmhead_vs_stock": paired_vs(tables, axes, "lora_lmhead", "stock_shim"),
             "lora_lmhead_vs_native": paired_vs(tables, axes, "lora_lmhead", "native")}
    cov = set(dg["covered_families"])
    by_cov = {}
    for grp in ("covered", "uncovered"):
        sub = [it for it in items if (it.family in cov) == (grp == "covered")]
        st = {nm: JS.build_table(sub, answers[nm]) for nm in ARMS}
        by_cov[grp] = {"n_items": len(sub), "native_vs_stock": paired_vs(st, axes, "native", "stock_shim"),
                       "lora_lmhead_vs_stock": paired_vs(st, axes, "lora_lmhead", "stock_shim")}
    ind = {}
    for name in ARMS:
        rows = [json.loads(x) for x in (d / f"{name}.indist.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in rows:
            cl = correct_and_logloss({"type": r["type"]}, r["gold"], r["answer"]) if r["status"] == "ok" else None
            r["_c"], r["_ll"] = (cl if cl else (0, math.log(1e12)))
        ind[name] = rows
    acc = {nm: {"acc": sum(r["_c"] for r in v) / len(v), "logloss": sum(r["_ll"] for r in v) / len(v), "n": len(v),
                "by_family": {f: sum(r["_c"] for r in v if r["family"] == f) / max(1, sum(1 for r in v if r["family"] == f))
                              for f in sorted({r["family"] for r in v})}} for nm, v in ind.items()}
    gain = family_cluster_ci(ind["native"], ind["stock_shim"])
    readout = pairs["lora_lmhead_vs_native"]["I"]
    r2_flat = dg.get("r2_delta_I_ci95")
    diag = {
        "READOUT": bool(readout["ci95"] and readout["ci95"][0] > 0),
        "TRANSFER": bool(gain["point"] >= 0.05 and gain["ci95"][0] > 0
                         and (r2_flat is None or (r2_flat[0] <= 0 <= r2_flat[1]))),
        "UNDERFIT": bool(gain["point"] < 0.02),
    }
    out = {"schema": SCHEMA, "jevkit": summ, "paired": pairs, "by_coverage": by_cov, "indist": acc,
           "indist_native_minus_stock": gain, "diagnosis": diag,
           "next": ("readout: train / serve with the LM-head label readout" if diag["READOUT"] else
                    "transfer: format + coverage (jev2 families) before more steps" if diag["TRANSFER"] else
                    "underfit: JEV2-4B budget (1 epoch) and LR x2" if diag["UNDERFIT"] else
                    "mixed: in-dist gain 2-5 points; run JEV2-4B (coverage + 1 epoch)"),
           "rules": __doc__.split("Diagnosis", 1)[1].strip()}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"diagnosis": diag, "next": out["next"]}))
    return 0


def valid_result(d: Any) -> bool:
    try:
        return (d.get("schema") == SCHEMA and set(d["jevkit"]) == set(ARMS) and isinstance(d["diagnosis"], dict)
                and all(isinstance(v, bool) for v in d["diagnosis"].values()))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.jev_diag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    for k in ("--cell", "--ckpt-dir", "--records", "--train-dir", "--guard-readiness", "--model-dir", "--data",
              "--params", "--out-dir"):
        s.add_argument(k, required=True)
    s.add_argument("--ckpt", default=None)
    s.add_argument("--teacher-dir", default=None)
    s.add_argument("--pool-dir", action="append", default=None)
    s.add_argument("--device", default="cuda")
    r = sub.add_parser("report")
    for k in ("--out-dir", "--params", "--out"):
        r.add_argument(k, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    sys.exit(main())

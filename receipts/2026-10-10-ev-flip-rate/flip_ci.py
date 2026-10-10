"""B-PREREG-FLIP scorer: per-operator flip rates + their average over operators, with a bootstrap CI over BASE items.

    python -X utf8 receipts/2026-10-10-ev-flip-rate/flip_ci.py --perturbed pert_public_seed0.jsonl \
        --model NAME=base_preds.jsonl,pert_preds.jsonl [--model ...] [--ref NAME] --B 1000 --seed 16 --out flip_ci.json

Point estimates use jevkit.perturb's own flip rule (frozen module, imported read-only: _dist / _decision / label_map) and
are cross-checked against jevkit.perturb.flip_report (must match exactly, else exit 3). Bootstrap: resample the 231 base
items with replacement (B resamples, numpy default_rng(seed)); every perturbed variant of a drawn base item comes along
with it; per operator flip rate = flips / valid pairs on the resample; average = mean over operators with >= 1 valid
pair. Paired deltas vs --ref use the SAME resamples. 95% CI = 2.5 / 97.5 percentiles. Invalid pairs are excluded from n
(reported separately), exactly as flip_report does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import items as IT  # noqa: E402
from jevkit import perturb as P  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402


def _read(p: str) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_perturbed(path: str) -> list[P.Perturbed]:
    out = []
    for d in _read(path):
        m = d.pop("perturbation")
        out.append(P.Perturbed(IT.Item.from_dict(d), m["op"], m["seed"], m["base_id"], m["label_map"]))
    return out


def outcomes(base_items, perturbed, base_ans, pert_ans):
    """list of (base_index, op, flip in {0,1} or None=invalid) - flip_report's rule, pair by pair."""
    base = {it.id: it for it in base_items}
    idx = {it.id: i for i, it in enumerate(base_items)}
    rows = []
    for p in perturbed:
        b = base[p.base_id]
        db, dp_ = P._dist(b, base_ans.get(b.id)), P._dist(p.item, pert_ans.get(p.item.id))
        if db is None or dp_ is None:
            rows.append((idx[b.id], p.op, None))
            continue
        back = {lab: dp_[p.label_map[lab]] for lab in b.labels}
        rows.append((idx[b.id], p.op, int(P._decision(b, db) != P._decision(b, back))))
    return rows


def rates(rows, ops, weights=None):
    """per-op flip rate on a resample given as per-base-item multiplicity weights (None = all once)."""
    out = {}
    for op in ops:
        n = f = 0.0
        for bi, o, fl in rows:
            if o != op or fl is None:
                continue
            w = 1.0 if weights is None else weights[bi]
            n += w
            f += w * fl
        out[op] = (f / n) if n else None
    vals = [v for v in out.values() if v is not None]
    out["_average"] = (sum(vals) / len(vals)) if vals else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbed", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=base.jsonl,pert.jsonl")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = IT.load_public()
    pert = load_perturbed(a.perturbed)
    ops = list(P.OPERATORS)
    res: dict = {"rule": "B-PREREG-FLIP", "B": a.B, "seed": a.seed, "n_base": len(items), "n_perturbed": len(pert),
                 "operators": ops, "models": {}}
    rows_by = {}
    for spec in a.model:
        name, files = spec.split("=", 1)
        bp, pp = files.split(",")
        ba, pa = preds_to_answers(_read(bp)), preds_to_answers(_read(pp))
        rows = outcomes(items, pert, ba, pa)
        rep = P.flip_report(items, pert, ba, pa)
        pt = rates(rows, ops)
        for op in ops:  # exact cross-check against the frozen reporter
            ref = (rep["per_operator"].get(op) or {}).get("flip_rate")
            if (ref is None) != (pt[op] is None) or (ref is not None and abs(ref - pt[op]) > 1e-12):
                print(f"MISMATCH {name} {op}: {pt[op]} vs flip_report {ref}", file=sys.stderr)
                return 3
        rows_by[name] = rows
        res["models"][name] = {"flip_report": rep, "point": pt,
                               "n_valid": {op: sum(1 for r in rows if r[1] == op and r[2] is not None) for op in ops},
                               "n_invalid": {op: sum(1 for r in rows if r[1] == op and r[2] is None) for op in ops}}
    rng = np.random.default_rng(a.seed)
    draws = {m: {k: [] for k in ops + ["_average"]} for m in rows_by}
    deltas = {m: {k: [] for k in ops + ["_average"]} for m in rows_by if a.ref and m != a.ref}
    for _ in range(a.B):
        w = np.bincount(rng.integers(0, len(items), size=len(items)), minlength=len(items)).astype(float)
        cur = {m: rates(r, ops, w) for m, r in rows_by.items()}
        for m, v in cur.items():
            for k, x in v.items():
                if x is not None:
                    draws[m][k].append(x)
            if m in deltas:
                for k, x in v.items():
                    y = cur[a.ref][k]
                    if x is not None and y is not None:
                        deltas[m][k].append(x - y)

    def ci(xs):
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None

    for m in rows_by:
        res["models"][m]["ci95"] = {k: ci(v) for k, v in draws[m].items()}
    if deltas:
        res["paired_minus_" + a.ref] = {m: {k: {"point": (None if res["models"][m]["point"][k] is None or
                                                          res["models"][a.ref]["point"][k] is None else
                                                          res["models"][m]["point"][k] - res["models"][a.ref]["point"][k]),
                                                "ci95": ci(v)} for k, v in d.items()} for m, d in deltas.items()}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    hdr = "model".ljust(22) + "".join(o[:10].rjust(11) for o in ops) + "    average [95% CI]"
    print(hdr)
    for m, v in res["models"].items():
        p, c = v["point"], v["ci95"]["_average"]
        line = m.ljust(22) + "".join((f"{100 * p[o]:10.1f}%" if p[o] is not None else "       n/a") for o in ops)
        print(line + (f"   {100 * p['_average']:.2f}% [{100 * c[0]:.2f}, {100 * c[1]:.2f}]" if c else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

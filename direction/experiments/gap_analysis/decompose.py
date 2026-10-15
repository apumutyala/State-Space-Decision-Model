"""Where does the one-pass Intelligence gap to H2O-Lightning-4B come from? (EVAL-ONLY public-231 diagnostics.)

Reads the private per-item shim-variant predictions of the reasoning-headroom probe (and, if present, extra systems
scored by ``score_shim.py``) from ``$SSDM_PRIVATE``. It fits nothing and selects nothing; it only re-scores and
counts. It writes aggregates to ``gap.json`` next to this file.

    python -X utf8 direction/experiments/gap_analysis/decompose.py

Intelligence (jevkit, JevBench v1.6.1): I = mean over {choice, yes/no, score} of the chance-corrected score CC_type,
and CC_type = tier-weighted mean of the per-tier CC (easy 0.1, standard 0.2, judge 0.3, hard 0.4, renormalised over
the tiers present). Choice CC = (acc - chance) / (1 - chance); yes/no CC = (acc - 0.5) / 0.5; score
CC = 1 - NMAE / NMAE_chance.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from jevkit import items as IT  # noqa: E402
from jevkit import scoring as SC  # noqa: E402
from jevkit import spec  # noqa: E402
from keel.paths import private_root  # noqa: E402

RAW = private_root() / "experiments" / "reasoning_headroom" / "raw"
EXTRA = private_root() / "gap_analysis"
SYSTEMS = {
    "qwen4b_one_pass": RAW / "Qwen3.5-4B.nothink.shim.preds.jsonl",
    "qwen4b_think": RAW / "Qwen3.5-4B.think.shim.preds.jsonl",
    "qwen9b_one_pass": RAW / "Qwen3.5-9B.nothink.shim.preds.jsonl",
    "h2o_one_pass": RAW / "h2o-lightning-4b.nothink.shim.preds.jsonl",
    "student_one_pass": EXTRA / "student.nothink.shim.preds.jsonl",
    "qwen4b_one_pass_rescored": EXTRA / "stock4b.nothink.shim.preds.jsonl",
}
REFERENCE, BASELINE = "h2o_one_pass", "qwen4b_one_pass"


def load(path: Path) -> dict:
    return {r["id"]: r["answer"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def cells(pub, table) -> dict:
    """CC and n per (type, tier), and the I-point weight of each cell (its share of I)."""
    out, by_type = {}, collections.defaultdict(dict)
    for typ in spec.TYPES:
        for tier in spec.TIERS:
            m = (table.typ == SC.TYPE_IX[typ]) & (table.tier == SC.TIER_IX[tier])
            if m.any():
                by_type[typ][tier] = (SC._cell_cc(table, m, typ), int(m.sum()))
    for typ, tiers in by_type.items():
        wsum = sum(spec.TIER_WEIGHTS[t] for t in tiers)
        for tier, (cc, n) in tiers.items():
            out[f"{typ}|{tier}"] = {"cc": cc, "n": n,
                                    "share_of_I": spec.TYPE_WEIGHTS[typ] * spec.TIER_WEIGHTS[tier] / wsum}
    return out


def biases(pub, answers) -> dict:
    """Label priors: yes-rate vs gold yes-rate (yes/no); predicted vs gold option position (choice)."""
    yes_pred = yes_gold = n_noul = 0
    acc_by_gold = collections.defaultdict(lambda: [0, 0])
    pos_pred, pos_gold, n_choice = collections.Counter(), collections.Counter(), 0
    score_err = []
    for it in pub:
        a = answers.get(it.id)
        if a is None:
            continue
        q = it.question
        if it.type == "noul":
            pred = "yes" if a["noul"] >= 0.5 else "no"
            n_noul += 1
            yes_pred += pred == "yes"
            yes_gold += it.expected == "yes"
            acc_by_gold[it.expected][0] += pred == it.expected
            acc_by_gold[it.expected][1] += 1
        elif it.type == "choice":
            names = list(q["criteria"])
            n_choice += 1
            pos_pred[min(names.index(a["choice"]), 3)] += 1
            pos_gold[min(names.index(str(it.expected)), 3)] += 1
        else:
            score_err.append(a["score"] - int(it.expected))
    pos = lambda c: {("ABC"[k] if k < 3 else "D+"): round(c[k] / n_choice, 3) for k in range(4)}
    return {"noul": {"n": n_noul, "pred_yes_rate": yes_pred / n_noul, "gold_yes_rate": yes_gold / n_noul,
                     "acc_gold_yes": acc_by_gold["yes"][0] / acc_by_gold["yes"][1],
                     "acc_gold_no": acc_by_gold["no"][0] / acc_by_gold["no"][1]},
            "choice_position": {"n": n_choice, "pred": pos(pos_pred), "gold": pos(pos_gold)},
            "score": {"n": len(score_err), "mean_signed_error_levels": sum(score_err) / len(score_err),
                      "mean_abs_error_levels": sum(map(abs, score_err)) / len(score_err)}}


def main() -> None:
    pub = IT.load_public()
    systems = {k: load(p) for k, p in SYSTEMS.items() if p.exists()}
    tables = {k: SC.build_table(pub, a) for k, a in systems.items()}
    report = {"systems": {}, "note": "EVAL-ONLY diagnostics; nothing fitted or selected on these items"}
    for k, t in tables.items():
        agg = SC.aggregate(t)
        report["systems"][k] = {"I": agg["intelligence"]["I"], "correct": int(t.correct.sum()), "n": len(t.ids),
                                "cells": cells(pub, t), "biases": biases(pub, systems[k])}
    ref, base = report["systems"][REFERENCE], report["systems"][BASELINE]
    report["gap_cells"] = {c: {"delta_cc": ref["cells"][c]["cc"] - base["cells"][c]["cc"],
                               "I_points": ref["cells"][c]["share_of_I"] * (ref["cells"][c]["cc"] - base["cells"][c]["cc"]),
                               "n": ref["cells"][c]["n"], "share_of_I": ref["cells"][c]["share_of_I"]}
                           for c in ref["cells"]}
    # item overlap: what the reference gets right that the baseline misses, and whether thinking recovers it
    ids = tables[BASELINE].ids
    cor = {k: dict(zip(t.ids, t.correct)) for k, t in tables.items()}
    typ = {it.id: it.type for it in pub}
    overlap = collections.defaultdict(collections.Counter)
    for i in ids:
        if typ[i] == "score":
            continue
        key = ("ref_only" if cor[REFERENCE][i] and not cor[BASELINE][i] else
               "base_only" if cor[BASELINE][i] and not cor[REFERENCE][i] else
               "both" if cor[BASELINE][i] else "neither")
        overlap[key]["n"] += 1
        if "qwen4b_think" in cor:
            overlap[key]["qwen4b_think_correct"] += bool(cor["qwen4b_think"][i])
        if "student_one_pass" in cor:
            overlap[key]["student_correct"] += bool(cor["student_one_pass"][i])
    report["overlap_choice_noul"] = {k: dict(v) for k, v in overlap.items()}
    (HERE / "gap.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"{'system':26s} {'I':>6s} {'correct':>8s}")
    for k, s in report["systems"].items():
        print(f"{k:26s} {s['I']:6.1f} {s['correct']:5d}/{s['n']}")
    print(f"\nI gap {REFERENCE} - {BASELINE} = {ref['I'] - base['I']:.1f}, by cell (share of I, n, dCC, I points):")
    for c, g in sorted(report["gap_cells"].items(), key=lambda x: -x[1]["I_points"]):
        print(f"  {c:18s} share {g['share_of_I']:.3f} n={g['n']:3d} dCC {g['delta_cc']:+6.1f} -> {g['I_points']:+5.1f} I")
    for k in (BASELINE, REFERENCE, "qwen4b_think", "student_one_pass"):
        if k in report["systems"]:
            b = report["systems"][k]["biases"]
            print(f"\n{k}: yes/no pred-yes {b['noul']['pred_yes_rate']:.2f} (gold {b['noul']['gold_yes_rate']:.2f}); "
                  f"acc gold-yes {b['noul']['acc_gold_yes']:.2f} gold-no {b['noul']['acc_gold_no']:.2f}; "
                  f"choice pred pos {b['choice_position']['pred']} gold {b['choice_position']['gold']}; "
                  f"score signed err {b['score']['mean_signed_error_levels']:+.2f}")
    print("\noverlap (choice + yes/no):", json.dumps(report["overlap_choice_noul"]))


if __name__ == "__main__":
    main()

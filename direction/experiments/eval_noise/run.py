"""How much of a JevBench Intelligence gap is sampling noise? (EVAL-ONLY public-231, CPU, existing predictions.)

Re-scores the gap-analysis systems with jevkit and resamples items two ways:
- jevkit's own paired stratified bootstrap (side x type x tier strata), as the board uses;
- a family-cluster bootstrap, where items from one generator family move together. It is more conservative
  because items within a family are correlated.

Reports each system's public Intelligence with 95% intervals, paired differences, and the minimum detectable
difference (MDD: 80% power, two-sided 5%, normal approximation 2.80 x SD of the paired difference), plus the number
of items needed to detect a 3-point gap. Nothing is fitted or selected.

    python -X utf8 direction/experiments/eval_noise/run.py   # writes results.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / "gap_analysis"))

import decompose  # noqa: E402  (system list and prediction loader of the gap analysis)
from jevkit import items as IT  # noqa: E402
from jevkit import scoring as SC  # noqa: E402

B, SEED, Z80 = 2000, 20261016, 1.959964 + 0.841621
NAMES = {"qwen4b_one_pass": "stock 4B", "qwen4b_one_pass_rescored": "stock 4B (second run)",
         "student_one_pass": "student", "qwen9b_one_pass": "9B", "h2o_one_pass": "H2O-Lightning-4B",
         "qwen4b_think": "4B thinking"}
PAIRS = [("h2o_one_pass", "student_one_pass"), ("h2o_one_pass", "qwen4b_one_pass"),
         ("student_one_pass", "qwen4b_one_pass"), ("qwen4b_think", "h2o_one_pass"),
         ("qwen9b_one_pass", "qwen4b_one_pass"), ("qwen4b_one_pass_rescored", "qwen4b_one_pass")]


def intelligence(table: SC.ItemTable, idx: np.ndarray | None = None) -> float:
    return SC.aggregate(table, idx, detail=False)["intelligence"]["I"]


def main() -> None:
    pub = IT.load_public()
    tables = {k: SC.build_table(pub, decompose.load(p)) for k, p in decompose.SYSTEMS.items() if p.exists()}
    ref = next(iter(tables.values()))
    assert all(t.ids == ref.ids for t in tables.values())
    family = {it.id: it.family for it in pub}
    clusters = defaultdict(list)
    for i, item_id in enumerate(ref.ids):
        clusters[family[item_id]].append(i)
    cluster_idx = [np.array(v) for v in clusters.values()]
    strata = SC.strata(ref)
    rng = np.random.default_rng(SEED)
    draws = {"stratified": defaultdict(list), "family_cluster": defaultdict(list)}
    for _ in range(B):
        idx_s = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        pick = rng.integers(0, len(cluster_idx), size=len(cluster_idx))
        idx_c = np.concatenate([cluster_idx[j] for j in pick])
        for name, table in tables.items():
            draws["stratified"][name].append(intelligence(table, idx_s))
            draws["family_cluster"][name].append(intelligence(table, idx_c))

    def summary(values: list[float]) -> dict:
        v = np.array(values)
        return {"sd": float(v.std(ddof=1)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}

    out = {"n_items": len(ref.ids), "n_families": len(cluster_idx), "B": B, "seed": SEED,
           "note": "EVAL-ONLY public-231; nothing fitted or selected", "systems": {}, "pairs": {}}
    for name, table in tables.items():
        out["systems"][NAMES.get(name, name)] = {
            "I": intelligence(table), "correct": int(table.correct.sum()),
            **{kind: summary(draws[kind][name]) for kind in draws}}
    for a, b in PAIRS:
        if a not in tables or b not in tables:
            continue
        entry = {"diff": intelligence(tables[a]) - intelligence(tables[b])}
        for kind in draws:
            d = np.array(draws[kind][a]) - np.array(draws[kind][b])
            sd = float(d.std(ddof=1))
            mdd = Z80 * sd
            entry[kind] = {"sd": sd, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                           "p_first_better": float((d > 0).mean()), "mdd80": mdd,
                           "items_for_3_points": int(np.ceil(len(ref.ids) * (mdd / 3.0) ** 2))}
        out["pairs"][f"{NAMES[a]} - {NAMES[b]}"] = entry
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(ref.ids)} items, {len(cluster_idx)} families, B={B}")
    for name, s in out["systems"].items():
        print(f"{name:24s} I={s['I']:5.1f}  stratified sd {s['stratified']['sd']:.2f}  "
              f"family sd {s['family_cluster']['sd']:.2f}  family CI {s['family_cluster']['ci95'][0]:.1f}..{s['family_cluster']['ci95'][1]:.1f}")
    for pair, e in out["pairs"].items():
        st, fc = e["stratified"], e["family_cluster"]
        print(f"{pair:40s} diff {e['diff']:+5.1f}  strat CI {st['ci95'][0]:+.1f}..{st['ci95'][1]:+.1f} MDD {st['mdd80']:.1f}  "
              f"family CI {fc['ci95'][0]:+.1f}..{fc['ci95'][1]:+.1f} MDD {fc['mdd80']:.1f} items@3pts {fc['items_for_3_points']}")


if __name__ == "__main__":
    main()

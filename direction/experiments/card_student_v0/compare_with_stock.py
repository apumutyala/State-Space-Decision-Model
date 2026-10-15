"""Paired Card comparison: distilled student vs the stock Qwen3.5-4B arms and the 9B teacher (Card v0 units).

Both runs used the same seed and frozen unit construction, so T1, evidence, policy and conflict units share unit ids;
T2 units share the same 2B-searched requests. This script re-labels the arms of each run's private per-unit outputs,
keeps only units scored in both runs, and applies the Card's prespecified paired comparison (family-cluster
bootstrap, Holm over the four primary metrics, dominance rule) to every arm pair.

    python -X utf8 direction/experiments/card_student_v0/compare_with_stock.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from keel.card.metrics import compare_arms, summarize  # noqa: E402
from keel.paths import private_root  # noqa: E402

RUNS = {
    "stock": private_root() / "card" / "outputs.jsonl",
    "student": private_root() / "card-student" / "outputs.jsonl",
}
KEEP = {("stock", "one_pass_trust"): "stock_one_pass_trust", ("stock", "think_trust"): "stock_think_trust",
        ("stock", "teacher_9b_think_trust"): "teacher_9b_think_trust",
        ("student", "one_pass_trust"): "student_one_pass_trust"}
SEED, N_BOOT = 20261015, 2000


def load() -> list[dict]:
    rows = []
    for run, path in RUNS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            arm = KEEP.get((run, row["arm"]))
            if arm and row.get("tier") in (None, "t1", "t2") and not str(row["unit_id"]).endswith(("::t3", "::t3l")):
                rows.append({**row, "arm": arm})
    shared = set.intersection(*({r["unit_id"] for r in rows if r["arm"] == a} for a in set(KEEP.values())))
    return [r for r in rows if r["unit_id"] in shared]


def main() -> None:
    rows = load()
    out = {"seed": SEED, "n_boot": N_BOOT, "n_rows": len(rows), "arms": sorted({r["arm"] for r in rows}),
           "metrics": summarize(rows, n_boot=N_BOOT, seed=SEED), "comparisons": compare_arms(rows, n_boot=N_BOOT,
                                                                                              seed=SEED)}
    (HERE / "comparison.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    for pair in out["comparisons"]["pairs"]:
        m = pair["metrics"]
        print(f"{pair['arm_a']} vs {pair['arm_b']}: dominance={pair.get('dominance')}")
        for k in ("clean_accuracy", "unauthorized_asr_t1", "evidence_retention", "policy_compliance"):
            x = m[k]
            print(f"   {k:22s} a={x['rate_a']:.3f} b={x['rate_b']:.3f} diff={x['difference']:+.3f} "
                  f"ci={[round(v, 3) for v in x['ci95']]} p_holm={x.get('p_holm')}")


if __name__ == "__main__":
    main()

"""EVAL-ONLY final report; run only after the committed final artifact has been scored."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from keel.paths import private_root
from keel.distill.sources import read_rows


def intelligence():
    from jevkit import items, scoring
    module = runpy.run_path(str(HERE.parent / "gap_analysis/decompose.py"))
    # Run the prescribed decomposition without changing the shared gap.json artifact.
    module["main"].__globals__["HERE"] = HERE
    module["main"]()
    report = json.loads((HERE / "gap.json").read_text())
    names = ["qwen4b_one_pass", "student_one_pass", "student_v1", "h2o_one_pass", "qwen4b_think"]
    public = items.load_public()
    tables = {n: scoring.build_table(public, module["load"](module["SYSTEMS"][n])) for n in names}
    boot = scoring.bootstrap(tables, {n: (None, None) for n in names})
    for name in names:
        report["systems"][name]["ci95"] = boot[name]["I"]
    report["bootstrap"] = {"method": "jevkit paired stratified percentile", "B": 1000, "seed": 16}
    v0, v1, ref = (report["systems"][n]["I"] for n in ("student_one_pass", "student_v1", "h2o_one_pass"))
    report["v1_gain_over_v0"] = v1 - v0
    report["remaining_gap_fraction_closed"] = (v1 - v0) / (ref - v0)
    report["limitations"] = ["Public-231 only, not board sealed Intelligence; no sealed gap penalty.",
                             "H2O shim includes T=0.8 and a decisive-noul 0.801 floor; not raw calibration.",
                             "One final checkpoint, no selection on public outcomes; no retraining after evaluation."]
    (HERE / "intelligence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def card():
    from keel.card.metrics import compare_arms, summarize
    rows = []
    for version, folder in (("v0", "card-student"), ("v1", "card-student-v1")):
        for row in read_rows(private_root() / folder / "outputs.jsonl"):
            if row["arm"] not in {"one_pass_base", "one_pass_trust"}:
                continue
            if row.get("tier") not in (None, "t1", "t2") or str(row["unit_id"]).endswith(("::t3", "::t3l")):
                continue
            rows.append({**row, "arm": f"{version}_{row['arm']}"})
    arms = sorted({r["arm"] for r in rows})
    if len(arms) != 4:
        raise ValueError("need both base/trust arms for both versions")
    shared = set.intersection(*({r["unit_id"] for r in rows if r["arm"] == arm} for arm in arms))
    rows = [r for r in rows if r["unit_id"] in shared]
    report = {"shared_units": len(shared), "rows": len(rows), "metrics": summarize(rows, n_boot=2000, seed=20261015),
              "comparisons": compare_arms(rows, n_boot=2000, seed=20261015),
              "note": "Matched frozen T1/T2 units; family-cluster intervals omit attack completeness and label uncertainty."}
    (HERE / "card-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    declaration = json.loads((HERE / "final-artifact.json").read_text())
    meta = json.loads((private_root() / "gap_analysis/student_v1.meta.json").read_text())
    if meta["model"] != declaration["checkpoint"]:
        raise ValueError("scored checkpoint does not match final declaration")
    if meta["missing_labels"]:
        raise ValueError("shim omitted label logprobs; do not silently claim a clean evaluation")
    i, c = intelligence(), card()
    print(json.dumps({"systems": {k: {"I": v["I"], "ci95": v.get("ci95")} for k, v in i["systems"].items()},
                      "gain_over_v0": i["v1_gain_over_v0"], "gap_fraction_closed": i["remaining_gap_fraction_closed"],
                      "card_shared_units": c["shared_units"]}, indent=2))


if __name__ == "__main__":
    main()

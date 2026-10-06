"""Receipt demo: paired bootstrap SE / MDE under group vs source-case clusters, for an
independent-noise fixture pair and correlated fixture pairs; calibration report.
Fixtures see the gold answer; they are scorer fixtures, not models.
Run: python -X utf8 -I receipts/2026-10-06-D-m0-harness/demo.py > demo-bootstrap-ece.json"""
import json
import random
import sys

from harness import bootstrap, calibration, mde, predictors, score
from harness.cli import load_slice

rows = load_slice("dev")
base = predictors.predict(rows, "noisy", seed=6, acc=0.55)
perfect = predictors.predict(rows, "perfect", seed=1)


def improve(res, frac, seed):
    """Correlated candidate: the baseline with a random `frac` of rows answered perfectly."""
    rng = random.Random(seed)
    return {k: (perfect[k] if rng.random() < frac else v) for k, v in res.items()}


from harness.cases import case_of as _case_of

_case = _case_of(rows)
_case_by_rid = {r["_evaluation"]["run_id"]: _case(r) for r in rows}


def improve_by_case(res, frac, seed):
    """Within-source correlated candidate: whole source cases flip to perfect together."""
    rng = random.Random(seed)
    cases = sorted(set(_case_by_rid.values()))
    chosen = {c for c in cases if rng.random() < frac}
    return {k: (perfect[k] if _case_by_rid[k] in chosen else v) for k, v in res.items()}


pairs = {
    "independent_noise_0.60_vs_0.55": predictors.predict(rows, "noisy", seed=7, acc=0.60),
    "correlated_base+11pct_perfect": improve(base, 0.11, 21),
    "correlated_base+4pct_perfect": improve(base, 0.04, 22),
    "case_correlated_base+11pct_cases_perfect": improve_by_case(base, 0.11, 23),
}
out = {"dev_rows": len(rows), "B": 2000, "seed": 0, "pairs": {}}
for name, cand in pairs.items():
    entry = {}
    for cl in ("group", "case"):
        c, b = score.decompose(rows, cand, clusters=cl), score.decompose(rows, base, clusters=cl)
        r = bootstrap.compare(c, b, B=2000, seed=0)
        n = sum(len(x.clusters) for x in b.values())
        d = r["diff"]["index"]
        unpaired = (r["candidate"]["index"]["se"] ** 2 + r["baseline"]["index"]["se"] ** 2) ** 0.5
        entry[cl] = {"clusters": n, "diff": d, "unpaired_se": unpaired, "mde": mde.mde(d["se"]), "mde_table": mde.table(d["se"], n)}
    out["pairs"][name] = entry
ses = [e["case"]["diff"]["se"] for e in out["pairs"].values()]
out["mde_range_case_clusters"] = {"low": mde.mde(min(ses)), "high": mde.mde(max(ses)), "note": "low = row-correlated pair (shared errors), high = independent-noise or case-correlated pair"}
cand = pairs["independent_noise_0.60_vs_0.55"]
out["calibration_independent_0.60_onehot"] = calibration.report(calibration.fields(rows, cand))
json.dump(out, sys.stdout, indent=1)

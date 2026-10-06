"""One command to reproduce the DEV parity and MDE headline claims of this receipt.

    python -X utf8 -I receipts/2026-10-06-D-m0-harness/reproduce.py > my-reproduce.json
    python -c "import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); print('MATCH' if a==b else 'DIFF')" \
        my-reproduce.json receipts/2026-10-06-D-m0-harness/expected-reproduce.json

Prerequisites: the 0.2.1 suite built and staged (receipt README "Commands"), the worktree venv
(kit 87d4650b --no-deps + `pip install -e ./harness`), and `harness slice` run once so that
`ssdm-data/slices/dev-rows.jsonl.gz` exists (DEV rows are re-checked against the committed
manifest hashes on load). TEST is never loaded. CPU only, deterministic, ~5 minutes.
Numbers are rounded to 4 decimals so the comparison is exact across platforms.
"""

import hashlib
import json
import random
import sys

from harness import bootstrap, config, mde, predictors, score
from harness.cases import case_of
from harness.cli import load_slice
from harness.suite import load_suite, verify


def r4(x):
    return round(float(x), 4)


out = {}
v = verify(load_suite())
out["suite"] = {"official_hash_match": v["match"], "added_match": v["added_match"], "subsets_match": v["subsets_match"], "missing_index_benchmarks": v["missing_index_benchmarks"], "base_scoreable": sum(c for k, c in v["scoreable_by_catalog"].items() if int(k) < 56), "hle_rows": v["scoreable_by_catalog"].get(45)}

rows = load_slice("dev")  # raises if any row hash differs from the committed manifest
out["dev"] = {"rows": len(rows), "manifest_sha256_gz": hashlib.sha256((config.MANIFEST_DIR / "dev-manifest.jsonl.gz").read_bytes()).hexdigest()}

# 1) parity: harness decomposition vs the unmodified kit scorer, three reference predictors
perfect = predictors.predict(rows, "perfect", seed=1)
parity = {}
for name, res in (("perfect", perfect), ("uniform", predictors.predict(rows, "uniform", seed=2)), ("noisy-0.7", predictors.predict(rows, "noisy", seed=3, acc=0.7, status_drop=0.02))):
    off = score.official(rows, res)
    dec = score.point(score.decompose(rows, res), renormalize=False)
    parity[name] = {"official_index": r4(off["scores"]["balanced_skill"]), "harness_index": r4(dec["index"]), "max_abs_benchmark_skill_gap": r4(max(abs(dec["per_benchmark"][n]["skill"] - off["values"][n]["skill"]) for n in dec["per_benchmark"]))}
out["parity"] = parity

# 2) MDE range: paired case-cluster bootstrap (B=2000, seed 0) on fixture pairs
base = predictors.predict(rows, "noisy", seed=6, acc=0.55)
cid = case_of(rows)
case_by_rid = {r["_evaluation"]["run_id"]: cid(r) for r in rows}


def improve(frac, seed):
    rng = random.Random(seed)
    return {k: (perfect[k] if rng.random() < frac else v) for k, v in base.items()}


def improve_by_case(frac, seed):
    rng = random.Random(seed)
    chosen = {c for c in sorted(set(case_by_rid.values())) if rng.random() < frac}
    return {k: (perfect[k] if case_by_rid[k] in chosen else v) for k, v in base.items()}


pairs = {
    "independent_noise_0.60_vs_0.55": predictors.predict(rows, "noisy", seed=7, acc=0.60),
    "baseline+11pct_rows_fixed": improve(0.11, 21),
    "baseline+4pct_rows_fixed": improve(0.04, 22),
    "baseline+11pct_cases_fixed": improve_by_case(0.11, 23),
}
mdes = {}
b_benches = score.decompose(rows, base)
for name, cand in pairs.items():
    r = bootstrap.compare(score.decompose(rows, cand), b_benches, B=2000, seed=0)
    d = r["diff"]["index"]
    mdes[name] = {"diff": r4(d["point"]), "ci95": [r4(d["lo"]), r4(d["hi"])], "se": r4(d["se"]), "mde": r4(mde.mde(d["se"]))}
out["mde_case_clusters"] = mdes
out["mde_range"] = [min(m["mde"] for m in mdes.values()), max(m["mde"] for m in mdes.values())]
json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()

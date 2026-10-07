"""Pre-registered DEV-slice SUBSET for the m1 jobs (probe9b, rwkv7): IDs only, built from the committed DEV manifest.

The full DEV slice is 12,104 rows (~1 s/row on the HF readout, two arms = ~7 h). The m1 decision only needs a paired,
stratified subset. Construction (deterministic, no row content needed):
  1. per benchmark (catalog_id) quota = max(min_per_bench, round(n_target * rows_c / rows_DEV)), capped at rows_c;
  2. inside a benchmark, the quota is split over request-length bins in proportion to the rows in each bin
     (largest remainder; a group's bin is the bin of its longest row);
  3. inside a (benchmark, bin) stratum, kit groups are taken in sha256(f"{seed}:{catalog_id}:{group_id}") order until
     the stratum quota is met (whole groups; a group never straddles the subset boundary).
Every benchmark is present, so the slice index renormalises nothing away. TEST is never touched: the manifest read is
the DEV manifest and scoring loads only `--slice dev` rows.

CLI (scoring needs the kit + the private DEV rows, i.e. the job image):
  python -m jobs.common.dev_subset build   [--n 2000] [--seed 20261007] [--out jobs/common/dev_subset_m1.json]
  python -m jobs.common.dev_subset rows    --rows /data/slices/dev-rows.jsonl.gz --out subset.jsonl.gz
  python -m jobs.common.dev_subset score   --results R
  python -m jobs.common.dev_subset compare --candidate A --baseline B
  python -m jobs.common.dev_subset ece     --results R
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV_MANIFEST = ROOT / "harness" / "manifests" / "dev-manifest.jsonl.gz"
SUBSET = HERE / "dev_subset_m1.json"
SEED = 20261007  # != harness SLICE_SEED, so the subset order is not the slice order
LENGTH_BINS = ("<128", "128-511", "512-2047", "2048-8191", ">=8192")


def priority(seed, catalog_id, group_id):
    return hashlib.sha256(f"{seed}:{catalog_id}:{group_id}".encode()).hexdigest()


def largest_remainder(total, weights):
    s = sum(weights.values())
    if total <= 0 or s == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[: total - sum(out.values())]:
        out[k] += 1
    return out


def build_subset(entries, n_target=2000, min_per_bench=12, seed=SEED):
    """entries: DEV manifest rows (run_id, catalog_id, group_id, length_bin). Returns sorted run_ids."""
    by_cat = collections.defaultdict(list)
    for e in entries:
        by_cat[e["catalog_id"]].append(e)
    n_all = len(entries)
    chosen = []
    for c, es in sorted(by_cat.items()):
        quota = min(len(es), max(min_per_bench, round(n_target * len(es) / n_all)))
        groups = collections.defaultdict(list)
        for e in es:
            groups[str(e["group_id"])].append(e)
        # a group's bin = the bin of its longest row (as harness.slicing.group_bin), so a group never straddles bins
        gbin = {g: max(rs, key=lambda e: e.get("proxy_tokens", 0))["length_bin"] for g, rs in groups.items()}
        bins = collections.Counter()
        for g, rs in groups.items():
            bins[gbin[g]] += len(rs)
        bq = largest_remainder(quota, bins)
        for b in sorted(bins, key=str):
            got = 0
            for g in sorted((g for g in groups if gbin[g] == b), key=lambda g: priority(seed, c, g)):
                if got >= bq[b]:
                    break
                chosen += [e["run_id"] for e in groups[g]]
                got += len(groups[g])
    return sorted(chosen)


def read_manifest(path=DEV_MANIFEST):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def summary(entries, ids):
    s = set(ids)
    sub = [e for e in entries if e["run_id"] in s]
    return {"rows": len(sub), "groups": len({(e["catalog_id"], str(e["group_id"])) for e in sub}),
            "benchmarks": len({e["catalog_id"] for e in sub}),
            "by_length_bin": {b: sum(e["length_bin"] == b for e in sub) for b in LENGTH_BINS},
            "by_benchmark": {str(c): n for c, n in sorted(collections.Counter(e["catalog_id"] for e in sub).items())}}


def manifest_hashes(path=DEV_MANIFEST):
    """Both hashes, named by the bytes they cover (uncompressed text = what harness/manifests/slices.json records)."""
    gz = Path(path).read_bytes()
    return {"dev_manifest_sha256_uncompressed": hashlib.sha256(gzip.decompress(gz)).hexdigest(),
            "dev_manifest_sha256_gz": hashlib.sha256(gz).hexdigest()}


def load_subset(path=SUBSET, check_manifest=True):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if ids_sha256(d["run_ids"]) != d["ids_sha256"]:
        raise SystemExit(f"{path}: run_ids do not match ids_sha256")
    if check_manifest and manifest_hashes()["dev_manifest_sha256_uncompressed"] != d["dev_manifest_sha256_uncompressed"]:
        raise SystemExit(f"{path}: built from a different DEV manifest than {DEV_MANIFEST}")
    return d


# ----------------------------------------------------------------------------- scoring (kit + DEV rows needed)
def _rows(subset):
    from harness.cli import load_slice  # DEV only; refuses TEST without the unseal env

    ids = set(subset["run_ids"])
    rows = [r for r in load_slice("dev") if r["_evaluation"]["run_id"] in ids]
    if len(rows) != len(ids):
        raise SystemExit(f"DEV rows matched {len(rows)} of {len(ids)} subset ids")
    return rows


def cmd_score(a, subset):
    from harness import bootstrap, score
    from harness.cli import load_results

    rows = _rows(subset)
    res = load_results(a.results, subset["run_ids"], a.allow_partial)
    rep = bootstrap.replicate_skills([score.decompose(rows, res)], B=a.B, seed=a.seed)[0]
    return {"subset_ids_sha256": subset["ids_sha256"], "rows": len(rows), "slice_index": bootstrap.summarize_one(rep)}


def cmd_compare(a, subset):
    from harness import bootstrap, mde, score
    from harness.cli import load_results

    rows = _rows(subset)
    c = score.decompose(rows, load_results(a.candidate, subset["run_ids"], a.allow_partial))
    b = score.decompose(rows, load_results(a.baseline, subset["run_ids"], a.allow_partial))
    out = bootstrap.compare(c, b, B=a.B, seed=a.seed)
    out["mde"] = mde.report(out["diff"]["index"]["se"], sum(len(x.clusters) for x in c.values()))
    out["subset_ids_sha256"], out["rows"] = subset["ids_sha256"], len(rows)
    return out


def cmd_ece(a, subset):
    from harness import calibration
    from harness.cli import load_results

    return {"subset_ids_sha256": subset["ids_sha256"],
            **calibration.report(calibration.fields(_rows(subset), load_results(a.results, subset["run_ids"],
                                                                                 a.allow_partial)))}


def cmd_rows(a, subset):
    ids = set(subset["run_ids"])
    n = 0
    with gzip.open(a.rows, "rt", encoding="utf-8") as fi, gzip.open(a.out, "wt", encoding="utf-8") as fo:
        for line in fi:
            if json.loads(line)["_evaluation"]["run_id"] in ids:
                fo.write(line)
                n += 1
    if n != len(ids):
        raise SystemExit(f"wrote {n} rows, subset has {len(ids)}")
    return {"rows": n, "subset_ids_sha256": subset["ids_sha256"]}


def main(argv=None):
    p = argparse.ArgumentParser(prog="dev_subset")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--min-per-bench", type=int, default=12)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", default=str(SUBSET))
    s = sub.add_parser("rows")
    s.add_argument("--rows", required=True)
    s.add_argument("--out", required=True)
    for name, args in (("score", ["--results"]), ("compare", ["--candidate", "--baseline"]), ("ece", ["--results"])):
        s = sub.add_parser(name)
        for x in args:
            s.add_argument(x, required=True)
        s.add_argument("--B", type=int, default=2000)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--allow-partial", action="store_true",
                       help="score a partial results file (missing subset rows = unanswered); default: exit 3")
    for s in sub.choices.values():
        s.add_argument("--subset", default=str(SUBSET))
    a = p.parse_args(argv)
    if a.cmd == "build":
        entries = read_manifest()
        ids = build_subset(entries, a.n, a.min_per_bench, a.seed)
        out = {"_comment": "Pre-registered m1 DEV subset (jobs/common/dev_subset.py). IDs only; rows stay private. "
                           "dev_manifest_sha256_uncompressed = sha256 of the gunzipped JSONL text (the value "
                           "harness/manifests/slices.json records as dev.manifest_sha256); dev_manifest_sha256_gz = "
                           "sha256 of the committed .gz file bytes. ids_sha256 = sha256 of the sorted run_ids joined "
                           "by newlines.",
               "seed": a.seed, "n_target": a.n, "min_per_bench": a.min_per_bench, **manifest_hashes(),
               "ids_sha256": ids_sha256(ids),
               "summary": summary(entries, ids), "run_ids": ids}
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: out[k] for k in ("ids_sha256", "dev_manifest_sha256_uncompressed")} | {"rows": len(ids)}))
        return 0
    subset = load_subset(a.subset)
    from harness.jsonl import EXIT_RESULTS, ResultsFileError

    try:
        res = {"rows": cmd_rows, "score": cmd_score, "compare": cmd_compare, "ece": cmd_ece}[a.cmd](a, subset)
    except ResultsFileError as e:
        print(f"dev_subset {a.cmd}: results file unusable: {e}", file=sys.stderr)
        return EXIT_RESULTS
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

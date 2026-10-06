"""python -I -m harness <command>

  verify                         official hash check + per-catalog counts of the local suite
  slice   [--seed S]             build DEV/TEST manifests (IDs + hashes) and materialise DEV rows
  score   --results R [--slice dev]          official point estimate + cluster-bootstrap CI
  compare --candidate A --baseline B [--slice dev]   paired CI for A - B
  ece     --results R [--slice dev]          board ECE, debiased ECE, SmoothECE per length bin
  mde     --se SE --clusters N               MDE table
  selfcheck [--slice dev]                    parity + perfect/uniform/noisy-oracle checks

TEST rows are only loaded with --slice test AND the environment variable
SSDM_UNSEAL_TEST=1; nothing prints row contents, only IDs and aggregate numbers.
"""

import argparse
import json
import os
import sys
import time

from harness import bootstrap, calibration, cases, config, mde, predictors, score, slicing
from harness.suite import dump_json, file_hashes, index_rows, load_suite, read_rows, verify, write_rows

DEV_ROWS = config.SLICE_ROWS_DIR / "dev-rows.jsonl.gz"


def manifest_path(name):
    return config.MANIFEST_DIR / f"{name}-manifest.jsonl.gz"


UNSEAL_LOG = config.REPO_ROOT / "receipts" / "test-unseal-log.jsonl"


def log_unseal(prereg):
    """Every TEST load is appended to a committed log (who/when/why). The seal is procedural:
    anyone with the suite can read the rows, so the log is what reviewers check."""
    import datetime
    import getpass
    import subprocess

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, text=True).strip()
    except Exception:
        head = None
    UNSEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "user": getpass.getuser(), "commit": head, "prereg": prereg, "argv": sys.argv}
    with UNSEAL_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"TEST unsealed; logged to {UNSEAL_LOG} -- commit that file", file=sys.stderr)


def load_slice(name):
    if name == "test":
        if os.environ.get("SSDM_UNSEAL_TEST") != "1" or not os.environ.get("SSDM_PREREG"):
            sys.exit("TEST slice is sealed. Unseal only for a pre-registered final evaluation: set SSDM_UNSEAL_TEST=1 and SSDM_PREREG=<path or URL of the committed pre-registration>.")
        log_unseal(os.environ["SSDM_PREREG"])
    manifest = slicing.read_manifest(manifest_path(name))
    if name == "dev" and DEV_ROWS.exists():
        rows = read_rows(DEV_ROWS)
        ids = [m["run_id"] for m in manifest]
        if [r["_evaluation"]["run_id"] for r in rows] != ids:
            raise SystemExit("materialised DEV rows do not match the manifest; re-run `slice`")
        return slicing.select(rows, manifest)
    return slicing.select(index_rows(load_suite()), manifest)


def load_results(path):
    from decision_index.scoring.report import load_results as kit_load

    return kit_load(path)


def cmd_verify(a):
    rep = verify(load_suite())
    rep["files"] = file_hashes(load_suite())
    print(json.dumps(rep, indent=1))


def cmd_slice(a):
    t = time.time()
    suite = load_suite()
    rows = list(index_rows(suite))
    slices, table = slicing.build(rows, seed=a.seed, area_target=a.area_target, cap=a.cap, max_case_share=a.max_case_share)
    out = {"seed": a.seed, "area_target": a.area_target, "cap": a.cap, "max_case_share": a.max_case_share, "kit_commit": config.KIT_COMMIT, "edition": config.EDITION, "suite_files": file_hashes(suite), "index_rows_available": len(rows)}
    for name in ("dev", "test"):
        out[name] = {"manifest": manifest_path(name).name, "manifest_sha256": slicing.write_manifest(manifest_path(name), slices[name]), **slicing.summarize(slices[name])}
    dev_ids = {r["_evaluation"]["run_id"] for r in slices["dev"]}
    test_ids = {r["_evaluation"]["run_id"] for r in slices["test"]}
    dev_g = {slicing.group_key(r) for r in slices["dev"]}
    test_g = {slicing.group_key(r) for r in slices["test"]}
    out["disjoint_rows"] = not (dev_ids & test_ids)
    out["disjoint_groups"] = not (dev_g & test_g)
    out["source_overlap_dev_test"] = cases.overlap_report(slices["dev"], slices["test"])
    out["disjoint_source_cases"] = not out["source_overlap_dev_test"]
    out["allocation"] = {str(k): v for k, v in sorted(table.items())}
    dump_json(config.MANIFEST_DIR / "slices.json", out)
    config.SLICE_ROWS_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(DEV_ROWS, slices["dev"])
    print(json.dumps({k: out[k] for k in ("disjoint_rows", "disjoint_groups", "disjoint_source_cases", "source_overlap_dev_test")} | {n: {"rows": out[n]["rows"], "by_area": out[n]["by_area"]} for n in ("dev", "test")}, indent=1))
    print(f"done in {time.time() - t:.1f}s", file=sys.stderr)


def cmd_score(a):
    rows = load_slice(a.slice)
    res = load_results(a.results)
    off = score.official(rows, res)
    benches = score.decompose(rows, res)
    rep = bootstrap.replicate_skills([benches], B=a.B, seed=a.seed)[0]
    out = {"official_index_missing_as_zero": off["scores"]["balanced_skill"], "slice_index": bootstrap.summarize_one(rep)}
    print(json.dumps(out, indent=1))


def cmd_compare(a):
    rows = load_slice(a.slice)
    c = score.decompose(rows, load_results(a.candidate))
    b = score.decompose(rows, load_results(a.baseline))
    out = bootstrap.compare(c, b, B=a.B, seed=a.seed)
    out["mde"] = mde.report(out["diff"]["index"]["se"], sum(len(x.clusters) for x in c.values()))
    print(json.dumps(out, indent=1))


def cmd_ece(a):
    rows = load_slice(a.slice)
    fl = calibration.fields(rows, load_results(a.results))
    print(json.dumps(calibration.report(fl), indent=1))


def cmd_mde(a):
    print(json.dumps(mde.table(a.se, a.clusters, alpha=a.alpha, power=a.power), indent=1))


def cmd_selfcheck(a):
    rows = load_slice(a.slice)
    out = {}
    for name, res in (
        ("perfect", predictors.predict(rows, "perfect", seed=1)),
        ("uniform", predictors.predict(rows, "uniform", seed=2)),
        ("noisy-0.7", predictors.predict(rows, "noisy", seed=3, acc=0.7, status_drop=0.02)),
    ):
        off = score.official(rows, res)
        dec = score.point(score.decompose(rows, res), renormalize=False)
        out[name] = {
            "official_index": off["scores"]["balanced_skill"],
            "decomposed_index": dec["index"],
            "slice_index_renormalized": score.point(score.decompose(rows, res))["index"],
            "max_abs_benchmark_skill_gap": max(abs(dec["per_benchmark"][n]["skill"] - off["values"][n]["skill"]) for n in dec["per_benchmark"]),
        }
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    s = sub.add_parser("slice")
    s.add_argument("--seed", type=int, default=config.SLICE_SEED)
    s.add_argument("--area-target", type=int, default=2400)
    s.add_argument("--cap", type=float, default=0.45)
    s.add_argument("--max-case-share", type=float, default=0.25)
    for name in ("score", "ece"):
        s = sub.add_parser(name)
        s.add_argument("--results", required=True)
        s.add_argument("--slice", default="dev", choices=["dev", "test"])
        s.add_argument("--B", type=int, default=2000)
        s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("compare")
    s.add_argument("--candidate", required=True)
    s.add_argument("--baseline", required=True)
    s.add_argument("--slice", default="dev", choices=["dev", "test"])
    s.add_argument("--B", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("mde")
    s.add_argument("--se", type=float, required=True)
    s.add_argument("--clusters", type=int, required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.8)
    s = sub.add_parser("selfcheck")
    s.add_argument("--slice", default="dev", choices=["dev", "test"])
    a = p.parse_args(argv)
    {"verify": cmd_verify, "slice": cmd_slice, "score": cmd_score, "compare": cmd_compare, "ece": cmd_ece, "mde": cmd_mde, "selfcheck": cmd_selfcheck}[a.cmd](a)


if __name__ == "__main__":
    main()

"""DEV splits for the proxy loop (train/proxy_rule.py): the SELECT subset is jobs/common/dev_subset_m1.json; the HOLDOUT
split is built here, disjoint from it, by the same stratified procedure. IDs only (rows stay private).

    python -X utf8 -m jobs.q9_feature_cache.splits build-holdout [--out jobs/q9_feature_cache/dev_holdout.json]
    python -X utf8 -m jobs.q9_feature_cache.splits check

The output has the dev_subset file shape, so ``jobs.common.dev_subset score|compare --subset <it>`` scores it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jobs.common import dev_subset as D
from train import proxy_rule as R

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / R.HOLDOUT_SUBSET


def build_holdout(entries: list[dict], select_ids: set[str], n: int = R.HOLDOUT_N, seed: int = R.HOLDOUT_SEED,
                  min_per_bench: int = 12) -> list[str]:
    """Stratified (benchmark x length bin, whole groups) over the DEV rows outside SELECT. SELECT took whole kit groups,
    so excluding its rows excludes its groups; a group is never split between the two."""
    sel_groups = {(e["catalog_id"], str(e["group_id"])) for e in entries if e["run_id"] in select_ids}
    rest = [e for e in entries if (e["catalog_id"], str(e["group_id"])) not in sel_groups]
    return D.build_subset(rest, n, min_per_bench, seed)


def holdout_doc(entries: list[dict], ids: list[str]) -> dict:
    return {"_comment": "HOLDOUT DEV split for the proxy loop (train/proxy_rule.py; jobs/q9_feature_cache/splits.py): "
                        "disjoint from jobs/common/dev_subset_m1.json (SELECT) by kit group; same construction "
                        "(jobs.common.dev_subset.build_subset) over the remaining DEV manifest rows. IDs only.",
            "seed": R.HOLDOUT_SEED, "n_target": R.HOLDOUT_N, "min_per_bench": 12, "disjoint_from": R.SELECT_SUBSET,
            "disjoint_from_ids_sha256": R.SELECT_IDS_SHA256, **D.manifest_hashes(), "ids_sha256": D.ids_sha256(ids),
            "summary": D.summary(entries, ids), "run_ids": ids}


def check(path: Path = HOLDOUT) -> dict:
    entries = D.read_manifest()
    sel = D.load_subset(ROOT / R.SELECT_SUBSET)
    hol = D.load_subset(path)
    ids = build_holdout(entries, set(sel["run_ids"]))
    groups = lambda s: {(e["catalog_id"], str(e["group_id"])) for e in entries if e["run_id"] in s}  # noqa: E731
    out = {"rebuild_equal": ids == hol["run_ids"], "select_sha_ok": sel["ids_sha256"] == R.SELECT_IDS_SHA256,
           "row_overlap": len(set(sel["run_ids"]) & set(hol["run_ids"])),
           "group_overlap": len(groups(set(sel["run_ids"])) & groups(set(hol["run_ids"]))),
           "rows": len(hol["run_ids"]), "benchmarks": hol["summary"]["benchmarks"]}
    out["ok"] = out["rebuild_equal"] and out["select_sha_ok"] and not out["row_overlap"] and not out["group_overlap"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9_feature_cache.splits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-holdout")
    b.add_argument("--out", default=str(HOLDOUT))
    sub.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "build-holdout":
        entries = D.read_manifest()
        sel = D.load_subset(ROOT / R.SELECT_SUBSET)
        ids = build_holdout(entries, set(sel["run_ids"]))
        Path(a.out).write_text(json.dumps(holdout_doc(entries, ids), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"rows": len(ids), "ids_sha256": D.ids_sha256(ids)}))
        return 0
    res = check()
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

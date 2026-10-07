"""Final report of a feature-cache run: every arm's summary + every shard re-hashed against its MANIFEST.

    python -X utf8 -m jobs.q9_feature_cache.report --cache <out>/cache --arms "stock conv1" --result <cache_result.json>

``valid_result`` is the jobs/common/finalize.py validator: the marker is written only when every requested arm exited 0,
every split of it is present with re-verified shards, and the selfcheck passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPLITS = ("select", "holdout", "train")


def arm_report(d: Path) -> dict:
    from .extract import sha256_file

    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    out = {"exit": s.get("exit"), "reason": s.get("reason"), "selfcheck": s.get("selfcheck"),
           "volume": s.get("volume"), "timing": s.get("timing"), "train_rows": s.get("train_rows"),
           "provenance": s.get("provenance"), "dry_run": s.get("dry_run"), "splits": {}}
    for sp in SPLITS:
        man = d / sp / "MANIFEST.jsonl"
        if not man.exists():
            out["splits"][sp] = {"present": False}
            continue
        lines = [json.loads(x) for x in man.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = [m["file"] for m in lines if sha256_file(d / sp / m["file"]) != m["sha256"]]
        out["splits"][sp] = {"present": True, "shards": len(lines), "requests": sum(m["requests"] for m in lines),
                             "questions": sum(m["questions"] for m in lines), "bytes": sum(m["bytes"] for m in lines),
                             "bad_shards": bad, "manifest_sha256": sha256_file(man)}
    return out


def valid_result(r: dict) -> bool:
    try:
        if not r["arms"]:
            return False
        for a in r["arms"].values():
            if a["exit"] != 0 or not a["selfcheck"]:
                return False
            for sp in SPLITS:
                x = a["splits"][sp]
                if not x["present"] or x["bad_shards"] or x["requests"] <= 0:
                    return False
            vol = a["volume"]
            if a["splits"]["select"]["requests"] != vol["select"]["requests"]:
                return False
            if a["splits"]["holdout"]["requests"] != vol["holdout"]["requests"]:
                return False
        return True
    except (KeyError, TypeError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9_feature_cache.report")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--result", required=True)
    a = ap.parse_args(argv)
    r = {"arms": {arm: arm_report(Path(a.cache) / arm) for arm in a.arms.split()}}
    r["valid"] = valid_result(r)
    Path(a.result).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"valid": r["valid"], "arms": {k: {sp: v["splits"][sp].get("requests") for sp in SPLITS}
                                                   for k, v in r["arms"].items()}}))
    return 0 if r["valid"] else 9


if __name__ == "__main__":
    sys.exit(main())

"""Final report of a feature-cache run: every arm's summary + every shard re-hashed against its MANIFEST.

    python -X utf8 -m jobs.q9_feature_cache.report --cache <out>/cache --arms stock --splits select,holdout \
        --max-gib 4.5 --result <cache_result.json>

``valid_result`` is the jobs/common/finalize.py validator: the marker is written only when the part's arm exited 0,
every split of the part is present with re-verified shards AND the part's lex_vocab_<part>.pt (sha256 = the summary's), every part
file listed in a MANIFEST exists, the selfcheck passed, and the whole output directory is <= max_gib (review B1: the
platform packages the output as ONE tar artifact bounded by 5 GiB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPLITS = ("select", "holdout", "train")


def arm_report(d: Path, splits=SPLITS) -> dict:
    from .extract import sha256_file

    s = json.loads((d / f"summary_{'+'.join(splits)}.json").read_text(encoding="utf-8"))
    out = {"exit": s.get("exit"), "reason": s.get("reason"), "selfcheck": s.get("selfcheck"),
           "volume": s.get("volume"), "timing": s.get("timing"), "train_rows": s.get("train_rows"),
           "provenance": s.get("provenance"), "dry_run": s.get("dry_run"), "splits": {}}
    out["part_splits"] = list(splits)
    for sp in splits:
        man = d / sp / "MANIFEST.jsonl"
        if not man.exists():
            out["splits"][sp] = {"present": False}
            continue
        lines = [json.loads(x) for x in man.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = [m["file"] for m in lines if not (d / sp / m["file"]).is_file()
               or sha256_file(d / sp / m["file"]) != m["sha256"]]
        lx = s.get("lex_vocab") or {}
        lv = d / str(lx.get("file"))
        lex_ok = sp in (lx.get("splits") or []) and lv.is_file() and sha256_file(lv) == lx.get("sha256")
        out["splits"][sp] = {"present": True, "shards": len(lines), "requests": sum(m["requests"] for m in lines),
                             "questions": sum(m["questions"] for m in lines), "bytes": sum(m["bytes"] for m in lines),
                             "bad_shards": bad, "lex_vocab_ok": lex_ok, "manifest_sha256": sha256_file(man)}
    return out


def valid_result(r: dict) -> bool:
    try:
        if not r["arms"] or not r["output_bytes"] or r["output_bytes"] > r["max_bytes"]:
            return False
        for a in r["arms"].values():
            if a["exit"] != 0 or not a["selfcheck"]:
                return False
            for sp in a["part_splits"]:
                x = a["splits"][sp]
                if not x["present"] or x["bad_shards"] or x["requests"] <= 0 or not x["lex_vocab_ok"]:
                    return False
                if x["requests"] != a["volume"][sp]["requests"]:
                    return False
        return True
    except (KeyError, TypeError):
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9_feature_cache.report")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--output-dir", default=None, help="the whole job output directory (its size is the artifact's)")
    ap.add_argument("--max-gib", type=float, default=4.5)
    a = ap.parse_args(argv)
    splits = [s for s in SPLITS if s in a.splits.split(",")]
    r = {"arms": {arm: arm_report(Path(a.cache) / arm, splits) for arm in a.arms.split()}}
    od = Path(a.output_dir or Path(a.cache).parent)
    r["output_bytes"] = sum(p.stat().st_size for p in od.rglob("*") if p.is_file())
    r["max_bytes"] = int(a.max_gib * (1 << 30))
    r["valid"] = valid_result(r)
    Path(a.result).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"valid": r["valid"], "output_gib": round(r["output_bytes"] / (1 << 30), 3),
                      "arms": {k: {sp: v["splits"][sp].get("requests") for sp in splits} for k, v in r["arms"].items()}}))
    return 0 if r["valid"] else 9


if __name__ == "__main__":
    sys.exit(main())

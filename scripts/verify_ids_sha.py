"""Verify a draw's ids sha256 against an expected pin under BOTH conventions (sorted ids joined by "\\n", with and
without one trailing "\\n"). Streams records.jsonl.gz; stdlib only (no torch, low RAM). Exit 0 = a convention matches
(prints which), 1 = neither matches (STOP: do not pin, do not submit).

    python -X utf8 -I scripts/verify_ids_sha.py --records <train records.jsonl.gz> --expect feb681f3... \
        [--records <slice records.jsonl.gz> --expect 844233f2...]
"""
import argparse
import gzip
import hashlib
import json
import sys


def ids_of(path: str) -> list[str]:
    ids = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["id"])
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", required=True)
    ap.add_argument("--expect", action="append", required=True)
    a = ap.parse_args(argv)
    if len(a.records) != len(a.expect):
        raise SystemExit("one --expect per --records")
    ok = True
    out = []
    for path, want in zip(a.records, a.expect):
        ids = ids_of(path)
        s = "\n".join(sorted(ids))
        both = {"no_trailing_nl": hashlib.sha256(s.encode()).hexdigest(),
                "trailing_nl": hashlib.sha256((s + "\n").encode()).hexdigest()}
        match = next((k for k, v in both.items() if v == want), None)
        ok &= match is not None
        out.append({"records": path, "n_ids": len(ids), "n_unique": len(set(ids)), "expect": want, "both": both,
                    "matched_convention": match})
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

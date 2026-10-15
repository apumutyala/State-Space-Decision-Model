"""STRICT full-row production Index Fingerprint: every string leaf of the WHOLE row (state + instructions + criteria),
not only the item tier the build check probes. 0 hits required.

    python -X utf8 -I scripts/jev3_strict_fp.py --records R1.jsonl.gz [R2 ...] --out strict-fp.json [--ids-out f]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from data import config
    from data.dedup import Fingerprint
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out")
    a = ap.parse_args(argv)
    fp = Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    hits, by, n = {}, collections.Counter(), 0
    for path in a.records:
        for line in gzip.open(path, "rt", encoding="utf-8"):
            r = json.loads(line)
            n += 1
            h = {k: sorted(v) for k, v in fp.check(r["state"], r["questions"]).items() if v}
            if h:
                hits[r["id"]] = h
                for k in h:
                    by[f"{r['aug']['jev1']['family8']}/{k}"] += 1
            if len(getattr(fp, "_seg_cache", ())) > 200_000:
                fp._seg_cache.clear()
                fp._txt_cache.clear()
    rep = {"records": a.records, "n": n, "n_hit": len(hits), "by_family_matcher": dict(by),
           "index_fingerprint": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256", "rows")},
           "rule": "data.dedup.Fingerprint.check(state, questions) on the full row (all matchers: exact, ngram13, "
                   "minhash, edit, contain)", "hits": dict(list(hits.items())[:200])}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if a.ids_out:
        Path(a.ids_out).write_text("\n".join(sorted(hits)) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "n_hit", "by_family_matcher")}))
    return 0 if not hits else 1


if __name__ == "__main__":
    raise SystemExit(main())

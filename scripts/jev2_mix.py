"""jev1 + jev2 mix manifest: 18 draw families (jev1's 8 + jev2's 10), with equal per-family weighting as the default.

Reads each component pointer, checks the records sha256 against it (streamed), counts rows per ``aug.jev1.family8``
and tests cross-draw exact-content overlap (train.cohort.record_content_hash). It writes
data/manifests/bschema-mix-jev1-jev2.json:

  components     pointer path + pointer sha256 + records sha256 + n per draw
  families       family -> {draw, rows, weight_equal (1/18), row_weight_equal (1/(18 * rows)), weight_natural}
  options        "equal_family" (default): sample a family uniformly, then a row uniformly within it (each family 1/18
                 of the draws); "natural": concatenate all rows (family share = its row share)
  cross_draw_content_overlap   must be 0

    python -X utf8 -I scripts/jev2_mix.py --pointers data/manifests/bschema-draw-jev1.pointer.json
        data/manifests/bschema-draw-jev2a.pointer.json data/manifests/bschema-draw-jev2b.pointer.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from data import config
    from train.cohort import record_content_hash

    ap = argparse.ArgumentParser()
    ap.add_argument("--pointers", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "data" / "manifests" / "bschema-mix-jev1-jev2.json"))
    ap.add_argument("--replace", nargs="*", default=["long_policy=bschema-draw-jev2c-v2"],
                    help="family=build: take this family ONLY from that build (JEV1_AUDIT: jev1 long_policy is the "
                         "harmful template monoculture; jev2c-v2 replaces it)")
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args(argv)
    repl = dict(x.split("=", 1) for x in a.replace)
    comps, fams, owner = [], {}, {}
    seen: dict[str, str] = {}
    overlap = collections.Counter()
    for p in a.pointers:
        raw = Path(p).read_bytes()
        ptr = json.loads(raw)
        build = ptr["build"].split(" ", 1)[0]
        rec = config.BUILD_DIR / build / "records.jsonl.gz"
        h = hashlib.sha256()
        c = collections.Counter()
        n = 0
        with gzip.open(rec, "rt", encoding="utf-8") as fh:
            for line in fh:
                h.update(line.encode())
                r = json.loads(line)
                n += 1
                f = r["aug"]["jev1"]["family8"]
                if f in repl and repl[f] != build:
                    c[f + " (excluded: replaced)"] += 1
                    continue
                c[f] += 1
                ch = record_content_hash(r)
                if ch in seen and seen[ch] != build:
                    overlap[f"{seen[ch]}|{build}"] += 1
                seen.setdefault(ch, build)
        if h.hexdigest() != ptr["records_uncompressed_sha256"]:
            raise SystemExit(f"{rec}: records sha256 does not match {p}")
        comps.append({"pointer": str(Path(p).as_posix()).split("State-Space-Decision-Model/")[-1],
                      "pointer_sha256": hashlib.sha256(raw).hexdigest(), "build": build,
                      "records_uncompressed_sha256": ptr["records_uncompressed_sha256"], "n_records": n,
                      "families": dict(c)})
        for f, k in c.items():
            if f.endswith("(excluded: replaced)"):
                continue
            if f in fams:
                raise SystemExit(f"family {f} in two draws (use --replace)")
            fams[f] = k
            owner[f] = build
    total = sum(fams.values())
    nf = len(fams)
    out = {"mix": "bschema-mix-jev1-jev2", "n_families": nf, "n_records": total, "default_option": "equal_family",
           "options": {"equal_family": f"sample a family uniformly (each 1/{nf} of the draws), then a row uniformly "
                                       "within it; small families repeat rows, large ones subsample",
                       "natural": "concatenate every row once (a family's share = its row share)"},
           "components": comps,
           "families": {f: {"draw": owner[f], "rows": k, "weight_equal": round(1 / nf, 6),
                            "row_weight_equal": 1 / (nf * k), "weight_natural": round(k / total, 6)}
                        for f, k in sorted(fams.items())},
           "replace": repl, "max_len_shim_tokens": a.max_len,
           "max_len_note": "every component row is <= its build's token cap (jev1 / jev2b 2,048; jev2a-v2 / jev2c-v2 "
                           "4,096) under jrt-h2o-shim:qwen3.5-4b, so max_len 4,096 truncates nothing",
           "cross_draw_content_overlap": dict(overlap), "cross_draw_content_overlap_total": sum(overlap.values()),
           "loader_contract": "rows keep their stored criteria order (never re-sort option keys); family = "
                              "aug.jev1.family8"}
    Path(a.out).write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("n_families", "n_records", "cross_draw_content_overlap_total")}),
          hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
    return 0 if not overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())

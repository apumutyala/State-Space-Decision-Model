"""Preflight for jobs/ev_pool_sample (lint L-08 rate guard): refuse before the build if it cannot finish inside its cap.
The build streams the two 0.2.1 suite files twice; the probe streams the DEV rows once and scales by the compressed
byte ratio.

    python -m jobs.ev_pool_sample.preflight --cap-min <phase cap> [--data /data]

Exit 0 = proceed; exit 15 = projected past the cap (nothing else runs); exit 10 = inputs missing."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

#: 2 suite passes, each parsing every row and (pass 1) running harness.cases.link_keys; x2 safety on the byte ratio
PASSES, SAFETY = 2, 2.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ev_pool_sample.preflight")
    ap.add_argument("--cap-min", type=float, required=True)
    ap.add_argument("--data", default=os.environ.get("SSDM_DATA", "/data"))
    a = ap.parse_args(argv)
    dev = Path(a.data) / "slices" / "dev-rows.jsonl.gz"
    suite = [Path(a.data) / "suite-0.2" / f for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")]
    missing = [str(p) for p in [dev, *suite] if not p.is_file()]
    if missing:
        print(f"preflight: missing {missing}", file=sys.stderr)
        return 10
    t0 = time.time()
    n = 0
    with gzip.open(dev, "rt", encoding="utf-8") as f:
        for line in f:
            json.loads(line)
            n += 1
    probe = time.time() - t0
    ratio = sum(p.stat().st_size for p in suite) / dev.stat().st_size
    proj_min = probe * ratio * PASSES * SAFETY / 60.0
    rec = {"probe_rows": n, "probe_s": round(probe, 2), "byte_ratio": round(ratio, 2), "passes": PASSES,
           "safety": SAFETY, "projected_min": round(proj_min, 2), "cap_min": a.cap_min, "ok": proj_min <= a.cap_min}
    print(json.dumps(rec), flush=True)
    return 0 if rec["ok"] else 15


if __name__ == "__main__":
    sys.exit(main())

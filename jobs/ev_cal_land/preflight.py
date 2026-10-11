"""Preflight for jobs/ev_cal_land (lint L-08 rate guard): refuse before the landing phase if it cannot finish inside
its cap. The work is fixed-size (12,104 rows x a few files, B = 2000 bootstrap), so the projection is the measured
wall time of a probe (loading the DEV rows once) times a fixed multiplier for the remaining passes.

    python -m jobs.ev_cal_land.preflight --cap-min <phase cap> --passes <n> [--data /data]

Exit 0 = proceed; exit 15 = projected past the cap (nothing else runs); exit 10 = inputs missing."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: each pass = load rows + results + decompose + B = 2000 bootstrap; the bootstrap costs <= this many row-load probes
PASS_MULTIPLIER = 6.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ev_cal_land.preflight")
    ap.add_argument("--cap-min", type=float, required=True)
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--data", default=os.environ.get("SSDM_DATA", "/data"))
    a = ap.parse_args(argv)
    rows = Path(a.data) / "slices" / "dev-rows.jsonl.gz"
    if not rows.is_file():
        print(f"preflight: {rows} missing", file=sys.stderr)
        return 10
    t0 = time.time()
    from harness import cli

    n = len(cli.load_slice("dev"))
    probe = time.time() - t0
    proj_min = probe * PASS_MULTIPLIER * a.passes / 60.0
    rec = {"probe_rows": n, "probe_s": round(probe, 2), "passes": a.passes, "multiplier": PASS_MULTIPLIER,
           "projected_min": round(proj_min, 2), "cap_min": a.cap_min, "ok": proj_min <= a.cap_min}
    print(json.dumps(rec), flush=True)
    return 0 if rec["ok"] else 15


if __name__ == "__main__":
    sys.exit(main())

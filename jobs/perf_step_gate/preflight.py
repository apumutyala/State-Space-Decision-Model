"""Measured-rate guard of jobs/perf_step_gate (lint L-08), run after the baseline variant and before the others:
the baseline's median timed step must be within SLOW_FACTOR of job-tvuat's ckpt_rows_lean median on the same rows
(the same config). A slower box would make every later number incomparable with tvuat and burn the caps; exit 15.

    python -m jobs.perf_step_gate.preflight --out DIR      # reads DIR/variant_greedy.json, writes DIR/preflight.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

#: job-tvuat ckpt_rows_lean median step seconds over the same 20 timed steps (receipts/2026-10-07-D-cell-step-profile-tvuat)
TVUAT_MEDIAN_STEP_S = 8.2475
SLOW_FACTOR = 1.5
DRY_OK = "dryrun"


def check(rec: dict) -> dict:
    timed = [s["s"] for s in rec.get("steps", []) if s.get("timed") and "loss" in s and not s.get("oom")]
    if rec.get("env", {}).get("dryrun"):
        return {"ok": True, "basis": DRY_OK, "median_step_s": statistics.median(timed) if timed else None}
    if not timed:
        return {"ok": False, "reason": "baseline completed no timed step"}
    m = statistics.median(timed)
    return {"ok": m <= SLOW_FACTOR * TVUAT_MEDIAN_STEP_S, "median_step_s": m, "tvuat_median_step_s": TVUAT_MEDIAN_STEP_S,
            "slow_factor": SLOW_FACTOR, "ratio": m / TVUAT_MEDIAN_STEP_S}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    p = a.out / "variant_greedy.json"
    res = check(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else {"ok": False, "reason": "no record"}
    (a.out / "preflight.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0 if res["ok"] else 15


if __name__ == "__main__":
    sys.exit(main())

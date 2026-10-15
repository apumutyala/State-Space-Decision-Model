"""Reproduction check: the (7) flip battery of scripts/ev_cell_rescore.py vs claude/ev-flip-rate @4a65dd18
receipts/2026-10-10-ev-flip-rate/run2/flip_ci_8op.json (B-PREREG-FLIP run 2, job-z8jcd). The kit is run with
``--flip8 <h2o preds>,<h2o preds_pert> --flip-baseline qwen35_4b_shim=<qwen preds>,<qwen preds_pert>``; every point,
CI and paired delta must be EXACTLY equal.

    python -X utf8 receipts/2026-10-11-ev-cell-kit/repro_flip.py --kit <out>/cell_rescore.json --flip-json <run2/flip_ci_8op.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--flip-json", required=True)
    a = ap.parse_args(argv)
    k = json.loads(Path(a.kit).read_text(encoding="utf-8"))["7_flips"]
    r = json.loads(Path(a.flip_json).read_text(encoding="utf-8"))
    pairs = [("cell", "h2o_lightning_4b"), ("qwen35_4b_shim", "qwen35_4b_shim")]
    n = bad = 0
    for km, rm in pairs:
        for key in ("point", "ci95", "n_valid", "n_invalid"):
            for op, v in r["models"][rm][key].items():
                n += 1
                if k["models"][km][key].get(op) != v:
                    bad += 1
                    print(f"DIFF {km}.{key}.{op}: kit {k['models'][km][key].get(op)} vs receipt {v}")
    pm = r["paired_minus_qwen35_4b_shim"]["h2o_lightning_4b"]
    for op, v in pm.items():
        n += 1
        if k["models"]["cell"]["paired_minus_qwen35_4b_shim"][op] != v:
            bad += 1
            print(f"DIFF paired.{op}: kit {k['models']['cell']['paired_minus_qwen35_4b_shim'][op]} vs receipt {v}")
    print(f"{'REPRODUCED' if not bad else 'NOT REPRODUCED'}: {n - bad}/{n} flip values exactly equal "
          f"(qwen35_4b_shim average {100 * r['models']['qwen35_4b_shim']['point']['_average']:.2f}%)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

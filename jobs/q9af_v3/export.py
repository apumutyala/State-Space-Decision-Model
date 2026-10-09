"""Export a Q9-AF v3 GDN checkpoint (stage A or the final job-B model) with jobs/q9af_v2/export.py's verified exporter
(same file format, sha256 on two reads, tensor re-load equality, HF strict load; consumed by jobs.q9af_cell unchanged).

    python -m jobs.q9af_v3.export --src <ckpt>.pt --model-dir <9B snapshot> --out-dir <dir> --meta-out <json> \
        [--mode final|salvage] [--stage A|B]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jobs.q9af_v2.export import export


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_v3.export")
    ap.add_argument("--src", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--mode", choices=["final", "salvage"], default="final")
    ap.add_argument("--stage", choices=["A", "B"], default="B")
    a = ap.parse_args(argv)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    try:
        meta = export(a.src, a.model_dir, a.out_dir, a.mode)
        meta.update({"ok": True, "producer": f"jobs/q9af_v3 stage {a.stage} (convert.v3)", "stage": a.stage})
    except Exception as e:  # noqa: BLE001
        meta = {"mode": a.mode, "ok": False, "stage": a.stage, "error": repr(e)[:500]}
    Path(a.meta_out).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0 if meta["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())

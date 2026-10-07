"""Export pin of the Q9-AF cell: fill it when job-2xe5w lands, check it on the box.

    python -m jobs.q9af_cell.pin fill --export-json <export.json> --gdn <gdn_converted.pt|gdn_partial.pt> [--hybrid]
        writes converted_sha256 / export_json_sha256 / converted_from into BOTH cell configs (q9af_cell_r64/r256.json)
        and gdn.sha256 (the data pin of the launch context). Refuses a partial export unless --hybrid, and --hybrid on a
        full export (jobs.q9af_cell.hooks.check_export). Commit the result in its own commit before any launch.
    python -m jobs.q9af_cell.pin check --config <cell.json> --export-json <export.json> --gdn <pt> --out <label.json>
        the runner's setup check (exit 10 on any refusal); label.json carries the attention-free / HYBRID label that
        every report repeats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jobs.q9af_cell.hooks import check_export, sha256

HERE = Path(__file__).resolve().parent
CONFIGS = ("q9af_cell_r64.json", "q9af_cell_r256.json")


def fill(export_json: str, gdn: str, hybrid: bool = False, configs=CONFIGS, root: Path = HERE) -> dict:
    meta = json.loads(Path(export_json).read_text(encoding="utf-8"))
    pins = {"converted_sha256": sha256(gdn), "export_json_sha256": sha256(export_json),
            "converted_from": f"job-2xe5w {meta.get('file')} (jobs/full8 export_gdn, mode {meta.get('mode')}, "
                              f"converted {meta.get('converted')})"}
    out = {}
    for name in configs:
        p = root / name
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["q9af"] = {**cfg["q9af"], **pins, "hybrid": bool(hybrid)}
        out[name] = check_export(cfg, export_json, gdn)  # refuses before anything is written
        p.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    fname = meta.get("file") or Path(gdn).name
    (root / "gdn.sha256").write_text(f"{pins['converted_sha256']}  gdn/{fname}\n"
                                     f"{pins['export_json_sha256']}  gdn/export.json\n", encoding="utf-8", newline="\n")
    return {"pins": pins, "labels": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.q9af_cell.pin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill")
    f.add_argument("--export-json", required=True)
    f.add_argument("--gdn", required=True)
    f.add_argument("--hybrid", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--config", required=True)
    c.add_argument("--export-json", required=True)
    c.add_argument("--gdn", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "fill":
            print(json.dumps(fill(a.export_json, a.gdn, a.hybrid), indent=1))
            return 0
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        label = check_export(cfg, a.export_json, a.gdn)
    except (ValueError, OSError, KeyError) as e:
        print(f"export pin: {e}", file=sys.stderr)
        return 10
    Path(a.out).write_text(json.dumps(label, indent=1), encoding="utf-8")
    print(json.dumps(label))
    return 0


if __name__ == "__main__":
    sys.exit(main())

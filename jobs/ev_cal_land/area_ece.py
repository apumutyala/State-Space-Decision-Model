"""Full-DEV calibration per Decision-Index area for several results files (EV-PREREG-AREA-TEMP reporting).

    python -I -X utf8 jobs/ev_cal_land/area_ece.py --out area_ece.json name=results.jsonl [name=results.jsonl ...]

Everything goes through the frozen harness: rows from ``harness.cli.load_slice("dev")`` (0.2.1), results through the
strict ``cli.load_results``, fields through ``harness.calibration.fields`` (board field rule: board-excluded benchmarks
dropped, noul confidence max(p, 1-p)), area labels from ``harness.suite.area_of("0.2.1")`` (fixed catalog map).
Reported per file: overall and per area n, accuracy, mean confidence, ECE 10-bin (equal width), debiased L2 ECE
(15 equal-mass bins) and the board-protocol ECE. Top-1 accuracy per file is reported so tempered vs untempered can be
asserted equal by the caller. CPU only; result files are data (python -I)."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from harness import calibration, cli, suite  # noqa: E402


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def block(fl: list) -> dict:
    if len(fl) < 2:
        return {"n": len(fl), "available": False}
    conf, cor = [f[3] for f in fl], [f[4] for f in fl]
    ece10, _ = calibration.binned_ece(conf, cor)
    return {"n": len(fl), "acc": sum(cor) / len(cor), "conf": sum(conf) / len(conf), "ece_10bin": ece10,
            "debiased15": calibration.debiased_ece(conf, cor, bins=15)["debiased_l2"]}


def run(files: dict[str, str]) -> dict:
    rows = cli.load_slice("dev")
    ids = cli._ids(rows)
    area = suite.area_of("0.2.1")
    out = {"rows": len(rows), "edition": "0.2.1", "files": {}, "overall": {}, "by_area": {}, "board_ece": {}}
    for name, path in files.items():
        res = cli.load_results(path, ids, False)
        fl = calibration.fields(rows, res)
        out["files"][name] = {"path": path, "sha256": _sha(path)}
        out["overall"][name] = block(fl)
        out["board_ece"][name] = calibration.board_ece(fl).get("ece")
        by = collections.defaultdict(list)
        for f in fl:
            by[area.get(f[0], "unmapped")].append(f)
        out["by_area"][name] = {a: block(by[a]) for a in sorted(by)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="area_ece")
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+", help="name=path")
    a = ap.parse_args(argv)
    files = dict(x.split("=", 1) for x in a.files)
    out = run(files)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name in files:
        o = out["overall"][name]
        print(name, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in o.items()}),
              "board", round(out["board_ece"][name] or float("nan"), 5))
        for ar, b in out["by_area"][name].items():
            print("  ", ar, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in b.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

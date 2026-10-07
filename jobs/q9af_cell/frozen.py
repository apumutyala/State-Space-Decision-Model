"""Frozen-family check of the cell's training draw (heldout-draw-v1, INDEX_0.3.md: no training record may target one of
the 9 FROZEN held-out benchmarks of the new-domain proxy).

    python -X utf8 -m jobs.q9af_cell.frozen --config <cell.json> --train-dir <150k dir> --out <frozen.json>
    python -X utf8 -m jobs.q9af_cell.frozen --config <cell.json> --toy-records <toy.jsonl.gz> --out <frozen.json>   (dry run)

Re-draws exactly the cell's records (``train.train_set.draw``: pointer, cohort v3 exclusion, pinned held-out
exclusions) and runs ``harness.heldout.check_record`` on every drawn record in its ORIGINAL form (manifest-verified
line). Exit 0 = no record targets a frozen family; 34 = at least one does (ids listed); 10 = the draw itself refused.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def check(records, check_record) -> dict:
    hits, n = [], 0
    for r in records:
        n += 1
        try:
            check_record(r)
        except Exception as e:  # noqa: BLE001 - FrozenFamilyError (harness.heldout)
            hits.append({"id": r.get("id"), "error": str(e)[:300]})
    return {"n_checked": n, "hits": hits, "pass": not hits and n > 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.q9af_cell.frozen")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--toy-records", default=None, help="dry run only: synthetic records instead of the 150k draw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from harness.heldout import check_record

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.toy_records:
        with gzip.open(a.toy_records, "rt", encoding="utf-8") as f:
            recs = [json.loads(x) for x in f if x.strip()]
        ident = {"train_set": "DRY RUN: synthetic toy records (not the 150k set)"}
    else:
        from train.data import load_m2_records, read_manifest
        from train.train_set import TrainSetMismatch, draw, files, read_pointer

        try:
            _, ids, ident = draw(cfg, a.train_dir)
        except ValueError as e:  # TrainSetMismatch / CohortLeak
            print(f"draw: {e}", file=sys.stderr)
            return 10
        fs = files(a.train_dir, read_pointer(cfg["train_set"]["pointer"]))
        want = set(ids)
        recs, _ = load_m2_records(fs["records"], read_manifest(fs["manifest"]), keep=lambda r: r["id"] in want)
        if {r["id"] for r in recs} != want:
            raise TrainSetMismatch("drawn records missing on re-read")
    rep = check(recs, check_record)
    rep["draw"] = {k: ident.get(k) for k in ("train_set", "n_drawn", "drawn_ids_sha256", "cohort") if k in ident}
    from harness import config as hc

    rep["heldout_draw_v1_sha256"] = getattr(hc, "HELDOUT_DRAW_V1_SHA256", None)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"n_checked": rep["n_checked"], "hits": len(rep["hits"]), "pass": rep["pass"]}))
    return 0 if rep["pass"] else 34


if __name__ == "__main__":
    sys.exit(main())

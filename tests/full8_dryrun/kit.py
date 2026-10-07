"""Dry-run stand-in for `python -m decision_index run` (the pinned kit is GPU-image only; the DEV rows are private).

It builds the arm's model through the REAL engine path (jobs.common.readout_engine.build_plugin -> the REAL
convert.readout.load_converted_plugin), with only convert.model.load swapped for the tiny stand-in on CPU, so the
converted arm really loads the EXPORTED GDN file the runner hands it (keys, layer list, sha256 into the revision).
It then runs the plugin's forward and its prefix-fork state ops once, and writes one result line per subset row
(labelled dry run). Scores are not computed here (tests.full8_dryrun.stubs subset).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path


def main(argv=None):
    if os.path.isdir("/opt/ssdm/m1"):
        raise SystemExit("dry-run kit refused inside the job image")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--option", action="append", default=[])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    opts = dict(o.split("=", 1) for o in a.option)
    import torch

    import convert.model as CM
    from jobs.common import readout_engine
    from tests.full8_dryrun.tiny import tiny_convertible, tokenizer

    CM.load = lambda model_dir=None, device="cuda", dtype=None, **k: (tiny_convertible(0), tokenizer(),
                                                                      CM.resolve_model_dir(model_dir))
    assert a.engine == "jobs.common.readout_engine:ReadoutIndexEngine", a.engine
    plugin = readout_engine.build_plugin(opts["family"], {**opts, "device": "cpu", "dtype": "float32"})
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    logits = plugin.forward(ids, 1)
    _, st = plugin.state_ops.prefill(ids)
    ext, _ = plugin.state_ops.extend(st, torch.tensor([[11, 12], [13, 14]]), 2)
    assert torch.isfinite(logits).all() and torch.isfinite(ext).all()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"dry_run": True, "notes": plugin.notes, "files": plugin.files,
                                                 "model_id": plugin.model_id}, indent=1, default=str))
    with gzip.open(a.rows, "rt", encoding="utf-8") as fi, open(out / "results.jsonl", "w", encoding="utf-8") as fo:
        for line in fi:
            rid = json.loads(line)["_evaluation"]["run_id"]
            fo.write(json.dumps({"run_id": rid, "dry_run": True}) + "\n")
    print(json.dumps({"dry_run": True, "converted_layers": plugin.notes["converted_layers"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load a draw with the TRAINER's own loader (no sidecar): ``train.train_set.draw`` + ``heldout_gate`` from a trainer
checkout (claude/b-train-v3w-template @9c3e435 lineage), using a cell config whose ``train_set`` block names the draw's
pointer (``heldout_excluded`` emptied: the three 150k-set D1-B ids are not in a b_schema draw). CPU only.

    python -X utf8 -I scripts/b_schema_trainer_draw.py --trainer-root <checkout> --cell <cell json in that checkout>
        --pointer <abs pointer path> --train-dir <draw dir> --n 24000 [--n 300000] --out <receipt json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-root", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--n", type=int, action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.trainer_root)
    from train.train_set import draw, heldout_gate

    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    cfg["train_set"] = {"pointer": a.pointer}
    rep = {"trainer_root": a.trainer_root, "cell": a.cell, "pointer": a.pointer, "draws": []}
    for n in a.n:
        cfg["n_train_records"] = n
        recs, ids, ident = draw(cfg, a.train_dir)
        rep["draws"].append({"n": n, "n_drawn": len(ids), "identity": ident,
                             "first_ids": ids[:3], "n_questions": sum(len(r["labels"]) for r in recs.values())})
        del recs
    hg = Path(a.out).with_name(Path(a.out).stem + "-heldout-gate.json")
    rep["heldout_gate_exit"] = heldout_gate(cfg, a.train_dir, hg)
    rep["pass"] = rep["heldout_gate_exit"] == 0 and all(d["n_drawn"] == d["n"] for d in rep["draws"])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], "draws": [(d["n"], d["n_drawn"]) for d in rep["draws"]]}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

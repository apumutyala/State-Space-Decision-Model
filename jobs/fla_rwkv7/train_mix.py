"""The m2 first cell's training-row SHAPES, for the fla_rwkv7 ``truth_v2`` training-speed phase (no row content).

    # local, CPU, once ($0): the first (warmup + timed) x grad_accum rows of the cell's own epoch-0 data order
    python -X utf8 -B -m jobs.fla_rwkv7.train_mix build --code-root <m2-first-cell checkout> \
        --cell-config <m2-first-cell checkout>/jobs/m2_first_cell/m2_first_cell_rwkv7.json \
        --train-dir <dir with the pointer-pinned 150k files> --tokenizer-dir <RWKV7-G1k tokenizer dir>

The training-speed phase must time the cell's ACTUAL step (``train.trainer.Trainer.train_step``: fork head in cache mode,
LoRA, bf16 autocast, batch x grad_accum, no checkpointing) on the cell's actual row-length / option-count mix. The rows
themselves are private training data and do not ship. What ships (``train_mix_v2.json``) is, per row, only integers:
prefix length, continuation lengths, the spans the head pools over and the question type (no gold label). The job
rebuilds rows of exactly those shapes with token ids drawn from a fixed generator; the RWKV-7 step's cost depends on the
shapes, not on the ids.

Which rows: the cell draws ``n_train_records`` records (``train.train_set.draw``), lays them out (``cell_items``, cap
``max(t_ladder)``) and the Trainer visits them in ``randperm(len(items), seed * 100003 + epoch)`` order, one row per
micro-batch (batch_size 1). The first (WARMUP + TIMED) x grad_accum rows of epoch 0 are the rows of the cell's first
optimiser steps (assuming the ladder keeps ``T_max = max(t_ladder)``, the cell's intended cap). ``guard`` is not applied
here: it only raises on a protected record and never changes a row; the real cell applies it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "train_mix_v2.json"
#: the TrainConfig fields of the cell that the timed step depends on (copied from the cell config at build time)
CELL_KEYS = ("head", "head_kwargs", "layout", "template", "loss_arm", "lora_rank", "lr_head", "lr_lora",
             "warmup_steps", "batch_size", "grad_accum", "max_k", "t_ladder", "seed", "n_train_records")


def row_shape(item) -> dict:
    """Integers (and the question type) only: what the timed step's cost (and the head's pooling) depends on."""
    r = item.row
    if r.layout != "ordered":
        raise ValueError("the m2 cell uses the ordered layout")
    return {"T": len(r.ids), "conts": [len(c) for c in r.continuations], "qtype": r.qtype,
            "state_span": list(r.state_span), "field_span": list(r.field_span),
            "option_spans": [list(s) for s in r.option_spans]}


def rows_from_shapes(shapes: list[dict], vocab: int, seed: int = 0, template: str = "synthetic"):
    """train.data.Item rows with the recorded shapes and token ids in [1, vocab) from a fixed generator (0 is pad)."""
    import torch

    from heads.layout import LayoutRow
    from train.data import Item

    g = torch.Generator().manual_seed(seed)
    ids = lambda n: torch.randint(1, vocab, (n,), generator=g).tolist()  # noqa: E731
    out = []
    for i, s in enumerate(shapes):
        k = len(s["conts"])
        row = LayoutRow(qid=f"q{i}", qtype=s["qtype"], option_ids=[str(j) for j in range(k)], ids=ids(s["T"]),
                        state_span=tuple(s["state_span"]), field_span=tuple(s["field_span"]),
                        option_spans=[tuple(x) for x in s["option_spans"]], option_seqs=[],
                        continuations=[ids(n) for n in s["conts"]], layout="ordered",
                        meta={"template": template, "encoder": {"kind": "synthetic-ids", "vocab": vocab, "seed": seed}})
        out.append(Item(f"mix/{i}", row, 0))  # gold index: not shipped; the step's cost does not depend on it
    return out


def summary(shapes: list[dict]) -> dict:
    n = len(shapes)
    return {"n": n, "mean_T": sum(s["T"] for s in shapes) / max(n, 1), "max_T": max((s["T"] for s in shapes), default=0),
            "mean_K": sum(len(s["conts"]) for s in shapes) / max(n, 1),
            "max_K": max((len(s["conts"]) for s in shapes), default=0),
            "tokens": sum(s["T"] for s in shapes), "cont_tokens": sum(sum(s["conts"]) for s in shapes)}


def mix_sha256(d: dict) -> str:
    return hashlib.sha256(json.dumps({"cell": d["cell"], "rows": d["rows"]}, sort_keys=True).encode()).hexdigest()


def load(path: Path | None = None) -> dict:
    d = json.loads(Path(path or OUT).read_text(encoding="utf-8"))
    if d.get("sha256") != mix_sha256(d):
        raise ValueError("train_mix_v2.json content does not match its sha256")
    return d


def build(cell_config: str, train_dir: str, tokenizer_dir: str, n_rows: int) -> dict:
    import torch

    from train.run_cell import cell_items, parent_encoder

    cfg_bytes = Path(cell_config).read_bytes()
    cfg = json.loads(cfg_bytes)
    if cfg.get("batch_size") != 1:
        raise SystemExit("the shape table assumes one row per micro-batch (batch_size 1)")
    train_recs, tr_ids, ident = _draw(cfg, train_dir)
    enc = parent_encoder("rwkv7", tokenizer_dir)
    # workers=1: spawned workers would re-import this module by name from the --code-root checkout, which lacks it
    items = cell_items(train_recs, tr_ids, cfg, enc, None, max(cfg["t_ladder"]), workers=1)
    g = torch.Generator().manual_seed(cfg["seed"] * 100_003 + 0)  # train.trainer.Trainer._order(epoch=0)
    order = torch.randperm(len(items), generator=g).tolist()
    shapes = [row_shape(items[i]) for i in order[:n_rows]]
    d = {"_comment": __doc__.split("\n\n")[0], "cell_config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
         "cell": {k: cfg.get(k) for k in CELL_KEYS}, "train_set": ident, "n_items": len(items),
         "items_summary": summary([row_shape(x) for x in items]), "order": "randperm(n_items, seed*100003+0)[:n]",
         "rows": shapes, "rows_summary": summary(shapes)}
    d["sha256"] = mix_sha256(d)
    return d


def _draw(cfg: dict, train_dir: str):
    from train.cohort import Cohort
    from train.train_set import draw

    return draw(cfg, train_dir, Cohort(expected_sha256=cfg["cohort_sha256"]))  # (records by id, ids, identity)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("--code-root", required=True, help="checkout of the m2 cell's code (train.train_set lives there)")
    ap.add_argument("--cell-config", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--rows", type=int, default=None, help="default: (V2_TRAIN_WARMUP + V2_TRAIN_STEPS) x grad_accum")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.code_root).resolve()))
    from jobs.fla_rwkv7 import fla_equivalence as E

    cfg = json.loads(Path(a.cell_config).read_text(encoding="utf-8"))
    n = a.rows or (E.V2_TRAIN_WARMUP + E.V2_TRAIN_STEPS) * cfg["grad_accum"] * cfg["batch_size"]
    d = build(a.cell_config, a.train_dir, a.tokenizer_dir, n)
    Path(a.out).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({k: d[k] for k in ("n_items", "items_summary", "rows_summary", "sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

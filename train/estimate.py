"""Memory / throughput / cost estimator for m2 head training on one H100-80GB (planning only).

Every number here is an ESTIMATE from analytic formulas and stated assumptions, not a
measurement. The first paid rung (train/M2_RUN_SPEC.md) exists to replace these with receipts.

Model of a training step (frozen bf16 backbone, LoRA, head, no LM head):
* weights          params * 2 B
* LoRA             rank * (in + out) per targeted projection; fp32 master + grad + Adam m, v = 16 B/param
* head             parameters counted by instantiating the real head on the ``meta`` device; 16 B/param
* backbone acts    with checkpointing: one bf16 [B, T, D] tensor per layer boundary + one layer's
                   working set (``WORK_FACTOR`` x [B, T, D] bf16) + delta-rule chunk states for that
                   layer's backward ((T / 64) states of ``state_scalars_per_layer`` fp32)
                   without checkpointing: every layer's working set
* head acts        autograd keeps per-chunk tensors for the whole sequence (see per-head formulas)
* fork             cache-copy fork: K copies of the full recurrent state + K continuation rows;
                   recompute fork (what heads/features.py does today): K full rows
* FLOPs/token      frozen backbone: 2N forward + 2N input-gradient backward (+ 2N recompute with
                   checkpointing); LoRA adds 6 * lora_params; head adds 3x its forward FLOPs/token
* throughput       mfu_assumed (per backbone spec) * 989 TFLOP/s dense bf16
* cost             $2.97/h batch, $3.96/h interactive (SF Compute rate card, B-review §1 row 10)

    python -m train.estimate --json train/estimates/m2_estimates.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from heads.backbone import BACKBONES, BackboneSpec
from heads.heads import make_head
from heads.typed import CumulativeLink

from .lora import lora_param_count

GPU = {"name": "H100-80GB SXM", "mem_gb": 80.0, "bf16_dense_tflops": 989.0, "usable_gb": 76.0}
RATE = {"batch": 2.97, "interactive": 3.96}
WORK_FACTOR = 24  # one layer's live activations in units of a [B,T,D] bf16 tensor (assumption)
CHUNK = 64  # delta-rule chunk length for chunkwise kernels
GB = 1024**3

HEAD_DEFAULTS = {
    "integrator": {"channels": 64},
    "fork": {"hidden": 256},
    "snapshot": {"channels": 64, "max_snapshots": 8},
    "head_b": {"d_state": 64},
    "clef_control": {"width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096},
}


def head_params(name: str, d_model: int) -> int:
    with torch.device("meta"):
        h = make_head(name, d_model, **HEAD_DEFAULTS[name])
        t = CumulativeLink(d_model)
    return sum(p.numel() for p in h.parameters()) + sum(p.numel() for p in t.parameters())


def head_cost(name: str, d_model: int, T: int, K: int) -> dict:
    with torch.device("meta"):
        h = make_head(name, d_model, **HEAD_DEFAULTS[name])
    return h.cost(T, K)


def head_activation_bytes(name: str, B: int, T: int, K: int, d_model: int) -> float:
    if name == "integrator":
        return B * T * K * HEAD_DEFAULTS[name]["channels"] * 4 * 3  # u*v product, silu, gate
    if name == "snapshot":
        return B * T * K * HEAD_DEFAULTS[name]["channels"] * 4 * 3 + B * T * 8 * 4
    if name == "head_b":
        return B * T * K * HEAD_DEFAULTS[name]["d_state"] * 4 * 6  # log_a, cumsum, decay, b, product, pre-acts
    if name == "fork":
        return B * K * d_model * 4 * 4
    if name == "clef_control":
        c = HEAD_DEFAULTS[name]
        return B * (c["routing_layers"] * (c["heads"] * K * T * 4 * 2 + 2 * T * c["width"] * 2) + T * c["width"] * 2)
    raise KeyError(name)


def train_point(spec: BackboneSpec, head: str, T: int, K: int, B: int = 1, rank: int = 256,
                grad_ckpt: bool = True, fork_mode: str = "cache", cont_len: int = 8) -> dict:
    D, L = spec.d_model, spec.n_layers
    weights = spec.params * 2
    lora = lora_param_count(spec.lora_shapes, rank, L)
    hp = head_params(head, D)
    act_unit = B * T * D * 2
    chunk_states = B * (T / CHUNK) * spec.state_scalars_per_layer * spec.state_dtype_bytes
    acts = (L * act_unit + WORK_FACTOR * act_unit + chunk_states) if grad_ckpt else L * (WORK_FACTOR * act_unit + chunk_states)
    hacts = head_activation_bytes(head, B, T, K, D)
    fork_bytes, fork_tokens = 0.0, 0
    if head == "fork":
        if fork_mode == "cache":
            state_bytes = L * spec.state_scalars_per_layer * spec.state_dtype_bytes
            fork_bytes = B * K * (state_bytes + (L if grad_ckpt else L * WORK_FACTOR) * cont_len * D * 2)
            fork_tokens = K * cont_len
        else:
            fork_bytes = K * acts
            fork_tokens = K * (T + cont_len)
    overhead = 2.0 * GB
    total = weights + 16 * (lora + hp) + acts + hacts + fork_bytes + overhead
    per_tok_head = head_cost(head, D, T, K)["flops"] / T
    f_tok = (6 if grad_ckpt else 4) * spec.params + 6 * lora + 3 * per_tok_head
    tokens_per_row = T + fork_tokens
    tps = spec.mfu_assumed * GPU["bf16_dense_tflops"] * 1e12 / f_tok * T / tokens_per_row  # prefix tokens/s
    return {
        "backbone": spec.name, "head": head, "T": T, "K": K, "B": B, "lora_rank": rank, "grad_ckpt": grad_ckpt,
        "mem_gb": {"weights": weights / GB, "lora_opt": 16 * lora / GB, "head_opt": 16 * hp / GB,
                   "backbone_acts": acts / GB, "head_acts": hacts / GB, "fork": fork_bytes / GB,
                   "overhead": overhead / GB, "total": total / GB},
        "fits_80gb": total / GB <= GPU["usable_gb"],
        "lora_params": lora, "head_params": hp,
        "train_flops_per_prefix_token": f_tok * tokens_per_row / T,
        "train_prefix_tokens_per_s": tps,
        "usd_per_1m_train_tokens_batch": 1e6 / tps / 3600 * RATE["batch"],
        "attention_free_head": head != "clef_control",
    }


def max_batch(spec: BackboneSpec, head: str, T: int, K: int, **kw) -> int:
    b = 0
    while b < 256 and train_point(spec, head, T, K, B=b + 1, **kw)["fits_80gb"]:
        b += 1
    return b


def eval_tokens_per_s(spec: BackboneSpec) -> float:
    return spec.mfu_assumed * GPU["bf16_dense_tflops"] * 1e12 / (2 * spec.params)


def cell_cost(spec: BackboneSpec, head: str, rows: int, T: int, K: int, epochs: float = 1.0, **kw) -> dict:
    p = train_point(spec, head, T, K, **kw)
    tokens = rows * T * epochs
    hours = tokens / p["train_prefix_tokens_per_s"] / 3600
    return {"backbone": spec.name, "head": head, "rows": rows, "T": T, "K": K, "epochs": epochs,
            "train_tokens": tokens, "gpu_hours": hours, "usd_batch": hours * RATE["batch"],
            "max_micro_batch": max_batch(spec, head, T, K, **kw)}


def report() -> dict:
    out: dict = {"gpu": GPU, "rates_usd_per_h": RATE, "assumptions": {
        "work_factor": WORK_FACTOR, "delta_chunk": CHUNK, "lm_head": "not run (heads read hidden states)",
        "status": "ESTIMATE ONLY - replace with receipts from the m2 qualification rung"},
        "backbones": {k: asdict(v) for k, v in BACKBONES.items()}, "points": [], "max_batch": [], "cells": []}
    for name, spec in BACKBONES.items():
        out["backbones"][name]["eval_tokens_per_s"] = eval_tokens_per_s(spec)
        for head in HEAD_DEFAULTS:
            for T, K in ((2048, 4), (8192, 16), (8192, 255)):
                out["points"].append(train_point(spec, head, T, K))
            out["max_batch"].append({"backbone": name, "head": head,
                                     **{f"T{T}_K{K}": max_batch(spec, head, T, K) for T, K in ((2048, 4), (8192, 16))}})
            # m2 cell under the pilot-data assumption: 10k records x 3 questions, T=2048, K=6
            out["cells"].append(cell_cost(spec, head, rows=30_000, T=2048, K=6))
    return out


def _table(rep: dict) -> str:
    lines = ["| backbone | head | T | K | mem GB (B=1) | fits | prefix tok/s | $/1M tok |", "|---|---|---|---|---|---|---|---|"]
    for p in rep["points"]:
        lines.append(f"| {p['backbone']} | {p['head']} | {p['T']} | {p['K']} | {p['mem_gb']['total']:.1f} | "
                     f"{'y' if p['fits_80gb'] else 'NO'} | {p['train_prefix_tokens_per_s']:.0f} | "
                     f"{p['usd_per_1m_train_tokens_batch']:.3f} |")
    lines += ["", "| backbone | head | GPU-h per cell | $ per cell (batch) | max micro-batch |", "|---|---|---|---|---|"]
    for c in rep["cells"]:
        lines.append(f"| {c['backbone']} | {c['head']} | {c['gpu_hours']:.2f} | {c['usd_batch']:.2f} | {c['max_micro_batch']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--table", action="store_true", help="print a markdown summary")
    a = ap.parse_args(argv)
    rep = report()
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if a.table or not a.json:
        print(_table(rep))


if __name__ == "__main__":
    main()

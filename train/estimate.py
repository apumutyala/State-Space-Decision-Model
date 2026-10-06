"""Memory / throughput / cost estimator for m2 head training on one H100-80GB (planning only).

Every number here is an ESTIMATE from analytic formulas and stated assumptions, not a
measurement. The first paid rung (train/M2_RUN_SPEC.md) exists to replace these with receipts.

Model of a training step (frozen bf16 backbone, LoRA, head, no LM head):
* weights          resident params (embeddings, no LM head) * 2 B
* LoRA             rank * (in + out) per targeted projection; fp32 master + grad + Adam m, v = 16 B/param
* head             parameters counted by instantiating the real head on the ``meta`` device; 16 B/param
* backbone acts    with checkpointing: one bf16 [B, T, D] tensor per layer boundary + one layer's
                   working set (``WORK_FACTOR`` x [B, T, D] bf16) + delta-rule chunk states for that
                   layer's backward ((T / 64) states of ``state_scalars_per_layer`` fp32);
                   without checkpointing: every layer's working set
* head acts        the fp32 feature copy h[B,T,D] every head consumes, plus what autograd keeps
                   for the whole sequence (per-head formulas; snapshot adds its [B,T,S] one-hot
                   and [B,S,D] boundary features). Heads are non-streaming: the O(S*K) / O(K*d)
                   figures in ``cost()`` are accumulator sizes, not peak memory.
* FLOPs/token      2 * non-embedding params forward + 2x for input-gradient backward
                   (+ 2x recompute with checkpointing); LoRA adds 6 * lora_params; head 3x its FLOPs
* throughput       mfu_assumed (per backbone spec) * 989 TFLOP/s dense bf16
* cost             $2.97/h batch, $3.96/h interactive (SF Compute rate card)

Final-state fork (the only head whose backbone cost depends on K), two modes, both implemented:
* ``recompute``  K extra full rows (prefix + continuation) per question, all alive for backward:
                 tokens x (1 + K), memory K x activations. Infeasible for large K on long rows.
* ``cache``      one prefix pass, K state copies extended by the continuation. HF drops caches
                 under gradient checkpointing (the code refuses rather than silently mis-forking),
                 so the cache fork trains WITHOUT checkpointing: full prefix activations.

G16 (``head="g16"``, controlsafe/g16.py) and the G15 trust gate (``trust_gate=True``): two streams. D runs over all
T prefix tokens (cache fork: + K continuations), P over the trusted tokens only (``TRUSTED_FRAC`` of T, an
ASSUMPTION; identical trusted subsequences of the K forks run once). The gated arm refuses gradient checkpointing,
so both streams keep full activations. Bridges (L x 2 x d x bridge_rank) train with LoRA. G16's own head is tiny
(2 d r + d + 256 (r + 1) params). ``g16_pair`` prices the paired cells of jobs/m2_pilot (G16 vs the cache-fork head
on the same gated backbone) and the ratio to the ungated cache-fork cell.

    python -m train.estimate --json train/estimates/m2_estimates.json --table
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
MIN_USEFUL_TPS = 500.0  # prefix tokens/s below which a configuration is marked infeasible
CELL = {"rows": 30_000, "T": 2048, "K": 6}  # pilot data assumption: 10k records x 3 questions
TRUSTED_FRAC = 0.5  # ASSUMPTION: share of prefix tokens the provenance policy labels trusted (template + schema)
G16_DEFAULTS = {"r": 8, "max_options": 256, "bridge_rank": 8}

HEAD_DEFAULTS = {
    "integrator": {"channels": 64},
    "fork": {"hidden": 256},
    "snapshot": {"channels": 64, "max_snapshots": 8},
    "head_b": {"d_state": 64},
    "clef_control": {"width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096},
}


def head_params(name: str, d_model: int) -> int:
    if name == "g16":  # A, W_e: d x r; w: d; u, beta: max_options x (r + 1). No typed output (softmax decision).
        r, m = G16_DEFAULTS["r"], G16_DEFAULTS["max_options"]
        return 2 * d_model * r + d_model + m * (r + 1)
    with torch.device("meta"):
        h = make_head(name, d_model, **HEAD_DEFAULTS[name])
        t = CumulativeLink(d_model)
    return sum(p.numel() for p in h.parameters()) + sum(p.numel() for p in t.parameters())


def head_cost(name: str, d_model: int, T: int, K: int) -> dict:
    if name == "g16":
        r = G16_DEFAULTS["r"]
        return {"flops": 2 * K * (2 * d_model * r + d_model) + 2 * K * r, "accumulator_scalars": K * (2 * r + 1)}
    with torch.device("meta"):
        h = make_head(name, d_model, **HEAD_DEFAULTS[name])
    return h.cost(T, K)


def head_activation_bytes(name: str, B: int, T: int, K: int, d_model: int) -> float:
    return B * T * d_model * 4 + _head_specific_bytes(name, B, T, K, d_model)  # + fp32 copy of h


def _head_specific_bytes(name: str, B: int, T: int, K: int, d_model: int) -> float:
    if name == "integrator":
        return B * T * K * HEAD_DEFAULTS[name]["channels"] * 4 * 3  # u*v product, silu, gate
    if name == "snapshot":
        S = HEAD_DEFAULTS[name]["max_snapshots"]
        inc = B * T * K * HEAD_DEFAULTS[name]["channels"] * 4 * 3  # integrator increments, all chunks
        return inc + B * T * S * 4 + B * S * d_model * 4 * 3  # one-hot [B,T,S]; boundary feats + products
    if name == "head_b":
        return B * T * K * HEAD_DEFAULTS[name]["d_state"] * 4 * 6  # log_a, cumsum, decay, b, product, pre-acts
    if name == "fork":
        return B * K * d_model * 4 * 4
    if name == "g16":  # h_P and h_D fork features + E, a products
        return B * K * d_model * 4 * 8
    if name == "clef_control":
        c = HEAD_DEFAULTS[name]
        return B * (c["routing_layers"] * (c["heads"] * K * T * 4 * 2 + 2 * T * c["width"] * 2) + T * c["width"] * 2)
    raise KeyError(name)


def _acts(spec: BackboneSpec, B: int, T: int, grad_ckpt: bool) -> float:
    D, L = spec.d_model, spec.n_layers
    unit = B * T * D * 2
    chunk_states = B * (T / CHUNK) * spec.state_scalars_per_layer * spec.state_dtype_bytes
    if grad_ckpt:
        return L * unit + WORK_FACTOR * unit + chunk_states
    return L * (WORK_FACTOR * unit + chunk_states)


def train_point(spec: BackboneSpec, head: str, T: int, K: int, B: int = 1, rank: int = 256,
                grad_ckpt: bool = True, fork_mode: str = "recompute", cont_len: int = 8, trust_gate: bool = False,
                trusted_frac: float = TRUSTED_FRAC) -> dict:
    D, L = spec.d_model, spec.n_layers
    if head == "g16":
        trust_gate, fork_mode = True, "cache"  # G16 = two-stream cache fork over the trust-gated backbone
    if head == "fork" and fork_mode == "cache":
        grad_ckpt = False  # HF drops caches under checkpointing; see module docstring
    if trust_gate:
        grad_ckpt = False  # the trust gate refuses checkpointing (bridge injection lives in forward hooks)
    weights = spec.resident_params * 2
    lora = lora_param_count(spec.lora_shapes, rank)
    hp = head_params(head, D)
    acts = _acts(spec, B, T, grad_ckpt)
    hacts = head_activation_bytes(head, B, T, K, D)
    fork_bytes, fork_tokens = 0.0, 0
    p_bytes, p_tokens = 0.0, 0.0
    if trust_gate:  # P stream over the trusted tokens (once per distinct trusted subsequence) + bridges
        p_tokens = trusted_frac * T
        p_bytes = _acts(spec, B, max(1, int(p_tokens)), False) + L * B * T * D * 2  # + carried P features per layer
        lora += L * 2 * D * G16_DEFAULTS["bridge_rank"]
    if head in ("fork", "g16"):
        if fork_mode == "cache":
            state_bytes = L * spec.state_scalars_per_layer * spec.state_dtype_bytes
            # K state copies (plus their grads) + K continuation rows of activations
            fork_bytes = B * K * (2 * state_bytes + _acts(spec, 1, cont_len, False))
            fork_tokens = K * cont_len
        elif fork_mode == "recompute":
            fork_bytes = K * _acts(spec, B, T + cont_len, grad_ckpt)
            fork_tokens = K * (T + cont_len)
        else:
            raise ValueError(fork_mode)
    overhead = 2.0 * GB
    total = weights + 16 * (lora + hp) + acts + hacts + fork_bytes + p_bytes + overhead
    per_tok_head = head_cost(head, D, T, K)["flops"] / T
    f_tok = (6 if grad_ckpt else 4) * spec.params + 6 * lora + 3 * per_tok_head
    tokens_per_row = T + fork_tokens + p_tokens
    tps = spec.mfu_assumed * GPU["bf16_dense_tflops"] * 1e12 / f_tok * T / tokens_per_row  # prefix tokens/s
    fits = total / GB <= GPU["usable_gb"]
    reason = None
    if not fits:
        reason = f"needs {total / GB:.0f} GB at B={B} > {GPU['usable_gb']:.0f} GB usable"
    elif tps < MIN_USEFUL_TPS:
        reason = f"{tps:.0f} prefix tok/s < {MIN_USEFUL_TPS:.0f}"
    return {
        "backbone": spec.name, "head": head, "fork_mode": fork_mode if head in ("fork", "g16") else None,
        "trust_gate": trust_gate, "trusted_frac": trusted_frac if trust_gate else None,
        "backbone_tokens_per_row": tokens_per_row,
        "T": T, "K": K, "B": B, "lora_rank": rank, "grad_ckpt": grad_ckpt,
        "mem_gb": {"weights": weights / GB, "lora_opt": 16 * lora / GB, "head_opt": 16 * hp / GB,
                   "backbone_acts": acts / GB, "head_acts": hacts / GB, "fork": fork_bytes / GB,
                   "p_stream": p_bytes / GB, "overhead": overhead / GB, "total": total / GB},
        "fits_80gb": fits,
        "feasible": reason is None,
        "infeasible_reason": reason,
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
    return {"backbone": spec.name, "head": head, "fork_mode": p["fork_mode"], "trust_gate": p["trust_gate"],
            "rows": rows, "T": T, "K": K, "backbone_tokens_per_row": p["backbone_tokens_per_row"],
            "epochs": epochs, "train_tokens": tokens, "gpu_hours": hours, "usd_batch": hours * RATE["batch"],
            "max_micro_batch": max_batch(spec, head, T, K, **kw), "feasible": p["feasible"],
            "infeasible_reason": p["infeasible_reason"]}


def g16_pair(spec: BackboneSpec, rows: int, T: int, K: int, **kw) -> dict:
    """G16 cell vs its paired non-G16 cell (cache fork on the SAME trust-gated backbone) vs the ungated cache fork."""
    g = cell_cost(spec, "g16", rows, T, K, **kw)
    base = cell_cost(spec, "fork", rows, T, K, fork_mode="cache", trust_gate=True, **kw)
    plain = cell_cost(spec, "fork", rows, T, K, fork_mode="cache", **kw)
    return {"backbone": spec.name, "g16": g, "pair_base_gated_fork": base, "ungated_fork_cache": plain,
            "g16_over_ungated_fork": g["gpu_hours"] / plain["gpu_hours"],
            "g16_over_pair_base": g["gpu_hours"] / base["gpu_hours"], "trusted_frac_assumed": TRUSTED_FRAC}


def _variants(head: str):
    return [{"fork_mode": "recompute"}, {"fork_mode": "cache"}] if head == "fork" else [{}]


def report() -> dict:
    out: dict = {"gpu": GPU, "rates_usd_per_h": RATE, "cell_assumption": CELL, "assumptions": {
        "work_factor": WORK_FACTOR, "delta_chunk": CHUNK, "lm_head": "not run (heads read hidden states)",
        "min_useful_prefix_tps": MIN_USEFUL_TPS,
        "status": "ESTIMATE ONLY - replace with receipts from the m2 qualification rung"},
        "backbones": {}, "points": [], "max_batch": [], "cells": [], "g16_pair": []}
    for name, spec in BACKBONES.items():
        b = asdict(spec)
        b["lora_params_r256"] = lora_param_count(spec.lora_shapes, 256)
        b["lora_opt_gb_r256"] = 16 * b["lora_params_r256"] / GB
        b["eval_tokens_per_s"] = eval_tokens_per_s(spec)
        out["backbones"][name] = b
        for head in HEAD_DEFAULTS:
            for kw in _variants(head):
                for T, K in ((2048, 4), (2048, 6), (8192, 16), (8192, 255)):
                    out["points"].append(train_point(spec, head, T, K, **kw))
                out["max_batch"].append({"backbone": name, "head": head, **kw,
                                         **{f"T{T}_K{K}": max_batch(spec, head, T, K, **kw) for T, K in ((2048, 4), (8192, 16))}})
                out["cells"].append(cell_cost(spec, head, **CELL, **kw))
        out["g16_pair"].append(g16_pair(spec, **CELL))
        for T, K in ((2048, 4), (2048, 6), (8192, 16)):
            out["points"].append(train_point(spec, "g16", T, K))
    return out


def _table(rep: dict) -> str:
    lines = ["| backbone | head | fork | T | K | mem GB (B=1) | feasible | prefix tok/s | $/1M tok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in rep["points"]:
        lines.append(f"| {p['backbone']} | {p['head']} | {p['fork_mode'] or ''} | {p['T']} | {p['K']} | "
                     f"{p['mem_gb']['total']:.1f} | {'y' if p['feasible'] else 'NO: ' + p['infeasible_reason']} | "
                     f"{p['train_prefix_tokens_per_s']:.0f} | {p['usd_per_1m_train_tokens_batch']:.3f} |")
    lines += ["", f"Cells: {CELL['rows']} rows x T={CELL['T']}, K={CELL['K']}, 1 epoch", "",
              "| backbone | head | fork | GPU-h per cell | $ per cell (batch) | max micro-batch | feasible |",
              "|---|---|---|---|---|---|---|"]
    for c in rep["cells"]:
        lines.append(f"| {c['backbone']} | {c['head']} | {c['fork_mode'] or ''} | {c['gpu_hours']:.2f} | "
                     f"{c['usd_batch']:.2f} | {c['max_micro_batch']} | {'y' if c['feasible'] else 'NO'} |")
    lines += ["", f"G16 pair (trusted_frac assumed {TRUSTED_FRAC}): GPU-h per cell", "",
              "| backbone | G16 | gated cache fork (pair base) | ungated cache fork | G16 / ungated | G16 / pair base |"
              " $ G16 cell |", "|---|---|---|---|---|---|---|"]
    for q in rep.get("g16_pair", []):
        lines.append(f"| {q['backbone']} | {q['g16']['gpu_hours']:.2f} | {q['pair_base_gated_fork']['gpu_hours']:.2f} | "
                     f"{q['ungated_fork_cache']['gpu_hours']:.2f} | {q['g16_over_ungated_fork']:.2f}x | "
                     f"{q['g16_over_pair_base']:.2f}x | {q['g16']['usd_batch']:.2f} |")
    lines += ["", "| backbone | LoRA params (r=256) | LoRA optimizer GB |", "|---|---|---|"]
    for n, b in rep["backbones"].items():
        lines.append(f"| {n} | {b['lora_params_r256'] / 1e6:.0f}M | {b['lora_opt_gb_r256']:.1f} |")
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

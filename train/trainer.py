"""m2 trainer: frozen backbone + LoRA + decision head, with checkpoint/resume and eval cards.

Precision: the backbone forward runs under bf16 autocast when ``bf16`` is set (CUDA or CPU);
head, typed outputs, losses and LoRA master weights stay fp32. Gradient checkpointing uses
the backbone's own ``gradient_checkpointing_enable`` (non-reentrant).

Determinism and resume: the data order is a pure function of (seed, epoch); the checkpoint
stores step, epoch cursor, trainable weights, optimizer and scheduler state and RNG states,
so ``train(N)`` equals ``train(k); save; resume; train(N - k)`` exactly on CPU. Checkpoints
are written atomically (tmp + replace) and never deleted by the trainer.

Usage (paid runs only after the spend ladder in COORDINATION.md):

    cfg = TrainConfig(head="integrator", loss=LossConfig(arm="clef"), lora_rank=256, ...)
    Trainer(model, cfg, train_items, eval_items).train()
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from heads.model import DecisionModel

from .data import Item
from .evalcard import run_card
from .guard import SplitGuard
from .lineage import diff as lineage_diff
from .lineage import lineage
from .lora import apply_lora, lora_parameters
from .losses import LossConfig, add_order_consistency, aligned_order_js, decision_loss


def teacher_tensor(batch, lp: torch.Tensor) -> torch.Tensor:
    """[B, K] teacher probabilities (``Item.teacher``, in ``row.option_ids`` order; ``train.teacher.attach``), zero on
    padded options; a row without a teacher is all NaN (``decision_loss`` falls back to the arm for it)."""
    t = torch.full(lp.shape, float("nan"), dtype=lp.dtype, device=lp.device)
    for b, it in enumerate(batch):
        tp = getattr(it, "teacher", None)
        if tp is None:
            continue
        if len(tp) != it.row.n_options:
            raise ValueError(f"{it.id}: teacher has {len(tp)} options, row has {it.row.n_options}")
        t[b] = 0.0
        t[b, : len(tp)] = torch.tensor(tp, dtype=lp.dtype, device=lp.device)
    return t


@dataclass
class TrainConfig:
    head: str = "integrator"
    head_kwargs: dict = field(default_factory=dict)
    layout: str = "ordered"
    fork_mode: str = "cache"  # final-state fork: "cache" (state copies) or "recompute" (K full rows)
    lora_rank: int = 256
    lora_alpha: float | None = None
    lora_dropout: float = 0.0
    lora_targets: str | None = None  # None -> backbone spec default
    loss: LossConfig = field(default_factory=LossConfig)
    lr_head: float = 1e-3
    lr_lora: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 8
    grad_accum: int = 1
    max_steps: int = 100
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    bf16: bool = False
    grad_checkpointing: bool = False
    seed: int = 0
    ckpt_dir: str | None = None
    ckpt_every: int = 0
    eval_every: int = 0
    require_guard: bool = True  # refuse to train without a SplitGuard (GOAL.md integrity rule)
    # optional controlsafe arm (G15 trust gate): {"trust_gate": true, "bridge_rank": 8, "readout": "decision",
    # "trust_policy": "state"}. Empty = off; an empty dict is left out of the config hash (old hashes unchanged).
    controlsafe: dict = field(default_factory=dict)
    # ---- execution strategy (jobs/cell_step_profile; defaults = the original path). The same math, so they are left
    # out of the config hash (a checkpoint resumes under either setting); equivalence: tests/test_cell_perf.py.
    #: per-block activation checkpointing that keeps the cache fork (heads/act_ckpt.py: RWKV-7, Qwen3.5 / Q9-AF)
    act_ckpt: bool = False
    #: rows per backbone call within one optimiser step (1 = one call per micro-batch row, as before). > 1: the step's
    #: batch_size x grad_accum rows are sorted by length (longest first) and run in groups of up to row_batch rows, each
    #: group in ONE prefix call + ONE continuation call (heads.backbone.fork_features_rows); every group's loss is
    #: weighted by its share of the step's rows, so the step's gradient is the same sum over rows.
    row_batch: int = 1
    #: with row_batch > 1: max padded tokens per call (rows x longest row of the group); 0 = no limit. A row longer
    #: than the budget runs alone. Bounds the per-call activation memory (jobs/cell_step_profile JOB_SPEC, A1).
    row_tokens: int = 0
    #: with row_batch > 1: how the step's length-sorted rows are cut into groups. "greedy" = ``row_groups`` (fill from
    #: the longest row). "min_cost" = ``row_groups_min_cost``: the contiguous cut of the same sorted order, under the
    #: same row_batch / row_tokens caps, that minimises (groups x GROUP_COST_TOKENS + padded tokens). Same rows, same
    #: per-group loss weights, so the same step gradient up to float summation order.
    row_plan: str = "greedy"
    #: with row_batch > 1: "per_group" = one prefix + one continuation call + one backward per group (as before);
    #: "single" = the groups' prefix calls (same groups, same padding), their states joined, ONE continuation call for
    #: all the step's rows and ONE backward of the step-mean loss. Same loss and gradient up to float summation order;
    #: every group's checkpointed block inputs stay alive until the backward (more memory: the ladder decides).
    cont_plan: str = "per_group"
    #: with act_ckpt (RWKV-7): run each checkpointed block body through one shared torch.compile (dynamic shapes; the
    #: WKV chunk loop stays eager via torch.compiler.disable): heads.act_ckpt ``compile_backend``. Kernel fusion changes
    #: float rounding (not the math): gated on GPU by the probe-loss / gradient-cosine rule (jobs/perf_step_gate).
    compile_blocks: bool = False
    #: with row_batch > 1: max forked continuations per call (sum of the group's option counts K); 0 = no limit. Each
    #: continuation carries its own copy of the recurrent state (RWKV-7 13.3B: 61 x 64 x 64 x 64 fp32 = 64 MB), so a
    #: group of long catalogs (K = 77..255) can hold thousands of copies at once; a row whose K alone is over the cap
    #: runs alone. Grouping only: the same rows, losses and gradients (tests/test_cell_perf.py).
    row_options: int = 0
    axis_loss_weight: float = 0.0  # opt-in MSE on HeadOutput.semantic_axes, targets live only on Item
    # ---- anti-collapse (research-notes/D10_DIVERGENCE.md). Every default = the old behaviour; at their defaults these
    # fields are left out of to_json / the hash, so old configs, records and checkpoints are byte-identical.
    #: "constant" = warmup then constant LR (as before); "cosine" = warmup then cosine decay to lr_min_frac x LR at the
    #: schedule horizon (max_steps, or the clock-projected final step set by ``set_horizon``; stored in the checkpoint)
    lr_schedule: str = "constant"
    lr_min_frac: float = 0.1
    #: every N optimiser steps run the dev card on the eval items; 0 = off. ``dev_card_rows`` > 0: a hash-stratified
    #: sample of that many rows (``dev_card_sample``: sources from the ``m2:<source>:...`` ids, sha256(seed, id) order
    #: within each source, round-robin across sources); 0 = all eval items. ``dev_card_seed`` fixes the sample.
    dev_card_every: int = 0
    dev_card_rows: int = 0
    dev_card_seed: int = 0
    #: keep the best dev card's checkpoint (lowest NLL, tie -> higher accuracy); latest.json gets a ``best`` pointer and
    #: scoring (``Trainer.scored``) loads it instead of the last checkpoint
    best_ckpt: bool = False
    #: collapse guard: 50-step mean loss > collapse_loss_frac x the uniform predictor's loss on the same rows, or a dev
    #: card accuracy < collapse_acc_frac x the best card's -> reload the best ckpt, halve both LRs (max events, then stop).
    #: The loss trigger arms only after one full window below the threshold (a fresh head starts at the uniform loss).
    collapse_guard: bool = False
    collapse_window: int = 50
    collapse_loss_frac: float = 0.8
    collapse_acc_frac: float = 0.6
    collapse_max_events: int = 3

    def to_json(self) -> dict:
        d = asdict(self)
        d["loss"] = self.loss.identity()  # optional KD/order-consistency fields only when enabled (old hashes unchanged)
        if d["axis_loss_weight"] == 0.0:
            d.pop("axis_loss_weight")  # preserve legacy config identities
        for k in self.EXECUTION + self.ANTI_COLLAPSE:  # at their defaults these are left out: old records identical
            if d[k] == getattr(TrainConfig, k):
                d.pop(k)
        return d

    #: anti-collapse fields (left out at their defaults). The schedule and the guard change the trajectory, so when set
    #: they are in the hash; dev_card_every / dev_card_rows / dev_card_seed / best_ckpt only observe, so they are
    #: OPERATIONAL, EXCEPT with collapse_guard on: then the dev cards set the accuracy-trigger timing and the rollback
    #: target (they change the trajectory), so the dev_card_* keys are in the hash (``GUARD_HASHED``).
    ANTI_COLLAPSE = ("lr_schedule", "lr_min_frac", "dev_card_every", "dev_card_rows", "dev_card_seed", "best_ckpt",
                     "collapse_guard", "collapse_window", "collapse_loss_frac", "collapse_acc_frac",
                     "collapse_max_events")
    #: operational fields that may differ between a run and its resumption
    OPERATIONAL = ("ckpt_dir", "ckpt_every", "eval_every", "max_steps", "require_guard", "dev_card_every",
                   "dev_card_rows", "dev_card_seed", "best_ckpt")
    #: operational dev-card keys that ARE hashed when collapse_guard is on
    GUARD_HASHED = ("dev_card_every", "dev_card_rows", "dev_card_seed")
    #: execution-strategy fields: same loss and gradients, not part of the hash (old hashes unchanged)
    EXECUTION = ("act_ckpt", "row_batch", "row_tokens", "row_plan", "cont_plan", "compile_blocks", "row_options")

    def hash(self) -> str:
        keep = self.GUARD_HASHED if self.collapse_guard else ()
        d = {k: v for k, v in self.to_json().items() if k in keep or k not in self.OPERATIONAL + self.EXECUTION}
        if not d.get("controlsafe"):
            d.pop("controlsafe", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def with_execution(cfg: TrainConfig, cell: dict) -> TrainConfig:
    """``cfg`` with the execution-strategy keys of a cell config (``act_ckpt``, ``row_batch``, ``row_tokens``; absent =
    the original path). For ``train.run_cell.train_config`` callers: ``with_execution(train_config(cell, ...), cell)``."""
    import dataclasses

    return dataclasses.replace(cfg, act_ckpt=bool(cell.get("act_ckpt", False)), row_batch=int(cell.get("row_batch", 1)),
                               row_tokens=int(cell.get("row_tokens", 0)), row_plan=str(cell.get("row_plan", "greedy")),
                               cont_plan=str(cell.get("cont_plan", "per_group")),
                               compile_blocks=bool(cell.get("compile_blocks", False)),
                               row_options=int(cell.get("row_options", 0)))


#: fixed cost of one group (prefix + continuation call, forward + recompute + backward) in padded-token equivalents:
#: job-tvuat ckpt_rows_lean fit over its 20 timed steps, step_s = 0.36 + 2.02 x groups + 0.462e-3 x padded tokens
#: (r2 0.995), so 2.02 / 0.462e-3 ~ 4,400 tokens.
GROUP_COST_TOKENS = 4400


def row_groups_min_cost(lengths: list[int], row_batch: int, row_tokens: int = 0, options: list[int] | None = None,
                        row_options: int = 0, group_cost: int = GROUP_COST_TOKENS) -> list[list[int]]:
    """Like ``row_groups`` (same longest-first stable order, every group <= ``row_batch`` rows and, if ``row_tokens``
    > 0, <= ``row_tokens`` padded tokens and, if ``row_options`` > 0, <= ``row_options`` summed option counts K,
    unless it is a single row), but the cut points minimise ``group_cost x groups + padded tokens`` (exact dynamic
    programme over contiguous cuts of the sorted order). The caps are hard constraints of the programme, so
    ``row_options`` composes with the cost model exactly as with the greedy fill."""
    if row_options and (options is None or len(options) != len(lengths)):
        raise ValueError("row_options needs one option count per row")
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    n = len(order)
    best, cut = [0] + [None] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - row_batch), j):
            pad = (j - i) * lengths[order[i]]
            if j - i > 1 and row_tokens and pad > row_tokens:
                continue
            if j - i > 1 and row_options and sum(options[order[x]] for x in range(i, j)) > row_options:
                continue
            c = best[i] + group_cost + pad
            if best[j] is None or c < best[j]:
                best[j], cut[j] = c, i
    groups, j = [], n
    while j:
        groups.append(order[cut[j]:j])
        j = cut[j]
    return groups[::-1]


def single_cont_fits(options: list[int], row_options: int = 0) -> bool:
    """``cont_plan="single"`` forks ONE continuation call over every row of the step: sum(K) recurrent-state copies at
    once. With ``row_options`` > 0 that sum must stay within the cap; otherwise the step falls back to the per-group
    path (one continuation call per ``row_options``-capped group). No cap = always fits (D1 behaviour unchanged)."""
    return not row_options or sum(options) <= row_options


def row_groups(lengths: list[int], row_batch: int, row_tokens: int = 0, options: list[int] | None = None,
               row_options: int = 0) -> list[list[int]]:
    """Indices of the step's rows grouped for ``row_batch`` > 1: sorted by length, longest first (stable), filled
    greedily while a group has <= ``row_batch`` rows and, if ``row_tokens`` > 0, (rows x its longest row) <=
    ``row_tokens`` padded tokens and, if ``row_options`` > 0, the sum of its rows' ``options`` (K, one forked state
    copy each) <= ``row_options``. Every row lands in exactly one group."""
    if row_options and (options is None or len(options) != len(lengths)):
        raise ValueError("row_options needs one option count per row")
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups: list[list[int]] = []
    for i in order:
        g = groups[-1] if groups else None
        if g is not None and len(g) < row_batch and (
                not row_tokens or (len(g) + 1) * lengths[g[0]] <= row_tokens) and (
                not row_options or sum(options[j] for j in g) + options[i] <= row_options):
            g.append(i)
        else:
            groups.append([i])
    return groups


def row_groups_with_views(base_lengths: list[int], view_lengths: list[int], row_batch: int,
                          row_tokens: int = 0) -> list[list[int]]:
    """Group base/view rows under the total padded-token budget for their separate forward calls.

    ``view_lengths[i] == 0`` means row i has no paired view. A group consumes
    ``n_base * max(base_lengths) + n_views * max(view_lengths for paired rows)`` padded tokens while both
    autograd graphs are live. This avoids undercounting when the longest base and view belong to different rows.
    """
    if len(base_lengths) != len(view_lengths):
        raise ValueError("base_lengths and view_lengths must have the same length")
    if row_batch < 1 or row_tokens < 0:
        raise ValueError("row_batch must be positive and row_tokens nonnegative")
    if any(type(x) is not int or x < 1 for x in base_lengths) or any(type(x) is not int or x < 0 for x in view_lengths):
        raise ValueError("base lengths must be positive integers and view lengths nonnegative integers")

    def padded(group: list[int]) -> int:
        base_tokens = len(group) * max(base_lengths[i] for i in group)
        paired = [i for i in group if view_lengths[i] > 0]
        view_tokens = len(paired) * max((view_lengths[i] for i in paired), default=0)
        return base_tokens + view_tokens

    order = sorted(range(len(base_lengths)), key=lambda i: -max(base_lengths[i], view_lengths[i]))
    groups: list[list[int]] = []
    for i in order:
        group = groups[-1] if groups else None
        if group is not None and len(group) < row_batch and (not row_tokens or padded(group + [i]) <= row_tokens):
            group.append(i)
        else:
            groups.append([i])
    return groups


#: torch.compile backend for ``TrainConfig.compile_blocks`` (tests set "aot_eager": no C++/Triton toolchain on CPU)
COMPILE_BACKEND = os.environ.get("SSDM_COMPILE_BACKEND", "inductor")


def lora_init_seed(seed: int) -> int:
    return seed * 1_000_003 + 17


def prepare_model(model: DecisionModel, cfg: TrainConfig) -> list[str]:
    """Freeze the backbone, add LoRA (rank 0 = head-only), enable checkpointing."""
    targets = cfg.lora_targets or (model.backbone.spec.lora_groups if model.backbone.spec else None)
    replaced: list[str] = []
    if cfg.lora_rank > 0:
        if targets is None:
            raise ValueError("no LoRA targets: set cfg.lora_targets or give the backbone a spec")
        # LoRA init draws only from this generator: a function of cfg.seed, never the ambient
        # (CPU or CUDA) RNG, and it does not advance the ambient RNG either.
        gen = torch.Generator().manual_seed(lora_init_seed(cfg.seed))
        found = apply_lora(model.backbone.model, targets, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
                           generator=gen)
        replaced = sorted({n for names in found.values() for n in names})
    else:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if cfg.controlsafe.get("trust_gate") and not getattr(model.backbone, "wants_trust", False):
        raise ValueError("cfg.controlsafe.trust_gate is set but the backbone is not a controlsafe "
                         "TrustGatedBackbone: wrap it with controlsafe.trust_gate.wrap(...) before DecisionModel")
    for p in getattr(model.backbone, "trainable_bridge_parameters", lambda: [])():
        p.requires_grad_(True)  # P -> D bridges train with the LoRA group (named lora_A / lora_B)
    model.backbone.fork_mode = cfg.fork_mode
    if cfg.grad_checkpointing:
        model.backbone.enable_gradient_checkpointing()
    from heads import act_ckpt

    if cfg.compile_blocks and not cfg.act_ckpt:
        raise ValueError("compile_blocks needs act_ckpt (the compiled region is the checkpointed block body)")
    backend = COMPILE_BACKEND if cfg.compile_blocks else None
    if act_ckpt.is_enabled(model.backbone.model) and act_ckpt.compile_backend_of(model.backbone.model) != backend:
        act_ckpt.disable_block_checkpointing(model.backbone.model)  # reused backbone, other compile setting
    if cfg.act_ckpt:
        # RWKV-7 or Qwen3.5 / Q9-AF; raises otherwise
        act_ckpt.enable_block_checkpointing(model.backbone.model, compile_backend=backend)
    elif act_ckpt.is_enabled(model.backbone.model):  # a backbone reused across Trainers (run_cell ladder)
        act_ckpt.disable_block_checkpointing(model.backbone.model)
    if cfg.row_batch < 1:
        raise ValueError(f"row_batch must be >= 1, got {cfg.row_batch}")
    if cfg.row_plan not in ("greedy", "min_cost"):
        raise ValueError(f"row_plan must be 'greedy' or 'min_cost', got {cfg.row_plan!r}")
    if cfg.cont_plan not in ("per_group", "single"):
        raise ValueError(f"cont_plan must be 'per_group' or 'single', got {cfg.cont_plan!r}")
    if cfg.cont_plan == "single" and cfg.loss.order_consistency_weight:
        raise ValueError("cont_plan='single' is not implemented for the order-consistency arm (views)")
    if cfg.row_options < 0:
        raise ValueError(f"row_options must be >= 0, got {cfg.row_options}")
    if cfg.row_options and cfg.loss.order_consistency_weight:
        raise ValueError("row_options is not implemented for order-consistency views (row_groups_with_views)")
    model.backbone.batch_rows = cfg.row_batch > 1
    return replaced


def item_source(item_id: str) -> str:
    """The cohort source of an item id ``m2:<source>:<record>/<qid>`` ("" for ids without one, e.g. toy rows)."""
    parts = str(item_id).split(":")
    return parts[1] if len(parts) >= 3 else ""


def dev_card_sample(items: list[Item], rows: int, seed: int = 0) -> list[Item]:
    """The dev-card rows (B1, PR #232 review): ``rows`` <= 0 or >= len(items) = every item (input order). Else a
    deterministic hash-stratified sample: items grouped by ``item_source``, each source ordered by
    sha256(f"{seed}:{id}") (ties by id), then taken round-robin over the sources (sorted by name) until ``rows`` are
    chosen, so every source is represented (cohort dev ids are sorted by id, i.e. by source: a prefix covered 4 of
    14). Independent of the input order."""
    if rows <= 0 or rows >= len(items):
        return list(items)
    by: dict[str, list[tuple[str, str, Item]]] = {}
    for it in items:
        key = hashlib.sha256(f"{seed}:{it.id}".encode()).hexdigest()
        by.setdefault(item_source(it.id), []).append((key, it.id, it))
    queues = [sorted(by[s], key=lambda t: (t[0], t[1])) for s in sorted(by)]
    out: list[Item] = []
    r = 0
    while len(out) < rows:
        for q in queues:
            if r < len(q) and len(out) < rows:
                out.append(q[r][2])
        r += 1
    return out


def dev_card_record(items: list[Item], rows: int, seed: int = 0) -> dict[str, Any]:
    """cell.json record of the dev-card sample: the selected ids (selection order), their sha256, per-source counts."""
    ids = [it.id for it in dev_card_sample(items, rows, seed)]
    per: dict[str, int] = {}
    for i in ids:
        per[item_source(i)] = per.get(item_source(i), 0) + 1
    return {"rows": rows, "seed": seed, "n": len(ids), "n_pool": len(items),
            "design": "hash-stratified by source: sha256(seed:id) within source, round-robin across sources",
            "per_source": dict(sorted(per.items())), "ids": ids,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def check_anti_collapse(cfg: TrainConfig) -> None:
    if cfg.lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {cfg.lr_schedule!r}")
    if not 0.0 <= cfg.lr_min_frac <= 1.0:
        raise ValueError(f"lr_min_frac must be in [0, 1], got {cfg.lr_min_frac}")
    if cfg.dev_card_every < 0 or cfg.dev_card_rows < 0:
        raise ValueError("dev_card_every and dev_card_rows must be >= 0")
    if type(cfg.dev_card_seed) is not int:
        raise ValueError(f"dev_card_seed must be an int, got {cfg.dev_card_seed!r}")
    if cfg.best_ckpt and not cfg.dev_card_every:
        raise ValueError("best_ckpt needs dev_card_every > 0 (the best is chosen by dev card)")
    if cfg.collapse_guard and not cfg.best_ckpt:
        raise ValueError("collapse_guard needs best_ckpt (a collapse reloads the best checkpoint)")
    if cfg.collapse_guard and (cfg.collapse_window < 1 or cfg.collapse_max_events < 0):
        raise ValueError("collapse_window must be >= 1 and collapse_max_events >= 0")


class Trainer:
    def __init__(self, model: DecisionModel, cfg: TrainConfig, train_items: list[Item],
                 eval_items: list[Item] | None = None, prepared: bool = False,
                 guard: SplitGuard | None = None) -> None:
        if cfg.require_guard and guard is None:
            raise ValueError("TrainConfig.require_guard: pass a SplitGuard built from the Index dev/test "
                             "manifests (or SplitGuard.synthetic_only for synthetic toys)")
        init_seed = getattr(model, "init_seed", None)
        if init_seed is not None and init_seed != cfg.seed:
            raise ValueError(f"model head was initialised with seed {init_seed} but cfg.seed = {cfg.seed}; "
                             "build DecisionModel(..., seed=cfg.seed) so one seed fixes every random init")
        # Seed the ambient RNGs FIRST, before any initialisation below. Head init (DecisionModel)
        # and LoRA init (explicit generator) are already independent of the ambient state; this
        # makes everything after construction (dropout, data order helpers) a function of cfg.seed.
        if cfg.loss.kd and not any(getattr(i, "teacher", None) is not None for i in train_items):
            raise ValueError("loss.kd_alpha > 0 but no training item carries a teacher distribution: call "
                             "train.teacher.attach(items, load_teacher(...)) first (a KD run without a teacher is the "
                             "clef arm under another name)")
        if cfg.loss.order_consistency_weight > 0:
            if cfg.loss.kd:
                raise ValueError("order-consistency arm currently requires KD off; view-teacher semantics are not registered")
            if cfg.axis_loss_weight != 0.0:
                raise ValueError("order-consistency arm currently requires axis_loss_weight=0; view-axis semantics are not registered")
            paired = [i for i in train_items if getattr(i, "view", None) is not None]
            if not paired:
                raise ValueError("order_consistency_weight > 0 but no training items carry eligible views")
            if any(getattr(i, "view", None) is not None or getattr(i, "view_perm", None) is not None
                   for i in (eval_items or [])):
                raise ValueError("order-consistency views are training-only; eval_items must not contain views")
            for item in train_items:
                view, perm = getattr(item, "view", None), getattr(item, "view_perm", None)
                if (view is None) != (perm is None):
                    raise ValueError(f"{item.id}: view and view_perm must either both be present or both be absent")
                if view is None:
                    continue
                k = len(item.row.option_ids)
                if (item.row.qtype != "choice" or view.row.qtype != "choice" or k < 2
                        or view.view is not None or view.view_perm is not None):
                    raise ValueError(f"{item.id}: malformed or nested order-consistency view")
                if tuple(item.row.option_ids) != tuple(view.row.option_ids) or len(perm) != k:
                    raise ValueError(f"{item.id}: view must preserve the canonical option-ID domain")
                if sorted(perm) != list(range(k)) or perm[view.label] != item.label:
                    raise ValueError(f"{item.id}: view label/permutation does not preserve the gold option content")
        elif any(getattr(i, "view", None) is not None or getattr(i, "view_perm", None) is not None
                 for i in (eval_items or [])):
            raise ValueError("eval_items must not contain order-consistency views")
        self._seed_all(cfg.seed)
        self.guard = guard
        self.model, self.cfg = model, cfg
        self.train_items, self.eval_items = train_items, eval_items or []
        self.lora_modules = [] if prepared else prepare_model(model, cfg)
        groups = [{"params": model.head_parameters(), "lr": cfg.lr_head}]
        lora = lora_parameters(model.backbone)
        if lora:
            groups.append({"params": lora, "lr": cfg.lr_lora})
        self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        check_anti_collapse(cfg)
        #: anti-collapse state (saved in the checkpoint; untouched at the defaults): the cosine horizon, the guard's LR
        #: multiplier, the dev-card log, the best card and the collapse events
        self.horizon = cfg.max_steps
        self.horizon_fixed = False
        self.lr_scale = 1.0
        self.dev_cards: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.collapse_events: list[dict[str, Any]] = []
        self.stopped: str | None = None
        self.guard_armed = False
        self._guard_window: list[tuple[float, float]] = []  # (step loss, uniform-predictor loss) since the last reload
        self._step_uniform = 0.0
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: self.lr_factor(s))
        self.step = 0
        self.epoch = 0
        self.cursor = 0
        self.history: list[dict[str, Any]] = []
        self.cards: list[dict[str, Any]] = []
        self.lineage = lineage(model.backbone, train_items)

    # ------------------------------------------------------------------ LR schedule
    def lr_factor(self, s: int) -> float:
        """LambdaLR factor after ``s`` scheduler steps. "constant": ``min(1, (s+1)/warm)`` (the original lambda,
        bit-identical); "cosine": the same warmup, then ``min_frac + (1 - min_frac)(1 + cos(pi p))/2`` with ``p`` the
        fraction of the post-warmup span to ``self.horizon`` (clamped: min_frac after the horizon). Times the collapse
        guard's ``lr_scale`` (1.0 unless the guard fired)."""
        cfg = self.cfg
        warm = max(cfg.warmup_steps, 0)
        f = min(1.0, (s + 1) / warm) if warm else 1.0
        if cfg.lr_schedule == "cosine" and s >= warm:
            span = max(1, self.horizon - warm)
            p = min(1.0, (s - warm) / span)
            f = cfg.lr_min_frac + (1.0 - cfg.lr_min_frac) * 0.5 * (1.0 + math.cos(math.pi * p))
        return f * self.lr_scale if self.lr_scale != 1.0 else f

    def set_horizon(self, final_step: int) -> None:
        """Cosine horizon = ``min(max_steps, final_step)`` (run_cell: the clock-projected final step from the in-training
        rate check). Fixed once per run and stored in the checkpoint, so a resume keeps the same schedule."""
        if self.horizon_fixed:
            return
        self.horizon = max(1, min(self.cfg.max_steps, int(final_step)))
        self.horizon_fixed = True
        self._apply_lr()

    def _apply_lr(self) -> None:
        """Set every group's LR for the current scheduler step (after a horizon / lr_scale change)."""
        f = self.lr_factor(self.sched.last_epoch)
        for g, base in zip(self.opt.param_groups, self.sched.base_lrs):
            g["lr"] = base * f
        self.sched._last_lr = [g["lr"] for g in self.opt.param_groups]

    # ------------------------------------------------------------------ dev cards, best checkpoint, collapse guard
    def dev_items(self, items: list[Item] | None = None) -> list[Item]:
        its = self.eval_items if items is None else items
        return dev_card_sample(its, self.cfg.dev_card_rows, self.cfg.dev_card_seed)

    def dev_card_due(self) -> bool:
        return bool(self.cfg.dev_card_every) and self.step % self.cfg.dev_card_every == 0

    def run_dev_card(self, items: list[Item] | None = None, out_dir=None) -> dict[str, Any] | None:
        """Run the dev card (``evalcard.run_card``) on the fixed dev set and pass it to ``consider_card``."""
        its = self.dev_items(items)
        if not its:
            return None
        card = run_card(self.model, its, self.step, out_dir)
        card.pop("units", None)
        self.consider_card(card)
        return card

    @staticmethod
    def _card_key(card: dict) -> tuple[float, float]:
        raw = card.get("raw") or {}
        nll, acc = raw.get("nll", float("nan")), raw.get("accuracy", float("nan"))
        return (nll if math.isfinite(nll) else float("inf"), -(acc if math.isfinite(acc) else -1.0))

    def consider_card(self, card: dict) -> dict:
        """Log a dev card; if it is the best so far (lowest NLL, tie -> higher accuracy) and ``best_ckpt``, save its
        checkpoint as the best. With ``collapse_guard``, an accuracy below collapse_acc_frac x the best card's fires
        the guard. Returns the log entry."""
        raw = card.get("raw") or {}
        entry = {"step": card["step"], "n_items": card.get("n_items"), "nll": raw.get("nll"),
                 "accuracy": raw.get("accuracy"), "best": False}
        if (self.cfg.collapse_guard and not self.stopped and self.best is not None and entry["accuracy"] is not None
                and entry["accuracy"] < self.cfg.collapse_acc_frac * self.best["accuracy"]):
            self.dev_cards.append(entry)
            self.collapse("dev_accuracy", accuracy=entry["accuracy"], best_accuracy=self.best["accuracy"])
            return entry
        if self.best is None or self._card_key(card) < (self.best["nll"], -self.best["accuracy"]):
            entry["best"] = True
            nll, acc = self._card_key(card)
            self.best = {"step": card["step"], "nll": nll, "accuracy": -acc}
            if self.cfg.best_ckpt and self.cfg.ckpt_dir:
                self.best["path"] = f"ckpt-step{self.step:07d}.pt"
                self.save()  # the step's checkpoint; latest.json carries the best pointer
        self.dev_cards.append(entry)
        return entry

    def _uniform_loss(self, lp: torch.Tensor, y: torch.Tensor, qtypes: list, teacher) -> torch.Tensor | None:
        """Collapse guard (else None): the configured decision loss of the UNIFORM predictor on the same rows (padded
        options = -inf stay -inf). For the CE arm this is -mean log(1/K). ``_supervised_loss`` adds the other terms of
        the logged objective and accumulates it into ``_step_uniform`` with the step's row weights."""
        if not self.cfg.collapse_guard:
            return None
        with torch.no_grad():
            u = torch.where(torch.isfinite(lp), torch.zeros_like(lp), torch.full_like(lp, float("-inf")))
            u = torch.log_softmax(u, dim=-1)
            return decision_loss(u, y, qtypes, self.cfg.loss, teacher).detach()

    def _guard_step(self, loss: float) -> None:
        if not self.cfg.collapse_guard or self.stopped:
            return
        self._guard_window.append((loss, self._step_uniform))
        w = self.cfg.collapse_window
        del self._guard_window[:-w]
        if len(self._guard_window) < w:
            return
        ml = sum(x for x, _ in self._guard_window) / w
        mu = sum(u for _, u in self._guard_window) / w
        above = ml > self.cfg.collapse_loss_frac * mu
        if not self.guard_armed:
            # a fresh head starts AT the uniform loss: the loss trigger arms only once a full window is below the
            # threshold (the model has learned); a later return above it is the collapse (D10: 0.40 -> 0.90-0.99)
            self.guard_armed = not above
            return
        if above:
            self.collapse("train_loss", mean_loss=ml, mean_uniform=mu)

    def collapse(self, trigger: str, **detail) -> dict:
        """A collapse event: reload the best checkpoint's trainable weights and optimiser moments (the step counter,
        data cursor, history and RNG keep going forward), halve both LRs, clear the loss window. After
        ``collapse_max_events`` reloads the next event stops training (``self.stopped``); scoring uses the best."""
        ev = {"step": self.step, "trigger": trigger, **detail, "best_step": (self.best or {}).get("step")}
        if len(self.collapse_events) >= self.cfg.collapse_max_events:
            ev["action"] = "stop"
            self.stopped = "collapse_max_events"
        elif not (self.best and self.best.get("path") and self.cfg.ckpt_dir):
            ev["action"] = "stop"  # nothing to roll back to
            self.stopped = "collapse_no_best"
        else:
            st = torch.load(Path(self.cfg.ckpt_dir) / self.best["path"], map_location="cpu", weights_only=False)
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, t in st["trainable"].items():
                    params[n].copy_(t)
            if st.get("optimizer") is not None:
                self.opt.load_state_dict(st["optimizer"])
            else:  # train.chain: a best carried trainable-only across a link (5 GiB artifact cap) has no moments
                ev["optimizer"] = "kept: the best was carried across a chain link without its AdamW moments"
            self.lr_scale *= 0.5
            self._apply_lr()
            ev.update(action="reload_best", lr_scale=self.lr_scale, lrs=[g["lr"] for g in self.opt.param_groups])
        self._guard_window.clear()
        self.collapse_events.append(ev)
        return ev

    # ------------------------------------------------------------------ data order
    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)  # also seeds every CUDA device

    def _order(self, epoch: int) -> list[int]:
        g = torch.Generator().manual_seed(self.cfg.seed * 100_003 + epoch)
        return torch.randperm(len(self.train_items), generator=g).tolist()

    def _next_batch(self) -> list[Item]:
        bs = self.cfg.batch_size
        order = self._order(self.epoch)
        if self.cursor + bs > len(order):
            self.epoch += 1
            self.cursor = 0
            order = self._order(self.epoch)
        batch = [self.train_items[i] for i in order[self.cursor : self.cursor + bs]]
        self.cursor += bs
        return batch

    # ------------------------------------------------------------------ steps
    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def _autocast(self):
        if not self.cfg.bf16:
            return nullcontext()
        dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)

    def _row_loss(self, batch: list[Item], weight: float) -> float:
        with self._autocast():
            fw = self.model([i.row for i in batch])
        loss = self._supervised_loss(fw, batch, weight) * weight
        loss.backward()
        return float(loss.detach())

    def _supervised_loss(self, fw, batch: list[Item], weight: float = 1.0) -> torch.Tensor:
        lp = fw.log_probs.float()
        y = torch.tensor([i.label for i in batch], device=lp.device)
        teacher = teacher_tensor(batch, lp) if self.cfg.loss.kd else None
        loss = decision_loss(lp, y, [i.row.qtype for i in batch], self.cfg.loss, teacher)
        row_weight = weight
        # collapse guard: the uniform predictor's loss under the SAME objective the step logs (base term, plus the
        # order-consistency view term with js = 0 and the zero predictor's axis MSE when those losses are on)
        ref = self._uniform_loss(lp.detach(), y, [i.row.qtype for i in batch], teacher)
        weight_order = self.cfg.loss.order_consistency_weight
        if weight_order:
            paired = [(i, item) for i, item in enumerate(batch) if getattr(item, "view", None) is not None]
            if paired:
                with self._autocast():
                    view_fw = self.model([item.view.row for _, item in paired])
                view_lp = view_fw.log_probs.float()
                view_y = torch.tensor([item.view.label for _, item in paired], device=view_lp.device)
                view_supervised = decision_loss(
                    view_lp, view_y, [item.view.row.qtype for _, item in paired], self.cfg.loss
                )
                js = []
                for j, (base_i, item) in enumerate(paired):
                    k = len(item.row.option_ids)
                    js.append(aligned_order_js(lp[base_i, :k], view_lp[j, :k], tuple(item.view_perm)))
                # Express extra-view terms per base row. _row_loss weights micro-batches by base-row count;
                # normalising each group by base rows makes gradients invariant to row_batch/group composition.
                loss = add_order_consistency(loss, view_supervised, torch.stack(js).mean(),
                                             len(batch), len(paired), weight_order)
                if ref is not None:  # the uniform predictor is order-consistent: its aligned JS is exactly 0
                    view_ref = self._uniform_loss(view_lp.detach(), view_y,
                                                  [item.view.row.qtype for _, item in paired], None)
                    ref = add_order_consistency(ref, view_ref, torch.zeros((), device=ref.device), len(batch),
                                                len(paired), weight_order)
        weight = self.cfg.axis_loss_weight
        if type(weight) not in (int, float) or not 0 <= weight < float("inf"):
            raise ValueError("axis_loss_weight must be finite and nonnegative")
        if weight:
            z = fw.head.semantic_axes
            if (z is None or z.shape != (len(batch), 6)
                    or any(i.axis_target is None or len(i.axis_target) != 6 for i in batch)):
                raise ValueError("axis supervision requires a [B,6] head port and six targets per Item")
            targets = torch.tensor([i.axis_target for i in batch], device=z.device, dtype=z.dtype)
            if not torch.isfinite(targets).all() or not torch.isfinite(z).all():
                raise ValueError("axis targets and port must be finite")
            loss = loss + weight * torch.nn.functional.mse_loss(z, targets)
            if ref is not None:  # the zero predictor of the axes (the uniform predictor carries no axis signal)
                ref = ref + weight * torch.nn.functional.mse_loss(torch.zeros_like(targets), targets).detach().float()
        if ref is not None:
            self._step_uniform += float(ref) * row_weight
        return loss

    def train_step(self) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        self._step_uniform = 0.0
        total = 0.0
        if self.cfg.row_batch > 1:
            # the same rows as the micro-batch loop below (same _next_batch calls, so epoch roll-over is unchanged); the
            # step loss sum_m mean(m) / grad_accum = sum_rows / (batch_size x grad_accum) = sum_g mean(g) x |g| / n
            rows = [it for _ in range(self.cfg.grad_accum) for it in self._next_batch()]
            n = len(rows)
            if self.cfg.loss.order_consistency_weight:
                base_lengths = [len(it.row.ids) for it in rows]
                view_lengths = [len(it.view.row.ids) if it.view is not None else 0 for it in rows]
                groups = row_groups_with_views(base_lengths, view_lengths, self.cfg.row_batch, self.cfg.row_tokens)
            else:
                plan = {"greedy": row_groups, "min_cost": row_groups_min_cost}[self.cfg.row_plan]
                groups = plan([len(it.row.ids) for it in rows], self.cfg.row_batch, self.cfg.row_tokens,
                              [it.row.n_options for it in rows], self.cfg.row_options)
            if self.cfg.cont_plan == "single" and len(groups) > 1 and single_cont_fits(
                    [it.row.n_options for it in rows], self.cfg.row_options):
                self.model.backbone.prefix_groups = groups
                try:
                    total += self._row_loss(rows, 1.0)
                finally:
                    self.model.backbone.prefix_groups = None
                return self._finish_step(total)
            for g in groups:
                total += self._row_loss([rows[j] for j in g], len(g) / n)
            return self._finish_step(total)
        for _ in range(self.cfg.grad_accum):
            batch = self._next_batch()
            with self._autocast():
                fw = self.model([i.row for i in batch])
            loss = self._supervised_loss(fw, batch, 1.0 / self.cfg.grad_accum) / self.cfg.grad_accum
            loss.backward()
            total += float(loss.detach())
        return self._finish_step(total)

    def _finish_step(self, total: float) -> float:
        params = [p for g in self.opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        self.opt.step()
        self.sched.step()
        self.step += 1
        self.history.append({"step": self.step, "loss": total})
        self._guard_step(total)
        return total

    def train(self, steps: int | None = None) -> list[dict[str, Any]]:
        target = self.cfg.max_steps if steps is None else self.step + steps
        if self.cfg.dev_card_every and self.step == 0 and not self.dev_cards and self.eval_items:
            self.run_dev_card()  # the step-0 card: the first best (the guard always has a checkpoint to go back to)
        while self.step < target and not self.stopped:
            self.train_step()
            if self.stopped:
                break
            if self.cfg.ckpt_dir and self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                self.save()
            if self.eval_items and self.cfg.eval_every and self.step % self.cfg.eval_every == 0:
                self.cards.append(run_card(self.model, self.eval_items, self.step, self.cfg.ckpt_dir))
            if self.eval_items and self.dev_card_due():
                self.run_dev_card()
        return self.history

    # ------------------------------------------------------------------ checkpoints
    def state(self) -> dict[str, Any]:
        return {
            "step": self.step, "epoch": self.epoch, "cursor": self.cursor,
            "config": self.cfg.to_json(), "config_hash": self.cfg.hash(),
            "trainable": self.trainable_state(),
            "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
            "rng": {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "lineage": self.lineage,
            "init_seed": getattr(self.model, "init_seed", None),
            "history": self.history,
            "backbone": getattr(self.model.backbone.spec, "name", None),
            "lora_modules": self.lora_modules,
            "guard": None if self.guard is None else self.guard.identity(),
            **({"anti_collapse": self.anti_collapse_state()} if self.anti_collapse_on() else {}),
        }

    def anti_collapse_on(self) -> bool:
        c = self.cfg
        return c.lr_schedule != "constant" or bool(c.dev_card_every) or c.best_ckpt or c.collapse_guard

    def anti_collapse_state(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "horizon_fixed": self.horizon_fixed, "lr_scale": self.lr_scale, "dev_cards": list(self.dev_cards),
                "best": None if self.best is None else dict(self.best), "collapse_events": list(self.collapse_events),
                "stopped": self.stopped, "guard_armed": self.guard_armed,
                # the loss trigger's window (<= collapse_window pairs): a resume (or chain link) continues it exactly
                **({"guard_window": [list(x) for x in self._guard_window]} if self.cfg.collapse_guard else {})}

    def save(self, path: str | Path | None = None) -> Path:
        d = Path(path or self.cfg.ckpt_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"ckpt-step{self.step:07d}.pt"
        tmp = p.with_suffix(".pt.tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, p)
        rec = {"path": p.name, "step": self.step, "config_hash": self.cfg.hash()}
        if self.cfg.best_ckpt and self.best and self.best.get("path"):
            rec["best"] = {k: self.best[k] for k in ("path", "step", "nll", "accuracy")}
        (d / "latest.json").write_text(json.dumps(rec), encoding="utf-8")
        return p

    @staticmethod
    def latest(ckpt_dir: str | Path) -> Path | None:
        """The LAST checkpoint (resume point)."""
        f = Path(ckpt_dir) / "latest.json"
        if not f.exists():
            return None
        return Path(ckpt_dir) / json.loads(f.read_text(encoding="utf-8"))["path"]

    @staticmethod
    def scored(ckpt_dir: str | Path) -> Path | None:
        """The checkpoint to SCORE: latest.json's ``best`` pointer when the run kept one (``best_ckpt``), else the
        last checkpoint (old runs: unchanged)."""
        f = Path(ckpt_dir) / "latest.json"
        if not f.exists():
            return None
        rec = json.loads(f.read_text(encoding="utf-8"))
        p = Path(ckpt_dir) / ((rec.get("best") or {}).get("path") or rec["path"])
        if not p.is_file():  # callers map None to their identity exit (no traceback, no silent fallback to the last)
            import sys

            print(f"scored checkpoint {p.name} named by latest.json is missing", file=sys.stderr)
            return None
        return p

    def load(self, path: str | Path, strict_config: bool = True, strict_lineage: bool = True) -> None:
        """Restore a ``.pt`` checkpoint (full: trainable + optimizer + scheduler + RNG), or an INFERENCE export
        (``train.ckpt_export``: ``trainable-stepN.safetensors`` + sidecar; trainable + step / history only, for eval,
        same config-hash and lineage checks), or a ``train.chain`` trainable-only best (``.pt`` with optimizer None:
        trainable + step / history, for scoring). ``self.loaded_format`` records which ("full" / "inference" /
        "trainable"); ``resume`` accepts only "full"."""
        from .ckpt_export import _jsonable, is_inference, read_inference

        if is_inference(path):
            tensors, st = read_inference(path)
            st["trainable"] = tensors
        else:
            st = torch.load(path, map_location="cpu", weights_only=False)
        if strict_config and st["config_hash"] != self.cfg.hash():
            raise ValueError(f"checkpoint config {st['config_hash']} != current {self.cfg.hash()}")
        if strict_lineage:
            # an export's sidecar is JSON: compare against the JSON form of this run's lineage (tuples -> lists)
            bad = lineage_diff(st.get("lineage") or {}, _jsonable(self.lineage) if is_inference(path) else self.lineage)
            if bad:
                raise ValueError(f"checkpoint lineage differs from this run: {bad} "
                                 "(frozen backbone / tokenizer / layout / training data must match to resume)")
        params = dict(self.model.named_parameters())
        missing = [n for n in st["trainable"] if n not in params]
        if missing:
            raise KeyError(f"checkpoint parameters not in model: {missing[:5]}")
        with torch.no_grad():
            for n, t in st["trainable"].items():
                params[n].copy_(t.to(params[n].dtype))
        self.step, self.epoch, self.cursor = st["step"], st["epoch"], st["cursor"]
        self.history = st["history"]
        ac = st.get("anti_collapse")
        if ac is not None:
            self.horizon, self.lr_scale = int(ac["horizon"]), float(ac["lr_scale"])
            self.horizon_fixed = bool(ac.get("horizon_fixed"))
            self.dev_cards, self.best = list(ac["dev_cards"]), ac["best"]
            self.collapse_events, self.stopped = list(ac["collapse_events"]), ac.get("stopped")
            self.guard_armed = bool(ac.get("guard_armed"))
            self._guard_window[:] = [(float(x), float(u)) for x, u in ac.get("guard_window") or []]
        if is_inference(path):
            self.loaded_format = "inference"
            return
        if st.get("optimizer") is None:  # train.chain trainable-only best: the fp32 weights, for scoring
            self.loaded_format = "trainable"
            return
        self.loaded_format = "full"
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["rng"]["torch"])
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        if st["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])

    def resume(self) -> bool:
        if not self.cfg.ckpt_dir:
            return False
        p = self.latest(self.cfg.ckpt_dir)
        if p is None:
            return False
        if str(p).endswith(".safetensors"):
            raise ValueError(f"{p.name} is an inference export (no optimizer / scheduler / RNG state): eval only, "
                             "resume needs the full .pt in the checkpoint dir")
        self.load(p)
        if self.loaded_format != "full":
            raise ValueError(f"{p.name} has no optimizer / scheduler / RNG state ({self.loaded_format}): not a resume point")
        return True

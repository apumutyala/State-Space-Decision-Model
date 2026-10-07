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
from .losses import LossConfig, decision_loss


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
    #: per-block activation checkpointing of the RWKV-7 backbone that keeps the cache fork (heads/act_ckpt.py)
    act_ckpt: bool = False
    #: rows per backbone call within one optimiser step (1 = one call per micro-batch row, as before). > 1: the step's
    #: batch_size x grad_accum rows are sorted by length (longest first) and run in groups of up to row_batch rows, each
    #: group in ONE prefix call + ONE continuation call (heads.backbone.fork_features_rows); every group's loss is
    #: weighted by its share of the step's rows, so the step's gradient is the same sum over rows.
    row_batch: int = 1
    #: with row_batch > 1: max padded tokens per call (rows x longest row of the group); 0 = no limit. A row longer
    #: than the budget runs alone. Bounds the per-call activation memory (jobs/cell_step_profile JOB_SPEC, A1).
    row_tokens: int = 0
    axis_loss_weight: float = 0.0  # opt-in MSE on HeadOutput.semantic_axes, targets live only on Item

    def to_json(self) -> dict:
        d = asdict(self)
        d["loss"] = asdict(self.loss)
        if d["axis_loss_weight"] == 0.0:
            d.pop("axis_loss_weight")  # preserve legacy config identities
        for k in self.EXECUTION:  # at their defaults the execution fields are left out: old records stay identical
            if d[k] == getattr(TrainConfig, k):
                d.pop(k)
        return d

    #: operational fields that may differ between a run and its resumption
    OPERATIONAL = ("ckpt_dir", "ckpt_every", "eval_every", "max_steps", "require_guard")
    #: execution-strategy fields: same loss and gradients, not part of the hash (old hashes unchanged)
    EXECUTION = ("act_ckpt", "row_batch", "row_tokens")

    def hash(self) -> str:
        d = {k: v for k, v in self.to_json().items() if k not in self.OPERATIONAL + self.EXECUTION}
        if not d.get("controlsafe"):
            d.pop("controlsafe", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def with_execution(cfg: TrainConfig, cell: dict) -> TrainConfig:
    """``cfg`` with the execution-strategy keys of a cell config (``act_ckpt``, ``row_batch``, ``row_tokens``; absent =
    the original path). For ``train.run_cell.train_config`` callers: ``with_execution(train_config(cell, ...), cell)``."""
    import dataclasses

    return dataclasses.replace(cfg, act_ckpt=bool(cell.get("act_ckpt", False)), row_batch=int(cell.get("row_batch", 1)),
                               row_tokens=int(cell.get("row_tokens", 0)))


def row_groups(lengths: list[int], row_batch: int, row_tokens: int = 0) -> list[list[int]]:
    """Indices of the step's rows grouped for ``row_batch`` > 1: sorted by length, longest first (stable), filled
    greedily while a group has <= ``row_batch`` rows and, if ``row_tokens`` > 0, (rows x its longest row) <=
    ``row_tokens`` padded tokens. Every row lands in exactly one group."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups: list[list[int]] = []
    for i in order:
        g = groups[-1] if groups else None
        if g is not None and len(g) < row_batch and (
                not row_tokens or (len(g) + 1) * lengths[g[0]] <= row_tokens):
            g.append(i)
        else:
            groups.append([i])
    return groups


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

    if cfg.act_ckpt:
        act_ckpt.enable_block_checkpointing(model.backbone.model)  # raises unless the backbone is the vendored RWKV-7
    elif act_ckpt.is_enabled(model.backbone.model):  # a backbone reused across Trainers (run_cell ladder)
        act_ckpt.disable_block_checkpointing(model.backbone.model)
    if cfg.row_batch < 1:
        raise ValueError(f"row_batch must be >= 1, got {cfg.row_batch}")
    model.backbone.batch_rows = cfg.row_batch > 1
    return replaced


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
        warm = max(cfg.warmup_steps, 0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
        self.step = 0
        self.epoch = 0
        self.cursor = 0
        self.history: list[dict[str, Any]] = []
        self.cards: list[dict[str, Any]] = []
        self.lineage = lineage(model.backbone, train_items)

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
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)

    def _row_loss(self, batch: list[Item], weight: float) -> float:
        with self._autocast():
            fw = self.model([i.row for i in batch])
        loss = self._supervised_loss(fw, batch) * weight
        loss.backward()
        return float(loss.detach())

    def _supervised_loss(self, fw, batch: list[Item]) -> torch.Tensor:
        lp = fw.log_probs.float()
        y = torch.tensor([i.label for i in batch], device=lp.device)
        loss = decision_loss(lp, y, [i.row.qtype for i in batch], self.cfg.loss)
        weight = self.cfg.axis_loss_weight
        if not isinstance(weight, (int, float)) or not 0 <= weight < float("inf"):
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
        return loss

    def train_step(self) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        if self.cfg.row_batch > 1:
            # the same rows as the micro-batch loop below (same _next_batch calls, so epoch roll-over is unchanged); the
            # step loss sum_m mean(m) / grad_accum = sum_rows / (batch_size x grad_accum) = sum_g mean(g) x |g| / n
            rows = [it for _ in range(self.cfg.grad_accum) for it in self._next_batch()]
            n = len(rows)
            for g in row_groups([len(it.row.ids) for it in rows], self.cfg.row_batch, self.cfg.row_tokens):
                total += self._row_loss([rows[j] for j in g], len(g) / n)
            return self._finish_step(total)
        for _ in range(self.cfg.grad_accum):
            batch = self._next_batch()
            with self._autocast():
                fw = self.model([i.row for i in batch])
            loss = self._supervised_loss(fw, batch) / self.cfg.grad_accum
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
        return total

    def train(self, steps: int | None = None) -> list[dict[str, Any]]:
        target = self.cfg.max_steps if steps is None else self.step + steps
        while self.step < target:
            self.train_step()
            if self.cfg.ckpt_dir and self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                self.save()
            if self.eval_items and self.cfg.eval_every and self.step % self.cfg.eval_every == 0:
                self.cards.append(run_card(self.model, self.eval_items, self.step, self.cfg.ckpt_dir))
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
        }

    def save(self, path: str | Path | None = None) -> Path:
        d = Path(path or self.cfg.ckpt_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"ckpt-step{self.step:07d}.pt"
        tmp = p.with_suffix(".pt.tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, p)
        (d / "latest.json").write_text(json.dumps({"path": p.name, "step": self.step,
                                                   "config_hash": self.cfg.hash()}), encoding="utf-8")
        return p

    @staticmethod
    def latest(ckpt_dir: str | Path) -> Path | None:
        f = Path(ckpt_dir) / "latest.json"
        if not f.exists():
            return None
        return Path(ckpt_dir) / json.loads(f.read_text(encoding="utf-8"))["path"]

    def load(self, path: str | Path, strict_config: bool = True, strict_lineage: bool = True) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        if strict_config and st["config_hash"] != self.cfg.hash():
            raise ValueError(f"checkpoint config {st['config_hash']} != current {self.cfg.hash()}")
        if strict_lineage:
            bad = lineage_diff(st.get("lineage") or {}, self.lineage)
            if bad:
                raise ValueError(f"checkpoint lineage differs from this run: {bad} "
                                 "(frozen backbone / tokenizer / layout / training data must match to resume)")
        params = dict(self.model.named_parameters())
        missing = [n for n in st["trainable"] if n not in params]
        if missing:
            raise KeyError(f"checkpoint parameters not in model: {missing[:5]}")
        with torch.no_grad():
            for n, t in st["trainable"].items():
                params[n].copy_(t)
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.step, self.epoch, self.cursor = st["step"], st["epoch"], st["cursor"]
        self.history = st["history"]
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
        self.load(p)
        return True

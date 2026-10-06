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

    def to_json(self) -> dict:
        d = asdict(self)
        d["loss"] = asdict(self.loss)
        return d

    #: operational fields that may differ between a run and its resumption
    OPERATIONAL = ("ckpt_dir", "ckpt_every", "eval_every", "max_steps", "require_guard")

    def hash(self) -> str:
        d = {k: v for k, v in self.to_json().items() if k not in self.OPERATIONAL}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def prepare_model(model: DecisionModel, cfg: TrainConfig) -> list[str]:
    """Freeze the backbone, add LoRA (rank 0 = head-only), enable checkpointing."""
    targets = cfg.lora_targets or (model.backbone.spec.lora_groups if model.backbone.spec else None)
    replaced: list[str] = []
    if cfg.lora_rank > 0:
        if targets is None:
            raise ValueError("no LoRA targets: set cfg.lora_targets or give the backbone a spec")
        with torch.random.fork_rng(devices=[]):  # LoRA init depends only on cfg.seed, never the ambient RNG
            torch.manual_seed(cfg.seed * 1_000_003 + 17)
            found = apply_lora(model.backbone.model, targets, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
        replaced = sorted({n for names in found.values() for n in names})
    else:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    model.backbone.fork_mode = cfg.fork_mode
    if cfg.grad_checkpointing:
        model.backbone.enable_gradient_checkpointing()
    return replaced


class Trainer:
    def __init__(self, model: DecisionModel, cfg: TrainConfig, train_items: list[Item],
                 eval_items: list[Item] | None = None, prepared: bool = False,
                 guard: SplitGuard | None = None) -> None:
        if cfg.require_guard and guard is None:
            raise ValueError("TrainConfig.require_guard: pass a SplitGuard built from the Index dev/test "
                             "manifests (or SplitGuard.synthetic_only for synthetic toys)")
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
        self._seed_all(cfg.seed)

    # ------------------------------------------------------------------ data order
    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

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

    def train_step(self) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.cfg.grad_accum):
            batch = self._next_batch()
            with self._autocast():
                fw = self.model([i.row for i in batch])
            lp = fw.log_probs.float()
            y = torch.tensor([i.label for i in batch], device=lp.device)
            loss = decision_loss(lp, y, [i.row.qtype for i in batch], self.cfg.loss) / self.cfg.grad_accum
            loss.backward()
            total += float(loss.detach())
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

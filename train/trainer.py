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


def _checkpoint_float(value: Any) -> bool:
    return type(value) in (int, float) and np.isfinite(value)


def _validate_optimizer_checkpoint(saved: Any, optimizer: Any) -> None:
    """Preflight AdamW's positional state mapping before any model/checkpoint mutation."""
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state must contain exactly state and param_groups")
    saved_groups, states = saved["param_groups"], saved["state"]
    current_state = optimizer.state_dict()
    groups = optimizer.param_groups
    current_groups = current_state["param_groups"]
    if (not isinstance(saved_groups, list) or len(saved_groups) != len(groups)
            or len(current_groups) != len(groups) or not isinstance(states, dict)):
        raise ValueError("checkpoint optimizer group/state structure differs")
    id_to_param: dict[int, torch.nn.Parameter] = {}
    id_to_group: dict[int, dict[str, Any]] = {}
    all_ids: set[int] = set()
    group_keys = set(current_groups[0]) if current_groups else set()
    allowed_group_keys = {"params", "lr", "initial_lr", "betas", "eps", "weight_decay", "amsgrad",
                          "maximize", "foreach", "capturable", "differentiable", "fused",
                          "decoupled_weight_decay"}
    if group_keys - allowed_group_keys:
        raise ValueError("current optimizer uses unsupported parameter-group options")
    for index, (saved_group, current_group) in enumerate(zip(saved_groups, groups)):
        saved_ids = saved_group.get("params") if isinstance(saved_group, dict) else None
        current_params = current_group.get("params")
        if (not isinstance(saved_group, dict) or set(saved_group) != group_keys
                or not isinstance(saved_ids, list) or not isinstance(current_params, list)
                or len(saved_ids) != len(current_params)):
            raise ValueError(f"checkpoint optimizer parameter group {index} differs")
        if (type(saved_group.get("amsgrad")) is not bool or type(saved_group.get("maximize")) is not bool
                or type(saved_group.get("capturable")) is not bool
                or type(saved_group.get("differentiable")) is not bool
                or ("decoupled_weight_decay" in saved_group
                    and saved_group["decoupled_weight_decay"] is not True)
                or any(saved_group.get(key) is not None and type(saved_group.get(key)) is not bool
                       for key in ("foreach", "fused"))):
            raise ValueError(f"checkpoint optimizer boolean options in group {index} are malformed")
        if (not _checkpoint_float(saved_group.get("lr")) or saved_group["lr"] < 0
                or not _checkpoint_float(saved_group.get("eps")) or saved_group["eps"] < 0
                or not _checkpoint_float(saved_group.get("weight_decay")) or saved_group["weight_decay"] < 0
                or ("initial_lr" in saved_group
                    and (not _checkpoint_float(saved_group["initial_lr"]) or saved_group["initial_lr"] < 0))
                or not isinstance(saved_group.get("betas"), (tuple, list)) or len(saved_group["betas"]) != 2
                or any(not _checkpoint_float(beta) or not 0 <= beta < 1 for beta in saved_group["betas"])):
            raise ValueError(f"checkpoint optimizer numeric options in group {index} are malformed")
        for saved_id, param in zip(saved_ids, current_params):
            if type(saved_id) is not int or saved_id in all_ids or not isinstance(param, torch.nn.Parameter):
                raise ValueError("checkpoint optimizer parameter IDs are duplicated or malformed")
            all_ids.add(saved_id)
            id_to_param[saved_id] = param
            id_to_group[saved_id] = saved_group
    if any(type(param_id) is not int or param_id not in id_to_param for param_id in states):
        raise ValueError("checkpoint optimizer has state for an unknown parameter ID")
    for param_id, state in states.items():
        expected_state = {"step", "exp_avg", "exp_avg_sq"}
        if id_to_group[param_id].get("amsgrad"):
            expected_state.add("max_exp_avg_sq")
        if not isinstance(state, dict) or set(state) != expected_state:
            raise ValueError("checkpoint optimizer parameter state is malformed")
        param = id_to_param[param_id]
        for name, value in state.items():
            if name == "step":
                if isinstance(value, torch.Tensor):
                    if (value.ndim != 0 or value.is_complex() or value.dtype == torch.bool
                            or not torch.isfinite(value).all() or float(value.item()) < 0
                            or not float(value.item()).is_integer()):
                        raise ValueError("checkpoint optimizer step is malformed")
                elif (not _checkpoint_float(value) or value < 0 or not float(value).is_integer()):
                    raise ValueError("checkpoint optimizer step is malformed")
            elif (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(param.shape)
                  or value.dtype != param.dtype or value.layout != torch.strided
                  or not torch.isfinite(value).all()):
                raise ValueError(f"checkpoint optimizer {name} shape/dtype/value differs for parameter {param_id}")


def _validate_scheduler_checkpoint(saved: Any, scheduler: Any, group_count: int) -> None:
    if not isinstance(saved, dict) or set(saved) != set(scheduler.state_dict()):
        raise ValueError("checkpoint scheduler structure differs")
    for key in ("last_epoch", "_step_count"):
        if key in saved and (type(saved[key]) is not int or saved[key] < -1):
            raise ValueError(f"checkpoint scheduler {key} is malformed")
    for key in ("base_lrs", "_last_lr"):
        if key in saved and (not isinstance(saved[key], list) or len(saved[key]) != group_count
                             or any(not _checkpoint_float(x) for x in saved[key])):
            raise ValueError(f"checkpoint scheduler {key} is malformed")
    if "lr_lambdas" in saved and (not isinstance(saved["lr_lambdas"], list)
                                  or len(saved["lr_lambdas"]) != group_count
                                  or any(value is not None for value in saved["lr_lambdas"])):
        raise ValueError("checkpoint scheduler lambda state is unsupported")


def _validate_rng_checkpoint(saved: Any) -> None:
    if not isinstance(saved, dict) or set(saved) != {"torch", "python", "numpy", "cuda"}:
        raise ValueError("checkpoint RNG state structure differs")
    cpu_state = saved["torch"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
        raise ValueError("checkpoint Torch CPU RNG state is malformed")
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
        random.Random().setstate(saved["python"])
        np.random.RandomState().set_state(saved["numpy"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"checkpoint CPU RNG state is malformed: {type(exc).__name__}") from None
    cuda_states = saved["cuda"]
    if cuda_states is not None:
        if (not isinstance(cuda_states, list)
                or any(not isinstance(s, torch.Tensor) or s.dtype != torch.uint8 or s.ndim != 1
                       for s in cuda_states)
                or (torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count())):
            raise ValueError("checkpoint CUDA RNG state is malformed")
        if torch.cuda.is_available():
            try:
                # Validate each device state on an independent generator before touching the process RNG.
                for device, state in enumerate(cuda_states):
                    torch.Generator(device=f"cuda:{device}").set_state(state)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"checkpoint CUDA RNG state is malformed: {type(exc).__name__}") from None
from .lora import apply_lora, lora_parameters
from .losses import LossConfig, decision_loss


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
    axis_loss_weight: float = 0.0  # opt-in MSE on HeadOutput.semantic_axes, targets live only on Item

    def to_json(self) -> dict:
        d = asdict(self)
        d["loss"] = self.loss.identity()  # KD fields only when KD is on (old hashes unchanged)
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
        act_ckpt.enable_block_checkpointing(model.backbone.model)  # RWKV-7 or Qwen3.5 / Q9-AF; raises otherwise
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
        if cfg.loss.kd and not any(getattr(i, "teacher", None) is not None for i in train_items):
            raise ValueError("loss.kd_alpha > 0 but no training item carries a teacher distribution: call "
                             "train.teacher.attach(items, load_teacher(...)) first (a KD run without a teacher is the "
                             "clef arm under another name)")
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
        teacher = teacher_tensor(batch, lp) if self.cfg.loss.kd else None
        loss = decision_loss(lp, y, [i.row.qtype for i in batch], self.cfg.loss, teacher)
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
        if not isinstance(st, dict):
            raise ValueError("checkpoint root must be a dictionary")
        if strict_config and st["config_hash"] != self.cfg.hash():
            raise ValueError(f"checkpoint config {st['config_hash']} != current {self.cfg.hash()}")
        if strict_lineage:
            bad = lineage_diff(st.get("lineage") or {}, self.lineage)
            if bad:
                raise ValueError(f"checkpoint lineage differs from this run: {bad} "
                                 "(frozen backbone / tokenizer / layout / training data must match to resume)")
        # Use the same duplicate-eliding parameter names as trainable_state(). Validate the entire resume payload
        # before copying the first weight or restoring optimizer/scheduler/RNG state. The strict flags only govern
        # config and data lineage; neither disables trainable-state integrity.
        params = dict(self.model.named_parameters())
        expected = {name: param for name, param in params.items() if param.requires_grad}
        saved = st.get("trainable")
        if not isinstance(saved, dict):
            raise ValueError("checkpoint trainable state must be a name-to-tensor dictionary")
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        if missing or unexpected:
            raise ValueError(f"checkpoint trainable keys differ: missing={missing[:5]}, unexpected={unexpected[:5]}")
        for name, param in expected.items():
            tensor = saved[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tuple(tensor.shape) != tuple(param.shape) or tensor.dtype != param.dtype
                    or not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
                raise ValueError(f"checkpoint trainable tensor shape/dtype/layout/value differs for {name}")

        _validate_optimizer_checkpoint(st.get("optimizer"), self.opt)
        _validate_scheduler_checkpoint(st.get("scheduler"), self.sched, len(self.opt.param_groups))
        for key in ("step", "epoch", "cursor"):
            if type(st.get(key)) is not int or st[key] < 0:
                raise ValueError(f"checkpoint {key} counter is malformed")
        if st["cursor"] > len(self.train_items) or not isinstance(st.get("history"), list):
            raise ValueError("checkpoint cursor or history is malformed")
        _validate_rng_checkpoint(st.get("rng"))

        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(saved[name])
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

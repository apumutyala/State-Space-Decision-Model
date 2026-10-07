"""Fast-iteration proxy for the m2 training recipe: a head trained on CACHED frozen-backbone features (PREMORTEM.md §8).

The backbone is run ONCE per training item under ``no_grad`` and its ``HeadInput`` (prefix features, option
embeddings, fork features) is stored on CPU. Every recipe iteration after that touches only the head: no backbone
forward, no backward through the backbone, no LoRA. On a 9B backbone the cache costs one inference pass over the
training items (the parent-arm readout rate), and each head iteration then runs in seconds to minutes on one GPU or
on CPU.

Exactness (tested, ``tests/test_premortem_probes.py``): for LoRA rank 0, ``train_head_cached`` reproduces
``train.trainer.Trainer`` (``batch_size=1``) step for step, with the same data order, optimiser, scheduler and
clipping. So every head-only recipe decision (loss arm, label smoothing, Brier/RPS weights, head type and width,
learning rate, warmup, data mix and order, calibration) is evaluated EXACTLY, not approximately. What the proxy cannot
see is the LoRA term; ``kendall_tau`` quantifies how well a recipe ranking under the proxy carries over to head + LoRA
training (the transfer measurement in the same test file).

Not for the trust-gated (G15/G16) row heads: they have no shared ``HeadInput``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch

from heads.features import HeadInput, build_head_input

from .losses import decision_loss
from .trainer import TrainConfig


@dataclass
class CachedItem:
    x: HeadInput
    label: int
    qtype: str


def _to(x: HeadInput, dev) -> HeadInput:
    kw = {}
    for f in ("h", "token_mask", "evidence_mask", "q_pre", "q_end", "e", "option_mask", "fork"):
        v = getattr(x, f)
        kw[f] = None if v is None else v.detach().to(dev)
    return HeadInput(ks=list(x.ks), **kw)


@torch.no_grad()
def cache_items(model, items: Sequence, device: str = "cpu") -> list[CachedItem]:
    """One frozen-backbone pass per item (B = 1, so no batch padding enters any feature)."""
    if model.row_head:
        raise ValueError("row heads (G16) have no shared HeadInput; the proxy does not apply")
    out = []
    for it in items:
        x = build_head_input(model.backbone, [it.row], need_fork=model.head.needs_fork, pad_id=model.pad_id)
        out.append(CachedItem(_to(x, device), it.label, it.row.qtype))
    return out


def _order(seed: int, epoch: int, n: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)  # == train.trainer.Trainer._order
    return torch.randperm(n, generator=g).tolist()


def _lp(model, c: CachedItem) -> torch.Tensor:
    out = model.head(c.x)
    return model.typed(out.logits, [c.qtype], c.x.ks, c.x.e)


def train_head_cached(model, cached: list[CachedItem], cfg: TrainConfig, steps: int | None = None) -> dict:
    """Head-only training on cached features; the update rule of ``Trainer`` with ``batch_size == 1``."""
    if cfg.batch_size != 1:
        raise ValueError("the proxy batches by gradient accumulation; use batch_size=1, grad_accum=N")
    if cfg.lora_rank:
        raise ValueError("the proxy has no backbone in the loop: lora_rank must be 0")
    torch.manual_seed(cfg.seed)
    params = model.head_parameters()
    opt = torch.optim.AdamW([{"params": params, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    warm = max(cfg.warmup_steps, 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
    n, epoch, cursor, hist = len(cached), 0, 0, []
    t0 = time.perf_counter()
    for _ in range(cfg.max_steps if steps is None else steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.grad_accum):
            order = _order(cfg.seed, epoch, n)
            if cursor + 1 > n:
                epoch, cursor = epoch + 1, 0
                order = _order(cfg.seed, epoch, n)
            c = cached[order[cursor]]
            cursor += 1
            lp = _lp(model, c).float()
            y = torch.tensor([c.label], device=lp.device)
            loss = decision_loss(lp, y, [c.qtype], cfg.loss) / cfg.grad_accum
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        hist.append(total)
    return {"history": hist, "seconds": time.perf_counter() - t0}


@torch.no_grad()
def eval_cached(model, cached: list[CachedItem]) -> dict:
    nll, acc = 0.0, 0
    for c in cached:
        lp = _lp(model, c)[0].float()
        nll -= float(lp[c.label])
        acc += int(int(lp.argmax()) == c.label)
    return {"nll": nll / len(cached), "acc": acc / len(cached)}


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Tau-a between two score lists over the same recipes (ties count as neither concordant nor discordant)."""
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += (1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1 if (a[i] - a[j]) * (b[i] - b[j]) < 0 else 0)
    return 2 * s / (n * (n - 1))

"""Minimal LoRA (no peft dependency): frozen base Linear + low-rank update B @ A, B zero-init.

``apply_lora`` replaces every ``nn.Linear`` whose qualified name matches ``targets`` (regex,
``re.search``) and freezes every non-LoRA parameter of the module tree. Rank is configurable;
the m2 default is 256 (Clef-flash recipe, reviews/01). With alpha = rank the update scale is 1.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:  # some HF code paths read .weight (dtype/device probes)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return self.base(x) + (self.dropout(x) @ a.t() @ b.t()) * self.scaling


def apply_lora(model: nn.Module, targets: str, rank: int = 256, alpha: float | None = None,
               dropout: float = 0.0) -> list[str]:
    pattern = re.compile(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and pattern.search(name):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    if not replaced:
        raise ValueError(f"LoRA target pattern {targets!r} matched no nn.Linear")
    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}


def lora_param_count(shapes, rank: int, n_layers: int) -> int:
    """Analytic LoRA parameter count for per-layer (in, out) projection shapes."""
    return n_layers * sum(rank * (i + o) for i, o in shapes)

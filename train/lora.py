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
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0,
                 generator: torch.Generator | None = None) -> None:
        """``lora_A`` is drawn on CPU from ``generator`` (an explicit CPU generator: no global or
        CUDA RNG is read or advanced), then moved to the base layer's device; fp32 master weights."""
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        dev = base.weight.device
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
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


def apply_lora(model: nn.Module, targets, rank: int = 256, alpha: float | None = None,
               dropout: float = 0.0, check_counts: bool = True,
               generator: torch.Generator | None = None) -> dict[str, list[str]]:
    """Wrap matching ``nn.Linear`` modules. ``targets`` is a regex or a sequence of
    ``(group, regex, expected_count)``. Every group must match at least one module, and with
    ``check_counts`` exactly ``expected_count`` (when > 0): a wrong module-name convention
    (e.g. fla vs HF RWKV7 names) fails loudly instead of silently training a subset."""
    groups = [("custom", targets, 0)] if isinstance(targets, str) else list(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear)]
    found: dict[str, list[str]] = {}
    for gname, rx, expected in groups:
        pat = re.compile(rx)
        names = [n for n, _ in linears if pat.search(n)]
        if not names:
            raise ValueError(f"LoRA group {gname!r} ({rx!r}) matched no nn.Linear: wrong module names for this backbone?")
        if check_counts and expected and len(names) != expected:
            raise ValueError(f"LoRA group {gname!r} matched {len(names)} modules, expected {expected}")
        found[gname] = names
    done: set[str] = set()
    for names in found.values():
        for name in names:
            if name in done:
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, LoRALinear(model.get_submodule(name), rank, alpha, dropout, generator))
            done.add(name)
    return found


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}


def lora_param_count(shapes, rank: int) -> int:
    """Analytic LoRA parameter count for (in, out, count) projection shapes."""
    return sum(rank * (i + o) * c for i, o, c in shapes)

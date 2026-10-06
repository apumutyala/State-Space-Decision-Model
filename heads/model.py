"""DecisionModel = frozen backbone (+ LoRA, applied by the trainer) + head + typed outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from .backbone import HiddenStateBackbone
from .features import HeadInput, build_head_input
from .heads import DecisionHead, HeadOutput, make_head
from .layout import EncodeFn, LayoutRow, build_row
from .typed import TypedOutput, to_wire


@dataclass
class Forward:
    log_probs: torch.Tensor  # [B, K]
    head: HeadOutput
    inputs: HeadInput


class DecisionModel(nn.Module):
    def __init__(self, backbone: HiddenStateBackbone, head: DecisionHead | str, head_kwargs: dict | None = None,
                 pad_id: int = 0, seed: int = 0) -> None:
        """Head and typed-output parameters are initialised from ``seed`` inside a forked RNG, so
        the result never depends on (or disturbs) the ambient torch RNG. A pre-built head module
        is used as given (``init_seed`` is then None)."""
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        self.init_seed = seed if isinstance(head, str) else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = make_head(head, d, **(head_kwargs or {})) if isinstance(head, str) else head
            self.typed = TypedOutput(d)
        self.pad_id = pad_id

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.typed.parameters())

    def forward(self, rows: Sequence[LayoutRow], return_attribution: bool = False) -> Forward:
        x = build_head_input(self.backbone, rows, need_fork=self.head.needs_fork, pad_id=self.pad_id)
        out = self.head(x, return_attribution=return_attribution)
        lp = self.typed(out.logits, [r.qtype for r in rows], x.ks, x.e)
        return Forward(lp, out, x)

    @torch.no_grad()
    def decide(self, request: dict[str, Any], encode: EncodeFn, layout: str = "ordered",
               template: str = "jrt-plain-v1") -> dict[str, Any]:
        """SystemOne ``answers`` for a validated request (one row per question)."""
        answers = {}
        for qid, q in request["questions"].items():
            row = build_row(request, qid, encode, template=template, layout=layout)
            lp = self.forward([row]).log_probs[0]
            answers[qid] = to_wire(q, row.option_ids, lp)
        return answers

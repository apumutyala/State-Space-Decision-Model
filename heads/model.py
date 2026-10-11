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
    inputs: HeadInput | None  # None for row heads (G16: two streams per row, no shared HeadInput)


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
        # torch.manual_seed also seeds CUDA, so fork (and restore) every CUDA device too.
        cuda = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed)
            if head == "g16":  # controlsafe G16 (protected judge + bounded evidence) over the trust-gated backbone
                if not getattr(backbone, "wants_trust", False):
                    raise ValueError("head 'g16' needs a controlsafe TrustGatedBackbone (cfg.controlsafe.trust_gate)")
                from controlsafe.g16 import G16DecisionHead

                self.head = G16DecisionHead(backbone, seed=seed, **(head_kwargs or {}))
            else:
                self.head = make_head(head, d, **(head_kwargs or {})) if isinstance(head, str) else head
            self.typed = TypedOutput(d)
        if self.row_head:
            for p in self.typed.parameters():  # unused: G16 decides by softmax over its logits for every qtype
                p.requires_grad_(False)
        self.pad_id = pad_id

    @property
    def row_head(self) -> bool:
        return bool(getattr(self.head, "wants_rows", False))

    def head_parameters(self):
        if self.row_head:  # the head shares the backbone module: its parameters train in the LoRA group
            return self.head.own_parameters()
        return list(self.head.parameters()) + list(self.typed.parameters())

    def forward(self, rows: Sequence[LayoutRow], return_attribution: bool = False,
                shared_prefix: bool = False) -> Forward:
        if self.row_head:
            for r in rows:
                if r.qtype == "noul" and r.n_options != 2:
                    raise ValueError("noul questions have exactly two options (true, false)")
            out = self.head.forward_rows(rows)
            return Forward(torch.log_softmax(out.logits, -1), out, None)
        x = build_head_input(self.backbone, rows, need_fork=self.head.needs_fork, pad_id=self.pad_id,
                             shared_prefix=shared_prefix)
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
            from .symmetric import sym_tie_info

            answers[qid] = to_wire(q, row.option_ids, lp, sym_tie_info(row))
        return answers

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


def record_key(row: LayoutRow):
    """Rows of one request share the token prefix template head + schema of ALL fields + state. ``meta["record"]``
    (when a caller sets it) wins; else that prefix (``meta["prefix_end"]``, set by ``build_row``; older rows: the state
    span end for state evidence) or the original state span (re-read evidence,
    ``meta["untrusted_spans"]``); a row with neither is its own record."""
    if row.meta.get("record") is not None:
        return ("record", row.meta["record"])
    if row.meta.get("prefix_end") is not None:
        return ("prefix", tuple(row.ids[: row.meta["prefix_end"]]))
    if row.meta.get("evidence", "state") == "state":
        return ("prefix", tuple(row.ids[: row.state_span[1]]))
    us = row.meta.get("untrusted_spans") or []
    if us:
        return ("prefix", tuple(row.ids[: us[0][1]]))
    return ("row", id(row))


def record_groups(rows: Sequence[LayoutRow]) -> list[list[int]]:
    """Row indices grouped by record (first-appearance order), each group in SCHEMA order (field position)."""
    groups: dict[Any, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(record_key(r), []).append(i)
    return [sorted(g, key=lambda i: (rows[i].field_span[0], i)) for g in groups.values()]


def _input_embedding(backbone) -> nn.Module:
    m = backbone.model
    get = getattr(m, "get_input_embeddings", None)
    emb = get() if callable(get) else None
    if emb is None:
        emb = next((x for x in m.modules() if isinstance(x, nn.Embedding)), None)
    if emb is None:
        raise ValueError("joint_record lexical prior: the backbone has no input embedding table")
    return emb


@torch.no_grad()
def lexical_vectors(backbone, rows: Sequence[LayoutRow], kmax: int) -> torch.Tensor:
    """[B, K, D] frozen input-embedding mean over each option's text (ordered and g1v2 layouts: the option's schema
    span, id + description; else, or on an empty span, the option-id tokens of its continuation -- for g1v2 code rows
    that would be the bare code letter, the same for every question, so the schema span is used there). Zero on
    padded options. No gradient (the table is frozen)."""
    emb = _input_embedding(backbone)
    dev = emb.weight.device
    out = torch.zeros((len(rows), kmax, emb.weight.shape[1]), dtype=torch.float32, device=dev)
    for i, r in enumerate(rows):
        for k in range(r.n_options):
            if r.layout in ("ordered", "g1v2") and r.option_spans and r.option_spans[k][1] > r.option_spans[k][0]:
                a, b = r.option_spans[k]
                toks = r.ids[a:b]
            else:
                sp = (r.meta.get("cont_spans") or [[]] * r.n_options)[k]
                toks = [t for (a, b, _p) in sp for t in r.continuations[k][a:b]] or r.continuations[k]
            if toks:
                out[i, k] = emb(torch.tensor(toks, dtype=torch.long, device=dev)).float().mean(0)
    return out


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
        if getattr(self.head, "needs_record", False):  # joint_record head: lexical vectors + record groups
            x.lex = lexical_vectors(self.backbone, rows, x.e.shape[1])
            x.groups = record_groups(rows)
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

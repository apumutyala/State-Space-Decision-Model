"""Turn LayoutRows into a HeadInput by running the backbone once (plus forks if asked).

Feature definitions (all from the backbone's final features unless ``layer`` says otherwise):

* ``h``       token features of the shared prefix ``[B, T, D]``
* ``evidence_mask``  True on state tokens (the only positions heads integrate over)
* ``q_pre``   mean over the target field's tokens in the schema. Known *before* the state, so
              streaming heads (integrator, snapshot, Head B) may condition on it.
* ``q_end``   feature at the answer position (after the re-read). Only fork/control heads use it.
* ``e``       option embeddings ``[B, K, D]``: mean over the option's span (ordered layout) or
              over its own stand-alone sequence (symmetric layout).
* ``fork``    final-state fork features ``[B, K, D]``: feature at the last token of
              prefix + continuation_k (recompute fork; only when ``need_fork``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .backbone import HiddenStateBackbone, pad_batch
from .layout import LayoutRow


@dataclass
class HeadInput:
    h: torch.Tensor
    token_mask: torch.Tensor
    evidence_mask: torch.Tensor
    q_pre: torch.Tensor
    q_end: torch.Tensor
    e: torch.Tensor
    option_mask: torch.Tensor
    fork: torch.Tensor | None = None

    @property
    def n_options(self) -> torch.Tensor:
        return self.option_mask.sum(-1)

    @property
    def evidence_len(self) -> torch.Tensor:
        return self.evidence_mask.sum(-1)

    def permute_options(self, perm: torch.Tensor) -> "HeadInput":
        """Same input with the option axis permuted by ``perm`` (shape [K])."""
        return HeadInput(
            h=self.h, token_mask=self.token_mask, evidence_mask=self.evidence_mask, q_pre=self.q_pre,
            q_end=self.q_end, e=self.e[:, perm], option_mask=self.option_mask[:, perm],
            fork=None if self.fork is None else self.fork[:, perm],
        )


def _span_mean(h: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    a, b = span
    return h[a:b].mean(0)


def build_head_input(backbone: HiddenStateBackbone, rows: Sequence[LayoutRow], need_fork: bool = False,
                     pad_id: int = 0, fork_chunk: int = 64) -> HeadInput:
    ids, mask = pad_batch([r.ids for r in rows], pad_id)
    h = backbone.features(ids, mask).float()
    bsz, t, d = h.shape
    kmax = max(r.n_options for r in rows)
    ev = torch.zeros((bsz, t), dtype=torch.bool)
    q_pre = torch.zeros((bsz, d))
    q_end = torch.zeros((bsz, d))
    opt_mask = torch.zeros((bsz, kmax), dtype=torch.bool)
    e_rows: list[torch.Tensor] = []
    sym_seqs: list[list[int]] = []
    sym_index: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        a, b = r.state_span
        ev[i, a:b] = True
        q_pre[i] = _span_mean(h[i], r.field_span)
        q_end[i] = h[i, r.answer_pos]
        opt_mask[i, : r.n_options] = True
        if r.layout == "ordered":
            e_rows.append(torch.stack([_span_mean(h[i], sp) for sp in r.option_spans]))
        else:
            e_rows.append(torch.zeros((r.n_options, d)))
            for k, seq in enumerate(r.option_seqs):
                sym_seqs.append(seq)
                sym_index.append((i, k))
    e = torch.zeros((bsz, kmax, d))
    for i, er in enumerate(e_rows):
        e[i, : er.shape[0]] = er
    if sym_seqs:
        e = e.clone()
        oid, om = pad_batch(sym_seqs, pad_id)
        he = backbone.features(oid, om).float()
        denom = om.sum(-1, keepdim=True).clamp_min(1)
        pooled = (he * om.unsqueeze(-1)).sum(1) / denom
        idx_i = torch.tensor([x for x, _ in sym_index])
        idx_k = torch.tensor([y for _, y in sym_index])
        e = e.index_put((idx_i, idx_k), pooled)
    fork = None
    if need_fork:
        seqs, where = [], []
        for i, r in enumerate(rows):
            for k, cont in enumerate(r.continuations):
                seqs.append(r.ids + cont)
                where.append((i, k))
        outs = []
        for s in range(0, len(seqs), fork_chunk):
            chunk = seqs[s : s + fork_chunk]
            fid, fm = pad_batch(chunk, pad_id)
            hf = backbone.features(fid, fm).float()
            last = fm.sum(-1) - 1
            outs.append(hf[torch.arange(len(chunk)), last])
        flat = torch.cat(outs)
        fork = torch.zeros((bsz, kmax, d)).index_put(
            (torch.tensor([x for x, _ in where]), torch.tensor([y for _, y in where])), flat
        )
    return HeadInput(h=h, token_mask=mask, evidence_mask=ev, q_pre=q_pre, q_end=q_end, e=e,
                     option_mask=opt_mask, fork=fork)

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
              prefix + continuation_k. ``backbone.fork_mode`` picks recompute (K full rows)
              or cache (one prefix pass per row + K state copies extended by the continuation).
* ``ks``      number of options per row as python ints (no device sync needed to read them).

Every tensor is created on the backbone's device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ks: list[int] = field(default_factory=list)

    @property
    def device(self) -> torch.device:
        return self.h.device

    @property
    def n_options(self) -> torch.Tensor:
        return self.option_mask.sum(-1)

    @property
    def evidence_len(self) -> torch.Tensor:
        return self.evidence_mask.sum(-1)

    def permute_options(self, perm: torch.Tensor) -> "HeadInput":
        """Same input with the option axis permuted by ``perm`` (shape [K])."""
        perm = perm.to(self.device)
        return HeadInput(
            h=self.h, token_mask=self.token_mask, evidence_mask=self.evidence_mask, q_pre=self.q_pre,
            q_end=self.q_end, e=self.e[:, perm], option_mask=self.option_mask[:, perm],
            fork=None if self.fork is None else self.fork[:, perm], ks=list(self.ks),
        )


def _span_mean(h: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    a, b = span
    return h[a:b].mean(0)


def _scatter_rows(base: torch.Tensor, where: list[tuple[int, int]], values: torch.Tensor) -> torch.Tensor:
    dev = base.device
    ii = torch.tensor([x for x, _ in where], dtype=torch.long, device=dev)
    kk = torch.tensor([y for _, y in where], dtype=torch.long, device=dev)
    return base.index_put((ii, kk), values)


def build_head_input(backbone: HiddenStateBackbone, rows: Sequence[LayoutRow], need_fork: bool = False,
                     pad_id: int = 0, fork_chunk: int = 64) -> HeadInput:
    dev = backbone.device
    cache_fork = need_fork and backbone.fork_mode == "cache"
    ids, mask = pad_batch([r.ids for r in rows], pad_id, device=dev)
    fork_rows: list[torch.Tensor] = []
    if cache_fork:  # per-row prefix pass that also yields the K forked features
        per = [backbone.fork_features(r.ids, r.continuations) for r in rows]
        hs = []
        for hp, fk in per:
            hs.append(torch.nn.functional.pad(hp, (0, 0, 0, ids.shape[1] - hp.shape[0])))
            fork_rows.append(fk)
        h = torch.stack(hs).float()
    else:
        h = backbone.features(ids, mask).float()
    bsz, t, d = h.shape
    ks = [r.n_options for r in rows]
    kmax = max(ks)
    ev = torch.zeros((bsz, t), dtype=torch.bool, device=dev)
    opt_mask = torch.zeros((bsz, kmax), dtype=torch.bool, device=dev)
    q_pre_l, q_end_l, e_rows = [], [], []
    sym_seqs: list[list[int]] = []
    sym_index: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        a, b = r.state_span
        ev[i, a:b] = True
        opt_mask[i, : r.n_options] = True
        q_pre_l.append(_span_mean(h[i], r.field_span))
        q_end_l.append(h[i, r.answer_pos])
        if r.layout == "ordered":
            er = torch.stack([_span_mean(h[i], sp) for sp in r.option_spans])
        else:
            er = h.new_zeros((r.n_options, d))
            for k, seq in enumerate(r.option_seqs):
                sym_seqs.append(seq)
                sym_index.append((i, k))
        e_rows.append(torch.nn.functional.pad(er, (0, 0, 0, kmax - er.shape[0])))
    q_pre, q_end, e = torch.stack(q_pre_l), torch.stack(q_end_l), torch.stack(e_rows)
    if sym_seqs:
        oid, om = pad_batch(sym_seqs, pad_id, device=dev)
        he = backbone.features(oid, om).float()
        denom = om.sum(-1, keepdim=True).clamp_min(1)
        pooled = (he * om.unsqueeze(-1)).sum(1) / denom
        e = _scatter_rows(e, sym_index, pooled)
    fork = None
    if cache_fork:
        fork = torch.stack([torch.nn.functional.pad(f.float(), (0, 0, 0, kmax - f.shape[0])) for f in fork_rows])
    elif need_fork:
        seqs, where = [], []
        for i, r in enumerate(rows):
            for k, cont in enumerate(r.continuations):
                seqs.append(r.ids + cont)
                where.append((i, k))
        outs = []
        for s in range(0, len(seqs), fork_chunk):
            chunk = seqs[s : s + fork_chunk]
            fid, fm = pad_batch(chunk, pad_id, device=dev)
            hf = backbone.features(fid, fm).float()
            last = fm.sum(-1) - 1
            outs.append(hf[torch.arange(len(chunk), device=dev), last])
        fork = _scatter_rows(h.new_zeros((bsz, kmax, d)), where, torch.cat(outs))
    return HeadInput(h=h, token_mask=mask, evidence_mask=ev, q_pre=q_pre, q_end=q_end, e=e,
                     option_mask=opt_mask, fork=fork, ks=ks)

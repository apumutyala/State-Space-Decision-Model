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
    #: joint_record head only (``heads.model.DecisionModel`` fills them when ``head.needs_record``): lexical option
    #: vectors [B, K, D] (frozen input-embedding mean over the option's text) and the record groups (row indices of
    #: each request, in schema order). None for every other head.
    lex: torch.Tensor | None = None
    groups: list[list[int]] | None = None

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
            lex=None if self.lex is None else self.lex[:, perm], groups=self.groups,
        )


def _span_mean(h: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    a, b = span
    return h[a:b].mean(0)


def _scatter_rows(base: torch.Tensor, where: list[tuple[int, int]], values: torch.Tensor) -> torch.Tensor:
    dev = base.device
    ii = torch.tensor([x for x, _ in where], dtype=torch.long, device=dev)
    kk = torch.tensor([y for _, y in where], dtype=torch.long, device=dev)
    return base.index_put((ii, kk), values)


def _row_trust(backbone, row: LayoutRow) -> list[bool]:
    from controlsafe.trust_gate import row_trust

    return row_trust(row, backbone.trust_policy)


def _cont_trust(backbone, row: LayoutRow, k: int) -> list[bool]:
    from controlsafe.trust_gate import continuation_trust

    return continuation_trust(row, k, backbone.trust_policy)


def _opt_trust(backbone, row: LayoutRow, k: int) -> list[bool]:
    from controlsafe.trust_gate import option_seq_trust

    return option_seq_trust(row, k, backbone.trust_policy)


def build_head_input(backbone: HiddenStateBackbone, rows: Sequence[LayoutRow], need_fork: bool = False,
                     pad_id: int = 0, fork_chunk: int = 64, shared_prefix: bool = False) -> HeadInput:
    """``shared_prefix=True`` (evaluation only, no grad): rows of the same request read their common schema + state
    prefix once (``heads.prefix_features``); equal to the plain path up to float rounding (tests/test_perf_paths.py)."""
    from . import prefix_features as PF

    dev = backbone.device
    cache_fork = need_fork and backbone.fork_mode == "cache"
    ids, mask = pad_batch([r.ids for r in rows], pad_id, device=dev)
    # controlsafe trust gate (G15): a backbone with ``wants_trust`` gets per-token trust labels from the rows
    gated = getattr(backbone, "wants_trust", False)
    trusts = [_row_trust(backbone, r) for r in rows] if gated else None
    tkw = {"trust": pad_batch([[int(x) for x in t] for t in trusts], 0, device=dev)[0].bool()} if gated else {}
    fork_rows: list[torch.Tensor] = []
    pre_fork = None
    # The shared-prefix path runs the wrapped model directly on cached state; a trust-gated backbone computes a
    # protected stream + bridges per call and needs per-token trust labels, so it NEVER takes this path: gated rows
    # go through the labelled path below exactly as before (PF.supported also refuses ``wants_trust``).
    use_prefix = (shared_prefix and not gated and not torch.is_grad_enabled() and PF.supported(backbone))
    if use_prefix:
        h, pre_fork = PF.shared_prefix_features(backbone, rows, need_fork, pad_id)
    elif (cache_fork and not gated and len(rows) > 1 and getattr(backbone, "batch_rows", False)
          and backbone.can_fork_rows()):  # TrainConfig.row_batch: all rows in one prefix call + one continuation call
        h, fork_rows = backbone.fork_features_rows([r.ids for r in rows], [r.continuations for r in rows])
        h = h.float()
    elif cache_fork:  # per-row prefix pass that also yields the K forked features
        per = [backbone.fork_features(r.ids, r.continuations,
                                      **({"trust": t, "cont_trust": [_cont_trust(backbone, r, k) for k in
                                                                     range(len(r.continuations))]} if gated else {}))
               for r, t in zip(rows, trusts or [None] * len(rows))]
        hs = []
        for hp, fk in per:
            hs.append(torch.nn.functional.pad(hp, (0, 0, 0, ids.shape[1] - hp.shape[0])))
            fork_rows.append(fk)
        h = torch.stack(hs).float()
    else:
        h = backbone.features(ids, mask, **tkw).float()
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
        okw = {}
        if gated:
            ot = [_opt_trust(backbone, rows[i], k) for i, k in sym_index]
            okw = {"trust": pad_batch([[int(x) for x in t] for t in ot], 0, device=dev)[0].bool()}
        he = backbone.features(oid, om, **okw).float()
        denom = om.sum(-1, keepdim=True).clamp_min(1)
        pooled = (he * om.unsqueeze(-1)).sum(1) / denom
        e = _scatter_rows(e, sym_index, pooled)
    fork = None
    if use_prefix:
        fork = pre_fork
    elif cache_fork:
        fork = torch.stack([torch.nn.functional.pad(f.float(), (0, 0, 0, kmax - f.shape[0])) for f in fork_rows])
    elif need_fork:
        seqs, where, strust = [], [], []
        for i, r in enumerate(rows):
            for k, cont in enumerate(r.continuations):
                seqs.append(r.ids + cont)
                where.append((i, k))
                if gated:  # continuations labelled by the same policy as the prefix (option ids are data)
                    strust.append([int(x) for x in trusts[i]] + [int(x) for x in _cont_trust(backbone, r, k)])
        outs = []
        for s in range(0, len(seqs), fork_chunk):
            chunk = seqs[s : s + fork_chunk]
            fid, fm = pad_batch(chunk, pad_id, device=dev)
            ckw = {"trust": pad_batch(strust[s : s + fork_chunk], 0, device=dev)[0].bool()} if gated else {}
            hf = backbone.features(fid, fm, **ckw).float()
            last = fm.sum(-1) - 1
            outs.append(hf[torch.arange(len(chunk), device=dev), last])
        fork = _scatter_rows(h.new_zeros((bsz, kmax, d)), where, torch.cat(outs))
    return HeadInput(h=h, token_mask=mask, evidence_mask=ev, q_pre=q_pre, q_end=q_end, e=e,
                     option_mask=opt_mask, fork=fork, ks=ks)

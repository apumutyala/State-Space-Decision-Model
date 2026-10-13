"""Fast EVALUATION readout for the symmetric fork layouts (``sym-fork*``; docs/SYM_FAST.md). No grad, serving only.

Why: the served sym-fork path (``heads.prefix_features`` with ``shared_prefix=True``) makes THREE backbone calls for a
one-question JevBench decision -- the prefix up to ``common_prefix_len`` (capped one token short), a one-token
"suffix" call, then the K forked continuations -- plus two deep copies of the cache. The g1v2 shim makes ONE call. At
small token counts a Qwen3.5-4B forward is per-call bound, so the 3 calls are the ~2.9x (0.197 s vs 0.067 s p50).

Two execution modes (``backbone.sym_exec``, set by ``DecisionModelBackend(sym_exec=...)``; None = the old path):

``fork2``   2 calls per request, token-neutral. One prefix call, ONE batched continuation call.
            * one-question request: exactly ``HiddenStateBackbone.fork_features(row.ids, conts)`` -- the cache fork the
              model TRAINS with (prefix = the whole row in one call, K state copies, one continuation call). Its fork
              features equal the ``shared_prefix=False`` cache-fork path bit for bit (same calls, same shapes), so
              serving = training numerics (the old served path differed from training by float rounding: its last
              prefix token ran through the GDN single-token decode kernel);
            * Q > 1 questions: one call over the common prefix (state + schema), then ONE call over every
              (question suffix + option continuation) of every question, from sum(K_q) copies of the prefix state,
              right-padded (causal: positions <= last never read padding). Float-close to the old path (it splits the
              same tokens over different calls), never bitwise.
``packed1`` 1 call per request (per ``fork_chunk`` sequences), NOT token-neutral: every option runs as the full
            sequence ``row.ids + cont`` in ONE right-padded batch (the recompute fork packed into one call); the prefix
            features are read from the batch row of the question's first option (content order). Its fork features
            equal the ``fork_mode="recompute"`` path bit for bit (same batch, same order, same chunks). Costs K x the
            prefix tokens: wins only if a call is overhead bound at that size (measure: scripts/sym_fast_timing.py).

Invariance (docs/PROOF_INVARIANCE.md): the batch layout is CONTENT-determined. Within a row the options run in the
order of their continuation token ids (ties by axis index: equal continuations are equal content, equal values), and
outputs are un-permuted back onto the row's axis. A sym row of a permuted / relabeled request is the same row (except
``option_ids``), so every call here sees bitwise identical tensors and the served probabilities are bitwise identical.
Even a row whose AXIS was permuted (``build_row(option_order=...)``, no canonicalization helping) gets bitwise
permuted fork features here (the old path: equal up to batch-order float rounding only).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

MODES = ("fork2", "packed1")


def content_order(conts: Sequence[Sequence[int]]) -> list[int]:
    """Batch order of one row's options: by continuation token ids, ties (exact duplicates) by axis index. For a
    choice row built by ``heads.symmetric.build_row_sym`` without ``option_order`` this is the identity (the axis is
    already content-canonical); noul / score rows may be reordered here and are un-permuted after the call."""
    return sorted(range(len(conts)), key=lambda k: (list(conts[k]), k))


def eligible(backbone, rows: Sequence, need_fork: bool) -> bool:
    """The fast path runs only for: a mode set on the backbone, fork features wanted, no autograd (training keeps its
    own path), no controlsafe trust gate (it needs per-token trust labels per call), every row a sym layout with at
    least one option, and a backbone that captures hidden states."""
    from .symmetric import SYM_LAYOUTS

    mode = getattr(backbone, "sym_exec", None)
    if mode is None:
        return False
    if mode not in MODES:
        raise ValueError(f"sym_exec must be one of {MODES} or None, got {mode!r}")
    return (need_fork and not torch.is_grad_enabled() and not getattr(backbone, "wants_trust", False)
            and len(rows) > 0 and all(r.layout in SYM_LAYOUTS and len(r.continuations) > 0 for r in rows))


def _pf_supported(backbone) -> bool:
    from .prefix_features import supported

    return supported(backbone)


def _groups(rows: Sequence) -> list[list[int]]:
    from .prefix_features import _group_key

    g: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        g.setdefault(_group_key(r), []).append(i)
    return list(g.values())


def _flat(rows: Sequence, idx: Sequence[int]) -> list[tuple[int, int]]:
    """(row, axis index) of every option of the rows ``idx``, rows in request order, options in content order."""
    return [(i, k) for i in idx for k in content_order(rows[i].continuations)]


@torch.no_grad()
def _fork2_group(backbone, rows: Sequence, idx: list[int], per_h: list, per_fork: list, max_copies: int) -> None:
    from .prefix_features import _clone, _run, supported

    dev = backbone.device
    if len(idx) == 1 or not supported(backbone):
        # one question (every JevBench decision), or a backbone the shared-prefix runner does not drive: the training
        # cache fork, one prefix call + one continuation call per row
        for i in idx:
            r = rows[i]
            order = content_order(r.continuations)
            hp, fk = backbone.fork_features(r.ids, [r.continuations[k] for k in order])
            per_h[i] = hp
            out = torch.empty_like(fk)
            out[torch.tensor(order, device=fk.device)] = fk
            per_fork[i] = out
        return
    from decide.backends.state_fork import common_prefix_len, expand

    L = common_prefix_len([rows[i].ids for i in idx])
    hp, sp = _run(backbone, torch.tensor([rows[idx[0]].ids[:L]], dtype=torch.long, device=dev), None)
    flat = _flat(rows, idx)
    seqs = [list(rows[i].ids[L:]) + list(rows[i].continuations[k]) for i, k in flat]
    feats: dict[tuple[int, int], torch.Tensor] = {}
    suffix: dict[int, torch.Tensor] = {}
    step = max(1, int(max_copies))
    for s in range(0, len(seqs), step):  # chunks of the content-ordered list: the chunking is content-determined too
        part, where = seqs[s: s + step], flat[s: s + step]
        st = sp if len(seqs) <= step else _clone(sp)
        expand(st, len(part))
        w = max(len(x) for x in part)
        cid = torch.full((len(part), w), backbone.pad_id, dtype=torch.long, device=dev)
        for j, x in enumerate(part):
            cid[j, : len(x)] = torch.tensor(x, dtype=torch.long, device=dev)
        hc, _ = _run(backbone, cid, st)  # right padding: only positions <= last are read
        for j, (i, k) in enumerate(where):
            feats[(i, k)] = hc[j, len(part[j]) - 1]
            if i not in suffix:  # the question's suffix features: from its first option in content order
                suffix[i] = hc[j, : len(rows[i].ids) - L]
    for i in idx:
        per_h[i] = torch.cat([hp[0], suffix[i]], 0)
        per_fork[i] = torch.stack([feats[(i, k)] for k in range(len(rows[i].continuations))])


@torch.no_grad()
def _packed1(backbone, rows: Sequence, per_h: list, per_fork: list, fork_chunk: int) -> None:
    from .backbone import pad_batch

    dev = backbone.device
    flat = _flat(rows, range(len(rows)))
    seqs = [list(rows[i].ids) + list(rows[i].continuations[k]) for i, k in flat]
    feats: dict[tuple[int, int], torch.Tensor] = {}
    step = max(1, int(fork_chunk))
    for s in range(0, len(seqs), step):  # as build_head_input's recompute fork: chunks of fork_chunk sequences
        chunk, where = seqs[s: s + step], flat[s: s + step]
        fid, fm = pad_batch(chunk, backbone.pad_id, device=dev)
        hf = backbone.features(fid, fm).float()
        last = fm.sum(-1) - 1
        f = hf[torch.arange(len(chunk), device=dev), last]
        for j, (i, k) in enumerate(where):
            feats[(i, k)] = f[j]
            if per_h[i] is None:
                per_h[i] = hf[j, : len(rows[i].ids)]
    for i, r in enumerate(rows):
        per_fork[i] = torch.stack([feats[(i, k)] for k in range(len(r.continuations))])


@torch.no_grad()
def sym_fast_features(backbone, rows: Sequence, mode: str | None = None, max_copies: int = 256,
                      fork_chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (h [B, Tmax, D] float, right-padded and zero on padding; fork [B, Kmax, D] float, zero on padded options):
    the contract of ``heads.prefix_features.shared_prefix_features(..., need_fork=True)``."""
    mode = mode or getattr(backbone, "sym_exec", None)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    B = len(rows)
    per_h: list[torch.Tensor | None] = [None] * B
    per_fork: list[torch.Tensor | None] = [None] * B
    if mode == "fork2":
        for idx in _groups(rows):
            _fork2_group(backbone, rows, idx, per_h, per_fork, max_copies)
    else:
        _packed1(backbone, rows, per_h, per_fork, fork_chunk)
    T = max(len(r.ids) for r in rows)
    kmax = max(len(r.continuations) for r in rows)
    h = torch.stack([torch.nn.functional.pad(x.float(), (0, 0, 0, T - x.shape[0])) for x in per_h])
    fork = torch.stack([torch.nn.functional.pad(f.float(), (0, 0, 0, kmax - f.shape[0])) for f in per_fork])
    return h, fork


def predicted_calls(rows: Sequence, mode: str | None, max_copies: int = 256, fork_chunk: int = 64,
                    backbone: Any = None) -> int:
    """Backbone forward calls one ``sym_fast_features`` (or, for ``mode=None``, the old shared-prefix path) makes on
    these rows (the latency model of docs/SYM_FAST.md; tests/test_sym_fast.py checks it against a call counter)."""
    groups = _groups(rows)
    if mode == "packed1":
        n = sum(len(r.continuations) for r in rows)
        return -(-n // max(1, fork_chunk))
    if mode == "fork2":
        out = 0
        for idx in groups:
            if len(idx) == 1 or (backbone is not None and not _pf_supported(backbone)):
                out += 2 * len(idx)
            else:
                n = sum(len(rows[i].continuations) for i in idx)
                out += 1 + -(-n // max(1, max_copies))
        return out
    from decide.backends.state_fork import common_prefix_len

    out = 0  # old path (heads.prefix_features, need_fork): prefix + one call per suffix-length group + one per row
    for idx in groups:
        L = common_prefix_len([rows[i].ids for i in idx])
        out += 1 + len({len(rows[i].ids) - L for i in idx}) + len(idx)
    return out

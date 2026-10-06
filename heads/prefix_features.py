"""Shared-prefix features for decision-head EVALUATION (no grad): the rows of one request (one row per question) share
the token prefix ``template head + schema of all fields + state``; read it once and continue the cached state over
each row's own suffix, instead of re-reading the whole prompt per question.

Measured volumes on the m1 DEV subset (``jobs/common/readout_volume_rwkv7.json``): one full row per question is 55.1M
tokens (~171 min at the 5.36k tok/s fitted for RWKV7-13.3B), the shared prefix + suffixes ~5M. Used by
``heads.decide_backend.DecisionModelBackend`` (the m2 child's Index / steer arm) and the eval card.

Exact up to float rounding: features of ``P + S`` equal ``features(P)`` concatenated with the features of ``S``
continued from P's state (recurrent state / KV cache), for causal backbones. Rows are grouped by EQUAL suffix length,
so no padding enters any state that is read later (the fork continues from each row's own end state).
Supported: ``capture == "hidden_states"`` backbones whose model takes ``state=`` (bundled RWKV-7) or
``past_key_values`` (Qwen3.5 hybrid, all-GDN Q9-AF). Anything else uses the plain path.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from decide.backends.state_fork import _deepcopy_cache, common_prefix_len, expand, select_row


def supported(backbone) -> bool:
    from .backbone import _accepts

    m = backbone.model
    return (backbone.capture == "hidden_states" and not _accepts(m.forward, "cache_params")
            and (_accepts(m.forward, "state") or _accepts(m.forward, "past_key_values")))


def _run(backbone, ids: torch.Tensor, state: Any):
    """-> (features [B, t, D], state after ids)."""
    from .backbone import _accepts, _linear_safe_cache

    m = backbone.model
    kw: dict[str, Any] = {"use_cache": True, "output_hidden_states": True}
    if _accepts(m.forward, "state"):
        out = m(input_ids=ids, state=state, **kw)
        return backbone._pick(out), out.state
    if state is None:
        state = _linear_safe_cache(m.config)
        state.seen = 0
    out = m(input_ids=ids, past_key_values=state, **kw)
    cache = out.past_key_values
    cache.seen = getattr(state, "seen", 0) + ids.shape[1]  # positions for an all-linear (Q9-AF) cache
    return backbone._pick(out), cache


def _clone(st):
    c = _deepcopy_cache(st, [getattr(st, "config", None)])
    if hasattr(st, "seen"):
        c.seen = st.seen
    return c


def _group_key(r) -> tuple:
    return tuple(r.ids[: r.state_span[1]])  # identical schema + state = same request


@torch.no_grad()
def shared_prefix_features(backbone, rows: Sequence, need_fork: bool, pad_id: int = 0):
    """-> (h [B, Tmax, D] float, right-padded like pad_batch; fork [B, Kmax, D] float or None)."""
    dev = backbone.device
    B = len(rows)
    per_h: list[torch.Tensor | None] = [None] * B
    per_fork: list[torch.Tensor | None] = [None] * B
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_group_key(r), []).append(i)
    single = [idx[0] for idx in groups.values() if len(idx) == 1 and not need_fork]
    if single:  # nothing to share: one batched plain forward, as before
        from .backbone import pad_batch

        sid, sm = pad_batch([rows[i].ids for i in single], pad_id, device=dev)
        hf = backbone.features(sid, sm)
        for b, i in enumerate(single):
            per_h[i] = hf[b, : len(rows[i].ids)]
    for idx in groups.values():
        if len(idx) == 1 and not need_fork:
            continue
        L = common_prefix_len([rows[i].ids for i in idx])
        hp, sp = _run(backbone, torch.tensor([rows[idx[0]].ids[:L]], dtype=torch.long, device=dev), None)
        by_len: dict[int, list[int]] = {}
        for i in idx:
            by_len.setdefault(len(rows[i].ids) - L, []).append(i)
        for _, members in sorted(by_len.items()):
            st = _clone(sp)
            if len(members) > 1:
                expand(st, len(members))
            suf = torch.tensor([rows[i].ids[L:] for i in members], dtype=torch.long, device=dev)
            hs, ss = _run(backbone, suf, st)
            for b, i in enumerate(members):
                per_h[i] = torch.cat([hp[0], hs[b]], 0)
                if need_fork:
                    conts = rows[i].continuations
                    k = len(conts)
                    sb = _clone(ss)
                    if len(members) > 1:
                        select_row(sb, b)
                    if k > 1:
                        expand(sb, k)
                    w = max(len(c) for c in conts)
                    cid = torch.full((k, w), pad_id, dtype=torch.long, device=dev)
                    for j, c in enumerate(conts):
                        cid[j, : len(c)] = torch.tensor(c, dtype=torch.long, device=dev)
                    hc, _ = _run(backbone, cid, sb)  # right padding: only positions <= last are read
                    last = torch.tensor([len(c) - 1 for c in conts], device=dev)
                    per_fork[i] = hc[torch.arange(k, device=dev), last]
    T = max(len(r.ids) for r in rows)
    h = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[0])) for x in per_h]).float()
    fork = None
    if need_fork:
        kmax = max(r.n_options for r in rows)
        fork = torch.stack([torch.nn.functional.pad(f.float(), (0, 0, 0, kmax - f.shape[0])) for f in per_fork])
    return h, fork

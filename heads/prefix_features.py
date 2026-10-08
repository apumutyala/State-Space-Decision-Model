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
    if getattr(backbone, "wants_trust", False):  # controlsafe trust gate: two streams + labels, never this path
        return False
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
    return _assemble(rows, per_h, per_fork, need_fork)


def _assemble(rows, per_h, per_fork, need_fork: bool):
    T = max(len(r.ids) for r in rows)
    h = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, T - x.shape[0])) for x in per_h]).float()
    fork = None
    if need_fork:
        kmax = max(r.n_options for r in rows)
        fork = torch.stack([torch.nn.functional.pad(f.float(), (0, 0, 0, kmax - f.shape[0])) for f in per_fork])
    return h, fork


#: packed tails per call (``packed_prefix_features``): rows x padded width. Each row holds its own copy of the prefix
#: state (RWKV7-13.3B: ~66 MB fp32), and the backbone returns every layer's hidden states for the call
#: (62 x rows x width x 4,096 bf16 = 4.2 GB at 8,192 positions), so both are bounded.
PACK_MAX_ROWS = 64
PACK_MAX_TOKENS = 8192


@torch.no_grad()
def packed_prefix_features(backbone, rows: Sequence, need_fork: bool, pad_id: int = 0,
                           max_rows: int = PACK_MAX_ROWS, max_tokens: int = PACK_MAX_TOKENS):
    """Same contract and result as ``shared_prefix_features`` (up to float rounding), in fewer backbone calls.

    Per request group: one call over the common prefix P, then EVERY tail in packed calls from copies of P's state,
    where a tail is ``suffix_q + continuation_k`` for each (question q, option k) (or ``suffix_q`` alone without the
    fork). Right padding is exact: the backbone is causal and only positions <= a tail's last real token are read;
    the state after a pack is discarded. The suffix features of row q are read from its first tail's positions
    ``[0, |suffix_q|)`` (identical in every tail of q: causal), its fork feature k at ``|suffix_q| + len(cont_k) - 1``.
    The stock readout scores this way (``decide.backends.state_fork``); the grouped path made 1 + (distinct suffix
    lengths) + (questions) calls per request instead."""
    from decide.backends.state_fork import expand

    dev = backbone.device
    B = len(rows)
    per_h: list[torch.Tensor | None] = [None] * B
    per_fork: list[torch.Tensor | None] = [None] * B
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_group_key(r), []).append(i)
    for idx in groups.values():
        L = common_prefix_len([rows[i].ids for i in idx])  # capped: every row keeps >= 1 own token
        hp, sp = _run(backbone, torch.tensor([rows[idx[0]].ids[:L]], dtype=torch.long, device=dev), None)
        tails = []  # (row i, option k or -1, tail ids, suffix length)
        for i in idx:
            suf = list(rows[i].ids[L:])
            if need_fork:
                for k, c in enumerate(rows[i].continuations):
                    tails.append((i, k, suf + list(c), len(suf)))
            else:
                tails.append((i, -1, suf, len(suf)))
        order = sorted(range(len(tails)), key=lambda t: (-len(tails[t][2]), t))  # deterministic, longest first
        packs: list[list[int]] = []
        cur: list[int] = []
        for t in order:
            if cur and (len(cur) >= max_rows or len(tails[cur[0]][2]) * (len(cur) + 1) > max_tokens):
                packs.append(cur)
                cur = []
            cur.append(t)
        if cur:
            packs.append(cur)
        fork_parts: dict[int, dict[int, torch.Tensor]] = {}
        for pack in packs:
            W = len(tails[pack[0]][2])
            ids = torch.full((len(pack), W), pad_id, dtype=torch.long, device=dev)
            for b, t in enumerate(pack):
                ids[b, : len(tails[t][2])] = torch.tensor(tails[t][2], dtype=torch.long, device=dev)
            st = _clone(sp)
            if len(pack) > 1:
                expand(st, len(pack))
            hc, _ = _run(backbone, ids, st)
            for b, t in enumerate(pack):
                i, k, tid, ls = tails[t]
                if per_h[i] is None:
                    per_h[i] = torch.cat([hp[0], hc[b, :ls]], 0)
                if k >= 0:
                    fork_parts.setdefault(i, {})[k] = hc[b, len(tid) - 1]
        if need_fork:
            for i in idx:
                parts = fork_parts[i]
                per_fork[i] = torch.stack([parts[k] for k in range(len(rows[i].continuations))])
    return _assemble(rows, per_h, per_fork, need_fork)


# ---------------------------------------------------------------------------------------------------------------------
# tree fork (``heads.infer_opt`` ``tree_fork``): prefix -> question -> option, each level read ONCE
# ---------------------------------------------------------------------------------------------------------------------
def tree_supported(backbone) -> bool:
    """The tree needs each row's state at the end of its OWN suffix out of a right-padded call: a recurrent model that
    takes ``state=`` and a right-padding ``attention_mask`` and holds the state exactly on padding (the vendored
    RWKV-7: decay 1, k = v = kk = 0 on pads, token shift from the last real token; ``HiddenStateBackbone.can_fork_rows``).
    Anything else falls back to the packed path (same result, more tokens)."""
    return supported(backbone) and backbone.can_fork_rows()


def pack_order(lengths: Sequence[int], max_rows: int = PACK_MAX_ROWS, max_tokens: int = PACK_MAX_TOKENS):
    """Indices into ``lengths`` cut into right-padded packs, longest first (stable): a pack holds <= ``max_rows`` rows
    and (rows x its longest length) <= ``max_tokens`` (a longer single row runs alone). The packed path's rule."""
    order = sorted(range(len(lengths)), key=lambda t: (-lengths[t], t))
    packs: list[list[int]] = []
    cur: list[int] = []
    for t in order:
        if cur and (len(cur) >= max_rows or lengths[cur[0]] * (len(cur) + 1) > max_tokens):
            packs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        packs.append(cur)
    return packs


def tree_plan(rows_ids: Sequence[Sequence[int]], conts: Sequence[Sequence[Sequence[int]]] | None,
              max_rows: int = PACK_MAX_ROWS, max_tokens: int = PACK_MAX_TOKENS) -> dict:
    """Token accounting of ``tree_prefix_features`` for ONE request group (no model): ``tokens`` = real tokens read,
    ``positions`` = tokens incl. right padding (what the device computes), ``calls`` = backbone calls. ``conts=None``
    = no fork. Same packing as the executor (``pack_order``)."""
    L = common_prefix_len([list(r) for r in rows_ids])
    suf = [len(r) - L for r in rows_ids]
    tok, pos, calls = L, L, 1
    for pack in pack_order(suf, max_rows, max_tokens):
        tok += sum(suf[i] for i in pack)
        pos += suf[pack[0]] * len(pack)
        calls += 1
        if conts is not None:
            flat = [len(c) for i in pack for c in conts[i]]
            for cp in pack_order(flat, max_rows, max_tokens):
                tok += sum(flat[j] for j in cp)
                pos += flat[cp[0]] * len(cp)
                calls += 1
    return {"tokens": tok, "positions": pos, "calls": calls, "prefix": L}


def packed_plan(rows_ids: Sequence[Sequence[int]], conts: Sequence[Sequence[Sequence[int]]] | None,
                max_rows: int = PACK_MAX_ROWS, max_tokens: int = PACK_MAX_TOKENS) -> dict:
    """Token accounting of ``packed_prefix_features`` for ONE request group (no model): every (question, option)
    tail ``suffix_q + continuation_k`` re-reads the question suffix once per option."""
    L = common_prefix_len([list(r) for r in rows_ids])
    tails = []
    for r, cs in zip(rows_ids, conts if conts is not None else [None] * len(rows_ids)):
        s = len(r) - L
        tails += [s + len(c) for c in cs] if cs is not None else [s]
    tok, pos, calls = L + sum(tails), L, 1
    for pack in pack_order(tails, max_rows, max_tokens):
        pos += tails[pack[0]] * len(pack)
        calls += 1
    return {"tokens": tok, "positions": pos, "calls": calls, "prefix": L}


def _run_masked(backbone, ids: torch.Tensor, mask: torch.Tensor, state: Any):
    """-> (features [B, t, D], state after each row's LAST REAL token) for a right-padded call (vendored RWKV-7)."""
    out = backbone.model(input_ids=ids, attention_mask=mask.long(), state=state, use_cache=True,
                         output_hidden_states=True)
    if out.state is None:
        raise RuntimeError("backbone returned no state for the tree fork")
    return backbone._pick(out), out.state


def _select(st, idx: torch.Tensor):
    """A NEW cache holding batch rows ``idx`` of ``st`` (repeats allowed); ``st`` is left untouched."""
    from decide.backends.state_fork import cache_batch_map

    c = _clone(st)
    cache_batch_map(c, lambda t: t.index_select(0, idx.to(t.device)))
    return c


@torch.no_grad()
def tree_prefix_features(backbone, rows: Sequence, need_fork: bool, pad_id: int = 0,
                         max_rows: int = PACK_MAX_ROWS, max_tokens: int = PACK_MAX_TOKENS):
    """Same contract and result as ``shared_prefix_features`` / ``packed_prefix_features`` (up to float rounding),
    reading every token of the prefix tree once. Per request group:

    1. one call over the common prefix P (template head + schema + state);
    2. every question suffix ``S_q`` (the re-read, ending at the answer position) in right-padded, MASKED packs from
       copies of P's state. The masked recurrence leaves each row's state at the end of its OWN suffix, so one call
       yields both the suffix features and the per-question states (the grouped path needed one call per distinct
       suffix length; the packed path re-read ``S_q`` once per option);
    3. every option continuation ``C_k`` of the pack's questions, from copies of its question's state
       (``index_select`` on the batch axis), right-padded (causal; only positions <= a continuation's last real token
       are read; the state after the call is dropped).

    Tokens read: |P| + sum_q |S_q| + sum_{q,k} |C_k| instead of the packed path's |P| + sum_{q,k} (|S_q| + |C_k|)
    (``tree_plan`` / ``packed_plan`` count both without a model). Backbones without the masked-state property
    (``tree_supported``) take ``packed_prefix_features``."""
    from .backbone import pad_batch

    if not tree_supported(backbone):
        return packed_prefix_features(backbone, rows, need_fork, pad_id, max_rows, max_tokens)
    dev = backbone.device
    B = len(rows)
    per_h: list[torch.Tensor | None] = [None] * B
    per_fork: list[torch.Tensor | None] = [None] * B
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_group_key(r), []).append(i)
    for idx in groups.values():
        L = common_prefix_len([rows[i].ids for i in idx])  # capped: every row keeps >= 1 own token
        hp, sp = _run(backbone, torch.tensor([rows[idx[0]].ids[:L]], dtype=torch.long, device=dev), None)
        sufs = [list(rows[i].ids[L:]) for i in idx]
        for pack in pack_order([len(s) for s in sufs], max_rows, max_tokens):
            sid, sm = pad_batch([sufs[j] for j in pack], pad_id, device=dev)
            hs, ss = _run_masked(backbone, sid, sm, _select(sp, torch.zeros(len(pack), dtype=torch.long)))
            for b, j in enumerate(pack):
                per_h[idx[j]] = torch.cat([hp[0], hs[b, : len(sufs[j])]], 0)
            if not need_fork:
                continue
            flat = [(b, k, list(c)) for b, j in enumerate(pack) for k, c in enumerate(rows[idx[j]].continuations)]
            parts: dict[int, dict[int, torch.Tensor]] = {}
            for cp in pack_order([len(f[2]) for f in flat], max_rows, max_tokens):
                cid, _ = pad_batch([flat[t][2] for t in cp], pad_id, device=dev)
                sc = _select(ss, torch.tensor([flat[t][0] for t in cp], dtype=torch.long))
                hc, _ = _run(backbone, cid, sc)  # right padding: only positions <= last are read
                for r_, t in enumerate(cp):
                    b, k, c = flat[t]
                    parts.setdefault(b, {})[k] = hc[r_, len(c) - 1]
            for b, j in enumerate(pack):
                i = idx[j]
                per_fork[i] = torch.stack([parts[b][k] for k in range(len(rows[i].continuations))])
    return _assemble(rows, per_h, per_fork, need_fork)

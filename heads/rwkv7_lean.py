"""``chunked_lean``: the vendored RWKV-7 chunk-parallel WKV (``modeling_rwkv7.rwkv7_chunked``) with fewer kernel launches
and autograd nodes per call, for TRAINING (jobs/cell_step_profile). Same algebra, same compute dtype (fp32), same chunk
size (64), same packed-sequence contract; registered from OUR code through the model's ``RWKV7_WKV_FUNCTIONS`` hook
(the vendored files stay sha-pinned and unmodified).

    from heads.rwkv7_wkv import select_wkv
    select_wkv(model, "chunked_lean")      # default off; the checkpoint's own value is "chunked"

Why: the m2 cell's train step is launch-bound (job-mden7: ~2 s of fixed cost per row, flat in tokens). The vendored
chunk loop issues ~11 forward kernels per 64-token chunk per layer and, in backward, one FULL-SIZE zero-filled
gradient (``select_backward``) per chunk for every tensor it indexes with ``x[:, i]`` -- 8 of them -- plus the adds that
accumulate them: O(chunks^2) bytes and ~3x the forward's launches. Changes, each one exact or within fp32 rounding
(``tests/test_cell_perf.py`` checks outputs, final state and every input gradient against the vendored function):

1. ``x.unbind(1)`` once instead of ``x[:, i]`` per chunk: identical forward values; backward is one ``stack`` instead of
   ``chunks`` zero-filled full-size tensors summed (each element receives exactly one non-zero term either way: exact).
2. The four intra-chunk score products (q~b~, q~k~, r~k~, r~b~) are ONE batched matmul of [q~; r~] with [b~; k~] (a
   2x2 block product) and the two value products (q~k~ V, r~k~ V) one more; the masks are applied as one multiply.
   Same dot products, different GEMM shapes: equal within fp32 rounding (not bitwise on every BLAS).
3. Only the state-dependent terms stay in the serial loop (state product, solve, state update). The output
   ``O = R~ S + tril(R~ K~^T) V - tril(R~ B~^T) U`` is formed for ALL chunks after the loop in one batched matmul.
4. The constant masks are built once per (device, dtype, chunk) and cached.

Forward chunk loop: 7 kernels per chunk (was ~11), no per-chunk selects; backward has no O(chunks^2) zero fills.
"""
from __future__ import annotations

import math

import torch

from third_party.rwkv7_hf import modeling_rwkv7 as M

NAME = "chunked_lean"
_MASKS: dict = {}


def _masks(span: int, device, dtype):
    """[2*span, 2*span] mask for the stacked score product rows (q~, r~) x cols (b~, k~): strict lower for the q~ rows,
    lower-with-diagonal for the r~ rows (exactly the vendored ``tri`` / ``causal``), and the identity."""
    key = (span, str(device), dtype)
    m = _MASKS.get(key)
    if m is None:
        tri = torch.ones(span, span, device=device, dtype=dtype).tril(-1)
        causal = torch.ones(span, span, device=device, dtype=dtype).tril(0)
        stacked = torch.cat([torch.cat([tri, tri], 1), torch.cat([causal, causal], 1)], 0)
        m = (stacked, torch.eye(span, device=device, dtype=dtype))
        _MASKS[key] = m
    return m


def rwkv7_chunked_lean(r, w_log, k, v, kk, a, state, chunk_size: int = 64, compute_dtype=torch.float32):
    """Drop-in for ``modeling_rwkv7.rwkv7_chunked`` (same signature, same returns, same chunk-size guard)."""
    batch, seq_len, num_heads, head_dim = r.shape
    dtype = r.dtype
    r, w_log, k, v, kk, a = (t.to(compute_dtype) for t in (r, w_log, k, v, kk, a))
    state_dtype = state.dtype
    state = state.transpose(-2, -1).to(compute_dtype)
    chunk_size = min(chunk_size, seq_len)
    widest = int(math.log(torch.finfo(compute_dtype).max) / M._INV_SQRT_E)
    if chunk_size > widest:
        raise ValueError(f"chunk_size={chunk_size} overflows {compute_dtype}: the widest finite chunk is {widest}")
    chunks = (seq_len + chunk_size - 1) // chunk_size
    padded = chunks * chunk_size
    if padded != seq_len:
        pad = (0, 0, 0, 0, 0, padded - seq_len)
        r, k, v, kk, a, w_log = (torch.nn.functional.pad(t, pad) for t in (r, k, v, kk, a, w_log))
    # [chunks, batch*heads, span, head_dim], contiguous: every product below is a plain batched matmul and the serial
    # loop indexes nothing (one unbind per tensor)
    span = chunk_size
    bh = batch * num_heads

    def lay(t):
        return t.reshape(batch, chunks, span, num_heads, head_dim).permute(1, 0, 3, 2, 4).reshape(chunks, bh, span,
                                                                                                head_dim)

    rg, kg, vg, kkg, ag, wg = (lay(t) for t in (r, k, v, kk, a, w_log))
    w_c = torch.exp(wg)
    c = torch.cumprod(w_c, dim=2)
    c_prev = c / w_c
    bg = kkg * ag
    b_t, k_t = bg / c, kg / c
    q_t, r_t = kkg * c_prev, rg * c
    mask, eye = _masks(span, r.device, r.dtype)

    # [q~; r~] x [b~; k~]^T in one product: blocks qb | qk / rb | rk, masked (tri for q~ rows, causal for r~ rows)
    scores = torch.matmul(torch.cat([q_t, r_t], 2), torch.cat([b_t, k_t], 2).transpose(-1, -2)) * mask
    qb, rb = scores[..., :span, :span], scores[..., span:, :span]
    lhs = eye + qb
    # (qk V ; rk V) in one product: the k~ columns of both row blocks times V; rows :span = qkv, span: = rkv
    qrkv = torch.matmul(scores[..., span:], vg)
    kv = torch.matmul(k_t.transpose(-1, -2), vg)  # [C, BH, N, V]
    c_last = c[:, :, -1].unsqueeze(-1)  # [C, BH, N, 1]
    b_tt = b_t.transpose(-1, -2).contiguous()  # [C, BH, N, S]

    state = state.reshape(bh, head_dim, head_dim)  # (key, value)
    q_c, qkv_c, lhs_c = q_t.unbind(0), qrkv[:, :, :span].contiguous().unbind(0), lhs.unbind(0)
    kv_c, b_c, cl_c = kv.unbind(0), b_tt.unbind(0), c_last.unbind(0)
    states, us = [], []
    for i in range(chunks):
        states.append(state)
        u = torch.linalg.solve_triangular(lhs_c[i], torch.bmm(q_c[i], state) + qkv_c[i], upper=False,
                                          unitriangular=True)
        us.append(u)
        state = cl_c[i] * (state + kv_c[i] - torch.bmm(b_c[i], u))
    s_all = torch.stack(states, 0)  # [C, BH, N, V]: the state entering each chunk
    u_all = torch.stack(us, 0)  # [C, BH, S, V]
    out = torch.matmul(r_t, s_all) + qrkv[:, :, span:] - torch.matmul(rb, u_all)  # [C, BH, S, V]
    out = out.reshape(chunks, batch, num_heads, span, head_dim).permute(1, 0, 3, 2, 4).reshape(batch, padded,
                                                                                              num_heads, head_dim)
    state = state.reshape(batch, num_heads, head_dim, head_dim).transpose(-2, -1).contiguous().to(state_dtype)
    return out[:, :seq_len].to(dtype), state


def rwkv7_parallel_lean(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kwargs):
    """Same packed-sequence contract as the vendored ``rwkv7_parallel``."""
    if cu_seq_lens is not None:
        bounds = cu_seq_lens.tolist()
        outputs = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop <= start:
                continue
            out, state = rwkv7_chunked_lean(r[:, start:stop], w_log[:, start:stop], k[:, start:stop],
                                            v[:, start:stop], kk[:, start:stop], a[:, start:stop],
                                            torch.zeros_like(state))
            outputs.append(out)
        return torch.cat(outputs, dim=1), state
    return rwkv7_chunked_lean(r, w_log, k, v, kk, a, state)


rwkv7_parallel_lean.__name__ = "rwkv7_parallel_lean"


def register() -> None:
    """Idempotent: add the lean entry to the vendored registry (selection is separate: ``select_wkv``)."""
    M.RWKV7_WKV_FUNCTIONS[NAME] = rwkv7_parallel_lean

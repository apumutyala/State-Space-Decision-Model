"""Gated DeltaNet (GDN) token mixer, parameter-compatible with HF `Qwen3_5GatedDeltaNet`.

Three interchangeable kernels for the gated delta rule
    S_t = exp(g_t) * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ,   o_t = S_t^T q_t   (q scaled by d^-1/2)
are provided:

* ``recurrent``  -- naive per-token loop (pure PyTorch, fp32). The numerical reference.
* ``chunk``      -- chunked WY/UT-transform form (pure PyTorch, fp32). Same math, O(T/C) sequential steps.
* ``fla``        -- flash-linear-attention Triton kernel (works on Windows via ``triton-windows``).

``tests/test_gdn.py`` checks all three against each other (forward and gradients) and checks the module
against the HF reference layer on real Qwen3.5-0.8B weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_FLA = None


def _fla_chunk():
    global _FLA
    if _FLA is None:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore

            _FLA = chunk_gated_delta_rule
        except Exception:  # pragma: no cover - depends on environment
            _FLA = False
    return _FLA


def fla_available() -> bool:
    return bool(_fla_chunk()) and torch.cuda.is_available()


def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdr_recurrent(q, k, v, g, beta, initial_state=None, qk_l2norm=True):
    """Naive gated delta rule. q,k: [B,T,H,Dk]; v: [B,T,H,Dv]; g,beta: [B,T,H]. Returns o [B,T,H,Dv], S [B,H,Dk,Dv]."""
    if torch.is_autocast_enabled(v.device.type):  # the reference stays fp32 under autocast (the v3 decode path)
        with torch.autocast(device_type=v.device.type, enabled=False):
            return gdr_recurrent(q, k, v, g, beta, initial_state, qk_l2norm)
    dt = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if qk_l2norm:
        q, k = _l2n(q), _l2n(k)
    q = q * q.shape[-1] ** -0.5
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, device=v.device) if initial_state is None else initial_state.float()
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt, bt = k[:, t], v[:, t], beta[:, t][..., None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, (vt - pred) * bt)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outs, 1).to(dt), S


def gdr_chunk(q, k, v, g, beta, initial_state=None, qk_l2norm=True, chunk_size=64):
    """Chunked gated delta rule (WY / UT transform), written independently of the HF fallback."""
    dt = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if qk_l2norm:
        q, k = _l2n(q), _l2n(k)
    q = q * q.shape[-1] ** -0.5
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    # -> [B,H,N,C,D]
    def prep(x):
        x = x.transpose(1, 2)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(B, H, -1, C, x.shape[-1])

    q, k, v = prep(q), prep(k), prep(v)
    g = F.pad(g.transpose(1, 2), (0, pad)).reshape(B, H, -1, C)
    beta = F.pad(beta.transpose(1, 2), (0, pad)).reshape(B, H, -1, C)
    N = q.shape[2]
    G = g.cumsum(-1)  # log decay from chunk start to (and including) position i
    tril = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    # decay between j and i (i >= j): exp(G_i - G_j)
    Dmat = (G[..., :, None] - G[..., None, :]).masked_fill(~tril, float("-inf")).exp()
    kb = k * beta[..., None]
    # Delta-rule within chunk: u_i = beta_i (v_i - sum_{j<i} exp(G_i-G_j) (k_i.k_j) u_j - exp(G_i) k_i^T S0)
    # => (I + A) U = beta*V - diag(beta exp(G)) K S0, with A_ij = beta_i exp(G_i-G_j) k_i.k_j (j<i)
    A = (kb @ k.transpose(-1, -2)) * Dmat
    A = A.masked_fill(~strict, 0.0)
    I = torch.eye(C, device=q.device).expand_as(A)
    Tm = torch.linalg.solve_triangular(I + A, I.contiguous(), upper=False, unitriangular=True)  # (I+A)^-1
    Ub = Tm @ (v * beta[..., None])  # part from values
    Wk = Tm @ (kb * G.exp()[..., None])  # part multiplying S0
    attn = ((q @ k.transpose(-1, -2)) * Dmat).masked_fill(~tril, 0.0)
    S = torch.zeros(B, H, Dk, Dv, device=q.device) if initial_state is None else initial_state.float()
    out = torch.empty(B, H, N, C, Dv, device=q.device)
    for n in range(N):
        U = Ub[:, :, n] - Wk[:, :, n] @ S  # [B,H,C,Dv]
        qn = q[:, :, n] * G[:, :, n].exp()[..., None]
        out[:, :, n] = qn @ S + attn[:, :, n] @ U
        last = G[:, :, n, -1]
        kdec = k[:, :, n] * (last[..., None] - G[:, :, n]).exp()[..., None]
        S = S * last.exp()[..., None, None] + kdec.transpose(-1, -2) @ U
    out = out.reshape(B, H, N * C, Dv)[:, :, :T].transpose(1, 2)
    return out.to(dt), S


def gdr_fla(q, k, v, g, beta, initial_state=None, qk_l2norm=True):
    fn = _fla_chunk()
    if not fn:
        raise RuntimeError("flash-linear-attention not available")
    o, S = fn(q, k, v, g=g.float(), beta=beta, initial_state=initial_state, output_final_state=True,
              use_qk_l2norm_in_kernel=qk_l2norm)
    return o, S


KERNELS = {"recurrent": gdr_recurrent, "chunk": gdr_chunk, "fla": gdr_fla}


class RMSNormGated(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x, z):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dt)
        return (x * F.silu(z.float())).to(dt)


class GDN(nn.Module):
    """Parameter names/shapes identical to HF Qwen3_5GatedDeltaNet so state dicts are interchangeable."""

    def __init__(self, hidden_size, n_k_heads=16, n_v_heads=16, dk=128, dv=128, conv_k=4, eps=1e-6, kernel="fla"):
        super().__init__()
        self.hidden_size, self.nk, self.nv, self.dk, self.dv = hidden_size, n_k_heads, n_v_heads, dk, dv
        self.key_dim, self.value_dim = dk * n_k_heads, dv * n_v_heads
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_k
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_k, groups=self.conv_dim, bias=False, padding=conv_k - 1)
        self.dt_bias = nn.Parameter(torch.ones(n_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(n_v_heads).uniform_(0.01, 16)))
        self.norm = RMSNormGated(dv, eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, n_v_heads, bias=False)
        self.kernel = kernel

    @classmethod
    def from_config(cls, cfg, kernel="fla"):
        return cls(cfg.hidden_size, cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                   cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim, cfg.rms_norm_eps, kernel)

    def forward(self, x, return_state=False):
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        qkv = F.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], -1)
        q = q.reshape(B, T, self.nk, self.dk)
        k = k.reshape(B, T, self.nk, self.dk)
        v = v.reshape(B, T, self.nv, self.dv)
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        if self.nv // self.nk > 1:
            r = self.nv // self.nk
            q, k = q.repeat_interleave(r, 2), k.repeat_interleave(r, 2)
        kern = self.kernel
        if kern == "fla" and not (x.is_cuda and fla_available()):
            kern = "chunk"
        o, S = KERNELS[kern](q, k, v, g, beta)
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(B, T, -1)
        out = self.out_proj(o)
        return (out, S) if return_state else out

    def step(self, x, state=None):
        """Continue from ``state`` = (conv_tail [B, conv_dim, K-1] of pre-conv inputs, S [B, H, Dk, Dv]) or None
        (fresh sequence). Returns (out, new_state). ``step`` over consecutive chunks equals ``forward`` over their
        concatenation (the causal conv sees the real left context instead of zero padding)."""
        B, T, _ = x.shape
        K = self.conv_kernel_size
        pre = self.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        if state is None:
            tail = pre.new_zeros(B, self.conv_dim, K - 1)
            S0 = None
        else:
            tail, S0 = state
        full = torch.cat([tail.to(pre.dtype), pre], -1)
        qkv = F.silu(F.conv1d(full, self.conv1d.weight.to(pre.dtype), None, groups=self.conv_dim)).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], -1)
        q = q.reshape(B, T, self.nk, self.dk)
        k = k.reshape(B, T, self.nk, self.dk)
        v = v.reshape(B, T, self.nv, self.dv)
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        if self.nv // self.nk > 1:
            r = self.nv // self.nk
            q, k = q.repeat_interleave(r, 2), k.repeat_interleave(r, 2)
        kern = self.kernel
        if kern == "fla" and not (x.is_cuda and fla_available()):
            kern = "chunk"
        o, S = KERNELS[kern](q, k, v, g, beta, initial_state=S0)
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(B, T, -1)
        return self.out_proj(o), (full[..., -(K - 1):].detach(), S)

    def state_bytes(self, dtype_bytes=4):
        """Fixed decode-state size (recurrent matrix + conv tail) per sequence."""
        return (self.nv * self.dk * self.dv + self.conv_dim * (self.conv_kernel_size - 1)) * dtype_bytes

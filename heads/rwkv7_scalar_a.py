"""Contractive-by-construction RWKV-7 state: the in-context learning rate ``a`` SHARED per head (opt-in module swap).

docs/STATE_STABILITY.md (claude/lead-state-stability @836bc6f6) T2: the per-step transition of one RWKV-7 head is
``A_t = diag(w_t) - diag(a_t) k_t k_t^T`` (key-first; ``k`` = the l2-normalised ``kk``), with the per-input certificate
``||A_t||_2 <= U_t = w_max,t + sigma_a,t`` (sigma_a = the kk^2-weighted spread of ``a`` over the head's channels). With
per-channel ``a`` (the stock model) U_t can exceed 1 and two admissible steps can alternate with rho(A2 A1)^(1/2) = 1.105
(T2(e)): no input-independent contraction exists. If ``a`` is one scalar per head and step, sigma_a = 0, A_t is
symmetric with eigenvalues in [w_min - a, w_max], and ``||A_t||_2 <= w_max,t < 1`` for EVERY input (T2(c), exact
arithmetic; float caveats F3 there: the chunked WKV casts to fp32 before exp, a bf16 recurrent path may round w to 1).

The swap (``enable_scalar_a``): each ``Rwkv7Attention.lora_gates`` keeps computing the stock per-channel gate
``a_c = sigmoid(a0 + x A1 A2)`` and then replaces it, per head and token, by

    a_h = sigmoid(logit(mean_{c in h} a_c) + b_h),      b_h = ``scalar_a.logit_bias`` (zero-init, trainable)

broadcast over the head's channels. At b = 0 this is EXACTLY the mean of the pretrained per-channel ``a`` (so the swap
starts from the parent's average rate, no new projection to learn); ``a_h`` stays in (0, 1) for any parameters, so
the guarantee holds through fine-tuning. The shared ``a`` also enters the write key ``k * (1 + (a - 1) k_a)``
(a ``B_t`` term: it does not affect the contraction). Opt-in only: a model without ``enable_scalar_a`` runs the
vendored code unchanged (bit-identical). The vendored files stay unmodified (the hook is an instance attribute).

Training: the bias is named ``...scalar_a.logit_bias``; ``train.lora.apply_lora`` leaves parameters with that suffix
trainable and ``train.lora.lora_parameters`` / ``lora_state_dict`` include them (the cell's LoRA group + export).
"""
from __future__ import annotations

import types

import torch
from torch import nn

SUFFIX = "scalar_a.logit_bias"
_EPS = 1e-6


def share_per_head(a: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[..., H*N] per-channel rate -> its per-head mean broadcast back over the head's channels."""
    shp = a.shape
    h = a.reshape(*shp[:-1], num_heads, shp[-1] // num_heads)
    return h.mean(-1, keepdim=True).expand_as(h).reshape(shp)


class ScalarA(nn.Module):
    """Per-head logit offset of the shared rate (zero-init: the swap starts at the mean of the per-channel ``a``)."""

    def __init__(self, num_heads: int, device=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.logit_bias = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        shp = a.shape
        h = a.reshape(*shp[:-1], self.num_heads, shp[-1] // self.num_heads)
        m = h.float().mean(-1, keepdim=True)
        s = torch.sigmoid(torch.logit(m, eps=_EPS) + self.logit_bias.view(self.num_heads, 1))
        return s.to(a.dtype).expand_as(h).reshape(shp)


def _gates(self, xw, xa, xg, xv, tokenwise):
    w_log, a, g, v_gate = type(self).lora_gates(self, xw, xa, xg, xv, tokenwise)
    return w_log, self.scalar_a(a), g, v_gate


def attention_modules(model: nn.Module) -> list[nn.Module]:
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7Attention

    return [m for m in model.modules() if isinstance(m, Rwkv7Attention)]


def enable_scalar_a(model: nn.Module) -> int:
    """Install the per-head shared ``a`` on every RWKV-7 attention block of ``model`` (idempotent). Returns the number
    of blocks switched. Refuses a model with no RWKV-7 block (a wrong config would otherwise be a silent no-op)."""
    mods = attention_modules(model)
    if not mods:
        raise ValueError("rwkv7_scalar_a: the model has no Rwkv7Attention block")
    for m in mods:
        if getattr(m, "scalar_a", None) is None:
            m.scalar_a = ScalarA(m.num_heads, device=m.a0.device)
            m.lora_gates = types.MethodType(_gates, m)
    return len(mods)


def is_enabled(model: nn.Module) -> bool:
    mods = attention_modules(model)
    return bool(mods) and all(getattr(m, "scalar_a", None) is not None for m in mods)


def scalar_a_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(SUFFIX)]


# ----------------------------------------------------------------------------------------- certificate (T2(b), T2(c))
def transition(w_log: torch.Tensor, kk: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Per-step, per-head transition ``A = diag(w) - diag(a) kk kk^T`` (key-first) from the WKV inputs
    ``[..., N]`` (one head's channels) -> ``[..., N, N]`` in float64."""
    w = torch.exp(w_log.double())
    kk, a = kk.double(), a.double()
    return torch.diag_embed(w) - a.unsqueeze(-1) * kk.unsqueeze(-1) * kk.unsqueeze(-2)


def certificate(w_log: torch.Tensor, kk: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """T2(b) general form (non-unit kk allowed): ``max(w_max, a_bar q - w_min) + sqrt(q) ||(diag(a) - a_bar) kk||``,
    ``q = ||kk||^2``, ``a_bar = sum a kk^2 / q``. With a shared per head the second term is 0."""
    w = torch.exp(w_log.double())
    kk, a = kk.double(), a.double()
    q = (kk * kk).sum(-1)
    abar = (a * kk * kk).sum(-1) / q.clamp_min(1e-300)
    spread = ((a - abar.unsqueeze(-1)) * kk).norm(dim=-1)
    return torch.maximum(w.max(-1).values, abar * q - w.min(-1).values) + q.sqrt() * spread


def spectral_norm(A: torch.Tensor, iters: int = 200, seed: int = 0) -> torch.Tensor:
    """||A||_2 by power iteration on A^T A (batched over leading dims), float64."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*A.shape[:-1], 1, generator=g, dtype=A.dtype)
    ata = A.transpose(-1, -2) @ A
    for _ in range(iters):
        x = ata @ x
        x = x / x.norm(dim=-2, keepdim=True).clamp_min(1e-300)
    return (A @ x).norm(dim=-2).squeeze(-1)

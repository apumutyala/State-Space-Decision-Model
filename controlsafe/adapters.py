"""Family adapters: one view of RWKV-7 (vendored HF code) and GDN / Q9-AF (HF ``Qwen3_5``, all
``linear_attention``) as the same linear-in-state recurrence.

Every layer's recurrent state is written key-axis first, ``X_t in R^{H x Nk x Nv}``, and both
families are then

    X_t = A_t X_{t-1} + k~_t v~_t^T        y_t = X_t^T r_t           (per head)

with a key-axis transition ``A_t`` (``Nk x Nk``) that depends on the layer input at t (and, through
the token shift / causal conv, on the few inputs before it):

* RWKV-7  (state stored ``(value, key)``; ``S <- S M``):  ``A_t = M_t^T = diag(w_t) - (kk_t * a_t) kk_t^T``
* GDN     (state stored ``(key, value)``):               ``A_t = alpha_t (I - beta_t k_t k_t^T)``

The linear map on the whole state is ``I_v (x) A_t``, so its induced 2-norm equals ``||A_t||_2``.

``trace(model)`` swaps each layer's recurrence kernel for a reference token-by-token implementation
(the RWKV-7 registry is open by design; GDN kernels are instance attributes) that records, per layer,
the transition factors, writes, readout vectors and every per-step state, differentiably. The traced
kernels are the reference recurrences (``rwkv7_recurrent`` / ``torch_recurrent_gated_delta_rule``)
rewritten in the key-first convention; they are for analysis (influence, Gramians, spectra) and agree
with the stock chunked kernels to float tolerance (tested), not bitwise.

``Runner`` is the prefill/continue/clone interface the snapshot certifier uses.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import nn

FAMILIES = ("rwkv7", "gdn")
TRACE_IMPL = "controlsafe_trace"


# ----------------------------------------------------------------------------------- structure
def family_of(model: nn.Module) -> str:
    name = type(model).__name__
    if name.startswith("Rwkv7"):
        return "rwkv7"
    if name.startswith("Qwen3_5"):
        return "gdn"
    raise ValueError(f"controlsafe supports RWKV-7 (vendored HF) and Qwen3.5/Q9-AF GDN models, not {name}")


def inner_model(model: nn.Module) -> nn.Module:
    """The module that owns the block list (``Rwkv7Model`` / ``Qwen3_5TextModel``)."""
    for attr in ("rwkv7", "model"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module) and (hasattr(sub, "blocks") or hasattr(sub, "layers")):
            return sub
    return model


def blocks(model: nn.Module) -> nn.ModuleList:
    m = inner_model(model)
    out = getattr(m, "blocks", None)
    if out is None:
        out = getattr(m, "layers")
    return out


def mixer(block: nn.Module) -> nn.Module:
    """The recurrent token mixer of a block (RWKV ``att`` / GDN ``linear_attn``)."""
    for attr in ("att", "linear_attn"):
        if hasattr(block, attr):
            return getattr(block, attr)
    raise ValueError(f"block {type(block).__name__} has no recurrent mixer (attention layer in a hybrid?)")


def recurrent_layer_ids(model: nn.Module) -> list[int]:
    out = []
    for i, b in enumerate(blocks(model)):
        if hasattr(b, "att") or hasattr(b, "linear_attn"):
            out.append(i)
    return out


def config(model: nn.Module):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "text_config", None) or cfg


def local_window(model: nn.Module) -> int:
    """How many positions after t a perturbation of a layer input at t still changes that layer's
    transition or write: 1 for the RWKV-7 token shift, ``conv_kernel - 1`` for the GDN causal conv."""
    fam = family_of(model)
    if fam == "rwkv7":
        return 1
    return int(getattr(config(model), "linear_conv_kernel_dim", 4)) - 1


def embed(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return inner_model(model).get_input_embeddings()(ids)


def hidden_states(model: nn.Module, ids: torch.Tensor | None = None, embeds: torch.Tensor | None = None,
                  attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """``output_hidden_states`` tuple, from ids or input embeddings (no cache)."""
    kw: dict[str, Any] = {"output_hidden_states": True, "use_cache": False}
    if embeds is not None:
        kw["inputs_embeds"] = embeds
    else:
        kw["input_ids"] = ids
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    return model(**kw).hidden_states


# ----------------------------------------------------------------------------------- tracing
@dataclass
class LayerTrace:
    """Per layer, batch-first, key-first. ``states[:, t]`` is X_t (after token t)."""

    family: str
    layer: int
    states: torch.Tensor  # [B, T, H, Nk, Nv]
    readout: torch.Tensor  # [B, T, H, Nk]   (RWKV r; GDN q * scale)
    write_k: torch.Tensor  # [B, T, H, Nk]
    write_v: torch.Tensor  # [B, T, H, Nv]
    init_state: torch.Tensor  # [B, H, Nk, Nv]
    factors: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.states.shape[1]

    def transition(self, t: int | slice | None = None) -> torch.Tensor:
        """A_t, ``[B, T', H, Nk, Nk]`` (all positions when t is None)."""
        f = self.factors
        sel = slice(None) if t is None else t
        if self.family == "rwkv7":
            w, kk, a = f["w"][:, sel], f["kk"][:, sel], f["a"][:, sel]
            return torch.diag_embed(w) - (kk * a).unsqueeze(-1) * kk.unsqueeze(-2)
        alpha, beta, k = f["alpha"][:, sel], f["beta"][:, sel], f["k"][:, sel]
        eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
        return alpha[..., None, None] * (eye - beta[..., None, None] * k.unsqueeze(-1) * k.unsqueeze(-2))


class Recorder:
    def __init__(self) -> None:
        self.layers: dict[int, LayerTrace] = {}
        self.current: int | None = None


def _rwkv_trace_kernel(rec: Recorder):
    def kernel(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **_):
        if cu_seq_lens is not None:
            raise NotImplementedError("controlsafe trace: packed rows are not supported")
        cd = torch.float64 if r.dtype == torch.float64 else torch.float32
        B, T, H, N = r.shape
        X = state.to(cd).transpose(-1, -2)  # (value,key) -> key-first
        X0 = X
        w = torch.exp(w_log.to(cd))
        kk_, a_, k_, v_, r_ = kk.to(cd), a.to(cd), k.to(cd), v.to(cd), r.to(cd)
        outs, states = [], []
        for t in range(T):
            # X <- A_t X + k v^T with A_t = diag(w) - (kk*a) kk^T   (== reference S <- S M + v k^T)
            kt = kk_[:, t]
            proj = (kt.unsqueeze(-1) * X).sum(-2)  # kk^T X : [B,H,Nv]
            X = w[:, t].unsqueeze(-1) * X - (kt * a_[:, t]).unsqueeze(-1) * proj.unsqueeze(-2)
            X = X + k_[:, t].unsqueeze(-1) * v_[:, t].unsqueeze(-2)
            states.append(X)
            outs.append((X * r_[:, t].unsqueeze(-1)).sum(-2))  # X^T r
        if rec.current is None:
            raise RuntimeError("trace: layer id unknown (pre-hook missing)")
        rec.layers[rec.current] = LayerTrace(
            "rwkv7", rec.current, torch.stack(states, 1), r_, k_, v_, X0,
            {"w": w, "kk": kk_, "a": a_})
        out = torch.stack(outs, 1).to(r.dtype)
        return out, X.transpose(-1, -2).to(state.dtype)

    return kernel


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _gdn_trace_kernel(rec: Recorder, layer: int):
    def kernel(query, key, value, g=None, beta=None, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, **_):
        cd = torch.float64 if query.dtype == torch.float64 else torch.float32
        if use_qk_l2norm_in_kernel:
            query, key = _l2norm(query), _l2norm(key)
        q, k, v, b, gg = (x.to(cd) for x in (query, key, value, beta, g))  # [B,T,H,*]
        B, T, H, Nk = k.shape
        Nv = v.shape[-1]
        q = q * (Nk ** -0.5)
        X = (torch.zeros(B, H, Nk, Nv, dtype=cd, device=k.device) if initial_state is None
             else initial_state.to(cd))
        X0 = X
        alpha = torch.exp(gg)
        outs, states = [], []
        for t in range(T):
            X = alpha[:, t, :, None, None] * X
            kv_mem = (X * k[:, t].unsqueeze(-1)).sum(-2)
            delta = (v[:, t] - kv_mem) * b[:, t].unsqueeze(-1)
            X = X + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            states.append(X)
            outs.append((X * q[:, t].unsqueeze(-1)).sum(-2))
        rec.layers[layer] = LayerTrace(
            "gdn", layer, torch.stack(states, 1), q, b.unsqueeze(-1) * k, v, X0,
            {"alpha": alpha, "beta": b, "k": k})
        out = torch.stack(outs, 1).to(value.dtype)
        return out, (X.to(value.dtype) if output_final_state else None)

    return kernel


@contextlib.contextmanager
def trace(model: nn.Module) -> Iterator[Recorder]:
    """Within the context every forward records a ``LayerTrace`` per recurrent layer (overwriting the
    previous forward's). Note ``write_k`` for GDN is ``beta * k`` and ``write_v`` is the raw value: the
    GDN write is ``beta k (v - alpha X^T k)^T``, i.e. ``A_t`` already contains the erase term."""
    fam = family_of(model)
    rec = Recorder()
    handles = []
    if fam == "rwkv7":
        from third_party.rwkv7_hf import modeling_rwkv7 as M

        cfg = config(model)
        old_impl = cfg.wkv_implementation
        M.RWKV7_WKV_FUNCTIONS[TRACE_IMPL] = _rwkv_trace_kernel(rec)
        cfg.wkv_implementation = TRACE_IMPL
        for i, b in enumerate(blocks(model)):
            def pre(mod, args, _i=i):
                rec.current = _i
            handles.append(b.att.register_forward_pre_hook(pre))
        try:
            yield rec
        finally:
            cfg.wkv_implementation = old_impl
            M.RWKV7_WKV_FUNCTIONS.pop(TRACE_IMPL, None)
            for h in handles:
                h.remove()
        return
    saved = []
    for i, b in enumerate(blocks(model)):
        if not hasattr(b, "linear_attn"):
            continue
        la = b.linear_attn
        saved.append((la, la.chunk_gated_delta_rule, la.recurrent_gated_delta_rule))
        la.chunk_gated_delta_rule = _gdn_trace_kernel(rec, i)
        la.recurrent_gated_delta_rule = _gdn_trace_kernel(rec, i)
    try:
        yield rec
    finally:
        for la, c, r in saved:
            la.chunk_gated_delta_rule, la.recurrent_gated_delta_rule = c, r


# ----------------------------------------------------------------------------------- universal bounds
def _eff_weight(lin: nn.Module) -> torch.Tensor:
    """Effective weight of an ``nn.Linear`` or a ``train.lora.LoRALinear`` (base + B A * scaling)."""
    base = getattr(lin, "base", lin)
    w = base.weight.detach().double()
    if hasattr(lin, "lora_A"):
        w = w + (lin.lora_B.detach().double() @ lin.lora_A.detach().double()) * lin.scaling
    return w


def universal_decay_bound(model: nn.Module) -> dict[int, dict[str, Any]]:
    """Per layer, an upper bound on the transition that holds for EVERY input (exact arithmetic).

    GDN: ``||A_t||_2 = alpha_t * ||I - beta k k^T||_2 = alpha_t`` (``beta`` in (0,1), ``||k|| <= 1`` after the
    kernel's l2norm), and ``alpha_t = exp(-exp(A_log) softplus(W_a x + dt_bias))`` with ``x`` the output of the
    layer's input RMSNorm ``x = (1+gamma) * xhat`` with ``||xhat||_2 <= sqrt(d)``, so ``W_a[h] x >= -||W_a[h] * (1+gamma)||_2 sqrt(d)``.
    That gives ``rho_h < 1`` for every head: a CERTIFIED common induced-norm (2-norm, key axis) bound, hence
    ``||A_T ... A_1||_2 <= rho^T`` for every input sequence.

    RWKV-7: only the diagonal (decay) part has a universal bound: ``tanh`` is in [-1, 1], so the decay
    pre-activation is ``>= w0_c - ||W2[:, c]||_1`` and ``w_c <= exp(-e^-0.5 sigmoid(w0_c - ||W2[:, c]||_1))``.
    The full transition ``diag(w) - (kk*a) kk^T`` is NOT 2-norm bounded by 1 in general (non-symmetric when
    ``a`` varies across channels; ``tests/controlsafe`` has an explicit counterexample), so for RWKV-7 the
    full-transition bound is reported as ``None`` and only observed sups are available (EMPIRICAL)."""
    fam = family_of(model)
    out: dict[int, dict[str, Any]] = {}
    for i, b in enumerate(blocks(model)):
        if fam == "gdn":
            if not hasattr(b, "linear_attn"):
                continue
            la, norm = b.linear_attn, b.input_layernorm
            d = la.hidden_size
            gamma = norm.weight.detach().double()
            xmax = (d ** 0.5) * float((1.0 + gamma).abs().max())
            wa = _eff_weight(la.in_proj_a)  # [H, d]
            # W_a[h] x = sum_i (W_a[h,i] (1+gamma_i)) xhat_i with ||xhat|| <= sqrt(d): the tight constant
            lo = -(wa * (1.0 + gamma)).norm(dim=-1) * (d ** 0.5) + la.dt_bias.detach().double()
            alpha = torch.exp(-la.A_log.detach().double().exp() * torch.nn.functional.softplus(lo))
            out[i] = {"family": "gdn", "norm": "2 (key axis, common to all t)", "per_head": alpha.tolist(),
                      "rho": float(alpha.max()), "certified": bool(alpha.max() < 1.0), "x_norm_bound": xmax}
        else:
            att = b.att
            w2 = att.w2.detach().double()
            w0 = att.w0.detach().double().reshape(-1)
            lo = w0 - w2.abs().sum(0)
            wbar = torch.exp(-0.6065306597126334 * torch.sigmoid(lo))
            out[i] = {"family": "rwkv7", "norm": "diagonal decay part only", "per_channel_max": float(wbar.max()),
                      "decay_rho": float(wbar.max()), "rho": None, "certified": False,
                      "note": "delta-rule term (kk*a) kk^T has no universal 2-norm bound < 1; see observed sups"}
    return out


# ----------------------------------------------------------------------------------- state runner
class Runner:
    """prefill / continue / clone for snapshot reuse. ``run(ids, state)`` returns ``(state, last_feature)``
    where the feature is the backbone's final hidden state (``hidden_states[-1]``) at the last token.
    The state passed in is consumed (HF caches update in place): clone first to keep it."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.family = family_of(model)
        self.tokens = 0
        self.calls = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def run(self, ids, state=None):
        ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)[None]
        self.tokens += ids.shape[1]
        self.calls += 1
        if self.family == "rwkv7":
            out = self.model(input_ids=ids, state=state, use_cache=True, output_hidden_states=True)
            return out.state, out.hidden_states[-1][0, -1]
        from heads.backbone import _linear_safe_cache, _set_seen

        if state is None:
            state = _linear_safe_cache(config(self.model))
            state.seen = 0
        seen = state.seen
        out = self.model(input_ids=ids, past_key_values=state, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        _set_seen(cache, seen + ids.shape[1])
        return cache, out.hidden_states[-1][0, -1]

    @staticmethod
    def clone(state):
        return copy.deepcopy(state)

    @torch.no_grad()
    def full(self, ids) -> torch.Tensor:
        """Final-token feature of one fresh full pass (the recompute baseline)."""
        _, h = self.run(ids, None)
        return h

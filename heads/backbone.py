"""Backbone-agnostic hidden-state interface.

A head never touches a backbone directly. It sees token features ``[B, T, D]``
produced by ``HiddenStateBackbone.features``, which wraps any ``nn.Module`` LM:

* HF models that accept ``output_hidden_states`` (Qwen3.5 hybrid, a Q9-AF
  checkpoint saved with all-``linear_attention`` ``layer_types``, HF Mamba,
  RWKV7 via its custom ``modeling_rwkv7.py``) -> ``capture="hidden_states"``.
* Anything else (``mamba_ssm`` Mamba-3 ``MambaLMHeadModel``, ``fla`` RWKV7) ->
  ``capture="<module name>"``: a forward hook on the final norm captures features,
  and the LM head is never run (no vocab-sized logits, which matters at 248k vocab).

Device: every tensor the wrapper builds lives on the backbone's parameter device.

Right padding only. Recurrent backbones without a padding mask are exact on
right-padded rows because outputs at valid positions never depend on later
positions.

Final-state fork, two modes (``fork_mode``):

``recompute``  run prefix + continuation_k for every option (K full rows). Exact and
               backbone-agnostic, but costs K x prefix tokens: infeasible at K = 255 on
               long rows (train/estimate.py prices it).
``cache``      run the prefix once with a cache, replicate the cache K times
               (``reorder_cache`` with an all-zeros index = a state copy per option),
               then extend each copy by its continuation token by token. Cost: one
               prefix + K x continuation tokens + K state copies. Implemented for the
               tiny functional backbone (differentiable) and HF caches (Qwen3.5 hybrid,
               all-GDN Q9-AF layout, HF Mamba); tests check it equals recompute.

``BACKBONES`` records the m1 candidates. ``verified`` flags say which dimensions
were checked against the published ``config.json`` (internal review of PR #14).
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

#: (group name, regex over qualified module names, expected matches in the full model)
LoraGroup = tuple[str, str, int]
#: (in_features, out_features, count in the full model)
LoraShape = tuple[int, int, int]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str  # qwen3.5 | rwkv7 | mamba3 | mamba | tiny
    hf_id: str
    params: float  # non-embedding text parameters (FLOPs per token ~ 2 * params)
    n_layers: int
    d_model: int
    lora_groups: tuple[LoraGroup, ...]
    capture: str  # "hidden_states" or a module name for a forward hook
    attention_free: bool
    params_memory: float = 0.0  # resident weights (embeddings included, LM head dropped); 0 -> params
    lora_shapes: tuple[LoraShape, ...] = ()
    state_scalars_per_layer: int = 0  # recurrent state per layer (fork/state pricing)
    state_dtype_bytes: int = 4
    mfu_assumed: float = 0.30
    verified: dict[str, bool] = field(default_factory=dict)
    notes: str = ""

    @property
    def lora_targets(self) -> str:
        return "|".join(f"(?:{rx})" for _, rx, _ in self.lora_groups)

    @property
    def resident_params(self) -> float:
        return self.params_memory or self.params


_GDN = r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$"
_QATTN = r"self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
_QMLP = r"mlp\.(gate_proj|up_proj|down_proj)$"

_QWEN_NOTE = ("Qwen/Qwen3.5-9B config.json (verified in PR #14 review): 32 layers, hidden 4096, "
              "intermediate 12288, GDN 16 key x 128 / 32 value x 128 heads; 6.92e9 non-embedding, "
              "7.94e9 with input embeddings (LM head dropped).")

BACKBONES: dict[str, BackboneSpec] = {
    "q9af": BackboneSpec(
        name="q9af",
        family="qwen3.5",
        hf_id="Qwen/Qwen3.5-9B with its 8 full-attention layers distilled to GDN at m1",
        params=6.92e9,
        params_memory=7.94e9,
        n_layers=32,
        d_model=4096,
        lora_groups=(("gdn", _GDN, 32 * 3), ("mlp", _QMLP, 32 * 3)),
        capture="hidden_states",
        attention_free=True,
        lora_shapes=((4096, 8192, 32), (4096, 4096, 32), (4096, 4096, 32),
                     (4096, 12288, 32), (4096, 12288, 32), (12288, 4096, 32)),
        state_scalars_per_layer=32 * 128 * 128 + 33_000,
        state_dtype_bytes=4,  # mamba_ssm_dtype float32
        mfu_assumed=0.35,
        verified={"n_layers": True, "d_model": True, "gdn_heads": True, "intermediate_size": True,
                  "params": True, "module_names": True},
        notes=_QWEN_NOTE + " Params of the 8 converted layers change by <1% (GDN vs attention projections).",
    ),
    "qwen3.5-9b": BackboneSpec(
        name="qwen3.5-9b",
        family="qwen3.5",
        hf_id="Qwen/Qwen3.5-9B-Base (stock hybrid; rung-1 shape proxy for Q9-AF)",
        params=6.92e9,
        params_memory=7.94e9,
        n_layers=32,
        d_model=4096,
        lora_groups=(("gdn", _GDN, 24 * 3), ("attn", _QATTN, 8 * 4), ("mlp", _QMLP, 32 * 3)),
        capture="hidden_states",
        attention_free=False,
        lora_shapes=((4096, 8192, 24), (4096, 4096, 24), (4096, 4096, 24),
                     (4096, 8192, 8), (4096, 1024, 8), (4096, 1024, 8), (4096, 4096, 8),
                     (4096, 12288, 32), (4096, 12288, 32), (12288, 4096, 32)),
        state_scalars_per_layer=32 * 128 * 128 + 33_000,
        mfu_assumed=0.35,
        verified={"n_layers": True, "d_model": True, "attn_proj_shapes": False, "module_names": True},
        notes=_QWEN_NOTE + " Attention projection shapes (gated q) unverified; only LoRA size depends on them.",
    ),
    "rwkv7-13.3b": BackboneSpec(
        name="rwkv7-13.3b",
        family="rwkv7",
        hf_id="RWKV/RWKV7-G1k-13.3B-20260930",
        params=12.73e9,
        params_memory=13.0e9,
        n_layers=61,
        d_model=4096,
        lora_groups=(("att", r"att\.(receptance|key|value|output)$", 61 * 4),
                     ("ffn", r"ffn\.(key|value)$", 61 * 2)),
        capture="hidden_states",
        attention_free=True,
        lora_shapes=((4096, 4096, 61 * 4), (4096, 16384, 61), (16384, 4096, 61)),
        state_scalars_per_layer=64 * 64 * 64 + 2 * 4096,
        state_dtype_bytes=4,
        mfu_assumed=0.25,
        verified={"n_layers": True, "d_model": True, "heads": True, "ffn": True, "module_names": True,
                  "params_split": False},
        notes=("13.27e9 total (card); non-embedding/resident split assumes vocab 65536 x 4096 embeddings "
               "and an untied head. Module names are the HF custom modeling_rwkv7.py ones "
               "(att.receptance/key/value/output, ffn.key/value); fla-converted repos use attn.*_proj "
               "and need their own spec. Bundled WKV7 training throughput unverified -> low MFU."),
    ),
    "mamba3-1.5b": BackboneSpec(
        name="mamba3-1.5b",
        family="mamba3",
        hf_id="state-spaces/mamba3-siso-1.5b",
        params=1.24e9,
        params_memory=1.5e9,
        n_layers=24,
        d_model=2048,
        lora_groups=(("mixer", r"mixer\.(in_proj|out_proj)$", 24 * 2), ("mlp", r"mlp\.(fc1|fc2)$", 24 * 2)),
        capture="backbone.norm_f",
        attention_free=True,
        # in_proj width = 2*d_inner + 2*d_state*ngroups + 3*nheads + rope_angles (rope_angles = 64 assumed)
        lora_shapes=((2048, 2 * 4096 + 2 * 128 + 3 * 64 + 64, 24), (4096, 2048, 24),
                     (2048, 2 * 4096, 24), (4096, 2048, 24)),
        state_scalars_per_layer=64 * 64 * 128,
        state_dtype_bytes=4,
        mfu_assumed=0.30,
        verified={"params": True, "state_size": True, "n_layers": True, "d_intermediate": True,
                  "d_model": False, "in_proj_rope_angles": False},
        notes=("config: n_layer 24, d_intermediate 4096 (GatedMLP fc1/fc2 per block), ctx 2048, state 128. "
               "d_model 2048 inferred from 1.5e9 params; embeddings ~0.26e9 (llama-3.1 vocab)."),
    ),
}


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class HiddenStateBackbone(nn.Module):
    """Uniform ``features(input_ids, attention_mask) -> [B, T, D]`` over any LM module."""

    FORK_MODES = ("recompute", "cache")

    def __init__(self, model: nn.Module, capture: str = "hidden_states", layer: int = -1,
                 spec: BackboneSpec | None = None, pad_id: int = 0, fork_mode: str = "recompute") -> None:
        super().__init__()
        if fork_mode not in self.FORK_MODES:
            raise ValueError(fork_mode)
        self.model = model
        self.capture = capture
        self.layer = layer
        self.spec = spec
        self.pad_id = pad_id
        self.fork_mode = fork_mode
        self._captured: torch.Tensor | None = None
        self._hook = None
        if capture != "hidden_states":
            target = dict(model.named_modules())[capture]
            self._hook = target.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._captured = output[0] if isinstance(output, tuple) else output

    @property
    def device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    @property
    def d_model(self) -> int:
        cfg = getattr(self.model, "config", None)
        for key in ("hidden_size", "d_model"):
            if cfg is not None and getattr(cfg, key, None):
                return int(getattr(cfg, key))
        raise AttributeError("cannot infer d_model; set config.hidden_size")

    def enable_gradient_checkpointing(self) -> bool:
        fn = getattr(self.model, "gradient_checkpointing_enable", None)
        if fn is None:
            return False
        try:
            fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            fn()
        return True

    # ------------------------------------------------------------------ plain features
    def _pick(self, out) -> torch.Tensor:
        if self.capture == "hidden_states":
            return out.hidden_states[self.layer]
        if self._captured is None:
            raise RuntimeError(f"hook on {self.capture} did not fire")
        return self._captured

    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        dev = self.device
        input_ids = input_ids.to(dev)
        self._captured = None
        if self.capture == "hidden_states":
            kwargs: dict[str, Any] = {"input_ids": input_ids, "output_hidden_states": True}
            if attention_mask is not None and getattr(self.model, "_accepts_mask", True):
                kwargs["attention_mask"] = attention_mask.to(dev)
            try:
                kwargs["use_cache"] = False
                out = self.model(**kwargs)
            except TypeError:
                kwargs.pop("use_cache", None)
                kwargs.pop("attention_mask", None)
                out = self.model(**kwargs)
            return self._pick(out)
        self.model(input_ids)
        return self._pick(None)

    def forward(self, input_ids, attention_mask=None):  # pragma: no cover - alias
        return self.features(input_ids, attention_mask)

    # ------------------------------------------------------------------ cache-copy fork
    def fork_features(self, prefix: Sequence[int], conts: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """One prefix, K continuations -> (prefix features [T, D], fork features [K, D]).

        The prefix runs once; its recurrent state (or HF cache) is copied K times and each
        copy is extended by its continuation. Equals the recompute fork (tested)."""
        dev = self.device
        k = len(conts)
        ids = torch.tensor([list(prefix)], dtype=torch.long, device=dev)
        cids, cmask = pad_batch(conts, self.pad_id, device=dev)
        last = cmask.sum(-1) - 1
        if hasattr(self.model, "prefill"):  # functional state API: differentiable
            h, state = self.model.prefill(ids)
            hc, _ = self.model.extend(cids, self.model.expand_state(state, k))
            return h[0], hc[torch.arange(k, device=dev), last]
        return self._hf_fork(ids, cids, last)

    def _hf_fork(self, ids: torch.Tensor, cids: torch.Tensor, last: torch.Tensor):
        dev = ids.device
        m = self.model
        if m.training and torch.is_grad_enabled() and getattr(m, "is_gradient_checkpointing", False):
            # HF silently drops the cache under gradient checkpointing; the fork would then be wrong.
            raise RuntimeError("cache fork is unavailable while HF gradient checkpointing is on in training: "
                               "use fork_mode='recompute' or disable checkpointing for the fork head")
        uses_cache_params = _accepts(m.forward, "cache_params")
        base: dict[str, Any] = {"use_cache": True}
        if self.capture == "hidden_states":
            base["output_hidden_states"] = True
        self._captured = None
        if not uses_cache_params and _accepts(m.forward, "state"):
            return self._state_fork(ids, cids, last, base)
        if uses_cache_params:
            out = m(input_ids=ids, **base)
            cache = out.cache_params
        else:
            out = m(input_ids=ids, past_key_values=_linear_safe_cache(m.config), **base)
            cache = out.past_key_values
        if cache is None:
            raise RuntimeError("backbone returned no cache for the cache fork")
        h = self._pick(out)[0]
        n = ids.shape[1]
        _set_seen(cache, n)
        k = cids.shape[0]
        cache.reorder_cache(torch.zeros(k, dtype=torch.long, device=dev))  # K state copies
        _assign_not_copy(cache)  # HF linear-attention caches write in place, which breaks autograd
        feats = torch.zeros((k, h.shape[-1]), dtype=h.dtype, device=dev)
        for t in range(cids.shape[1]):  # token by token: the cached-decode path every HF SSM supports
            self._captured = None
            kw = dict(base)
            kw["cache_params" if uses_cache_params else "past_key_values"] = cache
            if uses_cache_params:
                kw["cache_position"] = torch.tensor([n + t], device=dev)
            out = m(input_ids=cids[:, t : t + 1], **kw)
            cache = out.cache_params if uses_cache_params else out.past_key_values
            _set_seen(cache, n + t + 1)
            step = self._pick(out)[:, -1]
            feats = torch.where((last == t).unsqueeze(-1), step, feats)
        return h, feats

    def _state_fork(self, ids: torch.Tensor, cids: torch.Tensor, last: torch.Tensor, base: dict[str, Any]):
        """Cache fork for models whose recurrent state travels as ``state=`` (RWKV-7, bundled HF code in
        ``third_party/rwkv7_hf``). The K copies are made with ``batch_repeat_interleave`` (new tensors by
        assignment); ``reorder_cache`` would ``copy_`` into the prefix's buffers in place, which breaks
        autograd. ``Rwkv7Cache.write`` already rebinds instead of copying when a state requires grad."""
        m = self.model
        out = m(input_ids=ids, **base)
        cache = out.state
        if cache is None:
            raise RuntimeError("backbone returned no state for the cache fork")
        h = self._pick(out)[0]
        k = cids.shape[0]
        cache.batch_repeat_interleave(k)  # K state copies
        feats = torch.zeros((k, h.shape[-1]), dtype=h.dtype, device=ids.device)
        for t in range(cids.shape[1]):
            self._captured = None
            out = m(input_ids=cids[:, t : t + 1], state=cache, **base)
            cache = out.state
            step = self._pick(out)[:, -1]
            feats = torch.where((last == t).unsqueeze(-1), step, feats)
        return h, feats


def _linear_safe_cache(config):
    """HF DynamicCache that also works when every layer is linear attention (Q9-AF layout).

    transformers 5.14 raises in get_seq_length/get_mask_sizes when a cache has no attention
    layer; the model only needs the number of tokens already consumed."""
    from transformers import DynamicCache

    class _Cache(DynamicCache):
        seen = 0

        def get_seq_length(self, layer_idx=0):
            try:
                return super().get_seq_length(layer_idx)
            except ValueError:
                return self.seen

        def get_mask_sizes(self, query_length, layer_idx=0):
            try:
                return super().get_mask_sizes(query_length, layer_idx)
            except ValueError:
                return self.seen + query_length, 0

    return _Cache(config=config)


def _assign_not_copy(cache) -> None:
    """Make HF ``LinearAttentionLayer`` cache updates functional (assignment, not ``copy_``).

    transformers 5.14 updates GDN conv/recurrent states with in-place ``copy_`` (static
    addresses for CUDA graphs). During training the step that reads a state also overwrites
    it, so backward fails. Rebinding the two update methods on this cache instance keeps
    the same values (tested against recompute) and lets gradients flow through the fork."""
    import types

    for layer in getattr(cache, "layers", []) or []:
        if not hasattr(layer, "update_recurrent_state") or not hasattr(layer, "conv_states"):
            continue
        orig_conv, orig_rec = layer.update_conv_state, layer.update_recurrent_state

        def conv(self, conv_states, state_idx=0, conv_kernel_size=None, _orig=orig_conv, **kw):
            if not self.is_conv_states_initialized[state_idx] or not self.has_previous_state[state_idx]:
                return _orig(conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size, **kw)
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
            self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx]:]
            return full

        def rec(self, recurrent_states, state_idx=0, _orig=orig_rec, **kw):
            if not self.is_recurrent_states_initialized[state_idx]:
                return _orig(recurrent_states, state_idx=state_idx, **kw)
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

        layer.update_conv_state = types.MethodType(conv, layer)
        layer.update_recurrent_state = types.MethodType(rec, layer)


def _set_seen(cache, n: int) -> None:
    if hasattr(cache, "seen"):
        cache.seen = n


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int = 0,
              device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence. Returns (ids [B,T], mask [B,T] bool) on ``device``."""
    t = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), t), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = True
    if device is not None:
        ids, mask = ids.to(device), mask.to(device)
    return ids, mask


# --------------------------------------------------------------------------------------
# Tiny attention-free test backbone (random init; CPU; used only by tests and toys)
# --------------------------------------------------------------------------------------


@dataclass
class TinyConfig:
    vocab_size: int = 260
    hidden_size: int = 32
    num_hidden_layers: int = 2
    expand: int = 2


class _Out:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _TinyBlock(nn.Module):
    """Gated diagonal linear recurrence (no softmax over positions, fixed-size state)."""

    def __init__(self, d: int, expand: int) -> None:
        super().__init__()
        e = d * expand
        self.e = e
        self.norm = _RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * e)
        self.gate_proj = nn.Linear(d, e)
        self.out_proj = nn.Linear(e, d)
        self.norm2 = _RMSNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        a = torch.sigmoid(self.gate_proj(self.norm(x)) + 2.0)  # slow decay at init
        if h is None:
            h = torch.zeros_like(u[:, 0])
        outs = []
        for t in range(u.shape[1]):
            h = a[:, t] * h + (1 - a[:, t]) * u[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1) * torch.nn.functional.silu(z)
        x = x + self.out_proj(y)
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x)))), h


class TinyRecurrentLM(nn.Module):
    """HF-shaped tiny LM: ``model(input_ids, output_hidden_states=True).hidden_states``, plus a
    functional state API (``prefill`` / ``expand_state`` / ``extend``) used by the cache fork."""

    _accepts_mask = False

    def __init__(self, config: TinyConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyConfig()
        c = self.config
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(_TinyBlock(c.hidden_size, c.expand) for _ in range(c.num_hidden_layers))
        self.norm_f = _RMSNorm(c.hidden_size)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def _run(self, input_ids, state=None):
        x = self.embed(input_ids)
        hs, new_state = [x], []
        for i, layer in enumerate(self.layers):
            h0 = None if state is None else state[i]
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x, h = checkpoint(layer, x, h0, use_reentrant=False)
            else:
                x, h = layer(x, h0)
            hs.append(x)
            new_state.append(h)
        hs[-1] = self.norm_f(x)
        return hs, new_state

    def forward(self, input_ids, output_hidden_states=False, **_):
        hs, _ = self._run(input_ids)
        return _Out(tuple(hs))

    def prefill(self, input_ids):
        hs, state = self._run(input_ids)
        return hs[-1], state

    @staticmethod
    def expand_state(state, n: int):
        return [h.expand(n, -1) for h in state]

    def extend(self, input_ids, state):
        hs, new_state = self._run(input_ids, state)
        return hs[-1], new_state


def _tiny_groups(n_layers: int) -> tuple[LoraGroup, ...]:
    return (("mixer", r"(in_proj|gate_proj|out_proj)$", 3 * n_layers), ("mlp", r"(up|down)$", 2 * n_layers))


def tiny_spec(n_layers: int = 2, d: int = 32) -> BackboneSpec:
    return BackboneSpec(name="tiny", family="tiny", hf_id="random", params=0, n_layers=n_layers, d_model=d,
                        lora_groups=_tiny_groups(n_layers), capture="hidden_states", attention_free=True)


TINY_SPEC = tiny_spec()


def tiny_backbone(seed: int = 0, hidden: int = 32, layers: int = 2, fork_mode: str = "recompute") -> HiddenStateBackbone:
    torch.manual_seed(seed)
    model = TinyRecurrentLM(TinyConfig(hidden_size=hidden, num_hidden_layers=layers))
    return HiddenStateBackbone(model, spec=tiny_spec(layers, hidden), fork_mode=fork_mode)


def tiny_hf_backbone(family: str, seed: int = 0, hidden: int = 32, fork_mode: str = "recompute") -> HiddenStateBackbone:
    """Tiny randomly initialised HF models (CPU torch fallbacks; no kernels needed).

    ``qwen3.5``: stock hybrid layout (3 GDN : 1 attention). ``q9af``: the same class with
    every layer ``linear_attention`` (the Q9-AF target layout). ``mamba``: HF Mamba-1
    (stand-in for the hook/HF path; Mamba-3 lives in ``mamba_ssm``).
    """
    import transformers as tf

    torch.manual_seed(seed)
    if family in ("qwen3.5", "q9af"):
        kw = dict(vocab_size=260, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=4,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                  linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
        if family == "q9af":
            kw["layer_types"] = ["linear_attention"] * 4
            groups = (("gdn", _GDN, 4 * 3), ("mlp", _QMLP, 4 * 3))
        else:
            groups = (("gdn", _GDN, 3 * 3), ("attn", _QATTN, 1 * 4), ("mlp", _QMLP, 4 * 3))
        model = tf.Qwen3_5ForCausalLM(tf.Qwen3_5TextConfig(**kw))
        spec = BackboneSpec(name=f"tiny-{family}", family="qwen3.5", hf_id="random", params=0, n_layers=4,
                            d_model=hidden, lora_groups=groups, capture="hidden_states",
                            attention_free=family == "q9af")
        return HiddenStateBackbone(model, spec=spec, fork_mode=fork_mode)
    if family == "mamba":
        cfg = tf.MambaConfig(vocab_size=260, hidden_size=hidden, state_size=4, num_hidden_layers=2, expand=2)
        model = tf.MambaForCausalLM(cfg)
        spec = BackboneSpec(name="tiny-mamba", family="mamba", hf_id="random", params=0, n_layers=2,
                            d_model=hidden, lora_groups=(("mixer", r"mixer\.(in_proj|out_proj|x_proj)$", 2 * 3),),
                            capture="backbone.norm_f", attention_free=True)
        return HiddenStateBackbone(model, capture="backbone.norm_f", spec=spec, fork_mode=fork_mode)
    if family == "rwkv7":
        # the bundled HF RWKV-7 code of RWKV/RWKV7-G1k-13.3B-20260930 (vendored, pinned), pure-torch "chunked" WKV
        from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
        from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7Model

        cfg = Rwkv7Config(vocab_size=260, hidden_size=hidden, num_heads=hidden // 8, head_dim=8, num_hidden_layers=2,
                          intermediate_size=2 * hidden, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                          v_low_rank_dim=8)
        model = Rwkv7Model(cfg)
        with torch.no_grad():  # the release init zeroes most mixing parameters; randomise so the test is not vacuous
            for p in model.parameters():
                p.normal_(0.0, 0.1)
        full = BACKBONES["rwkv7-13.3b"]
        groups = tuple((g, rx, cnt * 2 // full.n_layers) for g, rx, cnt in full.lora_groups)  # the real regexes
        spec = BackboneSpec(name="tiny-rwkv7", family="rwkv7", hf_id="random", params=0, n_layers=2, d_model=hidden,
                            lora_groups=groups, capture="hidden_states", attention_free=True)
        return HiddenStateBackbone(model, spec=spec, fork_mode=fork_mode)
    raise ValueError(family)


#: RWKV/RWKV7-G1k-13.3B-20260930 at this revision; the bundled code is vendored in third_party/rwkv7_hf
RWKV7_REPO = "RWKV/RWKV7-G1k-13.3B-20260930"
RWKV7_REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
#: README "Training context: 25600 tokens" (source checkpoint ...-ctx25600.pth); config.json has no max_position_embeddings
RWKV7_TRAINED_CTX = 25600


def load_rwkv7_hf_backbone(model_dir: str, device: str = "cuda", dtype: str = "bfloat16",
                           fork_mode: str = "cache") -> HiddenStateBackbone:
    """RWKV7-G1k-13.3B for m2 training: the vendored bundled classes (no trust_remote_code), after checking that
    the snapshot's own modeling/config/tokenizer code is byte-identical to the vendored copy. The backbone is the
    base ``Rwkv7Model`` (no 65,536-wide LM head over every position)."""
    from third_party.rwkv7_hf import verify_snapshot_code
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    verify_snapshot_code(model_dir)
    lm = Rwkv7ForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype), device_map={"": device})
    return HiddenStateBackbone(lm.rwkv7, spec=BACKBONES["rwkv7-13.3b"], pad_id=0, fork_mode=fork_mode)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def group_matches(model: nn.Module, groups: Sequence[LoraGroup]) -> dict[str, list[str]]:
    """Linear modules matched by each LoRA group (by qualified name)."""
    out: dict[str, list[str]] = {}
    for gname, rx, _ in groups:
        pat = re.compile(rx)
        out[gname] = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]
    return out


"""Backbone-agnostic hidden-state interface.

A head never touches a backbone directly. It sees token features ``[B, T, D]``
produced by ``HiddenStateBackbone.features``, which wraps any ``nn.Module`` LM:

* HF models that accept ``output_hidden_states`` (Qwen3.5 hybrid, a Q9-AF
  checkpoint saved with all-``linear_attention`` ``layer_types``, HF Mamba/Mamba-2,
  RWKV7 in HF format once Transformers >= 5.15 ships it) -> ``capture="hidden_states"``.
* Anything else (``mamba_ssm`` Mamba-3 ``MambaLMHeadModel``, ``fla`` RWKV7) ->
  ``capture="<module name>"``: a forward hook on the final norm captures features,
  and the LM head is never run (no vocab-sized logits, which matters at 248k vocab).

Right padding only. Recurrent backbones without a padding mask are exact on
right-padded rows because outputs at valid positions never depend on later
positions.

The final-state fork is implemented by recompute (prefix + continuation per
option). That is backbone-agnostic and exact; a cache-copy fork is an
optimisation whose per-option state bytes are priced in ``train/estimate.py``.

``BACKBONES`` records the three m1 candidates. Dimensions marked
``verified=False`` are planning inputs, to be confirmed against ``config.json``
at m1 qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str  # qwen3.5 | rwkv7 | mamba3 | tiny
    hf_id: str
    params: float  # text-path parameters
    n_layers: int
    d_model: int
    lora_targets: str  # regex over module names
    capture: str  # "hidden_states" or a module name for a forward hook
    attention_free: bool
    #: per-layer projections that LoRA targets: list of (in, out) per layer
    lora_shapes: tuple[tuple[int, int], ...] = ()
    #: recurrent state scalars per layer (for fork/state pricing)
    state_scalars_per_layer: int = 0
    state_dtype_bytes: int = 4
    mfu_assumed: float = 0.30
    verified: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


_QWEN_LORA = r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"

BACKBONES: dict[str, BackboneSpec] = {
    "q9af": BackboneSpec(
        name="q9af",
        family="qwen3.5",
        hf_id="Qwen/Qwen3.5-9B (8 full-attention layers distilled to GDN at m1)",
        params=8.5e9,
        n_layers=32,
        d_model=4096,
        lora_targets=_QWEN_LORA,
        capture="hidden_states",
        attention_free=True,
        # GDN: in_proj_qkv 4096->8192, in_proj_z 4096->4096, out_proj 4096->4096;
        # MLP gate/up 4096->I, down I->4096 with I = 12288 (unverified)
        lora_shapes=((4096, 8192), (4096, 4096), (4096, 4096), (4096, 12288), (4096, 12288), (12288, 4096)),
        state_scalars_per_layer=32 * 128 * 128 + 33_000,
        state_dtype_bytes=4,  # mamba_ssm_dtype float32
        mfu_assumed=0.35,
        verified={"n_layers": True, "d_model": True, "gdn_heads": True, "intermediate_size": False, "params": False},
        notes="B-review §1 row 1/§8A; GDN 32 value heads x 128, 16 key heads x 128; text params ~8.5e9 (estimate)",
    ),
    "rwkv7-13.3b": BackboneSpec(
        name="rwkv7-13.3b",
        family="rwkv7",
        hf_id="RWKV/RWKV7-G1k-13.3B-20260930",
        params=13.27e9,
        n_layers=61,
        d_model=4096,
        lora_targets=r"(attn\.(r_proj|k_proj|v_proj|o_proj)|ffn\.(key|value))$",
        capture="hidden_states",
        attention_free=True,
        lora_shapes=((4096, 4096),) * 4 + ((4096, 16384), (16384, 4096)),
        state_scalars_per_layer=64 * 64 * 64 + 2 * 4096,
        state_dtype_bytes=4,
        mfu_assumed=0.25,
        verified={"n_layers": True, "d_model": True, "heads": True, "params": True, "module_names": False},
        notes="reviews/01 row 5; module names for LoRA are unverified (fla RWKV7 naming); bundled WKV7 training throughput unverified -> low MFU",
    ),
    "mamba3-1.5b": BackboneSpec(
        name="mamba3-1.5b",
        family="mamba3",
        hf_id="state-spaces/mamba3-siso-1.5b",
        params=1.5e9,
        n_layers=48,
        d_model=2048,
        lora_targets=r"mixer\.(in_proj|out_proj)$",
        capture="backbone.norm_f",
        attention_free=True,
        lora_shapes=((2048, 2 * 4096 + 2 * 128 + 64), (4096, 2048)),
        state_scalars_per_layer=64 * 64 * 128,
        state_dtype_bytes=4,
        mfu_assumed=0.30,
        verified={"params": True, "state_size": True, "n_layers": False, "d_model": False},
        notes="reviews/01: 1.50B params, ctx 2048, state 128; layer count / width unverified; mamba_ssm module, hook capture",
    ),
}


class HiddenStateBackbone(nn.Module):
    """Uniform ``features(input_ids, attention_mask) -> [B, T, D]`` over any LM module."""

    def __init__(self, model: nn.Module, capture: str = "hidden_states", layer: int = -1,
                 spec: BackboneSpec | None = None, pad_id: int = 0) -> None:
        super().__init__()
        self.model = model
        self.capture = capture
        self.layer = layer
        self.spec = spec
        self.pad_id = pad_id
        self._captured: torch.Tensor | None = None
        self._hook = None
        if capture != "hidden_states":
            target = dict(model.named_modules())[capture]
            self._hook = target.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._captured = output[0] if isinstance(output, tuple) else output

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

    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.capture == "hidden_states":
            kwargs: dict[str, Any] = {"input_ids": input_ids, "output_hidden_states": True}
            if attention_mask is not None and getattr(self.model, "_accepts_mask", True):
                kwargs["attention_mask"] = attention_mask
            try:
                kwargs["use_cache"] = False
                out = self.model(**kwargs)
            except TypeError:
                kwargs.pop("use_cache", None)
                kwargs.pop("attention_mask", None)
                out = self.model(**kwargs)
            return out.hidden_states[self.layer]
        self._captured = None
        self.model(input_ids)
        if self._captured is None:
            raise RuntimeError(f"hook on {self.capture} did not fire")
        return self._captured

    def forward(self, input_ids, attention_mask=None):  # pragma: no cover - alias
        return self.features(input_ids, attention_mask)


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence. Returns (ids [B,T], mask [B,T] bool)."""
    t = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), t), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = True
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
        self.norm = _RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * e)
        self.gate_proj = nn.Linear(d, e)
        self.out_proj = nn.Linear(e, d)
        self.norm2 = _RMSNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        a = torch.sigmoid(self.gate_proj(self.norm(x)) + 2.0)  # slow decay at init
        h = torch.zeros_like(u[:, 0])
        outs = []
        for t in range(u.shape[1]):
            h = a[:, t] * h + (1 - a[:, t]) * u[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1) * torch.nn.functional.silu(z)
        x = x + self.out_proj(y)
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x))))


class TinyRecurrentLM(nn.Module):
    """HF-shaped tiny LM: ``model(input_ids, output_hidden_states=True).hidden_states``."""

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

    def forward(self, input_ids, output_hidden_states=False, **_):
        x = self.embed(input_ids)
        hs = [x]
        for layer in self.layers:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            hs.append(x)
        hs[-1] = self.norm_f(x)
        return _Out(tuple(hs))


TINY_SPEC = BackboneSpec(
    name="tiny", family="tiny", hf_id="random", params=0, n_layers=2, d_model=32,
    lora_targets=r"(in_proj|gate_proj|out_proj|up|down)$", capture="hidden_states", attention_free=True,
)


def tiny_backbone(seed: int = 0, hidden: int = 32, layers: int = 2) -> HiddenStateBackbone:
    torch.manual_seed(seed)
    model = TinyRecurrentLM(TinyConfig(hidden_size=hidden, num_hidden_layers=layers))
    return HiddenStateBackbone(model, spec=TINY_SPEC)


def tiny_hf_backbone(family: str, seed: int = 0, hidden: int = 32) -> HiddenStateBackbone:
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
        cfg = tf.Qwen3_5TextConfig(**kw)
        model = tf.Qwen3_5ForCausalLM(cfg)
        spec = BackboneSpec(name=f"tiny-{family}", family="qwen3.5", hf_id="random", params=0, n_layers=4,
                            d_model=hidden, lora_targets=_QWEN_LORA, capture="hidden_states",
                            attention_free=family == "q9af")
        return HiddenStateBackbone(model, spec=spec)
    if family == "mamba":
        cfg = tf.MambaConfig(vocab_size=260, hidden_size=hidden, state_size=4, num_hidden_layers=2, expand=2)
        model = tf.MambaForCausalLM(cfg)
        spec = BackboneSpec(name="tiny-mamba", family="mamba", hf_id="random", params=0, n_layers=2,
                            d_model=hidden, lora_targets=r"mixer\.(in_proj|out_proj|x_proj)$",
                            capture="backbone.norm_f", attention_free=True)
        return HiddenStateBackbone(model, capture="backbone.norm_f", spec=spec)
    raise ValueError(family)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


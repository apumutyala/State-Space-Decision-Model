"""Per-block activation checkpointing that works WITH the cache fork, for the vendored RWKV-7 (``third_party/rwkv7_hf``)
and for HF Qwen3.5-family decoders: Q9-AF (every layer a Gated DeltaNet) and its hybrids (a partial conversion that
keeps some attention layers).

    from heads.act_ckpt import enable_block_checkpointing
    enable_block_checkpointing(backbone.model)        # TrainConfig.act_ckpt (default off)

Why not HF's ``gradient_checkpointing_enable``: in training it forces ``use_cache=False`` / ``past_key_values=None``
(the replay would re-read a state the first pass already advanced), so the cache fork gets no state and
``HiddenStateBackbone._hf_fork`` refuses to run. Here the recurrent state is moved OUT of the checkpointed region.

RWKV-7 (``Rwkv7Model.blocks``): each block's forward becomes

    att_shift, ffn_shift, wkv = state.read(i)                       # outside: plain tensors in
    h, v_first, att_shift, ffn_shift, wkv = checkpoint(block, h, v_first, att_shift, ffn_shift, wkv)
    state.write(i, att_shift, ffn_shift, wkv)                       # outside: the cache is updated once

Qwen3.5 / Q9-AF (``Qwen3_5DecoderLayer``, PREMORTEM blocker #4): the same pattern over the HF cache layer of block i,

    GDN block:        conv_tail, S = cache.layers[i].conv_states[0], .recurrent_states[0]     # outside
                      h, conv_tail, S = checkpoint(block, h, conv_tail, S)                      # private cache inside
                      cache.layers[i].conv_states[0], .recurrent_states[0] = conv_tail, S      # outside, once
    attention block:  K, V = cache.layers[i].keys, .values  ->  checkpoint  ->  cache.layers[i].keys, .values = K', V'

The block's own ``forward`` runs inside on a PRIVATE copy of the cache whose slot i is a fresh one-layer holder built
from the incoming tensors (the other slots are never touched by block i), so the replay in backward sees exactly the
inputs of the first pass and never writes into the real cache. The holder's GDN updates are functional (assignment,
the ``heads.backbone._assign_not_copy`` semantics; the first write of a fresh layer is HF's own lazy init), so the
values equal the un-checkpointed cache-fork path bit for bit. Only each block's inputs (the residual stream, the
incoming conv tail / recurrent state or K/V) are kept for backward; everything inside a block (the GDN chunk
intermediates, the 3x-wide MLP activations, LoRA intermediates, attention scores) is recomputed
(``torch.utils.checkpoint``, non-reentrant, RNG state preserved). Gradients equal the un-checkpointed ones
(``tests/test_q9af_act_ckpt.py``; RWKV-7: ``tests/test_cell_perf.py``).

Active only while autograd records (``torch.is_grad_enabled()``): evaluation under ``no_grad`` runs the block as is.
Model files are not modified: the instance attribute ``forward`` shadows the class method and is removed again by
``disable_block_checkpointing``. A cache layer of an unknown kind (offloaded, past-recording, a fused
linear+attention layer) is refused (``TypeError``), never silently mishandled.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.checkpoint import checkpoint

_ATTR = "_ssdm_ckpt_orig_forward"
_KIND = "_ssdm_ckpt_kind"


class _OneLayerState:
    """Duck-typed ``Rwkv7Cache`` for one block: ``read`` returns the given tensors, ``write`` records the new ones."""

    __slots__ = ("vals", "out")

    def __init__(self, att_shift, ffn_shift, wkv):
        self.vals = (att_shift, ffn_shift, wkv)
        self.out = None

    def read(self, layer_idx):
        return self.vals

    def write(self, layer_idx, att_shift, ffn_shift, wkv):
        self.out = (att_shift, ffn_shift, wkv)


# ------------------------------------------------------------------------------------------------ backbone dispatch
def _rwkv7_blocks(model):
    blocks = getattr(model, "blocks", None)
    if blocks is None and hasattr(model, "rwkv7"):
        blocks = model.rwkv7.blocks
    return blocks


def _hf_text_model(model):
    """The HF Qwen3.5 text model inside ``model`` (the text model itself, or a ``*ForCausalLM`` holding it)."""
    for m in (model, getattr(model, "model", None)):
        cfg = getattr(m, "config", None)
        layers = getattr(m, "layers", None)
        if m is None or layers is None or cfg is None or not hasattr(cfg, "layer_types"):
            continue
        if all(hasattr(l, "mlp") and hasattr(l, "input_layernorm") and
               (hasattr(l, "linear_attn") or hasattr(l, "self_attn")) for l in layers):
            return m
    return None


def _kind(model):
    if _rwkv7_blocks(model) is not None:
        return "rwkv7"
    if _hf_text_model(model) is not None:
        return "qwen3.5"
    return None


def _blocks(model):
    blocks = _rwkv7_blocks(model)
    if blocks is not None:
        return blocks
    text = _hf_text_model(model)
    if text is not None:
        return text.layers
    raise ValueError("block checkpointing supports the vendored RWKV-7 (Rwkv7Model.blocks) and HF Qwen3.5-family "
                     "decoders (Q9-AF / hybrids: text_model.layers of Qwen3_5DecoderLayer) only")


# ------------------------------------------------------------------------------------------------ RWKV-7
def _wrap(block):
    orig = block.forward  # the bound class method

    def forward(hidden_states, v_first, state, keep=None, cu_seq_lens=None):
        if not torch.is_grad_enabled():
            return orig(hidden_states, v_first, state, keep, cu_seq_lens)
        has_state = state is not None
        att, ffn, wkv = state.read(block.layer_id) if has_state else (None, None, None)

        def run(h, vf, a, f, w):
            st = _OneLayerState(a, f, w) if has_state else None
            h2, vf2, _ = orig(h, vf, st, keep, cu_seq_lens)
            return (h2, vf2) + (st.out if has_state else ())

        outs = checkpoint(run, hidden_states, v_first, att, ffn, wkv, use_reentrant=False)
        if has_state:
            state.write(block.layer_id, *outs[2:])
        return outs[0], outs[1], state

    return forward


# ------------------------------------------------------------------------------------------------ Qwen3.5 / Q9-AF
def _private_linear_layer_cls():
    """``LinearAttentionLayer`` with functional updates once a state exists (``_assign_not_copy`` semantics); the
    first write of a fresh layer is HF's own lazy init + copy (what the un-checkpointed prefix call does)."""
    from transformers.cache_utils import LinearAttentionLayer

    class _PrivateLinearLayer(LinearAttentionLayer):
        def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kw):
            if not self.is_conv_states_initialized[state_idx] or not self.has_previous_state[state_idx]:
                return super().update_conv_state(conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size,
                                                 **kw)
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
            self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx]:]
            return full

        def update_recurrent_state(self, recurrent_states, state_idx=0, **kw):
            if not self.is_recurrent_states_initialized[state_idx]:
                return super().update_recurrent_state(recurrent_states, state_idx=state_idx, **kw)
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

    return _PrivateLinearLayer


_LIN_FLAGS = ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state",
              "conv_kernel_size")


class _HFSlot:
    """Block i's slice of an HF cache: read (tensors in), a private holder for the checkpointed run, write (once)."""

    def __init__(self, cache, idx: int):
        from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

        self.cache, self.idx = cache, idx
        if getattr(cache, "offloading", False):
            raise TypeError("block checkpointing: offloaded caches are not supported")
        layer = cache.layers[idx] if idx < len(cache.layers) else None
        if layer is None:
            raise TypeError(f"block checkpointing: the cache has no layer {idx} (a lazily grown cache is not supported)")
        self.layer = layer
        if isinstance(layer, LinearAttentionCacheLayerMixin) and isinstance(layer, DynamicLayer):
            raise TypeError("block checkpointing: fused linear+attention cache layers are not supported")
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            if layer.record_past or getattr(layer, "number_of_states", 1) != 1:
                raise TypeError("block checkpointing: past-recording / multi-state linear cache layers are not supported")
            self.kind = "linear"
            self.flags = {f: dict(getattr(layer, f)) for f in _LIN_FLAGS}
            self.present = tuple(t is not None for t in (layer.conv_states[0], layer.recurrent_states[0]))
            self.inputs = tuple(t for t in (layer.conv_states[0], layer.recurrent_states[0]) if t is not None)
            self.meta = (layer.device, layer.dtype)
        elif type(layer) is DynamicLayer:
            self.kind = "attention"
            self.init = bool(layer.is_initialized)
            self.inputs = (layer.keys, layer.values) if self.init else ()
            self.meta = (getattr(layer, "device", None), getattr(layer, "dtype", None))
        else:
            raise TypeError(f"block checkpointing: unsupported cache layer {type(layer).__name__}")

    def holder(self, tensors):
        """A shallow copy of the cache whose slot ``idx`` is a fresh private layer holding ``tensors``."""
        from transformers.cache_utils import DynamicLayer

        if self.kind == "linear":
            priv = _private_linear_layer_cls()(number_of_states=1)
            for f, v in self.flags.items():
                setattr(priv, f, dict(v))
            it = iter(tensors)
            # a clone: the single-token decode path (causal_conv1d_update) updates the conv tail IN PLACE, which must
            # never hit the checkpoint's saved input (the un-checkpointed path mutates the cache tensor the same way)
            priv.conv_states[0] = next(it).clone() if self.present[0] else None
            priv.recurrent_states[0] = next(it) if self.present[1] else None
            priv.device, priv.dtype = self.meta
        else:
            priv = DynamicLayer()
            if self.init:
                priv.keys, priv.values = tensors
                priv.is_initialized = True
                priv.device, priv.dtype = self.meta
        h = copy.copy(self.cache)
        h.layers = list(self.cache.layers)
        h.layers[self.idx] = priv
        return h, priv

    @staticmethod
    def outputs(kind, priv):
        if kind == "linear":
            return priv.conv_states[0], priv.recurrent_states[0]
        return (priv.keys, priv.values) if priv.is_initialized else (None, None)

    def write(self, priv, outs):
        """Bind the checkpoint's output tensors (the same objects the first pass produced) into the real cache."""
        layer = self.layer
        if self.kind == "linear":
            for f in _LIN_FLAGS:
                setattr(layer, f, dict(getattr(priv, f)))
            layer.conv_states[0], layer.recurrent_states[0] = outs
            if layer.device is None:
                layer.device, layer.dtype = priv.device, priv.dtype
        else:
            if outs[0] is None:
                return
            layer.keys, layer.values = outs
            if not layer.is_initialized:
                layer.dtype, layer.device = outs[0].dtype, outs[0].device
                layer.is_initialized = True


def _wrap_hf(layer, idx: int):
    orig = layer.forward  # the bound class method (Qwen3_5DecoderLayer.forward)

    def forward(hidden_states, *args, past_key_values=None, **kwargs):
        if not torch.is_grad_enabled():
            return orig(hidden_states, *args, past_key_values=past_key_values, **kwargs)
        if past_key_values is None:  # no cache (plain features / recompute fork): checkpoint the block as is
            return checkpoint(lambda h: orig(h, *args, past_key_values=None, **kwargs), hidden_states,
                              use_reentrant=False)
        slot = _HFSlot(past_key_values, idx)
        box = {}

        def run(h, *state):
            holder, priv = slot.holder(state)
            box.setdefault("priv", priv)  # the FIRST pass's holder (the backward replay makes its own, discarded)
            out = orig(h, *args, past_key_values=holder, **kwargs)
            return (out,) + _HFSlot.outputs(slot.kind, priv)

        outs = checkpoint(run, hidden_states, *slot.inputs, use_reentrant=False)
        slot.write(box.pop("priv"), outs[1:])  # the forward's holder is not kept alive until backward (review nb 2)
        return outs[0]

    return forward


# ------------------------------------------------------------------------------------------------ public API
def enable_block_checkpointing(model) -> int:
    """Idempotent. Returns the number of checkpointed blocks."""
    kind = _kind(model)
    blocks = _blocks(model)
    n = 0
    for i, block in enumerate(blocks):
        if _ATTR not in block.__dict__:
            block.__dict__[_ATTR] = True
            block.__dict__[_KIND] = kind
            block.forward = _wrap(block) if kind == "rwkv7" else _wrap_hf(block, i)
        n += 1
    return n


def disable_block_checkpointing(model) -> int:
    n = 0
    for block in _blocks(model):
        if block.__dict__.pop(_ATTR, None):
            block.__dict__.pop(_KIND, None)
            del block.forward
            n += 1
    return n


def is_enabled(model) -> bool:
    try:
        return any(_ATTR in b.__dict__ for b in _blocks(model))
    except ValueError:
        return False


def supported(model) -> str | None:
    """``"rwkv7"`` / ``"qwen3.5"`` when ``enable_block_checkpointing`` supports ``model``, else None."""
    return _kind(model)

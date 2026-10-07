"""Per-block activation checkpointing for the vendored RWKV-7 (``third_party/rwkv7_hf``) that works WITH the cache fork.

    from heads.act_ckpt import enable_block_checkpointing
    enable_block_checkpointing(backbone.model)        # TrainConfig.act_ckpt (default off)

Why not HF's ``gradient_checkpointing_enable``: ``Rwkv7Model.forward`` then forces ``use_cache=False`` in training (the
replay would re-read a state the first pass already advanced), so the cache fork gets no state and
``HiddenStateBackbone._hf_fork`` refuses to run. Here the recurrent state is moved OUT of the checkpointed region: each
block's forward becomes

    att_shift, ffn_shift, wkv = state.read(i)                       # outside: plain tensors in
    h, v_first, att_shift, ffn_shift, wkv = checkpoint(block, h, v_first, att_shift, ffn_shift, wkv)
    state.write(i, att_shift, ffn_shift, wkv)                       # outside: the cache is updated once

with the block's own ``forward`` run inside on a private one-layer state holder, so the replay in backward sees exactly
the inputs of the first pass (``torch.utils.checkpoint``, non-reentrant, RNG state preserved). Only each block's inputs
(the residual stream, ``v_first`` and the incoming state) are kept for backward; everything inside a block (the chunked
WKV's fp32 intermediates, the FFN's 4x-wide activations, LoRA intermediates) is recomputed. Forward values are
unchanged bit for bit; gradients equal the un-checkpointed ones (``tests/test_cell_perf.py``).

Active only while autograd records (``torch.is_grad_enabled()``): evaluation under ``no_grad`` runs the block as is.
The vendored files are not modified: the instance attribute ``forward`` shadows the class method and is removed again by
``disable_block_checkpointing``.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

_ATTR = "_ssdm_ckpt_orig_forward"


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


def _blocks(model):
    blocks = getattr(model, "blocks", None)
    if blocks is None and hasattr(model, "rwkv7"):
        blocks = model.rwkv7.blocks
    if blocks is None:
        raise ValueError("block checkpointing supports the vendored RWKV-7 (Rwkv7Model.blocks) only")
    return blocks


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


def enable_block_checkpointing(model) -> int:
    """Idempotent. Returns the number of checkpointed blocks."""
    n = 0
    for block in _blocks(model):
        if _ATTR not in block.__dict__:
            block.__dict__[_ATTR] = True
            block.forward = _wrap(block)
        n += 1
    return n


def disable_block_checkpointing(model) -> int:
    n = 0
    for block in _blocks(model):
        if block.__dict__.pop(_ATTR, None):
            del block.forward
            n += 1
    return n


def is_enabled(model) -> bool:
    try:
        return any(_ATTR in b.__dict__ for b in _blocks(model))
    except ValueError:
        return False

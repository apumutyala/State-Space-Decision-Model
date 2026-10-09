"""RWKV07C readout plug-in: OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c (attention-free RWKV hxa07c conversion of
Qwen3-30B-A3B-Instruct; 48 RWKV-7-style layers, Qwen3-MoE FFN with 128 experts / 8 active; backbone scout).

The repo's bundled code is vendored unmodified and sha-pinned in ``third_party/rwkv07c_hf`` (no
``trust_remote_code``). It was written against transformers 4.5x; ``install_compat`` bridges 5.17 from outside, the
files unchanged:
  * config defaults 5.x no longer sets (``pad_token_id`` ...), and ``ROPE_INIT_FUNCTIONS["default"]`` (this model
    DOES apply RoPE inside its recurrent layers, ``use_rope: true``; the 4.x default inv_freq, rope_theta 1e7, dim 128);
  * the per-expert checkpoint (``mlp.experts.<e>.{gate,up,down}_proj.weight``) vs 5.x's fused experts
    (``mlp.experts.gate_up_proj`` / ``down_proj``): transformers' OWN qwen3_moe checkpoint conversion is registered for
    model_type ``rwkv07c`` (``register_checkpoint_conversion_mapping``), so the merge is the library's, not ours;
  * fla 0.5.2 ``fused_recurrent_rwkv7`` raises on any ``head_first`` kwarg (removed; inputs are [B, T, H, K]); the code
    passes ``head_first=False``, i.e. exactly the layout 0.5.2 assumes, so the module global is wrapped to drop a
    ``head_first=False`` (``True`` still raises);
  * ``_tied_weights_keys`` list -> {} (``tie_word_embeddings: false``);
  * ``create_causal_mask``: 5.x renamed its ``input_embeds`` kwarg, so the call raises. With no attention layer the
    mask's only consumer is the RWKV layer's ``v * mask[..., -T:]`` (padding zeroing), and for unpadded inputs (the
    readout never pads or passes ``attention_mask``) the 4.x mask is None or all-ones, i.e. a no-op; the module global
    is replaced by one that returns None and refuses a real padding mask.
Every weight must load (missing / unexpected / mismatched keys refuse), and every layer must be ``RWKV07CAttention``.
Readout and fork: as ``decide.backends.rwkv6qwen2`` (``RWKV6StateOps`` with ``logits_to_keep``), template
``jrt-chatml-nothink-v1`` (the bundled chat_template.json generation prompt is ``<|im_start|>assistant\\n``, no think block).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..untrusted import UntrustedEncoder
from .hf_readout import ModelPlugin, _hash_files, _pkg_versions, _snapshot_of
from .rwkv6qwen2 import CONFIG_COMPAT_DEFAULTS, RWKV6StateOps, _default_rope

#: per sequence 48 x 32 x 128 x 128 fp32 = 101 MB of state; 16 rows per packed call as the RWKV6-Qwen2 screens
FORK_MAX_ROWS = 16


def _drop_head_first(fn):
    def wrapped(*args, head_first=None, **kw):
        if head_first:
            raise ValueError("head_first=True layout is not supported by fla 0.5.2")
        return fn(*args, **kw)

    wrapped.__wrapped__ = fn
    return wrapped


def _no_causal_mask(*args, attention_mask=None, **kw):
    if attention_mask is not None and not bool((attention_mask == 1).all()):
        raise ValueError("padded attention_mask: not supported by the attention-free readout shim")
    return None


def install_compat() -> None:
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    import third_party.rwkv07c_hf.modeling_rwkv07c as M

    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
    if get_checkpoint_conversion_mapping("rwkv07c") is None:
        register_checkpoint_conversion_mapping("rwkv07c", get_checkpoint_conversion_mapping("qwen3_moe"))
    if isinstance(M.RWKV07CForCausalLM.__dict__.get("_tied_weights_keys"), list):
        M.RWKV07CForCausalLM._tied_weights_keys = {}
    M.create_causal_mask = _no_causal_mask
    M.create_sliding_window_causal_mask = _no_causal_mask
    fr = getattr(M, "fused_recurrent_rwkv7", None)
    if fr is not None and not hasattr(fr, "__wrapped__"):
        M.fused_recurrent_rwkv7 = _drop_head_first(fr)


def compat_config(cfg):
    install_compat()
    for k, v in CONFIG_COMPAT_DEFAULTS.items():
        if k == "rope_scaling":
            continue  # stored (null) in config.json
        try:
            getattr(cfg, k)
        except AttributeError:
            setattr(cfg, k, v)
    return cfg


def load_model(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16"):
    import torch

    from third_party.rwkv07c_hf.configuration_rwkv07c import RWKV07CConfig
    from third_party.rwkv07c_hf.modeling_rwkv07c import RWKV07CForCausalLM

    config = compat_config(RWKV07CConfig.from_pretrained(model_dir))
    if any(0 <= i < config.num_hidden_layers for i in config.transformer_layers):
        raise RuntimeError(f"transformer_layers {config.transformer_layers}: not attention-free (and the mask shim is wrong)")
    if getattr(config, "tie_word_embeddings", False):
        raise RuntimeError("tied embeddings: install_compat's empty _tied_weights_keys would be wrong")
    model, info = RWKV07CForCausalLM.from_pretrained(model_dir, config=config, dtype=getattr(torch, dtype),
                                                     device_map={"": device}, output_loading_info=True)
    bad = {k: sorted(map(str, v)) for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        raise RuntimeError(f"{model_dir}: weights do not match the vendored model: {json.dumps(bad)[:2000]}")
    layers = {type(layer.self_attn).__name__ for layer in model.model.layers}
    if layers != {"RWKV07CAttention"}:
        raise RuntimeError(f"expected every layer to be RWKV07CAttention, got {sorted(layers)}")
    return model.eval()


def load_rwkv07c(model_dir: str | Path, repo: str, device: str = "cuda", dtype: str = "bfloat16",
                 input_cap: int | None = None) -> ModelPlugin:
    from transformers import AutoTokenizer

    import third_party.rwkv07c_hf as V

    if repo != V.REPO:
        raise RuntimeError(f"{repo!r} is not the pinned RWKV07C repo {V.REPO}")
    model_dir = Path(model_dir)
    code = V.verify_snapshot_code(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir, device, dtype)
    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tok(text, add_special_tokens=False).input_ids
        return untrusted.encode(text)

    def forward(ids, n_last: int):
        return model(input_ids=ids, use_cache=False, logits_to_keep=n_last).logits

    cap = input_cap or V.INPUT_CAP
    return ModelPlugin(
        family="rwkv07c", model_id=repo, snapshot=_snapshot_of(model_dir), encode=encode, forward=forward,
        special_ids=untrusted.control_ids,
        files={"weights": _hash_files(model_dir, ["*.safetensors"]), "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
               "config": _hash_files(model_dir, ["config.json"]), "code": V._sha(V.HERE / "modeling_rwkv07c.py")},
        max_context=cap, dtype=dtype, device=device,
        kernel_stack={**_pkg_versions("transformers", "torch", "flash-linear-attention", "fla-core", "triton"),
                      "rwkv7": "fla.ops.rwkv7.fused_recurrent_rwkv7 (head_first kwarg dropped)",
                      "moe": "transformers qwen3_moe fused experts via the qwen3_moe checkpoint conversion"},
        state_ops=RWKV6StateOps(model, logits_kw="logits_to_keep"),
        notes={"model_class": type(model).__name__, "layers": {"RWKV07CAttention": len(model.model.layers)},
               "attention_free": True, "input_cap": cap, "parent": "Qwen3-30B-A3B-Instruct (model card)",
               "revision": V.REVISION, "vendored_code": code},
    )

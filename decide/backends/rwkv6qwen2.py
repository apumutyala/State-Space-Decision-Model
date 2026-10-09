"""RWKV6-Qwen2 readout plug-in (attention-free RADLADS conversions of Qwen2.5-32B-Instruct / QwQ-32B; backbone scout).

The model code is the repos' own bundled code, vendored unmodified and sha-pinned in ``third_party/rwkv6qwen2_hf``
(no ``trust_remote_code``). Scoring is the stock LM-logit readout (``HFReadoutBackend``: sum log p(' ' + option id +
terminator) after a ChatML JRT prompt), the same readout as the Qwen3.5-9B stock arm (job-5cmat), through the
prefix fork (``decide.backends.state_fork``) with ``RWKV6StateOps`` below.

Templates (ChatML; the parents' own chat format):
  ``jrt-chatml-v1``          (decide.prompt) the Qwen3 / QwQ no-think tail ``<think>\\n\\n</think>\\n\\nANSWER:``; for the
                             QwQ conversion, whose chat template opens every assistant turn with ``<think>``.
  ``jrt-chatml-nothink-v1``  (here) the same head, tail ``<|im_start|>assistant\\nANSWER:``; for the Qwen2.5-Instruct
                             conversion (Qwen2.5 has no think block). Kept out of ``decide.prompt.TEMPLATES`` on purpose
                             (that set is pinned by tests); passed to the backend as a Template object.

State (``RWKV6State``): per layer a GLA state [B, 40, 128, 128] fp32 + a token-shift state [B, 1, 5120]; ~168 MB per
sequence for the 32B models, independent of the prefix length. ``update`` copies into the state's own tensors in
place, so every ``extend`` works on a deep copy. Long prefixes are prefilled in ``PREFILL_CHUNK`` pieces (exact for a
recurrent model: each piece continues from the previous piece's state), which bounds the activation peak next to the
~65 GiB of bf16 weights on one H100.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..prompt import Template
from ..untrusted import UntrustedEncoder
from .hf_readout import ModelPlugin, _hash_files, _pkg_versions, _snapshot_of

CHATML_NOTHINK = Template(
    id="jrt-chatml-nothink-v1",
    head="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
    tail="<|im_end|>\n<|im_start|>assistant\nANSWER:",
    terminator="<|im_end|>",
)
PREFILL_CHUNK = 8192
#: rows per packed fork call: each row carries its own ~168 MB state copy (16 x 168 MB = 2.7 GB next to 65 GiB weights)
FORK_MAX_ROWS = 16


#: transformers 4.x PretrainedConfig defaults the vendored (4.43-era) code reads; 5.x no longer sets them. Filled only
#: when absent (config.json carries none of them), never overriding a stored value. Not a code change: the model
#: classes and every pinned file stay byte-identical.
CONFIG_COMPAT_DEFAULTS = {"pad_token_id": None, "rope_scaling": None, "output_attentions": False,
                          "output_hidden_states": False, "use_return_dict": True}


def _default_rope(config, device=None, *args, **kwargs):
    """transformers 4.x ``_compute_default_rope_parameters`` (removed from ROPE_INIT_FUNCTIONS in 5.x). The vendored
    model builds its rotary module unconditionally but applies it only when ``config.use_rope`` (False in both pinned
    configs), so these values never reach a forward pass; computed the 4.x way anyway."""
    import torch

    base = getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def install_compat() -> None:
    """Two 4.x -> 5.x API gaps, filled from outside (the pinned files stay byte-identical):
    ROPE_INIT_FUNCTIONS["default"] (see ``_default_rope``), and ``_tied_weights_keys``, a list in 4.x and a
    {target: source} dict in 5.x. Both pinned configs have ``tie_word_embeddings: false`` (lm_head is its own
    tensor in the checkpoint), so the 5.x equivalent is the empty dict."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from third_party.rwkv6qwen2_hf.modeling_rwkv6qwen2 import RWKV6Qwen2ForCausalLM

    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
    if isinstance(RWKV6Qwen2ForCausalLM.__dict__.get("_tied_weights_keys"), list):
        RWKV6Qwen2ForCausalLM._tied_weights_keys = {}


def compat_config(cfg):
    install_compat()
    for k, v in CONFIG_COMPAT_DEFAULTS.items():
        try:
            getattr(cfg, k)
        except AttributeError:
            setattr(cfg, k, v)
    return cfg


def _batch(state) -> int:
    return state.layer_kv_states[0].shape[0] if state is not None and state.layer_kv_states else 1


def _map_state(state, fn) -> None:
    state.layer_kv_states = [fn(t) for t in state.layer_kv_states]
    state.layer_shift_states = [fn(t) for t in state.layer_shift_states]


class RWKV6StateOps:
    """prefill / extend / select over ``RWKV6State`` (``past_key_values``); see ``decide.backends.state_fork``."""

    def __init__(self, model, prefill_chunk: int = PREFILL_CHUNK):
        self.model = model
        self.prefill_chunk = prefill_chunk

    def _fwd(self, ids, state, n_last):
        out = self.model(input_ids=ids, past_key_values=state, use_cache=True, num_logits_to_keep=n_last)
        return out.logits, out.past_key_values

    def prefill(self, ids):
        state, logits = None, None
        for s in range(0, ids.shape[1], self.prefill_chunk):
            logits, state = self._fwd(ids[:, s: s + self.prefill_chunk], state, 1)
        return logits, state

    def extend(self, state, ids, n_last):
        st = copy.deepcopy(state)
        if ids.shape[0] > 1 and _batch(st) == 1:
            B = ids.shape[0]
            _map_state(st, lambda t: t.repeat_interleave(B, 0))
        return self._fwd(ids, st, n_last)

    def select(self, state, i):
        st = copy.deepcopy(state)
        _map_state(st, lambda t: t[i: i + 1].clone())
        return st


def load_model(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16"):
    """The vendored class from a snapshot dir (config through ``compat_config``); refuses any missing / unexpected /
    mismatched weight, so a renamed tensor can never be silently random-initialised."""
    import torch

    from third_party.rwkv6qwen2_hf.configuration_rwkv6qwen2 import RWKV6Qwen2Config
    from third_party.rwkv6qwen2_hf.modeling_rwkv6qwen2 import RWKV6Qwen2ForCausalLM

    config = compat_config(RWKV6Qwen2Config.from_pretrained(model_dir))
    if getattr(config, "tie_word_embeddings", False):
        raise RuntimeError("tied embeddings: install_compat's empty _tied_weights_keys would be wrong")
    model, info = RWKV6Qwen2ForCausalLM.from_pretrained(model_dir, config=config, dtype=getattr(torch, dtype),
                                                        device_map={"": device}, output_loading_info=True)
    bad = {k: sorted(map(str, v)) for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        raise RuntimeError(f"{model_dir}: weights do not match the vendored model: {json.dumps(bad)[:2000]}")
    return model.eval()


def load_rwkv6qwen2(model_dir: str | Path, repo: str, device: str = "cuda", dtype: str = "bfloat16",
                    input_cap: int | None = None) -> ModelPlugin:
    """A pinned RWKV6-Qwen2 snapshot (``third_party.rwkv6qwen2_hf.REPOS``) as a readout plug-in. Refuses a snapshot whose
    code/config differs from the pins, and any missing / unexpected / mismatched weight (no silent random init)."""
    import torch
    from transformers import AutoTokenizer

    import third_party.rwkv6qwen2_hf as V

    model_dir = Path(model_dir)
    code = V.verify_snapshot_code(model_dir, repo)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir, device, dtype)
    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tok(text, add_special_tokens=False).input_ids
        return untrusted.encode(text)

    def forward(ids, n_last: int):
        return model(input_ids=ids, use_cache=False, num_logits_to_keep=n_last).logits

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = [type(layer.self_attn).__name__ for layer in model.model.layers]
    if set(layers) != {"RWKV6Attention"}:
        raise RuntimeError(f"{repo}: expected every layer to be RWKV6Attention, got {sorted(set(layers))}")
    return ModelPlugin(
        family="rwkv6qwen2",
        model_id=repo,
        snapshot=_snapshot_of(model_dir),
        encode=encode,
        forward=forward,
        special_ids=untrusted.control_ids,
        files={
            "weights": _hash_files(model_dir, ["*.safetensors"]),
            "tokenizer": _hash_files(model_dir, ["tokenizer.json"]),
            "config": _hash_files(model_dir, ["config.json"]),
            "code": V._sha(V.HERE / "modeling_rwkv6qwen2.py"),
        },
        max_context=input_cap or V.REPOS[repo]["input_cap"],
        dtype=dtype,
        device=device,
        kernel_stack={**_pkg_versions("transformers", "torch", "flash-linear-attention", "fla-core", "triton"),
                      "gla": "fla.ops.gla.fused_recurrent_gla"},
        state_ops=RWKV6StateOps(model),
        notes={"model_class": type(model).__name__, "layers": {"RWKV6Attention": len(layers)}, "attention_free": True,
               "input_cap": input_cap or V.REPOS[repo]["input_cap"], "parent": V.REPOS[repo]["parent"],
               "revision": V.REPOS[repo]["revision"], "vendored_code": code, "num_hidden_layers": cfg.get("num_hidden_layers")},
    )

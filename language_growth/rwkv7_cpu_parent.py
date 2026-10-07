"""Isolated, pinned FLA-G1 checkpoint conversion to the vendored CPU RWKV7.

No FLA import, AutoModel, snapshot Python, pickle, network or GPU. The reviewed
FLA 0.5.2 equations match the vendored model in FP32; kernel rounding is different.
This is an exploratory parent dependency, not strong-parent qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("C:/Users/apumu/research/CHLabs/ssdm-data/hf/models--fla-hub--rwkv7-0.1B-g1/"
                "snapshots/7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8")
MODEL_ID = "fla-hub/rwkv7-0.1B-g1"
PINNED_HASHES = {
    "config.json": "dd5da97874e0d4134d8114de3c722b7ba35798cd92870a774166c225b8b20968",
    "model.safetensors": "051b3684ffa896aa842522e5bbc2f355ac00783e12e373a8eb7e4907dd5ead70",
    "rwkv_vocab_v20230424.txt": "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89",
}


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def conversion_spec(c: int, layers: int, vocab: int, head: int,
                    inner: int, w: int, a: int, v: int, g: int) -> dict[str, tuple[str, str, list[int]]]:
    """Source name -> (target name, transform, expected source shape)."""
    spec = {
        "lm_head.weight": ("head.weight", "identity", [vocab, c]),
        "model.embeddings.weight": ("rwkv7.emb.weight", "identity", [vocab, c]),
    }
    for suffix in ("weight", "bias"):
        spec[f"model.norm.{suffix}"] = (f"rwkv7.ln_out.{suffix}", "identity", [c])
    for i in range(layers):
        src, dst = f"model.layers.{i}.", f"rwkv7.blocks.{i}."
        norms = {"attn_norm": "ln1", "ffn_norm": "ln2"}
        if i == 0:
            norms["pre_norm"] = "ln0"
        for left, right in norms.items():
            for suffix in ("weight", "bias"):
                spec[src+left+"."+suffix] = (dst+right+"."+suffix, "identity", [c])
        for left, right in {"r_proj": "receptance", "k_proj": "key",
                            "v_proj": "value", "o_proj": "output"}.items():
            spec[src+"attn."+left+".weight"] = (dst+"att."+right+".weight", "identity", [c, c])
        for name in ("r", "w", "k", "v", "a", "g"):
            spec[src+"attn.x_"+name] = (dst+"att.x_"+name, "identity", [1, 1, c])
        for name in ("k_k", "k_a"):
            spec[src+"attn."+name] = (dst+"att."+name, "channel", [c])
        spec[src+"attn.r_k"] = (dst+"att.r_k", "identity", [c // head, head])
        for suffix in ("weight", "bias"):
            spec[src+"attn.g_norm."+suffix] = (dst+"att.ln_x."+suffix, "identity", [c])
        for name, rank in (("w", w), ("a", a), ("g", g), ("v", v)):
            if name == "v" and i == 0:
                continue  # FLA omits the unused layer-0 value gate.
            spec[src+f"attn.{name}_lora.lora.0.weight"] = (dst+f"att.{name}1", "transpose", [rank, c])
            spec[src+f"attn.{name}_lora.lora.2.weight"] = (dst+f"att.{name}2", "transpose", [c, rank])
            if name != "g":
                spec[src+f"attn.{name}_lora.lora.2.bias"] = (dst+f"att.{name}0", "channel", [c])
        spec[src+"ffn.x_k"] = (dst+"ffn.x_k", "channel", [c])
        spec[src+"ffn.key.weight"] = (dst+"ffn.key.weight", "identity", [inner, c])
        spec[src+"ffn.value.weight"] = (dst+"ffn.value.weight", "identity", [c, inner])
    return spec


def inspect_snapshot(snapshot: Path = SNAPSHOT) -> dict[str, Any]:
    """Data-only preflight; reads JSON/header and hashes bytes, never tensor data."""
    from third_party.rwkv7_hf import CODE_SHA256, vendored_ok
    if not all(vendored_ok().values()):
        raise RuntimeError("vendored implementation hash changed")
    hashes = {name: sha(snapshot / name) for name in PINNED_HASHES}
    if hashes != PINNED_HASHES:
        raise RuntimeError("CPU loader accepts only the reviewed cached checkpoint")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    # Only this shape/architecture was statically reviewed. Unknown semantics fail closed.
    if (config["attn"] is not None or config["hidden_act"] != "sqrelu"
            or config["norm_first"] is not True or config["fuse_norm"] is not False
            or config["tie_word_embeddings"] is not False):
        raise RuntimeError("checkpoint architecture is outside reviewed CPU conversion")
    with (snapshot / "model.safetensors").open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        if not 0 < length < 1024**2:
            raise RuntimeError("unbounded safetensors header")
        header = json.loads(file.read(length))
    spec = conversion_spec(768, 12, 65536, 64, 3072, 64, 64, 32, 128)
    entries = {k: val for k, val in header.items() if k != "__metadata__"}
    if set(entries) != set(spec) or any(val["dtype"] != "BF16" or val["shape"] != spec[k][2]
                                      for k, val in entries.items()):
        raise RuntimeError("checkpoint schema differs from reviewed FLA conversion")
    implementation_hashes = {f"third_party/rwkv7_hf/{name}": value for name, value in CODE_SHA256.items()}
    for name in ("language_growth/rwkv7_cpu_parent.py", "decide/rwkv_world_tokenizer.py"):
        implementation_hashes[name] = sha(ROOT / name)
    return {"snapshot": str(snapshot), "snapshot_revision": snapshot.name,
            "file_sha256": hashes, "implementation_sha256": implementation_hashes,
            "config": config, "weight_header": header, "conversion_spec": spec,
            "stored_parameters": 191034624, "effective_heads": 12,
            "stored_bytes": 382069248, "fp32_parameter_bytes": 764138496,
            "unused_layer0_buffers": ["rwkv7.blocks.0.att.v0", "rwkv7.blocks.0.att.v1", "rwkv7.blocks.0.att.v2"]}


def convert_tensor(tensor, transform: str):
    if transform == "transpose":
        return tensor.T.contiguous()
    if transform == "channel":
        return tensor.reshape(1, 1, -1)
    if transform != "identity":
        raise ValueError("unknown conversion transform")
    return tensor


def load_cpu_parent(snapshot: Path = SNAPSHOT):
    """Explicit opt-in FP32 loader. Caller must own RAM/time guards."""
    evidence = inspect_snapshot(snapshot)  # gate before importing Torch or reading tensors
    import torch
    from safetensors import safe_open
    from transformers.initialization import no_init_weights
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM
    from decide.backends.hf_readout import ModelPlugin
    from decide.rwkv_world_tokenizer import RWKVWorldTokenizer

    config = Rwkv7Config(vocab_size=65536, hidden_size=768, num_hidden_layers=12,
                        head_dim=64, num_heads=12, intermediate_size=3072,
                        decay_low_rank_dim=64, a_low_rank_dim=64, v_low_rank_dim=32,
                        gate_low_rank_dim=128, wkv_implementation="chunked", wkv_state_dtype="float32")
    with no_init_weights(), torch.device("meta"):
        model = Rwkv7ForCausalLM(config)
    expected = model.state_dict()
    weights = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as file:
        for source, (target, transform, _) in evidence["conversion_spec"].items():
            value = convert_tensor(file.get_tensor(source).float(), transform)
            if list(value.shape) != list(expected[target].shape):
                raise RuntimeError(f"mapped tensor shape mismatch: {source}")
            weights[target] = value
    for name in evidence["unused_layer0_buffers"]:
        weights[name] = torch.zeros(expected[name].shape, dtype=torch.float32)
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    if any(t.device.type != "cpu" for t in model.state_dict().values()):
        raise RuntimeError("CPU model contains non-CPU tensors")
    tok = RWKVWorldTokenizer(snapshot / "rwkv_vocab_v20230424.txt")

    def forward(ids, n_last):
        if ids.device.type != "cpu" or ids.shape[0] > 2 or ids.shape[1] > 2048:
            raise RuntimeError("bounded CPU readout shape exceeded")
        return model(input_ids=ids, logits_to_keep=n_last, use_cache=False).logits

    source_identity = hashlib.sha256(json.dumps(evidence["implementation_sha256"], sort_keys=True).encode()).hexdigest()
    plugin = ModelPlugin(family="rwkv7", model_id=MODEL_ID, snapshot=snapshot.name,
                         encode=lambda text, trusted=True: tok.encode(text),
                         untrusted_encoder_id="rwkv-world-greedy (no control tokens)", forward=forward,
                         files={"weights": PINNED_HASHES["model.safetensors"],
                                "config": PINNED_HASHES["config.json"],
                                "tokenizer": PINNED_HASHES["rwkv_vocab_v20230424.txt"], "code": source_identity},
                         max_context=2048, dtype="float32", device="cpu",
                         kernel_stack={"wkv_implementation": "vendored-pytorch-chunked", "code_sha256": source_identity},
                         notes={"attention_free": True, "model_class": type(model).__name__,
                                "trained_ctx": 2048, "effective_heads": 12,
                                "scope": "exploratory CPU parent; no strong-parent qualification",
                                "conversion": "fla-0.5.2-to-vendored-rwkv7-v1"})
    return plugin, model, evidence

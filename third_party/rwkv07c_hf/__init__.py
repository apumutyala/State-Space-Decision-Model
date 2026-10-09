"""Bundled Transformers RWKV07C code of OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c (RWKV hxa07c conversion of
Qwen3-30B-A3B-Instruct, every layer an RWKV-7-style recurrent time mix, Qwen3-MoE FFN), vendored UNMODIFIED
(Apache-2.0, model card). ``verify_snapshot_code`` refuses a snapshot whose code or config.json differs.

Attention-free: config ``transformer_layers: [-1]`` (``is_layer_attention`` is ``layer_id in transformer_layers``, so no
layer index matches) and ``rwkv_layers`` lists all 48 layers; the loader also asserts every layer's ``self_attn`` is
``RWKV07CAttention``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c"
REVISION = "2fb46556362db60336d4abf951aee3ebfb7db449"
CODE_SHA256 = {
    "modeling_rwkv07c.py": "b2d037f2615380474574d968f74e362be847d44f2bb14995de123d9c5805c8b1",
    "configuration_rwkv07c.py": "8ba500eed019e4f04884fb7f66ec607c48bfb52bc9c2c587f1506ce34e84af60",
}
CONFIG_SHA256 = "435918fcf6d37fce4eda9869d3c10044b38c41d1f4921ce7aea3961dc3a563ca"
#: backend input cap: the card's NIAH falls from 84 at 8K to 37.8 at 16K; 16,384 as for the RWKV6-Qwen2 screens
INPUT_CAP = 16384
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in CODE_SHA256.items()}


def verify_snapshot_code(model_dir: str | Path) -> dict[str, str]:
    d = Path(model_dir)
    got = {f: _sha((d / f).resolve()) for f in CODE_SHA256}
    got["config.json"] = _sha((d / "config.json").resolve())
    want = {**CODE_SHA256, "config.json": CONFIG_SHA256}
    bad = {f: got[f] for f in want if got[f] != want[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"RWKV07C snapshot code/config differs from the pins: {bad} (vendored ok: {vendored_ok()})")
    return got

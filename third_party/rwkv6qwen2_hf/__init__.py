"""Bundled Transformers RWKV6-Qwen2 code (RADLADS conversions of Qwen2.5 / QwQ to RWKV-6 linear attention), vendored
UNMODIFIED (Apache-2.0; see NOTICE.md). Byte-identical in both pinned repos below (fetched as text 2026-10-08).
Loading these classes directly avoids ``trust_remote_code``; ``verify_snapshot_code`` refuses a snapshot whose own
code or config.json differs from the pins, so the weights always run under the code that was reviewed.

Attention-free: every decoder layer's ``self_attn`` is ``RWKV6Attention`` (fla ``fused_recurrent_gla`` with an
RWKV-6 data-dependent decay); no softmax attention anywhere (no ``attn_layers`` / hybrid list in the code or configs).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: repo -> pinned revision + config.json sha256 (HF API sha / raw file at that revision, 2026-10-08). input_cap: the
#: backend input cap; the Instruct card states 16K training context, the QwQ card states none (same RADLADS recipe),
#: so both are capped at 16,384 (recurrent models degrade silently past their trained length).
REPOS = {
    "recursal/QRWKV6-32B-Instruct-Preview-v0.1": {
        "revision": "2da855ca372cefab920716eca30f11ed65d6c708",
        "config_sha256": "8d5ad345f0c6d8e63b19b092779d5b30407af4ada9a75ff0790d83ad60782823",
        "license": "apache-2.0", "parent": "Qwen/Qwen2.5-32B-Instruct", "input_cap": 16384},
    "featherless-ai/QRWKV-QwQ-32B": {
        "revision": "6801e98e72a3e03b4611cdd816a0532411f6e1e0",
        "config_sha256": "cac3e8cdfee0b234600d5eece6dda592d23229f20d6b243f76300448cc56fcb4",
        "license": "apache-2.0", "parent": "Qwen/QwQ-32B", "input_cap": 16384},
}
CODE_SHA256 = {
    "modeling_rwkv6qwen2.py": "dd4f49b1af2c666c9be00f30257680403cd07e3214782150c6e869db42b6806a",
    "configuration_rwkv6qwen2.py": "d66834fe2b15148b2d75357263b12870a68700e416b2b698fdaea1beeda267dc",
}
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in CODE_SHA256.items()}


def verify_snapshot_code(model_dir: str | Path, repo: str) -> dict[str, str]:
    """Raise unless the snapshot's code files and config.json are byte-identical to the pinned ones for ``repo``."""
    if repo not in REPOS:
        raise RuntimeError(f"{repo!r} is not a pinned RWKV6-Qwen2 repo: {sorted(REPOS)}")
    d = Path(model_dir)
    got = {f: _sha((d / f).resolve()) for f in CODE_SHA256}
    got["config.json"] = _sha((d / "config.json").resolve())
    want = {**CODE_SHA256, "config.json": REPOS[repo]["config_sha256"]}
    bad = {f: got[f] for f in want if got[f] != want[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"RWKV6-Qwen2 snapshot code/config differs from the pins: {bad} (vendored ok: {vendored_ok()})")
    return got

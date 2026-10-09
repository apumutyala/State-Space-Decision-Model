"""LiquidAI/d1-3B's bundled code (``modeling_d1.py``, ``lfm2_vl.py``, ``hybrid.py``, ``runner.py``, ``prompt.py``,
``api.py``), vendored UNMODIFIED (LFM Open License v1.0; see LICENSE and NOTICE.md) so the JevBench host scan reads d1-3B
through its documented readout, ``D1Model.system_one(state, questions)`` (one forward pass, softmax over each option's
single-token code / yes-no forms / score digits at the answer slot), without ``trust_remote_code``. ``verify_snapshot``
refuses a snapshot whose own code differs from this copy, so the weights always run under the reviewed code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
#: sha256 of the files at REVISION (fetched 2026-10-09); the vendored copies must match too
CODE_SHA256 = {
    "modeling_d1.py": "7fc6a001f80fc2d146f2a51128478f3b3bd4f1423a3ac129bc10f3fd0caf1d98",
    "lfm2_vl.py": "f9e8d3354572a169237865283594b0364209fdb858901121b9f99147328b5d95",
    "hybrid.py": "ce579827b952a6978a3560f70e74be9e228f9029de46048996b0c5f58d7f1f39",
    "runner.py": "24d3ae718e9f9717771c0df39c93d324d00aa27bf3e12bab085c0922096cb0dd",
    "prompt.py": "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31",
    "api.py": "94e9f82eabfdf92c4abe1153e9a6dc4dd9821f9cc2fec3221e79e70723d8ada0",
}
CONFIG_SHA256 = "0cbac0f580bd8036ef94ac83aa632a57a170e66d20f381ef47c3f6486c9a3e8c"
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in CODE_SHA256.items()}


def verify_snapshot_code(model_dir: str | Path) -> dict[str, str]:
    """Raise unless the snapshot's code files and config.json are byte-identical to the pinned ones."""
    d = Path(model_dir)
    got = {f: _sha((d / f).resolve()) for f in CODE_SHA256}
    got["config.json"] = _sha((d / "config.json").resolve())
    want = {**CODE_SHA256, "config.json": CONFIG_SHA256}
    bad = {f: got[f] for f in want if got[f] != want[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"d1-3B snapshot code/config differs from the pinned vendored copy: {bad} "
                           f"(vendored ok: {vendored_ok()})")
    return got


def load_d1(model_dir: str | Path, device: str = "cuda", dtype: str = "bfloat16"):
    """The vendored ``D1Model`` on ``model_dir`` (weights + tokenizer from the snapshot; code from here)."""
    import torch

    if not all(vendored_ok().values()):
        raise RuntimeError(f"vendored d1-3B code bytes changed: {vendored_ok()}")
    from .modeling_d1 import D1Model

    verify_snapshot_code(model_dir)
    model = D1Model.from_pretrained(str(model_dir), dtype=getattr(torch, dtype)).to(device).eval()
    return model

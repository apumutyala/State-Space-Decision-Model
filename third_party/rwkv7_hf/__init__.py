"""Bundled Transformers RWKV-7 code of RWKV/RWKV7-G1k-13.3B-20260930, vendored UNMODIFIED (Apache-2.0; see LICENSE,
NOTICE and NOTICE.md). Loading these classes directly avoids ``trust_remote_code``; ``verify_snapshot_code`` refuses a
snapshot whose own code differs from this copy, so the weights always run under the code that was reviewed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "RWKV/RWKV7-G1k-13.3B-20260930"
REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
#: sha256 of the files at REVISION (fetched as text 2026-10-06); the vendored copies must match too
CODE_SHA256 = {
    "modeling_rwkv7.py": "28fc373673b2c98805275cc55381db0dd1dd9325ec0c1b34264d49ca1bc16958",
    "configuration_rwkv7.py": "14050c631e27d66cd2f52ca5ec53b62eef04e602c4f1fc4e3696c328bc8e200e",
    "tokenization_rwkv7.py": "58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65",
}
CONFIG_SHA256 = "abb02dc8556ed94926d28593fce202430500f132f77273c7a158fab2eb0beccf"
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
        raise RuntimeError(f"RWKV7 snapshot code/config differs from the pinned vendored copy: {bad} "
                           f"(vendored ok: {vendored_ok()})")
    return got

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
#: the other RWKV7-G1k sizes of the same release ship byte-identical code and tokenizer files (sha256 checked
#: 2026-10-09 at the revisions below); only config.json differs, pinned per repo
CONFIG_SHA256_BY_REPO = {
    REPO: CONFIG_SHA256,
    "RWKV/RWKV7-G1k-7.2B-20260930": "13d0c1ba3972f33198485455bc1d5fd98603b1618bfd37524ed8191925ed9b10",  # @3badfea3
    "RWKV/RWKV7-G1k-2.9B-20260930": "f6bd6f715ed95780f3cab7540eab8447aa52bc4b5413544325164dc66c51c3c1",  # @c3a1ef92
    "RWKV/RWKV7-G1k-1.5B-20260930": "95f3524181116fca466ce5b2e8c827dcf1a026ebfb4967d3ad28af9feee96d1f",  # @4923bf40
}
#: RWKV7 releases that ship NO modeling code (transformers-native layout, e.g. the G1i 20260805 sizes): they are run
#: under this vendored G1k code when their config.json is pinned here and their tokenizer.json is the G1k one (the
#: weight names were checked equal to the G1k 2.9B index, 1,062 tensors, 2026-10-09)
TOKENIZER_SHA256 = "1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf"
CODELESS_CONFIG_SHA256 = {
    "RWKV/RWKV7-2.9B-20260805": "6c091162fff38d7b7d0aeacaf69e08e0c96712e8c59e3f0fd325ee2957b8fc9d",  # @aae64511 (G1i, wkv_implementation "eager")
}
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in CODE_SHA256.items()}


def verify_snapshot_code(model_dir: str | Path, repo: str = REPO) -> dict[str, str]:
    """Raise unless the snapshot's code files and config.json are byte-identical to the pinned ones (``repo``: which
    G1k size's config.json; the code files are the same for every size)."""
    d = Path(model_dir)
    if repo in CODELESS_CONFIG_SHA256:
        got = {f: _sha((d / f).resolve()) for f in ("config.json", "tokenizer.json")}
        want = {"config.json": CODELESS_CONFIG_SHA256[repo], "tokenizer.json": TOKENIZER_SHA256}
        bad = {f: got[f] for f in want if got[f] != want[f]}
        if bad or not all(vendored_ok().values()) or any((d / f).exists() for f in CODE_SHA256):
            raise RuntimeError(f"{repo}: config/tokenizer differ from the pins, or the snapshot ships its own code: "
                               f"{bad} (vendored ok: {vendored_ok()})")
        return got
    got = {f: _sha((d / f).resolve()) for f in CODE_SHA256}
    got["config.json"] = _sha((d / "config.json").resolve())
    want = {**CODE_SHA256, "config.json": CONFIG_SHA256_BY_REPO[repo]}
    bad = {f: got[f] for f in want if got[f] != want[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"RWKV7 snapshot code/config differs from the pinned vendored copy: {bad} "
                           f"(vendored ok: {vendored_ok()})")
    return got

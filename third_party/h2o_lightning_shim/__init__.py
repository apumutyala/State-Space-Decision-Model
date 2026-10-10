"""H2O-Lightning-4B v1.1 shim (h2oai/h2o-lightning-4b @ tag v1.1), vendored UNMODIFIED (Apache-2.0; see LICENSE and
NOTICE.md) so the JevBench host scan reads H2O-Lightning-4B exactly as its published board row did: the shim's own
prompt (``Contract.question_prompt``), its label table, temperature, yes/no floor and answer formatting
(``format_answer``). Only the vLLM HTTP call is replaced (jobs/jev_host_scan/host_scan.py: one transformers forward,
log-probabilities of the label tokens at the answer slot, the lm head in fp32 when config.json says
``head_dtype: float32``, as vLLM does). ``verify_snapshot`` refuses a snapshot whose shim or serve_config.json differ
from this copy, so the weights always run under the reviewed contract.

Importing the shim module only defines functions and classes; its HTTP server (``main``) is never started here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO = "h2oai/h2o-lightning-4b"
TAG = "v1.1"
REVISION = "193ad740925b176a3b70a5a13a7cff2f2fadd01e"
#: sha256 of the files at REVISION (fetched 2026-10-09); the vendored copies must match too
FILES_SHA256 = {
    "h2o_lightning_shim.py": "355791c3f7b50a5dc9fa759dfa1b861874678d6e04c70dbf8b86fddccd3c0932",
    "serve_config.json": "dfe5fd08b5b89cfd1be2797baf7a570c32f2883c6520d2b047cd5528591cee7a",
}
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in FILES_SHA256.items()}


def verify_snapshot(model_dir: str | Path) -> dict[str, str]:
    """Raise unless the snapshot's shim and serve_config.json are byte-identical to the vendored (pinned) ones."""
    d = Path(model_dir)
    got = {f: _sha((d / f).resolve()) for f in FILES_SHA256}
    bad = {f: got[f] for f in FILES_SHA256 if got[f] != FILES_SHA256[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"H2O shim snapshot differs from the pinned vendored copy: {bad} (vendored ok: {vendored_ok()})")
    return got


def load_shim():
    """The vendored shim module (definitions only), after the sha256 check of the vendored bytes."""
    if not all(vendored_ok().values()):
        raise RuntimeError(f"vendored H2O shim bytes changed: {vendored_ok()}")
    spec = importlib.util.spec_from_file_location("h2o_lightning_shim_v11", HERE / "h2o_lightning_shim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_contract(env: dict | None = None):
    """The shim's ``Contract`` over the vendored serve_config.json; ``env`` defaults to {} so no SHIM_TEMPERATURE /
    SHIM_NOUL_FLOOR override from the job's environment can change the evaluated configuration."""
    mod = load_shim()
    cfg = json.loads((HERE / "serve_config.json").read_text(encoding="utf-8"))
    return mod, mod.Contract(cfg, env={} if env is None else env)

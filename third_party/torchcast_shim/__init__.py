"""Torchcast Decision 12B shim (torchcast-ai/torchcast-decision-12b @ 49107b55 = github Torchcast-AI/torchcast-decision-12b
v1.0.0 serving/), vendored UNMODIFIED (MIT, see LICENSE-serving-MIT and NOTICE.md) so the kit-parity job reads the
checkpoint exactly as its author's public-231 release run did: the shim's own system prompt, prompt (``build_prompt``),
option table (``options_from``), letter-mass readout (``letter_probs`` over the top-20 log-probabilities at the answer
slot) and per-type temperatures (``readout_config.json``). Only the vLLM HTTP call is replaced
(jobs/jev_host_scan/host_scan.py ``TorchcastEngine``: the chat template with thinking off, one transformers forward, the
raw top-20 log-probabilities, as vLLM's default raw-logprob mode returns them before any structured-output mask).

Importing the shim module only defines functions and classes; its HTTP server (``__main__``) is never started here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

REPO = "torchcast-ai/torchcast-decision-12b"
REVISION = "49107b5589bf2dc4ec3f5acd54316971dd5978b1"
#: sha256 of the files at REVISION (fetched 2026-10-10; byte-identical to github v1.0.0 serving/); vendored copies match
FILES_SHA256 = {
    "torchcast_shim.py": "1dff956ead46f251fe7c60d91b4cf08b241d56a69a45eba835463bf4fc252694",
    "readout_config.json": "8a4de5ef4a26a24d8be69c123a6a6622016f487693b0bc8fc683b92f0d7ade97",
}
HERE = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vendored_ok() -> dict[str, bool]:
    return {f: _sha(HERE / f) == h for f, h in FILES_SHA256.items()}


def verify_snapshot(model_dir: str | Path) -> dict[str, str]:
    """Raise unless the snapshot's shim and readout_config.json are byte-identical to the vendored (pinned) ones."""
    d = Path(model_dir)
    got = {f: _sha((d / f).resolve()) for f in FILES_SHA256}
    bad = {f: got[f] for f in FILES_SHA256 if got[f] != FILES_SHA256[f]}
    if bad or not all(vendored_ok().values()):
        raise RuntimeError(f"torchcast shim snapshot differs from the pinned vendored copy: {bad} (vendored ok: {vendored_ok()})")
    return got


def load_shim():
    """The vendored shim module (definitions only) after the sha256 check. Its import reads the temperatures from the
    vendored readout_config.json; every SHIM_* variable is removed from the environment first, so nothing in the job's
    environment can change the evaluated configuration."""
    if not all(vendored_ok().values()):
        raise RuntimeError(f"vendored torchcast shim bytes changed: {vendored_ok()}")
    for k in [k for k in os.environ if k.startswith("SHIM_")]:
        del os.environ[k]
    spec = importlib.util.spec_from_file_location("torchcast_shim_v100", HERE / "torchcast_shim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.TEMPS != {"choice": 1.0, "score": 1.0, "noul": 0.2}:
        raise RuntimeError(f"torchcast shim temperatures {mod.TEMPS} != the published readout_config.json")
    return mod

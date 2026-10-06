"""Canonical scoring path: drive Track D's ``decide()`` adapter (decide/, PR #6) from safety_eval.

safety_eval does NOT encode untrusted text itself on the gated path. Every gated number comes from
``decide.Decider.decide`` with whatever backend is under test (HF readout for Qwen3.5 / RWKV7 / Mamba-3, oracle or
replay in tests), so the untrusted-text encoding, control-token refusal, canonical option order, calibration and
policy conditioning are exactly what production runs. When decide/ changes its encoder (e.g. split special tokens +
neutralise non-special added tokens such as ``<think>``), safety_eval inherits it with no code change here.

Translation (safety_eval request -> decide wire request):
  state, questions                -> unchanged (SystemOne-compatible top level)
  envelope schema                 -> ssdm.schema = decide.versions.ENVELOPE_SCHEMA (read from the installed decide/,
                                     currently "ssdm.decide.v0.1"; never hard-coded here)
  policy.allowed                  -> ssdm.policy_restriction (core in contract 0.1; no flag needed)
  request_id, parent_revision     -> ssdm.request_id, ssdm.parent_revision
  provenance.untrusted_fields     -> NOT transmitted. decide 0.2.0 carries ssdm.provenance only behind
                                     enable_experimental_provenance and only for a STRING state; without it every
                                     request field is untrusted data (the safe superset). ISE/ASIDE arms need the 0.2
                                     provenance extension ratified before m2.
decide response -> safety_eval response:
  raw_probabilities               <- ssdm.raw_probabilities (pre-calibration, unmasked)
  answers[q].probabilities        <- policy-conditioned probabilities over all options (0 outside the allowed set);
  answers[q].choice                  choice = their argmax (= what the host executes). A question with a restriction
                                     whose conditioning stage is not "ok" (e.g. unavailable_zero_served_mass) executes
                                     NOTHING: status "unsupported", error "policy_conditioned_unavailable" (never a
                                     fallback to the unrestricted served argmax). A restriction sent but no
                                     conditioning returned is a contract violation (ValueError -> error row).
  served (calibrated, unmasked)   -> diagnostics.served_probabilities
  non-ok status                   -> {"status": status, "error": ssdm.error.code, "error_message": ...}; a refusal
                                     on an attacked request is a blocked attack, on a clean request a utility cost
                                     (metrics.refusal_*). The legacy top-level error shape is used only if ssdm is
                                     absent.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

ENV_DECIDE_PATH = "SSDM_DECIDE_PATH"  # repo root that contains decide/ (until PR #6 is on agent-main)
ENV_QWEN_DIR = "SSDM_QWEN_DIR"        # Qwen3.5 snapshot directory (shared with PR #6)


def import_decide():
    """Import the decide package (repo root on sys.path, or ``$SSDM_DECIDE_PATH``)."""
    try:
        return importlib.import_module("decide")
    except ImportError:
        root = os.environ.get(ENV_DECIDE_PATH)
        if not root:
            raise ImportError(f"decide/ not importable; set {ENV_DECIDE_PATH} to a checkout containing decide/")
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("decide")


def envelope_schema() -> str:
    import_decide()
    from decide import versions
    return versions.ENVELOPE_SCHEMA


def to_decide_request(req: dict, model: str = "ssdm-eval") -> dict:
    env: dict[str, Any] = {"schema": envelope_schema()}
    if req.get("request_id") is not None:
        env["request_id"] = req["request_id"]
    if req.get("parent_revision") is not None:
        env["parent_revision"] = req["parent_revision"]
    allowed = (req.get("policy") or {}).get("allowed")
    if allowed:
        env["policy_restriction"] = {q: list(v) for q, v in allowed.items()}
    return {"model": model, "state": req["state"], "questions": req["questions"], "ssdm": env}


def from_decide_response(req: dict, resp: dict) -> dict:
    ss = resp.get("ssdm")
    if not isinstance(ss, dict):  # legacy-only response shape (no envelope)
        err = resp.get("error") or {}
        code = err.get("code") if isinstance(err, dict) else str(err)
        return {"status": "error", "error": code or "no_ssdm_envelope", "error_message": None, "diagnostics": {}}
    status = ss.get("status", "ok")
    if status != "ok":
        err = ss.get("error") or {}
        return {"status": status, "error": err.get("code", "unknown"), "error_message": err.get("message"),
                "diagnostics": ss.get("diagnostics", {})}
    raw = ss["raw_probabilities"]
    served = ss["served_probabilities"]
    pcp = ss.get("policy_conditioned_probabilities")
    restricted = set(((req.get("policy") or {}).get("allowed") or {}))
    if restricted and not isinstance(pcp, dict):
        raise ValueError("policy_restriction sent but no policy_conditioned_probabilities returned (contract 0.1)")
    pcp = pcp or {}
    answers = {}
    for q, spec in req["questions"].items():
        keys = list(spec["criteria"])
        if q in restricted:
            stage = pcp.get(q)
            if not isinstance(stage, dict):
                raise ValueError(f"{q}: restriction sent but no conditioning stage returned (contract 0.1)")
            if stage.get("status", "ok") != "ok" or stage.get("probabilities") is None:
                return {"status": "unsupported", "error": "policy_conditioned_unavailable",
                        "error_message": f"{q}: conditioning stage {stage.get('status')}",
                        "diagnostics": ss.get("diagnostics", {})}
            cond = stage["probabilities"]
            probs = {k: float(cond.get(k, 0.0)) for k in keys}
        else:  # no restriction for this question: the host executes the served argmax
            probs = {k: float(served[q][k]) for k in keys}
        best = max(sorted(keys), key=lambda k: probs[k])  # tie -> smallest id, as decide.wire_answer
        answers[q] = {"type": "choice", "choice": best, "probabilities": probs}
    diag = dict(ss.get("diagnostics", {}))
    diag["served_probabilities"] = served
    diag["path"] = "decide_bridge"
    return {"status": "ok", "answers": answers, "raw_probabilities": raw,
            "policy_conditioned_probabilities": {q: a["probabilities"] for q, a in answers.items()},
            "diagnostics": diag}


class DecideBridge:
    """safety_eval ``decide(request) -> response`` over a ``decide.Decider``."""

    def __init__(self, decider, model: str = "ssdm-eval"):
        # contract 0.1: policy_restriction is core, so any Decider works; the response is still checked per call
        self.decider, self.model = decider, model

    def revision(self) -> dict:
        rev = getattr(self.decider, "revision", None)  # the served revision (backend + versioned calibration)
        return rev() if callable(rev) else self.decider.backend.revision()

    def __call__(self, req: dict) -> dict:
        return from_decide_response(req, self.decider.decide(to_decide_request(req, self.model)))


def build_hf_bridge(model_dir: str | None = None, family: str = "qwen3.5", device: str = "cpu",
                    dtype: str = "float32", tokenizer_dir: str | None = None) -> DecideBridge:
    """HF-readout bridge for a local snapshot (default ``$SSDM_QWEN_DIR``). CPU by default."""
    import_decide()
    from decide import Decider
    from decide.backends import hf_readout as H
    model_dir = model_dir or os.environ.get(ENV_QWEN_DIR)
    if not model_dir:
        raise ValueError(f"pass model_dir or set {ENV_QWEN_DIR}")
    templates = {"qwen3.5": "jrt-chatml-v1", "rwkv7": "jrt-rwkv-g1-v1", "rwkv7-hf": "jrt-rwkv-g1-v1",
                 "mamba3": "jrt-plain-v1"}
    if family == "qwen3.5":
        plug = H.load_qwen35(model_dir, device=device, dtype=dtype)
    elif family == "rwkv7":
        plug = H.load_rwkv7(model_dir, device=device, dtype=dtype)
    elif family == "rwkv7-hf":  # RWKV/RWKV7-G1k-*-20260930 (bundled HF code, vendored in third_party/rwkv7_hf)
        plug = H.load_rwkv7_hf(model_dir, device=device, dtype=dtype)
    elif family == "mamba3":
        plug = H.load_mamba3(model_dir, tokenizer_dir, device=device, dtype=dtype)
    else:
        raise ValueError(family)
    backend = H.HFReadoutBackend(plug, templates[family])
    return DecideBridge(Decider(backend, backend_max_input_tokens=plug.max_context))


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_files_sha(model_dir: str | Path) -> dict:
    """sha256 of weights / tokenizer / config in a local snapshot (resolves HF-cache symlinks)."""
    d = Path(model_dir)
    out = {}
    for role, pats in {"weights": ["*.safetensors"], "tokenizer": ["tokenizer.json"], "config": ["config.json"]}.items():
        files = sorted(p for pat in pats for p in d.glob(pat))
        if not files:
            out[role] = "missing"
        elif len(files) == 1:
            out[role] = sha256_file(files[0].resolve())
        else:
            out[role] = hashlib.sha256("".join(f"{f.name}:{sha256_file(f.resolve())}\n"
                                               for f in files).encode()).hexdigest()
    return out

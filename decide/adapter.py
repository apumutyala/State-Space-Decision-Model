"""``decide(request, state_handle=None) -> response``.

Implements the decision adapter of ``interfaces/DECISION_LANGUAGE_V0.md``.

Transport: the SystemOne-compatible fields (``model``, ``state``, ``questions``)
stay at the top level, untouched, and the canonical v0 fields travel in a
versioned extension envelope ``"ssdm"`` beside them (request and response). A
legacy SystemOne client that ignores ``ssdm`` sees exactly the SystemOne wire
response; the envelope never changes the meaning of the legacy fields.

Request envelope (all optional; legacy requests are accepted without it)::

    "ssdm": {
      "schema": "ssdm.decide.v0",
      "request_id": str,             # identity only, not a permission token
      "parent_revision": str | null,  # must equal the backend's revision_id if given
      "context_ref": str, "policy_ref": str, "registry_ref": str,
      "audience": "public" | "private",
      "limits": {"max_input_tokens", "max_choices", "max_program_nodes",
                 "max_inference_calls", "deadline_ms"},   # positive ints
      "policy_restriction": {qid: [allowed option ids]}   # v0.1 delta, flag-gated
    }

Response: legacy ``model`` / ``answers`` / ``usage`` plus ``"ssdm"`` with
``status`` (ok | unsupported | exhausted | error), ``error`` {code, message},
``abstention`` (explicit, separate from status), ``raw_probabilities``,
``served_probabilities`` + ``served_config``, ``diagnostics``,
``proposed_evidence_refs`` and ``state_handle``. With ``enable_v0_1_deltas``:
``policy_conditioned_probabilities`` (when a restriction is supplied) and
``attribution`` (only if the backend produces accounting attribution).
``semantic_records`` is never emitted until Track D's semantic-interface gate.

Guarantees: no silent truncation, option pruning or model fallback; stateless
(an incoming ``state_handle`` is discarded and the result recomputed, as the
interface prescribes on mismatch); every non-ok status carries a reason code
and no answers.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

from . import versions
from .backends.base import Backend, BackendUnsupported, DeadlineExceeded
from .calibration import OptionCountTemperature, softmax
from .schema import (
    RequestRejected,
    canonical_options,
    request_fingerprint,
    validate_legacy,
    wire_answer,
)

LIMIT_KEYS = (
    "max_input_tokens",
    "max_choices",
    "max_program_nodes",
    "max_inference_calls",
    "deadline_ms",
)
ENVELOPE_KEYS = {
    "schema",
    "request_id",
    "parent_revision",
    "context_ref",
    "policy_ref",
    "registry_ref",
    "audience",
    "limits",
    "policy_restriction",
}
DEFAULT_LIMITS = {
    "max_input_tokens": 16384,  # Clef encode_record default
    "max_choices": 64,  # Workers AI: 1-64 questions per request
    "max_program_nodes": 8,  # recorded only; decide() emits no program nodes
    "max_inference_calls": 4096,
    "deadline_ms": None,  # no deadline unless the host sets one
}


class Decider:
    """Binds a backend, a calibration config and the optional-delta flag."""

    def __init__(
        self,
        backend: Backend,
        calibration: OptionCountTemperature | None = None,
        enable_v0_1_deltas: bool = False,
        backend_max_input_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.calibration = calibration or OptionCountTemperature()
        self.enable_v0_1_deltas = enable_v0_1_deltas
        self.backend_max_input_tokens = backend_max_input_tokens

    # ------------------------------------------------------------------ helpers
    def _envelope(self, request: dict[str, Any]) -> dict[str, Any]:
        env = request.get("ssdm")
        if env is None:
            return {}
        if not isinstance(env, dict):
            raise RequestRejected("error", "invalid_request", "ssdm envelope must be an object")
        unknown = set(env) - ENVELOPE_KEYS
        if unknown:
            raise RequestRejected(
                "error", "invalid_request", f"unknown ssdm envelope keys: {sorted(unknown)}"
            )
        if env.get("schema", versions.ENVELOPE_SCHEMA) != versions.ENVELOPE_SCHEMA:
            raise RequestRejected(
                "unsupported", "envelope_version_unsupported", f"ssdm.schema must be {versions.ENVELOPE_SCHEMA}"
            )
        if env.get("audience") not in (None, "public", "private"):
            raise RequestRejected("error", "invalid_request", "audience must be public or private")
        return env

    def _limits(self, env: dict[str, Any]) -> dict[str, Any]:
        given = env.get("limits") or {}
        if not isinstance(given, dict):
            raise RequestRejected("error", "invalid_request", "limits must be an object")
        unknown = set(given) - set(LIMIT_KEYS)
        if unknown:
            raise RequestRejected("error", "invalid_request", f"unknown limit keys: {sorted(unknown)}")
        limits = dict(DEFAULT_LIMITS)
        for k, v in given.items():
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise RequestRejected("error", "invalid_request", f"limits.{k} must be a positive integer")
            limits[k] = v
        limits["_source"] = {k: ("request" if k in given else "adapter_default") for k in LIMIT_KEYS}
        if self.backend_max_input_tokens is not None and (
            limits["max_input_tokens"] is None or limits["max_input_tokens"] > self.backend_max_input_tokens
        ):
            limits["max_input_tokens"] = self.backend_max_input_tokens
            limits["_source"]["max_input_tokens"] = "backend_context_limit"
        return limits

    def _restriction(self, env: dict[str, Any], request: dict[str, Any]) -> dict[str, list[str]] | None:
        restriction = env.get("policy_restriction")
        if restriction is None:
            return None
        if not self.enable_v0_1_deltas:
            raise RequestRejected(
                "unsupported",
                "policy_restriction_requires_v0_1_deltas",
                "policy_restriction is a v0.1 delta; enable_v0_1_deltas is off",
            )
        if not isinstance(restriction, dict):
            raise RequestRejected("error", "invalid_request", "policy_restriction must be an object")
        out = {}
        for qid, allowed in restriction.items():
            if qid not in request["questions"]:
                raise RequestRejected("error", "invalid_request", f"policy_restriction: unknown question {qid}")
            ids = {o for o, _ in canonical_options(request["questions"][qid])}
            if not isinstance(allowed, list) or not allowed or not set(allowed) <= ids:
                raise RequestRejected(
                    "error", "invalid_request", f"policy_restriction.{qid}: non-empty subset of option ids required"
                )
            out[qid] = sorted(set(allowed))
        return out

    # --------------------------------------------------------------------- main
    def decide(self, request: dict[str, Any], state_handle: Any = None) -> dict[str, Any]:
        t_start = time.perf_counter()
        env: dict[str, Any] = {}
        request_id = None
        try:
            if isinstance(request, dict) and isinstance(request.get("ssdm"), dict):
                request_id = request["ssdm"].get("request_id")
            validate_legacy(request)
            env = self._envelope(request)
            request_id = env.get("request_id")
            limits = self._limits(env)
            restriction = self._restriction(env, request)
            revision = self.backend.revision()
            wanted = env.get("parent_revision")
            if wanted is not None and wanted != revision["revision_id"]:
                raise RequestRejected(
                    "unsupported",
                    "parent_revision_mismatch",
                    f"requested {wanted}, backend is {revision['revision_id']}; no fallback",
                )
            n_questions = len(request["questions"])
            if limits["max_choices"] is not None and n_questions > limits["max_choices"]:
                raise RequestRejected(
                    "exhausted", "choices_exceeded", f"{n_questions} questions > max_choices {limits['max_choices']}"
                )
            try:
                plan = self.backend.plan(request)
            except BackendUnsupported as exc:
                raise RequestRejected("unsupported", exc.code, exc.message) from exc
            if limits["max_input_tokens"] is not None and plan.max_call_input_tokens > limits["max_input_tokens"]:
                raise RequestRejected(
                    "exhausted",
                    "input_tokens_exceeded",
                    f"longest model input {plan.max_call_input_tokens} tokens > max_input_tokens "
                    f"{limits['max_input_tokens']} (no truncation)",
                )
            if limits["max_inference_calls"] is not None and plan.inference_calls > limits["max_inference_calls"]:
                raise RequestRejected(
                    "exhausted",
                    "inference_calls_exceeded",
                    f"plan needs {plan.inference_calls} calls > max_inference_calls {limits['max_inference_calls']}",
                )
            deadline_ms = limits["deadline_ms"]

            def check_deadline() -> None:
                if deadline_ms is not None and (time.perf_counter() - t_start) * 1000.0 > deadline_ms:
                    raise DeadlineExceeded()

            try:
                result = self.backend.score(request, check_deadline)
                check_deadline()
            except DeadlineExceeded as exc:
                raise RequestRejected(
                    "exhausted", "deadline_exceeded", f"deadline_ms {deadline_ms} exceeded; no partial answers"
                ) from exc
            except BackendUnsupported as exc:
                raise RequestRejected("unsupported", exc.code, exc.message) from exc
        except RequestRejected as rej:
            return self._failure(request, request_id, rej.status, rej.code, rej.message, t_start, state_handle)
        except Exception as exc:  # backend crash: explicit error, never a fallback
            return self._failure(
                request, request_id, "error", "backend_error", f"{type(exc).__name__}: {exc}", t_start, state_handle
            )

        return self._success(request, env, request_id, limits, restriction, revision, plan, result, t_start, state_handle)

    def _base(self, request: Any, request_id: Any) -> dict[str, Any]:
        model = request.get("model") if isinstance(request, dict) else None
        return {"model": model, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def _versions(self) -> dict[str, str]:
        return {
            "adapter": versions.ADAPTER_VERSION,
            "interface": versions.INTERFACE_VERSION,
            "deltas": versions.DELTAS_VERSION if self.enable_v0_1_deltas else "disabled",
            "wire": versions.WIRE_COMPAT,
        }

    def _failure(self, request, request_id, status, code, message, t_start, state_handle) -> dict[str, Any]:
        resp = self._base(request, request_id)
        try:
            revision = self.backend.revision()
        except Exception:  # pragma: no cover - revision itself failed
            revision = {"revision_id": "unknown"}
        resp["ssdm"] = {
            "schema": versions.ENVELOPE_SCHEMA,
            "request_id": request_id,
            "status": status,
            "error": {"code": code, "message": message},
            "abstention": {"abstained": False, "reason": None},
            "raw_probabilities": {},
            "served_probabilities": {},
            "served_config": None,
            "proposed_evidence_refs": [],
            "state_handle": None,
            "diagnostics": {
                "revision_id": revision.get("revision_id"),
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                "inference_calls": 0,
                "state_handle_in": None if state_handle is None else "discarded:stateless",
                "state_reuse": False,
                "versions": self._versions(),
            },
        }
        return resp

    def _success(self, request, env, request_id, limits, restriction, revision, plan, result, t_start, state_handle):
        questions = request["questions"]
        raw, served, entropy, temps = {}, {}, {}, {}
        for qid, q in questions.items():
            option_ids = [o for o, _ in canonical_options(q)]
            q_logits = result.logits[qid]
            if set(q_logits) != set(option_ids):  # backend must score every supplied option
                return self._failure(
                    request, request_id, "error", "backend_error",
                    f"{qid}: backend scored {sorted(q_logits)} but options are {option_ids}", t_start, state_handle,
                )
            vec = [float(q_logits[o]) for o in option_ids]
            if not all(math.isfinite(x) for x in vec):
                return self._failure(
                    request, request_id, "error", "non_finite_logits", f"{qid}: non-finite logits", t_start, state_handle
                )
            k = len(option_ids)
            temp = self.calibration.temperature(k)
            raw[qid] = dict(zip(option_ids, softmax(vec, 1.0)))
            served[qid] = dict(zip(option_ids, softmax(vec, temp)))
            temps[qid] = temp
            entropy[qid] = -sum(p * math.log(p) for p in served[qid].values() if p > 0)

        answers = {qid: wire_answer(q, served[qid]) for qid, q in questions.items()}
        ssdm: dict[str, Any] = {
            "schema": versions.ENVELOPE_SCHEMA,
            "request_id": request_id,
            "status": "ok",
            "error": None,
            "abstention": {"abstained": False, "reason": None},
            "raw_probabilities": raw,
            "served_probabilities": served,
            "served_config": {
                "calibration": self.calibration.identity(),
                "temperature_per_question": temps,
                "policy_masking": "none (served covers every supplied option)",
            },
            "proposed_evidence_refs": [],
            "state_handle": None,
            "diagnostics": {
                "revision_id": revision["revision_id"],
                "model_revision": revision,
                "request_fingerprint": request_fingerprint(request),
                "context_ref": env.get("context_ref"),
                "policy_ref": env.get("policy_ref"),
                "registry_ref": env.get("registry_ref"),
                "audience": env.get("audience"),
                "input_counts": {
                    "questions": len(questions),
                    "options": {qid: len(raw[qid]) for qid in questions},
                    "max_call_input_tokens": plan.max_call_input_tokens,
                    "total_input_tokens": result.total_input_tokens,
                },
                "inference_calls": result.inference_calls,
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                "entropy_nats": entropy,
                "limits_applied": limits,
                "state_handle_in": None if state_handle is None else "discarded:stateless",
                "state_reuse": False,
                "backend": result.extras,
                "versions": self._versions(),
            },
        }
        if self.enable_v0_1_deltas:
            if restriction:
                pcp = {}
                for qid, allowed in restriction.items():
                    mass = sum(served[qid][o] for o in allowed)
                    pcp[qid] = {
                        "probabilities": {o: served[qid][o] / mass for o in allowed} if mass > 0 else None,
                        "allowed": allowed,
                        "excluded_served_mass": 1.0 - mass,
                        "note": "trusted restriction; never substituted for served/benchmark probabilities",
                    }
                ssdm["policy_conditioned_probabilities"] = pcp
            if result.attribution is not None:
                ssdm["attribution"] = {
                    "label": "accounting attribution, not causal effect",
                    "items": result.attribution,
                }
        return {
            "model": request["model"],
            "answers": answers,
            "usage": {"input_tokens": result.total_input_tokens, "output_tokens": 0},
            "ssdm": ssdm,
        }


_DEFAULT: Decider | None = None


def set_default_backend(backend: Backend, **kwargs: Any) -> Decider:
    global _DEFAULT
    _DEFAULT = Decider(backend, **kwargs)
    return _DEFAULT


def decide(request: dict[str, Any], state_handle: Any = None, *, decider: Decider | None = None) -> dict[str, Any]:
    """Module-level entry point: ``decide(request, state_handle=None) -> response``."""
    d = decider or _DEFAULT
    if d is None:
        raise RuntimeError("no backend configured: pass decider= or call set_default_backend()")
    return d.decide(request, state_handle)

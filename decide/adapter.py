"""``decide(request, state_handle=None) -> response``.

Implements the decision adapter of ``interfaces/DECISION_LANGUAGE_V0.md`` (version 0.1).

Transport: the SystemOne-compatible fields (``model``, ``state``, ``questions``)
stay at the top level, untouched, and the canonical v0 fields travel in a
versioned extension envelope ``"ssdm"`` beside them (request and response). A
legacy SystemOne client that ignores ``ssdm`` sees exactly the SystemOne wire
response; the envelope never changes the meaning of the legacy fields.

Request envelope (all optional; legacy requests are accepted without it)::

    "ssdm": {
      "schema": "ssdm.decide.v0.1",
      "request_id": str,             # identity only, not a permission token
      "parent_revision": str | null,  # must equal the backend's revision_id if given
      "context_ref": str, "policy_ref": str, "registry_ref": str,
      "audience": "public" | "private",
      "limits": {"max_input_tokens", "max_questions", "max_options_per_question",
                 "max_program_nodes", "max_inference_calls", "deadline_ms",
                 "max_choices"},  # positive ints; max_choices = legacy alias of max_questions only
      "policy_restriction": {qid: [allowed option ids]},  # nonempty subset, no duplicates
      "provenance": {...}   # planned 0.2; needs enable_experimental_provenance (decide/provenance.py)
    }

Limit units (contract 0.1): ``max_questions`` caps questions per request;
``max_options_per_question`` caps supplied options in each question (the wire
maximum is 255 regardless); ``max_inference_calls`` caps model forward
invocations in this request; input tokens are per model input (row); deadline is
wall-clock ms checked between calls. L-only budgets (``max_proposal_calls``,
``max_repair_calls``) are unknown keys here and rejected.

Response: legacy ``model`` / ``answers`` / ``usage`` plus ``"ssdm"`` with
``status`` (ok | unsupported | exhausted | error), ``error`` {code, message},
``abstention`` (explicit, separate from status), ``raw_probabilities``,
``served_probabilities`` + ``served_config``, ``diagnostics``,
``proposed_evidence_refs`` and ``state_handle``; ``policy_conditioned_probabilities``
when a restriction is supplied. With ``enable_v0_1_deltas``: ``attribution`` (only
if the backend produces accounting attribution).
``semantic_records`` is never emitted until Track D's semantic-interface gate.

Guarantees: no silent truncation, option pruning or model fallback; stateless
(an incoming ``state_handle`` is discarded and the result recomputed, as the
interface prescribes on mismatch); every non-ok status carries a reason code
and no answers.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from . import versions
from .backends.base import Backend, BackendUnsupported, CallBudgetExceeded, DeadlineExceeded
from .calibration import FeatureTemperature, OptionCountTemperature, softmax
from .provenance import validate_provenance
from .schema import (
    canonical_json,
    sha256_text,
    RequestRejected,
    canonical_options,
    request_fingerprint,
    validate_legacy,
    wire_answer,
)

LIMIT_KEYS = (
    "max_input_tokens",
    "max_questions",
    "max_options_per_question",
    "max_choices",  # legacy alias of max_questions only
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
    "provenance",
}
#: Failure messages are fixed text per code; raw exception/backend text never reaches a response.
SAFE_MESSAGES = {
    "backend_error": "the backend failed or returned a malformed result; details withheld",
    "internal_error": "the adapter failed; details withheld",
    "non_finite_logits": "the backend returned non-finite scores; details withheld",
    "control_tokens_in_input": "request text would encode to control tokens; refused",
    "replay_miss": "no recording for this request; no model fallback",
    "tokenization_failed": "tokenization failed; details withheld",
    "deadline_exceeded": "deadline exceeded; no partial answers",
    "inference_calls_exceeded": "the backend needed more inference calls than max_inference_calls; no partial answers",
    "calibration_features_missing": "the backend did not report the features the served calibration needs; refused",
}
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
MAX_MESSAGE_CHARS = 300


def _backend_code(code: Any) -> str:
    return code if isinstance(code, str) and _CODE_RE.fullmatch(code) else "backend_unsupported"


def _backend_message(code: str) -> str:
    return SAFE_MESSAGES.get(code, "the backend cannot serve this request; details withheld")


#: No abstention mechanism exists in the v0 adapter; this is not a measured non-abstention.
ABSTENTION_V0 = {"abstained": False, "reason": None, "mechanism": "not_implemented_v0"}
DEFAULT_LIMITS = {
    "max_input_tokens": 16384,  # Clef encode_record default
    "max_questions": 64,  # Workers AI: 1-64 questions per request
    "max_options_per_question": 255,  # SystemOne wire cardinality
    "max_program_nodes": 8,  # recorded only; decide() emits no program nodes
    "max_inference_calls": 4096,
    "deadline_ms": None,  # no deadline unless the host sets one
}


class Decider:
    """Binds a backend, a calibration config and the optional-delta flag."""

    def __init__(
        self,
        backend: Backend,
        calibration: OptionCountTemperature | FeatureTemperature | None = None,
        enable_v0_1_deltas: bool = False,
        backend_max_input_tokens: int | None = None,
        enable_experimental_provenance: bool = False,
        readout_rules: bool = False,
    ) -> None:
        self.enable_experimental_provenance = enable_experimental_provenance
        #: Opt-in parameter-free readout rules (``decide.readout_rules``, id ``ssdm.readout_rules.r1``): BFCL
        #: exactly-one and SATA top-m on the served probabilities, served under the 32,768-token input cap. Off by
        #: default, so every existing readout's responses are byte-identical.
        self.readout_rules = readout_rules
        self.backend = backend
        self.calibration = calibration or OptionCountTemperature()
        assert_servable = getattr(self.calibration, "assert_servable", None)
        if callable(assert_servable):  # a fitted temperature without a calibration/eval disjointness proof is refused
            assert_servable()
        self.enable_v0_1_deltas = enable_v0_1_deltas
        self.backend_max_input_tokens = backend_max_input_tokens

    def revision(self) -> dict[str, Any]:
        """The served revision: the backend's, plus the calibration when it is versioned into the revision
        (``FeatureTemperature``: ``revision_id = <backend revision_id>+T:<temperature id>``). ``parent_revision``
        is checked against this id, so a client pinned to one fitted temperature never silently gets another.
        The legacy ``OptionCountTemperature`` keeps the backend id (its identity is in ``served_config``)."""
        rev = self.backend.revision()
        if self.readout_rules:
            from .readout_rules import READOUT_RULES_CAP, RULES_ID

            rev = {**rev, "revision_id": f"{rev['revision_id']}+R:{RULES_ID}",
                   "readout_rules": {"id": RULES_ID, "max_input_tokens_default": READOUT_RULES_CAP}}
        if not getattr(self.calibration, "versioned_in_revision", False):
            return rev
        cal = self.calibration.identity()
        return {**rev, "revision_id": f"{rev['revision_id']}+T:{cal['id']}",
                "backend_revision_id": rev["revision_id"], "calibration": cal}

    def _temperature(self, qid: str, k: int, result: Any) -> float:
        needs = getattr(self.calibration, "needs", ())
        if not needs:
            return self.calibration.temperature(k)
        feats = (getattr(result, "features", None) or {}).get(qid)
        try:
            return self.calibration.temperature_for(k, feats)
        except (KeyError, TypeError, ValueError) as exc:
            raise RequestRejected("error", "calibration_features_missing",
                                  SAFE_MESSAGES["calibration_features_missing"]) from exc

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
        if self.readout_rules:
            from .readout_rules import READOUT_RULES_CAP

            limits["max_input_tokens"] = READOUT_RULES_CAP
        for k, v in given.items():
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise RequestRejected("error", "invalid_request", f"limits.{k} must be a positive integer")
        given = dict(given)
        if "max_choices" in given:  # legacy alias: questions per request, never options
            alias = given.pop("max_choices")
            if "max_questions" in given and given["max_questions"] != alias:
                raise RequestRejected(
                    "error", "invalid_request", "limits.max_choices (alias of max_questions) conflicts with max_questions"
                )
            given["max_questions"] = alias
            alias_used = True
        else:
            alias_used = False
        for k, v in given.items():
            limits[k] = v
        limits["_source"] = {
            k: ("request" if k in given else "adapter_default") for k in LIMIT_KEYS if k != "max_choices"
        }
        if alias_used:
            limits["_source"]["max_questions"] = "request:max_choices_alias"
        if self.readout_rules and "max_input_tokens" not in given:
            limits["_source"]["max_input_tokens"] = "readout_rules_cap"
        if self.backend_max_input_tokens is not None and (
            limits["max_input_tokens"] is None or limits["max_input_tokens"] > self.backend_max_input_tokens
        ):
            limits["max_input_tokens"] = self.backend_max_input_tokens
            limits["_source"]["max_input_tokens"] = "backend_context_limit"
        return limits

    @staticmethod
    def _restriction(env: dict[str, Any], request: dict[str, Any]) -> dict[str, list[str]] | None:
        restriction = env.get("policy_restriction")
        if restriction is None:
            return None
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
            if len(set(allowed)) != len(allowed):
                raise RequestRejected("error", "invalid_request", f"policy_restriction.{qid}: duplicate option ids")
            out[qid] = sorted(allowed)
        return out

    # --------------------------------------------------------------------- main
    def decide(self, request: dict[str, Any], state_handle: Any = None) -> dict[str, Any]:
        """Error boundary: every path, including building the success response, returns an
        explicit status. Work is accounted for every attempted inference call."""
        t_start = time.perf_counter()
        env: dict[str, Any] = {}
        request_id = None
        # calls_started: inference calls the backend began (it calls check_deadline before each);
        # result_calls: what a returned ScoreResult reported; score_entered: score() was invoked.
        work: dict[str, Any] = {"calls_started": 0, "result_calls": None, "score_entered": False}
        try:
            if isinstance(request, dict) and isinstance(request.get("ssdm"), dict):
                request_id = request["ssdm"].get("request_id")
            validate_legacy(request)
            env = self._envelope(request)
            request_id = env.get("request_id")
            limits = self._limits(env)
            restriction = self._restriction(env, request)
            provenance = None
            if env.get("provenance") is not None:
                if not self.enable_experimental_provenance:
                    raise RequestRejected(
                        "unsupported", "provenance_not_enabled",
                        "ssdm.provenance is a planned 0.2 extension; this adapter has it disabled",
                    )
                provenance = validate_provenance(request, env["provenance"])
            revision = self.revision()
            wanted = env.get("parent_revision")
            if wanted is not None and wanted != revision["revision_id"]:
                raise RequestRejected(
                    "unsupported",
                    "parent_revision_mismatch",
                    f"requested {wanted}, backend is {revision['revision_id']}; no fallback",
                )
            n_questions = len(request["questions"])
            if limits["max_questions"] is not None and n_questions > limits["max_questions"]:
                raise RequestRejected(
                    "exhausted", "questions_exceeded",
                    f"{n_questions} questions > max_questions {limits['max_questions']}",
                )
            n_opts = max(len(canonical_options(q)) for q in request["questions"].values())
            if limits["max_options_per_question"] is not None and n_opts > limits["max_options_per_question"]:
                raise RequestRejected(
                    "exhausted", "options_exceeded",
                    f"a question has {n_opts} options > max_options_per_question "
                    f"{limits['max_options_per_question']} (no pruning)",
                )
            try:
                plan = self.backend.plan(request)
            except BackendUnsupported as exc:
                code = _backend_code(exc.code)
                raise RequestRejected("unsupported", code, _backend_message(code)) from exc
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

            def expired() -> bool:
                return deadline_ms is not None and (time.perf_counter() - t_start) * 1000.0 > deadline_ms

            max_calls = limits["max_inference_calls"]

            def check_deadline() -> None:  # called by backends before each inference call
                if expired():
                    work["stopped_by_deadline"] = True  # exact: this callback counted every started call
                    raise DeadlineExceeded()
                # Enforce the call budget at run time, not only against the plan: a backend whose
                # plan under-reports its calls is stopped before the first call over the cap.
                if max_calls is not None and work["calls_started"] >= max_calls:
                    work["stopped_by_call_budget"] = True
                    raise CallBudgetExceeded()
                work["calls_started"] += 1

            work["score_entered"] = True
            try:
                result = self.backend.score(request, check_deadline)
                work["result_calls"] = getattr(result, "inference_calls", None)
                if expired():
                    raise DeadlineExceeded()
                reported = work["result_calls"]
                if (
                    max_calls is not None
                    and isinstance(reported, int)
                    and not isinstance(reported, bool)
                    and reported > max_calls
                ):  # backend did more calls than the cap without using the callback for each
                    raise CallBudgetExceeded()
            except DeadlineExceeded as exc:
                raise RequestRejected("exhausted", "deadline_exceeded", SAFE_MESSAGES["deadline_exceeded"]) from exc
            except CallBudgetExceeded as exc:
                raise RequestRejected(
                    "exhausted", "inference_calls_exceeded", SAFE_MESSAGES["inference_calls_exceeded"]
                ) from exc
            except BackendUnsupported as exc:
                code = _backend_code(exc.code)
                raise RequestRejected("unsupported", code, _backend_message(code)) from exc
            return self._success(
                request, env, request_id, limits, restriction, provenance, revision, plan, result, t_start,
                state_handle, work,
            )
        except RequestRejected as rej:
            return self._failure(request, request_id, rej.status, rej.code, rej.message[:MAX_MESSAGE_CHARS],
                                 t_start, state_handle, work)
        except Exception:  # never a fallback, never the raw exception text
            code = "backend_error" if work["score_entered"] else "internal_error"
            try:
                return self._failure(request, request_id, "error", code, SAFE_MESSAGES[code], t_start,
                                     state_handle, work)
            except Exception:  # pragma: no cover - last-resort fixed response
                return {"model": None, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                        "error": {"status": "error", "code": "internal_error",
                                  "message": SAFE_MESSAGES["internal_error"]}}

    @staticmethod
    def _calls(work: dict[str, Any] | None) -> tuple[int | None, str]:
        """Attempted inference calls. Unknown work is reported as null, never as zero."""
        if not work or not work["score_entered"]:
            return 0, "no inference call attempted"
        started, reported = work["calls_started"], work["result_calls"]
        if isinstance(reported, int) and not isinstance(reported, bool) and reported >= 0:
            return max(started, reported), "max(calls started, calls reported by the backend)"
        if started > 0 or work.get("stopped_by_deadline") or work.get("stopped_by_call_budget"):
            return started, "calls started before failure/deadline/call budget (counted by the adapter callback)"
        return None, "score() was attempted but the backend failed before reporting work; count unknown"

    @staticmethod
    def _provenance_diag(provenance, plan) -> dict[str, Any]:
        if provenance is None:
            return {"supplied": False, "applied": False}
        from .untrusted import UNTRUSTED_ENCODER_ID

        labels = plan.prompt_token_labels
        return {
            "supplied": True,
            "applied": bool(plan.provenance_applied),
            "schema": provenance["schema"],
            "serializer": provenance["serializer"],
            "default_trust": provenance["default_trust"],
            "path_encoding": provenance["path_encoding"],
            "segments": provenance["segments"],
            "instruction_tokens": None if labels is None else labels.get("instruction", 0),
            "data_tokens": None if labels is None else labels.get("data", 0),
            "template_tokens": None if labels is None else labels.get("template", 0),
            "encoder_id": UNTRUSTED_ENCODER_ID,
        }

    def _base(self, request: Any, request_id: Any) -> dict[str, Any]:
        model = request.get("model") if isinstance(request, dict) else None
        return {"model": model, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def _versions(self) -> dict[str, str]:
        return {
            "adapter": versions.ADAPTER_VERSION,
            "interface": versions.INTERFACE_VERSION,
            "deltas": versions.DELTAS_VERSION if self.enable_v0_1_deltas else "disabled",
            "experimental_provenance": "enabled" if self.enable_experimental_provenance else "disabled",
            "wire": versions.WIRE_COMPAT,
        }

    def _failure(self, request, request_id, status, code, message, t_start, state_handle,
                 work: dict[str, Any] | None = None) -> dict[str, Any]:
        calls, calls_note = self._calls(work)
        resp = self._base(request, request_id)
        # Legacy-visible failure signal (SystemOne clients ignore ``ssdm``).
        resp["error"] = {"status": status, "code": code, "message": message}
        try:
            revision = self.revision()
        except Exception:  # pragma: no cover - revision itself failed
            revision = {"revision_id": "unknown"}
        resp["ssdm"] = {
            "schema": versions.ENVELOPE_SCHEMA,
            "request_id": request_id,
            "status": status,
            "error": {"code": code, "message": message},
            "abstention": dict(ABSTENTION_V0),
            "raw_probabilities": {},
            "served_probabilities": {},
            "served_config": None,
            "proposed_evidence_refs": [],
            "state_handle": None,
            "diagnostics": {
                "revision_id": revision.get("revision_id"),
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                "inference_calls": calls,
                "inference_calls_accounting": calls_note,
                "state_handle_in": None if state_handle is None else "discarded:stateless",
                "state_reuse": False,
                "versions": self._versions(),
            },
        }
        return resp

    def _success(
        self, request, env, request_id, limits, restriction, provenance, revision, plan, result, t_start,
        state_handle, work,
    ):
        questions = request["questions"]
        raw, served, entropy, temps = {}, {}, {}, {}
        logits = getattr(result, "logits", None)
        calls, _ = self._calls(work)
        total_tokens = getattr(result, "total_input_tokens", None)
        if not isinstance(logits, dict) or set(logits) != set(questions) or not (
            isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens >= 0
        ):
            return self._failure(request, request_id, "error", "backend_error", SAFE_MESSAGES["backend_error"],
                                 t_start, state_handle, work)
        for qid, q in questions.items():
            option_ids = [o for o, _ in canonical_options(q)]
            q_logits = logits[qid]
            if not isinstance(q_logits, dict) or set(q_logits) != set(option_ids):  # every supplied option
                return self._failure(request, request_id, "error", "backend_error", SAFE_MESSAGES["backend_error"],
                                     t_start, state_handle, work)
            try:
                vec = [float(q_logits[o]) for o in option_ids]
            except (TypeError, ValueError):
                vec = [float("nan")]
            if not all(math.isfinite(x) for x in vec):
                return self._failure(request, request_id, "error", "non_finite_logits",
                                     SAFE_MESSAGES["non_finite_logits"], t_start, state_handle, work)
            k = len(option_ids)
            try:
                temp = self._temperature(qid, k, result)
            except RequestRejected as rej:
                return self._failure(request, request_id, rej.status, rej.code, rej.message, t_start,
                                     state_handle, work)
            raw[qid] = dict(zip(option_ids, softmax(vec, 1.0)))
            served[qid] = dict(zip(option_ids, softmax(vec, temp)))
            temps[qid] = temp
            entropy[qid] = -sum(p * math.log(p) for p in served[qid].values() if p > 0)

        answers = {qid: wire_answer(q, served[qid]) for qid, q in questions.items()}
        rules_diag = None
        if self.readout_rules:
            from .readout_rules import apply_rules

            rules_diag = apply_rules(request, answers, served)
        ssdm: dict[str, Any] = {
            "schema": versions.ENVELOPE_SCHEMA,
            "request_id": request_id,
            "status": "ok",
            "error": None,
            "abstention": dict(ABSTENTION_V0),
            "raw_probabilities": raw,
            "served_probabilities": served,
            "served_config": {
                "calibration": self.calibration.identity(),
                "temperature_per_question": temps,
                "policy_masking": "none (served covers every supplied option)",
                **({"readout_rules": rules_diag} if rules_diag is not None else {}),
            },
            "proposed_evidence_refs": [],
            "state_handle": None,
            "diagnostics": {
                "revision_id": revision["revision_id"],
                "model_revision": revision,
                "request_fingerprint": request_fingerprint(request),
                "request_fingerprint_algorithm": (
                    versions.FINGERPRINT_ALGORITHM if provenance is None
                    else versions.FINGERPRINT_ALGORITHM_PROVENANCE
                ),
                "provenance": self._provenance_diag(provenance, plan),
                "probability_stages": {
                    "raw": {"domain": "all_supplied_options", "transform": "softmax(logits, T=1)"},
                    "served": {"domain": "all_supplied_options",
                               "transform": self.calibration.identity()["id"]},
                    "policy_conditioned": "present" if restriction else "absent (no restriction supplied)",
                },
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
                "inference_calls": calls,
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                "entropy_nats": entropy,
                "limits_applied": limits,
                "state_handle_in": None if state_handle is None else "discarded:stateless",
                "state_reuse": False,
                "backend": result.extras,
                "versions": self._versions(),
            },
        }
        if restriction:
            pcp = {}
            for qid, allowed in restriction.items():
                mass = sum(served[qid][o] for o in allowed)
                stage = {
                    "allowed": allowed,
                    "domain": "all_supplied_options (zero on excluded)",
                    "restriction_id": sha256_text(canonical_json({"policy_ref": env.get("policy_ref"),
                                                                  "question": qid, "allowed": allowed})),
                    "excluded_served_mass": 1.0 - mass,
                    "note": "trusted restriction; never substituted for served/benchmark probabilities",
                }
                if mass > 0:
                    stage["status"] = "ok"
                    stage["probabilities"] = {o: (served[qid][o] / mass if o in allowed else 0.0)
                                              for o in served[qid]}
                else:
                    stage["status"] = "unavailable_zero_served_mass"
                    stage["probabilities"] = None
                pcp[qid] = stage
            ssdm["policy_conditioned_probabilities"] = pcp
        if self.enable_v0_1_deltas:
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


def select_actions(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Host action boundary for choice questions; never reads legacy ``answers``.

    Require an explicit host restriction for every action (allow all IDs explicitly
    for an unrestricted action). Bind the response to the current request and policy,
    then use only a valid conditioned stage. Any failure defers the entire request.
    This checks transport/policy consistency, not model correctness or a certificate.
    """
    def defer(code: str) -> dict[str, Any]:
        return {"status": "unsupported", "error": code, "action": "defer", "actions": {}, "answers": {},
                "abstention": {"abstained": True, "reason": code, "mechanism": "policy_action_boundary"}}

    def distribution(probs: Any, ids: set[str]) -> bool:
        return (isinstance(probs, dict) and set(probs) == ids
                and all(isinstance(p, (int, float)) and not isinstance(p, bool)
                        and math.isfinite(p) and p >= 0 for p in probs.values())
                and abs(sum(probs.values()) - 1.0) <= 1e-9)

    try:
        validate_legacy(request)
        env = request.get("ssdm") or {}
        restriction = Decider._restriction(env, request)
        questions = request["questions"]
        if not restriction or set(restriction) != set(questions):
            return defer("policy_conditioned_unavailable")
        if any(q["type"] != "choice" for q in questions.values()):
            return defer("action_type_unsupported")
        ss = response.get("ssdm")
        if (not isinstance(ss, dict) or ss.get("schema") != versions.ENVELOPE_SCHEMA
                or ss.get("status") != "ok" or response.get("error") is not None):
            return defer("decision_unavailable")
        abstention = ss.get("abstention")
        if not isinstance(abstention, dict) or abstention.get("abstained") is not False:
            return defer("decision_abstained")
        diag = ss.get("diagnostics") or {}
        algorithm = (versions.FINGERPRINT_ALGORITHM if env.get("provenance") is None
                     else versions.FINGERPRINT_ALGORITHM_PROVENANCE)
        if (response.get("model") != request["model"] or ss.get("request_id") != env.get("request_id")
                or diag.get("request_fingerprint_algorithm") != algorithm
                or diag.get("request_fingerprint") != request_fingerprint(request)
                or any(diag.get(k) != env.get(k) for k in ("context_ref", "policy_ref", "registry_ref", "audience"))
                or (env.get("parent_revision") is not None
                    and diag.get("revision_id") != env["parent_revision"])):
            return defer("stale_decision")
        pcp, served = ss.get("policy_conditioned_probabilities"), ss.get("served_probabilities")
        if not isinstance(pcp, dict) or not isinstance(served, dict):
            return defer("policy_conditioned_unavailable")
        answers, actions = {}, {}
        for qid, q in questions.items():
            stage = pcp.get(qid)
            if not isinstance(stage, dict) or stage.get("status") != "ok":
                return defer("policy_conditioned_unavailable")
            allowed = restriction[qid]
            expected_id = sha256_text(canonical_json({"policy_ref": env.get("policy_ref"),
                                                      "question": qid, "allowed": allowed}))
            if stage.get("restriction_id") != expected_id or stage.get("allowed") != allowed:
                return defer("stale_policy_conditioned")
            ids = {o for o, _ in canonical_options(q)}
            probs, base = stage.get("probabilities"), served.get(qid)
            if not distribution(probs, ids) or not distribution(base, ids):
                return defer("invalid_policy_conditioned")
            mass = sum(base[o] for o in allowed)
            if mass <= 0:
                return defer("policy_conditioned_unavailable")
            if any(probs[o] != 0.0 for o in ids - set(allowed)) or any(
                    abs(probs[o] - base[o] / mass) > 1e-9 for o in allowed):
                return defer("invalid_policy_conditioned")
            best = max(allowed, key=lambda o: probs[o])  # allowed is sorted; ties use smallest ID
            actions[qid] = best
            answers[qid] = {"type": "choice", "choice": best, "probabilities": dict(probs)}
        return {"status": "ok", "error": None, "action": "execute", "actions": actions, "answers": answers,
                "abstention": {"abstained": False, "reason": None, "mechanism": "policy_action_boundary"}}
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError, RequestRejected):
        return defer("invalid_action_response")


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

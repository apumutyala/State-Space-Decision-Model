"""Discovery-only oracle-scaffolded relation screen; scorer files are never accepted here.

The fixed scaffold chooses operation and operands from exact task wording plus visible audience-filtered records.
The relation request is forwarded unchanged to the supplied Decider. Its response is preserved; no relation oracle,
retry, correction or fallback exists in this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Sequence

from language_growth.parent_bridge import DECISION_SCHEMA, FINGERPRINT_ALGORITHM, legacy_request_fingerprint
from language_growth.planner import plan_checked_answer


SCREEN_SCHEMA = "ssdm.l0i_causal_relation_screen.v1"
DISCOVERY_INPUT_COUNT = 1152
PINNED_DISCOVERY_INPUT_SHA256 = "5bb73a63c7f6ed085b850e8680576ac4670be5d74c36e5da9af188c1d8f7065f"
MIN_START_AVAILABLE_BYTES = 9 * 1024**3
MIN_RUNTIME_AVAILABLE_BYTES = 3 * 1024**3
SCREEN_OFFSETS = (1, 2, 5, 7, 8, 11, 13, 14, 17)
RELATION_DESCRIPTIONS = {"less_than", "equal", "greater_than"}
_TASK_RE = re.compile(
    r"Compare (?P<left>.+?)'s (?P<field>[A-Za-z_][A-Za-z0-9_]*) with "
    r"(?P<right>.+?)'s (?P=field) in October 2026, in that order\."
)
_SCRIPTED_SLOTS = ("operation", "left", "right")
_COMPARE_DESCRIPTION = "Compare two selected typed values and assert the selected relation"
_POLICY_MASKING = "none (served covers every supplied option)"
_LAYOUT_BACKEND = "rwkv7_layout_bound_causal_experiment"


class ScaffoldError(ValueError):
    """The fixed input-derived operation/operand scaffold cannot safely serve this request."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _available_memory_bytes() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


def _relation_request(request: dict[str, Any]) -> tuple[str, str, str, list[dict[str, Any]]]:
    try:
        state = json.loads(request["state"])
        task = state["task_input"]
        records = state["records"]
        context_ref = state["context_ref"]
        request_context = request["ssdm"]["context_ref"]
    except (KeyError, TypeError, ValueError):
        raise ScaffoldError("invalid serialized task/evidence state") from None
    if (not isinstance(state, dict) or not isinstance(task, str) or not isinstance(records, list)
            or context_ref != request_context or state.get("accepted_partial_program") != []
            or not isinstance(state.get("proposed_slot_choices"), list)):
        raise ScaffoldError("state/envelope context or prefix mismatch")
    match = _TASK_RE.fullmatch(task)
    if match is None:
        raise ScaffoldError("unsupported comparison wording")
    return match.group("left"), match.group("right"), match.group("field"), records


def _unique_record(records: list[dict[str, Any]], entity: str, field: str) -> str:
    matches = [record for record in records
               if isinstance(record, dict) and record.get("entity_ref") == entity]
    if len(matches) != 1:
        raise ScaffoldError("entity must bind to exactly one visible record")
    fields = matches[0].get("fields")
    if not isinstance(fields, dict) or field not in fields:
        raise ScaffoldError("named field is not visible on the selected record")
    record_ref = matches[0].get("record_ref")
    if not isinstance(record_ref, str) or not record_ref:
        raise ScaffoldError("visible record has no exact record_ref")
    return record_ref


def _unique_description(options: Any, expected: str) -> str:
    if not isinstance(options, dict) or not options or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in options.items()):
        raise ScaffoldError("choice options must be a nonempty ID-to-description object")
    matches = [option_id for option_id, description in options.items() if description == expected]
    if len(matches) != 1:
        raise ScaffoldError("expected exactly one offered description")
    return matches[0]


def _scaffold_response(request: dict[str, Any], qid: str, options: dict[str, str], selected: str,
                       *, parent_revision: str, calibration_identity: dict[str, Any], temperature: float) -> dict[str, Any]:
    probabilities = {option_id: float(option_id == selected) for option_id in options}
    rid = request["ssdm"]["request_id"]
    fingerprint = legacy_request_fingerprint(request)
    return {
        "model": request.get("model"),
        "answers": {qid: {"type": "choice", "choice": selected,
                          "confidence": 1.0, "probabilities": dict(probabilities)}},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "ssdm": {
            "schema": DECISION_SCHEMA, "request_id": rid, "status": "ok", "error": None,
            "abstention": {"abstained": False, "reason": None, "mechanism": "not_implemented_v0"},
            "raw_probabilities": {qid: dict(probabilities)},
            "served_probabilities": {qid: dict(probabilities)},
            "served_config": {"calibration": calibration_identity,
                              "temperature_per_question": {qid: float(temperature)},
                              "policy_masking": _POLICY_MASKING},
            "proposed_evidence_refs": [], "state_handle": None,
            "diagnostics": {
                "revision_id": parent_revision, "request_fingerprint": fingerprint,
                "request_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
                "provenance": {"supplied": False, "applied": False},
                "state_handle_in": None, "state_reuse": False, "inference_calls": 0,
                "backend": {"kind": "input_derived_scaffold", "scripted": True,
                            "parent_selection": False},
            },
        },
    }


class RelationScaffoldProvider:
    """Three fixed input-derived planner choices; delegate the fourth relation decision to the parent unchanged."""

    def __init__(self, parent_decider):
        if not callable(getattr(parent_decider, "decide", None)) or not callable(getattr(parent_decider, "revision", None)):
            raise ValueError("a pinned Decider-like parent is required")
        revision = parent_decider.revision()
        calibration = getattr(parent_decider, "calibration", None)
        identity = getattr(calibration, "identity", None)
        if (not isinstance(revision, dict) or not isinstance(revision.get("revision_id"), str)
                or not isinstance(revision.get("model_id"), str) or not callable(identity)):
            raise ValueError("parent revision and calibration identity are required")
        self.parent = parent_decider
        self.parent_revision = revision["revision_id"]
        self.model_id = revision["model_id"]
        self.calibration_identity = identity()
        self.history: list[dict[str, str]] = []
        self.scaffold_events: list[dict[str, Any]] = []
        self.parent_relation_events: list[dict[str, Any]] = []
        self._stage = 0
        self._request_identity_sha256: str | None = None

    def _temperature(self, option_count: int) -> float:
        calibration = self.parent.calibration
        if getattr(calibration, "needs", ()):
            # The one-hot scaffold distribution is unchanged by any positive temperature. Feature calibrators are
            # identified exactly in the envelope; only the actual parent relation response uses model features.
            return 1.0
        temperature = getattr(calibration, "temperature", None)
        value = temperature(option_count) if callable(temperature) else 1.0
        if type(value) not in (int, float) or value <= 0:
            raise ScaffoldError("scripted calibration temperature is invalid")
        return float(value)

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict) or not isinstance(request.get("questions"), dict) or len(request["questions"]) != 1:
            raise ScaffoldError("one planner choice question is required")
        qid, question = next(iter(request["questions"].items()))
        if not isinstance(qid, str) or not isinstance(question, dict) or question.get("type") != "choice":
            raise ScaffoldError("only one choice question is supported")
        suffix = qid.rsplit("-", 1)[-1]
        expected_sequence = (*_SCRIPTED_SLOTS, "relation")
        if self._stage >= len(expected_sequence) or suffix != expected_sequence[self._stage]:
            raise ScaffoldError("planner slots must occur exactly operation, left, right, relation")
        try:
            state = json.loads(request["state"])
            prefix = state["proposed_slot_choices"]
        except (KeyError, TypeError, ValueError):
            raise ScaffoldError("proposed prefix is missing or malformed") from None
        if not isinstance(state, dict):
            raise ScaffoldError("serialized task state must be an object")
        state_identity = {key: value for key, value in state.items() if key != "proposed_slot_choices"}
        envelope_identity = {key: value for key, value in request["ssdm"].items() if key != "request_id"}
        request_identity = _sha(_canonical({"model": request.get("model"),
                                            "state": state_identity,
                                            "envelope": envelope_identity}))
        if self._request_identity_sha256 is None:
            self._request_identity_sha256 = request_identity
        elif request_identity != self._request_identity_sha256:
            raise ScaffoldError("task/evidence/context/model identity changed across planner slots")
        if prefix != self.history:
            raise ScaffoldError("proposal prefix differs from exact prior scaffold choices")
        options = question.get("criteria")
        if not isinstance(options, dict):
            raise ScaffoldError("choice criteria are missing")

        if suffix == "relation":
            descriptions = list(options.values())
            if (question.get("instructions") != "Select the comparison relation."
                    or len(options) != 3 or len(set(descriptions)) != 3
                    or set(descriptions) != {f"Relation {name}" for name in RELATION_DESCRIPTIONS}):
                raise ScaffoldError("relation choice must retain the complete three-relation domain")
            _relation_request(request)  # revalidate the same exact task/context before forwarding to the parent
            try:
                available = _available_memory_bytes()
            except Exception as exc:
                self.parent_relation_events.append({
                    "qid": qid, "request_sha256": _sha(_canonical(request)),
                    "context_ref": request["ssdm"].get("context_ref"),
                    "status": "runtime_memory_read_failed", "exception_type": type(exc).__name__,
                })
                raise ScaffoldError("runtime memory telemetry unavailable") from None
            if available < MIN_RUNTIME_AVAILABLE_BYTES:
                self.parent_relation_events.append({
                    "qid": qid, "request_sha256": _sha(_canonical(request)),
                    "context_ref": request["ssdm"].get("context_ref"),
                    "status": "runtime_memory_guard", "available_memory_bytes": available,
                    "required_memory_bytes": MIN_RUNTIME_AVAILABLE_BYTES,
                })
                raise ScaffoldError("runtime memory guard refused relation inference")
            self._stage += 1
            forwarded_request = deepcopy(request)
            event = {"qid": qid, "request_sha256": _sha(_canonical(forwarded_request)),
                     "request_identity_sha256": self._request_identity_sha256,
                     "request": deepcopy(forwarded_request), "context_ref": request["ssdm"].get("context_ref"),
                     "status": "forwarding"}
            try:
                response = self.parent.decide(deepcopy(forwarded_request))
            except Exception as exc:
                event.update({"status": "exception", "exception_type": type(exc).__name__})
                self.parent_relation_events.append(event)
                raise
            event["status"] = "returned"
            event["response"] = deepcopy(response)
            self.parent_relation_events.append(event)
            return response  # The parent's response is deliberately not edited, repaired, or replaced.

        expected_previous = list(_SCRIPTED_SLOTS[:self._stage])
        if [item.get("slot") for item in prefix if isinstance(item, dict)] != expected_previous:
            raise ScaffoldError("planner slot order or prefix length changed")
        if question.get("instructions") != {
                "operation": "Select the semantic operation for the user's task.",
                "left": "Select the left typed operand.",
                "right": "Select the right typed operand.",
        }[suffix]:
            raise ScaffoldError("unexpected planner question for scaffold stage")

        left_entity, right_entity, field, records = _relation_request(request)
        left_ref = _unique_record(records, left_entity, field)
        right_ref = _unique_record(records, right_entity, field)
        if suffix == "operation":
            selected = _unique_description(options, _COMPARE_DESCRIPTION)
        elif suffix == "left":
            selected = _unique_description(options, f"Fact {left_ref} field {field}")
        else:
            selected = _unique_description(options, f"Fact {right_ref} field {field}")
        description = options[selected]
        self.history.append({"slot": suffix, "option_id": selected, "description": description})
        event = {"qid": qid, "slot": suffix, "option_id": selected, "description": description,
                 "source": "scripted_input_scaffold", "model_inference_calls": 0,
                 "request_identity_sha256": self._request_identity_sha256}
        self.scaffold_events.append(event)
        self._stage += 1
        return _scaffold_response(request, qid, options, selected, parent_revision=self.parent_revision,
                                 calibration_identity=self.calibration_identity,
                                 temperature=self._temperature(len(options)))


def select_screen_inputs(inputs: Sequence[dict[str, Any]], *, expected_input_sha256: str
                         ) -> tuple[list[tuple[int, dict[str, Any]]], str]:
    """Select the fixed nine discovery inputs; accepts no scorer labels or manifest objects."""
    if not isinstance(inputs, (list, tuple)) or len(inputs) != DISCOVERY_INPUT_COUNT:
        raise ValueError("the pinned discovery input split must contain exactly 1,152 inputs")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"task_input", "host", "catalog"}:
            raise ValueError("only task_input, host and catalog inputs are permitted")
    input_sha = _sha(b"\n".join(_canonical(item) for item in inputs) + b"\n")
    if (not isinstance(expected_input_sha256, str) or len(expected_input_sha256) != 64
            or expected_input_sha256 != PINNED_DISCOVERY_INPUT_SHA256
            or input_sha != expected_input_sha256):
        raise ValueError("discovery input hash does not match the pinned manifest")
    selected = [(offset, deepcopy(inputs[offset])) for offset in SCREEN_OFFSETS]
    expected_tasks = (("forward_compare", "price"), ("reverse_compare", "price"),
                      ("unrelated_compare", "warranty_months")) * 3
    for (_offset, item), (kind, field) in zip(selected, expected_tasks):
        task = item["task_input"]
        parsed = _TASK_RE.fullmatch(task) if isinstance(task, str) else None
        if parsed is None or parsed.group("field") != field:
            raise ValueError("discovery offset/order/field differs from the pinned natural-contrast screen")
        if kind == "reverse_compare" and "in that order" not in task:
            raise ValueError("reverse comparison must preserve the explicit operand-order instruction")
    if not (selected[0][1]["task_input"] == selected[3][1]["task_input"] == selected[6][1]["task_input"]
            and selected[1][1]["task_input"] == selected[4][1]["task_input"] == selected[7][1]["task_input"]
            and selected[2][1]["task_input"] == selected[5][1]["task_input"] == selected[8][1]["task_input"]):
        raise ValueError("low/equal/high relation wording must remain fixed")
    return selected, input_sha


def run_relation_screen(inputs: Sequence[dict[str, Any]], parent_factory, *, expected_input_sha256: str,
                        expected_parent_revision: str,
                        expected_calibration_identity: dict[str, Any]) -> dict[str, Any]:
    """Run the nine selected discovery inputs via the normal checked planner; there is no scorer argument.

    The factory is called only after the 9 GiB start reserve passes. Before each parent relation call, the provider
    rechecks the 3 GiB runtime floor. Tests patch the private memory reader; production callers cannot pass a fake
    memory value through this API.
    """
    selected, input_sha = select_screen_inputs(inputs, expected_input_sha256=expected_input_sha256)
    if (not isinstance(expected_parent_revision, str) or not expected_parent_revision
            or not isinstance(expected_calibration_identity, dict) or not expected_calibration_identity):
        raise ValueError("frozen parent revision and calibration identity pins are required")
    base = {"schema": SCREEN_SCHEMA,
            "claim": "oracle_scaffolded_relation_selection_diagnostic_not_end_to_end_language_or_causal_fidelity",
            "split": "semantic_axis_discovery_only", "input_count": len(inputs),
            "inputs_sha256": input_sha, "selected_offsets": list(SCREEN_OFFSETS),
            "expected_parent_revision": expected_parent_revision,
            "expected_calibration_identity": deepcopy(expected_calibration_identity),
            "available_memory_start_bytes": None,
            "required_start_memory_bytes": MIN_START_AVAILABLE_BYTES,
            "results": []}
    try:
        available_start = _available_memory_bytes()
    except Exception as exc:
        return {**base, "status": "incomplete", "stop_reason": "start_memory_read_failed",
                "memory_exception_type": type(exc).__name__}
    base["available_memory_start_bytes"] = available_start
    if available_start < MIN_START_AVAILABLE_BYTES:
        return {**base, "status": "incomplete", "stop_reason": "start_memory_guard"}
    if not callable(parent_factory):
        return {**base, "status": "incomplete", "stop_reason": "parent_factory_required"}
    try:
        parent_decider = parent_factory()
    except Exception as exc:
        return {**base, "status": "incomplete", "stop_reason": "parent_load_failed",
                "parent_load_exception_type": type(exc).__name__}
    try:
        revision = parent_decider.revision()
        calibration = parent_decider.calibration
        calibration_identity = calibration.identity()
    except Exception as exc:
        return {**base, "status": "incomplete", "stop_reason": "parent_identity_failed",
                "parent_identity_exception_type": type(exc).__name__}
    if (not isinstance(revision, dict) or not isinstance(revision.get("model_id"), str)
            or not isinstance(revision.get("revision_id"), str)
            or revision.get("backend") != _LAYOUT_BACKEND or revision.get("edit_id") is not None
            or revision.get("edit_sign") is not None
            or type(revision.get("max_context_tokens")) is not int
            or revision["max_context_tokens"] > 2048
            or revision.get("layout_version") != "m2-layout-v2"
            or not isinstance(revision.get("runtime_identity_sha256"), str)
            or len(revision["runtime_identity_sha256"]) != 64):
        return {**base, "status": "incomplete", "stop_reason": "backend_not_unchanged_layout_bound_parent",
                "parent_revision": revision}
    if revision["revision_id"] != expected_parent_revision:
        return {**base, "status": "incomplete", "stop_reason": "parent_revision_mismatch",
                "parent_revision": revision}
    if calibration_identity != expected_calibration_identity:
        return {**base, "status": "incomplete", "stop_reason": "calibration_identity_mismatch",
                "parent_revision": revision, "calibration": calibration_identity}
    expected_config = {"calibration": expected_calibration_identity, "policy_masking": _POLICY_MASKING}
    results = []
    stop_reason = None
    for offset, item in selected:
        if parent_decider.revision() != revision:
            stop_reason = "parent_revision_changed"
            break
        scaffold = RelationScaffoldProvider(parent_decider)
        result = plan_checked_answer(item["task_input"], item["host"], item["catalog"], scaffold,
                                     model=revision["model_id"], parent_revision=revision["revision_id"],
                                     expected_served_config=expected_config,
                                     allow_repair=False, candidate_style="record_ref")
        results.append({
            "input_offset": offset,
            "input_sha256": _sha(_canonical(item)),
            "context_ref": item["host"]["context_ref"],
            "status": result.status,
            "reason": result.reason,
            "text": result.text,
            "program": result.program,
            "planner_trace": result.trace,
            "provider_calls": result.provider_calls,
            "model_forward_calls": result.forward_calls,
            "scripted_scaffold": scaffold.scaffold_events,
            "parent_relation_calls": scaffold.parent_relation_events,
        })
        memory_stop = next((event.get("status") for event in scaffold.parent_relation_events
                            if event.get("status") in ("runtime_memory_guard", "runtime_memory_read_failed")), None)
        if memory_stop is not None:
            stop_reason = memory_stop
            break
    return {**base, "status": "complete" if stop_reason is None and len(results) == len(selected) else "incomplete",
            "stop_reason": stop_reason, "parent_revision": revision,
            "calibration": expected_config["calibration"], "results": results}

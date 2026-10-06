"""Stateless Track L serialization for the legacy Track D decision wire.

Only the host's audience-filtered snapshot reaches ``state``.  Evaluator
labels, authored programs, expectations, and arbitrary host metadata are never
arguments to this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SERIALIZATION_REF = "ssdm.track_l_state.v0.1"
MAX_D_OPTIONS = 255


class BridgeError(ValueError):
    """A bounded precondition failure before any provider call."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BridgeError("invalid_host")
    return value


def _positive_limit(limits: dict[str, Any], key: str, default: int) -> int:
    value = limits.get(key, default)
    if type(value) is not int or value < 1:
        raise BridgeError("invalid_limit")
    return value


def planner_limits(host: dict[str, Any]) -> dict[str, int]:
    """L call budgets and D per-request budgets have separate units."""
    limits = _object(_object(host).get("limits"))
    out = {
        "max_program_nodes": _positive_limit(limits, "max_program_nodes", 8),
        "max_proposal_calls": _positive_limit(limits, "max_proposal_calls", 8),
        "max_repair_calls": limits.get("max_repair_calls", 0),
        "max_repair_attempts": limits.get("max_repair_attempts", 1),
        "max_questions": _positive_limit(limits, "max_questions", 1),
        "max_options_per_question": _positive_limit(limits, "max_options_per_question", MAX_D_OPTIONS),
    }
    if type(out["max_repair_calls"]) is not int or out["max_repair_calls"] < 0:
        raise BridgeError("invalid_limit")
    if type(out["max_repair_attempts"]) is not int or out["max_repair_attempts"] not in (0, 1):
        raise BridgeError("invalid_limit")
    if "max_choices" in limits and limits["max_choices"] != out["max_questions"]:
        raise BridgeError("conflicting_limit_alias")
    if out["max_options_per_question"] > MAX_D_OPTIONS:
        raise BridgeError("invalid_limit")
    return out


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BridgeError("invalid_host")
    return value


def _typed_field(raw: Any, *, constant: bool) -> dict[str, Any]:
    descriptor = _object(raw)
    required = {"value", "value_type", "unit", "temporal_scope"}
    allowed = required | {"quantity_kind"}
    if not required <= set(descriptor) or not set(descriptor) <= allowed or (constant and set(descriptor) != allowed):
        raise BridgeError("invalid_host")
    kind, value = descriptor["value_type"], descriptor["value"]
    valid = ((kind == "integer" and type(value) is int)
             or (kind == "boolean" and type(value) is bool)
             or (kind == "string" and isinstance(value, str))
             or (kind == "decimal" and isinstance(value, str)
                 and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value) is not None))
    if not valid or (descriptor["unit"] is not None and (not isinstance(descriptor["unit"], str)
                                                     or not descriptor["unit"])):
        raise BridgeError("invalid_host")
    _name(descriptor["temporal_scope"])
    if "quantity_kind" in descriptor:
        _name(descriptor["quantity_kind"])
    return {key: descriptor[key] for key in descriptor if key in allowed}


def _query_constants(host: dict[str, Any], registered_fields: set[str]) -> dict[str, dict[str, Any]]:
    constants = _object(host.get("query_constants", {}))
    result = {}
    for ref, raw in constants.items():
        _name(ref)
        descriptor = _object(raw)
        if set(descriptor) == {"entity_ref", "field"}:
            _name(descriptor["entity_ref"])
            if not isinstance(descriptor["field"], str) or descriptor["field"] not in registered_fields:
                raise BridgeError("invalid_host")
        elif set(descriptor) == {"value", "value_type", "unit", "temporal_scope", "quantity_kind"}:
            descriptor = _typed_field(descriptor, constant=True)
        else:
            raise BridgeError("invalid_host")
        result[ref] = dict(descriptor)
    return result


def _unmet_obligations(host: dict[str, Any], constants: dict[str, dict[str, Any]],
                       categories: set[str]) -> list[dict[str, Any]]:
    obligations = host.get("unmet_obligations", [])
    if not isinstance(obligations, list):
        raise BridgeError("invalid_host")
    result = []
    seen = set()
    for raw in obligations:
        ob = _object(raw)
        if set(ob) != {"obligation_id", "category", "allowed_refs"}:
            raise BridgeError("invalid_host")
        ident = _name(ob["obligation_id"])
        if ident in seen or not isinstance(ob["category"], str) or ob["category"] not in categories:
            raise BridgeError("invalid_host")
        refs = ob["allowed_refs"]
        if (not isinstance(refs, list) or not refs or any(not isinstance(ref, str)
                or ref not in constants or set(constants[ref]) != {"entity_ref", "field"} for ref in refs)
                or len(refs) != len(set(refs))):
            raise BridgeError("invalid_host")
        seen.add(ident)
        result.append({"obligation_id": ident, "category": ob["category"], "allowed_refs": list(refs)})
    return result


def audience_state(task_input: str, host: dict[str, Any], catalog: dict[str, Any],
                   accepted_partial_program: list[dict[str, Any]] | None = None) -> str:
    """Serialize only active, audience-permitted records and host task inputs.

    No label or task interpretation is accepted.  The first planner uses an
    empty accepted prefix; rejected attempts are never promoted to a prefix.
    """
    if not isinstance(task_input, str) or not task_input:
        raise BridgeError("invalid_task")
    h, c = _object(host), _object(catalog)
    audience = h.get("audience")
    if audience not in ("public", "private"):
        raise BridgeError("invalid_host")
    provider = _object(h.get("provider", {}))
    if (provider.get("state_handle") is not None or
            provider.get("state_history_context_refs") not in (None, []) or
            provider.get("state_binding") not in (None, {})):
        raise BridgeError("provider_state_forbidden")
    if provider.get("input_context_ref", h.get("context_ref")) != h.get("context_ref"):
        raise BridgeError("stale_context")
    snapshots = _object(c.get("snapshots"))
    policies = _object(c.get("policies"))
    registries = _object(c.get("registries"))
    snapshot = _object(snapshots.get(h.get("context_ref")))
    policy = _object(policies.get(h.get("policy_ref")))
    registry = _object(registries.get(h.get("registry_ref")))
    if snapshot.get("audience") != audience:
        raise BridgeError("stale_context")
    if policy.get("status") != "current" or policy.get("audience") != audience:
        raise BridgeError("stale_policy")
    refs = snapshot.get("record_refs")
    if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs) or len(refs) != len(set(refs)):
        raise BridgeError("invalid_host")
    records = c.get("records")
    if not isinstance(records, list):
        raise BridgeError("invalid_host")
    by_ref = {r.get("record_ref"): r for r in records if isinstance(r, dict)}
    if len(by_ref) != len(records):
        raise BridgeError("invalid_host")
    forbidden = policy.get("forbidden_audiences", [])
    if not isinstance(forbidden, list):
        raise BridgeError("invalid_host")
    sources = _object(c.get("sources"))
    visible = []
    for ref in refs:
        rec = _object(by_ref.get(ref))
        if rec.get("audience") in forbidden or (audience == "public" and rec.get("audience") != "public"):
            raise BridgeError("forbidden_snapshot")
        source = _object(sources.get(rec.get("source_ref")))
        raw_fields = _object(rec.get("fields"))
        fields = {_name(name): _typed_field(raw, constant=False) for name, raw in raw_fields.items()}
        visible.append({
            "record_ref": ref,
            "version": rec.get("version"),
            "entity_ref": rec.get("entity_ref"),
            "source_ref": rec.get("source_ref"),
            "audience": rec.get("audience"),
            "epistemic_class": rec.get("epistemic_class"),
            "freshness": rec.get("freshness"),
            "conflicts": sorted(x for x in rec.get("conflicts", []) if x in refs),
            "fields": fields,
            "source": {k: source[k] for k in ("trust", "premise_admission") if k in source},
        })
    if accepted_partial_program is None:
        accepted_partial_program = []
    if accepted_partial_program != []:
        raise BridgeError("invalid_prefix")
    registered_fields = set(registry["registered_fields"])
    query_constants = _query_constants(h, registered_fields)
    unmet_obligations = _unmet_obligations(h, query_constants, set(registry["ask_categories"]))
    body = {
        "schema": SERIALIZATION_REF,
        "task_input": task_input,
        "audience": audience,
        "context_ref": h["context_ref"],
        "policy_ref": h["policy_ref"],
        "registry_ref": h["registry_ref"],
        "records": visible,
        "registry": {k: registry[k] for k in ("registered_fields", "comparators", "ask_categories") if k in registry},
        "query_constants": query_constants,
        "unmet_obligations": unmet_obligations,
        "accepted_partial_program": [],
    }
    try:
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        raise BridgeError("invalid_host") from None


def legacy_request_fingerprint(request: dict[str, Any]) -> str:
    """Pinned D v0 fingerprint over the actual legacy state/questions wire."""
    questions = {}
    for qid, q in request["questions"].items():
        questions[str(qid)] = {
            "type": q["type"],
            "instructions": q.get("instructions"),
            "options": sorted((str(k), v) for k, v in q["criteria"].items()),
        }
    body = {"state": request["state"], "questions": questions}
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decision_request(*, task_input: str, host: dict[str, Any], catalog: dict[str, Any],
                     model: str, parent_revision: str, request_id: str,
                     question_id: str, question: str, options: dict[str, str],
                     supplied_state: str | None = None,
                     state_handle: Any = None,
                     proposed_slot_choices: list[dict[str, str]] | None = None) -> tuple[dict[str, Any], str]:
    """Return a D-compatible request and exact serialization digest."""
    limits = planner_limits(host)
    if (not isinstance(model, str) or not model or not isinstance(parent_revision, str)
            or not parent_revision or not isinstance(request_id, str) or not request_id
            or not isinstance(question_id, str) or not question_id or not isinstance(question, str)
            or not question or not isinstance(options, dict) or not options
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in options.items())):
        raise BridgeError("invalid_request")
    if len(options) > limits["max_options_per_question"]:
        raise BridgeError("options_exceeded")
    if state_handle is not None or host.get("state_handle") is not None:
        raise BridgeError("provider_state_forbidden")
    state = audience_state(task_input, host, catalog)
    choices = [] if proposed_slot_choices is None else proposed_slot_choices
    if (not isinstance(choices, list) or any(not isinstance(item, dict)
            or set(item) != {"slot", "option_id", "description"}
            or any(not isinstance(value, str) or not value for value in item.values())
            for item in choices)):
        raise BridgeError("invalid_prefix")
    body = json.loads(state)
    # Model proposals, never checked facts or accepted program nodes.
    body["proposed_slot_choices"] = choices
    state = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    if supplied_state is not None and supplied_state != state:
        raise BridgeError("state_mismatch")
    hlimits = host["limits"]
    dlimits: dict[str, int] = {"max_choices": 1, "max_program_nodes": limits["max_program_nodes"]}
    for key in ("max_input_tokens", "max_inference_calls", "deadline_ms"):
        if key in hlimits:
            value = hlimits[key]
            if type(value) is not int or value < 1:
                raise BridgeError("invalid_limit")
            dlimits[key] = value
    digest = hashlib.sha256(state.encode("utf-8")).hexdigest()
    return ({
        "model": model,
        "state": state,
        "questions": {question_id: {"type": "choice", "instructions": question, "criteria": options}},
        "ssdm": {
            "schema": "ssdm.decide.v0",
            "request_id": request_id,
            "parent_revision": parent_revision,
            "context_ref": host["context_ref"],
            "policy_ref": host["policy_ref"],
            "registry_ref": host["registry_ref"],
            "audience": host["audience"],
            "limits": dlimits,
        },
    }, digest)

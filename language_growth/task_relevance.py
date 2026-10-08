"""Fail-closed task relevance for the bounded public semantic projection.

This checks that a kernel-accepted program answers one of the exact supported
read/explain/compare/two-fact conjunction forms. It does not infer unrestricted
natural language.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .parent_bridge import BridgeError, audience_state
from .types import Comparison, ConstantValue, Fact, SpeechAct

_MAX_TASK_CHARS = 2048
_MAX_VISIBLE_RECORDS = 128
_MAX_ENTITIES = 32
_MAX_FIELDS = 16


@dataclass(frozen=True, slots=True)
class TaskContract:
    task_sha256: str
    visible_state_sha256: str
    authority_sha256: str
    kind: str
    # Host-resolved identities: (record_ref, field, temporal_scope).
    # Comparison order matters; conjunction order does not.
    operands: tuple[tuple[str, str, str], ...]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _authority_sha256(host: dict[str, Any], catalog: dict[str, Any],
                      visible_state: str) -> str:
    """Bind relevance to the exact public checking authorities and task state."""
    try:
        refs = {key: host[key] for key in
                ("audience", "context_ref", "policy_ref", "registry_ref",
                 "query_constants", "unmet_obligations", "limits")}
        body = {
            "visible_state": visible_state,
            "host": refs,
            "snapshot": catalog["snapshots"][host["context_ref"]],
            "policy": catalog["policies"][host["policy_ref"]],
            "registry": catalog["registries"][host["registry_ref"]],
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False)
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_authority_state") from None
    return _sha(encoded)


def _task_bindings(task_input: str, visible: dict[str, Any]) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    if not isinstance(task_input, str) or not task_input or len(task_input) > _MAX_TASK_CHARS:
        raise ValueError("unsupported_task")
    records = visible["records"]
    registry_fields = visible["registry"]["registered_fields"]
    if (not isinstance(records, list) or not isinstance(registry_fields, list)
            or len(records) > _MAX_VISIBLE_RECORDS):
        raise ValueError("invalid_visible_state")

    # Ambiguous case-insensitive entity names and field display names fail closed.
    entities: dict[str, list[str]] = {}
    for record in records:
        name = record.get("entity_ref")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("invalid_visible_state")
        entities.setdefault(name.casefold(), []).append(name)
    if any(len(names) != 1 for names in entities.values()) or len(entities) > _MAX_ENTITIES:
        raise ValueError("ambiguous_entity")
    fields: dict[str, list[str]] = {}
    for field in registry_fields:
        if not isinstance(field, str) or not field or len(field) > 64:
            raise ValueError("invalid_visible_state")
        display = field.replace("_", " ")
        fields.setdefault(display.casefold(), []).append(field)
    if any(len(names) != 1 for names in fields.values()) or len(fields) > _MAX_FIELDS:
        raise ValueError("ambiguous_field")

    # Enumerate only fixed grammar productions. Names and fields are escaped by
    # equality-by-generation, so punctuation cannot add syntax or suffixes.
    candidates: list[tuple[str, tuple[tuple[str, str], ...], bool]] = []
    templates = {
        "read": (
            ("What is {entity}'s {field} in October 2026?", True, "spaced"),
            ("Give the October 2026 {field} for {entity}.", True, "spaced"),
            ("State {entity}'s {field} for October 2026.", True, "spaced"),
            ("What is the recorded {field} for {entity} in October 2026?", True, "canonical"),
            ("Retrieve the October 2026 {field} entry belonging to {entity}.", True, "spaced"),
            ("Identify the recorded {field} associated with {entity} during October 2026.", True, "spaced"),
        ),
        "explain": (
            ("Explain the checked {field} fact for {entity}.", False, "spaced"),
            ("Show the checked derivation for {entity}'s {field}.", False, "spaced"),
            ("Provide a checked derivation of {entity}'s {field}.", False, "spaced"),
            ("Explain the recorded {field} for {entity} in October 2026.", True, "canonical"),
            ("Explain the evidence supporting {entity}'s October 2026 {field} entry.", True, "spaced"),
            ("Give an evidence-based explanation of the October 2026 {field} recorded for {entity}.", True, "spaced"),
        ),
        "compare": (
            ("Compare the October {field} of {left} and {right}.", True, "spaced"),
            ("How do {left}'s and {right}'s October {field} compare?", True, "spaced"),
            ("For October 2026, compare {left}'s and {right}'s {field}.", True, "spaced"),
            ("Compare {left}'s {field} with {right}'s {field} in October 2026, in that order.", True, "canonical"),
            ("Taking {left} first and {right} second, compare their October 2026 {field} entries.", True, "spaced"),
            ("In the order {left} then {right}, determine the relation between their recorded October 2026 {field}.", True, "spaced"),
        ),
    }
    unique_entities = sorted({name for names in entities.values() for name in names})
    unique_fields = sorted(registry_fields)
    for kind, forms in templates.items():
        for form, has_explicit_month, field_surface in forms:
            for field in unique_fields:
                display_field = field if field_surface == "canonical" else field.replace("_", " ")
                if kind in ("read", "explain"):
                    for entity in unique_entities:
                        rendered = form.format(entity=entity, field=display_field)
                        if rendered == task_input:
                            candidates.append((kind, ((entity, field),), has_explicit_month))
                else:
                    for left in unique_entities:
                        for right in unique_entities:
                            if left == right:
                                continue
                            rendered = form.format(left=left, right=right, field=display_field)
                            if rendered == task_input:
                                candidates.append((kind, ((left, field), (right, field)), has_explicit_month))
    # One exact, canonical-field production. Resolve all possible clause
    # bindings rather than choosing a split when names contain " and ".
    prefix, suffix = "State ", " for October 2026."
    if task_input.startswith(prefix) and task_input.endswith(suffix):
        body = task_input[len(prefix):-len(suffix)]
        clauses: dict[str, list[tuple[str, str]]] = {}
        for entity in unique_entities:
            for field in unique_fields:
                clauses.setdefault(f"{entity}'s {field}", []).append((entity, field))
        for clause, first_bindings in clauses.items():
            separator = clause + " and "
            if not body.startswith(separator):
                continue
            for first in first_bindings:
                for second in clauses.get(body[len(separator):], []):
                    candidates.append(("conjoin", (first, second), True))
    if len(candidates) != 1:
        raise ValueError("unsupported_or_ambiguous_task")
    kind, requested_facts, explicit_month = candidates[0]
    operands = []
    for name, field in requested_facts:
        matching_entities = [record for record in records
                             if isinstance(record.get("entity_ref"), str)
                             and record["entity_ref"] == name]
        matches = []
        for record in matching_entities:
            value = record.get("fields", {}).get(field)
            if not isinstance(value, dict):
                continue
            scope = value.get("temporal_scope")
            if not isinstance(scope, str) or not scope:
                continue
            if explicit_month and scope != "2026-10":
                continue
            matches.append((record.get("record_ref"), field, scope))
        if len(matches) != 1:
            raise ValueError("missing_or_ambiguous_binding")
        operands.append(matches[0])
    if kind in ("compare", "conjoin") and len({operand[2] for operand in operands}) != 1:
        raise ValueError("scope_mismatch")
    if kind == "conjoin" and len(set(operands)) != 2:
        raise ValueError("repeated_fact_binding")
    return kind, tuple(operands)


def build_task_contract(task_input: str, host: dict[str, Any],
                        catalog: dict[str, Any]) -> TaskContract:
    """Parse supported intent before provider calls, from audience-filtered state."""
    if not isinstance(host, dict) or host.get("audience") != "public":
        raise ValueError("public_audience_required")
    state = audience_state(task_input, host, catalog)
    visible = json.loads(state)
    kind, operands = _task_bindings(task_input, visible)
    return TaskContract(_sha(task_input), _sha(state),
                        _authority_sha256(host, catalog, state), kind, operands)


def _fact_binding(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, Fact):
        return value.record_ref, value.field, value.interval
    return None


def _matches_speech(contract: TaskContract, outputs: tuple[SpeechAct, ...]) -> bool:
    if not isinstance(outputs, tuple) or len(outputs) != 1 or not isinstance(outputs[0], SpeechAct):
        return False
    act = outputs[0]
    if contract.kind == "read":
        return act.kind == "assert" and _fact_binding(act.content) == contract.operands[0]
    if contract.kind == "explain":
        return (act.kind == "explain" and isinstance(act.content, SpeechAct)
                and act.content.kind == "assert"
                and _fact_binding(act.content.content) == contract.operands[0])
    if contract.kind == "compare":
        if act.kind != "assert" or not isinstance(act.content, Comparison):
            return False
        if isinstance(act.content.left, ConstantValue) or isinstance(act.content.right, ConstantValue):
            return False
        return (_fact_binding(act.content.left), _fact_binding(act.content.right)) == contract.operands
    if contract.kind == "conjoin":
        if act.kind != "conjoin" or not isinstance(act.content, tuple) or len(act.content) != 2:
            return False
        if any(not isinstance(part, SpeechAct) or part.kind != "assert" for part in act.content):
            return False
        bindings = tuple(_fact_binding(part.content) for part in act.content)
        return (None not in bindings and len(set(bindings)) == 2
                and set(bindings) == set(contract.operands))
    return False


def task_contract_matches(contract: TaskContract, task_input: str,
                          host: dict[str, Any], catalog: dict[str, Any],
                          outputs: tuple[SpeechAct, ...]) -> bool:
    """Rebind current task/state and require exact speech-act/operand agreement."""
    try:
        if _sha(task_input) != contract.task_sha256:
            return False
        current_state = audience_state(task_input, host, catalog)
        if _sha(current_state) != contract.visible_state_sha256:
            return False
        if _authority_sha256(host, catalog, current_state) != contract.authority_sha256:
            return False
        visible = json.loads(current_state)
        kind, operands = _task_bindings(task_input, visible)
        if kind != contract.kind or operands != contract.operands:
            return False
        return _matches_speech(contract, outputs)
    except (BridgeError, KeyError, TypeError, ValueError, AttributeError):
        return False

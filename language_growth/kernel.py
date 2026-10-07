"""Small deterministic checker for v0.1 and opt-in v0.2 semantic programs.

The host supplies authority and evidence. A proposal supplies only references and
operations. Acceptance is evidence-relative, not a judgment of task relevance.
"""

from __future__ import annotations

import json
import hashlib
import re
import weakref
from decimal import Decimal, InvalidOperation
from typing import Any

from .types import AskContent, CheckResult, Comparison, ConstantValue, ExactSum, Fact, OutputLimitError, SpeechAct

_SCHEMAS = {"ssdm.semantic_program.v0.1", "ssdm.semantic_program.v0.2"}
_OPS = {"ReadFact", "Compare", "ExactSum", "Assert", "Conjoin", "ExplainDerivation", "Ask"}
_REASONS = {"unknown_reference", "stale_context", "stale_policy", "type_unit_mismatch", "unsupported_assertion", "forbidden_dependency", "invalid_graph", "conflict", "exceeded_limit", "scope_mismatch", "unsupported_operation", "unsupported_field"}
_MINTED: weakref.WeakSet[CheckResult] = weakref.WeakSet()
_MAX_INTEGER_DIGITS = 4300
_MAX_SCALE = 5000
_INTEGER_BOUND = 10 ** _MAX_INTEGER_DIGITS
_NODE_KEYS = {
    "ReadFact": {"id", "op", "record_ref", "field"},
    "Compare": {"id", "op", "left", "right", "relation"},
    "ExactSum": {"id", "op", "rule_ref", "operands"},
    "Assert": {"id", "op", "proposition"},
    "Conjoin": {"id", "op", "speech_acts"},
    "ExplainDerivation": {"id", "op", "speech_act"},
    "Ask": {"id", "op", "category", "obligation_ref", "references"},
}


class _Reject(Exception):
    def __init__(self, reason: str):
        assert reason in _REASONS
        self.reason = reason


class _ProviderReject(_Reject):
    pass


def _fail(reason: str) -> None:
    raise _Reject(reason)


def _dict(value: Any, reason: str = "invalid_graph") -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        _fail(reason)
    return value


def _keys(value: dict[str, Any], required: set[str], optional: set[str] = set(), reason: str = "invalid_graph") -> None:
    if value.keys() - required - optional:
        _fail("unsupported_field")
    if not required <= value.keys():
        _fail(reason)


def _name(value: Any, reason: str = "invalid_graph") -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        _fail(reason)
    return value


def _sequence(value: Any, reason: str = "invalid_graph") -> list[Any]:
    if not isinstance(value, list):
        _fail(reason)
    return value


def _catalog(catalog: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    c = _dict(catalog)
    sources = _dict(c.get("sources"))
    records = _sequence(c.get("records"))
    by_ref: dict[str, Any] = {}
    for rec in records:
        r = _dict(rec)
        ref = _name(r.get("record_ref"))
        if ref in by_ref or type(r.get("version")) is not int or r["version"] < 0:
            _fail("invalid_graph")
        _name(r.get("entity_ref")); _name(r.get("source_ref")); _name(r.get("audience")); _name(r.get("epistemic_class"))
        if r["source_ref"] not in sources:
            _fail("invalid_graph")
        _dict(r.get("fields")); _sequence(r.get("conflicts"))
        by_ref[ref] = r
    snapshots = _dict(c.get("snapshots")); policies = _dict(c.get("policies")); registries = _dict(c.get("registries"))
    return by_ref, snapshots, policies, registries


def _environment(host: Any, catalog: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    h = _dict(host)
    context_ref = _name(h.get("context_ref")); policy_ref = _name(h.get("policy_ref")); registry_ref = _name(h.get("registry_ref")); audience = _name(h.get("audience"))
    by_ref, snapshots, policies, registries = _catalog(catalog)
    snapshot = _dict(snapshots.get(context_ref), "stale_context")
    policy = _dict(policies.get(policy_ref), "stale_policy")
    registry = _dict(registries.get(registry_ref), "unknown_reference")
    if snapshot.get("audience") != audience:
        _fail("stale_context")
    if policy.get("status") != "current" or policy.get("audience") != audience:
        _fail("stale_policy")
    refs = _sequence(snapshot.get("record_refs"))
    if len(refs) != len(set(refs)) or any(ref not in by_ref for ref in refs):
        _fail("invalid_graph")
    forbidden = policy.get("forbidden_audiences", [])
    if not isinstance(forbidden, list):
        _fail("invalid_graph")
    if any(by_ref[ref]["audience"] in forbidden or (audience == "public" and by_ref[ref]["audience"] != "public") for ref in refs):
        _fail("forbidden_dependency")
    for name in ("allowed_ops", "registered_fields", "comparators", "ask_categories", "style_ids"):
        _sequence(registry.get(name))
    predicates = _dict(registry.get("field_predicates"))
    if any(not isinstance(field, str) or not isinstance(predicate, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", predicate) for field, predicate in predicates.items()):
        _fail("invalid_graph")
    _dict(h.get("query_constants")); _sequence(h.get("unmet_obligations")); _dict(h.get("limits"))
    if type(h["limits"].get("max_program_nodes")) is not int or not 1 <= h["limits"]["max_program_nodes"] <= 1024:
        _fail("invalid_graph")
    provider = _dict(h.get("provider", {}))
    if provider.get("state_handle") is not None or provider.get("state_history_context_refs") not in (None, []):
        raise _ProviderReject("forbidden_dependency")
    if provider.get("input_context_ref", context_ref) != context_ref:
        raise _ProviderReject("stale_context")
    if provider.get("state_binding") not in (None, {}):
        raise _ProviderReject("forbidden_dependency")
    return h, by_ref, snapshot, policy, registry, policies, snapshots


def _numeric_or_typed(field: dict[str, Any]) -> str | int | bool | Decimal:
    value_type = field.get("value_type")
    value = field.get("value")
    if value_type == "integer" and type(value) is int:
        if abs(value) >= _INTEGER_BOUND:
            _fail("exceeded_limit")
        return value
    if value_type == "decimal" and isinstance(value, str):
        # Bound raw components, including leading zeros, before regex/Decimal.
        if len(value) > _MAX_INTEGER_DIGITS + _MAX_SCALE + 2:
            _fail("exceeded_limit")
        unsigned = value[1:] if value[:1] in ("+", "-") else value
        integer, _, fraction = unsigned.partition(".")
        if len(integer) > _MAX_INTEGER_DIGITS or len(fraction) > _MAX_SCALE:
            _fail("exceeded_limit")
        if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value) is None:
            _fail("type_unit_mismatch")
        try:
            result = Decimal(value)
            if result.is_finite():
                return result
        except InvalidOperation:
            pass
    if value_type == "boolean" and type(value) is bool:
        return value
    if value_type == "string" and isinstance(value, str):
        return value
    _fail("type_unit_mismatch")


def _field_shape(field: Any) -> dict[str, Any]:
    f = _dict(field)
    _keys(f, {"value", "value_type", "unit", "temporal_scope"}, {"quantity_kind"})
    if f["unit"] is not None:
        _name(f["unit"])
    _name(f["temporal_scope"])
    _numeric_or_typed(f)
    return f


def _read(ref: str, field_name: str, by_ref: dict[str, Any], snapshot: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any], catalog: dict[str, Any]) -> Fact:
    if ref not in snapshot["record_refs"]:
        # Repair feedback must not reveal existence outside the offered view.
        _fail("unknown_reference")
    if ref not in by_ref:
        _fail("unknown_reference")
    rec = by_ref[ref]
    if rec["audience"] in policy.get("forbidden_audiences", []) or (policy["audience"] == "public" and rec["audience"] != "public"):
        _fail("forbidden_dependency")
    if rec.get("freshness") not in ("valid_in_declared_snapshots", "current"):
        _fail("stale_context")
    if field_name not in registry["registered_fields"] or field_name not in registry["field_predicates"]:
        _fail("unsupported_field")
    if field_name not in rec["fields"]:
        _fail("unknown_reference")
    f = _field_shape(rec["fields"][field_name])
    source = _dict(catalog["sources"].get(rec["source_ref"]))
    admitted = source.get("premise_admission") == "accepted" or source.get("trust") in {"accepted", "accepted_structured_premises", "accepted_structured_premises_except_explicit_belief_record"}
    if not admitted:
        _fail("unsupported_assertion")
    if rec["epistemic_class"] != "authoritative_structured_premise":
        _fail("unsupported_assertion")
    active = set(snapshot["record_refs"])
    if any(conflict in active for conflict in rec["conflicts"]):
        _fail("conflict")
    # Functional values with identical entity/field/unit/time cannot disagree.
    for other_ref in active:
        if other_ref == ref:
            continue
        other = by_ref[other_ref]
        if other["entity_ref"] != rec["entity_ref"] or other["epistemic_class"] != "authoritative_structured_premise" or field_name not in other["fields"]:
            continue
        of = _field_shape(other["fields"][field_name])
        if (of["unit"], of["temporal_scope"]) == (f["unit"], f["temporal_scope"]) and _numeric_or_typed(of) != _numeric_or_typed(f):
            _fail("conflict")
    return Fact(rec["entity_ref"], registry["field_predicates"][field_name], _numeric_or_typed(f), f["unit"], f["temporal_scope"], rec["source_ref"], ref, rec["version"], field_name, f.get("quantity_kind", field_name))


def _constant(ref: str, host: dict[str, Any]) -> ConstantValue:
    constants = host["query_constants"]
    if ref not in constants:
        _fail("unknown_reference")
    raw = _dict(constants[ref])
    _keys(raw, {"value", "value_type", "unit", "temporal_scope", "quantity_kind"})
    f = _field_shape(raw)
    return ConstantValue(_numeric_or_typed(f), f["value_type"], f["unit"], f["temporal_scope"], _name(f["quantity_kind"]), ref)


def _operand(value: Any, prior: dict[str, Any], host: dict[str, Any]) -> Fact | ConstantValue:
    v = _dict(value)
    if set(v) == {"constant_ref"}:
        return _constant(_name(v["constant_ref"]), host)
    _keys(v, {"node", "project"})
    if v["project"] != "object" or v["node"] not in prior or not isinstance(prior[v["node"]], Fact):
        _fail("invalid_graph")
    return prior[v["node"]]


def _compare(left: Fact | ConstantValue, right: Fact | ConstantValue, relation: str, registry: dict[str, Any]) -> Comparison:
    if relation not in registry["comparators"] or relation not in {"greater_than", "less_than", "equal"}:
        _fail("unsupported_operation")
    lv, rv = (left.object if isinstance(left, Fact) else left.value), (right.object if isinstance(right, Fact) else right.value)
    if left.interval != right.interval:
        _fail("scope_mismatch")
    if left.unit != right.unit or left.quantity_kind != right.quantity_kind:
        _fail("type_unit_mismatch")
    if type(lv) is bool or type(rv) is bool:
        if relation != "equal" or type(lv) is not type(rv):
            _fail("type_unit_mismatch")
    elif isinstance(lv, (int, Decimal)) and isinstance(rv, (int, Decimal)):
        pass
    elif type(lv) is str and type(rv) is str and relation == "equal":
        pass
    else:
        _fail("type_unit_mismatch")
    valid = {"greater_than": lv > rv, "less_than": lv < rv, "equal": lv == rv}[relation]
    if not valid:
        _fail("unsupported_assertion")
    return Comparison(left, right, relation, left.interval)


def _fact_projection(value: Any, prior: dict[str, Any]) -> Fact:
    projection = _dict(value)
    _keys(projection, {"node", "project"})
    ref = _name(projection["node"])
    if projection["project"] != "object" or ref not in prior or not isinstance(prior[ref], Fact):
        _fail("invalid_graph")
    return prior[ref]


def _exact_parts(value: int | Decimal) -> tuple[int, int]:
    if type(value) is int:
        return value, 0
    if not isinstance(value, Decimal) or not value.is_finite():
        _fail("type_unit_mismatch")
    parts = value.as_tuple()
    coefficient = 0
    for digit in parts.digits:
        coefficient = coefficient * 10 + digit
    return -coefficient if parts.sign else coefficient, parts.exponent


def _exact_sum(operands: tuple[Fact, ...], rule_ref: str, registry: dict[str, Any],
               by_ref: dict[str, Any]) -> ExactSum:
    rules = _dict(registry.get("sum_rules", {}))
    if rule_ref not in rules:
        _fail("unknown_reference")
    rule = _dict(rules[rule_ref])
    _keys(rule, {"field", "predicate", "unit", "quantity_kind", "scope_policy", "max_terms"})
    for key in ("field", "predicate", "quantity_kind"):
        _name(rule[key])
    if rule["unit"] is not None:
        _name(rule["unit"])
    if rule["scope_policy"] != "identical":
        _fail("unsupported_operation")
    if type(rule["max_terms"]) is not int or not 2 <= rule["max_terms"] <= 16:
        _fail("exceeded_limit")
    if not 2 <= len(operands) <= rule["max_terms"]:
        _fail("exceeded_limit")
    interval = operands[0].interval
    identities = set()
    for operand in operands:
        # v0.1 ReadFact may infer a kind from its field. ExactSum requires the
        # explicit host metadata and cannot upgrade that compatibility default.
        raw = by_ref[operand.record_ref]["fields"][operand.field]
        _name(raw.get("quantity_kind"), "type_unit_mismatch")
        if operand.interval != interval:
            _fail("scope_mismatch")
        if (operand.field, operand.predicate, operand.unit, operand.quantity_kind) != (
                rule["field"], rule["predicate"], rule["unit"], rule["quantity_kind"]):
            _fail("type_unit_mismatch")
        identity = (operand.subject, operand.field, operand.predicate, operand.unit,
                    operand.quantity_kind, operand.interval)
        if identity in identities:
            _fail("invalid_graph")
        identities.add(identity)
    parts = [_exact_parts(operand.object) for operand in operands]
    exponent = min(exponent for _, exponent in parts)
    # All parts came from bounded lexemes/integers; scale alignment is <=5000.
    if not -_MAX_SCALE <= exponent <= 0:
        _fail("exceeded_limit")
    coefficient = sum(coefficient * 10 ** (part_exp - exponent)
                      for coefficient, part_exp in parts)
    # Always normalize results to Decimal, including integer-only sums. A carry
    # can exceed Python's int->str limit even when both input integers fit.
    # These constructors extract digits exactly, independent of the context.
    digits = Decimal(abs(coefficient)).as_tuple().digits
    if max(1, len(digits) + exponent) > _MAX_INTEGER_DIGITS + 2:
        _fail("exceeded_limit")
    value = Decimal((1 if coefficient < 0 else 0, digits, exponent))
    return ExactSum(operands, rule_ref, value, rule["unit"], interval, rule["quantity_kind"])


def _deps(value: Any) -> tuple[str, ...]:
    if isinstance(value, Fact):
        return (value.record_ref,)
    if isinstance(value, Comparison):
        return tuple(dict.fromkeys(_deps(value.left) + _deps(value.right)))
    if isinstance(value, ExactSum):
        return _deps(value.operands)
    if isinstance(value, SpeechAct):
        return value.dependencies
    if isinstance(value, tuple):
        return tuple(dict.fromkeys(ref for part in value for ref in _deps(part)))
    if isinstance(value, AskContent):
        return value.references
    return ()


def _scopes(act: SpeechAct) -> set[str]:
    if act.kind == "assert" and isinstance(act.content, (Fact, Comparison, ExactSum)):
        return {act.content.interval}
    if act.kind == "conjoin" and isinstance(act.content, tuple):
        return set().union(*(_scopes(part) for part in act.content))
    if act.kind == "explain" and isinstance(act.content, SpeechAct):
        return _scopes(act.content)
    return set()


def _check(program: Any, host: Any, catalog: Any) -> CheckResult:
    h, by_ref, snapshot, policy, registry, _, _ = _environment(host, catalog)
    p = _dict(program)
    _keys(p, {"schema", "context_ref", "policy_ref", "registry_ref", "nodes", "outputs", "style"})
    if p["schema"] not in _SCHEMAS:
        _fail("invalid_graph")
    if p["context_ref"] != h["context_ref"]:
        _fail("stale_context")
    if p["policy_ref"] != h["policy_ref"]:
        _fail("stale_policy")
    if p["registry_ref"] != h["registry_ref"]:
        _fail("unknown_reference")
    if p["style"] not in registry["style_ids"] or p["style"] != "plain":
        _fail("unsupported_field")
    nodes = _sequence(p["nodes"])
    if not nodes:
        _fail("invalid_graph")
    if len(nodes) > h["limits"]["max_program_nodes"]:
        _fail("exceeded_limit")
    prior: dict[str, Any] = {}
    ordinals: dict[str, int] = {}
    sum_steps: dict[str, tuple[int, ...]] = {}
    for ordinal, raw in enumerate(nodes, 1):
        n = _dict(raw)
        op = _name(n.get("op"), "unsupported_operation")
        if op not in _OPS or op not in registry["allowed_ops"]:
            _fail("unsupported_operation")
        if op == "ExactSum" and p["schema"] != "ssdm.semantic_program.v0.2":
            _fail("unsupported_operation")
        _keys(n, _NODE_KEYS[op])
        node_id = _name(n["id"])
        if re.fullmatch(r"n[0-9]{1,3}", node_id) is None or node_id in prior:
            _fail("invalid_graph")
        ordinals[node_id] = ordinal
        if op == "ReadFact":
            value = _read(_name(n["record_ref"]), _name(n["field"]), by_ref, snapshot, policy, registry, catalog)
        elif op == "Compare":
            value = _compare(_operand(n["left"], prior, h), _operand(n["right"], prior, h), _name(n["relation"]), registry)
        elif op == "ExactSum":
            projections = _sequence(n["operands"])
            if not 2 <= len(projections) <= 16:
                _fail("exceeded_limit")
            operands = tuple(_fact_projection(projection, prior) for projection in projections)
            value = _exact_sum(operands, _name(n["rule_ref"]), registry, by_ref)
            sum_steps[node_id] = tuple(ordinals[projection["node"]] for projection in projections)
        elif op == "Assert":
            target = n["proposition"]
            if not isinstance(target, str) or target not in prior or not isinstance(prior[target], (Fact, Comparison, ExactSum)):
                _fail("invalid_graph")
            content = prior[target]
            value = SpeechAct("assert", content, _deps(content), sum_steps.get(target, ()) + (ordinals[target], ordinal))
        elif op == "Conjoin":
            refs = _sequence(n["speech_acts"])
            if len(refs) < 2 or len(refs) != len(set(refs)) or any(ref not in prior or not isinstance(prior[ref], SpeechAct) or prior[ref].kind not in ("assert", "conjoin") for ref in refs):
                _fail("invalid_graph")
            acts = tuple(prior[ref] for ref in refs)
            intervals = set().union(*(_scopes(act) for act in acts))
            if len(intervals) > 1:
                _fail("scope_mismatch")
            value = SpeechAct("conjoin", acts, _deps(acts), tuple(ordinals[ref] for ref in refs) + (ordinal,))
        elif op == "ExplainDerivation":
            ref = n["speech_act"]
            if not isinstance(ref, str) or ref not in prior or not isinstance(prior[ref], SpeechAct):
                _fail("invalid_graph")
            act = prior[ref]
            value = SpeechAct("explain", act, act.dependencies, act.derivation + (ordinal,))
        else:
            category = _name(n["category"]); obligation_ref = _name(n["obligation_ref"])
            references = _sequence(n["references"])
            if category not in registry["ask_categories"] or not references or len(references) != len(set(references)) or any(not isinstance(ref, str) or ref not in h["query_constants"] for ref in references):
                _fail("unknown_reference")
            for ref in references:
                descriptor = _dict(h["query_constants"][ref])
                _keys(descriptor, {"entity_ref", "field"})
                _name(descriptor["entity_ref"])
                if descriptor["field"] not in registry["registered_fields"]:
                    _fail("unsupported_field")
            matches = [o for o in h["unmet_obligations"] if isinstance(o, dict) and o.get("obligation_id") == obligation_ref and o.get("category") == category]
            if len(matches) != 1 or any(ref not in matches[0].get("allowed_refs", []) for ref in references):
                _fail("unknown_reference")
            value = SpeechAct("ask", AskContent(category, obligation_ref, tuple(references)), tuple(references), (ordinal,))
        if isinstance(value, SpeechAct):
            # Apply the same cap to constructed acts before later scope walks
            # can repeatedly traverse an oversized shared conjunction graph.
            from .realize import _render_outputs
            value.to_dict()
            _render_outputs((value,))
        prior[node_id] = value
    outputs = _sequence(p["outputs"])
    if not outputs or len(outputs) != len(set(outputs)) or any(ref not in prior or not isinstance(prior[ref], SpeechAct) for ref in outputs):
        _fail("invalid_graph")
    acts = tuple(prior[ref] for ref in outputs)
    binding = {
        "snapshot": snapshot,
        "policy": policy,
        "registry": registry,
        "records": [by_ref[ref] for ref in snapshot["record_refs"]],
        "sources": catalog["sources"],
        "query_constants": h["query_constants"],
        "unmet_obligations": h["unmet_obligations"],
    }
    binding_digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    result = CheckResult("accepted", None, acts, _deps(acts), h["context_ref"], h["policy_ref"], h["registry_ref"], h["audience"], p["style"], json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=True), binding_digest)
    # Bound both complete destination representations before minting. The
    # renderer helper does not validate/recheck or consume any raw evidence.
    from .realize import _render_outputs
    result.to_dict()
    _render_outputs(result.outputs)
    return result


def check_program(program: Any, host: Any, catalog: Any) -> CheckResult:
    """Check every program node; all rejection reasons are bounded codes."""
    try:
        result = _check(program, host, catalog)
        _MINTED.add(result)
        return result
    except _ProviderReject as exc:
        return CheckResult("provider_rejected", exc.reason)
    except _Reject as exc:
        return CheckResult("rejected", exc.reason)
    except OutputLimitError:
        return CheckResult("rejected", "exceeded_limit")
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
        return CheckResult("rejected", "invalid_graph")


def validate_release(result: CheckResult, host: Any, catalog: Any) -> CheckResult:
    """Recheck current authority and snapshot before any destination rendering.

    A copied, forged or rejected result cannot be released within this process.
    This process-local mint check is not a cryptographic certificate. The live
    recheck also detects changed record values under a reused snapshot ID.
    """
    if not isinstance(result, CheckResult) or result not in _MINTED or result.status != "accepted" or result._program_json is None:
        raise ValueError("unreleasable result")
    try:
        fresh = check_program(json.loads(result._program_json), host, catalog)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("unreleasable result") from None
    if fresh.status != "accepted" or fresh.to_dict() != result.to_dict() or fresh._binding_digest != result._binding_digest:
        raise ValueError("unreleasable result")
    return result

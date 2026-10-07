"""Bounded parent-selected L0 semantic programs.

Every operation and argument is selected through ``provider.decide``.  Static
options are enumerated without executing candidate programs or consulting
fixture labels.  Only the kernel may accept a proposal and only the renderer
may produce an answer.
"""

from __future__ import annotations

import math
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .kernel import check_program
from .parent_bridge import (
    BridgeError, DECISION_SCHEMA, FINGERPRINT_ALGORITHM, SERIALIZATION_REF,
    audience_state, decision_request,
    legacy_request_fingerprint, planner_limits,
)
from .realize import render_checked

_RESPONSE_ENVELOPE_KEYS = {
    "schema", "request_id", "status", "error", "abstention", "raw_probabilities",
    "served_probabilities", "served_config", "proposed_evidence_refs", "state_handle",
    "diagnostics", "policy_conditioned_probabilities", "attribution",
}
CANDIDATE_STYLES = ("record_ref", "entity_name")
MAX_CANDIDATE_ENTITY_CHARS = 256


class PlannerFailure(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(slots=True)
class PlannerResult:
    status: str
    text: str | None = None
    program: dict[str, Any] | None = None
    reason: str | None = None
    provider_calls: int = 0
    repair_requests: int = 0
    forward_calls: int | None = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def _probabilities(stage: Any, option_ids: set[str]) -> dict[str, float] | None:
    if not isinstance(stage, dict) or set(stage) != option_ids:
        return None
    try:
        values = [float(p) for p in stage.values() if type(p) in (int, float)]
        if (len(values) != len(option_ids) or any(not math.isfinite(p) or p < 0 for p in values)
                or abs(math.fsum(values) - 1.0) > 1e-9):
            return None
    except (OverflowError, ValueError):
        return None
    return stage


def _check_stages(env: dict[str, Any], request: dict[str, Any], qid: str,
                  options: dict[str, str], expected_config: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    ids = set(options)
    config = env.get("served_config")
    if (not isinstance(config, dict) or set(config) != {"calibration", "temperature_per_question", "policy_masking"}
            or config["calibration"] != expected_config["calibration"]
            or config["policy_masking"] != expected_config["policy_masking"]
            or not isinstance(config["temperature_per_question"], dict)
            or set(config["temperature_per_question"]) != {qid}):
        raise PlannerFailure("provider_error", "served_config_mismatch")
    temperature = config["temperature_per_question"][qid]
    try:
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise PlannerFailure("provider_error", "served_config_mismatch")
    except OverflowError:
        raise PlannerFailure("provider_error", "served_config_mismatch") from None
    calibration = config["calibration"]
    if isinstance(calibration, dict) and calibration.get("kind") == "option_count_temperature":
        try:
            t0, t1, t_min = (calibration[k] for k in ("t0", "t1", "t_min"))
            if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (t0, t1, t_min))
                    or t_min <= 0 or abs(temperature - max(t_min, t0 + t1 * math.log(len(ids)))) > 1e-9):
                raise PlannerFailure("provider_error", "served_config_mismatch")
        except (KeyError, OverflowError, ValueError):
            raise PlannerFailure("provider_error", "served_config_mismatch") from None
    served_map = env.get("served_probabilities")
    if not isinstance(served_map, dict) or set(served_map) != {qid}:
        raise PlannerFailure("provider_error", "invalid_probabilities")
    served = _probabilities(served_map[qid], ids)
    if served is None:
        raise PlannerFailure("provider_error", "invalid_probabilities")
    candidate_set_id = hashlib.sha256(json.dumps(options, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=True).encode("ascii")).hexdigest()
    stages: dict[str, Any] = {
        "served": {"candidate_set_id": candidate_set_id, "domain": sorted(ids),
                   "transformation": calibration},
    }
    if "raw_probabilities" not in env:
        raise PlannerFailure("provider_error", "invalid_probabilities")
    raw_map = env["raw_probabilities"]
    if raw_map is None:
        stages["raw"] = {"available": False}
    else:
        if not isinstance(raw_map, dict) or set(raw_map) != {qid} or _probabilities(raw_map[qid], ids) is None:
            raise PlannerFailure("provider_error", "invalid_probabilities")
        stages["raw"] = {"available": True, "candidate_set_id": candidate_set_id,
                         "domain": sorted(ids), "transformation": "pre_calibration"}
    restriction = request["ssdm"].get("policy_restriction")
    policy_map = env.get("policy_conditioned_probabilities")
    if policy_map is not None:
        if (not isinstance(restriction, dict) or qid not in restriction or not isinstance(policy_map, dict)
                or set(policy_map) != {qid} or not isinstance(policy_map[qid], dict)):
            raise PlannerFailure("provider_error", "invalid_probabilities")
        allowed = restriction[qid]
        policy = policy_map[qid]
        if (not isinstance(allowed, list) or not allowed or any(not isinstance(x, str) for x in allowed)
                or len(allowed) != len(set(allowed)) or not set(allowed) <= ids
                or set(policy) != {"probabilities", "allowed", "domain", "restriction_id",
                                   "excluded_served_mass", "status", "note"}
                or not isinstance(policy["allowed"], list)
                or any(not isinstance(x, str) for x in policy["allowed"])
                or policy["allowed"] != sorted(allowed)
                or policy["domain"] != "all_supplied_options (zero on excluded)"
                or not isinstance(policy["note"], str)):
            raise PlannerFailure("provider_error", "invalid_probabilities")
        expected_restriction_id = hashlib.sha256(json.dumps(
            {"policy_ref": request["ssdm"].get("policy_ref"), "question": qid,
             "allowed": sorted(allowed)}, ensure_ascii=False, separators=(",", ":"),
            sort_keys=True).encode("utf-8")).hexdigest()
        if policy["restriction_id"] != expected_restriction_id:
            raise PlannerFailure("provider_error", "invalid_probabilities")
        mass = math.fsum(served[x] for x in allowed)
        excluded = policy["excluded_served_mass"]
        try:
            excluded_number = float(excluded) if type(excluded) in (int, float) else float("nan")
        except (OverflowError, ValueError):
            raise PlannerFailure("provider_error", "invalid_probabilities")
        if not math.isfinite(excluded_number) or abs(excluded_number - (1 - mass)) > 1e-9:
            raise PlannerFailure("provider_error", "invalid_probabilities")
        if mass <= 0:
            if policy["status"] != "unavailable_zero_served_mass" or policy["probabilities"] is not None:
                raise PlannerFailure("provider_error", "invalid_probabilities")
            stages["policy_conditioned"] = {"available": False, "allowed": sorted(allowed),
                                            "excluded_served_mass": excluded,
                                            "restriction_id": expected_restriction_id}
        else:
            conditioned = _probabilities(policy["probabilities"], ids)
            if (policy["status"] != "ok" or conditioned is None
                    or any(abs(conditioned[x] - (served[x] / mass if x in allowed else 0.0)) > 1e-9
                           for x in ids)):
                raise PlannerFailure("provider_error", "invalid_probabilities")
            stages["policy_conditioned"] = {"available": True, "candidate_set_id": candidate_set_id,
                                            "domain": sorted(ids), "allowed": sorted(allowed),
                                            "excluded_served_mass": excluded,
                                            "restriction_id": expected_restriction_id,
                                            "transformation": "host_policy_restriction"}
    elif restriction:
        raise PlannerFailure("provider_error", "invalid_probabilities")
    else:
        stages["policy_conditioned"] = {"available": False}
    return served, stages


def _candidate_sets(host: dict[str, Any], catalog: dict[str, Any], task_input: str,
                    candidate_style: str = "record_ref") -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    """Build syntactic choices from the visible snapshot, never from truth."""
    if candidate_style not in CANDIDATE_STYLES:
        raise ValueError("unsupported candidate style")
    state = json.loads(audience_state(task_input, host, catalog))
    registry = catalog["registries"][host["registry_ref"]]
    registered = set(registry["registered_fields"])
    facts = []
    for rec in state["records"]:
        for name in sorted(rec["fields"]):
            if name in registered:
                fact = {"record_ref": rec["record_ref"], "field": name}
                if candidate_style == "entity_name":
                    # Display-only cue copied from the exact audience-visible snapshot;
                    # executable bindings remain record_ref/field.
                    entity_ref = rec["entity_ref"]
                    if not isinstance(entity_ref, str) or not entity_ref or len(entity_ref) > MAX_CANDIDATE_ENTITY_CHARS:
                        raise ValueError("candidate entity name exceeds the display limit")
                    fact["entity_ref"] = entity_ref
                facts.append(fact)
    # A value-shaped host constant is a Compare operand; Ask descriptors are not.
    operands = [dict(fact, kind="fact") for fact in facts]
    for ref, descriptor in sorted(host.get("query_constants", {}).items()):
        if isinstance(descriptor, dict) and {"value", "value_type", "unit", "temporal_scope", "quantity_kind"} <= descriptor.keys():
            operands.append({"kind": "constant", "constant_ref": ref})
    obligations = []
    categories = set(registry["ask_categories"])
    for ob in host.get("unmet_obligations", []):
        if not isinstance(ob, dict) or ob.get("category") not in categories or not isinstance(ob.get("allowed_refs"), list):
            continue
        refs = [ref for ref in ob["allowed_refs"] if ref in host.get("query_constants", {})]
        if refs:
            obligations.append({"obligation_id": ob["obligation_id"], "category": ob["category"], "refs": refs})
    return facts, operands, obligations


def _fact_description(item: dict[str, Any]) -> str:
    if "entity_ref" not in item:
        return f"Fact {item['record_ref']} field {item['field']}"
    return f"Fact {item['entity_ref']} (record {item['record_ref']}) field {item['field']}"


def _as_options(items: list[Any], describe: Callable[[Any], str], prefix: str) -> tuple[dict[str, str], dict[str, Any]]:
    options = {f"{prefix}{i:03d}": describe(item) for i, item in enumerate(items, 1)}
    binding = {f"{prefix}{i:03d}": item for i, item in enumerate(items, 1)}
    return options, binding


class _Session:
    def __init__(self, task_input: str, host: dict[str, Any], catalog: dict[str, Any],
                 provider: Any, model: str, parent_revision: str,
                 expected_served_config: dict[str, Any]):
        self.task_input, self.host, self.catalog = task_input, host, catalog
        self.provider, self.model, self.parent_revision = provider, model, parent_revision
        if (not isinstance(expected_served_config, dict)
                or set(expected_served_config) != {"calibration", "policy_masking"}
                or not isinstance(expected_served_config["calibration"], dict)
                or not isinstance(expected_served_config["calibration"].get("id"), str)
                or not expected_served_config["calibration"]["id"]
                or not isinstance(expected_served_config["policy_masking"], str)
                or not expected_served_config["policy_masking"]):
            raise PlannerFailure("precondition_failed", "invalid_served_config_pin")
        try:
            self.expected_served_config = json.loads(json.dumps(expected_served_config, allow_nan=False))
        except (TypeError, ValueError, OverflowError):
            raise PlannerFailure("precondition_failed", "invalid_served_config_pin") from None
        self.limits = planner_limits(host)
        self.trace: list[dict[str, Any]] = []
        self.provider_calls = 0
        self.repair_requests = 0
        self.forward_calls: int | None = 0
        self.run_id = uuid.uuid4().hex
        self.attempt_trace_start = 0

    def choose(self, slot: str, question: str, options: dict[str, str], *, repair: bool) -> str:
        if not options:
            raise PlannerFailure("unsupported", "no_candidates")
        if self.provider_calls >= self.limits["max_proposal_calls"]:
            raise PlannerFailure("exhausted", "max_proposal_calls")
        if repair and self.repair_requests >= self.limits["max_repair_calls"]:
            raise PlannerFailure("exhausted", "max_repair_calls")
        qid = f"slot-{self.provider_calls + 1:03d}-{slot}"
        rid = f"{self.run_id}-{self.provider_calls + 1:03d}"
        try:
            request, state_digest = decision_request(
                task_input=self.task_input, host=self.host, catalog=self.catalog,
                model=self.model, parent_revision=self.parent_revision,
                request_id=rid, question_id=qid, question=question, options=options,
                proposed_slot_choices=[
                    {"slot": step["slot"], "option_id": step["selected_id"],
                     "description": step["options"][step["selected_id"]]}
                    for step in self.trace[self.attempt_trace_start:]
                    if "slot" in step and step.get("selected_id") is not None
                ],
            )
        except BridgeError as exc:
            raise PlannerFailure("precondition_failed", exc.code) from None
        step = {
            "slot": slot, "repair": repair, "request_id": rid,
            "decision_schema": DECISION_SCHEMA,
            "serialization_ref": SERIALIZATION_REF, "state_sha256": state_digest,
            "request_fingerprint": legacy_request_fingerprint(request),
            "request_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "options": dict(options), "selected_id": None,
            "status": "invoked", "forward_calls": None,
        }
        self.trace.append(step)
        self.provider_calls += 1  # include calls that raise or return a failure
        if repair:
            self.repair_requests += 1  # each repair provider request, even if it fails
        try:
            if hasattr(self.provider, "decide"):
                response = self.provider.decide(request)
            elif callable(self.provider):
                response = self.provider(request)
            else:
                raise TypeError("provider has no decide method")
        except Exception:
            step["status"] = "provider_exception"
            self.forward_calls = None
            raise PlannerFailure("provider_error", "provider_exception") from None
        if not isinstance(response, dict) or not isinstance(response.get("ssdm"), dict):
            step["status"] = "invalid_response"
            self.forward_calls = None
            raise PlannerFailure("provider_error", "invalid_response")
        env = response["ssdm"]
        diagnostics = env.get("diagnostics")
        if isinstance(diagnostics, dict) and type(diagnostics.get("inference_calls")) is int and diagnostics["inference_calls"] >= 0:
            step["forward_calls"] = diagnostics["inference_calls"]
            if self.forward_calls is not None:
                self.forward_calls += diagnostics["inference_calls"]
        else:
            self.forward_calls = None
        if set(env) - _RESPONSE_ENVELOPE_KEYS:
            step["status"] = "invalid_response"
            raise PlannerFailure("provider_error", "invalid_response")
        if env.get("schema") != DECISION_SCHEMA:
            step["status"] = "schema_mismatch"
            raise PlannerFailure("provider_error", "schema_mismatch")
        step["status"] = env.get("status")
        if env.get("status") != "ok":
            reason = env.get("status")
            if reason not in ("unsupported", "exhausted", "error"):
                reason = "invalid_response"
            step["status"] = reason
            raise PlannerFailure("provider_rejected", reason)
        if (env.get("request_id") != rid or not isinstance(diagnostics, dict)
                or diagnostics.get("revision_id") != self.parent_revision
                or diagnostics.get("request_fingerprint") != step["request_fingerprint"]
                or diagnostics.get("request_fingerprint_algorithm") != FINGERPRINT_ALGORITHM
                or not isinstance(diagnostics.get("provenance"), dict)
                or diagnostics["provenance"].get("supplied") is not False
                or diagnostics["provenance"].get("applied") is not False
                or diagnostics.get("state_handle_in") is not None
                or diagnostics.get("state_reuse") is not False
                or env.get("state_handle") is not None):
            step["status"] = "identity_mismatch"
            raise PlannerFailure("provider_error", "identity_mismatch")
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != {qid}:
            step["status"] = "invalid_response"
            raise PlannerFailure("provider_error", "invalid_response")
        answer = answers[qid]
        if (not isinstance(answer, dict) or set(answer) != {"type", "choice", "confidence", "probabilities"}
                or answer.get("type") != "choice"):
            step["status"] = "invalid_response"
            raise PlannerFailure("provider_error", "invalid_response")
        try:
            probs, stages = _check_stages(env, request, qid, options, self.expected_served_config)
        except PlannerFailure as exc:
            step["status"] = exc.reason
            raise
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in options or selected != max(sorted(options), key=lambda oid: probs[oid]):
            step["status"] = "invalid_choice"
            raise PlannerFailure("provider_error", "invalid_choice")
        answer_probs = answer.get("probabilities")
        confidence = answer.get("confidence")
        if (not isinstance(answer_probs, dict) or set(answer_probs) != set(options)
                or type(confidence) not in (int, float)):
            step["status"] = "invalid_probabilities"
            raise PlannerFailure("provider_error", "invalid_probabilities")
        try:
            # The pinned D legacy choice wire uses WIRE_ROUND = 4; stages keep full precision.
            if (any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1
                    or abs(p - round(probs[oid], 4)) > 1e-9 for oid, p in answer_probs.items())
                    or not math.isfinite(confidence)
                    or abs(confidence - round(probs[selected], 4)) > 1e-9):
                step["status"] = "invalid_probabilities"
                raise PlannerFailure("provider_error", "invalid_probabilities")
        except OverflowError:
            step["status"] = "invalid_probabilities"
            raise PlannerFailure("provider_error", "invalid_probabilities") from None
        step["selected_id"] = selected
        step["served_config"] = env["served_config"]
        step["probability_stages"] = stages
        return selected

    def _program(self, facts: list[dict[str, str]], operands: list[dict[str, str]],
                 obligations: list[dict[str, Any]], *, feedback: str | None) -> dict[str, Any]:
        registry = self.catalog["registries"][self.host["registry_ref"]]
        self.attempt_trace_start = len(self.trace)
        allowed_ops = set(registry["allowed_ops"])
        op_items = []
        if facts and {"ReadFact", "Assert"} <= allowed_ops:
            op_items.append(("read", "Assert one selected fact"))
            if "ExplainDerivation" in allowed_ops:
                op_items.append(("explain_read", "Explain a checked assertion of one selected fact"))
        if operands and facts and {"ReadFact", "Compare", "Assert"} <= allowed_ops:
            op_items.append(("compare", "Compare two selected typed values and assert the selected relation"))
        if obligations and "Ask" in allowed_ops:
            op_items.append(("ask", "Ask for a registered missing input"))
        op_options = {f"op{i:03d}": desc for i, (_, desc) in enumerate(op_items, 1)}
        op_binding = {f"op{i:03d}": kind for i, (kind, _) in enumerate(op_items, 1)}
        q = "Select the semantic operation for the user's task."
        if feedback is not None:
            q += f" Previous proposal rejection code: {feedback}."
        kind = op_binding[self.choose("operation", q, op_options, repair=feedback is not None)]
        nodes: list[dict[str, Any]] = []
        output: str

        def choose_operand(slot: str) -> dict[str, str]:
            opts, binding = _as_options(
                operands,
                lambda x: (_fact_description(x) if x["kind"] == "fact"
                           else f"Host constant {x['constant_ref']}"), "v",
            )
            return binding[self.choose(slot, f"Select the {slot} typed operand.", opts, repair=feedback is not None)]

        if kind in ("read", "explain_read"):
            opts, binding = _as_options(facts, _fact_description, "f")
            selected = binding[self.choose("fact", "Select the fact to assert.", opts, repair=feedback is not None)]
            fact = {key: selected[key] for key in ("record_ref", "field")}
            nodes = [{"id": "n1", "op": "ReadFact", **fact},
                     {"id": "n2", "op": "Assert", "proposition": "n1"}]
            output = "n2"
            if kind == "explain_read":
                nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
                output = "n3"
        elif kind == "compare":
            left = choose_operand("left")
            right = choose_operand("right")
            relations = list(registry["comparators"])
            opts, binding = _as_options(relations, lambda x: f"Relation {x}", "r")
            relation = binding[self.choose("relation", "Select the comparison relation.", opts, repair=feedback is not None)]

            def operand(item: dict[str, str]) -> dict[str, str]:
                if item["kind"] == "constant":
                    return {"constant_ref": item["constant_ref"]}
                node_id = f"n{len(nodes) + 1}"
                nodes.append({"id": node_id, "op": "ReadFact", "record_ref": item["record_ref"], "field": item["field"]})
                return {"node": node_id, "project": "object"}

            left_ref, right_ref = operand(left), operand(right)
            compare_id = f"n{len(nodes) + 1}"
            nodes.append({"id": compare_id, "op": "Compare", "left": left_ref, "right": right_ref, "relation": relation})
            output = f"n{len(nodes) + 1}"
            nodes.append({"id": output, "op": "Assert", "proposition": compare_id})
        else:
            opts, binding = _as_options(obligations, lambda x: f"Obligation {x['obligation_id']} category {x['category']}", "o")
            ob = binding[self.choose("obligation", "Select the unmet host obligation.", opts, repair=feedback is not None)]
            opts, binding = _as_options(ob["refs"], lambda x: f"Requested reference {x}", "a")
            ref = binding[self.choose("ask_reference", "Select the requested reference.", opts, repair=feedback is not None)]
            nodes = [{"id": "n1", "op": "Ask", "category": ob["category"],
                      "obligation_ref": ob["obligation_id"], "references": [ref]}]
            output = "n1"
        return {"schema": "ssdm.semantic_program.v0.1", "context_ref": self.host["context_ref"],
                "policy_ref": self.host["policy_ref"], "registry_ref": self.host["registry_ref"],
                "nodes": nodes, "outputs": [output], "style": "plain"}


def plan_checked_answer(task_input: str, host: dict[str, Any], catalog: dict[str, Any],
                        provider: Any, *, model: str, parent_revision: str,
                        expected_served_config: dict[str, Any],
                        allow_repair: bool = False,
                        candidate_style: str = "record_ref") -> PlannerResult:
    """Run one bounded planning attempt, optionally one or more declared repairs.

    ``provider`` can be a callable or an object with ``decide(request)``. Oracle
    and replay providers are test tools; this function never claims learning.
    ``candidate_style`` changes display descriptions only; it never chooses a
    fact or changes the executable record_ref/field bindings.
    ``candidate_style`` changes display descriptions only; it never chooses a
    fact or changes the executable record_ref/field bindings.
    """
    session: _Session | None = None
    try:
        session = _Session(task_input, host, catalog, provider, model, parent_revision,
                           expected_served_config)
        facts, operands, obligations = _candidate_sets(host, catalog, task_input, candidate_style)
        max_repairs = session.limits["max_repair_attempts"] if allow_repair else 0
        feedback = None
        last_program = None
        for attempt in range(max_repairs + 1):
            program = session._program(facts, operands, obligations, feedback=feedback)
            last_program = program
            if len(program["nodes"]) > session.limits["max_program_nodes"]:
                raise PlannerFailure("exhausted", "max_program_nodes")
            checked = check_program(program, host, catalog)
            session.trace.append({"attempt": attempt + 1, "kernel_status": checked.status,
                                  "kernel_reason": checked.reason, "program": program})
            if checked.status == "accepted":
                try:
                    rendered = render_checked(checked, host, catalog)
                except (TypeError, ValueError):
                    raise PlannerFailure("release_failed", "release_failed") from None
                return PlannerResult("accepted", rendered, program, None, session.provider_calls,
                                     session.repair_requests, session.forward_calls, session.trace)
            feedback = checked.reason or checked.status
            if attempt == max_repairs or session.repair_requests >= session.limits["max_repair_calls"]:
                return PlannerResult("kernel_rejected", None, last_program, feedback,
                                     session.provider_calls, session.repair_requests,
                                     session.forward_calls, session.trace)
        raise AssertionError("unreachable")
    except BridgeError as exc:
        failure = PlannerFailure("precondition_failed", exc.code)
    except PlannerFailure as exc:
        failure = exc
    return PlannerResult(failure.status, None, None, failure.reason,
                         session.provider_calls if session else 0,
                         session.repair_requests if session else 0,
                         session.forward_calls if session else 0,
                         session.trace if session else [])

"""Bounded free-running module choices through the existing pinned D bridge.

No task labels, evaluator programs or checker-filtered candidates enter this
planner. Invalid choices consume the one-shot denominator. This is a narrow
evaluation planner, not a replacement public task parser or unrestricted agent.
"""
from __future__ import annotations

import json
from typing import Any

from .kernel import check_program
from .parent_bridge import BridgeError
from .planner import PlannerFailure, PlannerResult, _Session, _candidate_sets, _as_options, _fact_description
from .realize import render_checked


def plan_gate_task(task: dict, parent: Any) -> PlannerResult:
    session = None
    program = None
    try:
        host, catalog = task["host"], task["catalog"]
        session = _Session(task["task_input"], host, catalog, parent, **parent.planner_kwargs())
        def choose(slot, items, describe):
            options, bindings = _as_options(items, describe, "v")
            oid = session.choose(slot, f"Select {slot} for the original task.", options, repair=False)
            return bindings[oid]
        if task.get("planner_profile") == "energy":
            from .domains.energy import PROGRAM_SCHEMA, MODULE_REF, check_energy_program, public_energy_answer
            operation = choose("energy-operation", ["ReadTechnology", "LowestMedian"], str)
            record = choose("energy-record", task["energy_module"]["rows"], lambda row: f"Technology {row['technology']} record {row['record_ref']}")
            program = {"schema": PROGRAM_SCHEMA, "module_ref": MODULE_REF, "op": operation, "record_ref": record["record_ref"]}
            checked = check_energy_program(program, task["energy_module"], host, catalog)
            response = public_energy_answer(task["task_input"], program, task["energy_module"], host, catalog)
            text = response.get("text")
        else:
            facts, _, _ = _candidate_sets(host, catalog, task["task_input"], "entity_name")
            registry = catalog["registries"][host["registry_ref"]]
            allowed = set(registry["allowed_ops"])
            operations = ["ReadFact", "Compare", "ExplainDerivation"]
            operations = [op for op in operations if op in allowed]
            operations.extend(op for op in ("ExactDifference", "Conjoin", "Hypothesize", "Plan") if op in allowed)
            operation = choose("semantic-operation", operations, str)
            nodes = []
            def add(op, **kwargs):
                node = f"n{len(nodes) + 1}"
                nodes.append({"id": node, "op": op, **kwargs})
                return node
            def fact(slot):
                selected = choose(slot, facts, _fact_description)
                return add("ReadFact", record_ref=selected["record_ref"], field=selected["field"])
            def operand(slot):
                return {"node": fact(slot), "project": "object"}
            if operation == "Plan":
                workflows = host.get("workflows", {})
                workflow = choose("workflow", list(workflows), lambda ref: f"Workflow {ref}: {json.dumps(workflows[ref], sort_keys=True)}")
                remaining = list(workflows[workflow]["steps"])
                steps = []
                # Every registered step is chosen, in a model-selected order;
                # no topological filtering or checker-assisted ordering.
                while remaining:
                    step = choose("plan-step", remaining, lambda ref: f"Step {ref}: {json.dumps(workflows[workflow]['steps'][ref], sort_keys=True)}")
                    steps.append(step)
                    remaining.remove(step)
                add("Plan", workflow_ref=workflow, step_refs=steps)
            elif operation == "Hypothesize":
                assumptions = host.get("assumptions", {})
                assumption = choose("assumption", list(assumptions), lambda ref: f"Assumption {ref}: {json.dumps(assumptions[ref], sort_keys=True)}")
                right = operand("right-operand")
                relation = choose("comparison-relation", list(registry["comparators"]), str)
                compared = add("Compare", left={"assumption_ref": assumption}, right=right, relation=relation)
                add("Hypothesize", proposition=compared)
            elif operation == "Conjoin":
                first = add("Assert", proposition=fact("first-fact"))
                second = add("Assert", proposition=fact("second-fact"))
                add("Conjoin", speech_acts=[first, second])
            elif operation in ("Compare", "ExactDifference"):
                left, right = operand("left-operand"), operand("right-operand")
                kwargs = {"left": left, "right": right}
                if operation == "Compare":
                    kwargs["relation"] = choose("comparison-relation", list(registry["comparators"]), str)
                compared = add(operation, **kwargs)
                add("Assert", proposition=compared)
            else:
                asserted = add("Assert", proposition=fact("fact"))
                if operation == "ExplainDerivation":
                    add("ExplainDerivation", speech_act=asserted)
            program = {"schema": "ssdm.semantic_program.v0.1",
                       **{key: host[key] for key in ("context_ref", "policy_ref", "registry_ref")},
                       "nodes": nodes, "outputs": [nodes[-1]["id"]], "style": "plain"}
            production = "plain"
            if task["level"] == "L3":
                production = choose("licensed-production", ["plain", "l3_fact_pair_coordination_v1", "l3_fact_comparison_contrast_v1"], str)
            session.trace.append({"selected_production": production})
            checked = check_program(program, host, catalog)
            text = render_checked(checked, host, catalog, production=production) if checked.status == "accepted" else None
        session.trace.append({"program": program, "kernel_status": checked.status, "kernel_reason": checked.reason})
        status = "accepted" if checked.status == "accepted" and text is not None else "kernel_rejected"
        return PlannerResult(status, text, program, checked.reason, session.provider_calls,
                             session.repair_requests, session.forward_calls, session.trace)
    except BridgeError as exc:
        status, reason = "precondition_failed", exc.code
    except PlannerFailure as exc:
        status, reason = exc.status, exc.reason
    except (ValueError, TypeError, KeyError):
        status, reason = "runner_error", "module_boundary_failed"
    return PlannerResult(status, None, program, reason,
                         session.provider_calls if session else 0, session.repair_requests if session else 0,
                         session.forward_calls if session else 0, session.trace if session else [])

"""Free-running scoring on synthetic DEVELOPMENT tasks only.

The planner receives an initial task, host and catalog. Reference programs,
slot targets and generated teacher-forced requests are used only by scoring.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .kernel import check_program
from .planner import PlannerResult, plan_checked_answer
from .realize import render_checked
from .types import Comparison, Fact, SpeechAct

_INVERSE = {"greater_than": "less_than", "less_than": "greater_than", "equal": "equal"}


def _asserted(output: SpeechAct) -> tuple[bool, Fact | Comparison] | None:
    explained = output.kind == "explain"
    act = output.content if explained else output
    if not isinstance(act, SpeechAct) or act.kind != "assert" or not isinstance(act.content, (Fact, Comparison)):
        return None
    return explained, act.content


def _same_comparison(actual: Comparison, expected: Comparison) -> bool:
    if actual.left == expected.left and actual.right == expected.right:
        return actual.relation == expected.relation
    if actual.left == expected.right and actual.right == expected.left:
        return actual.relation == _INVERSE[expected.relation]
    return False


def score_development_task(example: dict[str, Any], result: PlannerResult) -> bool:
    """Compare checked output meaning with the generated task's intended meaning.

    This uses the reference program only inside the scorer, after planning. A
    valid assertion about another fact is still a task failure. An explanation
    of the requested fact is credited for a read task; explanation tasks require
    an explanation. This measures answer meaning, not minimal verbosity.
    """
    if result.status != "accepted" or result.program is None or result.text is None:
        return False
    training = example["training_only"]
    host, catalog = training["host"], training["catalog"]
    intended = check_program(training["program"], host, catalog)
    actual = check_program(result.program, host, catalog)
    if intended.status != "accepted" or actual.status != "accepted":
        return False
    # Do not credit a correct program paired with altered surface output.
    if result.text != render_checked(actual, host, catalog):
        return False
    if len(intended.outputs) != 1 or len(actual.outputs) != 1:
        return False
    desired, offered = _asserted(intended.outputs[0]), _asserted(actual.outputs[0])
    if desired is None or offered is None:
        return False
    expected_explained, expected_content = desired
    actual_explained, actual_content = offered
    if training["kind"] == "explain" and not actual_explained:
        return False
    if training["kind"] in ("read", "explain"):
        return isinstance(expected_content, Fact) and isinstance(actual_content, Fact) and actual_content == expected_content
    if training["kind"] == "compare":
        return (isinstance(expected_content, Comparison) and isinstance(actual_content, Comparison)
                and _same_comparison(actual_content, expected_content))
    return False


def _initial_task(example: dict[str, Any]) -> str:
    if example.get("development_only") is not True:
        raise ValueError("development-only example required")
    # Only the first generated state is read for the original task wording.
    # Later model_inputs contain gold proposed-slot prefixes and are never read.
    first = example["model_inputs"][0]
    state = json.loads(first["state"])
    if state.get("proposed_slot_choices") != [] or state.get("accepted_partial_program") != []:
        raise ValueError("initial state must have an empty program prefix")
    task = state.get("task_input")
    if not isinstance(task, str) or not task:
        raise ValueError("missing task_input")
    return task


def _slot_diagnostics(example: dict[str, Any], result: PlannerResult | None) -> dict[str, Any]:
    """Score completed traces only; never construct requests or planner state.

    Slot accuracy is literal selection accuracy, separate from task meaning
    (a reversed comparison can be task-correct with wrong orientation slots).
    Shuffled training IDs are resolved by their generated option descriptions,
    since the free-running planner supplies its own canonical candidate IDs.
    Final totals count each expected slot once, including unreached slots; all
    earlier attempts are retained but cannot supply a missing final choice.
    """
    training = example.get("training_only", {}) if isinstance(example, dict) else {}
    targets = training.get("targets") if isinstance(training, dict) else None
    if (not isinstance(targets, list)
            or any(not isinstance(target, dict) or not isinstance(target.get("slot"), str)
                   or not isinstance(target.get("option_id"), str) for target in targets)):
        return {"status": "unavailable", "total": 0, "correct": 0,
                "missing_choice": 0, "attempt_total": 0, "slots": []}

    attempts: list[dict[str, dict[str, Any]]] = [{}]
    next_attempt = 1
    for step in result.trace if result is not None else []:
        if "slot" in step:
            while len(attempts) < next_attempt:
                attempts.append({})
            attempts[next_attempt - 1][step["slot"]] = step
        elif "attempt" in step:
            next_attempt = step["attempt"] + 1
    # A retry can fail before its first provider call, leaving no slot trace.
    if (result is not None and result.trace
            and result.trace[-1].get("kernel_status") not in (None, "accepted")
            and result.status != "kernel_rejected"):
        attempts.append({})

    generated_options = {}
    if training.get("choice_format") == "augmented":
        for request in example.get("model_inputs", []):
            if not isinstance(request, dict) or not isinstance(request.get("questions"), dict):
                continue
            for qid, question in request.get("questions", {}).items():
                if (isinstance(qid, str) and isinstance(question, dict)
                        and isinstance(question.get("criteria"), dict)):
                    generated_options[qid.rsplit("-", 1)[-1]] = question["criteria"]
    slots = []
    for target in targets:
        slot, target_id = target["slot"], target["option_id"]
        target_description = generated_options.get(slot, {}).get(target_id)
        history = []
        for attempt_number, choices in enumerate(attempts, 1):
            step = choices.get(slot, {})
            selected = step.get("selected_id")
            expected_id = target_id
            if training.get("choice_format") == "augmented":
                matches = [oid for oid, description in step.get("options", {}).items()
                           if description == target_description]
                expected_id = matches[0] if len(matches) == 1 else None
            history.append({
                "attempt": attempt_number, "repair": attempt_number > 1,
                "selected_id": selected, "expected_id": expected_id,
                "correct": selected is not None and selected == expected_id,
                "status": step.get("status", "not_reached"),
                "missing_choice": selected is None, "denominator": 1,
            })
        final = history[-1]
        slots.append({
            "slot": slot, "target_id": target_id,
            "comparison_basis": ("option_description" if training.get("choice_format") == "augmented"
                                 else "option_id"),
            "selected_ids": [row["selected_id"] for row in history],
            "attempts": history, "final_attempt": final["attempt"],
            **{key: final[key] for key in
               ("selected_id", "expected_id", "correct", "status", "missing_choice", "denominator")},
        })
    return {"status": "scored", "total": len(slots),
            "correct": sum(row["correct"] for row in slots),
            "missing_choice": sum(row["missing_choice"] for row in slots),
            "attempt_total": len(slots) * len(attempts), "slots": slots}


def run_development_tasks(examples: Iterable[dict[str, Any]], provider: Any, *,
                          model: str, parent_revision: str,
                          expected_served_config: dict[str, Any],
                          allow_repair: bool = False) -> dict[str, Any]:
    """Run each task from scratch and retain every outcome in the denominator.

    A requested repair arm must authorize one complete retry of the longest
    supported L0 program (four provider requests). Reject a misconfigured arm
    before calling the provider; never silently relabel one-shot access.
    """
    examples = list(examples)
    if allow_repair:
        for example in examples:
            try:
                limits = example["training_only"]["host"]["limits"]
                values = [limits[key] for key in
                          ("max_repair_attempts", "max_repair_calls", "max_proposal_calls")]
            except (KeyError, TypeError):
                raise ValueError("one_repair requires explicit host budgets") from None
            if (any(type(value) is not int for value in values)
                    or values[0] != 1 or values[1] < 4 or values[2] < 8):
                raise ValueError("one_repair requires attempts=1, repair calls>=4, proposal calls>=8")
    observations = []
    results = []
    for example in examples:
        task_id = example.get("task_id") if isinstance(example, dict) else None
        effective_budget = None
        result = None
        try:
            task_input = _initial_task(example)
            training = example["training_only"]
            limits = training["host"]["limits"]
            effective_budget = {
                "max_proposal_calls": limits["max_proposal_calls"],
                "max_repair_attempts": limits["max_repair_attempts"] if allow_repair else 0,
                "max_repair_calls": limits["max_repair_calls"] if allow_repair else 0,
            }
            result = plan_checked_answer(
                task_input, training["host"], training["catalog"], provider,
                model=model, parent_revision=parent_revision,
                expected_served_config=expected_served_config, allow_repair=allow_repair,
            )
            correct = score_development_task(example, result)
            observations.append({
                "task_id": task_id, "status": result.status, "reason": result.reason,
                "kernel_accepted": result.status == "accepted", "task_correct": correct,
                "text": result.text, "program": result.program,
                "provider_calls": result.provider_calls, "repair_requests": result.repair_requests,
                "forward_calls": result.forward_calls,
                "effective_budget": effective_budget,
            })
        except Exception:
            # An exception still occupies a development request. Work before an
            # unexpected exception is unknown rather than falsely counted as zero.
            observations.append({
                "task_id": task_id, "status": "runner_error", "reason": "runner_error",
                "kernel_accepted": False, "task_correct": False,
                "text": None, "program": None, "provider_calls": None,
                "repair_requests": None, "forward_calls": None,
                "effective_budget": effective_budget,
            })
        finally:
            results.append(result)
    # All provider calls in this batch have finished before targets are read.
    slot_totals: dict[str, dict[str, int]] = {}
    unavailable = 0
    for example, result, observation in zip(examples, results, observations):
        diagnostics = _slot_diagnostics(example, result)
        observation["slot_diagnostics"] = diagnostics
        unavailable += diagnostics["status"] == "unavailable"
        for row in diagnostics["slots"]:
            totals = slot_totals.setdefault(row["slot"], {"total": 0, "correct": 0, "missing_choice": 0})
            totals["total"] += row["denominator"]
            totals["correct"] += row["correct"]
            totals["missing_choice"] += row["missing_choice"]
    return {
        "condition": "one_repair" if allow_repair else "one_shot",
        "source": "synthetic_development_nonlearned_unless_provider_proven_otherwise",
        "total": len(observations),
        "kernel_accepted": sum(row["kernel_accepted"] for row in observations),
        "task_correct": sum(row["task_correct"] for row in observations),
        "slot_diagnostics": {
            "total": sum(row["total"] for row in slot_totals.values()),
            "correct": sum(row["correct"] for row in slot_totals.values()),
            "missing_choice": sum(row["missing_choice"] for row in slot_totals.values()),
            "unavailable_tasks": unavailable, "by_slot": slot_totals,
        },
        "observations": observations,
    }

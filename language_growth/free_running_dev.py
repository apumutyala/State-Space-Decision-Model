"""Free-running scoring on synthetic DEVELOPMENT tasks only.

The planner receives an initial task, host and catalog. Reference programs,
slot targets and generated teacher-forced requests are used only by scoring.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .kernel import check_program
from .planner import CANDIDATE_STYLES, PlannerResult, plan_checked_answer
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


def run_development_tasks(examples: Iterable[dict[str, Any]], provider: Any, *,
                          model: str, parent_revision: str,
                          expected_served_config: dict[str, Any],
                          allow_repair: bool = False,
                          candidate_style: str = "record_ref") -> dict[str, Any]:
    """Run each task from scratch and retain every outcome in the denominator.

    A requested repair arm must authorize one complete retry of the longest
    supported L0 program (four provider requests). Reject a misconfigured arm
    before calling the provider; never silently relabel one-shot access.
    """
    examples = list(examples)
    if candidate_style not in CANDIDATE_STYLES:
        raise ValueError("unsupported candidate style")
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
    for example in examples:
        task_id = example.get("task_id") if isinstance(example, dict) else None
        effective_budget = None
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
                candidate_style=candidate_style,
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
    return {
        "condition": "one_repair" if allow_repair else "one_shot",
        "candidate_style": candidate_style,
        "source": "synthetic_development_nonlearned_unless_provider_proven_otherwise",
        "total": len(observations),
        "kernel_accepted": sum(row["kernel_accepted"] for row in observations),
        "task_correct": sum(row["task_correct"] for row in observations),
        "observations": observations,
    }

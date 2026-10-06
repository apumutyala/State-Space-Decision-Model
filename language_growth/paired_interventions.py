"""External input-counterfactual response fidelity on synthetic comparisons.

Every intervention gets two fresh, one-shot free-running planner attempts.
References are consulted for scoring only after both attempts. This measures
external response fidelity, not internal neural causal mediation, learned L0
qualification, or superiority to random/output-only latent controls.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable

from .development_data import comparison_nuisance_counterfactual, comparison_swap_counterfactual
from .free_running_dev import _asserted, _initial_task, score_development_task
from .kernel import check_program
from .parent_bridge import audience_state
from .planner import PlannerResult, plan_checked_answer
from .realize import render_checked
from .types import Comparison

_INVERSE = {"greater_than": "less_than", "less_than": "greater_than", "equal": "equal"}
_INTERVENTIONS = {"relevant_swap": comparison_swap_counterfactual,
                  "nuisance_change": comparison_nuisance_counterfactual}
_FORBIDDEN_KEYS = {"training_only", "targets", "target_option_id", "task_id",
                   "program", "reference_program", "label", "labels"}
DEPENDENCY = {"pull_request": 51, "unmerged": True,
              "dependency_commit": "03db551", "original_source_commit": "4682e49"}


def _no_labels(value: Any) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_KEYS.intersection(value):
            raise ValueError("provider_request_label_or_program_leakage")
        for child in value.values():
            _no_labels(child)
    elif isinstance(value, list):
        for child in value:
            _no_labels(child)


def _initial_state(example: dict[str, Any]) -> dict[str, Any]:
    task = _initial_task(example)
    training = example["training_only"]
    state = json.loads(audience_state(task, training["host"], training["catalog"]))
    state["proposed_slot_choices"] = []
    # Reject poisoned initial requests; never forward generated gold prefixes.
    if state != json.loads(example["model_inputs"][0]["state"]):
        raise ValueError("initial_state_mismatch")
    _no_labels(state)
    return state


class _Capture:
    def __init__(self, provider: Any, initial_state: dict[str, Any]):
        self.provider, self.initial_state = provider, deepcopy(initial_state)
        self.calls: list[dict[str, Any]] = []

    def decide(self, request: dict[str, Any]) -> Any:
        record = {"request": deepcopy(request), "response": None,
                  "error": None, "provider_called": False}
        self.calls.append(record)
        try:
            _no_labels(request)
            state = json.loads(request["state"])
            _no_labels(state)
            choices = state.pop("proposed_slot_choices")
            initial = deepcopy(self.initial_state)
            initial.pop("proposed_slot_choices")
            if (state != initial or not isinstance(choices, list)
                    or (len(self.calls) == 1 and choices != [])):
                raise ValueError("provider_request_state_or_prefix_mismatch")
            if any(not isinstance(choice, dict)
                   or set(choice) != {"slot", "option_id", "description"} for choice in choices):
                raise ValueError("provider_request_prefix_mismatch")
            record["provider_called"] = True
            response = (self.provider.decide(request) if hasattr(self.provider, "decide")
                        else self.provider(request))
            record["response"] = deepcopy(response)
            return response
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise


def _attempt(example: dict[str, Any] | None, provider: Any, *, model: str,
             parent_revision: str, expected_served_config: dict[str, Any],
             preparation_error: dict[str, str] | None = None) -> tuple[PlannerResult, dict[str, Any]]:
    capture = None
    error = preparation_error
    state = None
    result = PlannerResult("runner_error", reason="runner_error", forward_calls=None)
    try:
        if example is None:
            raise ValueError("counterfactual_preparation_failed")
        state = _initial_state(example)
        capture = _Capture(provider, state)
        training = example["training_only"]
        result = plan_checked_answer(
            state["task_input"], training["host"], training["catalog"], capture,
            model=model, parent_revision=parent_revision,
            expected_served_config=expected_served_config, allow_repair=False)
    except Exception as exc:
        if error is None:
            error = {"type": type(exc).__name__, "message": str(exc)}
    calls = capture.calls if capture else []
    row = {"status": result.status, "reason": result.reason, "error": error,
           "text": result.text, "program": deepcopy(result.program),
           "provider_calls": sum(call["provider_called"] for call in calls),
           "request_attempts": len(calls), "forward_calls": result.forward_calls,
           "repair_requests": result.repair_requests, "calls": deepcopy(calls),
           "initial_state": state,
           "initial_state_sha256": hashlib.sha256(json.dumps(
               state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
           ).encode("ascii")).hexdigest() if state is not None else None,
           "kernel_accepted": False, "task_correct": False,
           "checked_outputs": None, "surface_verified": False,
           "reference_relation": None, "relation_in_reference_order": None}
    return result, row


def _score(example: dict[str, Any] | None, result: PlannerResult, row: dict[str, Any]) -> None:
    if example is None:
        return
    try:
        training = example["training_only"]
        host, catalog = training["host"], training["catalog"]
        row["task_correct"] = score_development_task(example, result)
        reference = check_program(training["program"], host, catalog)
        desired = _asserted(reference.outputs[0]) if len(reference.outputs) == 1 else None
        if reference.status != "accepted" or desired is None or not isinstance(desired[1], Comparison):
            raise ValueError("invalid_comparison_reference")
        expected = desired[1]
        row["reference_relation"] = expected.relation
        if result.status != "accepted" or result.program is None:
            return
        checked = check_program(result.program, host, catalog)
        row["kernel_accepted"] = checked.status == "accepted"
        if checked.status != "accepted":
            return
        row["checked_outputs"] = [output.to_dict() for output in checked.outputs]
        row["surface_verified"] = result.text == render_checked(checked, host, catalog)
        offered = _asserted(checked.outputs[0]) if len(checked.outputs) == 1 else None
        if offered is not None and isinstance(offered[1], Comparison):
            actual = offered[1]
            if actual.left == expected.left and actual.right == expected.right:
                row["relation_in_reference_order"] = actual.relation
            elif actual.left == expected.right and actual.right == expected.left:
                row["relation_in_reference_order"] = _INVERSE[actual.relation]
    except Exception as exc:
        row["task_correct"] = False
        row["scoring_error"] = {"type": type(exc).__name__, "message": str(exc)}


def _input_check(example: dict[str, Any], condition: str,
                 base: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    """Check exact evidence edits, allowing only the rebound context identity."""
    result = {"verified": False, "state_changed": None, "task_text_unchanged": None,
              "expected_state_change": None, "field": None}
    try:
        first, second = base["initial_state"], counterfactual["initial_state"]
        if first is None or second is None:
            return result
        field = example["training_only"]["program"]["nodes"][0]["field"]
        changed_field = field if condition == "relevant_swap" else (
            "warranty_months" if field == "price" else "price")
        result.update(field=changed_field, state_changed=first != second,
                      task_text_unchanged=first["task_input"] == second["task_input"])
        left, right = deepcopy(first), deepcopy(second)
        original = [record["fields"][changed_field] for record in left["records"]]
        edited = [record["fields"][changed_field] for record in right["records"]]
        if condition == "relevant_swap":
            edit_ok = edited == list(reversed(original))
            expected_change = original[0] != original[1]
        else:
            edit_ok = all(a != b and {k: v for k, v in a.items() if k != "value"}
                          == {k: v for k, v in b.items() if k != "value"}
                          for a, b in zip(original, edited))
            expected_change = True
        result["expected_state_change"] = expected_change
        for record in right["records"]:
            ref = record["record_ref"]
            source = next(rec for rec in left["records"] if rec["record_ref"] == ref)
            record["fields"][changed_field] = source["fields"][changed_field]
        left.pop("context_ref")
        right.pop("context_ref")
        result["verified"] = (edit_ok and left == right and
                              result["state_changed"] == expected_change)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def run_paired_interventions(examples: Iterable[dict[str, Any]], provider: Any, *,
                             model: str, parent_revision: str,
                             expected_served_config: dict[str, Any]) -> dict[str, Any]:
    """Run both interventions per comparison; no filtering, retries or repairs.

    Each condition has its own base attempt (four planner attempts per task).
    Providers must honor the pinned configuration and stateless request contract.
    Equal-relation relevant swaps remain in the relevant denominator as controls.
    """
    examples = list(examples)
    if not examples or any(not isinstance(example, dict)
                           or example.get("development_only") is not True
                           or example.get("training_only", {}).get("kind") != "compare"
                           for example in examples):
        raise ValueError("nonempty comparison-only development cohort required")
    pairs = []
    for example in examples:
        for condition, transform in _INTERVENTIONS.items():
            counterfactual, preparation_error = None, None
            try:
                counterfactual = transform(example)
            except Exception as exc:
                preparation_error = {"type": type(exc).__name__, "message": str(exc)}
            base_result, base = _attempt(example, provider, model=model,
                                        parent_revision=parent_revision,
                                        expected_served_config=expected_served_config)
            cf_result, changed = _attempt(counterfactual, provider, model=model,
                                         parent_revision=parent_revision,
                                         expected_served_config=expected_served_config,
                                         preparation_error=preparation_error)
            # Both attempts are complete before any task-label/reference scoring.
            _score(example, base_result, base)
            _score(counterfactual, cf_result, changed)
            input_check = _input_check(example, condition, base, changed)
            checked_available = base["kernel_accepted"] and changed["kernel_accepted"]
            surface_available = base["surface_verified"] and changed["surface_verified"]
            checked_invariant = (base["checked_outputs"] == changed["checked_outputs"]
                                 if checked_available else None)
            surface_invariant = base["text"] == changed["text"] if surface_available else None
            expected = (base["reference_relation"], changed["reference_relation"])
            observed = (base["relation_in_reference_order"], changed["relation_in_reference_order"])
            expected_change = (expected[0] != expected[1] if None not in expected else None)
            observed_change = (observed[0] != observed[1] if None not in observed else None)
            relation_behavior = (None not in expected and None not in observed and
                                 observed == expected and
                                 expected[1] == (_INVERSE[expected[0]] if condition == "relevant_swap"
                                                 else expected[0]))
            both_correct = base["task_correct"] and changed["task_correct"]
            passed = (input_check["verified"] and both_correct and relation_behavior and
                      (checked_invariant is True and surface_invariant is True
                       if condition == "nuisance_change" else True))
            pairs.append({"task_id": example.get("task_id"), "intervention": condition,
                          "base": base, "counterfactual": changed, "input_check": input_check,
                          "equal_relation_control": expected[0] == "equal",
                          "expected_relation_change": expected_change,
                          "observed_relation_change": observed_change,
                          "expected_relation_behavior": relation_behavior,
                          "checked_output_invariant": checked_invariant,
                          "surface_invariant": surface_invariant,
                          "surface_matches_expected_change": (surface_invariant == (not expected_change)
                                                               if surface_available and expected_change is not None
                                                               else None),
                          "both_task_correct": both_correct, "pair_pass": passed})
    summaries = {}
    for condition in _INTERVENTIONS:
        cohort = [pair for pair in pairs if pair["intervention"] == condition]
        summaries[condition] = {"total": len(cohort),
                                "pair_pass": sum(pair["pair_pass"] for pair in cohort),
                                "both_task_correct": sum(pair["both_task_correct"] for pair in cohort),
                                "equal_relation_controls": sum(pair["equal_relation_control"] for pair in cohort),
                                "checked_output_invariant": sum(pair["checked_output_invariant"] is True for pair in cohort),
                                "surface_invariant": sum(pair["surface_invariant"] is True for pair in cohort),
                                "expected_relation_behavior": sum(pair["expected_relation_behavior"] for pair in cohort)}
    return {"scope": "external_input_counterfactual_response_fidelity_only",
            "source": "synthetic_development_nonlearned_unless_provider_proven_otherwise",
            "claims_excluded": ["internal_neural_causal_mediation", "learned_L0_qualification",
                                "superiority_to_random_or_output_only_latent_controls"],
            "dependency": deepcopy(DEPENDENCY), "condition": "one_shot_no_repair",
            "model": model, "parent_revision": parent_revision,
            "expected_served_config": deepcopy(expected_served_config),
            "task_total": len(examples), "total": len(pairs),
            "pair_pass": sum(pair["pair_pass"] for pair in pairs),
            "interventions": summaries, "pairs": pairs}

"""Paired direct-choice and free-running planner views of synthetic comparison tasks.

Both arms start with the same permitted evidence and use one pinned parent. The
direct arm always offers all three relations; reference meaning stays in the
development scorer and is never used to construct a provider request.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Iterable
import uuid

from .free_running_dev import _initial_task, run_development_tasks
from .parent_bridge import BridgeError, decision_request
from .planner import PlannerFailure, _Session


RELATIONS = ("greater_than", "less_than", "equal")
DEFAULT_OPTION_IDS = {"greater_than": "r001", "less_than": "r002", "equal": "r003"}
DIRECT_SLOT = "direct-comparison-relation"
DIRECT_QID = f"slot-001-{DIRECT_SLOT}"
DIRECT_QUESTION = "Select the relation between the requested quantities."


def direct_comparison_request(example: dict[str, Any], *, model: str, parent_revision: str,
                              option_ids: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the same-evidence direct request without reading the target relation."""
    if example.get("development_only") is not True or example.get("training_only", {}).get("kind") != "compare":
        raise ValueError("comparison development example required")
    ids = DEFAULT_OPTION_IDS if option_ids is None else option_ids
    if (not isinstance(ids, dict) or set(ids) != set(RELATIONS)
            or any(not isinstance(value, str) or not value for value in ids.values())
            or len(set(ids.values())) != len(RELATIONS)):
        raise ValueError("option_ids must be a one-to-one map for all three relations")
    task_input = _initial_task(example)
    training = example["training_only"]
    options = {ids[relation]: f"Relation {relation}" for relation in RELATIONS}
    request, _ = decision_request(
        task_input=task_input, host=training["host"], catalog=training["catalog"],
        model=model, parent_revision=parent_revision,
        request_id=f"direct-{uuid.uuid4().hex}", question_id=DIRECT_QID,
        question=DIRECT_QUESTION, options=options,
    )
    if request["state"] != example["model_inputs"][0]["state"]:
        raise ValueError("direct and planner initial evidence differ")
    return request


def _target_relation(example: dict[str, Any]) -> str:
    """Evaluator-only label lookup, called after the provider has answered."""
    nodes = example["training_only"]["program"]["nodes"]
    comparisons = [node for node in nodes if node.get("op") == "Compare"]
    if len(comparisons) != 1 or comparisons[0].get("relation") not in RELATIONS:
        raise ValueError("comparison reference program has no single registered relation")
    return comparisons[0]["relation"]


class _Capture:
    def __init__(self, provider: Any):
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def decide(self, request: dict[str, Any]) -> Any:
        record = {"request": deepcopy(request), "response": None, "exception": False}
        self.calls.append(record)
        try:
            response = (self.provider.decide(request) if hasattr(self.provider, "decide")
                        else self.provider(request))
        except Exception:
            record["exception"] = True
            raise
        record["response"] = deepcopy(response)
        return response


def run_paired_comparisons(examples: Iterable[dict[str, Any]], provider: Any, *,
                           model: str, parent_revision: str,
                           expected_served_config: dict[str, Any],
                           option_ids: dict[str, str] | None = None) -> dict[str, Any]:
    """One-shot development comparison, with every task in both denominators."""
    examples = list(examples)
    if not examples or any(example.get("development_only") is not True
                           or example.get("training_only", {}).get("kind") != "compare" for example in examples):
        raise ValueError("pass a nonempty comparison-only development cohort")
    capture = _Capture(provider)
    pairs = []
    for example in examples:
        task_id = example.get("task_id")
        direct_request = None
        direct = {"status": "runner_error", "reason": "runner_error", "selected_id": None,
                  "target_id": None, "correct": False, "provider_calls": 0,
                  "forward_calls": 0, "response": None}
        direct_call_start = len(capture.calls)
        session = None
        try:
            training = example["training_only"]
            session = _Session(_initial_task(example), training["host"], training["catalog"],
                               capture, model, parent_revision, expected_served_config)
            preview = direct_comparison_request(example, model=model,
                                                parent_revision=parent_revision, option_ids=option_ids)
            options = preview["questions"][DIRECT_QID]["criteria"]
            selected = session.choose(DIRECT_SLOT, DIRECT_QUESTION, options, repair=False)
            sent = capture.calls[direct_call_start]["request"]
            if sent["state"] != preview["state"] or sent["questions"] != preview["questions"]:
                raise ValueError("direct request drifted from standalone builder")
            target = next(oid for oid, description in options.items()
                          if description == f"Relation {_target_relation(example)}")
            direct.update(status="ok", reason=None, selected_id=selected,
                          target_id=target, correct=selected == target)
        except PlannerFailure as exc:
            direct.update(status=exc.status, reason=exc.reason)
        except BridgeError as exc:
            direct.update(status="precondition_failed", reason=exc.code)
        except Exception:
            pass  # malformed development input still occupies the paired denominator
        finally:
            if session is not None:
                direct["provider_calls"] = session.provider_calls
                direct["forward_calls"] = session.forward_calls
            if len(capture.calls) > direct_call_start:
                sent = capture.calls[direct_call_start]
                direct_request = deepcopy(sent["request"])
                direct["response"] = deepcopy(sent["response"])
        first_planner_call = len(capture.calls)
        planner_report = run_development_tasks([example], capture, model=model,
                                               parent_revision=parent_revision,
                                               expected_served_config=expected_served_config)
        planner = planner_report["observations"][0]
        planner_calls = capture.calls[first_planner_call:]
        direct_hash = (hashlib.sha256(direct_request["state"].encode()).hexdigest()
                       if direct_request is not None else None)
        planner_hash = (hashlib.sha256(planner_calls[0]["request"]["state"].encode()).hexdigest()
                        if planner_calls else None)
        same_state = direct_hash == planner_hash if direct_hash is not None and planner_hash is not None else None
        pairs.append({"task_id": task_id, "direct": direct, "planner": planner,
                      "direct_request": direct_request,
                      "planner_calls": deepcopy(planner_calls),
                      "direct_initial_state_sha256": direct_hash,
                      "planner_initial_state_sha256": planner_hash,
                      "same_initial_state": same_state,
                      "paired_correct": same_state is True and direct["correct"] and planner["task_correct"]})
    return {"scope": "synthetic_development_comparison_only_one_shot",
            "source": "nonlearned_unless_provider_proven_otherwise",
            "model": model, "parent_revision": parent_revision,
            "expected_served_config": deepcopy(expected_served_config),
            "total": len(pairs), "direct_correct": sum(pair["direct"]["correct"] for pair in pairs),
            "planner_correct": sum(pair["planner"]["task_correct"] for pair in pairs),
            "paired_correct": sum(pair["paired_correct"] for pair in pairs), "pairs": pairs}

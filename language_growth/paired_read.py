"""Same-evidence plain-read development probe, never a learned-level gate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Iterable
import uuid

from .free_running_dev import _initial_task, run_development_tasks
from .paired_comparison import _Capture
from .parent_bridge import BridgeError, decision_request
from .planner import PlannerFailure, _Session, _as_options, _candidate_sets

DIRECT_SLOT = "direct-read-fact"
DIRECT_QID = f"slot-001-{DIRECT_SLOT}"
DIRECT_QUESTION = "Select the fact requested by the user's task."


def _read_view(example: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    if example.get("development_only") is not True or example.get("training_only", {}).get("kind") != "read":
        raise ValueError("plain-read development example required")
    task = _initial_task(example)
    training = example["training_only"]
    facts, _, _ = _candidate_sets(training["host"], training["catalog"], task)
    options, binding = _as_options(facts, lambda fact: f"Fact {fact['record_ref']} field {fact['field']}", "f")
    return task, options, binding


def direct_read_request(example: dict[str, Any], *, model: str,
                        parent_revision: str) -> dict[str, Any]:
    """Offer all permitted registered Facts without consulting reference targets."""
    task, options, _ = _read_view(example)
    training = example["training_only"]
    request, _ = decision_request(
        task_input=task, host=training["host"], catalog=training["catalog"],
        model=model, parent_revision=parent_revision,
        request_id=f"direct-{uuid.uuid4().hex}", question_id=DIRECT_QID,
        question=DIRECT_QUESTION, options=options,
    )
    if request["state"] != example["model_inputs"][0]["state"]:
        raise ValueError("direct and planner initial evidence differ")
    return request


def _target_fact(example: dict[str, Any]) -> dict[str, str]:
    """Evaluator-only lookup, called after the direct provider response."""
    program = example["training_only"]["program"]
    nodes = program["nodes"]
    if (len(nodes) != 2 or nodes[0].get("op") != "ReadFact"
            or nodes[1].get("op") != "Assert"
            or nodes[1].get("proposition") != nodes[0].get("id")
            or program["outputs"] != [nodes[1].get("id")]):
        raise ValueError("plain-read reference must contain one ReadFact and Assert")
    return {key: nodes[0][key] for key in ("record_ref", "field")}


def run_paired_reads(examples: Iterable[dict[str, Any]], provider: Any, *,
                     model: str, parent_revision: str,
                     expected_served_config: dict[str, Any]) -> dict[str, Any]:
    """Run both arms from scratch and preserve every read in both denominators.

    This constrained direct Fact probe supplements the strong unrestricted
    baseline and full parent-retention evaluation required for qualification.
    """
    examples = list(examples)
    if not examples or any(example.get("development_only") is not True
                           or example.get("training_only", {}).get("kind") != "read" for example in examples):
        raise ValueError("pass a nonempty plain-read-only development cohort")
    capture = _Capture(provider)
    pairs = []
    for example in examples:
        direct = {"status": "runner_error", "reason": "runner_error", "selected_id": None,
                  "target_id": None, "correct": False, "provider_calls": 0,
                  "forward_calls": 0, "response": None, "trace": []}
        direct_start = len(capture.calls)
        session = None
        try:
            training = example["training_only"]
            session = _Session(_initial_task(example), training["host"], training["catalog"],
                               capture, model, parent_revision, expected_served_config)
            preview = direct_read_request(example, model=model, parent_revision=parent_revision)
            _, options, binding = _read_view(example)
            selected = session.choose(DIRECT_SLOT, DIRECT_QUESTION, options, repair=False)
            direct["selected_id"] = selected
            sent = capture.calls[direct_start]["request"]
            if sent["state"] != preview["state"] or sent["questions"] != preview["questions"]:
                raise ValueError("direct request drifted from standalone builder")
            desired = _target_fact(example)
            target = next(oid for oid, fact in binding.items() if fact == desired)
            direct.update(status="ok", reason=None, target_id=target, correct=selected == target)
        except PlannerFailure as exc:
            direct.update(status=exc.status, reason=exc.reason)
        except BridgeError as exc:
            direct.update(status="precondition_failed", reason=exc.code)
        except Exception:
            pass  # malformed input/scoring still occupies the paired denominator
        finally:
            if session is not None:
                direct.update(provider_calls=session.provider_calls, forward_calls=session.forward_calls,
                              trace=deepcopy(session.trace))
        direct_calls = deepcopy(capture.calls[direct_start:])
        direct_request = direct_calls[0]["request"] if direct_calls else None
        if direct_calls:
            direct["response"] = direct_calls[0]["response"]
        planner_start = len(capture.calls)
        planner = run_development_tasks([example], capture, model=model, parent_revision=parent_revision,
                                        expected_served_config=expected_served_config)["observations"][0]
        planner_calls = deepcopy(capture.calls[planner_start:])
        direct_hash = hashlib.sha256(direct_request["state"].encode()).hexdigest() if direct_request else None
        planner_hash = (hashlib.sha256(planner_calls[0]["request"]["state"].encode()).hexdigest()
                        if planner_calls else None)
        same_state = direct_hash == planner_hash if direct_hash is not None and planner_hash is not None else None
        pairs.append({"task_id": example.get("task_id"), "direct": direct, "planner": planner,
                      "direct_request": direct_request, "direct_calls": direct_calls,
                      "planner_calls": planner_calls, "direct_initial_state_sha256": direct_hash,
                      "planner_initial_state_sha256": planner_hash, "same_initial_state": same_state,
                      "paired_correct": same_state is True and direct["correct"] and planner["task_correct"]})
    return {"scope": "synthetic_development_plain_read_only_one_shot",
            "source": "nonlearned_unless_provider_proven_otherwise",
            "qualification_requires": ["strong_unrestricted_comparator", "full_parent_retention"],
            "model": model, "parent_revision": parent_revision,
            "expected_served_config": deepcopy(expected_served_config),
            "total": len(pairs), "direct_correct": sum(pair["direct"]["correct"] for pair in pairs),
            "planner_correct": sum(pair["planner"]["task_correct"] for pair in pairs),
            "paired_correct": sum(pair["paired_correct"] for pair in pairs), "pairs": pairs}

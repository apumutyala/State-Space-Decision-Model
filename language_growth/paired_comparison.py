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

from decide.schema import wire_answer

from .free_running_dev import _initial_task, run_development_tasks
from .parent_bridge import (DECISION_SCHEMA, FINGERPRINT_ALGORITHM, decision_request,
                            legacy_request_fingerprint)
from .planner import PlannerFailure, _check_stages


RELATIONS = ("greater_than", "less_than", "equal")
DEFAULT_OPTION_IDS = {"greater_than": "r001", "less_than": "r002", "equal": "r003"}
DIRECT_QID = "direct-comparison-relation"


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
        question="Select the relation between the requested quantities.", options=options,
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


def _forward_calls(response: Any) -> int | None:
    if not isinstance(response, dict) or not isinstance(response.get("ssdm"), dict):
        return None
    diagnostic = response["ssdm"].get("diagnostics")
    calls = diagnostic.get("inference_calls") if isinstance(diagnostic, dict) else None
    return calls if type(calls) is int and calls >= 0 else None


def _score_direct(request: dict[str, Any], response: Any, parent_revision: str,
                  expected_served_config: dict[str, Any], target_relation: str) -> dict[str, Any]:
    row = {"status": "provider_error", "reason": "invalid_response", "selected_id": None,
           "target_id": None, "correct": False, "provider_calls": 1,
           "forward_calls": _forward_calls(response), "response": deepcopy(response)}
    if not isinstance(response, dict) or not isinstance(response.get("ssdm"), dict):
        return row
    env = response["ssdm"]
    if env.get("schema") != DECISION_SCHEMA:
        row["reason"] = "schema_mismatch"
        return row
    if env.get("status") != "ok":
        row["status"] = env["status"] if env["status"] in ("unsupported", "exhausted", "error") else "provider_error"
        row["reason"] = row["status"]
        return row
    diagnostics = env.get("diagnostics")
    if (env.get("request_id") != request["ssdm"]["request_id"]
            or not isinstance(diagnostics, dict)
            or diagnostics.get("revision_id") != parent_revision
            or diagnostics.get("request_fingerprint") != legacy_request_fingerprint(request)
            or diagnostics.get("request_fingerprint_algorithm") != FINGERPRINT_ALGORITHM
            or not isinstance(diagnostics.get("provenance"), dict)
            or diagnostics["provenance"].get("supplied") is not False
            or diagnostics["provenance"].get("applied") is not False
            or diagnostics.get("state_handle_in") is not None
            or diagnostics.get("state_reuse") is not False
            or env.get("state_handle") is not None):
        row["reason"] = "identity_mismatch"
        return row
    question = request["questions"][DIRECT_QID]
    options = question["criteria"]
    try:
        served, _ = _check_stages(env, request, DIRECT_QID, options, expected_served_config)
    except PlannerFailure as exc:
        row["reason"] = exc.reason
        return row
    answers = response.get("answers")
    answer = answers.get(DIRECT_QID) if isinstance(answers, dict) else None
    if answer != wire_answer(question, served) or not isinstance(answers, dict) or set(answers) != {DIRECT_QID}:
        row["reason"] = "invalid_choice"
        return row
    selected = answer["choice"]
    target = next(oid for oid, description in options.items()
                  if description == f"Relation {target_relation}")
    row.update(status="ok", reason=None, selected_id=selected, target_id=target, correct=selected == target)
    return row


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
        try:
            direct_request = direct_comparison_request(example, model=model,
                                                       parent_revision=parent_revision, option_ids=option_ids)
        except Exception:
            pass  # malformed development input still occupies the paired denominator
        else:
            try:
                response = capture.decide(direct_request)
            except Exception:
                direct.update(status="provider_error", reason="provider_exception",
                              provider_calls=1, forward_calls=None)
            else:
                try:
                    direct = _score_direct(direct_request, response, parent_revision,
                                           expected_served_config, _target_relation(example))
                except Exception:
                    direct.update(provider_calls=1, forward_calls=_forward_calls(response),
                                  response=deepcopy(response))
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

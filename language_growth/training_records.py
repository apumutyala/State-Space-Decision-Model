"""Export typed slot supervision to the shared parent's existing trainer.

Requests are copied verbatim. Earlier gold slots are teacher-forced context;
current-slot labels are separate and never inserted into the model request.
"""

from copy import deepcopy
import json
from typing import Any, Iterable

from .parent_bridge import audience_state


def export_training_records(examples: Iterable[dict[str, Any]], *,
                            split: str = "development_train") -> list[dict[str, Any]]:
    """Export a single named development domain, failing closed on mixed inputs.

    Callers must explicitly select development_validation for evaluation rows.
    These rows are never an untouched audit set or a learned-result receipt.
    """
    if split not in ("development_train", "development_validation"):
        raise ValueError("only named iterative development splits are supported")
    records = []
    seen = set()
    for example in examples:
        if example.get("development_only") is not True or example.get("development_split") != split:
            raise ValueError("development split mismatch")
        task_id = example.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("missing task identity")
        requests = example.get("model_inputs")
        targets = example.get("training_only", {}).get("targets")
        if (not isinstance(requests, list) or not requests or not isinstance(targets, list)
                or len(requests) != len(targets)):
            raise ValueError("requests and targets must match exactly")
        training = example["training_only"]
        first_state = json.loads(requests[0]["state"])
        expected_state = json.loads(audience_state(first_state["task_input"], training["host"], training["catalog"]))
        prior = []
        for index, (request, target) in enumerate(zip(requests, targets), 1):
            expected_state["proposed_slot_choices"] = prior
            canonical_state = json.dumps(expected_state, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False)
            if request.get("state") != canonical_state:
                raise ValueError("state must match permitted evidence and earlier target prefix exactly")
            questions = request.get("questions")
            if not isinstance(questions, dict) or len(questions) != 1:
                raise ValueError("one question per planner slot is required")
            qid, question = next(iter(questions.items()))
            slot = target.get("slot")
            if not isinstance(slot, str) or qid != f"slot-{index:03d}-{slot}":
                raise ValueError("target slot does not match request")
            gold = target.get("option_id")
            if (question.get("type") != "choice" or not isinstance(question.get("criteria"), dict)
                    or not isinstance(gold, str) or gold not in question["criteria"]):
                raise ValueError("target is not an offered typed choice")
            identity = f"{task_id}/{qid}"
            if identity in seen:
                raise ValueError("duplicate training record identity")
            seen.add(identity)
            records.append({"id": identity, "request": deepcopy(request), "labels": {qid: gold}})
            prior.append({"slot": slot, "option_id": gold, "description": question["criteria"][gold]})
    if not records:
        raise ValueError("no training records")
    return records

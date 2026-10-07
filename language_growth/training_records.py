"""Export typed slot supervision to the shared parent's existing trainer.

L0 requests are copied verbatim. Pinned L1 requests gain prior-slot gold
prefixes only; current-slot labels stay separate from the model request.
"""

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable

from .parent_bridge import audience_state

# L1 is deliberately a single committed train cohort, not a generic split
# extension. Validation, audit, and heldout data cannot enter this branch.
L1_TRAINING_ALLOWLIST = {
    "l1_development_train": {
        "sample_sha256": "2d455c7ad7e2a1116d026bb5801dec0c821de24602375ccecf66b2569011f007",
        "manifest_sha256": "c5d94f90e4c41daab59fedb6745ecac758bce48b2724acfa7ca13e1dc47e9e2d",
    },
}


def _prepare_l1_training(examples, split, manifest):
    from .multirecord_data import multirecord_manifest, multirecord_rows
    from .multirecord_eval import FIXTURE_SOURCE_COMMIT, _adapt_task, _canonical

    pins = L1_TRAINING_ALLOWLIST[split]
    if (not isinstance(manifest, dict)
            or hashlib.sha256((_canonical(manifest) + "\n").encode("ascii")).hexdigest()
            != pins["manifest_sha256"]
            or hashlib.sha256(multirecord_rows(examples)).hexdigest() != pins["sample_sha256"]):
        raise ValueError("L1 training cohort or manifest SHA-256 mismatch")
    if _canonical(multirecord_manifest(examples)) != _canonical(manifest):
        raise ValueError("L1 training manifest identity mismatch")
    prepared, lineage = [], {}
    worlds = {world["world_id"]: world for world in manifest["worlds"]}
    for task in examples:
        # Independently check every typed reference and evidence/identity pair.
        _adapt_task(task, split, manifest["seed"])
        gold = task["evaluator_only"]
        requests, prior = [], []
        if len(task["model_inputs"]) != len(gold["targets"]):
            raise ValueError("requests and targets must match exactly")
        for request, target in zip(task["model_inputs"], gold["targets"]):
            state = json.loads(request["state"])
            expected = json.loads(audience_state(state["task_input"], gold["host"], gold["catalog"]))
            expected["proposed_slot_choices"] = []
            if request["state"] != _canonical(expected):
                raise ValueError("authored L1 state must contain only permitted evidence with empty prefixes")
            copied = deepcopy(request)
            state["proposed_slot_choices"] = deepcopy(prior)
            copied["state"] = _canonical(state)
            requests.append(copied)
            question = next(iter(request["questions"].values()))
            prior.append({"slot": target["slot"], "option_id": target["option_id"],
                          "description": question["criteria"][target["option_id"]]})
        prepared.append({"development_only": True, "development_split": split,
                         "task_id": task["task_id"], "model_inputs": requests,
                         "training_only": {"host": gold["host"], "catalog": gold["catalog"],
                                           "targets": gold["targets"]}})
        world = worlds[task["world_id"]]
        lineage[task["task_id"]] = {
            "schema": "ssdm.l1_training_lineage.v1", "development_split": split,
            "world_id": task["world_id"], "task_id": task["task_id"],
            "fixture_source_commit": FIXTURE_SOURCE_COMMIT, **pins,
            "world_rows_sha256": world["rows_sha256"],
            "catalog_sha256": world["catalog_sha256"], "context_ref": world["context_ref"],
        }
    return prepared, lineage


def export_training_records(examples: Iterable[dict[str, Any]], *,
                            split: str = "development_train",
                            l1_manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Export a single named development domain, failing closed on mixed inputs.

    Callers must explicitly select development_validation for evaluation rows.
    L1 accepts only allowlisted raw train fixtures and their pinned manifest;
    its authored empty prefixes are filled with prior target choices only.
    These rows are never an untouched audit set or a learned-result receipt.
    """
    lineage = {}
    if split in L1_TRAINING_ALLOWLIST:
        examples, lineage = _prepare_l1_training(list(examples), split, l1_manifest)
    elif split not in ("development_train", "development_validation"):
        raise ValueError("only named iterative development splits are supported")
    elif l1_manifest is not None:
        raise ValueError("L1 manifest requires the explicit L1 train split")
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
            # D adds /qid to produce an Item.id. Keep the shared task as its
            # record identity so record-level calibration guards group all
            # teacher-forced slots of one world instead of splitting them.
            records.append({"id": task_id, "request": deepcopy(request), "labels": {qid: gold}})
            if lineage:
                records[-1]["metadata"] = deepcopy(lineage[task_id])
            prior.append({"slot": slot, "option_id": gold, "description": question["criteria"][gold]})
    if not records:
        raise ValueError("no training records")
    return records

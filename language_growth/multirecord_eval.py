"""Pinned L1 fixture adapter for the existing free-running development scorer.

This is evaluation infrastructure, not a training exporter or L1 qualification.
The legacy ``training_only`` name is required by the scorer; it is evaluator
metadata here. No per-slot requests or targets become training records.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .kernel import check_program
from .multirecord_data import multirecord_manifest
from .parent_bridge import audience_state

FIXTURE_SOURCE_COMMIT = "bd31c9051732ddfa325b17f6ad986c2d08574f49"
_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "receipts/2026-10-07-L1-multirecord-data"
# Pins are independent of the caller-supplied manifest. Updating the cohort is
# a reviewed source change, never a runtime option to accept another hash.
_PINS = {
    "l1_development_train": (
        "2d455c7ad7e2a1116d026bb5801dec0c821de24602375ccecf66b2569011f007",
        "c5d94f90e4c41daab59fedb6745ecac758bce48b2724acfa7ca13e1dc47e9e2d",
    ),
    "l1_development_validation": (
        "613d5b293cebbb883c46df6e8d0f361b61d81e65277d8ba8a3eeb8baacca141d",
        "6a6d3f8012c8b6089c6c8d40a2c6eb6ded60320ba7551d332fdd668bdd9d1cc1",
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _adapt_task(task: dict[str, Any], split: str, seed: int) -> dict[str, Any]:
    """Check identity/evidence consistency after the file pins are verified."""
    if (task.get("schema") != "ssdm.synthetic_l1_development_task.v1"
            or task.get("development_only") is not True
            or task.get("development_split") != split):
        raise ValueError("L1 fixture schema or split mismatch")
    gold = task["evaluator_only"]
    world_id = f"synthetic-{split}-{seed}-world-{gold['world_index']}"
    if (gold["seed"] != seed or task["world_id"] != world_id
            or task["task_id"] != f"{world_id}-task-{gold['task_index']}"
            or gold["template_id"] != f"{split}:{gold['kind']}:0"):
        raise ValueError("L1 fixture identity mismatch")
    host, catalog, program = gold["host"], gold["catalog"], gold["program"]
    first = json.loads(task["model_inputs"][0]["state"])
    expected = json.loads(audience_state(first["task_input"], host, catalog))
    expected["proposed_slot_choices"] = []
    if first != expected:
        raise ValueError("initial request must match permitted evidence with empty prefixes")
    if (any(program[key] != host[key] for key in ("context_ref", "policy_ref", "registry_ref"))
            or check_program(program, host, catalog).status != "accepted"):
        raise ValueError("L1 fixture reference program mismatch")
    return {
        "schema": "ssdm.l1_free_running_evaluation_task.v1",
        "development_only": True, "development_split": split,
        "world_id": world_id, "task_id": task["task_id"],
        # The runner reads only task wording from this state and constructs its
        # own requests. Discard every authored question, option ID and slot.
        "model_inputs": [{"state": _canonical(first)}],
        "training_only": copy.deepcopy({
            "kind": gold["kind"], "host": host, "catalog": catalog, "program": program,
        }),
    }


def load_multirecord_evaluation(split: str, *, fixture_dir: Path = _FIXTURE_DIR) -> list[dict[str, Any]]:
    """Load one exact committed iterative cohort, preserving its L1 split name.

    The optional directory permits relocated copies of these exact files, not
    arbitrary data or replacement manifests. All validation precedes return;
    this function never receives a provider and cannot call a model.
    """
    if split not in _PINS:
        raise ValueError("only pinned L1 development splits are supported")
    rows = (Path(fixture_dir) / f"{split}.jsonl").read_bytes()
    manifest_bytes = (Path(fixture_dir) / f"{split}-manifest.json").read_bytes()
    actual = (hashlib.sha256(rows).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest())
    if actual != _PINS[split]:
        raise ValueError("L1 fixture or manifest SHA-256 mismatch")
    tasks = [json.loads(line) for line in rows.splitlines()]
    manifest = json.loads(manifest_bytes)
    if (manifest["split"] != split
            or _canonical(multirecord_manifest(tasks)) != _canonical(manifest)):
        raise ValueError("L1 fixture manifest identity mismatch")
    return [_adapt_task(task, split, manifest["seed"]) for task in tasks]

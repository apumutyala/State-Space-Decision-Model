"""GatePlanner supervision replayed from the pinned L1 fit-world cohort.

This is diagnostic teacher-forced training data, not free-running capability
or qualification. The adapter verifies the whole-world partition first, then
replays only the 24 fit worlds through the current GatePlanner. Calibration,
development-validation and language-heldout labels never enter this export.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .gate_suites import ANSWER_FORMAT, canonical, digest
from .gate_training import _replay_task
from .multirecord_calibration_split import split_multirecord_training
from .multirecord_data import (
    MULTIWORLD_TRAIN_SPLIT,
    TASKS_PER_WORLD,
    multirecord_manifest,
    multirecord_rows,
)
from .multirecord_eval import MULTIWORLD_FIXTURE_DIR, _adapt_task, _canonical
from .training_records import L1_TRAINING_ALLOWLIST
from .l1_world_training import (
    CALIBRATION_RECORDS_SHA256,
    FIT_RECORDS_SHA256,
    PARTITION_MANIFEST_SHA256,
    _manifest_bytes,
    _record_bytes,
)
from .parent_bridge import audience_state


EXPORT_SCHEMA = "ssdm.l1_gate_training_export.v1"
EXPECTED_FIT_WORLDS = 24
EXPECTED_TASKS = EXPECTED_FIT_WORLDS * TASKS_PER_WORLD
EXPECTED_ROWS = EXPECTED_FIT_WORLDS * 128
EXPECTED_RELATIONS_PER_FIELD_PER_WORLD = {"equal": 2, "greater_than": 5, "less_than": 5}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _task_and_reference(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt one validated source task without carrying evaluator data forward."""
    if source.get("development_split") != MULTIWORLD_TRAIN_SPLIT:
        raise ValueError("only the pinned L1 development-training source is allowed")
    source_world_id, source_task_id = source.get("world_id"), source.get("task_id")
    gold = source.get("training_only")
    if (not isinstance(source_world_id, str) or not source_world_id
            or not isinstance(source_task_id, str) or not source_task_id
            or not isinstance(gold, dict)
            or set(gold) != {"kind", "host", "catalog", "program"}
            or gold["kind"] not in {"read", "explain", "compare"}):
        raise ValueError("pinned L1 fit task schema mismatch")

    initial = json.loads(source["model_inputs"][0]["state"])
    expected_state = json.loads(audience_state(initial.get("task_input"), gold["host"], gold["catalog"]))
    expected_state["proposed_slot_choices"] = []
    if initial != expected_state:
        raise ValueError("source request and pinned evaluator evidence differ")

    # Namespace IDs for compatibility with the existing train-only replay
    # validator while preserving exact source IDs in metadata below.
    world_id = f"language-development_train-l1-fit-{source_world_id}"
    task_id = f"language-development_train-l1-fit-{source_task_id}"
    task = {
        "task_id": task_id,
        "world_id": world_id,
        "level": "L1",
        "split": "development_train",
        "task_input": initial["task_input"],
        "host": gold["host"],
        "catalog": gold["catalog"],
        "planner_profile": None,
        "answer_format": ANSWER_FORMAT,
        "production": "plain",
    }
    reference = {
        "task_id": task_id,
        "world_id": world_id,
        "input_sha256": digest(task),
        "program": gold["program"],
    }
    return task, reference


def _fit_coverage(fit_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove the pinned fit source retains its declared pointer/field/relation coverage."""
    by_world: dict[str, list[dict[str, Any]]] = {}
    kind_counts: Counter[str] = Counter()
    global_relation_counts: dict[str, Counter[str]] = {
        "price": Counter(), "warranty_months": Counter(),
    }


    for task in fit_tasks:
        by_world.setdefault(task["world_id"], []).append(task)
        gold = task["training_only"]
        kind = gold["kind"]
        kind_counts[kind] += 1
        program = gold["program"]
        nodes = {node["id"]: node for node in program["nodes"]}
        read_nodes = [node for node in program["nodes"] if node["op"] == "ReadFact"]
        if kind in {"read", "explain"}:
            if len(read_nodes) != 1:
                raise ValueError("read/explain source task must contain one ReadFact")
        elif kind == "compare":
            compare_nodes = [node for node in program["nodes"] if node["op"] == "Compare"]
            if len(read_nodes) != 2 or len(compare_nodes) != 1:
                raise ValueError("comparison source task must contain two facts and one Compare")
            compare = compare_nodes[0]
            left, right = nodes[compare["left"]["node"]], nodes[compare["right"]["node"]]
            if (left["op"] != "ReadFact" or right["op"] != "ReadFact"
                    or left["field"] != right["field"]):
                raise ValueError("comparison operands must be same-field typed facts")
            global_relation_counts[left["field"]][compare["relation"]] += 1
        else:
            raise ValueError("unexpected L1 fit task kind")

    for world_id, tasks in by_world.items():
        if len(tasks) != TASKS_PER_WORLD:
            raise ValueError("fit world is missing source tasks")
        catalog_records = tasks[0]["training_only"]["catalog"]["records"]
        record_refs = {row["record_ref"] for row in catalog_records}
        fields = set(catalog_records[0]["fields"])
        expected_facts = {(ref, field) for ref in record_refs for field in fields}
        read_facts: dict[str, set[tuple[str, str]]] = {"read": set(), "explain": set()}
        comparisons: set[tuple[str, str, str]] = set()
        relations_by_field: dict[str, Counter[str]] = {field: Counter() for field in fields}
        for task in tasks:
            gold = task["training_only"]
            program = gold["program"]
            nodes = {node["id"]: node for node in program["nodes"]}
            if gold["kind"] in read_facts:
                fact = next(node for node in program["nodes"] if node["op"] == "ReadFact")
                read_facts[gold["kind"]].add((fact["record_ref"], fact["field"]))
            elif gold["kind"] == "compare":
                compare = next(node for node in program["nodes"] if node["op"] == "Compare")
                left = nodes[compare["left"]["node"]]
                right = nodes[compare["right"]["node"]]
                triple = (left["record_ref"], right["record_ref"], left["field"])
                if triple in comparisons:
                    raise ValueError("duplicate ordered comparison in a fit world")
                comparisons.add(triple)
                relations_by_field[left["field"]][compare["relation"]] += 1
        expected_comparisons = {(left, right, field) for left in record_refs for right in record_refs
                                if left != right for field in fields}
        if (fields != {"price", "warranty_months"}
                or any(read_facts[kind] != expected_facts for kind in ("read", "explain"))
                or comparisons != expected_comparisons
                or any(dict(counts) != EXPECTED_RELATIONS_PER_FIELD_PER_WORLD
                       for counts in relations_by_field.values())):
            raise ValueError(f"fit-world semantic coverage mismatch: {world_id}")

    expected_kinds = {"compare": 576, "explain": 192, "read": 192}
    if dict(kind_counts) != expected_kinds or len(by_world) != EXPECTED_FIT_WORLDS:
        raise ValueError("fit cohort kind/world distribution mismatch")
    expected_global_relations = {
        relation: count * EXPECTED_FIT_WORLDS
        for relation, count in EXPECTED_RELATIONS_PER_FIELD_PER_WORLD.items()
    }
    if any(dict(counts) != expected_global_relations for counts in global_relation_counts.values()):
        raise ValueError("fit cohort comparison-relation distribution mismatch")
    return {
        "fields": ["price", "warranty_months"],
        "records_per_world": 4,
        "read_record_field_pairs_per_world": 8,
        "explain_record_field_pairs_per_world": 8,
        "ordered_comparison_field_triples_per_world": 24,
        "comparison_relations_per_field_per_world": EXPECTED_RELATIONS_PER_FIELD_PER_WORLD,
        "comparison_relations_per_field_in_fit": expected_global_relations,
        "source_kind_counts": dict(sorted(kind_counts.items())),
    }


def _target_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count the labeled semantic option descriptions outside model requests."""
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        labels = row.get("labels")
        request = row.get("request")
        if not isinstance(labels, dict) or len(labels) != 1 or not isinstance(request, dict):
            raise ValueError("each GatePlanner slot row must have one separate label and request")
        qid, option_id = next(iter(labels.items()))
        question = request.get("questions", {}).get(qid)
        description = question.get("criteria", {}).get(option_id) if isinstance(question, dict) else None
        slot = row.get("metadata", {}).get("slot")
        if not isinstance(slot, str) or not isinstance(description, str):
            raise ValueError("training target must map to one offered semantic description")
        counts.setdefault(slot, Counter())[description] += 1
    return {slot: dict(sorted(values.items())) for slot, values in sorted(counts.items())}


def generate_l1_fit_training_records(*, model: str = "gate-training",
                                     parent_revision: str = "training-only:v1") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay all and only the exact 24 pinned L1 fit worlds via GatePlanner."""
    if not isinstance(model, str) or not model or not isinstance(parent_revision, str) or not parent_revision:
        raise ValueError("model and parent revision pins must be nonempty strings")

    # Verify the pinned train bytes and the whole-world fit/calibration split.
    # The separate development-validation input is not opened by this adapter.
    partitions, partition_manifest = split_multirecord_training()
    fit_records, calibration_records = partitions["fit"], partitions["calibration"]
    if (_sha(_record_bytes(fit_records)) != FIT_RECORDS_SHA256
            or _sha(_record_bytes(calibration_records)) != CALIBRATION_RECORDS_SHA256
            or _sha(_manifest_bytes(partition_manifest)) != PARTITION_MANIFEST_SHA256):
        raise ValueError("pinned L1 24/8 fit/calibration partition identity mismatch")
    fit_worlds = frozenset(row["metadata"]["world_id"] for row in fit_records)
    calibration_worlds = frozenset(row["metadata"]["world_id"] for row in calibration_records)
    if (len(fit_worlds) != EXPECTED_FIT_WORLDS or len(calibration_worlds) != 8
            or fit_worlds & calibration_worlds):
        raise ValueError("pinned 24/8/32 L1 world partition required")

    pins = L1_TRAINING_ALLOWLIST[MULTIWORLD_TRAIN_SPLIT]
    source_path = MULTIWORLD_FIXTURE_DIR / f"{MULTIWORLD_TRAIN_SPLIT}.jsonl"
    manifest_path = MULTIWORLD_FIXTURE_DIR / f"{MULTIWORLD_TRAIN_SPLIT}-manifest.json"
    source_bytes, source_manifest_bytes = source_path.read_bytes(), manifest_path.read_bytes()
    if (_sha(source_bytes) != pins["sample_sha256"]
            or _sha(source_manifest_bytes) != pins["manifest_sha256"]):
        raise ValueError("pinned L1 training source or manifest SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_bytes)
    raw_tasks = [json.loads(line) for line in source_bytes.splitlines()]
    if (multirecord_rows(raw_tasks) != source_bytes
            or _canonical(multirecord_manifest(raw_tasks)) != _canonical(source_manifest)):
        raise ValueError("pinned L1 source tasks differ from their canonical manifest")
    source_tasks = [_adapt_task(task, MULTIWORLD_TRAIN_SPLIT, source_manifest["seed"])
                    for task in raw_tasks]
    if len(source_tasks) != 32 * TASKS_PER_WORLD:
        raise ValueError("complete pinned 32-world L1 train cohort required")
    fit_tasks = [task for task in source_tasks if task.get("world_id") in fit_worlds]
    source_fit_worlds = frozenset(task.get("world_id") for task in fit_tasks)
    counts = Counter(task.get("world_id") for task in fit_tasks)
    if (len(fit_tasks) != EXPECTED_TASKS or source_fit_worlds != fit_worlds
            or set(counts.values()) != {TASKS_PER_WORLD}):
        raise ValueError("fit source must contain exactly 40 complete tasks in each pinned fit world")
    coverage = _fit_coverage(fit_tasks)

    rows: list[dict[str, Any]] = []
    source_task_ids: set[str] = set()
    for source in fit_tasks:
        task, reference = _task_and_reference(source)
        generated = _replay_task(task, reference, model=model, parent_revision=parent_revision)
        gold = source["training_only"]
        for row in generated:
            row["metadata"]["schema"] = EXPORT_SCHEMA
            row["metadata"]["source_cohort"] = MULTIWORLD_TRAIN_SPLIT
            row["metadata"]["source_task_id"] = source["task_id"]
            row["metadata"]["source_world_id"] = source["world_id"]
            row["metadata"]["source_kind"] = gold["kind"]
            row["metadata"]["source_program_sha256"] = digest(gold["program"])
            if source["world_id"] not in fit_worlds:
                raise ValueError("non-fit L1 source entered the training export")
        if source["task_id"] in source_task_ids:
            raise ValueError("duplicate source task identity")
        source_task_ids.add(source["task_id"])
        rows.extend(generated)

    slot_counts = Counter(row["metadata"]["slot"] for row in rows)
    kind_counts = Counter(task["training_only"]["kind"] for task in fit_tasks)
    expected_kind_counts = {"compare": 24 * 24, "explain": 8 * 24, "read": 8 * 24}
    expected_slots = {
        "semantic-operation": EXPECTED_TASKS,
        "fact": 384,
        "left-operand": 576,
        "right-operand": 576,
        "comparison-relation": 576,
    }
    if (len(rows) != EXPECTED_ROWS or dict(slot_counts) != expected_slots
            or dict(kind_counts) != expected_kind_counts):
        raise ValueError("GatePlanner slot counts differ from the pinned 24-world L1 fit cohort")

    task_ids = {row["id"] for row in rows}
    world_ids = {row["metadata"]["world_id"] for row in rows}
    if (len(task_ids) != EXPECTED_TASKS or len(world_ids) != EXPECTED_FIT_WORLDS
            or any(row["metadata"]["source_world_id"] not in fit_worlds for row in rows)
            or any("source_world_id" in repr(row["request"]) for row in rows)):
        raise ValueError("training export identity or request-isolation check failed")
    target_counts = _target_counts(rows)
    if {slot: sum(values.values()) for slot, values in target_counts.items()} != dict(slot_counts):
        raise ValueError("semantic target frequencies do not cover every exported slot row")
    if (target_counts.get("semantic-operation") != {
            "Compare": 576, "ExplainDerivation": 192, "ReadFact": 192}
            or target_counts.get("comparison-relation") != {
                "equal": 96, "greater_than": 240, "less_than": 240}):
        raise ValueError("semantic operation/relation targets differ from the pinned source distribution")

    source_file_hashes = {
        name: _sha((Path(__file__).with_name(name)).read_bytes())
        for name in ("l1_gate_training.py", "gate_training.py", "gate_suites.py",
                     "gate_planner.py", "planner.py", "parent_bridge.py",
                     "l1_world_training.py", "multirecord_calibration_split.py",
                     "multirecord_data.py", "multirecord_eval.py", "training_records.py")
    }
    manifest = {
        "schema": EXPORT_SCHEMA,
        "split": "development_train",
        "source": "pinned L1 24-world fit cohort replayed through GatePlanner",
        "source_split": MULTIWORLD_TRAIN_SPLIT,
        "source_jsonl_sha256": pins["sample_sha256"],
        "source_manifest_sha256": pins["manifest_sha256"],
        "fit_records_sha256": FIT_RECORDS_SHA256,
        "calibration_records_sha256": CALIBRATION_RECORDS_SHA256,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "fit_world_ids": sorted(fit_worlds),
        "records": len(rows),
        "tasks": len(task_ids),
        "worlds": len(world_ids),
        "tasks_per_world": TASKS_PER_WORLD,
        "slot_counts": dict(sorted(slot_counts.items())),
        "target_counts_by_slot": target_counts,
        "target_counts_sha256": digest(target_counts),
        "source_task_count": len(source_task_ids),
        "source_kind_counts": dict(sorted(kind_counts.items())),
        "source_coverage": coverage,
        "source_files_sha256": source_file_hashes,
        "model": model,
        "parent_revision": parent_revision,
        "diagnostic_teacher_forced": True,
        "free_running": False,
        "promotion_eligible": False,
        "validation_or_heldout_generated": False,
        "limitations": [
            "Synthetic checker-valid supervision is not independent audit or learned capability.",
            "Rows are teacher-forced planner slots; evaluate free-running behavior separately.",
            "Only the exact pinned 24 fit worlds are included; calibration remains reserved.",
        ],
    }
    return rows, manifest


def export_l1_fit_training(out: Path, **kwargs) -> dict[str, Any]:
    """Write a deterministic export atomically; refuse replacement of existing work."""
    out = Path(out)
    if out.exists():
        raise ValueError("training output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, manifest = generate_l1_fit_training_records(**kwargs)
    data = b"".join(canonical(row) + b"\n" for row in rows)
    compressed = gzip.compress(data, mtime=0)
    manifest = {
        **manifest,
        "training_records_sha256": _sha(data),
        "training_records_gzip_sha256": _sha(compressed),
    }

    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    try:
        (stage / "training_records.jsonl").write_bytes(data)
        (stage / "training_records.jsonl.gz").write_bytes(compressed)
        (stage / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        if out.exists():
            raise ValueError("training output already exists")
        os.replace(stage, out)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gate-training")
    parser.add_argument("--parent-revision", default="training-only:v1")
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    print(json.dumps(export_l1_fit_training(out, **args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

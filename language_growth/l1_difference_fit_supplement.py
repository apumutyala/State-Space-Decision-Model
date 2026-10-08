"""Balanced, train-only ExactDifference coverage for synthetic L1 worlds.

Enumerates every ordered pair of distinct records for both registered numeric
fields. Labels are emitted only by the existing GatePlanner replay helper;
this is teacher-forced supervision and not learned-capability evidence.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .gate_suites import ANSWER_FORMAT, _program, canonical, digest
from .gate_training import _replay_task
from .kernel import check_program
from .multirecord_data import MULTIWORLD_TRAIN_SPLIT, _world

SCHEMA = "ssdm.agent_c.l1_exactdifference_fit_supplement.v1"
SEED = 20261010
WORLDS = 24
FIELDS = ("price", "warranty_months")
MODEL = "gate-training-l1-difference-fit-v2"
PARENT_REVISION = "training-only-l1-difference-v2"
SOURCE_FILES = (
    "multirecord_data.py", "gate_training.py", "gate_suites.py", "gate_planner.py",
    "planner.py", "parent_bridge.py", "kernel.py", "realize.py", "gate_comparator.py",
    "gate_answer_format.py", "types.py", "l1_difference_fit_supplement.py",
)
EXPECTED_SIGNS = {
    "price": {"positive": 120, "negative": 120, "tie": 48},
    "warranty_months": {"positive": 120, "negative": 120, "tie": 48},
}


class DifferenceSupplementError(ValueError):
    pass


def _source_pins() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    return {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def generate_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signs: Counter[tuple[str, str]] = Counter()
    slot_labels: Counter[tuple[str, str]] = Counter()
    world_ids: list[str] = []
    task_ids: set[str] = set()

    for world_index in range(WORLDS):
        host, catalog = _world(MULTIWORLD_TRAIN_SPLIT, SEED, world_index)
        registry = catalog["registries"][host["registry_ref"]]
        if "ExactDifference" in registry["allowed_ops"]:
            raise DifferenceSupplementError("source registry unexpectedly already enables ExactDifference")
        registry["allowed_ops"].append("ExactDifference")

        old_context = host["context_ref"]
        try:
            snapshot = catalog["snapshots"].pop(old_context)
        except KeyError as exc:
            raise DifferenceSupplementError("source context snapshot is missing") from exc
        context = "synthetic-l1-difference-fit-v2:" + digest({"catalog": catalog, "snapshot": snapshot})
        host["context_ref"] = context
        host["provider"]["input_context_ref"] = context
        catalog["snapshots"][context] = snapshot

        world_id = f"language-development_train-{SEED}-difference-world-{world_index}"
        world_ids.append(world_id)
        records_by_ref = {record["record_ref"]: record for record in catalog["records"]}
        records = catalog["records"]
        if len(records) != 4 or len(records_by_ref) != 4:
            raise DifferenceSupplementError("fit source world must contain four unique records")

        for field in FIELDS:
            for left_index, left_record in enumerate(records):
                for right_index, right_record in enumerate(records):
                    if left_index == right_index:
                        continue
                    task_id = f"{world_id}-{field}-{left_index}-{right_index}"
                    if task_id in task_ids:
                        raise DifferenceSupplementError("duplicate task identity")
                    task_ids.add(task_id)
                    nodes = [
                        {"id": "n1", "op": "ReadFact", "record_ref": left_record["record_ref"], "field": field},
                        {"id": "n2", "op": "ReadFact", "record_ref": right_record["record_ref"], "field": field},
                        {"id": "n3", "op": "ExactDifference",
                         "left": {"node": "n1", "project": "object"},
                         "right": {"node": "n2", "project": "object"}},
                        {"id": "n4", "op": "Assert", "proposition": "n3"},
                    ]
                    task = {
                        "task_id": task_id,
                        "world_id": world_id,
                        "level": "L1",
                        "split": "development_train",
                        "task_input": (f"For October 2026, calculate {left_record['entity_ref']}'s {field} "
                                       f"minus {right_record['entity_ref']}'s {field}."),
                        "host": host,
                        "catalog": catalog,
                        "planner_profile": None,
                        "answer_format": ANSWER_FORMAT,
                        "production": "plain",
                    }
                    program = _program(host, nodes)
                    checked = check_program(program, host, catalog)
                    if checked.status != "accepted":
                        raise DifferenceSupplementError(f"reference rejected: {checked.reason}")
                    left = Decimal(str(left_record["fields"][field]["value"]))
                    right = Decimal(str(right_record["fields"][field]["value"]))
                    expected = left - right
                    content = checked.outputs[0].content
                    if content.value != expected:
                        raise DifferenceSupplementError("kernel result differs from exact Decimal subtraction")
                    sign = "positive" if expected > 0 else "negative" if expected < 0 else "tie"
                    signs[field, sign] += 1
                    reference = {"task_id": task_id, "world_id": world_id,
                                 "input_sha256": digest(task), "program": program}
                    task_rows = _replay_task(
                        task, reference, model=MODEL, parent_revision=PARENT_REVISION)
                    if len(task_rows) != 3:
                        raise DifferenceSupplementError("ExactDifference task did not replay to three slots")
                    for row in task_rows:
                        slot = row["metadata"]["slot"]
                        slot_labels[slot, next(iter(row["labels"].values()))] += 1
                    rows.extend(task_rows)

    sign_counts = {field: {sign: signs[field, sign] for sign in ("positive", "negative", "tie")}
                   for field in FIELDS}
    if sign_counts != EXPECTED_SIGNS:
        raise DifferenceSupplementError(f"unexpected signed/tie coverage: {sign_counts}")
    if (len(rows) != 1728 or len(task_ids) != 576 or len(set(world_ids)) != WORLDS
            or Counter(row["metadata"]["slot"] for row in rows) !=
            Counter({"semantic-operation": 576, "left-operand": 576, "right-operand": 576})):
        raise DifferenceSupplementError("unexpected task, row, world, or slot totals")
    label_counts = {slot: dict(sorted((label, count) for (found_slot, label), count in slot_labels.items()
                                      if found_slot == slot))
                    for slot in ("semantic-operation", "left-operand", "right-operand")}
    if (label_counts["semantic-operation"] != {"v004": 576}
            or any(label_counts[slot] != {f"v{i:03d}": 72 for i in range(1, 9)}
                   for slot in ("left-operand", "right-operand"))):
        raise DifferenceSupplementError("option-order target counts are not the registered balanced profile")
    return rows, {"world_ids": world_ids, "task_ids": len(task_ids), "sign_counts": sign_counts,
                  "slot_counts": {"semantic-operation": 576, "left-operand": 576, "right-operand": 576},
                  "target_counts_by_slot": label_counts}


def export_training(out: Path) -> dict[str, Any]:
    out = Path(out)
    if out.exists():
        raise DifferenceSupplementError("training output already exists")
    rows, coverage = generate_records()
    data = b"".join(canonical(row) + b"\n" for row in rows)
    compressed = gzip.compress(data, mtime=0)
    manifest = {
        "schema": SCHEMA,
        "split": "development_train",
        "level": "L1",
        "world_source": "language_growth.multirecord_data._world/l1_multiworld_development_train",
        "seed": SEED,
        "requested_worlds": WORLDS,
        "records": len(rows),
        "tasks": coverage["task_ids"],
        "worlds": len(coverage["world_ids"]),
        "fields": list(FIELDS),
        "slot_counts": coverage["slot_counts"],
        "target_counts_by_slot": coverage["target_counts_by_slot"],
        "difference_sign_counts_by_field": coverage["sign_counts"],
        "source_sha256": _source_pins(),
        "training_records_sha256": hashlib.sha256(data).hexdigest(),
        "training_records_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
        "diagnostic_teacher_forced": True,
        "free_running": False,
        "promotion_eligible": False,
        "validation_or_heldout_generated": False,
        "limitations": [
            "Synthetic checker-valid supervision is not independent audit or learned capability.",
            "Each world contains four synthetic records with two numeric fields.",
            "Tasks enumerate ordered distinct-record pairs; self-pairs are excluded.",
            "Ties between distinct records are included; no general arithmetic claim follows.",
            "Earlier target choices are diagnostic teacher forcing; free-running evaluation is separate.",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "training_records.jsonl").write_bytes(data)
    (out / "training_records.jsonl.gz").write_bytes(compressed)
    (out / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_training(args.out), sort_keys=True))

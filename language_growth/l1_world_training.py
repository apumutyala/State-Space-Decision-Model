"""Load the pinned whole-world L1 development partitions for the existing trainer.

This is a split-before-Item-conversion boundary. Fit records alone can be
converted to trainer Items; calibration stays reserved, and validation is
returned as the separate free-running task schema. No model is loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from .multirecord_calibration_split import split_multirecord_training
from .multirecord_data import MULTIWORLD_VALIDATION_SPLIT
from .multirecord_eval import MULTIWORLD_FIXTURE_DIR, load_multirecord_evaluation


FIT_RECORDS_SHA256 = "34909156035594729d7891d64c840efb8ed853dfe4fcb10a56f77b0fa8e91be0"
CALIBRATION_RECORDS_SHA256 = "f58d51a51cdb6621d9b7ae86d348d67406488e13bb0c6839c174aaa5c47dce28"
PARTITION_MANIFEST_SHA256 = "3ad9f962cc6896c4540a13633ea371a00d8537c46d061813b66188e43920b0ec"


def _record_bytes(rows: list[dict[str, Any]]) -> bytes:
    import json

    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii") for row in rows)


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    import json

    return (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class L1WorldTrainingData:
    """Separate optimizer data, reserved calibration records and free-running eval tasks."""

    fit_records: list[dict[str, Any]]
    calibration_records: list[dict[str, Any]]
    validation_tasks: list[dict[str, Any]]
    partition_manifest: dict[str, Any]
    fit_world_ids: frozenset[str]
    calibration_world_ids: frozenset[str]
    validation_world_ids: frozenset[str]

    def fit_items(self, encode, *, guard, layout: str = "ordered", template="jrt-plain-v1"):
        """Convert only fit rows, after whole-world assignment, preserving a metric sidecar.

        `world_by_item_id` is deliberately external to `Item.row` and to the
        model request. The validation tasks are not returned as Trainer eval
        Items: they must be evaluated by the free-running planner path.
        """
        if guard is None or not callable(getattr(guard, "check", None)):
            raise ValueError("fit Item conversion requires the protected-data SplitGuard")
        if hashlib.sha256(_record_bytes(list(self.fit_records))).hexdigest() != FIT_RECORDS_SHA256:
            raise ValueError("fit records changed after the pinned world split")
        current_fit_worlds = frozenset(
            r.get("metadata", {}).get("world_id") for r in self.fit_records
            if isinstance(r.get("metadata"), dict)
        )
        if (None in current_fit_worlds or current_fit_worlds != self.fit_world_ids
                or current_fit_worlds & self.calibration_world_ids
                or current_fit_worlds & self.validation_world_ids):
            raise ValueError("fit records no longer match the reserved world partition")
        from train.data import items_from_records

        items = items_from_records(self.fit_records, encode, layout=layout, template=template, guard=guard)
        world_by_task: dict[str, str] = {}
        for record in self.fit_records:
            task_id = record.get("id")
            metadata = record.get("metadata")
            world_id = metadata.get("world_id") if isinstance(metadata, dict) else None
            if not isinstance(task_id, str) or not isinstance(world_id, str):
                raise ValueError("fit record lost its task/world sidecar")
            if task_id in world_by_task and world_by_task[task_id] != world_id:
                raise ValueError("one task maps to multiple worlds")
            world_by_task[task_id] = world_id

        world_by_item: dict[str, str] = {}
        for item in items:
            task_id, sep, _qid = item.id.rpartition("/")
            if not sep or task_id not in world_by_task or item.id in world_by_item:
                raise ValueError("trainer Item identity does not map uniquely to a fit task")
            world_by_item[item.id] = world_by_task[task_id]
        expected_ids = {f"{r['id']}/{qid}" for r in self.fit_records for qid in r["labels"]}
        if set(world_by_item) != expected_ids:
            raise ValueError("fit Item conversion changed the pinned item set")
        return items, world_by_item


def load_l1_world_training_data(*, fixture_dir: Path | None = None) -> L1WorldTrainingData:
    """Load exact pinned train/calibration/validation cohorts and verify their split identity.

    `fixture_dir` only relocates the committed bytes; the source loaders retain
    their independent SHA-256 and manifest checks.
    """
    root = MULTIWORLD_FIXTURE_DIR if fixture_dir is None else Path(fixture_dir)
    partitions, manifest = split_multirecord_training(fixture_dir=root)
    fit, calibration = partitions["fit"], partitions["calibration"]
    if hashlib.sha256(_record_bytes(fit)).hexdigest() != FIT_RECORDS_SHA256:
        raise ValueError("fit partition differs from the reviewed whole-world split")
    if hashlib.sha256(_record_bytes(calibration)).hexdigest() != CALIBRATION_RECORDS_SHA256:
        raise ValueError("calibration partition differs from the reviewed whole-world split")
    if hashlib.sha256(_manifest_bytes(manifest)).hexdigest() != PARTITION_MANIFEST_SHA256:
        raise ValueError("L1 whole-world partition manifest differs from the reviewed split")

    validation = load_multirecord_evaluation(MULTIWORLD_VALIDATION_SPLIT, fixture_dir=root)
    fit_worlds = frozenset(r["metadata"]["world_id"] for r in fit)
    calibration_worlds = frozenset(r["metadata"]["world_id"] for r in calibration)
    validation_worlds = frozenset(r["world_id"] for r in validation)
    if (len(fit_worlds), len(calibration_worlds), len(validation_worlds)) != (24, 8, 32):
        raise ValueError("L1 requires exactly 24 fit, 8 calibration and 32 validation worlds")
    if ((fit_worlds & calibration_worlds) or (fit_worlds & validation_worlds)
            or (calibration_worlds & validation_worlds)):
        raise ValueError("fit, calibration and free-running validation worlds must be disjoint")
    if (len(fit), len(calibration), len(validation)) != (3072, 1024, 1280):
        raise ValueError("L1 whole-world partition row/task counts changed")

    # Metadata, current labels, and world identities are sidecars, never request content.
    for record in fit + calibration:
        if "metadata" not in record or "labels" not in record or "request" not in record:
            raise ValueError("L1 trainer row lost separated labels or lineage metadata")
        request_text = repr(record["request"])
        if record["metadata"]["world_id"] in request_text:
            raise ValueError("world identity leaked into the model request")
    for task in validation:
        if "evaluator_only" in task or "targets" in task:
            raise ValueError("validation adapter must expose only initial request and evaluator sidecar")

    return L1WorldTrainingData(
        fit_records=tuple(fit),
        calibration_records=tuple(calibration),
        validation_tasks=tuple(validation),
        partition_manifest=manifest,
        fit_world_ids=fit_worlds,
        calibration_world_ids=calibration_worlds,
        validation_world_ids=validation_worlds,
    )

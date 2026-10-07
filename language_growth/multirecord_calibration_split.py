"""Prepare a fixed world-level L1 fit/calibration split; $0, no model execution.

This is synthetic development data preparation, not training or calibrated
capability evidence. The separate validation cohort remains evaluation-only.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import tempfile

from .multirecord_data import MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT
from .multirecord_eval import MULTIWORLD_FIXTURE_DIR, _canonical, _PINS
from .multirecord_training import load_multirecord_training
from .training_records import L1_GENERATOR_SOURCE, L1_TRAINING_ALLOWLIST


# Preregistered constants: no caller-selected seed, size, labels or search.
SPLIT_SEED = 1708
SPLIT_METHOD = "sha256-canonical-[ssdm.l1_world_split.v1,seed,world_id]-ascending-first-8"
SOURCE_COMMIT = "e94c79793082fc8643bae34ac52813760f1950e8"
TRAINING_EXPORT_SHA256 = "a7806d12672be658f66f75d57a0d53a888cb00b93074b183a0a7c5bceb52d489"


def _rows_bytes(rows):
    return b"".join((_canonical(row) + "\n").encode("ascii") for row in rows)


def calibration_world_ids(world_ids):
    """Select solely from 32 unique world IDs using the fixed seed and rule."""
    ids = list(world_ids)
    if (len(ids) != 32 or any(not isinstance(w, str) or not w for w in ids)
            or len(set(ids)) != 32):
        raise ValueError("exactly 32 unique world IDs are required")

    def rank(world_id):
        payload = _canonical(["ssdm.l1_world_split.v1", SPLIT_SEED, world_id]).encode("ascii")
        return hashlib.sha256(payload).hexdigest(), world_id

    return frozenset(sorted(ids, key=rank)[:8])


def split_multirecord_training(*, split=MULTIWORLD_TRAIN_SPLIT,
                               fixture_dir=MULTIWORLD_FIXTURE_DIR):
    """Load the exact pinned train cohort/export before assigning any worlds.

    No arbitrary export or Item input is accepted. Row content and source order
    are preserved within each partition, including labels and world metadata.
    """
    records, source = load_multirecord_training(split=split, fixture_dir=fixture_dir)
    export_hash = hashlib.sha256(_rows_bytes(records)).hexdigest()
    if export_hash != TRAINING_EXPORT_SHA256:
        raise ValueError("pinned L1 training export SHA-256 mismatch")
    worlds = {w["world_id"]: w for w in source["worlds"]}
    calibration_ids = calibration_world_ids(worlds)
    partitions = {"fit": [], "calibration": []}
    world_rows = defaultdict(list)
    for row in records:
        world_id = row["metadata"]["world_id"]
        world_rows[world_id].append(row)
        partitions["calibration" if world_id in calibration_ids else "fit"].append(row)
    # These checks also make complete-world membership explicit in the manifest.
    for world_id, world in worlds.items():
        rows = world_rows[world_id]
        slots = [(r["id"], next(iter(r["request"]["questions"]))) for r in rows]
        if (len(rows) != 128 or len(set(slots)) != 128
                or set(r["id"] for r in rows) != set(world["task_ids"])
                or len(world["task_ids"]) != 40):
            raise ValueError("incomplete L1 task/world slot coverage")
    if set(world_rows) != set(worlds):
        raise ValueError("unknown L1 world identity")
    summaries = {}
    for name, rows in partitions.items():
        counts = Counter(r["metadata"]["world_id"] for r in rows)
        summaries[name] = {
            "file": f"{name}.jsonl", "world_count": len(counts),
            "task_count": len({r["id"] for r in rows}), "row_count": len(rows),
            "bytes": len(_rows_bytes(rows)),
            "sha256": hashlib.sha256(_rows_bytes(rows)).hexdigest(),
            "worlds": [{**worlds[w], "row_count": counts[w]} for w in sorted(counts)],
        }
    manifest = {
        "schema": "ssdm.l1_world_calibration_split.v1",
        "split_seed": SPLIT_SEED, "split_method": SPLIT_METHOD,
        "assignment_inputs": ["fixed split seed", "world IDs only"],
        "source_commit": SOURCE_COMMIT, "source_split": split,
        "source_training_export_sha256": export_hash,
        "source_world_count": 32, "source_task_count": 1280, "source_row_count": 4096,
        "source_cohort_hashes": L1_TRAINING_ALLOWLIST[split],
        "source_manifest": source, **L1_GENERATOR_SOURCE,
        "partitions": summaries,
        "evaluation_only_cohort": {
            "split": MULTIWORLD_VALIDATION_SPLIT, "world_count": 32,
            "sample_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][0],
            "manifest_sha256": _PINS[MULTIWORLD_VALIDATION_SPLIT][1],
            "used_in_split": False,
        },
        "scope": __doc__.strip(), "cost_usd": 0,
        "model_execution": False, "training_performed": False,
        "calibration_performed": False,
        "limitations": "Only 8 synthetic calibration worlds; no calibrated-performance claim. "
                       "No model/weights, GPU, paid compute, Index or heldout access.",
        "consumer_contract": "Honor the two files as separate partitions before Item conversion; "
                             "do not recombine and randomly resplit tasks or slots. "
                             "Calibration is development data, not final evaluation. "
                             "Use only the separate validation cohort for development evaluation.",
    }
    return partitions, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    partitions, manifest = split_multirecord_training()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Stage the entire three-file bundle beside its destination; one rename
    # publishes it only after every write succeeds, without partial final files.
    with tempfile.TemporaryDirectory(prefix=f".{args.out.name}-", dir=args.out.parent) as directory:
        staged = Path(directory)
        for name, rows in partitions.items():
            (staged / f"{name}.jsonl").write_bytes(_rows_bytes(rows))
        (staged / "split-manifest.json").write_bytes((_canonical(manifest) + "\n").encode("ascii"))
        staged.rename(args.out)


if __name__ == "__main__":
    main()

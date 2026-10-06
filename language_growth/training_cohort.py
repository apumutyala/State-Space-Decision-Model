"""Reproducible, synthetic development_train slots; CPU serialization only.

Run from the repository root with ``python -m language_growth.training_cohort``.
There is no input-data path: every world comes from development_data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .development_data import development_manifest, generate_development_tasks
from .parent_bridge import decision_request
from .training_records import export_training_records

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "development_train"
ENCODING = "sorted compact ensure_ascii JSON per LF line, final LF"
SCOPE = ("synthetic development_train semantic-choice supervision with earlier gold slots "
         "as teacher-forced context; no validation, audit, Index or held-out inputs; "
         "no model load, training, capability or free-running qualification")


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_records(tasks: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the entire request boundary and parse actual D rows without a model."""
    # The exporter checks exact permitted state and earlier target prefix. A
    # full bridge reconstruction also rejects labels/programs in other fields.
    if records != export_training_records(tasks, split=SPLIT):
        raise ValueError("records differ from development_train exporter")
    offset = 0
    for task in tasks:
        training = task["training_only"]
        for request in task["model_inputs"]:
            if set(request["questions"]) != set(records[offset]["labels"]):
                raise ValueError("question/label mismatch")
            qid, question = next(iter(request["questions"].items()))
            if set(question) != {"type", "instructions", "criteria"}:
                raise ValueError("unexpected question fields: possible supervision leakage")
            state = json.loads(request["state"])
            expected, _ = decision_request(
                task_input=state["task_input"], host=training["host"], catalog=training["catalog"],
                model="development-only", parent_revision="synthetic-no-model:v1",
                request_id=request["ssdm"]["request_id"], question_id=qid,
                question=question["instructions"], options=question["criteria"],
                proposed_slot_choices=state["proposed_slot_choices"],
            )
            if request != expected:
                raise ValueError("unexpected request fields: possible supervision leakage")
            offset += 1

    from heads.layout import ByteEncoder
    from train.data import items_from_records
    from train.guard import SplitGuard

    guard = SplitGuard.synthetic_only("fresh language_growth development_train generator only")
    items = items_from_records(records, ByteEncoder(), guard=guard)
    if len(items) != len(records) or len({item.id for item in items}) != len(items):
        raise ValueError("shared D adapter slot identity/count mismatch")
    for record, item in zip(records, items):
        qid, gold = next(iter(record["labels"].items()))
        if (item.id != f"{record['id']}/{qid}" or item.row.qtype != "choice"
                or item.row.option_ids[item.label] != gold):
            raise ValueError("shared D adapter label mismatch")
    return {"adapter": "train.data.items_from_records", "encoder": ByteEncoder().identity(),
            "layout": "ordered", "template": "jrt-plain-v1", "item_count": len(items),
            "labels_verified": True, "request_boundary_verified": True,
            "model_loaded": False, "guard": guard.identity()}


def _source() -> dict[str, Any]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                    cwd=ROOT, text=True).strip()
    paths = ("language_growth/training_cohort.py", "language_growth/development_data.py",
             "language_growth/training_records.py", "language_growth/parent_bridge.py",
             "language_growth/kernel.py", "train/data.py", "train/guard.py", "train/cohort.py",
             "heads/layout.py", "decide/prompt.py", "decide/schema.py", "decide/provenance.py")
    return {"sha": sha, "worktree_dirty": bool(dirty),
            "files_sha256": {path: _sha((ROOT / path).read_bytes()) for path in paths}}


def _empty_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty; refusing overwrite")


def write_training_cohort(output: str | Path, *, seed: int, count: int,
                          choice_format: str = "mixed", split: str = SPLIT,
                          command: list[str] | None = None) -> dict[str, Any]:
    """Generate, verify, then write records, a development manifest and receipt."""
    if split != SPLIT:
        raise ValueError("training cohort requires development_train")
    output = Path(output).resolve()
    _empty_output(output)
    tasks = generate_development_tasks(seed, count, choice_format=choice_format, split=SPLIT)
    records = export_training_records(tasks, split=SPLIT)
    verification = verify_records(tasks, records)
    manifest = development_manifest(tasks)
    record_lines = [_json(record) for record in records]
    record_bytes = b"".join(record_lines)
    manifest_bytes = _json(manifest)
    canonical_command = [sys.executable, "-m", "language_growth.training_cohort", "--output",
                         str(output), "--seed", str(seed), "--count", str(count),
                         "--choice-format", choice_format, "--split", SPLIT]
    receipt = {
        "schema": "ssdm.track_l_training_cohort.v1", "source": _source(),
        "seed": seed, "count": count, "split": SPLIT, "choice_format": choice_format,
        "allow_one_repair": False, "command": command if command is not None else canonical_command,
        "command_cwd": str(ROOT), "scope": SCOPE,
        "hardware": {"device": "cpu", "machine": platform.machine(),
                     "platform": platform.platform(), "python": platform.python_version(),
                     "gpu_used": False},
        "cost": {"paid_execution": False, "incremental_paid_usd": 0},
        "counts": {"tasks": len(tasks), "records": len(records), "slots": len(records),
                   "by_kind_tasks": dict(sorted(Counter(t["training_only"]["kind"] for t in tasks).items())),
                   "by_kind_slots": dict(sorted(Counter(t["training_only"]["kind"]
                                                  for t in tasks for _ in t["model_inputs"]).items())),
                   "by_slot": dict(sorted(Counter(target["slot"] for t in tasks
                                                   for target in t["training_only"]["targets"]).items())),
                   "by_choice_format_tasks": manifest["choice_formats"]},
        "hashes": {"encoding": ENCODING, "tasks_sha256": manifest["sample_sha256"],
                   "training_records_sha256": _sha(record_bytes),
                   "development_manifest_sha256": _sha(manifest_bytes),
                   "tasks": [{"id": t["task_id"], "sha256": _sha(_json(t))} for t in tasks],
                   "records": [{"item_id": f"{r['id']}/{next(iter(r['labels']))}", "sha256": _sha(line)}
                               for r, line in zip(records, record_lines)]},
        "artifacts": {"training_records": "training_records.jsonl",
                      "development_manifest": "development_manifest.json", "receipt": "receipt.json"},
        "verification": verification,
    }
    _empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in (("training_records.jsonl", record_bytes),
                      ("development_manifest.json", manifest_bytes), ("receipt.json", _json(receipt))):
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--choice-format", choices=("production", "augmented", "mixed"), default="mixed")
    parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    args = parser.parse_args(argv)
    invocation = [sys.executable, "-m", "language_growth.training_cohort", *(sys.argv[1:] if argv is None else argv)]
    try:
        receipt = write_training_cohort(args.output, seed=args.seed, count=args.count,
                                       choice_format=args.choice_format, split=args.split, command=invocation)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output.resolve()), "counts": receipt["counts"],
                      "records_sha256": receipt["hashes"]["training_records_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

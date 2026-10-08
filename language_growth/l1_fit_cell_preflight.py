"""Model-free, fit-only preflight for the first L1 GatePlanner learning cell.

This module reads only the pinned fit export and its exact tokenizer census.
It does not import the model stack, inspect other splits, or authorize training.
The generated plan reserves whole fit worlds for diagnostic evaluation; those
rows remain development_train and are not promotion or qualification evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

FIT_PIN = {
    "training_records.jsonl": "02345dfb60c92742e137d96aa9e34576d04a77181dfc4aa0ac5d308730cb4ccd",
    "training_records.jsonl.gz": "d767627b781596f5f1e2882f2ed2e0b69587c055b3522117d165332af2d859ef",
    "manifest.json": "0c08fd0da13a175f6a66107e728ee27eff9a69c6d7ff115a072be18938f694c5",
}
CENSUS_PIN = {
    "summary.json": "a467706aca9b2365ecb3044ffd58a844d6ad35d713719e2165d03f5864ceeb13",
    "row_counts.jsonl": "bd1d192f20888ea31339ed022f21cae220d3a3f82fb928c242a7a0622c8bd3fa",
}
SLOT_COUNTS = {
    "semantic-operation": 960,
    "fact": 384,
    "left-operand": 576,
    "right-operand": 576,
    "comparison-relation": 576,
}
TOKENIZER_REVISION = "ce3630b854cedbd673569f4600c6cf744bca9dba"
ROW_TEMPLATE = "jrt-rwkv-g1-v1"
LAYOUT = "ordered"
MAX_TOKENS = 2048
MAX_OPTIONS = 8


class PreflightError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(directory: Path, pins: dict[str, str]) -> dict[str, bytes]:
    contents = {}
    for name, expected in pins.items():
        path = directory / name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PreflightError(f"missing pinned input: {path}") from exc
        actual = _sha(raw)
        if actual != expected:
            raise PreflightError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        contents[name] = raw
    return contents


def _records(raw: bytes) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in raw.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("fit JSONL is not valid UTF-8 JSONL") from exc


def _split_worlds(world_ids: list[str], diagnostic_worlds: int) -> tuple[list[str], list[str]]:
    if type(diagnostic_worlds) is not int or not 1 <= diagnostic_worlds < len(world_ids):
        raise PreflightError("diagnostic_worlds must leave at least one fit world for training")
    ranked = sorted(world_ids, key=lambda world: hashlib.sha256(
        f"l1-fit-diagnostic-v1:{world}".encode("utf-8")).hexdigest())
    diagnostic = sorted(ranked[:diagnostic_worlds])
    training = sorted(set(world_ids) - set(diagnostic))
    if set(training) & set(diagnostic) or set(training) | set(diagnostic) != set(world_ids):
        raise PreflightError("world split is not a disjoint partition")
    return training, diagnostic


def build_plan(records: list[dict[str, Any]], token_rows: list[dict[str, Any]], *,
               diagnostic_worlds: int = 6) -> dict[str, Any]:
    """Validate fit-only rows against census identities and return a non-runnable plan."""
    if len(records) != 3072 or len(token_rows) != len(records):
        raise PreflightError("expected exactly 3,072 fit rows and one census row per fit row")
    slot_counts: dict[str, int] = {}
    per_world_slots: dict[str, dict[str, int]] = {}
    per_world_tasks: dict[str, set[str]] = {}
    identities = set()
    world_ids = set()
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise PreflightError(f"fit row {index} is not a record with metadata")
        meta = row.get("metadata", {})
        slot = meta.get("slot")
        world = meta.get("world_id")
        identity = (row.get("id"), slot)
        if (not isinstance(row.get("id"), str) or not row["id"] or not isinstance(slot, str)
                or not isinstance(world, str) or not world or slot not in SLOT_COUNTS
                or identity in identities or row.get("group_id") != world
                or meta.get("split") != "development_train"
                or meta.get("source_cohort") != "l1_multiworld_development_train"
                or meta.get("diagnostic_teacher_forced") is not True or meta.get("free_running") is not False
                or meta.get("promotion_eligible") is not False):
            raise PreflightError(f"fit row {index} has invalid identity/provenance")
        request = row.get("request")
        questions = request.get("questions") if isinstance(request, dict) else None
        if not isinstance(questions, dict) or len(questions) != 1:
            raise PreflightError(f"fit row {index} must contain exactly one question")
        qid, question = next(iter(questions.items()))
        if not isinstance(question, dict) or not isinstance(question.get("criteria"), dict):
            raise PreflightError(f"fit row {index} has malformed question criteria")
        options = question.get("criteria", {})
        labels = row.get("labels")
        if not isinstance(labels, dict) or len(labels) != 1 or qid not in labels or labels[qid] not in options:
            raise PreflightError(f"fit row {index} has an invalid separated option label")
        identities.add(identity)
        world_ids.add(world)
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
        world_slots = per_world_slots.setdefault(world, {})
        world_slots[slot] = world_slots.get(slot, 0) + 1
        per_world_tasks.setdefault(world, set()).add(row["id"])

    if slot_counts != SLOT_COUNTS:
        raise PreflightError(f"unexpected GatePlanner slot distribution: {slot_counts}")
    if len(world_ids) != 24:
        raise PreflightError(f"expected 24 fit worlds, got {len(world_ids)}")
    expected_world_slots = {"semantic-operation": 40, "fact": 16, "left-operand": 24,
                            "right-operand": 24, "comparison-relation": 24}
    if any(per_world_slots.get(world) != expected_world_slots or len(per_world_tasks.get(world, ())) != 40
           for world in world_ids):
        raise PreflightError("each fit world must contain 40 complete tasks and the exact 128-slot distribution")

    census_identities = set()
    for index, census_row in enumerate(token_rows, 1):
        if not isinstance(census_row, dict):
            raise PreflightError(f"token census row {index} is not an object")
        if type(census_row.get("record_line")) is not int or census_row["record_line"] != index:
            raise PreflightError("token census order does not match fit JSONL line order")
        record = records[index - 1]
        census_slot = census_row.get("slot", "")
        if not isinstance(census_slot, str):
            raise PreflightError(f"token census row {index} has malformed slot")
        slot = "-".join(census_slot.split("-")[2:]) if census_slot.startswith("slot-") else ""
        identity = (census_row.get("record_id"), slot)
        if identity != (record.get("id"), record.get("metadata", {}).get("slot")) or identity in census_identities:
            raise PreflightError(f"token census identity mismatch at row {index}")
        if (type(census_row.get("prompt_tokens")) is not int or not 1 <= census_row["prompt_tokens"] <= MAX_TOKENS
                or type(census_row.get("options")) is not int or not 1 <= census_row["options"] <= MAX_OPTIONS
                or type(census_row.get("longest_prompt_plus_continuation_tokens")) is not int
                or census_row["longest_prompt_plus_continuation_tokens"] < census_row["prompt_tokens"]
                or census_row["longest_prompt_plus_continuation_tokens"] > MAX_TOKENS
                or type(census_row.get("max_continuation_tokens")) is not int
                or not 1 <= census_row["max_continuation_tokens"] <= 16):
            raise PreflightError(f"token census row {index} exceeds the planned T/K/continuation limits")
        qid, question = next(iter(record["request"]["questions"].items()))
        if len(question["criteria"]) != census_row["options"]:
            raise PreflightError(f"option count differs from tokenizer census at row {index}")
        census_identities.add(identity)

    world_ids_sorted = sorted(world_ids)
    train_worlds, diagnostic = _split_worlds(world_ids_sorted, diagnostic_worlds)
    train_world_set = set(train_worlds)
    train_rows = sum(r["group_id"] in train_world_set for r in records)
    diagnostic_rows = len(records) - train_rows
    return {
        "schema": "ssdm.agent_c.l1_fit_cell_preflight.v1",
        "status": "preflight_only_not_authorized_to_train",
        "source": {
            "cohort": "pinned L1 development_train fit export v3",
            "records": len(records), "worlds": len(world_ids_sorted),
            "slot_counts": dict(sorted(slot_counts.items())),
            "split": "whole-world fit-only diagnostic; never qualification or promotion evidence",
            "training_worlds": train_worlds, "diagnostic_worlds": diagnostic,
            "training_rows": train_rows, "diagnostic_rows": diagnostic_rows,
        },
        "sequence": {"tokenizer_revision": TOKENIZER_REVISION, "template": ROW_TEMPLATE,
                     "layout": LAYOUT, "max_tokens": MAX_TOKENS, "max_options": MAX_OPTIONS},
        "parent": {"line": "RWKV7-G1k-13.3B", "checkpoint_sha256": None,
                   "stage1_report_sha256": None, "qualified": False},
        "training_authorized": False,
        "blockers": ["qualified RWKV7 parent and reviewed Stage 1 report are not pinned",
                     "A-ordered language run and suitable isolated resources are not present"],
        "limitations": ["fit-only rows are diagnostic teacher-forced supervision",
                        "a fit-world diagnostic split does not replace development validation or language-heldout gates",
                        "no model, Torch, validation, calibration, heldout, or GPU data was accessed"],
    }


def preflight(fit_dir: Path, census_dir: Path, *, diagnostic_worlds: int = 6) -> dict[str, Any]:
    fit = _read_pinned(Path(fit_dir), FIT_PIN)
    census = _read_pinned(Path(census_dir), CENSUS_PIN)
    try:
        compressed = gzip.decompress(fit["training_records.jsonl.gz"])
        manifest = json.loads(fit["manifest.json"])
        summary = json.loads(census["summary.json"])
        census_rows = [json.loads(line) for line in census["row_counts.jsonl"].splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("a pinned input is malformed") from exc
    if compressed != fit["training_records.jsonl"]:
        raise PreflightError("pinned compressed and plain fit exports differ")
    if (manifest.get("training_records_sha256") != _sha(fit["training_records.jsonl"])
            or manifest.get("training_records_gzip_sha256") != _sha(fit["training_records.jsonl.gz"])
            or manifest.get("split") != "development_train" or manifest.get("records") != 3072
            or summary.get("tokenizer_revision") != TOKENIZER_REVISION
            or summary.get("template") != ROW_TEMPLATE or summary.get("layout") != LAYOUT
            or summary.get("model_loaded") is not False or summary.get("torch_imported") is not False
            or summary.get("aggregate", {}).get("rows") != 3072
            or summary.get("aggregate", {}).get("prompt_tokens", {}).get("max", MAX_TOKENS + 1) > MAX_TOKENS
            or summary.get("aggregate", {}).get("max_options") != MAX_OPTIONS
            or summary.get("aggregate", {}).get("longest_prompt_plus_continuation_tokens", {}).get(
                "max", MAX_TOKENS + 1) > MAX_TOKENS):
        raise PreflightError("manifest or token census aggregate does not match the planned cell")
    plan = build_plan(_records(fit["training_records.jsonl"]), census_rows,
                      diagnostic_worlds=diagnostic_worlds)
    plan["pins"] = {**{name: _sha(raw) for name, raw in fit.items()},
                    **{name: _sha(raw) for name, raw in census.items()},
                    "l1_fit_cell_preflight.py": _sha(Path(__file__).read_bytes())}
    return plan


def write_new_plan(path: Path, plan: dict[str, Any]) -> None:
    """Create the output exactly once, rejecting existing/dangling links and races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise PreflightError(f"refusing to overwrite existing plan: {path}")
    content = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    except FileExistsError as exc:
        raise PreflightError(f"refusing to overwrite existing plan: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-worlds", type=int, default=6)
    args = parser.parse_args(argv)
    plan = preflight(args.fit_dir, args.census_dir, diagnostic_worlds=args.diagnostic_worlds)
    write_new_plan(args.out, plan)
    print(json.dumps({"out": str(args.out), "status": plan["status"],
                      "training_rows": plan["source"]["training_rows"],
                      "diagnostic_rows": plan["source"]["diagnostic_rows"],
                      "training_authorized": plan["training_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

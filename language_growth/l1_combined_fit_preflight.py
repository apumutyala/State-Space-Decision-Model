"""Combine pinned multirecord fit data with a train-only ExactDifference slice.

This is a dependency-free data/census preflight. It reads only the pinned
development_train artifacts, regenerates the synthetic L1 train supplement
from its pinned source code, and reserves whole worlds in each source for
diagnostics. It never authorizes or starts model training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .gate_suites import canonical, digest
from .gate_training import generate_training_records
from .l1_fit_cell_preflight import (
    CENSUS_PIN as BASE_CENSUS_PIN,
    FIT_PIN as BASE_FIT_PIN,
    MAX_OPTIONS,
    MAX_TOKENS,
    ROW_TEMPLATE,
    TOKENIZER_REVISION,
    _records,
    _sha,
    _split_worlds,
    preflight as preflight_base,
    write_new_plan,
)

BASE_PLAN_SHA256 = "29066e2e137c5374d401ea1c9b47c165995969e8a951f930057e51dbda1a02f0"
SUPPLEMENT_PIN = {
    "training_records.jsonl": "dd22bc825d964654b896036d13b8de78b658ebb313867a4802f4f016a752688d",
    "training_records.jsonl.gz": "5a875060c816b52caa1e3a212fa6fb035c3981f3c50976cc974b12f13561861c",
    "manifest.json": "db6ee72c14173eca8aa62d397e90dda23f971f99d02d6276c1b49d68f98a2acd",
}
SUPPLEMENT_CENSUS_PIN = {
    "sha256.json": "78e50da15951efbbe6334f393ae0cf9b87d6f56c384caa3c53dc1a118022b7d2",
    "summary.json": "89334b73a598d285a2a5b64e5f5cae477df5442129afcd43845bbd3ff6581500",
    "row_counts.jsonl": "21ae770c1d75916d06048f2be88abce5de8b023924136cdaec5afc5d159bc893",
    "census.py": "e02454315aa187e0f46deba5b1dbf649049e882a7aef57ccce72ef480e405474",
}
SUPPLEMENT_SEED = 20261009
SUPPLEMENT_WORLDS = 24
SUPPLEMENT_MODEL = "gate-training-l1-supplement"
SUPPLEMENT_PARENT_REVISION = "training-only-l1-difference-v1"
EXPECTED_SUPPLEMENT_SLOTS = {
    "semantic-operation": 72,
    "fact": 24,
    "left-operand": 48,
    "right-operand": 48,
    "comparison-relation": 24,
}


class CombinedPreflightError(ValueError):
    pass


def _read_pinned(directory: Path, pins: dict[str, str]) -> dict[str, bytes]:
    contents = {}
    for name, expected in pins.items():
        try:
            raw = (Path(directory) / name).read_bytes()
        except OSError as exc:
            raise CombinedPreflightError(f"missing pinned input: {name}") from exc
        actual = _sha(raw)
        if actual != expected:
            raise CombinedPreflightError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        contents[name] = raw
    return contents


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CombinedPreflightError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _target_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        slot = record["metadata"]["slot"]
        qid, question = next(iter(record["request"]["questions"].items()))
        target = question["criteria"][record["labels"][qid]]
        slot_counts = counts.setdefault(slot, {})
        slot_counts[target] = slot_counts.get(target, 0) + 1
    return {slot: dict(sorted(values.items())) for slot, values in sorted(counts.items())}


def _verify_world_partition(records: list[dict], training: set[str], diagnostic: set[str], source: str) -> None:
    if training & diagnostic:
        raise CombinedPreflightError(f"{source} training and diagnostic worlds overlap")
    try:
        actual = {record["group_id"] for record in records}
    except (KeyError, TypeError) as exc:
        raise CombinedPreflightError(f"{source} record is missing a world identity") from exc
    if not actual or not all(isinstance(world, str) and world for world in actual):
        raise CombinedPreflightError(f"{source} record has an invalid world identity")
    if training | diagnostic != actual:
        raise CombinedPreflightError(f"{source} plan does not partition the actual record worlds")


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CombinedPreflightError("pinned JSON input is malformed") from exc


def _supplement(corpus_dir: Path, census_dir: Path, *, diagnostic_worlds: int) -> tuple[dict, list]:
    source = _read_pinned(corpus_dir, SUPPLEMENT_PIN)
    census = _read_pinned(census_dir, SUPPLEMENT_CENSUS_PIN)
    try:
        plain = source["training_records.jsonl"]
        compressed = gzip.decompress(source["training_records.jsonl.gz"])
        manifest = _json(source["manifest.json"])
        summary = _json(census["summary.json"])
        census_hashes = _json(census["sha256.json"])
        token_rows = [_json(line) for line in census["row_counts.jsonl"].splitlines() if line]
    except (OSError, EOFError) as exc:
        raise CombinedPreflightError("supplement gzip artifact is malformed") from exc
    if compressed != plain:
        raise CombinedPreflightError("supplement compressed and plain records differ")
    if (census_hashes != {name: _sha(census[name]) for name in ("census.py", "row_counts.jsonl", "summary.json")}
            or summary.get("row_counts_sha256") != _sha(census["row_counts.jsonl"])
            or summary.get("fit_sha256") != _sha(source["training_records.jsonl.gz"])
            or summary.get("tokenizer_revision") != TOKENIZER_REVISION
            or summary.get("template") != ROW_TEMPLATE or summary.get("layout") != "ordered"
            or summary.get("model_loaded") is not False or summary.get("torch_imported") is not False
            or summary.get("aggregate", {}).get("rows") != 216
            or summary.get("aggregate", {}).get("max_options") != MAX_OPTIONS
            or summary.get("aggregate", {}).get("prompt_tokens", {}).get("max", MAX_TOKENS + 1) > MAX_TOKENS
            or summary.get("aggregate", {}).get("longest_prompt_plus_continuation_tokens", {}).get(
                "max", MAX_TOKENS + 1) > MAX_TOKENS):
        raise CombinedPreflightError("supplement token census pins or aggregate limits mismatch")
    if (manifest.get("schema") != "ssdm.gate_training_export.v1"
            or manifest.get("source") != "language_growth.gate_suites.build_suite/development_train"
            or manifest.get("levels") != ["L1"] or manifest.get("split") != "development_train"
            or manifest.get("seed") != SUPPLEMENT_SEED or manifest.get("requested_worlds") != SUPPLEMENT_WORLDS
            or manifest.get("tasks") != 72 or manifest.get("worlds") != SUPPLEMENT_WORLDS
            or manifest.get("records") != 216 or manifest.get("slot_counts") != EXPECTED_SUPPLEMENT_SLOTS
            or manifest.get("validation_or_heldout_generated") is not False
            or manifest.get("free_running") is not False or manifest.get("promotion_eligible") is not False
            or manifest.get("training_records_sha256") != _sha(plain)
            or manifest.get("training_records_gzip_sha256") != _sha(source["training_records.jsonl.gz"])):
        raise CombinedPreflightError("supplement manifest does not match the registered L1 train slice")
    source_dir = Path(__file__).resolve().parent
    for name, expected in manifest.get("source_sha256", {}).items():
        actual = _sha((source_dir / name).read_bytes())
        if actual != expected:
            raise CombinedPreflightError(f"supplement source pin changed: {name}")

    generated = generate_training_records(levels=("L1",), split="development_train",
                                           worlds=SUPPLEMENT_WORLDS, seed=SUPPLEMENT_SEED,
                                           model=SUPPLEMENT_MODEL, parent_revision=SUPPLEMENT_PARENT_REVISION)
    regenerated = b"".join(canonical(row) + b"\n" for row in generated)
    if regenerated != plain:
        raise CombinedPreflightError("supplement records do not exactly replay from the pinned L1 train generator")
    records = _records(plain)
    target_counts = _target_counts(records)
    worlds = {row.get("group_id") for row in records}
    tasks = {row.get("id") for row in records}
    if len(worlds) != SUPPLEMENT_WORLDS or len(tasks) != 72 or len(token_rows) != len(records):
        raise CombinedPreflightError("supplement task/world/token census count mismatch")

    per_world_slots: dict[str, dict[str, int]] = {}
    per_world_tasks: dict[str, set[str]] = {}
    kinds = {"ReadFact": 0, "Compare": 0, "ExactDifference": 0}
    for index, (record, token) in enumerate(zip(records, token_rows, strict=True), 1):
        meta = record.get("metadata", {})
        world, task_id, slot = meta.get("world_id"), record.get("id"), meta.get("slot")
        if (meta.get("split") != "development_train" or meta.get("free_running") is not False
                or meta.get("promotion_eligible") is not False or record.get("group_id") != world
                or not isinstance(world, str) or not isinstance(task_id, str)):
            raise CombinedPreflightError(f"supplement row {index} has invalid train-only provenance")
        if token.get("record_line") != index or token.get("record_id") != task_id:
            raise CombinedPreflightError(f"supplement census row identity/order mismatch at {index}")
        slot_name = token.get("slot")
        slot_from_census = "-".join(slot_name.split("-")[2:]) if isinstance(slot_name, str) and slot_name.startswith("slot-") else ""
        if slot_from_census != slot or type(token.get("options")) is not int or not 1 <= token["options"] <= MAX_OPTIONS:
            raise CombinedPreflightError(f"supplement slot/options census mismatch at {index}")
        prompt = token.get("prompt_tokens")
        longest = token.get("longest_prompt_plus_continuation_tokens")
        continuation = token.get("max_continuation_tokens")
        if (type(prompt) is not int or not 1 <= prompt <= MAX_TOKENS or type(longest) is not int
                or not prompt <= longest <= MAX_TOKENS or type(continuation) is not int
                or not 1 <= continuation <= 16):
            raise CombinedPreflightError(f"supplement token cap exceeded at row {index}")
        qid, question = next(iter(record["request"]["questions"].items()))
        if len(question["criteria"]) != token["options"]:
            raise CombinedPreflightError(f"supplement option census mismatch at row {index}")
        if slot == "semantic-operation":
            target = question["criteria"][record["labels"][qid]]
            if target not in kinds:
                raise CombinedPreflightError(f"unexpected supplement operation target: {target}")
            kinds[target] += 1
        by_world = per_world_slots.setdefault(world, {})
        by_world[slot] = by_world.get(slot, 0) + 1
        per_world_tasks.setdefault(world, set()).add(task_id)
    expected_world_slots = {"semantic-operation": 3, "fact": 1, "left-operand": 2,
                            "right-operand": 2, "comparison-relation": 1}
    if (any(per_world_slots.get(world) != expected_world_slots or len(per_world_tasks.get(world, ())) != 3
            for world in worlds)
            or kinds != {"ReadFact": 24, "Compare": 24, "ExactDifference": 24}
            or target_counts.get("semantic-operation") != {
                "Compare": 24, "ExactDifference": 24, "ReadFact": 24}):
        raise CombinedPreflightError("supplement must have one checked read, compare and exact-difference task per world")

    train_worlds, diagnostic = _split_worlds(sorted(worlds), diagnostic_worlds)
    train_world_set = set(train_worlds)
    train_rows = sum(record["group_id"] in train_world_set for record in records)
    plan = {
        "schema": "ssdm.agent_c.l1_exactdifference_supplement_preflight.v1",
        "source": {"records": len(records), "tasks": len(tasks), "worlds": len(worlds),
                   "slot_counts": EXPECTED_SUPPLEMENT_SLOTS,
                   "operation_counts": kinds, "target_counts_by_slot": target_counts,
                   "training_worlds": train_worlds,
                   "diagnostic_worlds": diagnostic, "training_rows": train_rows,
                   "diagnostic_rows": len(records) - train_rows},
        "token_limits": {"tokenizer_revision": TOKENIZER_REVISION, "template": ROW_TEMPLATE,
                         "layout": "ordered", "max_tokens": MAX_TOKENS, "max_options": MAX_OPTIONS,
                         "prompt_tokens_max": summary["aggregate"]["prompt_tokens"]["max"],
                         "prompt_plus_continuation_max": summary["aggregate"]["longest_prompt_plus_continuation_tokens"]["max"]},
        "pins": {**{name: _sha(raw) for name, raw in source.items()},
                 **{name: _sha(raw) for name, raw in census.items()}},
    }
    return plan, records


def combine(base_plan: dict, supplement_plan: dict, base_records: list[dict],
            supplement_records: list[dict], *, base_summary: dict) -> dict:
    base_source = base_plan["source"]
    supplement_source = supplement_plan["source"]
    base_train, base_diag = set(base_source["training_worlds"]), set(base_source["diagnostic_worlds"])
    supplement_train, supplement_diag = (set(supplement_source["training_worlds"]),
                                         set(supplement_source["diagnostic_worlds"]))
    base_worlds = base_train | base_diag
    supplement_worlds = supplement_train | supplement_diag
    _verify_world_partition(base_records, base_train, base_diag, "base")
    _verify_world_partition(supplement_records, supplement_train, supplement_diag, "supplement")
    if base_worlds & supplement_worlds:
        raise CombinedPreflightError("base/supplement world split is not a disjoint source-preserving partition")
    base_ids = {r["id"] for r in base_records}
    supplement_ids = {r["id"] for r in supplement_records}
    if base_ids & supplement_ids:
        raise CombinedPreflightError("base and supplemental task IDs overlap")
    base_target_counts = _target_counts(base_records)
    supplement_target_counts = _target_counts(supplement_records)
    if (base_target_counts.get("semantic-operation") != {
            "Compare": 576, "ExplainDerivation": 192, "ReadFact": 192}
            or base_target_counts.get("comparison-relation") != {
                "equal": 96, "greater_than": 240, "less_than": 240}
            or supplement_target_counts.get("semantic-operation") != {
                "Compare": 24, "ExactDifference": 24, "ReadFact": 24}):
        raise CombinedPreflightError("base or supplemental operation targets differ from their pinned fit profiles")
    merged_target_counts = {}
    for counts in (base_target_counts, supplement_target_counts):
        for slot, values in counts.items():
            merged = merged_target_counts.setdefault(slot, {})
            for target, count in values.items():
                merged[target] = merged.get(target, 0) + count
    merged_target_counts = {slot: dict(sorted(values.items()))
                            for slot, values in sorted(merged_target_counts.items())}
    base_train_rows = base_source["training_rows"]
    base_diag_rows = base_source["diagnostic_rows"]
    supplement_train_rows = supplement_source["training_rows"]
    supplement_diag_rows = supplement_source["diagnostic_rows"]
    return {
        "schema": "ssdm.agent_c.l1_combined_fit_preflight.v1",
        "status": "preflight_only_not_authorized_to_train",
        "sources": {"pinned_multirecord_fit": base_plan, "generated_l1_exactdifference": supplement_plan},
        "source_counts": {"records": len(base_records) + len(supplement_records),
                          "tasks": len(base_ids) + len(supplement_ids),
                          "worlds": len(base_worlds) + len(supplement_worlds),
                          "slot_counts": {slot: base_plan["source"]["slot_counts"].get(slot, 0)
                                          + supplement_plan["source"]["slot_counts"].get(slot, 0)
                                          for slot in sorted(set(base_plan["source"]["slot_counts"])
                                                             | set(supplement_plan["source"]["slot_counts"]))},
                          "target_counts_by_slot": merged_target_counts,
                          "target_counts_sha256": digest(merged_target_counts)},
        "split": {"training_worlds": sorted(base_train | supplement_train),
                  "diagnostic_worlds": sorted(base_diag | supplement_diag),
                  "training_rows": base_train_rows + supplement_train_rows,
                  "diagnostic_rows": base_diag_rows + supplement_diag_rows,
                  "by_source": {"pinned_multirecord_fit": {"training_rows": base_train_rows,
                                                               "diagnostic_rows": base_diag_rows,
                                                               "training_worlds": sorted(base_train),
                                                               "diagnostic_worlds": sorted(base_diag)},
                                "generated_l1_exactdifference": {"training_rows": supplement_train_rows,
                                                                 "diagnostic_rows": supplement_diag_rows,
                                                                 "training_worlds": sorted(supplement_train),
                                                                 "diagnostic_worlds": sorted(supplement_diag)}},
                  "unit": "whole world; each source split independently with l1-fit-diagnostic-v1"},
        "token_limits": {"tokenizer_revision": TOKENIZER_REVISION, "template": ROW_TEMPLATE,
                         "layout": "ordered", "max_tokens": MAX_TOKENS, "max_options": MAX_OPTIONS,
                         "max_prompt_tokens": max(base_summary["aggregate"]["prompt_tokens"]["max"],
                                                  supplement_plan["token_limits"]["prompt_tokens_max"]),
                         "max_prompt_plus_continuation_tokens": max(
                             base_summary["aggregate"]["longest_prompt_plus_continuation_tokens"]["max"],
                             supplement_plan["token_limits"]["prompt_plus_continuation_max"])},
        "training_authorized": False,
        "parent": {"line": "RWKV7-G1k-13.3B", "checkpoint_sha256": None,
                   "stage1_report_sha256": None, "qualified": False},
        "blockers": ["qualified Stage 1 RWKV7 parent is not pinned",
                     "A-ordered language run is not present",
                     "reviewed hash-only DEV/TEST SplitGuard artifact/fingerprint is not pinned",
                     "suitable isolated resources are not present"],
        "limitations": ["all rows are diagnostic teacher-forced development_train supervision",
                        "the added ExactDifference slice is generated synthetic coverage, not independent evidence",
                        "this plan does not claim L1 usefulness, parent retention, joint soundness, or intervention gains"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fit-dir", type=Path, required=True)
    parser.add_argument("--base-census-dir", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--supplement-census-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    base_bytes = args.base_plan.read_bytes()
    if _sha(base_bytes) != BASE_PLAN_SHA256:
        raise SystemExit("base L1 preflight plan hash mismatch")
    base_plan = _json(base_bytes)
    replayed_base = preflight_base(args.base_fit_dir, args.base_census_dir, diagnostic_worlds=6)
    if base_plan != replayed_base:
        raise SystemExit("base L1 preflight does not match the pinned current inputs")
    base_data = _read_pinned(args.base_fit_dir, BASE_FIT_PIN)["training_records.jsonl"]
    base_records = _records(base_data)
    base_summary = _json(_read_pinned(args.base_census_dir, BASE_CENSUS_PIN)["summary.json"])
    supplement_plan, supplement_records = _supplement(
        args.supplement_dir, args.supplement_census_dir, diagnostic_worlds=6)
    combined = combine(base_plan, supplement_plan, base_records, supplement_records,
                       base_summary=base_summary)
    combined["pins"] = {"base_plan_sha256": BASE_PLAN_SHA256,
                        "base_fit": base_plan["pins"],
                        "supplement": supplement_plan["pins"],
                        "combined_preflight_source_sha256": _sha(Path(__file__).read_bytes())}
    write_new_plan(args.out, combined)
    print(json.dumps({"out": str(args.out), "status": combined["status"],
                      "records": combined["source_counts"]["records"],
                      "training_rows": combined["split"]["training_rows"],
                      "diagnostic_rows": combined["split"]["diagnostic_rows"],
                      "training_authorized": combined["training_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

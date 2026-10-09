"""Combine the pinned L1 cohort with balanced train-only ExactDifference rows."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .gate_suites import canonical, digest
from .l1_combined_fit_preflight import (
    BASE_CENSUS_PIN,
    BASE_FIT_PIN,
    BASE_PLAN_SHA256,
    CombinedPreflightError,
    _json,
    _read_pinned,
    _records,
    _sha,
    _split_worlds,
    _target_counts,
    _verify_world_partition,
    preflight_base,
    write_new_plan,
)
from .l1_difference_fit_supplement import (
    EXPECTED_SIGNS,
    FIELDS,
    SCHEMA as SUPPLEMENT_SCHEMA,
    SEED,
    SOURCE_FILES,
    WORLDS,
    generate_records,
)
from .l1_fit_cell_preflight import MAX_OPTIONS, MAX_TOKENS, ROW_TEMPLATE, TOKENIZER_REVISION

SUPPLEMENT_PIN = {
    "training_records.jsonl": "0cdac25e223b20bffa076bbc9c749455cbae4b23f367fea83eb7d05ee3e8b345",
    "training_records.jsonl.gz": "bee5faefa8a9551b68da4811744dbc1d70d9fd16f612fe8895af6d7af7a63fb1",
    "manifest.json": "407ba9f16c6768e7519cc38d9c85a8c221e786fe4f1a3b764b70d6a5bc6d93f9",
}
CENSUS_PIN = {
    "census.py": "ad84cb34be27a066b8e623d0530438bd07dddf3f60fe0ff2c5e68c51720a8089",
    "row_counts.jsonl": "a2147057af2c882298c77e232e9f57cffdedc94c4d205f93b187125f12c2ca27",
    "summary.json": "1e7dd3f22a7021a7911d4ad3745ade709898b2e21b9a629e6239f3807fe26406",
    "sha256.json": "c2e8d69aabf61e58212d0d0fc4b78e2189f264612dd547508e77ac991a4bc67a",
}


def _verify_census_source(summary: dict, base_summary: dict) -> None:
    if (summary.get("tokenizer_revision") != TOKENIZER_REVISION
            or summary.get("template") != ROW_TEMPLATE or summary.get("layout") != "ordered"
            or summary.get("model_loaded") is not False or summary.get("torch_imported") is not False
            or summary.get("cost_usd") != 0
            or summary.get("tokenizer_sha256") != base_summary.get("tokenizer_sha256")):
        raise CombinedPreflightError("difference census tokenizer/encoder identity differs from base census")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = summary.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise CombinedPreflightError("difference census is missing source-code pins")
    for relative, expected in source_hashes.items():
        try:
            actual = _sha((source_root / relative).read_bytes())
        except OSError as exc:
            raise CombinedPreflightError(f"difference census source file is missing: {relative}") from exc
        if actual != expected:
            raise CombinedPreflightError(f"difference census source pin changed: {relative}")


def _load_supplement(corpus_dir: Path, census_dir: Path, base_summary: dict) -> tuple[dict, list[dict]]:
    source = _read_pinned(corpus_dir, SUPPLEMENT_PIN)
    census = _read_pinned(census_dir, CENSUS_PIN)
    try:
        plain = source["training_records.jsonl"]
        compressed = gzip.decompress(source["training_records.jsonl.gz"])
        manifest = _json(source["manifest.json"])
        summary = _json(census["summary.json"])
        census_hashes = _json(census["sha256.json"])
        token_rows = [_json(line) for line in census["row_counts.jsonl"].splitlines() if line]
    except (OSError, EOFError) as exc:
        raise CombinedPreflightError("difference supplement or census JSON is malformed") from exc
    if compressed != plain:
        raise CombinedPreflightError("difference supplement gzip and plain records differ")
    if (census_hashes != {name: _sha(census[name]) for name in ("census.py", "row_counts.jsonl", "summary.json")}
            or summary.get("census_script_sha256") != _sha(census["census.py"])
            or summary.get("row_counts_sha256") != _sha(census["row_counts.jsonl"])
            or summary.get("fit_sha256") != _sha(source["training_records.jsonl.gz"])
            or summary.get("records") != 1728 or summary.get("question_rows") != 1728
            or summary.get("aggregate", {}).get("max_options") != MAX_OPTIONS
            or summary.get("aggregate", {}).get("prompt_tokens", {}).get("max", MAX_TOKENS + 1) > MAX_TOKENS
            or summary.get("aggregate", {}).get("longest_prompt_plus_continuation_tokens", {}).get(
                "max", MAX_TOKENS + 1) > MAX_TOKENS):
        raise CombinedPreflightError("difference census pins or aggregate limits mismatch")
    _verify_census_source(summary, base_summary)
    if (manifest.get("schema") != SUPPLEMENT_SCHEMA or manifest.get("level") != "L1"
            or manifest.get("split") != "development_train" or manifest.get("seed") != SEED
            or manifest.get("requested_worlds") != WORLDS or manifest.get("worlds") != WORLDS
            or manifest.get("tasks") != 576 or manifest.get("records") != 1728
            or manifest.get("difference_sign_counts_by_field") != EXPECTED_SIGNS
            or manifest.get("fields") != list(FIELDS)
            or manifest.get("slot_counts") != {"semantic-operation": 576,
                                                "left-operand": 576, "right-operand": 576}
            or manifest.get("target_counts_by_slot") != {
                "semantic-operation": {"v004": 576},
                "left-operand": {f"v{i:03d}": 72 for i in range(1, 9)},
                "right-operand": {f"v{i:03d}": 72 for i in range(1, 9)}}
            or manifest.get("validation_or_heldout_generated") is not False
            or manifest.get("free_running") is not False or manifest.get("promotion_eligible") is not False
            or manifest.get("training_records_sha256") != _sha(plain)
            or manifest.get("training_records_gzip_sha256") != _sha(source["training_records.jsonl.gz"])):
        raise CombinedPreflightError("difference manifest does not match the registered train-only profile")
    source_root = Path(__file__).resolve().parent
    if manifest.get("source_sha256") != {
            name: _sha((source_root / name).read_bytes()) for name in SOURCE_FILES}:
        raise CombinedPreflightError("difference supplement generator/source pin changed")

    rows, coverage = generate_records()
    replayed = b"".join(canonical(row) + b"\n" for row in rows)
    if replayed != plain:
        raise CombinedPreflightError("difference records do not replay from the pinned development_train generator")
    records = _records(plain)
    if len(token_rows) != len(records):
        raise CombinedPreflightError("difference census row count differs from training records")
    worlds = set(coverage["world_ids"])
    tasks: set[str] = set()
    for line, (record, token) in enumerate(zip(records, token_rows, strict=True), 1):
        metadata = record.get("metadata", {})
        task_id = record.get("id")
        slot = metadata.get("slot")
        if (metadata.get("split") != "development_train" or metadata.get("level") != "L1"
                or metadata.get("free_running") is not False or metadata.get("promotion_eligible") is not False
                or metadata.get("world_id") not in worlds or record.get("group_id") != metadata.get("world_id")
                or not isinstance(task_id, str)):
            raise CombinedPreflightError(f"difference record provenance mismatch at row {line}")
        tasks.add(task_id)
        if (token.get("record_line") != line or token.get("record_id") != task_id
                or token.get("slot") != next(iter(record["request"]["questions"]))
                or type(token.get("options")) is not int
                or token.get("options") != len(next(iter(record["request"]["questions"].values()))["criteria"])
                or not 1 <= token["options"] <= MAX_OPTIONS
                or type(token.get("prompt_tokens")) is not int or not 1 <= token["prompt_tokens"] <= MAX_TOKENS
                or type(token.get("longest_prompt_plus_continuation_tokens")) is not int
                or not token["prompt_tokens"] <= token["longest_prompt_plus_continuation_tokens"] <= MAX_TOKENS
                or type(token.get("max_continuation_tokens")) is not int
                or not 1 <= token["max_continuation_tokens"] <= 16):
            raise CombinedPreflightError(f"difference row/token census mismatch at row {line}")
    if len(tasks) != 576:
        raise CombinedPreflightError("difference task count mismatch")

    train_worlds, diagnostic_worlds = _split_worlds(sorted(worlds), 6)
    plan = {
        "schema": "ssdm.agent_c.l1_difference_fit_preflight.v1",
        "source": {"records": len(records), "tasks": len(tasks), "worlds": len(worlds),
                   "slot_counts": coverage["slot_counts"],
                   "operation_counts": {"ExactDifference": 576},
                   "target_counts_by_slot": _target_counts(records),
                   "difference_sign_counts_by_field": coverage["sign_counts"],
                   "training_worlds": train_worlds, "diagnostic_worlds": diagnostic_worlds,
                   "training_rows": sum(r["group_id"] in set(train_worlds) for r in records),
                   "diagnostic_rows": sum(r["group_id"] in set(diagnostic_worlds) for r in records)},
        "token_limits": {"tokenizer_revision": TOKENIZER_REVISION, "template": ROW_TEMPLATE,
                         "layout": "ordered", "max_tokens": MAX_TOKENS, "max_options": MAX_OPTIONS,
                         "prompt_tokens_max": summary["aggregate"]["prompt_tokens"]["max"],
                         "prompt_plus_continuation_max": summary["aggregate"][
                             "longest_prompt_plus_continuation_tokens"]["max"]},
        "pins": {**{name: _sha(raw) for name, raw in source.items()},
                 **{name: _sha(raw) for name, raw in census.items()}},
    }
    return plan, records


def _combine(base_plan: dict, supplement_plan: dict, base_records: list[dict],
             supplement_records: list[dict], base_summary: dict) -> dict:
    base = base_plan["source"]
    supplement = supplement_plan["source"]
    base_train, base_diag = set(base["training_worlds"]), set(base["diagnostic_worlds"])
    supplement_train = set(supplement["training_worlds"])
    supplement_diag = set(supplement["diagnostic_worlds"])
    _verify_world_partition(base_records, base_train, base_diag, "base")
    _verify_world_partition(supplement_records, supplement_train, supplement_diag, "difference supplement")
    base_worlds, supplement_worlds = base_train | base_diag, supplement_train | supplement_diag
    if base_worlds & supplement_worlds:
        raise CombinedPreflightError("base and difference supplement worlds overlap")
    base_ids = {row["id"] for row in base_records}
    supplement_ids = {row["id"] for row in supplement_records}
    if base_ids & supplement_ids:
        raise CombinedPreflightError("base and difference supplement task IDs overlap")

    base_targets = _target_counts(base_records)
    supplement_targets = _target_counts(supplement_records)
    if (base_targets.get("semantic-operation") != {
            "Compare": 576, "ExplainDerivation": 192, "ReadFact": 192}
            or supplement_targets.get("semantic-operation") != {"ExactDifference": 576}):
        raise CombinedPreflightError("semantic operation target frequencies differ from pinned profiles")
    merged_targets: dict[str, dict[str, int]] = {}
    for counts in (base_targets, supplement_targets):
        for slot, values in counts.items():
            destination = merged_targets.setdefault(slot, {})
            for target, count in values.items():
                destination[target] = destination.get(target, 0) + count
    merged_targets = {slot: dict(sorted(values.items())) for slot, values in sorted(merged_targets.items())}
    base_train_rows, base_diag_rows = base["training_rows"], base["diagnostic_rows"]
    supplement_train_rows = supplement["training_rows"]
    supplement_diag_rows = supplement["diagnostic_rows"]
    return {
        "schema": "ssdm.agent_c.l1_combined_difference_preflight.v1",
        "status": "preflight_only_not_authorized_to_train",
        "sources": {"pinned_multirecord_fit": base_plan, "balanced_exactdifference_fit": supplement_plan},
        "source_counts": {"records": len(base_records) + len(supplement_records),
                          "tasks": len(base_ids) + len(supplement_ids),
                          "worlds": len(base_worlds) + len(supplement_worlds),
                          "slot_counts": {slot: base_plan["source"]["slot_counts"].get(slot, 0)
                                          + supplement_plan["source"]["slot_counts"].get(slot, 0)
                                          for slot in sorted(set(base_plan["source"]["slot_counts"])
                                                             | set(supplement_plan["source"]["slot_counts"]))},
                          "target_counts_by_slot": merged_targets,
                          "target_counts_sha256": digest(merged_targets)},
        "split": {"training_worlds": sorted(base_train | supplement_train),
                  "diagnostic_worlds": sorted(base_diag | supplement_diag),
                  "training_rows": base_train_rows + supplement_train_rows,
                  "diagnostic_rows": base_diag_rows + supplement_diag_rows,
                  "by_source": {
                      "pinned_multirecord_fit": {"training_rows": base_train_rows,
                                                  "diagnostic_rows": base_diag_rows,
                                                  "training_worlds": sorted(base_train),
                                                  "diagnostic_worlds": sorted(base_diag)},
                      "balanced_exactdifference_fit": {"training_rows": supplement_train_rows,
                                                       "diagnostic_rows": supplement_diag_rows,
                                                       "training_worlds": sorted(supplement_train),
                                                       "diagnostic_worlds": sorted(supplement_diag)}},
                  "unit": "whole world; each source split independently with l1-fit-diagnostic-v1"},
        "semantic_operation_counts": merged_targets["semantic-operation"],
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
        "limitations": ["all rows are synthetic checker-valid development_train teacher-forced supervision",
                        "balanced ExactDifference examples are not independent evidence or learned arithmetic",
                        "the plan does not claim L1 usefulness, parent retention, joint soundness, or intervention gains"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fit-dir", type=Path, required=True)
    parser.add_argument("--base-census-dir", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    base_raw = args.base_plan.read_bytes()
    if _sha(base_raw) != BASE_PLAN_SHA256:
        raise SystemExit("base L1 fit plan hash mismatch")
    base_plan = _json(base_raw)
    replay = preflight_base(args.base_fit_dir, args.base_census_dir, diagnostic_worlds=6)
    if replay != base_plan:
        raise SystemExit("base L1 fit plan no longer matches pinned inputs")
    base_records = _records(_read_pinned(args.base_fit_dir, BASE_FIT_PIN)["training_records.jsonl"])
    base_summary = _json(_read_pinned(args.base_census_dir, BASE_CENSUS_PIN)["summary.json"])
    supplement_plan, supplement_records = _load_supplement(args.supplement_dir, args.census_dir, base_summary)
    combined = _combine(base_plan, supplement_plan, base_records, supplement_records, base_summary)
    combined["pins"] = {"base_plan_sha256": BASE_PLAN_SHA256,
                        "base_fit": base_plan["pins"],
                        "difference_supplement": supplement_plan["pins"],
                        "combined_preflight_source_sha256": _sha(Path(__file__).read_bytes())}
    write_new_plan(args.out, combined)
    print(json.dumps({"out": str(args.out), "status": combined["status"],
                      "records": combined["source_counts"]["records"],
                      "tasks": combined["source_counts"]["tasks"],
                      "training_rows": combined["split"]["training_rows"],
                      "diagnostic_rows": combined["split"]["diagnostic_rows"],
                      "semantic_operation_counts": combined["semantic_operation_counts"],
                      "training_authorized": combined["training_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

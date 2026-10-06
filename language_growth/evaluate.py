"""Execute synthetic L0 proposals and report observed boundary behavior.

This is a deterministic fixture replay, not a learned planner evaluation. Authored
task labels are annotations; no independent task-correctness oracle runs here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any
from decimal import Decimal, InvalidOperation

from fixtures.language_growth.validate import read_json, validate_pack

from .kernel import check_program
from .realize import render_checked


def _attempt(program: dict[str, Any], host: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    result = check_program(program, host, catalog)
    observed = result.to_dict()
    observed["rendered_text"] = None
    observed["release_status"] = "not_applicable"
    if result.status == "accepted":
        try:
            observed["rendered_text"] = render_checked(result, host, catalog)
            observed["release_status"] = "released"
        except ValueError:
            observed["release_status"] = "denied"
    return observed


def evaluate_case(case: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Run the authored proposal schedule without consulting expected labels."""
    host = case["host"]
    first = _attempt(case["proposal"], host, catalog)
    attempts = [first]
    repair = case.get("repair")
    repair_state = "not_requested"
    if repair and repair.get("requested"):
        repair_state = "not_needed" if repair.get("permitted") and first["status"] == "accepted" else "not_permitted"
        max_attempts = repair.get("max_scripted_attempts", 1)
        if type(max_attempts) is not int or max_attempts not in (1, 2):
            raise ValueError("invalid scripted attempt budget")
        if (
            repair.get("permitted")
            and first["status"] == "rejected"
            and max_attempts >= 2
            and repair.get("repaired_proposal") is not None
        ):
            feedback = repair.get("feedback_to_planner", {}).get("reason_code")
            if feedback is not None and feedback != first["reason"]:
                repair_state = "feedback_mismatch"
            else:
                attempts.append(_attempt(repair["repaired_proposal"], host, catalog))
                repair_state = "scripted_repair_executed"
        elif max_attempts < 2 and first["status"] != "accepted":
            repair_state = "budget_exhausted"
    final = attempts[-1]
    provider_status = "fail" if first["status"] == "provider_rejected" else "pass"
    kernel_status = "not_run" if first["status"] == "provider_rejected" else final["status"]
    speech_kinds = [output["kind"] for output in final["outputs"]]
    observed = {
        "provider_precondition": provider_status,
        "kernel_status": kernel_status,
        "kernel_reason_category": final["reason"] if kernel_status == "rejected" else None,
        "release_status": final["release_status"],
        "rendered_text": final["rendered_text"],
        "speech_kinds": speech_kinds,
        "ask_emitted": "ask" in speech_kinds and final["release_status"] == "released",
        "proposal_attempts": len(attempts),
        "actual_provider_calls": 0,
        "budget_scope": "authored_AST_schedule_only_not_planner_call_qualification",
        "scripted_repair_attempts": len(attempts) - 1,
        "repair_state": repair_state,
        "task_correctness": "not_independently_scored",
        "useful_complete_answer": None,
    }
    expected = case["expected"]
    comparisons = {
        "provider_precondition_matches_authored": provider_status == expected["provider_precondition"],
        "kernel_status_matches_authored": kernel_status == expected["kernel_status"],
        "kernel_reason_matches_authored": observed["kernel_reason_category"] == expected["kernel_reason_category"],
    }
    if repair and "initial_expected" in repair:
        comparisons["initial_kernel_status_matches_authored"] = first["status"] == repair["initial_expected"]["kernel_status"]
        comparisons["initial_reason_matches_authored"] = first["reason"] == repair["initial_expected"]["kernel_reason_category"]
    return {
        "case_id": case["case_id"],
        "world_id": case["world_id"],
        "decision_question_id": case["decision_only"]["question_id"],
        "attempts": attempts,
        "observed": observed,
        "authored_expectation": deepcopy(expected),
        "authored_label_provenance": case["label_provenance"],
        "comparison": comparisons,
    }


def evaluate_pack(fixtures: Path, out: Path) -> dict[str, Any]:
    fixtures = fixtures.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError("output directory already exists; choose a new directory")
    validate_pack(fixtures)
    manifest = read_json(fixtures / "manifest.json")
    catalog = read_json(fixtures / manifest["catalog"])
    rows = [evaluate_case(read_json(fixtures / entry["path"]), catalog) for entry in manifest["cases"]]
    counts = Counter(row["observed"]["kernel_status"] for row in rows)
    provider_counts = Counter(row["observed"]["provider_precondition"] for row in rows)
    repair_counts = Counter(row["observed"]["repair_state"] for row in rows)
    mismatches = [row["case_id"] for row in rows if not all(row["comparison"].values())]
    release_failures = [row["case_id"] for row in rows if row["observed"]["release_status"] == "denied"]
    summary = {
        "schema": "ssdm.l0_fixture_evaluation.v0",
        "scope": "synthetic_authored_fixture_replay_no_model_run",
        "case_count": len(rows),
        "provider_precondition_counts": dict(provider_counts),
        "observed_kernel_status_counts": dict(counts),
        "observed_repair_state_counts": dict(repair_counts),
        "observed_ask_emitted_count": sum(row["observed"]["ask_emitted"] for row in rows),
        "proposal_attempts": sum(row["observed"]["proposal_attempts"] for row in rows),
        "scripted_repair_attempts": sum(row["observed"]["scripted_repair_attempts"] for row in rows),
        "authored_boundary_expectation_mismatches": mismatches,
        "release_failures": release_failures,
        "task_correctness": "not_independently_scored",
        "learned_model_accuracy": None,
        "learned_model_coverage": None,
        "decision_only_model_calls": 0,
        "note": "Expected task outcomes and useful-answer labels are authored annotations, not observed kernel or model results.",
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / "case_results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = evaluate_pack(args.fixtures, args.out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["authored_boundary_expectation_mismatches"] or summary["release_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recompute scores from preserved programs and aggregate seeds by task cluster."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from language_growth.curriculum_probe import load_rows, paired_ci, write, SEEDS, OLD_RECORD_HASH
from language_growth.free_running_dev import score_development_task
from language_growth.kernel import check_program
from language_growth.planner import PlannerResult


def audit(root, old_records):
    receipt_raw = (root / "run/receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    for name, digest in receipt["artifacts_sha256"].items():
        assert hashlib.sha256((root / "run" / name).read_bytes()).hexdigest() == digest, name
    data = json.loads((root / "data/manifest.json").read_bytes())
    for name, info in data["files"].items():
        assert hashlib.sha256((root / "data" / name).read_bytes()).hexdigest() == info["file_sha256"], name
        assert len(load_rows(root / "data" / name, info["jsonl_sha256"])) == info["rows"]
    examples = load_rows(root / "data/development_transfer_v2-tasks.jsonl.gz")
    old_raw = old_records.read_bytes()
    assert hashlib.sha256(old_raw).hexdigest() == OLD_RECORD_HASH
    old = [json.loads(line) for line in old_raw.splitlines()]
    curriculum = load_rows(root / "data/development_curriculum-records.jsonl.gz")
    assert len(old) == len(curriculum) == 13332
    changed = 0
    for before, after in zip(old, curriculum):
        assert before["labels"] == after["labels"]
        assert before["request"]["questions"] == after["request"]["questions"]
        a, b = json.loads(before["request"]["state"]), json.loads(after["request"]["state"])
        changed += a.pop("task_input") != b.pop("task_input")
        assert a == b
    arms = {}
    teacher_rows = {}
    tie_counts = {}
    for name, result in receipt["results"].items():
        tasks = load_rows(root / "run" / f"{name}-tasks.jsonl.gz")
        calls = load_rows(root / "run" / f"{name}-free-calls.jsonl.gz")
        slots = load_rows(root / "run" / f"{name}-teacher-slots.jsonl.gz")
        free = load_rows(root / "run" / f"{name}-free-slots.jsonl.gz")
        assert len(tasks) == 300 and len(slots) == len(free) == 800
        assert sum(r["provider_calls"] for r in tasks) == len(calls)
        for example, observed in zip(examples, tasks):
            assert example["task_id"] == observed["task_id"]
            predicted = PlannerResult(status=observed["status"], text=observed["text"], program=observed["program"])
            assert score_development_task(example, predicted) == observed["task_correct"]
            assert (observed["program"] == example["training_only"]["program"]) == observed["exact_program_correct"]
            if observed["program"] is not None:
                evidence = example["training_only"]
                accepted = check_program(observed["program"], evidence["host"], evidence["catalog"]).status == "accepted"
                assert accepted == observed["kernel_accepted"]
            else:
                assert not observed["kernel_accepted"]
        summary = result["summary"]
        assert sum(t["task_correct"] for t in tasks) == summary["semantic_correct"]
        assert sum(t["exact_program_correct"] for t in tasks) == summary["exact_program_correct"]
        assert sum(t["kernel_accepted"] for t in tasks) == summary["kernel_accepted"]
        assert sum(t["correct"] for t in slots) == summary["teacher_slot_correct"]
        assert sum(t["correct"] for t in free) == summary["free_slot_correct"]
        for rows in (slots, free):
            for row in rows:
                assert row["correct"] == (row["gold_description"] == row["prediction_description"])
        tie_counts[name] = sum(sum(score == max(call["logits"]) for score in call["logits"]) > 1 for call in calls)
        arms[name] = tasks
        teacher_rows[name] = {(r["task_id"], r["slot"]): r for r in slots}
    for seed in SEEDS:
        a = receipt["results"][f"old_cohort-seed{seed}"]
        b = receipt["results"][f"curriculum-seed{seed}"]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        for baseline_name in (f"old_cohort-seed{seed}", "pr64_frozen_seed20261006"):
            key = f"curriculum-seed{seed}_minus_{baseline_name}"
            for metric, reported in receipt["paired_comparisons"][key].items():
                assert paired_ci(arms[baseline_name], arms[b["name"]], metric,
                                 slots=metric.endswith("slot_correct")) == reported
    metrics = ("task_correct", "exact_program_correct", "kernel_accepted", "slot_correct", "free_slot_correct")
    average = {}
    for arm in ("old_cohort", "curriculum"):
        average[arm] = []
        for index, example in enumerate(examples):
            seed_rows = [arms[f"{arm}-seed{seed}"][index] for seed in SEEDS]
            average[arm].append({"task_id": example["task_id"], "kind": seed_rows[0]["kind"],
                "slot_total": seed_rows[0]["slot_total"],
                **{metric: sum(float(r[metric]) for r in seed_rows) / len(SEEDS) for metric in metrics}})
    per_slot = {}
    for slot in ("operation", "fact", "left", "right", "relation"):
        slot_arms = {}
        for arm in ("old_cohort", "curriculum"):
            slot_arms[arm] = []
            for index, example in enumerate(examples):
                tid = example["task_id"]
                rows = [teacher_rows[f"{arm}-seed{seed}"].get((tid, slot)) for seed in SEEDS]
                active = rows[0] is not None
                slot_arms[arm].append({"task_id": tid, "kind": example["training_only"]["kind"],
                    "slot_total": int(active),
                    "correct": sum(r["correct"] for r in rows) / len(rows) if active else 0.0})
        per_slot[slot] = paired_ci(slot_arms["old_cohort"], slot_arms["curriculum"], "correct", slots=True)
    output = {"receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(), "verified": True,
        "task_count": 300, "models": len(arms), "free_call_top_logit_ties": tie_counts,
        "latent_world_match": {"old_records_sha256": OLD_RECORD_HASH, "matched_records": len(old),
            "changed_wording_records": changed, "equal_after_stripping_only_task_input": True},
        "mean_seed_teacher_by_slot_comparisons": per_slot,
        "mean_seed_matched_comparisons": {metric: paired_ci(average["old_cohort"], average["curriculum"], metric,
            slots=metric.endswith("slot_correct")) for metric in metrics},
        "method": "average paired seed outcomes within each task, then resample task clusters; seeds are fixed",
        "data_source": "raw free-running programs, requests, and separate frozen gold tasks"}
    write(root / "audit.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-records", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.old_records)

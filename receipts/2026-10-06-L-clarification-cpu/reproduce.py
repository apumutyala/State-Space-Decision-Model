"""Regenerate scripted CPU clarification outcomes; no model, network, or audit data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from language_growth.development_data import development_manifest, generate_development_tasks
from language_growth.tests.test_clarification import SemanticProvider, run


def main() -> None:
    out = Path(sys.argv[1])
    if out.exists():
        raise ValueError("output directory must be fresh")
    out.mkdir(parents=True)
    rows = []
    manifests = {}
    for split, seed in (("development_train", 17), ("development_validation", 19)):
        tasks = generate_development_tasks(seed, 25, split=split, choice_format="mixed",
                                           clarification_mode=True)
        manifests[split] = development_manifest(tasks)
        report = run(tasks, SemanticProvider())
        if (report["total"], report["kernel_accepted"], report["task_correct"]) != (25, 25, 25):
            raise AssertionError((split, report))
        for task, row in zip(tasks, report["observations"]):
            rows.append({"condition": "scripted_semantic_choice", "split": split,
                         "task_id": task["task_id"], "kind": task["training_only"]["kind"],
                         "status": row["status"], "kernel_accepted": row["kernel_accepted"],
                         "task_correct": row["task_correct"], "provider_calls": row["provider_calls"],
                         "program": row["program"], "text": row["text"]})
    five = generate_development_tasks(31, 5, choice_format="mixed", clarification_mode=True)
    for condition, task, provider in (
        ("wrong_question_ask", five[3], SemanticProvider(wrong_ask=True)),
        ("ask_on_answerable_task", five[4], SemanticProvider(ask_even_when_answerable=True)),
    ):
        row = run([task], provider)["observations"][0]
        if (row["status"], row["kernel_accepted"], row["task_correct"]) != ("accepted", True, False):
            raise AssertionError((condition, row))
        rows.append({"condition": condition, "split": task["development_split"],
                     "task_id": task["task_id"], "kind": task["training_only"]["kind"],
                     "status": row["status"], "kernel_accepted": row["kernel_accepted"],
                     "task_correct": row["task_correct"], "provider_calls": row["provider_calls"],
                     "program": row["program"], "text": row["text"]})
    with (out / "outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    summary = {"scope": "scripted nonlearned CPU development only", "total": len(rows),
               "scripted_correct": 50, "wrong_question_ask_task_wrong": True,
               "ask_on_answerable_task_wrong": True, "manifests": manifests}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    print(json.dumps({"total": len(rows), "scripted_correct": 50,
                      "negative_controls_task_wrong": 2}, sort_keys=True))


if __name__ == "__main__":
    main()

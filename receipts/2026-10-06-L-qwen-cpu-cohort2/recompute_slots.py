"""Recompute L0 per-slot grounding outcomes from the committed raw cohort trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from language_growth.development_data import generate_development_tasks

SEED = 271828
INDICES = (2, 5)
TASK_HASHES = {
    2: "e996f537fea463e480f691b36ed68b5f3a8aa1da014e798b706a672473029e0c",
    5: "cc9d1d2418ef38891870e517dd1fcc75d01e238023e65dcc80e9de9aac783b0f",
}
RAW_HASH = "b2bc174efc6d3face4dde864399097e3d85e2d1db8a27b3c68e566bc6f872ae3"


def canonical_hash(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    out = Path(__file__).resolve().parent
    raw_path = out / "raw.jsonl"
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != RAW_HASH:
        raise SystemExit("raw trace hash mismatch")
    events = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()]

    generated = generate_development_tasks(SEED, max(INDICES) + 1,
                                           choice_format="production",
                                           split="development_validation")
    examples = {}
    for index in INDICES:
        task = generated[index]
        expected_id = f"synthetic-development_validation-{SEED}-{index}"
        if (task["task_id"] != expected_id
                or canonical_hash(task) != TASK_HASHES[index]):
            raise SystemExit(f"task identity/hash mismatch at index {index}")
        examples[task["task_id"]] = task

    targets: dict[str, dict[str, str]] = {}
    for task_id, task in examples.items():
        mapping = {}
        for request, target in zip(task["model_inputs"], task["training_only"]["targets"]):
            qid, question = next(iter(request["questions"].items()))
            if qid.rsplit("-", 1)[-1] != target["slot"]:
                raise SystemExit("training target/request slot mismatch")
            mapping[target["slot"]] = question["criteria"][target["option_id"]]
        targets[task_id] = mapping

    selected: dict[str, dict[str, dict[str, object]]] = {task_id: {} for task_id in examples}
    pending: list[tuple[str, str, dict[str, str], str]] = []
    for event in events:
        if event.get("event") == "provider_request":
            task_id = event.get("task_id")
            request = event["request"]
            if task_id not in examples:
                raise SystemExit("provider request has unexpected task identity")
            qid, question = next(iter(request["questions"].items()))
            state = json.loads(request["state"])
            if ("targets" in request or "training_only" in request or "program" in request
                    or state.get("proposed_slot_choices") is None):
                raise SystemExit("unexpected label/program leak in provider request")
            slot = qid.split("-", 2)[-1]
            pending.append((task_id, slot, question["criteria"], qid))
        elif event.get("event") in ("provider_response", "provider_exception"):
            if not pending:
                raise SystemExit("provider response without request")
            task_id, slot, criteria, qid = pending.pop(0)
            if slot not in targets[task_id]:  # direct comparison arm is not a planner slot
                continue
            if event["event"] == "provider_exception":
                selected_id, status = None, "provider_exception"
            else:
                response = event["response"]
                choice = response.get("answers", {}).get(qid, {}).get("choice")
                selected_id = choice if isinstance(choice, str) else None
                status = response.get("ssdm", {}).get("status", "invalid_response")
            selected_description = criteria.get(selected_id) if selected_id is not None else None
            target_description = targets[task_id][slot]
            selected[task_id][slot] = {
                "selected_id": selected_id,
                "expected_runtime_description": target_description,
                "selected_description": selected_description,
                "correct": selected_description == target_description,
                "status": status,
                "missing_choice": selected_id is None,
            }
    if pending:
        raise SystemExit("unpaired provider request at end of raw trace")

    by_slot = {}
    rows = []
    for task_id, mapping in targets.items():
        row = {"task_id": task_id, "slots": {}}
        for slot, target_description in mapping.items():
            result = selected[task_id].get(slot, {
                "selected_id": None,
                "expected_runtime_description": target_description,
                "selected_description": None,
                "correct": False,
                "status": "not_reached",
                "missing_choice": True,
            })
            row["slots"][slot] = result
            total = by_slot.setdefault(slot, {"total": 0, "correct": 0, "missing_choice": 0})
            total["total"] += 1
            total["correct"] += int(result["correct"])
            total["missing_choice"] += int(result["missing_choice"])
        rows.append(row)

    summary = {
        "source_commit": "4f4c3c0dfdd93c96a82a2c4bf2f992fca573dd55",
        "raw_sha256": RAW_HASH,
        "development_seed": SEED,
        "split": "development_validation",
        "preselected_indices": list(INDICES),
        "total_expected_slots": sum(values["total"] for values in by_slot.values()),
        "correct_slots": sum(values["correct"] for values in by_slot.values()),
        "by_slot": by_slot,
        "tasks": rows,
        "confidence_interval": None,
        "scope": "Post-call slot analysis of two preselected hybrid-Qwen development diagnostics; no population estimate.",
    }
    (out / "slot_diagnostics.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

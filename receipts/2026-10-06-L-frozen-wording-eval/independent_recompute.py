"""Independent reviewer's raw-only reconstruction; no inference or fitting."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from language_growth.kernel import check_program
from language_growth.realize import render_checked

root = Path("receipts/2026-10-06-L-frozen-wording-eval")
receipt = json.loads((root / "receipt.json").read_text())
tasks = [json.loads(line) for line in gzip.decompress((root / "evaluation_tasks.jsonl.gz").read_bytes()).splitlines()]
lookup = {task["task_id"]: task for task in tasks}
matrix = []
for run in receipt["runs"]:
    path = root / run["raw_artifact"]["filename"]
    compressed = path.read_bytes()
    encoded = gzip.decompress(compressed)
    assert hashlib.sha256(compressed).hexdigest() == run["raw_artifact"]["sha256"]
    assert hashlib.sha256(encoded).hexdigest() == run["raw_artifact"]["decompressed_sha256"]
    raw = json.loads(encoded)
    assert [o["task_id"] for o in raw["observations"]] == list(lookup)
    counts = collections.defaultdict(collections.Counter)
    correct = []
    for observation in raw["observations"]:
        target = lookup[observation["task_id"]]["training_only"]
        actual, expected = observation["program"], target["program"]
        checked = check_program(actual, target["host"], target["catalog"])
        reference = check_program(expected, target["host"], target["catalog"])
        assert [n["op"] for n in actual["nodes"]] == ["ReadFact", "Assert"]
        semantic = (target["kind"] == "read"
            and actual["nodes"][0]["record_ref"] == expected["nodes"][0]["record_ref"]
            and actual["nodes"][0]["field"] == expected["nodes"][0]["field"]
            and observation["text"] == render_checked(checked, target["host"], target["catalog"]))
        exact_program = actual == expected
        exact_text = observation["text"] == render_checked(reference, target["host"], target["catalog"])
        assert semantic == observation["task_correct"]
        assert exact_program == observation["exact_program_correct"]
        assert exact_text == observation["exact_text_correct"]
        counts[target["kind"]].update(total=1, semantic=semantic, exact_program=exact_program,
            exact_text=exact_text, accepted=checked.status == "accepted")
        correct.append(semantic)
    calls_by_task = collections.defaultdict(list)
    operation_choices = collections.Counter()
    for call in raw["calls"]:
        calls_by_task[call["task_id"]].append(call)
    for task_id in lookup:
        calls = calls_by_task[task_id]
        assert len(calls) == 2
        prior = []
        for call in calls:
            request = call["request"]
            state = json.loads(request["state"])
            qid = next(iter(request["questions"]))
            slot = qid.rsplit("-", 1)[1]
            assert state["proposed_slot_choices"] == prior
            assert not ({"targets", "kind", "template_id", "training_only", "program", "gold"} & state.keys())
            choice = call["response"]["answers"][qid]["choice"]
            logits = call["logits"][qid]
            assert choice == max(sorted(logits), key=logits.get)
            if slot == "operation":
                operation_choices[choice] += 1
            prior.append({"slot": slot, "option_id": choice,
                "description": request["questions"][qid]["criteria"][choice]})
    matrix.append(correct)
    print(run["seed"], dict(counts), len(raw["calls"]), dict(operation_choices))
pooled = np.array(matrix, dtype=float).mean(axis=0)
rng = np.random.default_rng(1729)
draws = rng.integers(0, 300, (10000, 300))
print("accuracy", pooled.mean())
print("95% CI", np.quantile(pooled[draws].mean(axis=1), [.025, .975]).tolist())
print("elapsed", receipt["elapsed_seconds"], "cost", receipt["cost"])

"""Train-only diagnostic supervision for the actual bounded GatePlanner.

Targets are read from exact synthetic reference programs, outside the provider
request. Earlier target choices are replayed as proposed_slot_choices, so these
are teacher-forced training rows, never free-running results or qualification.
Only build_suite(development_train) is called; no external dataset is accepted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
from typing import Any

from .gate_planner import plan_gate_task
from .gate_suites import LEVELS, build_suite, canonical, digest
from .parent_bridge import DECISION_SCHEMA, FINGERPRINT_ALGORITHM, legacy_request_fingerprint
from .planner import _fact_description

TRAIN_SPLIT = "development_train"
SCHEMA = "ssdm.gate_training_export.v1"
REPLAY_CONFIG = {
    "calibration": {"kind": "option_count_temperature", "id": "gate-training-replay-v1",
                    "t0": 1.0, "t1": 0.0, "t_min": 0.05},
    "policy_masking": "none (served covers every supplied option)",
}


def _target_choices(task: dict, reference: dict) -> list[tuple[str, str]]:
    """Map the exact reference syntax to slot descriptions, never search truth.

    Offered domains still come exclusively from the live GatePlanner. Full
    descriptions must match uniquely; there is no substring or first-match
    fallback. The final replay must reconstruct the exact reference program.
    """
    program = reference["program"]
    if task.get("planner_profile") == "energy":
        row = next(r for r in task["energy_module"]["rows"] if r["record_ref"] == program["record_ref"])
        return [("energy-operation", program["op"]),
                ("energy-record", f"Technology {row['technology']} record {row['record_ref']}")]
    nodes = {node["id"]: node for node in program["nodes"]}
    output = nodes[program["outputs"][0]]
    root = nodes[output["proposition"]] if output["op"] == "Assert" else output
    if root["op"] == "ExplainDerivation":
        operation = "ExplainDerivation"
        root = nodes[nodes[root["speech_act"]]["proposition"]]
    else:
        operation = root["op"]
    choices = [("semantic-operation", operation)]

    def fact(slot: str, node: dict) -> None:
        if node["op"] != "ReadFact":
            raise ValueError("reference operand is not a ReadFact")
        record = next(r for r in task["catalog"]["records"] if r["record_ref"] == node["record_ref"])
        choices.append((slot, _fact_description({"record_ref": node["record_ref"],
                                                "field": node["field"], "entity_ref": record["entity_ref"]})))

    def operand(slot: str, value: dict) -> None:
        if set(value) != {"node", "project"} or value["project"] != "object":
            raise ValueError("unsupported reference operand")
        fact(slot, nodes[value["node"]])

    if operation == "Plan":
        workflow = root["workflow_ref"]
        definition = task["host"]["workflows"][workflow]
        choices.append(("workflow", f"Workflow {workflow}: {json.dumps(definition, sort_keys=True)}"))
        for step in root["step_refs"]:
            choices.append(("plan-step", f"Step {step}: {json.dumps(definition['steps'][step], sort_keys=True)}"))
    elif operation == "Hypothesize":
        compared = nodes[root["proposition"]]
        assumption = compared["left"]["assumption_ref"]
        definition = task["host"]["assumptions"][assumption]
        choices.append(("assumption", f"Assumption {assumption}: {json.dumps(definition, sort_keys=True)}"))
        operand("right-operand", compared["right"])
        choices.append(("comparison-relation", compared["relation"]))
    elif operation == "Conjoin":
        if len(root["speech_acts"]) != 2:
            raise ValueError("only the actual two-fact GatePlanner is supported")
        for slot, act in zip(("first-fact", "second-fact"), root["speech_acts"]):
            fact(slot, nodes[nodes[act]["proposition"]])
    elif operation in ("Compare", "ExactDifference"):
        operand("left-operand", root["left"])
        operand("right-operand", root["right"])
        if operation == "Compare":
            choices.append(("comparison-relation", root["relation"]))
    elif operation in ("ReadFact", "ExplainDerivation"):
        fact("fact", root)
    else:
        raise ValueError("unsupported reference operation")
    if task["level"] == "L3":
        choices.append(("licensed-production", task["production"]))
    return choices


class _TrainingReplay:
    """Nonlearned label owner. It captures, but never augments, model requests."""

    def __init__(self, choices: list[tuple[str, str]], *, model: str, parent_revision: str):
        self.choices, self.model, self.parent_revision = choices, model, parent_revision
        self.rows: list[dict[str, Any]] = []

    def planner_kwargs(self) -> dict:
        return {"model": self.model, "parent_revision": self.parent_revision,
                "expected_served_config": deepcopy(REPLAY_CONFIG)}

    def decide(self, request: dict) -> dict:
        index = len(self.rows)
        slot, description = self.choices[index]
        qid, question = next(iter(request["questions"].items()))
        if qid != f"slot-{index + 1:03d}-{slot}":
            raise ValueError("reference slot differs from actual planner")
        options = question["criteria"]
        matches = [oid for oid, text in options.items() if text == description]
        if len(matches) != 1:
            raise ValueError("reference target is not a unique offered option")
        chosen = matches[0]
        self.rows.append({"request": deepcopy(request), "labels": {qid: chosen}})
        probabilities = {oid: float(oid == chosen) for oid in options}
        return {"answers": {qid: {"type": "choice", "choice": chosen,
                                   "probabilities": probabilities, "confidence": 1.0}},
                "ssdm": {"schema": DECISION_SCHEMA, "status": "ok",
                         "request_id": request["ssdm"]["request_id"],
                         "raw_probabilities": {qid: dict(probabilities)},
                         "served_probabilities": {qid: probabilities},
                         "served_config": {**deepcopy(REPLAY_CONFIG), "temperature_per_question": {qid: 1.0}},
                         "diagnostics": {"revision_id": self.parent_revision,
                                         "request_fingerprint": legacy_request_fingerprint(request),
                                         "request_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
                                         "provenance": {"supplied": False, "applied": False},
                                         "state_handle_in": None, "state_reuse": False, "inference_calls": 0},
                         "state_handle": None}}


def _replay_task(task: dict, reference: dict, *, model: str, parent_revision: str) -> list[dict]:
    if task.get("split") != TRAIN_SPLIT or not task["task_id"].startswith("language-development_train-"):
        raise ValueError("only development_train tasks may be exported")
    if task["level"] != "L4" and not task["world_id"].startswith("language-development_train-"):
        raise ValueError("only development_train worlds may be exported")
    if (reference.get("task_id") != task["task_id"] or reference.get("world_id") != task["world_id"]
            or reference.get("input_sha256") != digest(task)):
        raise ValueError("reference input identity mismatch")
    replay = _TrainingReplay(_target_choices(task, reference), model=model, parent_revision=parent_revision)
    result = plan_gate_task(task, replay)
    if (result.status != "accepted" or result.program != reference["program"]
            or len(replay.rows) != len(replay.choices) or result.repair_requests):
        raise ValueError(f"reference replay failed: {result.status}/{result.reason}")
    rows = []
    for index, row in enumerate(replay.rows):
        # Rekey only the transport correlation ID. The fingerprinted D model
        # input (state/questions) remains exactly as emitted by _Session.
        request = row["request"]
        qid = next(iter(row["labels"]))
        request["ssdm"]["request_id"] = digest([SCHEMA, task["task_id"], qid])
        rows.append({"id": task["task_id"], "group_id": task["world_id"], **row, "metadata": {
            "schema": SCHEMA, "split": TRAIN_SPLIT, "world_id": task["world_id"],
            "level": task["level"], "slot": replay.choices[index][0],
            "input_sha256": digest(task), "reference_program_sha256": digest(reference["program"]),
            "request_fingerprint": legacy_request_fingerprint(request),
            "diagnostic_teacher_forced": True, "earlier_target_choices": index,
            "free_running": False, "promotion_eligible": False,
        }})
    return rows


def generate_training_records(*, levels=LEVELS, split=TRAIN_SPLIT, worlds=4, seed=20261008,
                              model="gate-training", parent_revision="training-only:v1") -> list[dict]:
    """Generate small train cohorts only; never load caller-supplied datasets.

    L4 has a single shared historical source-world and only ReadTechnology
    training tasks. LowestMedian's validation reference is never imported.
    """
    levels = tuple(levels)
    if (split != TRAIN_SPLIT or not levels or len(set(levels)) != len(levels)
            or any(level not in LEVELS for level in levels)
            or type(worlds) is not int or not 1 <= worlds <= 64 or type(seed) is not int
            or not isinstance(model, str) or not model or not isinstance(parent_revision, str) or not parent_revision):
        raise ValueError("invalid train-only bounded export configuration")
    rows = []
    for level in levels:
        tasks, references = build_suite(level, TRAIN_SPLIT, worlds=worlds, seed=seed)
        for task, reference in zip(tasks, references, strict=True):
            rows.extend(_replay_task(task, reference, model=model, parent_revision=parent_revision))
    identities = [(row["id"], next(iter(row["labels"]))) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate training slot identity")
    return rows


def export_training(out: Path, **kwargs) -> dict:
    out = Path(out)
    if out.exists():
        raise ValueError("training output already exists")
    rows = generate_training_records(**kwargs)
    data = b"".join(canonical(row) + b"\n" for row in rows)
    compressed = gzip.compress(data, mtime=0)
    manifest = {"schema": SCHEMA, "split": TRAIN_SPLIT, "records": len(rows),
                "tasks": len({row["id"] for row in rows}),
                "worlds": len({row["metadata"]["world_id"] for row in rows}),
                "levels": sorted({row["metadata"]["level"] for row in rows}),
                "slot_counts": dict(sorted(Counter(row["metadata"]["slot"] for row in rows).items())),
                "seed": kwargs.get("seed", 20261008), "requested_worlds": kwargs.get("worlds", 4),
                "source": "language_growth.gate_suites.build_suite/development_train",
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("gate_training.py", "gate_suites.py", "gate_planner.py", "planner.py", "parent_bridge.py")},
                "training_records_sha256": hashlib.sha256(data).hexdigest(),
                "training_records_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                "diagnostic_teacher_forced": True, "free_running": False, "promotion_eligible": False,
                "checker_filtered_candidates": False, "validation_or_heldout_generated": False,
                "limitations": ["Synthetic checker-valid supervision is not independent audit or learned capability.",
                                "L4 shares one fixed public source-world; train contains ReadTechnology only.",
                                "Earlier target choices are diagnostic teacher forcing; evaluate free-running separately."]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "training_records.jsonl").write_bytes(data)
    (out / "training_records.jsonl.gz").write_bytes(compressed)
    (out / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--split", choices=(TRAIN_SPLIT,), default=TRAIN_SPLIT)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--model", default="gate-training")
    parser.add_argument("--parent-revision", default="training-only:v1")
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    print(json.dumps(export_training(out, **args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

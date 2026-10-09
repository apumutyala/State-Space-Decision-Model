"""Development-only synthetic operation-selection pair generation.

This module deliberately avoids importing model runtimes so pair generation
and exact GatePlanner request checks stay cheap and usable on CPU-only hosts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .gate_suites import canonical, digest

_WORDING = {
    "development_train": {
        "Compare": (
            "Compare the October 2026 {field} values for {left} and {right}.",
            "State whether {left} has a greater, smaller, or equal October 2026 {field} value than {right}.",
            "Give the October 2026 {field} ordering for {left} relative to {right}.",
            "Is {left}'s October 2026 {field} higher, lower, or equal to {right}'s? State the relation.",
        ),
        "ExactDifference": (
            "For October 2026, calculate {left}'s {field} minus {right}'s {field}.",
            "Report the signed {field} gap as {left} relative to {right} for October 2026.",
            "How much more or less {field} did {left} record than {right} in October 2026?",
            "Give the oriented difference in October 2026 {field} from {right} to {left}.",
        ),
    },
    "development_validation": {
        "Compare": (
            "Which ordering holds between {left}'s and {right}'s {field} values for October 2026?",
            "Relative to {right}, does {left} record a greater, equal, or lesser {field} value in October 2026?",
            "Place {left} and {right} in order by their October 2026 {field} values.",
            "Report the comparison relation for the October 2026 {field} values of {left} versus {right}.",
        ),
        "ExactDifference": (
            "What signed amount is obtained by taking {left}'s October 2026 {field} relative to {right}'s?",
            "State the October 2026 {field} surplus or shortfall of {left} against {right}, with sign.",
            "By how much does {left}'s October 2026 {field} value exceed or fall short of {right}'s?",
            "What is the signed amount from {right}'s October 2026 {field} value to {left}'s?",
        ),
    },
}


def development_operation_pairs(*, split: str, worlds: int, seed: int,
                                model: str = "gate-operation-fit",
                                parent_revision: str = "operation-fit:v1") -> list[dict]:
    """Build paired Compare -> ExactDifference K4 operation examples.

    This is a fresh synthetic development cohort, replayed through the actual
    GatePlanner with empty proposal histories. The two requests in each pair
    share the same sorted catalog, host and operation question; only the task
    wording (and the corresponding reference operation) differs. Labels stay
    in fitter metadata and never enter either request.
    """
    from .gate_suites import build_suite
    from .gate_planner import plan_gate_task
    from .gate_training import _TrainingReplay, _target_choices

    if (split not in ("development_train", "development_validation")
            or type(worlds) is not int or not 1 <= worlds <= 4096 or type(seed) is not int
            or not isinstance(model, str) or not model
            or not isinstance(parent_revision, str) or not parent_revision):
        raise ValueError("bounded synthetic development split and configuration required")
    tasks, references = build_suite("L1", split, worlds=worlds, seed=seed)
    by_world = {}
    world_template_index = {}
    world_order = sorted({task["world_id"] for task in tasks if task["level"] == "L1"})
    prompt_index = {world_id: index % 4 for index, world_id in enumerate(world_order)}
    field_index = {world_id: int(index >= (len(world_order) + 1) // 2)
                   for index, world_id in enumerate(world_order)}
    for task, reference in zip(tasks, references, strict=True):
        kinds = [node.get("op") for node in reference["program"]["nodes"]
                 if node.get("op") in ("Compare", "ExactDifference")]
        if not kinds:
            continue
        if len(kinds) != 1:
            raise ValueError("reference must contain exactly one Compare or ExactDifference operation")
        kind = kinds[0]
        # build_suite independently shuffles record order per task. Normalize
        # it, including the snapshot order, so the pair changes only intent.
        task = deepcopy(task)
        task["catalog"]["records"].sort(key=lambda row: row["record_ref"])
        for snapshot in task["catalog"].get("snapshots", {}).values():
            if "record_refs" in snapshot:
                snapshot["record_refs"] = sorted(snapshot["record_refs"])
        reference = deepcopy(reference)
        index = prompt_index[task["world_id"]]
        world_template_index[task["world_id"]] = index
        field = "price" if field_index[task["world_id"]] else "warranty_months"
        if field == "price":
            for node in reference["program"]["nodes"]:
                if node.get("op") == "ReadFact" and node.get("field") == "warranty_months":
                    node["field"] = "price"
        nodes = {node["id"]: node for node in reference["program"]["nodes"]}
        operation_node = next(node for node in reference["program"]["nodes"]
                              if node.get("op") in ("Compare", "ExactDifference"))
        left_ref = nodes[operation_node["left"]["node"]]["record_ref"]
        right_ref = nodes[operation_node["right"]["node"]]["record_ref"]
        records = {record["record_ref"]: record for record in task["catalog"]["records"]}
        if field == "price":
            for node in reference["program"]["nodes"]:
                if node.get("op") == "Compare":
                    left_value = records[left_ref]["fields"][field]["value"]
                    right_value = records[right_ref]["fields"][field]["value"]
                    node["relation"] = ("greater_than" if left_value > right_value else
                                        "less_than" if left_value < right_value else "equal")
        left_name = records[left_ref]["entity_ref"]
        right_name = records[right_ref]["entity_ref"]
        task["task_input"] = _WORDING[split][kind][index].format(
            left=left_name, right=right_name, field=field)
        reference["input_sha256"] = digest(task)
        if (reference.get("task_id") != task["task_id"]
                or reference.get("world_id") != task["world_id"]
                or reference.get("input_sha256") != digest(task)):
            raise ValueError("reference input identity mismatch")
        replay = _TrainingReplay(_target_choices(task, reference), model=model,
                                 parent_revision=parent_revision)
        result = plan_gate_task(task, replay)
        if (result.status != "accepted" or result.program != reference["program"]
                or len(replay.rows) != len(replay.choices) or result.repair_requests):
            raise ValueError(f"reference GatePlanner replay failed: {result.status}/{result.reason}")
        operation_rows = [row for index, row in enumerate(replay.rows)
                          if replay.choices[index][0] == "semantic-operation"]
        if len(operation_rows) != 1:
            raise ValueError("GatePlanner must emit exactly one first-call semantic-operation row")
        row = operation_rows[0]
        request = row["request"]
        qid, question = next(iter(request["questions"].items()))
        import json
        state = json.loads(request["state"])
        if qid != "slot-001-semantic-operation" or state.get("proposed_slot_choices") != []:
            raise ValueError("operation pair must use the actual first empty-prefix GatePlanner call")
        spec = {"task_input": task["task_input"], "host": deepcopy(task["host"]),
                "catalog": deepcopy(task["catalog"]), "question_id": qid,
                "question": question["instructions"], "options": deepcopy(question["criteria"])}
        target = next(iter(row["labels"].values()))
        record = by_world.setdefault(task["world_id"], {})
        if kind in record:
            raise ValueError("duplicate operation task in one synthetic world")
        record[kind] = (spec, target)

    pairs = []
    for world_id, variants in sorted(by_world.items()):
        if set(variants) != {"Compare", "ExactDifference"}:
            raise ValueError("each development world must have a matched Compare and ExactDifference task")
        source, source_target = variants["Compare"]
        effect, effect_target = variants["ExactDifference"]
        if (source["host"] != effect["host"] or source["catalog"] != effect["catalog"]
                or any(source[key] != effect[key] for key in ("question_id", "question", "options"))
                or source["task_input"] == effect["task_input"]):
            raise ValueError("operation pair must hold planner schema and shared state fixed")
        if (source_target == effect_target or source_target not in source["options"]
                or effect_target not in source["options"]):
            raise ValueError("GatePlanner operation targets must contrast in one unchanged K4 domain")
        field = "price" if field_index[world_id] else "warranty_months"
        template_id = f"{split.removeprefix('development_')}-{world_template_index[world_id] + 1:02d}"
        pairs.append({"pair_id": f"{world_id}-operation-prompt-{field}-{template_id}", "world_id": world_id,
                      "split": split, "concept": "Compare_to_ExactDifference",
                      "source": source, "effect": effect,
                      "source_target": source_target, "effect_target": effect_target})
    if len(pairs) != worlds:
        raise ValueError("operation pair cohort must contain exactly the requested number of worlds")
    if len({digest(row) for row in pairs}) != len(pairs):
        raise ValueError("duplicate operation pair content")
    return pairs


def export_operation_pairs(out: Path, *, train_worlds: int = 32, train_seed: int = 20261011,
                           validation_worlds: int = 32, validation_seed: int = 20261012) -> dict:
    """Write separate deterministic discovery/validation inputs for the fitter."""
    out = Path(out)
    if out.exists():
        raise ValueError("preserve existing operation-pair artifacts; choose a fresh output directory")
    train = development_operation_pairs(split="development_train", worlds=train_worlds, seed=train_seed)
    validation = development_operation_pairs(split="development_validation", worlds=validation_worlds,
                                             seed=validation_seed)
    visible = [digest({key: pair[side][key] for key in
                       ("task_input", "host", "catalog", "question_id", "question", "options")})
               for pair in train + validation for side in ("source", "effect")]
    if len(visible) != len(set(visible)):
        raise ValueError("discovery and validation must have disjoint visible request content")
    train_bytes = canonical(train) + b"\n"
    validation_bytes = canonical(validation) + b"\n"
    sources = ("gate_operation_pairs.py", "gate_suites.py", "gate_training.py", "gate_planner.py",
               "gate_comparator.py", "planner.py", "parent_bridge.py", "kernel.py")
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in sources}
    manifest = {
        "schema": "ssdm.l1_semantic_operation_pairs.v1",
        "concept": "Compare_to_ExactDifference",
        "target_slot": "semantic-operation",
        "source": "synthetic language_growth.gate_suites L1 development worlds, replayed through GatePlanner",
        "train": {"split": "development_train", "seed": train_seed, "worlds": train_worlds,
                  "pairs": len(train), "path": "discovery-pairs.json",
                  "sha256": hashlib.sha256(train_bytes).hexdigest(), "canonical_digest": digest(train)},
        "validation": {"split": "development_validation", "seed": validation_seed,
                       "worlds": validation_worlds, "pairs": len(validation),
                       "path": "validation-pairs.json",
                       "sha256": hashlib.sha256(validation_bytes).hexdigest(),
                       "canonical_digest": digest(validation)},
        "source_sha256": source_hashes,
        "target_ids": {"Compare": "v002", "ExactDifference": "v004"},
        "wording_templates": {
                split: {f"{split.removeprefix('development_')}-{field}-{index + 1:02d}": {
                "Compare": _WORDING[split]["Compare"][index].replace("{field}", field),
                "ExactDifference": _WORDING[split]["ExactDifference"][index].replace("{field}", field),
            } for field in ("warranty_months", "price") for index in range(4)}
            for split in ("development_train", "development_validation")},
        "template_pair_counts": {
            "train": _template_counts(train), "validation": _template_counts(validation)},
        "field_pair_counts": {"train": _field_counts(train), "validation": _field_counts(validation)},
        "limitations": ["Synthetic development contrasts only; no model fitting or language gate measured.",
                        "The cohort uses two synthetic numeric L1 fields and four fixed, split-disjoint prompt pairs per field.",
                        "Validation pairs must remain separate from discovery fitting and scale selection."],
    }
    manifest_bytes = canonical(manifest) + b"\n"
    out.mkdir(parents=True, exist_ok=False)
    (out / "discovery-pairs.json").write_bytes(train_bytes)
    (out / "validation-pairs.json").write_bytes(validation_bytes)
    (out / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def _template_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {}
    for pair in pairs:
        template_id = pair["pair_id"].rsplit("-operation-prompt-", 1)[1]
        counts[template_id] = counts.get(template_id, 0) + 1
    return dict(sorted(counts.items()))


def _field_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {}
    for pair in pairs:
        field = pair["pair_id"].rsplit("-operation-prompt-", 1)[1].rsplit("-", 2)[0]
        counts[field] = counts.get(field, 0) + 1
    return dict(sorted(counts.items()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-worlds", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20261011)
    parser.add_argument("--validation-worlds", type=int, default=32)
    parser.add_argument("--validation-seed", type=int, default=20261012)
    args = parser.parse_args(argv)
    print(json.dumps(export_operation_pairs(
        args.out, train_worlds=args.train_worlds, train_seed=args.train_seed,
        validation_worlds=args.validation_worlds, validation_seed=args.validation_seed),
        sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

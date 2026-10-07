"""Synthetic four-record L1 preparation; never a learned-level qualification.

Only ``model_inputs`` may be sent to a decision provider. Authored labels,
programs and cohort identities are evaluator metadata, not model inputs.
No L0 data, model weights, audit corpus or provider is accessed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .kernel import check_program
from .parent_bridge import decision_request

_SPLITS = {
    "l1_development_train": {
        "names": ("Moraine", "Nacre", "Oriel", "Peregrine", "Quartz", "Rill"),
        "warranty_range": (61, 84), "price_cents_range": (30001, 45000),
        "templates": {
            "read": "Retrieve the October 2026 {field} entry belonging to {entity}.",
            "explain": "Explain the evidence supporting {entity}'s October 2026 {field} entry.",
            "compare": "Taking {left} first and {right} second, compare their October 2026 {field} entries.",
        },
    },
    "l1_development_validation": {
        "names": ("Sundew", "Tamarack", "Umber", "Vervain", "Wicket", "Yarrow"),
        "warranty_range": (85, 108), "price_cents_range": (45001, 60000),
        "templates": {
            "read": "Identify the recorded {field} associated with {entity} during October 2026.",
            "explain": "Give an evidence-based explanation of the October 2026 {field} recorded for {entity}.",
            "compare": "In the order {left} then {right}, determine the relation between their recorded October 2026 {field}.",
        },
    },
}
# Distinct cohort identities retain the already disjoint development domains.
# Do not rename or alter the original smoke splits: their bytes are receipts.
MULTIWORLD_TRAIN_SPLIT = "l1_multiworld_development_train"
MULTIWORLD_VALIDATION_SPLIT = "l1_multiworld_development_validation"
_SPLITS[MULTIWORLD_TRAIN_SPLIT] = copy.deepcopy(_SPLITS["l1_development_train"])
_SPLITS[MULTIWORLD_VALIDATION_SPLIT] = copy.deepcopy(_SPLITS["l1_development_validation"])
MULTIWORLD_COUNT = 32
MULTIWORLD_SEED = 1707
_FIELDS = ("price", "warranty_months")
_REFS = tuple(f"l1-development-record-{i}" for i in range(1, 5))
_OPS = (("read", "Assert one selected fact"),
        ("explain", "Explain a checked assertion of one selected fact"),
        ("compare", "Compare two selected typed values and assert the selected relation"))
_RELATIONS = ("greater_than", "less_than", "equal")
_CONTEXT = "synthetic-l1-development-context-v1"
_POLICY = "synthetic-l1-development-public-v1"
_REGISTRY = "synthetic-l1-development-registry-v1"
_LABEL_STATUS = "synthetically_authored_checker_valid_not_independently_validated"
TASKS_PER_WORLD = 40


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _rng(*identity: Any) -> random.Random:
    return random.Random(int(_sha(["ssdm-l1-multirecord-v1", *identity]), 16))


def multirecord_split_spec(split: str) -> dict[str, Any]:
    """Return the declared iterative development domain, never an audit split."""
    if split not in _SPLITS:
        raise ValueError("unknown L1 development split")
    return copy.deepcopy(_SPLITS[split])


def _world(split: str, seed: int, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _SPLITS[split]
    rng = _rng(split, seed, index, "world")
    names = rng.sample(spec["names"], 4)
    values = {}
    for field, bounds in (("price", spec["price_cents_range"]),
                          ("warranty_months", spec["warranty_range"])):
        distinct = rng.sample(range(bounds[0], bounds[1] + 1), 3)
        values[field] = [*distinct, distinct[0]]
        rng.shuffle(values[field])
    records = []
    for i, (name, ref) in enumerate(zip(names, _REFS)):
        cents = values["price"][i]
        records.append({
            "record_ref": ref, "version": 1, "entity_ref": name,
            "source_ref": "synthetic-l1-premises", "audience": "public",
            "epistemic_class": "authoritative_structured_premise",
            "freshness": "current", "conflicts": [],
            "fields": {
                "price": {"value": f"{cents // 100}.{cents % 100:02d}", "value_type": "decimal",
                          "unit": "USD", "temporal_scope": "2026-10", "quantity_kind": "price"},
                "warranty_months": {"value": values["warranty_months"][i], "value_type": "integer",
                                    "unit": "month", "temporal_scope": "2026-10",
                                    "quantity_kind": "warranty_months"},
            },
        })
    catalog = {
        "schema": "ssdm.synthetic_l1_development_catalog.v1",
        "sources": {"synthetic-l1-premises": {"premise_admission": "accepted"}},
        "records": records,
        "snapshots": {_CONTEXT: {"audience": "public", "record_refs": list(_REFS)}},
        "policies": {_POLICY: {"status": "current", "audience": "public", "forbidden_audiences": ["private"]}},
        "registries": {_REGISTRY: {
            "allowed_ops": ["ReadFact", "Compare", "Assert", "ExplainDerivation"],
            "registered_fields": list(_FIELDS),
            "field_predicates": {"price": "Price", "warranty_months": "WarrantyMonths"},
            "comparators": list(_RELATIONS), "ask_categories": [], "style_ids": ["plain"],
        }},
    }
    context_ref = f"{_CONTEXT}:{_sha(catalog)}"
    catalog["snapshots"][context_ref] = catalog["snapshots"].pop(_CONTEXT)
    host = {
        "context_ref": context_ref, "policy_ref": _POLICY, "registry_ref": _REGISTRY,
        "audience": "public", "query_constants": {}, "unmet_obligations": [],
        "limits": {"max_program_nodes": 8, "max_questions": 1, "max_options_per_question": 16,
                   "max_proposal_calls": 4, "max_repair_calls": 0, "max_repair_attempts": 0},
        "provider": {"input_context_ref": context_ref, "state_handle": None, "state_history_context_refs": []},
    }
    return host, catalog


def _options(items: list[tuple[str, str]], prefix: str, rng: random.Random,
             augmented: bool) -> tuple[dict[str, str], dict[str, str]]:
    items = list(items)
    if augmented:
        rng.shuffle(items)
    options, ids = {}, {}
    for i, (key, description) in enumerate(items, 1):
        option_id = f"{prefix}{i:03d}"
        options[option_id], ids[key] = description, option_id
    return options, ids


def _task(split: str, seed: int, world_index: int, task_index: int, kind: str,
          field: str, left: int, right: int | None, fmt: str,
          host: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    world_id = f"synthetic-{split}-{seed}-world-{world_index}"
    task_id = f"{world_id}-task-{task_index}"
    records = catalog["records"]
    text = _SPLITS[split]["templates"][kind].format(
        entity=records[left]["entity_ref"], left=records[left]["entity_ref"],
        right=records[right]["entity_ref"] if right is not None else "",
        field=field.replace("_", " "))
    nodes = [{"id": "n1", "op": "ReadFact", "record_ref": _REFS[left], "field": field}]
    relation = None
    if kind == "compare":
        # Exact integer cents, avoiding binary floating point for decimal prices.
        a = records[left]["fields"][field]["value"]
        b = records[right]["fields"][field]["value"]
        if field == "price":
            a, b = int(a.replace(".", "")), int(b.replace(".", ""))
        relation = "equal" if a == b else ("greater_than" if a > b else "less_than")
        nodes.extend([
            {"id": "n2", "op": "ReadFact", "record_ref": _REFS[right], "field": field},
            {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"},
             "right": {"node": "n2", "project": "object"}, "relation": relation},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
        ])
    else:
        nodes.append({"id": "n2", "op": "Assert", "proposition": "n1"})
        if kind == "explain":
            nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
    program = {"schema": "ssdm.semantic_program.v0.1", "context_ref": host["context_ref"],
               "policy_ref": _POLICY, "registry_ref": _REGISTRY,
               "nodes": nodes, "outputs": [nodes[-1]["id"]], "style": "plain"}
    checked = check_program(program, host, catalog)
    if checked.status != "accepted":
        raise ValueError(f"synthetic L1 program rejected: {checked.reason}")
    rng = _rng(split, seed, world_index, task_index, "choice-order")
    facts = [(f"{ref}:{f}", f"Fact {ref} field {f}") for ref in _REFS for f in _FIELDS]
    op_options, op_ids = _options(list(_OPS), "op", rng, fmt == "augmented")
    fact_options, fact_ids = _options(facts, "f", rng, fmt == "augmented")
    operand_options, operand_ids = _options(facts, "v", rng, fmt == "augmented")
    relation_options, relation_ids = _options([(r, f"Relation {r}") for r in _RELATIONS],
                                               "r", rng, fmt == "augmented")
    choices = [("operation", "Select the semantic operation for the user's task.", op_options, op_ids[kind])]
    if kind == "compare":
        choices.extend([
            ("left", "Select the left typed operand.", operand_options, operand_ids[f"{_REFS[left]}:{field}"]),
            ("right", "Select the right typed operand.", operand_options, operand_ids[f"{_REFS[right]}:{field}"]),
            ("relation", "Select the comparison relation.", relation_options, relation_ids[relation]),
        ])
    else:
        choices.append(("fact", "Select the fact to assert.", fact_options, fact_ids[f"{_REFS[left]}:{field}"]))
    requests, targets = [], []
    for position, (slot, question, options, target) in enumerate(choices, 1):
        request, _ = decision_request(
            task_input=text, host=host, catalog=catalog, model="development-only",
            parent_revision="synthetic-no-model:v1",
            request_id=_sha([task_id, fmt, position, host["context_ref"]])[:24],
            question_id=f"slot-{position:03d}-{slot}", question=question, options=options,
            proposed_slot_choices=[])
        requests.append(request)
        targets.append({"slot": slot, "option_id": target})
    return {
        "schema": "ssdm.synthetic_l1_development_task.v1", "development_only": True,
        "development_split": split, "world_id": world_id, "task_id": task_id,
        "model_inputs": requests,
        "evaluator_only": {
            "label_status": _LABEL_STATUS, "seed": seed, "world_index": world_index,
            "task_index": task_index, "kind": kind, "field": field, "choice_format": fmt,
            "template_id": f"{split}:{kind}:0", "host": copy.deepcopy(host),
            "catalog": copy.deepcopy(catalog), "program": program, "targets": targets,
            "comparison_pair_id": (f"{world_id}:{field}:{min(left, right)}:{max(left, right)}"
                                   if right is not None else None),
        },
    }


def generate_multirecord_tasks(seed: int, world_count: int, *,
                               split: str = "l1_development_train",
                               choice_format: str = "mixed") -> list[dict[str, Any]]:
    """Generate complete 40-task worlds, with paired A/B and B/A comparisons.

    Every field has three distinct values across four records, including one
    equal pair. Every ordered distinct-record pair is queried for each field.
    Requests have no authored choice prefix; their slot schedule is authored
    diagnostic scaffolding and must not substitute for a free-running planner.
    """
    if type(seed) is not int or seed < 0 or type(world_count) is not int or not 1 <= world_count <= 100:
        raise ValueError("seed must be nonnegative and world_count must be 1..100")
    multirecord_split_spec(split)
    if choice_format not in ("production", "augmented", "mixed"):
        raise ValueError("invalid choice format")
    tasks = []
    for world_index in range(world_count):
        host, catalog = _world(split, seed, world_index)
        queries = [(kind, field, left, None) for kind in ("read", "explain")
                   for field in _FIELDS for left in range(4)]
        queries += [("compare", field, left, right) for field in _FIELDS
                    for left in range(4) for right in range(4) if left != right]
        for task_index, (kind, field, left, right) in enumerate(queries):
            fmt = ("production" if task_index % 2 == 0 else "augmented") if choice_format == "mixed" else choice_format
            tasks.append(_task(split, seed, world_index, task_index, kind, field, left, right, fmt, host, catalog))
    return tasks


def multirecord_rows(tasks: list[dict[str, Any]]) -> bytes:
    """Exact canonical JSONL bytes used by the manifest, including final LF."""
    return b"".join(_json(task) + b"\n" for task in tasks)


def multirecord_manifest(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pin exact rows and their evaluator identities; reject incomplete worlds."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("nonempty L1 task list required")
    split, seed = tasks[0]["development_split"], tasks[0]["evaluator_only"]["seed"]
    spec = multirecord_split_spec(split)
    if any(t["development_split"] != split or t["evaluator_only"]["seed"] != seed for t in tasks):
        raise ValueError("mixed development domains")
    worlds: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        worlds.setdefault(task["world_id"], []).append(task)
    if len({t["task_id"] for t in tasks}) != len(tasks) or any(
            len(rows) != TASKS_PER_WORLD or {t["evaluator_only"]["task_index"] for t in rows} != set(range(TASKS_PER_WORLD))
            for rows in worlds.values()):
        raise ValueError("complete unique 40-task worlds required")
    comparisons = [t for t in tasks if t["evaluator_only"]["kind"] == "compare"]
    return {
        "schema": "ssdm.synthetic_l1_development_manifest.v1", "split": split,
        "seed": seed, "world_count": len(worlds), "task_count": len(tasks),
        "records_per_world": 4, "tasks_per_world": TASKS_PER_WORLD,
        "label_status": _LABEL_STATUS, "split_spec": spec,
        "sample_sha256": hashlib.sha256(multirecord_rows(tasks)).hexdigest(),
        "sample_encoding": "sorted compact ensure_ascii JSON per LF line, final LF",
        "worlds": [{"world_id": world_id, "world_index": rows[0]["evaluator_only"]["world_index"],
                    "context_ref": rows[0]["evaluator_only"]["host"]["context_ref"],
                    "catalog_sha256": _sha(rows[0]["evaluator_only"]["catalog"]),
                    "task_ids": [t["task_id"] for t in rows],
                    "rows_sha256": hashlib.sha256(multirecord_rows(rows)).hexdigest()}
                   for world_id, rows in worlds.items()],
        "comparison_fields": {f: sum(t["evaluator_only"]["field"] == f for t in comparisons) for f in _FIELDS},
        "comparison_relations": {r: sum(t["evaluator_only"]["program"]["nodes"][2]["relation"] == r
                                        for t in comparisons) for r in _RELATIONS},
        "choice_formats": {fmt: sum(t["evaluator_only"]["choice_format"] == fmt for t in tasks)
                           for fmt in ("production", "augmented")},
        "scope": "synthetic iterative L1 preparation only; L0 gates still required; no capability or causal mediation proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--split", choices=tuple(_SPLITS), default="l1_development_train")
    parser.add_argument("--choice-format", choices=("production", "augmented", "mixed"), default="mixed")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    tasks = generate_multirecord_tasks(args.seed, args.world_count, split=args.split, choice_format=args.choice_format)
    manifest = multirecord_manifest(tasks)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "tasks.jsonl").write_bytes(multirecord_rows(tasks))
    (args.out / "manifest.json").write_bytes(_json(manifest) + b"\n")


if __name__ == "__main__":
    main()

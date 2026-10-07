"""Small synthetic factual counterfactuals, with no model or training integration.

Request files contain only decision requests. Axis metadata, labels, programs,
and request offsets are written to separate scorer files. Catalog-bound context
identities change with evidence; question text and choices do not.
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

BASE_COMMIT = "e94c79793082fc8643bae34ac52813760f1950e8"
WORLD_COUNT = 32
SEED = 2707
SPLITS = ("semantic_axis_discovery", "semantic_axis_validation")
LEVELS = ("low", "equal", "high")
TASKS = ("direct_read", "forward_compare", "reverse_compare", "explanation",
         "unrelated_read", "unrelated_compare")
FIELDS = ("price", "warranty_months")
REFS = tuple(f"record-{i}" for i in range(1, 5))
POLICY = "synthetic-public-v1"
REGISTRY = "synthetic-scalar-registry-v1"
LABEL_STATUS = "synthetically_authored_checker_valid_not_independently_validated"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "receipts/2026-10-07-L1-semantic-axis-counterfactuals"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def rows(values: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical(value) + b"\n" for value in values)


def _field(value: int, field: str) -> dict[str, Any]:
    return {"value": f"{value // 100}.{value % 100:02d}" if field == "price" else value,
            "value_type": "decimal" if field == "price" else "integer",
            "unit": "USD" if field == "price" else "month",
            "temporal_scope": "2026-10", "quantity_kind": field}


def _relation(left: int, right: int) -> str:
    return "equal" if left == right else "less_than" if left < right else "greater_than"


def _world(split: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = SPLITS.index(split)
    # Disjoint entity identities and numeric ranges; shared schema/refs are intentional.
    names = ("Beryl", "Coral", "Flint", "Mica") if domain == 0 else ("Aspen", "Birch", "Maple", "Willow")
    records = []
    for position, name in enumerate(names):
        records.append({
            "record_ref": REFS[position], "version": 1,
            "entity_ref": f"{name}{index + 1:02d}", "source_ref": "synthetic-premises",
            "audience": "public", "epistemic_class": "authoritative_structured_premise",
            "freshness": "current", "conflicts": [],
            "fields": {"price": _field(70000 + domain * 40000 + index * 401 + position * 79, "price"),
                       "warranty_months": _field(150 + domain * 200 + index * 4 + position, "warranty_months")},
        })
    catalog = {
        "schema": "ssdm.synthetic_scalar_counterfactual_catalog.v1",
        "sources": {"synthetic-premises": {"premise_admission": "accepted"}},
        "records": records,
        "snapshots": {},
        "policies": {POLICY: {"status": "current", "audience": "public", "forbidden_audiences": ["private"]}},
        "registries": {REGISTRY: {
            "allowed_ops": ["ReadFact", "Compare", "Assert", "ExplainDerivation"],
            "registered_fields": list(FIELDS),
            "field_predicates": {"price": "Price", "warranty_months": "WarrantyMonths"},
            "comparators": ["less_than", "equal", "greater_than"],
            "ask_categories": [], "style_ids": ["plain"],
        }},
    }
    host = {"policy_ref": POLICY, "registry_ref": REGISTRY, "audience": "public",
            "query_constants": {}, "unmet_obligations": [],
            "limits": {"max_program_nodes": 8, "max_questions": 1, "max_options_per_question": 8,
                       "max_proposal_calls": 1, "max_repair_calls": 0, "max_repair_attempts": 0}}
    return host, catalog


def _bind(host: dict[str, Any], catalog: dict[str, Any]) -> None:
    # Hash content with the snapshot under a fixed placeholder to avoid a hash cycle.
    catalog["snapshots"] = {"content": {"audience": "public", "record_refs": list(REFS)}}
    context = f"synthetic-scalar:{sha(catalog)}"
    catalog["snapshots"][context] = catalog["snapshots"].pop("content")
    host["context_ref"] = context
    host["provider"] = {"input_context_ref": context, "state_handle": None, "state_history_context_refs": []}


def _value(catalog: dict[str, Any], record: int, field: str) -> int:
    value = catalog["records"][record]["fields"][field]["value"]
    return int(value.replace(".", "")) if field == "price" else value


def _program(host: dict[str, Any], kind: str, field: str, left: int,
             right: int, relation: str) -> dict[str, Any]:
    nodes = [{"id": "n1", "op": "ReadFact", "record_ref": REFS[left], "field": field}]
    if "compare" in kind:
        nodes.extend([
            {"id": "n2", "op": "ReadFact", "record_ref": REFS[right], "field": field},
            {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"},
             "right": {"node": "n2", "project": "object"}, "relation": relation},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
        ])
    else:
        nodes.append({"id": "n2", "op": "Assert", "proposition": "n1"})
        if kind == "explanation":
            nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
    return {"schema": "ssdm.semantic_program.v0.1", "context_ref": host["context_ref"],
            "policy_ref": POLICY, "registry_ref": REGISTRY, "nodes": nodes,
            "outputs": [nodes[-1]["id"]], "style": "plain"}


def _answer(value: int | str, field: str, kind: str, entity: str, ref: str) -> str:
    if "compare" in kind:
        return str(value)
    descriptor = _field(value, field)
    fact = f"{entity}: {field} = {descriptor['value']} {descriptor['unit']} in 2026-10"
    if kind == "explanation":
        return f"Read {ref}.{field} from accepted synthetic-premises, assert {fact}, then explain that derivation."
    return fact


def generate_split(split: str, world_count: int = WORLD_COUNT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return model-only requests and separate scorer-only world groups.

    These are independent stateless answer probes, not a planner training export.
    The reference program and answer value are scorer-only and never serialized
    into state, question wording, IDs or proposed choices.
    """
    if split not in SPLITS or type(world_count) is not int or not 1 <= world_count <= WORLD_COUNT:
        raise ValueError("unknown split or world_count outside 1..32")
    requests, groups = [], []
    for index in range(world_count):
        base_host, base_catalog = _world(split, index)
        for field in FIELDS:
            other = FIELDS[1 - FIELDS.index(field)]
            edited = index % 4
            comparator = (edited + 1 + (index // 4) % 3) % 4
            center = _value(base_catalog, comparator, field)
            step = 1 + index % 7
            values = (center - step, center, center + step)
            world_id = f"{split}-world-{index:02d}"
            group = {"world_id": world_id, "axis_group_id": f"{world_id}:{field}",
                     "split": split, "label_status": LABEL_STATUS,
                     "axis": {"record_ref": REFS[edited], "field": field,
                              "comparator_record_ref": REFS[comparator],
                              "typed_values": {level: _field(value, field) for level, value in zip(LEVELS, values)}},
                     "variants": []}
            for level, value in zip(LEVELS, values):
                host, catalog = copy.deepcopy(base_host), copy.deepcopy(base_catalog)
                catalog["records"][edited]["fields"][field] = _field(value, field)
                _bind(host, catalog)
                variant = {"level": level, "host": host, "catalog": catalog, "tasks": []}
                for task_index, kind in enumerate(TASKS):
                    task_field = other if kind.startswith("unrelated") else field
                    left, right = (comparator, edited) if kind == "reverse_compare" else (edited, comparator)
                    entity = catalog["records"][left]["entity_ref"]
                    right_entity = catalog["records"][right]["entity_ref"]
                    if "compare" in kind:
                        text = f"Compare {entity}'s {task_field} with {right_entity}'s {task_field} in October 2026, in that order."
                        candidates = ["less_than", "equal", "greater_than"]
                        target = _relation(_value(catalog, left, task_field), _value(catalog, right, task_field))
                    else:
                        text = (f"Explain the recorded {task_field} for {entity} in October 2026."
                                if kind == "explanation" else f"What is the recorded {task_field} for {entity} in October 2026?")
                        target = _value(catalog, left, task_field)
                        candidates = list(values) + [center + 2 * step] if task_field == field else [target - 2, target - 1, target, target + 1]
                    # Seed contains task/world identity, never the variant or its target.
                    rng = random.Random(int(sha([SEED, split, index, kind]), 16))
                    rng.shuffle(candidates)
                    options = {f"o{i + 1:03d}": _answer(candidate, task_field, kind, entity, REFS[left])
                               for i, candidate in enumerate(candidates)}
                    target_id = f"o{candidates.index(target) + 1:03d}"
                    program = _program(host, kind, task_field, left, right,
                                       target if "compare" in kind else "equal")
                    checked = check_program(program, host, catalog)
                    if checked.status != "accepted":
                        raise ValueError(f"authored program rejected: {checked.reason}")
                    request, _ = decision_request(
                        task_input=text, host=host, catalog=catalog, model="synthetic-no-model",
                        parent_revision="fixture-only:v1",
                        request_id=sha([host["context_ref"], task_index, text, options])[:24],
                        question_id="answer", question="Select the answer supported by the supplied records.",
                        options=options, proposed_slot_choices=[])
                    variant["tasks"].append({"kind": kind, "field": task_field,
                                             "request_index": len(requests), "target_option_id": target_id,
                                             "answer_value": catalog["records"][left]["fields"][task_field]["value"]
                                             if "compare" not in kind else target,
                                             "program": program, "checked_output": checked.outputs[0].to_dict()})
                    requests.append(request)
                group["variants"].append(variant)
            groups.append(group)
    return requests, groups


def manifest(split: str, requests: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "ssdm.semantic_axis_fixture_manifest.v1", "split": split,
            "fixture_only": True, "label_status": LABEL_STATUS, "seed": SEED,
            "world_count": len({g["world_id"] for g in groups}), "axis_group_count": len(groups),
            "variant_count": len(groups) * len(LEVELS),
            "request_count": len(requests), "program_count": len(requests),
            "tasks_per_variant": list(TASKS), "requests_sha256": hashlib.sha256(rows(requests)).hexdigest(),
            "scorer_sha256": hashlib.sha256(rows(groups)).hexdigest(),
            "axis_groups": [{"world_id": group["world_id"], "axis_group_id": group["axis_group_id"],
                             "scorer_sha256": sha(group),
                             "catalog_sha256": [sha(v["catalog"]) for v in group["variants"]],
                             "context_refs": [v["host"]["context_ref"] for v in group["variants"]]}
                            for group in groups],
            "claims": {"factual_counterfactual_responsiveness_fixture": True,
                       "future_semantic_state_edit_support": True,
                       "learned_recurrent_mediation": False, "calibrated_ability": False,
                       "alignment": False, "internal_interpretability": False,
                       "model_result": False, "independent_or_heldout_evaluation": False},
            "model_input_rule": "Only requests JSONL entries may reach a provider; scorer files and manifests are evaluator-only.",
            "identity_rule": "One typed fact value changes; catalog-bound context and request identities change too. All task wording and choices stay fixed.",
            "no_runtime_integration": True}


def write_pack(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        requests, groups = generate_split(split)
        (output_dir / f"{split}-requests.jsonl").write_bytes(rows(requests))
        (output_dir / f"{split}-scorer.jsonl").write_bytes(rows(groups))
        (output_dir / f"{split}-manifest.json").write_bytes(canonical(manifest(split, requests, groups)) + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    write_pack(parser.parse_args().output_dir)

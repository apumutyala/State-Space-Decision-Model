"""Small synthetic factual counterfactuals, with no model or training integration.

Input files contain only task_input, permitted host and catalog. Axis metadata,
labels, programs and input offsets are written to separate scorer files. The
existing planner creates typed decision requests; no answer choices are stored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .kernel import check_program
from .realize import render_checked

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
RECEIPT_DIR = Path(__file__).resolve().parents[1] / "receipts/2026-10-07-L1-semantic-axis-counterfactuals"


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
                       "max_proposal_calls": 4, "max_repair_calls": 0, "max_repair_attempts": 0}}
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


def _targets(kind: str, field: str, left: int, right: int, relation: str) -> list[dict[str, str]]:
    """Scorer-only choices in the existing planner's fixed syntactic menu."""
    operation = "op003" if "compare" in kind else "op002" if kind == "explanation" else "op001"
    targets = [{"slot": "operation", "option_id": operation}]
    if "compare" in kind:
        targets.extend([
            {"slot": "left", "option_id": f"v{left * 2 + FIELDS.index(field) + 1:03d}"},
            {"slot": "right", "option_id": f"v{right * 2 + FIELDS.index(field) + 1:03d}"},
            {"slot": "relation", "option_id": f"r{('less_than', 'equal', 'greater_than').index(relation) + 1:03d}"},
        ])
    else:
        targets.append({"slot": "fact", "option_id": f"f{left * 2 + FIELDS.index(field) + 1:03d}"})
    return targets


def generate_split(split: str, world_count: int = WORLD_COUNT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return planner inputs and separate scorer-only world groups.

    Feed task_input, host and catalog to plan_checked_answer. Its provider selects
    typed slots; only the existing kernel and renderer can release the result.
    Reference programs, rendered targets and slot choices remain scorer-only.
    """
    if split not in SPLITS or type(world_count) is not int or not 1 <= world_count <= WORLD_COUNT:
        raise ValueError("unknown split or world_count outside 1..32")
    inputs, groups = [], []
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
                        target = _relation(_value(catalog, left, task_field), _value(catalog, right, task_field))
                    else:
                        text = (f"Explain the recorded {task_field} for {entity} in October 2026."
                                if kind == "explanation" else f"What is the recorded {task_field} for {entity} in October 2026?")
                        target = _value(catalog, left, task_field)
                    program = _program(host, kind, task_field, left, right,
                                       target if "compare" in kind else "equal")
                    checked = check_program(program, host, catalog)
                    if checked.status != "accepted":
                        raise ValueError(f"authored program rejected: {checked.reason}")
                    variant["tasks"].append({"kind": kind, "field": task_field,
                                             "input_index": len(inputs),
                                             "targets": _targets(kind, task_field, left, right,
                                                                 target if "compare" in kind else "equal"),
                                             "answer_value": catalog["records"][left]["fields"][task_field]["value"]
                                             if "compare" not in kind else target,
                                             "program": program, "checked_output": checked.outputs[0].to_dict(),
                                             "rendered_text": render_checked(checked, host, catalog)})
                    inputs.append({"task_input": text, "host": copy.deepcopy(host),
                                   "catalog": copy.deepcopy(catalog)})
                group["variants"].append(variant)
            groups.append(group)
    return inputs, groups


def manifest(split: str, inputs: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "ssdm.semantic_axis_fixture_manifest.v1", "split": split,
            "fixture_only": True, "label_status": LABEL_STATUS, "seed": SEED,
            "world_count": len({g["world_id"] for g in groups}), "axis_group_count": len(groups),
            "variant_count": len(groups) * len(LEVELS),
            "input_count": len(inputs), "program_count": len(inputs),
            "tasks_per_variant": list(TASKS), "inputs_sha256": hashlib.sha256(rows(inputs)).hexdigest(),
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
            "model_input_rule": "Only inputs JSONL task_input/host/catalog may enter plan_checked_answer; scorer files and manifests are evaluator-only.",
            "identity_rule": "One typed fact value changes; catalog-bound context identities change too. Task wording and planner-generated typed menus stay fixed.",
            "planner_entry_point": "language_growth.planner.plan_checked_answer",
            "learned_provider_or_recurrent_edit_implemented": False}


def write_pack(output_dir: Path) -> None:
    """Publish the complete bundle to a fresh path; never overwrite evidence."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")
    bundle = {}
    for split in SPLITS:
        inputs, groups = generate_split(split)
        bundle[f"{split}-inputs.jsonl"] = rows(inputs)
        bundle[f"{split}-scorer.jsonl"] = rows(groups)
        bundle[f"{split}-manifest.json"] = canonical(manifest(split, inputs, groups)) + b"\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        for name, content in bundle.items():
            (staging / name).write_bytes(content)
        if output_dir.exists():
            raise FileExistsError(f"output path appeared during generation: {output_dir}")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            # Only this invocation's flat temporary bundle is reclaimed.
            for name in bundle:
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh output directory, preferably outside the checkout; existing paths are refused.")
    write_pack(parser.parse_args().output_dir)

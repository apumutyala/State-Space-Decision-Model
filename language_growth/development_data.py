"""Seeded synthetic DEVELOPMENT examples for the first typed L0 language pilot.

Each ``model_inputs`` entry is one stateless legacy D request. Current-slot
targets and checked reference programs live only under ``training_only``.
These examples are generated from local synthetic values, never audit data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from decimal import Decimal
from typing import Any

from .kernel import check_program
from .parent_bridge import decision_request

_NAMES = ("Aster", "Bracken", "Cinder", "Dovetail", "Ember", "Fallow",
          "Grove", "Hearth", "Indigo", "Juniper", "Kestrel", "Lumen")
_OPS = (("read", "Assert one selected fact"),
        ("explain", "Explain a checked assertion of one selected fact"),
        ("compare", "Compare two selected typed values and assert the selected relation"))
_RELATIONS = ("greater_than", "less_than", "equal")
_CONTEXT = "synthetic-development-context-v1"
_POLICY = "synthetic-development-public-v1"
_REGISTRY = "synthetic-development-registry-v1"


def _rng(seed: int, index: int, domain: str) -> random.Random:
    material = f"ssdm-development-v1:{seed}:{index}:{domain}".encode("ascii")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _catalog(names: tuple[str, str], warranties: tuple[int, int],
             prices: tuple[int, int]) -> dict[str, Any]:
    records = []
    for index, name in enumerate(names):
        records.append({
            "record_ref": f"development-record-{index + 1}", "version": 1,
            "entity_ref": name, "source_ref": "synthetic-premises",
            "audience": "public", "epistemic_class": "authoritative_structured_premise",
            "freshness": "current", "conflicts": [],
            "fields": {
                "warranty_months": {"value": warranties[index], "value_type": "integer",
                                    "unit": "month", "temporal_scope": "2026-10",
                                    "quantity_kind": "warranty_months"},
                "price": {"value": f"{prices[index] // 100}.{prices[index] % 100:02d}", "value_type": "decimal",
                          "unit": "USD", "temporal_scope": "2026-10", "quantity_kind": "price"},
            },
        })
    return {
        "schema": "ssdm.synthetic_development_catalog.v1",
        "sources": {"synthetic-premises": {"premise_admission": "accepted"}},
        "records": records,
        "snapshots": {_CONTEXT: {"audience": "public", "record_refs": [r["record_ref"] for r in records]}},
        "policies": {_POLICY: {"status": "current", "audience": "public", "forbidden_audiences": ["private"]}},
        "registries": {_REGISTRY: {
            "allowed_ops": ["ReadFact", "Compare", "Assert", "ExplainDerivation"],
            "registered_fields": ["price", "warranty_months"],
            "field_predicates": {"price": "Price", "warranty_months": "WarrantyMonths"},
            "comparators": list(_RELATIONS), "ask_categories": [], "style_ids": ["plain"],
        }},
    }


def _host() -> dict[str, Any]:
    return {
        "context_ref": _CONTEXT, "policy_ref": _POLICY, "registry_ref": _REGISTRY,
        "audience": "public", "query_constants": {}, "unmet_obligations": [],
        "limits": {"max_program_nodes": 8, "max_questions": 1,
                   "max_options_per_question": 16, "max_proposal_calls": 4,
                   "max_repair_calls": 0, "max_repair_attempts": 0},
        "provider": {"input_context_ref": _CONTEXT, "state_handle": None,
                     "state_history_context_refs": []},
    }


def _options(items: list[tuple[str, str]], prefix: str, rng: random.Random,
             *, augment: bool) -> tuple[dict[str, str], dict[str, str]]:
    shuffled = list(items)
    if augment:
        rng.shuffle(shuffled)
    options = {f"{prefix}{index:03d}": description for index, (_, description) in enumerate(shuffled, 1)}
    target_ids = {key: f"{prefix}{index:03d}" for index, (key, _) in enumerate(shuffled, 1)}
    return options, target_ids


def _program(kind: str, field: str, fact_index: int, relation: str | None) -> dict[str, Any]:
    if kind == "compare":
        nodes = [
            {"id": "n1", "op": "ReadFact", "record_ref": "development-record-1", "field": "price"},
            {"id": "n2", "op": "ReadFact", "record_ref": "development-record-2", "field": "price"},
            {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"},
             "right": {"node": "n2", "project": "object"}, "relation": relation},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
        ]
    else:
        nodes = [
            {"id": "n1", "op": "ReadFact", "record_ref": f"development-record-{fact_index + 1}", "field": field},
            {"id": "n2", "op": "Assert", "proposition": "n1"},
        ]
        if kind == "explain":
            nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
    return {"schema": "ssdm.semantic_program.v0.1", "context_ref": _CONTEXT,
            "policy_ref": _POLICY, "registry_ref": _REGISTRY,
            "nodes": nodes, "outputs": [nodes[-1]["id"]], "style": "plain"}


def _relation(prices: tuple[int, int]) -> str:
    return "greater_than" if prices[0] > prices[1] else "less_than"


def _build(seed: int, index: int, names: tuple[str, str], warranties: tuple[int, int],
           prices: tuple[int, int], choice_format: str) -> dict[str, Any]:
    kind = _OPS[index % len(_OPS)][0]
    rng = _rng(seed, index, "choice-order")
    wording = _rng(seed, index, "wording")
    field = wording.choice(("price", "warranty_months"))
    fact_index = wording.randrange(2)
    if kind == "compare":
        task_input = wording.choice((
            f"Compare the October prices of {names[0]} and {names[1]}.",
            f"How do {names[0]}'s and {names[1]}'s October prices compare?",
        ))
    elif kind == "explain":
        task_input = f"Explain the checked {field.replace('_', ' ')} fact for {names[fact_index]}."
    else:
        task_input = f"What is {names[fact_index]}'s {field.replace('_', ' ')} in October 2026?"
    catalog, host = _catalog(names, warranties, prices), _host()
    program = _program(kind, field, fact_index, _relation(prices) if kind == "compare" else None)
    # Context identity changes with evidence, including counterfactual worlds.
    digest = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    context_ref = f"{_CONTEXT}:{digest}"
    catalog["snapshots"][context_ref] = catalog["snapshots"].pop(_CONTEXT)
    host["context_ref"] = context_ref
    host["provider"]["input_context_ref"] = context_ref
    program["context_ref"] = context_ref
    checked = check_program(program, host, catalog)
    if checked.status != "accepted":
        raise ValueError(f"synthetic program rejected: {checked.reason}")

    augment = choice_format == "augmented"
    op_options, op_ids = _options(list(_OPS), "op", rng, augment=augment)
    facts = [(f"{record}-{field_name}", f"Fact development-record-{record} field {field_name}")
             for record in (1, 2) for field_name in ("price", "warranty_months")]
    fact_options, fact_ids = _options(facts, "f", rng, augment=augment)
    operand_options, operand_ids = _options(facts, "v", rng, augment=augment)
    relation_options, relation_ids = _options([(rel, f"Relation {rel}") for rel in _RELATIONS],
                                               "r", rng, augment=augment)
    choices: list[tuple[str, str, dict[str, str], str]] = [
        ("operation", "Select the semantic operation for the user's task.", op_options, op_ids[kind]),
    ]
    if kind == "compare":
        choices.extend([
            ("left", "Select the left typed operand.", operand_options, operand_ids["1-price"]),
            ("right", "Select the right typed operand.", operand_options, operand_ids["2-price"]),
            ("relation", "Select the comparison relation.", relation_options, relation_ids[_relation(prices)]),
        ])
    else:
        choices.append(("fact", "Select the fact to assert.", fact_options,
                        fact_ids[f"{fact_index + 1}-{field}"]))
    model_inputs = []
    targets = []
    prior = []
    for slot_index, (slot, question, options, target) in enumerate(choices, 1):
        request_id = hashlib.sha256(
            f"ssdm-development-request:{seed}:{index}:{slot_index}:{digest}".encode("ascii")
        ).hexdigest()[:24]
        request, _ = decision_request(
            task_input=task_input, host=host, catalog=catalog,
            model="development-only", parent_revision="synthetic-no-model:v1",
            request_id=request_id,
            question_id=f"slot-{slot_index:03d}-{slot}", question=question,
            options=options, proposed_slot_choices=prior,
        )
        model_inputs.append(request)
        targets.append({"slot": slot, "option_id": target})
        prior.append({"slot": slot, "option_id": target, "description": options[target]})
    return {
        "development_only": True, "task_id": f"synthetic-{seed}-{index}",
        "model_inputs": model_inputs,
        "training_only": {"kind": kind, "choice_format": choice_format,
                          "host": host, "catalog": catalog,
                          "program": program, "targets": targets},
    }


def generate_development_tasks(seed: int, count: int, *, choice_format: str = "mixed") -> list[dict[str, Any]]:
    """Generate only synthetic development tasks; no external data is read."""
    if type(seed) is not int or seed < 0 or type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("seed must be nonnegative and count must be 1..10000")
    if choice_format not in ("production", "augmented", "mixed"):
        raise ValueError("choice_format must be production, augmented, or mixed")
    tasks = []
    for index in range(count):
        names = tuple(_rng(seed, index, "names").sample(_NAMES, 2))
        values = _rng(seed, index, "values")
        warranties = (values.randint(1, 48), values.randint(1, 48))
        price_a = values.randint(5000, 25000)
        price_b = values.randint(5000, 25000)
        while price_b == price_a:
            price_b = values.randint(5000, 25000)
        fmt = ("production" if index % 2 == 0 else "augmented") if choice_format == "mixed" else choice_format
        tasks.append(_build(seed, index, names, warranties, (price_a, price_b), fmt))
    return tasks


def price_swap_counterfactual(example: dict[str, Any]) -> dict[str, Any]:
    """Development-only comparison with the two prices exchanged."""
    if example.get("training_only", {}).get("kind") != "compare":
        raise ValueError("comparison example required")
    original = example["training_only"]["catalog"]["records"]
    names = (original[0]["entity_ref"], original[1]["entity_ref"])
    warranties = tuple(r["fields"]["warranty_months"]["value"] for r in original)
    prices = tuple(int(Decimal(r["fields"]["price"]["value"]) * 100) for r in original)
    ident = example["task_id"].split("-")
    seed, index = int(ident[-2]), int(ident[-1])
    swapped = _build(seed, index, names, warranties, (prices[1], prices[0]),
                     example["training_only"]["choice_format"])
    swapped["task_id"] = f"{example['task_id']}-price-swap"
    return swapped

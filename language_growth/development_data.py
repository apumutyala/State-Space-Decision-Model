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

_SPLITS = {
    "development_train": {
        "names": ("Aster", "Bracken", "Cinder", "Dovetail", "Ember", "Fallow"),
        "warranty_range": (1, 36), "price_cents_range": (5000, 20000),
        "templates": {
            "read": ("What is {entity}'s {field} in October 2026?",
                     "Give the October 2026 {field} for {entity}."),
            "explain": ("Explain the checked {field} fact for {entity}.",
                        "Show the checked derivation for {entity}'s {field}."),
            "compare": ("Compare the October {field} of {left} and {right}.",
                        "How do {left}'s and {right}'s October {field} compare?"),
        },
    },
    "development_validation": {
        "names": ("Grove", "Hearth", "Indigo", "Juniper", "Kestrel", "Lumen"),
        "warranty_range": (37, 60), "price_cents_range": (20001, 30000),
        "templates": {
            "read": ("State {entity}'s {field} for October 2026.",),
            "explain": ("Provide a checked derivation of {entity}'s {field}.",),
            "compare": ("For October 2026, compare {left}'s and {right}'s {field}.",),
        },
    },
}
_CLARIFICATION_TEMPLATES = {
    "development_train": {
        "ask_missing": ("What is {entity}'s {field} in October 2026? Ask for the needed input if it is absent.",
                        "Give {entity}'s October 2026 {field}, or request the missing input."),
        "read_with_unrelated_obligation": ("What is {entity}'s {field} in October 2026? Ask for the needed input if it is absent.",
                                           "Give {entity}'s October 2026 {field}, or request the missing input."),
    },
    "development_validation": {
        "ask_missing": ("For October 2026, give {entity}'s {field}; request the missing evidence if needed.",),
        "read_with_unrelated_obligation": ("For October 2026, give {entity}'s {field}; request the missing evidence if needed.",),
    },
}
_OPS = (("read", "Assert one selected fact"),
        ("explain", "Explain a checked assertion of one selected fact"),
        ("compare", "Compare two selected typed values and assert the selected relation"))
_CLARIFICATION_KINDS = ("read", "explain", "compare", "ask_missing", "read_with_unrelated_obligation")
_RELATIONS = ("greater_than", "less_than", "equal")
_CONTEXT = "synthetic-development-context-v1"
_POLICY = "synthetic-development-public-v1"
_REGISTRY = "synthetic-development-registry-v1"


def _rng(split: str, seed: int, index: int, domain: str) -> random.Random:
    material = f"ssdm-development-v2:{split}:{seed}:{index}:{domain}".encode("ascii")
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


def _program(kind: str, field: str, fact_index: int, relation: str | None,
             left_index: int, ask_target: tuple[str, str] | None = None) -> dict[str, Any]:
    if kind == "ask_missing":
        if ask_target is None:
            raise ValueError("missing Ask target")
        obligation_id, reference = ask_target
        nodes = [{"id": "n1", "op": "Ask", "category": "missing_catalog_field",
                  "obligation_ref": obligation_id, "references": [reference]}]
    elif kind == "compare":
        nodes = [
            {"id": "n1", "op": "ReadFact", "record_ref": f"development-record-{left_index + 1}", "field": field},
            {"id": "n2", "op": "ReadFact", "record_ref": f"development-record-{2 - left_index}", "field": field},
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


def _relation(values: tuple[int, int], left_index: int) -> str:
    left, right = values[left_index], values[1 - left_index]
    return "equal" if left == right else ("greater_than" if left > right else "less_than")


def _build(seed: int, index: int, names: tuple[str, str], warranties: tuple[int, int],
           prices: tuple[int, int], choice_format: str, split: str,
           allow_one_repair: bool, clarification_mode: bool = False) -> dict[str, Any]:
    kind = (_CLARIFICATION_KINDS[index % len(_CLARIFICATION_KINDS)] if clarification_mode
            else _OPS[index % len(_OPS)][0])
    spec = _SPLITS[split]
    rng = _rng(split, seed, index, "choice-order")
    wording = _rng(split, seed, index, "wording")
    field = wording.choice(("price", "warranty_months"))
    fact_index = wording.randrange(2)
    missing_pairs = None
    if kind in ("ask_missing", "read_with_unrelated_obligation"):
        pairs = [(record_index, field_name) for record_index in range(2)
                 for field_name in ("price", "warranty_months")]
        # The visible catalog layout is independent of which operation is correct.
        missing_pairs = _rng(split, seed, index, "missing-pairs").sample(pairs, 2)
        available = (missing_pairs if kind == "ask_missing"
                     else [pair for pair in pairs if pair not in missing_pairs])
        fact_index, field = _rng(split, seed, index, "requested-pair").choice(available)
    left_index = _rng(split, seed, index, "named-order").randrange(2)
    templates = spec["templates"].get(kind) or _CLARIFICATION_TEMPLATES[split][kind]
    template_index = wording.randrange(len(templates))
    template = templates[template_index]
    field_text = field.replace("_", " ")
    if kind == "compare":
        task_input = template.format(left=names[left_index], right=names[1 - left_index], field=field_text)
    else:
        task_input = template.format(entity=names[fact_index], field=field_text)
    catalog, host = _catalog(names, warranties, prices), _host()
    ask_target = None
    if clarification_mode:
        registry = catalog["registries"][_REGISTRY]
        registry["allowed_ops"].append("Ask")
        registry["ask_categories"] = ["missing_catalog_field"]
        if missing_pairs is not None:
            # Both arms expose two real obligations. Ask depends on the requested pair,
            # not on the count or arrangement of missing fields alone.
            _rng(split, seed, index, "obligation-order").shuffle(missing_pairs)
            for number, (record_index, missing_field) in enumerate(missing_pairs, 1):
                del catalog["records"][record_index]["fields"][missing_field]
                reference, obligation_id = f"qref{number:03d}", f"ob{number:03d}"
                host["query_constants"][reference] = {"entity_ref": names[record_index], "field": missing_field}
                host["unmet_obligations"].append({"obligation_id": obligation_id,
                                                   "category": "missing_catalog_field",
                                                   "allowed_refs": [reference]})
                if (record_index, missing_field) == (fact_index, field):
                    ask_target = (obligation_id, reference)
    if allow_one_repair:
        host["limits"].update(max_proposal_calls=8, max_repair_calls=4, max_repair_attempts=1)
    compare_values = prices if field == "price" else warranties
    program = _program(kind, field, fact_index,
                       _relation(compare_values, left_index) if kind == "compare" else None,
                       left_index, ask_target)
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
    operations = list(_OPS)
    if host["unmet_obligations"]:
        operations.append(("ask", "Ask for a registered missing input"))
    op_options, op_ids = _options(operations, "op", rng, augment=augment)
    op_kind = "ask" if kind == "ask_missing" else "read" if kind == "read_with_unrelated_obligation" else kind
    facts = [(f"{record}-{field_name}", f"Fact development-record-{record} field {field_name}")
             for record in (1, 2) for field_name in ("price", "warranty_months")
             if field_name in catalog["records"][record - 1]["fields"]]
    fact_options, fact_ids = _options(facts, "f", rng, augment=augment)
    operand_options, operand_ids = _options(facts, "v", rng, augment=augment)
    relation_options, relation_ids = _options([(rel, f"Relation {rel}") for rel in _RELATIONS],
                                               "r", rng, augment=augment)
    choices: list[tuple[str, str, dict[str, str], str]] = [
        ("operation", "Select the semantic operation for the user's task.", op_options, op_ids[op_kind]),
    ]
    if kind == "compare":
        choices.extend([
            ("left", "Select the left typed operand.", operand_options, operand_ids[f"{left_index + 1}-{field}"]),
            ("right", "Select the right typed operand.", operand_options, operand_ids[f"{2 - left_index}-{field}"]),
            ("relation", "Select the comparison relation.", relation_options,
             relation_ids[_relation(compare_values, left_index)]),
        ])
    elif kind == "ask_missing":
        obligations = host["unmet_obligations"]
        obligation_options, obligation_ids = _options(
            [(ob["obligation_id"], f"Obligation {ob['obligation_id']} category {ob['category']}")
             for ob in obligations], "o", rng, augment=augment)
        choices.append(("obligation", "Select the unmet host obligation.", obligation_options,
                        obligation_ids[ask_target[0]]))
        reference_options, reference_ids = _options(
            [(ask_target[1], f"Requested reference {ask_target[1]}")], "a", rng, augment=augment)
        choices.append(("ask_reference", "Select the requested reference.", reference_options,
                        reference_ids[ask_target[1]]))
    else:
        choices.append(("fact", "Select the fact to assert.", fact_options,
                        fact_ids[f"{fact_index + 1}-{field}"]))
    model_inputs = []
    targets = []
    prior = []
    for slot_index, (slot, question, options, target) in enumerate(choices, 1):
        request_id = hashlib.sha256(
            f"ssdm-development-request:{split}:{seed}:{index}:{slot_index}:{digest}".encode("ascii")
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
    training_only = {"kind": kind, "choice_format": choice_format,
                     "seed": seed, "index": index, "template_id": f"{split}:{kind}:{template_index}",
                     "allow_one_repair": allow_one_repair,
                     "host": host, "catalog": catalog,
                     "program": program, "targets": targets}
    if clarification_mode:
        training_only["clarification_mode"] = True
    return {
        "development_only": True, "development_split": split,
        "task_id": f"synthetic-{split}-{seed}-{index}",
        "model_inputs": model_inputs,
        "training_only": training_only,
    }


def generate_development_tasks(seed: int, count: int, *, choice_format: str = "mixed",
                               split: str = "development_train",
                               allow_one_repair: bool = False,
                               clarification_mode: bool = False) -> list[dict[str, Any]]:
    """Generate one iterative development split; no audit/test data is read."""
    if type(seed) is not int or seed < 0 or type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("seed must be nonnegative and count must be 1..10000")
    if choice_format not in ("production", "augmented", "mixed"):
        raise ValueError("choice_format must be production, augmented, or mixed")
    if split not in _SPLITS or type(allow_one_repair) is not bool or type(clarification_mode) is not bool:
        raise ValueError("invalid development split or repair setting")
    spec = _SPLITS[split]
    tasks = []
    for index in range(count):
        names = tuple(_rng(split, seed, index, "names").sample(spec["names"], 2))
        values = _rng(split, seed, index, "values")
        warranty_lo, warranty_hi = spec["warranty_range"]
        price_lo, price_hi = spec["price_cents_range"]
        warranties = (values.randint(warranty_lo, warranty_hi), values.randint(warranty_lo, warranty_hi))
        price_a = values.randint(price_lo, price_hi)
        price_b = values.randint(price_lo, price_hi)
        while price_b == price_a:
            price_b = values.randint(price_lo, price_hi)
        # Equality is a real skill, not merely an always-wrong offered option.
        if _rng(split, seed, index, "equality-case").randrange(3) == 0:
            price_b = price_a
        fmt = ("production" if index % 2 == 0 else "augmented") if choice_format == "mixed" else choice_format
        tasks.append(_build(seed, index, names, warranties, (price_a, price_b), fmt,
                            split, allow_one_repair, clarification_mode))
    return tasks


def development_split_spec(split: str) -> dict[str, Any]:
    """Declared, disjoint iterative development domains; no audit split exists here."""
    if split not in _SPLITS:
        raise ValueError("unknown development split")
    return copy.deepcopy(_SPLITS[split])


def development_manifest(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash exact generated rows and identify the split used to produce them."""
    if not tasks or not isinstance(tasks, list):
        raise ValueError("nonempty development task list required")
    first = tasks[0]
    split = first["development_split"]
    spec = development_split_spec(split)
    seed = first["training_only"]["seed"]
    repair = first["training_only"]["allow_one_repair"]
    clarification_mode = first["training_only"].get("clarification_mode", False)
    if any(task["development_split"] != split or task["training_only"]["seed"] != seed
           or task["training_only"]["allow_one_repair"] != repair
           or task["training_only"].get("clarification_mode", False) != clarification_mode for task in tasks):
        raise ValueError("mixed development domains")
    rows = "".join(json.dumps(task, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False) + "\n" for task in tasks).encode("ascii")
    comparisons = [task for task in tasks if task["training_only"]["kind"] == "compare"]
    manifest = {
        "schema": "ssdm.synthetic_development_manifest.v2",
        "split": split, "seed": seed, "count": len(tasks),
        "allow_one_repair": repair,
        "choice_formats": {fmt: sum(task["training_only"]["choice_format"] == fmt for task in tasks)
                           for fmt in ("production", "augmented")},
        "name_pool": list(spec["names"]),
        "warranty_range": list(spec["warranty_range"]),
        "price_cents_range": list(spec["price_cents_range"]),
        "template_ids": {kind: [f"{split}:{kind}:{index}" for index in range(len(options))]
                         for kind, options in spec["templates"].items()},
        "comparison_fields": {field: sum(task["training_only"]["program"]["nodes"][0]["field"] == field
                                        for task in comparisons) for field in ("price", "warranty_months")},
        "comparison_left_records": {
            ref: sum(task["training_only"]["program"]["nodes"][0]["record_ref"] == ref
                     for task in comparisons)
            for ref in ("development-record-1", "development-record-2")
        },
        "comparison_relations": {relation: sum(task["training_only"]["program"]["nodes"][2]["relation"] == relation
                                              for task in comparisons) for relation in _RELATIONS},
        "sample_sha256": hashlib.sha256(rows).hexdigest(),
        "sample_encoding": "sorted compact ensure_ascii JSON per LF line, final LF",
        "scope": "iterative synthetic development only; no audit or held-out generalization claim",
    }
    if clarification_mode:
        manifest["clarification_mode"] = True
        manifest["kind_counts"] = {kind: sum(task["training_only"]["kind"] == kind for task in tasks)
                                   for kind in _CLARIFICATION_KINDS}
        manifest["template_ids"].update({
            kind: [f"{split}:{kind}:{index}" for index in range(len(templates))]
            for kind, templates in _CLARIFICATION_TEMPLATES[split].items()
        })
    return manifest


def comparison_swap_counterfactual(example: dict[str, Any]) -> dict[str, Any]:
    """Swap the selected comparison field's values in a development world."""
    if example.get("training_only", {}).get("kind") != "compare":
        raise ValueError("comparison example required")
    original = example["training_only"]["catalog"]["records"]
    names = (original[0]["entity_ref"], original[1]["entity_ref"])
    warranties = tuple(r["fields"]["warranty_months"]["value"] for r in original)
    prices = tuple(int(Decimal(r["fields"]["price"]["value"]) * 100) for r in original)
    field = example["training_only"]["program"]["nodes"][0]["field"]
    if field == "price":
        prices = (prices[1], prices[0])
    elif field == "warranty_months":
        warranties = (warranties[1], warranties[0])
    else:
        raise ValueError("unsupported comparison field")
    seed, index = example["training_only"]["seed"], example["training_only"]["index"]
    swapped = _build(seed, index, names, warranties, prices,
                     example["training_only"]["choice_format"], example["development_split"],
                     example["training_only"]["allow_one_repair"],
                     example["training_only"].get("clarification_mode", False))
    swapped["task_id"] = f"{example['task_id']}-comparison-swap"
    return swapped


def price_swap_counterfactual(example: dict[str, Any]) -> dict[str, Any]:
    """Compatibility alias; also swaps warranty values for warranty comparisons."""
    return comparison_swap_counterfactual(example)

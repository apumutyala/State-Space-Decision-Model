"""Frozen synthetic read/explain pairs for the L0 candidate-name screen.

Gold programs remain evaluator-only. Model requests are generated from the
visible host/catalog and contain no target or training metadata.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .kernel import check_program
from .parent_bridge import decision_request

_BASE_NAMES = (
    "Amber Kettle", "Birch Cabinet", "Copper Monitor", "Dawn Lantern",
    "Elm Router", "Fern Compass", "Gale Mixer", "Harbor Sensor",
    "Ivory Notebook", "Jade Speaker", "Keen Thermostat", "Lunar Battery",
    "Moss Blender", "Noble Projector", "Opal Microscope", "Pine Turbine",
    "Quiet Valve", "Ruby Telescope", "Silver Pump", "Tidal Workbench",
    "Umber Satellite", "Violet Charger", "Willow Keyboard", "Xenon Gateway",
    "Young Watercraft", "Zephyr Scanner", "Maple Instrument", "Quartz Lantern",
    "Cobalt Heater", "Meadow Console", "Velvet Generator", "Prairie Clock",
)
_RENAMED_NAMES = (
    "Arctic Kettle", "Bramble Cabinet", "Citron Monitor", "Distant Lantern",
    "Evergreen Router", "Frost Compass", "Granite Mixer", "Highland Sensor",
    "Larch Notebook", "Moraine Speaker", "Nectar Thermostat", "Lattice Battery",
    "Marble Blender", "Northern Projector", "Olive Microscope", "Pioneer Turbine",
    "Quarry Valve", "Raven Telescope", "Summit Pump", "Timber Workbench",
    "Upland Satellite", "Warden Charger", "Windward Keyboard", "Yellowstone Gateway",
    "Zenith Watercraft", "Alder Scanner", "Boreal Instrument", "Crescent Lantern",
    "Driftwood Heater", "Estuary Console", "Fable Generator", "Glimmer Clock",
)
_READ_TEMPLATES = (
    "What is {entity}'s {field}?",
    "Report the current {field} for {entity}.",
)
_EXPLAIN_TEMPLATES = (
    "Explain the supported {field} for {entity}.",
    "Why does the evidence support {entity}'s {field}?",
)
_OP_OPTIONS = {
    "op001": "Assert one selected fact",
    "op002": "Explain a checked assertion of one selected fact",
    "op003": "Compare two selected typed values and assert the selected relation",
}
_POLICY = "binding-screen-public-v1"
_REGISTRY = "binding-screen-registry-v1"
_SOURCE = "binding-screen-premises-v1"


def _catalog(names: tuple[str, str], prices: tuple[int, int], warranties: tuple[int, int],
             order: tuple[int, int]) -> dict[str, Any]:
    records = []
    for index in order:
        record_ref = f"binding-screen-record-{index + 1}"
        records.append({
            "record_ref": record_ref,
            "version": 1,
            "entity_ref": names[index],
            "source_ref": _SOURCE,
            "audience": "public",
            "epistemic_class": "authoritative_structured_premise",
            "freshness": "current",
            "conflicts": [],
            "fields": {
                "price": {"value": f"{prices[index] // 100}.{prices[index] % 100:02d}",
                          "value_type": "decimal", "unit": "USD", "temporal_scope": "2026-10",
                          "quantity_kind": "price"},
                "warranty_months": {"value": warranties[index], "value_type": "integer",
                                     "unit": "month", "temporal_scope": "2026-10",
                                     "quantity_kind": "warranty_months"},
            },
        })
    return {
        "schema": "ssdm.synthetic_binding_screen_catalog.v1",
        "sources": {_SOURCE: {"premise_admission": "accepted"}},
        "records": records,
        "snapshots": {"binding-screen-context": {"audience": "public",
            "record_refs": [f"binding-screen-record-{i + 1}" for i in order]}},
        "policies": {_POLICY: {"status": "current", "audience": "public",
                               "forbidden_audiences": ["private"]}},
        "registries": {_REGISTRY: {
            "allowed_ops": ["ReadFact", "Assert", "ExplainDerivation", "Compare"],
            "registered_fields": ["price", "warranty_months"],
            "field_predicates": {"price": "Price", "warranty_months": "WarrantyMonths"},
            "comparators": ["less_than", "equal", "greater_than"],
            "ask_categories": [], "style_ids": ["plain"],
        }},
    }


def _host(context_ref: str) -> dict[str, Any]:
    return {
        "context_ref": context_ref, "policy_ref": _POLICY, "registry_ref": _REGISTRY,
        "audience": "public", "query_constants": {}, "unmet_obligations": [],
        "limits": {"max_program_nodes": 8, "max_questions": 1,
                   "max_options_per_question": 16, "max_proposal_calls": 4,
                   "max_repair_calls": 0, "max_repair_attempts": 0},
        "provider": {"input_context_ref": context_ref, "state_handle": None,
                     "state_history_context_refs": []},
    }


def _program(kind: str, record_ref: str, field: str, context_ref: str) -> dict[str, Any]:
    nodes = [
        {"id": "n1", "op": "ReadFact", "record_ref": record_ref, "field": field},
        {"id": "n2", "op": "Assert", "proposition": "n1"},
    ]
    output = "n2"
    if kind == "explain":
        nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
        output = "n3"
    return {"schema": "ssdm.semantic_program.v0.1", "context_ref": context_ref,
            "policy_ref": _POLICY, "registry_ref": _REGISTRY,
            "nodes": nodes, "outputs": [output], "style": "plain"}


def _member(seed: int, world: int, member: int, kind: str, field: str, target_index: int,
            names: tuple[str, str], prices: tuple[int, int], warranties: tuple[int, int],
            order: tuple[int, int]) -> dict[str, Any]:
    catalog = _catalog(names, prices, warranties, order)
    catalog_digest = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    context_ref = f"binding-screen-context:{catalog_digest}"
    catalog["snapshots"][context_ref] = catalog["snapshots"].pop("binding-screen-context")
    host = _host(context_ref)
    record_ref = f"binding-screen-record-{target_index + 1}"
    entity = names[target_index]
    template_set = _READ_TEMPLATES if kind == "read" else _EXPLAIN_TEMPLATES
    template = template_set[(world + member) % len(template_set)]
    task_input = template.format(entity=entity, field=field.replace("_", " "))
    program = _program(kind, record_ref, field, context_ref)
    if check_program(program, host, catalog).status != "accepted":
        raise ValueError("binding-screen reference program did not pass the checker")
    request_id = hashlib.sha256(f"binding-screen:{seed}:{world}:{member}".encode("ascii")).hexdigest()[:24]
    request, _ = decision_request(task_input=task_input, host=host, catalog=catalog,
        model="development-only", parent_revision="binding-screen-no-model:v1",
        request_id=request_id, question_id="slot-001-operation",
        question="Select the semantic operation for the user's task.",
        options=_OP_OPTIONS, proposed_slot_choices=[])
    return {
        "development_only": True,
        "development_split": "development_binding_screen",
        "task_id": f"binding-screen-{seed}-{world:02d}-{member}",
        "pair_id": f"binding-screen-{seed}-{world:02d}",
        "pair_member": member,
        "model_inputs": [request],
        "training_only": {"kind": kind, "field": field, "target_record_ref": record_ref,
                          "seed": seed, "world_id": f"binding-screen-{seed}-{world:02d}",
                          "member": member, "template_id": f"binding-screen:{kind}:{member}",
                          "allow_one_repair": False, "host": host, "catalog": catalog,
                          "program": program, "targets": [{"slot": "fact", "record_ref": record_ref,
                                                              "field": field}]},
    }


def generate_binding_screen(seed: int = 20261007, worlds: int = 16) -> list[dict[str, Any]]:
    """Generate 16 read/explain worlds, each with a renamed/reordered partner."""
    if type(seed) is not int or seed < 0 or type(worlds) is not int or worlds != 16:
        raise ValueError("binding screen is frozen at a nonnegative seed and 16 worlds")
    rng = random.Random(seed)
    out = []
    for world in range(worlds):
        kind = "read" if world < 8 else "explain"
        field = "price" if world % 4 in (0, 1) else "warranty_months"
        target_index = world % 2
        prices = (rng.randint(12500, 92500), rng.randint(12500, 92500))
        warranties = (rng.randint(61, 120), rng.randint(61, 120))
        base_names = (_BASE_NAMES[2 * world], _BASE_NAMES[2 * world + 1])
        renamed_names = (_RENAMED_NAMES[2 * world], _RENAMED_NAMES[2 * world + 1])
        out.append(_member(seed, world, 0, kind, field, target_index,
                           base_names, prices, warranties, (0, 1)))
        out.append(_member(seed, world, 1, kind, field, target_index,
                           renamed_names, prices, warranties, (1, 0)))
    return out

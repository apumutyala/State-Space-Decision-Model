"""Offline, slot-specific supervision from typed gold programs only.

This module never parses options, candidate IDs, request text or embeddings.
The sidecar belongs on train.Item, outside the inference row and HeadInput.
"""
from dataclasses import replace

AXES = ("assert", "explain", "compare", "record", "field", "relation")
RECORDS = {"development-record-1": -1.0, "development-record-2": 1.0}
FIELDS = {"price": -1.0, "warranty_months": 1.0}
RELATIONS = {"less_than": -1.0, "equal": 0.0, "greater_than": 1.0}


def axis_target(program, slot):
    nodes = {node["id"]: node for node in program["nodes"]}
    if len(nodes) != len(program["nodes"]) or len(program["outputs"]) != 1:
        raise ValueError("one typed output and unique node identities required")
    output = nodes[program["outputs"][0]]
    explain = output["op"] == "ExplainDerivation"
    assertion = nodes[output["speech_act"]] if explain else output
    if assertion["op"] != "Assert":
        raise ValueError("axis targets support only checked assertions/explanations")
    proposition = nodes[assertion["proposition"]]
    compare = proposition["op"] == "Compare"
    if proposition["op"] not in ("ReadFact", "Compare") or (explain and compare):
        raise ValueError("unsupported supervised semantic program")
    result = [0.0] * 6
    if slot == "operation":
        result[2 if compare else (1 if explain else 0)] = 1.0
    elif slot in ("fact", "left", "right"):
        if slot == "fact":
            if compare:
                raise ValueError("fact target cannot bind a comparison")
            fact = proposition
        else:
            if not compare or proposition[slot].get("project") != "object":
                raise ValueError("operand target requires a bound comparison Fact")
            fact = nodes[proposition[slot]["node"]]
        if fact["op"] != "ReadFact" or fact["record_ref"] not in RECORDS or fact["field"] not in FIELDS:
            raise ValueError("axis target outside the frozen two-record/two-field domain")
        result[3], result[4] = RECORDS[fact["record_ref"]], FIELDS[fact["field"]]
    elif slot == "relation":
        if not compare or proposition["relation"] not in RELATIONS:
            raise ValueError("relation target requires a registered comparison")
        result[5] = RELATIONS[proposition["relation"]]
    else:
        raise ValueError("unsupported supervised slot")
    return tuple(result)


def targets_by_item(examples):
    result = {}
    for example in examples:
        training = example["training_only"]
        for index, target in enumerate(training["targets"], 1):
            slot = target["slot"]
            identity = f"{example['task_id']}/slot-{index:03d}-{slot}"
            if identity in result:
                raise ValueError("duplicate axis supervision identity")
            result[identity] = axis_target(training["program"], slot)
    return result


def attach_axis_targets(items, examples):
    targets = targets_by_item(examples)
    if len(items) != len(targets) or {item.id for item in items} != set(targets):
        raise ValueError("axis sidecar and exact training item identities differ")
    return [replace(item, axis_target=targets[item.id]) for item in items]

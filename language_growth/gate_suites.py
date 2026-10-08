"""Independent synthetic language suites; inputs never contain evaluator labels.

This generator does not read Decision Index, D1-B, or retention data. Heldout
material is generated only by an explicit prepare command, never during a dev
run. Worlds, entities, scalar ranges and task wording differ across splits.
Generated labels are checker-validated, not an independent human audit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .development_data import _catalog, _host
from .kernel import check_program
from .realize import render_checked
from .types import Comparison, Fact, SpeechAct, ExactDifference, ConditionalContent

LEVELS = ("L0", "L1", "L2", "L3", "L4")
SPLITS = ("development_train", "development_validation", "language_heldout")
SUITE_SCHEMA = "ssdm.language_gate_suite.v1"
SCOPE = {
    "L0": "single_fact_and_exact_comparison",
    "L1": "multi_record_selection_and_exact_difference",
    "L2": "conjunction_registered_hypothesis_and_registered_plan",
    "L3": "licensed_coordination_and_quantitative_contrast",
    "L4": "scoped_historical_electricity_knowledge",
}
from .gate_answer_format import answer_format

ANSWER_FORMAT = answer_format()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def answer_meaning(outputs: tuple[SpeechAct, ...]) -> dict[str, Any]:
    """A common answer schema for checked outputs and unrestricted generation.

    Comparator decoding is unconstrained: it generates this final-answer JSON
    without program grammar, checking, retries or gold labels. This measures
    narrow semantic usefulness, not unrestricted prose quality or fluency.
    """
    claims = []
    explanations = 0

    def visit(act: SpeechAct) -> None:
        nonlocal explanations
        if act.kind == "conjoin":
            for child in act.content:
                visit(child)
        elif act.kind == "explain":
            explanations += 1
            visit(act.content)
        elif act.kind == "hypothesize":
            content = act.content
            def hyp_operand(fact):
                return {"entity": fact.subject, "field": fact.field, "value": _plain(fact.object), "unit": fact.unit}
            claims.append({"kind": "hypothesis", "assumption_ref": content.assumption.assumption_ref,
                           "premise": hyp_operand(content.assumption.premise), "alternate_value": _plain(content.assumption.value),
                           "right": hyp_operand(content.conclusion.right), "relation": content.conclusion.relation,
                           "scope": content.conclusion.interval})
        elif act.kind == "plan":
            content = act.content
            claims.append({"kind": "plan", "workflow_ref": content.workflow_ref,
                           "steps": [{"step_ref": step.step_ref, "action_ref": step.action_ref, "after": list(step.after),
                                      "preconditions": [{"kind": "fact", "entity": fact.subject, "field": fact.field,
                                                         "value": _plain(fact.object), "unit": fact.unit, "scope": fact.interval}
                                                        for fact in step.preconditions]} for step in content.steps]})
        elif act.kind == "assert" and isinstance(act.content, (Fact, Comparison, ExactDifference)):
            item = act.content
            def operand(fact):
                if not isinstance(fact, Fact):
                    raise ValueError("suite supports fact operands only")
                return {"entity": fact.subject, "field": fact.field,
                        "value": _plain(fact.object), "unit": fact.unit}
            if isinstance(item, Fact):
                claims.append({"kind": "fact", "entity": item.subject,
                               "field": item.field, "value": _plain(item.object),
                               "unit": item.unit, "scope": item.interval})
            elif isinstance(item, ExactDifference):
                claims.append({"kind": "exact_difference", "left": operand(item.left), "right": operand(item.right),
                               "value": _plain(item.value), "unit": item.unit, "scope": item.interval})
            else:
                left, right, relation = operand(item.left), operand(item.right), item.relation
                # Normalize inverse comparisons for semantic equivalence.
                if canonical(left) > canonical(right):
                    left, right = right, left
                    relation = {"greater_than": "less_than", "less_than": "greater_than", "equal": "equal"}[relation]
                claims.append({"kind": "comparison", "left": left, "right": right,
                               "relation": relation, "scope": item.interval})
        else:
            raise ValueError("unsupported answer meaning")

    for output in outputs:
        visit(output)
    return {"claims": sorted(claims, key=canonical), "explanations": explanations}


def _program(host: dict, nodes: list[dict]) -> dict:
    return {"schema": "ssdm.semantic_program.v0.1",
            **{key: host[key] for key in ("context_ref", "policy_ref", "registry_ref")},
            "nodes": nodes, "outputs": [nodes[-1]["id"]], "style": "plain"}


def build_suite(level: str, split: str, *, worlds: int = 4, seed: int = 20261008) -> tuple[list[dict], list[dict]]:
    """Return separate provider inputs and evaluator-only references.

    Each synthetic world is one bootstrap unit across all operation variants.
    L4 uses the actual fixed historical module: it has ONE source-world, never
    fake independent copies, and cannot meet a multi-world qualification floor.
    """
    if level not in LEVELS or split not in SPLITS or type(worlds) is not int or not 1 <= worlds <= 10000:
        raise ValueError("invalid level, split or bounded world count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if level == "L4":
        return _energy_suite(split)
    inputs, references = [], []
    domain = SPLITS.index(split)
    for index in range(worlds):
        rng = random.Random(int(digest([SUITE_SCHEMA, split, seed, index]), 16))
        names = tuple(f"{split}-entity-{seed}-{index}-{j}" for j in range(4))
        offset = 1000 * (domain + 1)
        warranties = tuple(offset + rng.randrange(1, 200) for _ in range(4))
        prices = tuple(100 * offset + rng.randrange(1, 20000) for _ in range(4))
        old_refs = [f"development-record-{j}" for j in range(1, 5)]
        new_refs = rng.sample(old_refs, len(old_refs))
        ref_map = dict(zip(old_refs, new_refs))
        def relabel(value):
            if isinstance(value, str):
                return ref_map.get(value, value)
            if isinstance(value, list):
                return [relabel(item) for item in value]
            if isinstance(value, dict):
                return {key: relabel(item) for key, item in value.items()}
            return value
        host = _host()
        catalog = _catalog(names[:2], warranties[:2], prices[:2])
        extra = _catalog(names[2:], warranties[2:], prices[2:])["records"]
        for j, row in enumerate(extra, 3):
            row["record_ref"] = f"development-record-{j}"
        catalog["records"].extend(extra)
        registry = catalog["registries"][host["registry_ref"]]
        registry["allowed_ops"].extend({"L0": [], "L1": ["ExactDifference"],
                                       "L2": ["Conjoin", "Hypothesize", "Plan"], "L3": ["Conjoin"]}[level])
        catalog["snapshots"][host["context_ref"]]["record_refs"] = [r["record_ref"] for r in catalog["records"]]
        host["limits"]["max_proposal_calls"] = 8
        world_id = f"language-{split}-{seed}-world-{index}"
        variants = ("base",)
        kinds = {"L0": ("read", "compare"), "L1": ("read", "compare", "difference"),
                 "L2": ("conjoin", "hypothesis", "plan"), "L3": ("conjoin", "compare")}[level]
        for variant in variants:
            current = deepcopy(catalog)
            if variant == "corrected":
                first = current["records"][0]["fields"]["warranty_months"]
                first["value"] = current["records"][1]["fields"]["warranty_months"]["value"] + 100
                current["records"][0]["version"] = 2
            for kind in kinds:
                nodes = [{"id": "n1", "op": "ReadFact", "record_ref": "development-record-1", "field": "warranty_months"}]
                if kind in ("compare", "difference"):
                    left, right = (r["fields"]["warranty_months"]["value"] for r in current["records"][:2])
                    relation = "equal" if left == right else "greater_than" if left > right else "less_than"
                    nodes.extend([
                        {"id": "n2", "op": "ReadFact", "record_ref": "development-record-2", "field": "warranty_months"},
                        {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"},
                         "right": {"node": "n2", "project": "object"}, "relation": relation},
                        {"id": "n4", "op": "Assert", "proposition": "n3"}])
                    if kind == "difference":
                        nodes[2]["op"] = "ExactDifference"
                        nodes[2].pop("relation")
                elif kind == "conjoin":
                    nodes.extend([
                        {"id": "n2", "op": "Assert", "proposition": "n1"},
                        {"id": "n3", "op": "ReadFact", "record_ref": "development-record-2", "field": "price"},
                        {"id": "n4", "op": "Assert", "proposition": "n3"},
                        {"id": "n5", "op": "Conjoin", "speech_acts": ["n2", "n4"]}])
                elif kind == "hypothesis":
                    host["assumptions"] = {"alternate-warranty": {"record_ref": "development-record-1", "field": "warranty_months",
                        "value": {"value": warranties[1] + 100, "value_type": "integer", "unit": "month", "temporal_scope": "2026-10", "quantity_kind": "warranty_months"}}}
                    nodes = [{"id": "n1", "op": "ReadFact", "record_ref": "development-record-2", "field": "warranty_months"},
                             {"id": "n2", "op": "Compare", "left": {"assumption_ref": "alternate-warranty"}, "right": {"node": "n1", "project": "object"}, "relation": "greater_than"},
                             {"id": "n3", "op": "Hypothesize", "proposition": "n2"}]
                elif kind == "plan":
                    host["workflows"] = {"catalog-review": {"audience": "public", "status": "current", "steps": {
                        "inspect": {"action_ref": "inspect-catalog", "after": [], "preconditions": [{"record_ref": "development-record-1", "field": "warranty_months", "expected": deepcopy(current["records"][0]["fields"]["warranty_months"])}]},
                        "draft": {"action_ref": "draft-summary", "after": ["inspect"], "preconditions": [{"record_ref": "development-record-2", "field": "warranty_months", "expected": deepcopy(current["records"][1]["fields"]["warranty_months"])}]}}}}
                    nodes = [{"id": "n1", "op": "Plan", "workflow_ref": "catalog-review", "step_refs": ["inspect", "draft"]}]
                else:
                    nodes.append({"id": "n2", "op": "Assert", "proposition": "n1"})
                    if kind == "explain":
                        nodes.append({"id": "n3", "op": "ExplainDerivation", "speech_act": "n2"})
                wording = {
                    "development_train": f"Give {names[0]}'s warranty_months for October 2026.",
                    "development_validation": f"State the October 2026 warranty_months recorded for {names[0]}.",
                    "language_heldout": f"Report the warranty_months of {names[0]}, effective October 2026.",
                }[split]
                if kind in ("compare", "difference"):
                    wording = {"development_train": f"Compare {names[0]}'s and {names[1]}'s October 2026 warranty_months.",
                               "development_validation": f"For October 2026, how do the warranty_months of {names[0]} and {names[1]} compare?",
                               "language_heldout": f"Report the relation between {names[0]}'s warranty_months and {names[1]}'s warranty_months in October 2026."}[split]
                    if kind == "difference":
                        wording = f"For October 2026, calculate {names[0]}'s warranty_months minus {names[1]}'s warranty_months."
                elif kind == "conjoin":
                    wording = f"State {names[0]}'s warranty_months and {names[1]}'s price for October 2026."
                elif kind == "hypothesis":
                    wording = f"If registered assumption alternate-warranty holds, compare {names[0]}'s warranty_months with {names[1]}'s warranty_months for October 2026."
                elif kind == "plan":
                    wording = "Propose all steps of the registered catalog-review workflow in dependency order."
                elif kind == "explain":
                    wording = f"Explain the checked October 2026 warranty_months fact for {names[0]}."
                task_id = f"{world_id}-{level}-{kind}-{variant}"
                if level == "L3":
                    wording += " Use licensed coordination." if kind == "conjoin" else " Use licensed quantitative contrast."
                task = {"task_id": task_id, "world_id": world_id, "level": level,
                        "split": split, "task_input": wording,
                        "host": deepcopy(host), "catalog": current,
                        "planner_profile": "conjoin_two_facts" if kind == "conjoin" else None,
                        "answer_format": ANSWER_FORMAT,
                        "production": ("l3_fact_pair_coordination_v1" if kind == "conjoin" else "l3_fact_comparison_contrast_v1") if level == "L3" else "plain"}
                task = relabel(task)
                rng.shuffle(task["catalog"]["records"])
                program = relabel(_program(host, nodes))
                checked = check_program(program, task["host"], task["catalog"])
                if checked.status != "accepted":
                    raise ValueError(f"generated reference failed: {checked.reason}")
                inputs.append(task)
                references.append({"task_id": task_id, "world_id": world_id,
                                   "input_sha256": digest(task), "kind": kind, "variant": variant,
                                   "program": program, "meaning": answer_meaning(checked.outputs),
                                   "text": render_checked(checked, task["host"], task["catalog"], production=task["production"]),
                                   "label_provenance": "synthetic_checker_valid_not_independent_audit"})
    return inputs, references


def _energy_suite(split: str) -> tuple[list[dict], list[dict]]:
    from .domains.energy import load_energy_module, energy_environment, check_energy_program, public_energy_answer, LOWEST_TASK, MODULE_REF, PROGRAM_SCHEMA
    module = load_energy_module()
    host, catalog = energy_environment(module)
    host["limits"].update(max_questions=1, max_options_per_question=32, max_proposal_calls=8, max_repair_calls=0, max_repair_attempts=0)
    host["provider"].update(input_context_ref=host["context_ref"], state_history_context_refs=[])
    indices = {"development_train": (0, 1), "development_validation": (2, 3), "language_heldout": (4,)}[split]
    choices = [("ReadTechnology", module["rows"][i]) for i in indices]
    if split == "development_validation":
        choices.append(("LowestMedian", min(module["rows"], key=lambda r: Decimal(r["lifecycle_median"]))))
    inputs, references = [], []
    for op, row in choices:
        task = {"task_id": f"language-{split}-L4-{op}-{row['record_ref']}", "world_id": "ipcc-ar5-fixed-public-source-world",
                "level": "L4", "split": split, "task_input": LOWEST_TASK if op == "LowestMedian" else f"In IPCC AR5 (2014), what lifecycle emissions are reported for {row['technology']}?",
                "host": deepcopy(host), "catalog": deepcopy(catalog), "energy_module": deepcopy(module),
                "planner_profile": "energy", "production": "energy", "answer_format": ANSWER_FORMAT}
        proposal = {"schema": PROGRAM_SCHEMA, "module_ref": MODULE_REF, "op": op, "record_ref": row["record_ref"]}
        checked = check_energy_program(proposal, module, host, catalog)
        released = public_energy_answer(task["task_input"], proposal, module, host, catalog)
        if checked.status != "accepted" or released.get("status") != "ok":
            raise ValueError("energy reference failed")
        inputs.append(task)
        references.append({"task_id": task["task_id"], "world_id": task["world_id"], "input_sha256": digest(task),
                           "kind": op, "variant": "base", "program": proposal, "meaning": answer_meaning(checked.outputs),
                           "text": released["text"], "label_provenance": "public_historical_table_fixed_source_not_independent_audit"})
    return inputs, references


def export_suite(out: Path, level: str, split: str, *, worlds: int, seed: int) -> dict:
    """Write provider/evaluator files separately; never overwrite a cohort."""
    tasks, refs = build_suite(level, split, worlds=worlds, seed=seed)
    out = Path(out)
    if out.exists():
        raise ValueError("suite output already exists")
    input_bytes = b"".join(canonical(row) + b"\n" for row in tasks)
    reference_bytes = b"".join(canonical(row) + b"\n" for row in refs)
    manifest = {"schema": SUITE_SCHEMA, "level": level, "split": split, "seed": seed,
                "worlds": len({task["world_id"] for task in tasks}), "tasks": len(tasks), "scope": SCOPE[level],
                "inputs_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "evaluator_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                "promotion_eligible": False,
                "label_provenance": "synthetic_checker_valid_not_independent_audit",
                "training_allowed": split == "development_train"}
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs.jsonl").write_bytes(input_bytes)
    (out / "evaluator.jsonl").write_bytes(reference_bytes)
    (out / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest

"""Bounded, newly authored synthetic wording development cohort; never training."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .development_data import _catalog, _host, _program, _relation, _rng, _OPS
from .kernel import check_program
from .parent_bridge import decision_request

SPLIT = "development_paraphrase"
NAMES = ("Marlow", "Nacre", "Oriole", "Peregrine", "Quill", "Rill")
WARRANTY_RANGE = (61, 84)
PRICE_CENTS_RANGE = (30001, 45000)
# One fixed wording set authored before inference. Literal operation labels are
# absent from task text. Field names stay literal to isolate intent wording.
TEMPLATES = {
    "read": (
        "Retrieve {entity}'s {field} from the October 2026 catalog.",
        "I'd like the October 2026 {field} listed for {entity}.",
        "Tell me the {field} recorded for {entity} during October 2026.",
    ),
    "explain": (
        "Walk me through the evidence supporting {entity}'s October 2026 {field}.",
        "Trace the reasoning that establishes {entity}'s {field} for October 2026.",
        "Why is the October 2026 {field} attributed to {entity} justified by the catalog?",
    ),
    "compare": (
        "Contrast {left}'s October 2026 {field} with {right}'s, including whether they tie.",
        "Rank {left} and {right} by their October 2026 {field}; specify which is higher or whether they tie.",
        "How does {left}'s October 2026 {field} stack up against {right}'s: higher, lower, or the same?",
    ),
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def generate_paraphrase_tasks(seed: int = 161803) -> list[dict[str, Any]]:
    """Exactly 100 tasks per kind, with 34/33/33 phrasings per kind.

    Only an initial empty-prefix request is exported. No teacher-forced gold
    prefixes are present anywhere outside training_only. Runtime options use
    the existing production planner's fixed order; no option-shuffle diagnostic.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tasks = []
    for index in range(300):
        kind = ("read", "explain", "compare")[index % 3]
        template_index = (index // 3) % 3
        rng = _rng(SPLIT, seed, index, "world")
        names = tuple(rng.sample(NAMES, 2))
        warranties = tuple(rng.randint(*WARRANTY_RANGE) for _ in range(2))
        prices = tuple(rng.randint(*PRICE_CENTS_RANGE) for _ in range(2))
        # Preserve genuine equality as part of the comparison domain.
        if _rng(SPLIT, seed, index, "equality-case").randrange(3) == 0:
            prices = (prices[0], prices[0])
        field = rng.choice(("price", "warranty_months"))
        fact_index, left_index = rng.randrange(2), rng.randrange(2)
        wording = TEMPLATES[kind][template_index].format(
            entity=names[fact_index], left=names[left_index], right=names[1-left_index],
            field=field.replace("_", " "))
        catalog, host = _catalog(names, warranties, prices), _host()
        relation = _relation(prices if field == "price" else warranties, left_index)
        program = _program(kind, field, fact_index, relation if kind == "compare" else None, left_index)
        old_context = host["context_ref"]
        digest = hashlib.sha256(canonical(catalog).encode("ascii")).hexdigest()
        context = f"{old_context}:{digest}"
        catalog["snapshots"][context] = catalog["snapshots"].pop(old_context)
        host["context_ref"] = host["provider"]["input_context_ref"] = program["context_ref"] = context
        checked = check_program(program, host, catalog)
        if checked.status != "accepted":
            raise ValueError(f"reference rejected: {checked.reason}")
        operations = {f"op{i:03d}": description for i, (_, description) in enumerate(_OPS, 1)}
        request, _ = decision_request(
            task_input=wording, host=host, catalog=catalog,
            model="development-only", parent_revision="synthetic-no-model:v1",
            request_id=hashlib.sha256(f"{SPLIT}:{seed}:{index}:{digest}".encode()).hexdigest()[:24],
            question_id="slot-001-operation", question="Select the semantic operation for the user's task.",
            options=operations, proposed_slot_choices=[])
        fact_id = lambda record, prefix: f"{prefix}{record * 2 + (1 if field == 'price' else 2):03d}"
        targets = [{"slot": "operation", "option_id": f"op{index % 3 + 1:03d}"}]
        if kind == "compare":
            targets.extend([
                {"slot": "left", "option_id": fact_id(left_index, "v")},
                {"slot": "right", "option_id": fact_id(1-left_index, "v")},
                {"slot": "relation", "option_id": f"r{('greater_than', 'less_than', 'equal').index(relation)+1:03d}"},
            ])
        else:
            targets.append({"slot": "fact", "option_id": fact_id(fact_index, "f")})
        tasks.append({
            "development_only": True, "development_split": SPLIT,
            "task_id": f"synthetic-{SPLIT}-{seed}-{index}", "model_inputs": [request],
            "training_only": {"kind": kind, "seed": seed, "index": index,
                "template_id": f"{SPLIT}:{kind}:{template_index}", "allow_one_repair": False,
                "host": host, "catalog": catalog, "program": program, "targets": targets},
        })
    return tasks


def paraphrase_manifest(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tasks) != 300 or any(t["development_split"] != SPLIT for t in tasks):
        raise ValueError("exact bounded paraphrase cohort required")
    return {"schema": "ssdm.synthetic_paraphrase_manifest.v1", "split": SPLIT,
        "seed": tasks[0]["training_only"]["seed"], "count": 300,
        "name_pool": list(NAMES), "warranty_range": list(WARRANTY_RANGE),
        "price_cents_range": list(PRICE_CENTS_RANGE), "templates": TEMPLATES,
        "template_counts": {f"{SPLIT}:{kind}:{i}": sum(t["training_only"]["template_id"] == f"{SPLIT}:{kind}:{i}" for t in tasks)
                            for kind in TEMPLATES for i in range(3)},
        "sample_sha256": hashlib.sha256("".join(canonical(t)+"\n" for t in tasks).encode("ascii")).hexdigest(),
        "sample_encoding": "sorted compact ensure_ascii JSON per LF line, final LF",
        "scope": "new synthetic development wording only; no Index, official or sealed held-out claim"}

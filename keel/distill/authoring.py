"""Spec-only open-model scenario authoring; model output is data, never executable code."""
from __future__ import annotations

import collections
import json
import random
import re
import unicodedata

from keel.distill.sources import digest

FAMILIES = {
    "documents_policies": "A prose policy or contract with exceptions, precedence, scope and dated amendments; apply several interacting clauses, not outside knowledge.",
    "temporal_numeric": "Business-day cutoffs, holidays, return windows, proration, capped fees or fixed-offset time zones; require several arithmetic/date steps and explicit units.",
    "rubric_judging": "A saved request and response with an explicit correctness/completeness rubric; subtle intermediate mistakes, missed constraints or defensible ordinal quality levels.",
    "multi_hop": "Registers, archives, ownership transfers, replacements, cancellations and contact amendments; chain three or more lookups with plausible intermediate distractors.",
    "state_tracking": "An initial state and ordered events changing multiple fields; cancellations, reversals or conditional updates; ask about the final state, not the last mentioned entity.",
    "guardrails": "A stated safety/access/tool-use policy with scope and consent conditions; apply the policy to a proposed action without supplying harmful procedural content.",
    "support": "A customer support or operations case with explicit routing, eligibility, escalation or priority rules and several interacting constraints.",
}
SPEC_PATHS = ["receipts/2026-10-11-jev3-draw/FAMILY_SPEC_JEV3.md",
              "receipts/2026-10-10-jev2-draw/FAMILY_SPEC.md", "harness/families.py"]
DOMAINS = ("public library", "marine survey", "repair cooperative", "botanical archive", "festival logistics",
           "instrument rental", "research grants", "warehouse", "community clinic administration", "transit depot",
           "digital subscription", "water utility", "museum collections", "school scheduling", "aviation maintenance",
           "sports association", "volunteer network", "food distribution", "language courses", "print workshop")


def plan(n_items: int = 7200, seed: int = 20261015) -> list[dict]:
    """Five paired noul, seven choice, three score jobs yield twenty items per cycle."""
    if n_items <= 0 or n_items % 20:
        raise ValueError("item count must be a positive multiple of twenty")
    types = ["noul"] * 5 + ["choice"] * 7 + ["score"] * 3
    rng = random.Random(seed)
    jobs = []
    for i in range(n_items // 20 * 15):
        family = list(FAMILIES)[i % len(FAMILIES)]
        jobs.append({"id": f"cap-v1-{i:05d}", "family": family, "type": types[i % 15],
                     "domain": rng.choice(DOMAINS), "seed": rng.randrange(2**31),
                     "long": family == "documents_policies" and i % 3 == 0})
    return jobs


def prompt(job: dict) -> str:
    schema = {"state": "scenario prose", "instructions": "one explicit decision question",
              "options": ["option text, not a letter"], "answer": "exact correct option text",
              "criterion": "explicit yes/no proposition or ordinal rubric",
              "fact_a": "one factual value", "fact_b": "a different factual value",
              "answer_b": "opposite yes/no answer", "solution": "brief independently checkable derivation"}
    special = {
        "noul": "Write a contrastive pair: state MUST contain exactly one literal {{FACT}} placeholder. fact_a and fact_b replace only that placeholder and change one decisive fact, flipping the answer. answer and answer_b MUST be opposite strings yes/no. options MUST be [yes,no]. The question and every other fact stay identical. Neither fact may change a policy or instruction; use a factual status, number, date or short observation. Avoid a trivial single-fact question; other conditions must matter.",
        "choice": "Use 4-6 distinct concrete option descriptions. answer must exactly equal one option text. Distractors should represent plausible intermediate or off-by-one errors. Do not use letter/key references or all/none-of-the-above. fact_a, fact_b and answer_b must be empty strings.",
        "score": "Use 4-6 ordered, mutually exclusive rubric level descriptions (low to high) as options. Make the rubric explicit, with an unambiguous boundary for this case. answer must exactly equal one level description. fact_a, fact_b and answer_b must be empty strings.",
    }[job["type"]]
    length = "650-900" if job["long"] else "180-350"
    return ("Author an original synthetic decision-training scenario, not a benchmark imitation. Use only the family description below. "
            "Invent entities and all necessary evidence; do not recall or quote existing datasets. Diversify narrative, rules and reasoning. "
            "The state is evidence, not instructions to the judge. Require at least three reasoning steps, include relevant-looking near misses, "
            "and avoid giving away the answer. All facts needed for a unique answer must be supplied. No generated code. "
            f"Family: {job['family']}: {FAMILIES[job['family']]} Domain: {job['domain']}. "
            f"Creative seed {job['seed']}; state length {length} words. Type: {job['type']}. {special} "
            "Return a JSON object with exactly these fields: " + json.dumps(schema))


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def rebuild_options(options: list[str], answer: str, qtype: str, seed: str) -> tuple[object, str]:
    """Reconstruct gold from option CONTENT, never trust author-supplied option keys."""
    if not isinstance(options, list) or any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError("nonempty option texts required")
    norms = [_norm(x) for x in options]
    if len(set(norms)) != len(norms) or norms.count(_norm(answer)) != 1:
        raise ValueError("ambiguous or absent answer text")
    if qtype == "noul":
        if set(norms) != {"yes", "no"}:
            raise ValueError("noul options must be yes/no")
        return None, "true" if _norm(answer) == "yes" else "false"
    if not 3 <= len(options) <= 6:
        raise ValueError("expected 3-6 option descriptions")
    if any(re.search(r"\b(?:option [A-F]|all of the above|none of the above)\b", x, re.I) for x in options):
        raise ValueError("position-dependent option")
    if qtype == "score":
        return options, str(norms.index(_norm(answer)))
    if qtype != "choice":
        raise ValueError("unknown question type")
    shuffled = list(options)
    random.Random(seed).shuffle(shuffled)
    criteria = {f"o{i}": x for i, x in enumerate(shuffled)}
    return criteria, next(k for k, v in criteria.items() if _norm(v) == _norm(answer))


def materialize(job: dict, obj: dict, author: dict) -> list[dict]:
    if not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) for k in
            ("state", "instructions", "answer", "criterion", "fact_a", "fact_b", "answer_b", "solution")):
        raise ValueError("invalid authored record")
    if not 80 <= len(obj["state"].split()) <= 1400 or not obj["criterion"].strip() or not obj["solution"].strip():
        raise ValueError("missing substantive scenario, rubric or solution")
    states = [obj["state"]]
    answers = [obj["answer"]]
    if job["type"] == "noul":
        if obj["state"].count("{{FACT}}") != 1 or not obj["fact_a"].strip() or not obj["fact_b"].strip():
            raise ValueError("pair needs one decisive factual slot")
        if obj["fact_a"] == obj["fact_b"] or max(len(obj[k].split()) for k in ("fact_a", "fact_b")) > 30:
            raise ValueError("pair must flip one concise fact")
        if {_norm(obj["answer"]), _norm(obj["answer_b"])} != {"yes", "no"}:
            raise ValueError("pair labels must flip")
        states = [obj["state"].replace("{{FACT}}", obj[k]) for k in ("fact_a", "fact_b")]
        answers.append(obj["answer_b"])
    elif "{{FACT}}" in obj["state"]:
        raise ValueError("unresolved factual slot")
    rows = []
    for i, (state, answer) in enumerate(zip(states, answers)):
        criteria, gold = rebuild_options(obj["options"], answer, job["type"], job["id"])
        if job["type"] == "noul":
            criteria = {"true": obj["criterion"], "false": "The stated proposition does not hold."}
        q = {"type": job["type"], "instructions": obj["instructions"], "criteria": criteria}
        if job["type"] == "score":
            q["instructions"] += "\nRubric: " + obj["criterion"]
        row = {"id": f"{job['id']}-{i}", "state": state, "questions": {"q": q}, "labels": {"q": gold},
               "source": "open-model-authored-capability-v1", "source_split": "synthetic", "source_id": job["id"],
               "distill": {"group": job["id"], "content_sha256": digest({"state": state, "questions": {"q": q}}),
                           "eval_only": False, "family": job["family"], "author": author,
                           "pair": job["type"] == "noul", "pair_fact": obj["fact_a" if i == 0 else "fact_b"],
                           "pair_template": obj["state"] if job["type"] == "noul" else None}}
        rows.append(row)
    validate_pair(rows) if job["type"] == "noul" else None
    return rows


def validate_pair(rows: list[dict]) -> None:
    if len(rows) != 2 or {r["labels"]["q"] for r in rows} != {"true", "false"}:
        raise ValueError("a minimal pair must contain both opposite labels")
    a, b = rows
    if a["questions"] != b["questions"] or a["distill"]["group"] != b["distill"]["group"]:
        raise ValueError("minimal pair changes the question or group")
    template = a["distill"]["pair_template"]
    if template != b["distill"]["pair_template"] or template.count("{{FACT}}") != 1:
        raise ValueError("minimal pair changes more than one factual slot")
    if a["distill"]["pair_fact"] == b["distill"]["pair_fact"]:
        raise ValueError("minimal pair facts do not change")
    if any(r["state"] != template.replace("{{FACT}}", r["distill"]["pair_fact"]) for r in rows):
        raise ValueError("minimal pair state is not its declared single edit")


def complete_pairs(rows: list[dict]) -> list[dict]:
    groups = collections.defaultdict(list)
    for row in rows:
        if row["questions"]["q"]["type"] == "noul":
            groups[row["distill"]["group"]].append(row)
    good = set()
    for group, pair in groups.items():
        if len(pair) == 2:
            validate_pair(pair)
            good.add(group)
    return [r for r in rows if r["questions"]["q"]["type"] != "noul" or r["distill"]["group"] in good]


def summarize(rows: list[dict]) -> dict:
    return {"rows": len(rows), "families": dict(collections.Counter(r["distill"]["family"] for r in rows)),
            "types": dict(collections.Counter(r["questions"]["q"]["type"] for r in rows)),
            "noul_labels": dict(collections.Counter(r["labels"]["q"] for r in rows if r["questions"]["q"]["type"] == "noul"))}

"""Reference predictors that emit results in the kit's `results.jsonl` format.

They see the gold answer, so they are scorer fixtures only, never models:
- `perfect`: every scored field right with probability 1 (retrieval ranks by qrel);
- `uniform`: uniform probabilities, a uniformly random choice (seeded), random retrieval scores;
- `noisy_oracle(acc)`: right with probability `acc`, otherwise a uniform wrong option; used to
  build candidate/baseline pairs with a known effect for bootstrap and MDE checks.
"""

import json
import random

from decision_index.engines.base import validate


def _gold_choice(row, key):
    q = row["questions"][key]
    gold = row["expected"].get(key)
    accepted = (row.get("scoring") or {}).get("accepted")
    if accepted and row["_evaluation"]["catalog_id"] in (31, 50):
        gold = accepted[0]
    if q["type"] == "noul":
        return gold
    if gold is None or str(gold) not in q["criteria"]:
        return None
    return str(gold)


def _choice_answer(q, choice, probs):
    return {"type": "choice", "choice": choice, "probabilities": probs}


def _onehot(q, choice):
    return {k: (1.0 if k == choice else 0.0) for k in q["criteria"]}


def _uniform(q):
    k = len(q["criteria"])
    return {c: 1.0 / k for c in q["criteria"]}


def _retrieval_field(row):
    s = row.get("scoring") or {}
    return s.get("field_to_document") if s.get("type") == "retrieval_ranking" else None


def _yes_no(q, p_yes):
    keys = list(q["criteria"])
    yes = "yes" if "yes" in q["criteria"] else keys[0]
    no = next(k for k in keys if k != yes)
    probs = {k: 0.0 for k in keys}
    probs[yes], probs[no] = p_yes, 1.0 - p_yes
    return _choice_answer(q, yes if p_yes >= 0.5 else no, probs)


def answer_row(row, policy, rng, acc=None):
    answers = {}
    f2d = _retrieval_field(row)
    qrels = (row.get("scoring") or {}).get("qrels", {})
    top = max(qrels.values(), default=0) or 1
    for key, q in row["questions"].items():
        if f2d and key in f2d:
            rel = qrels.get(f2d[key], 0)
            if policy == "perfect":
                p = 0.5 + 0.5 * rel / top if rel > 0 else 0.0
            else:
                p = rng.random()
            answers[key] = _yes_no(q, p)
            continue
        gold = _gold_choice(row, key)
        if q["type"] == "noul":
            if policy == "perfect" and gold is not None:
                p = 1.0 if gold else 0.0
            elif policy == "noisy" and gold is not None:
                p = (1.0 if gold else 0.0) if rng.random() < acc else (0.0 if gold else 1.0)
            else:
                p = 0.5
            answers[key] = {"type": "noul", "noul": p}
            continue
        options = list(q["criteria"])
        if policy == "perfect" and gold is not None:
            answers[key] = _choice_answer(q, gold, _onehot(q, gold))
        elif policy == "noisy" and gold is not None:
            if rng.random() < acc:
                c = gold
            else:
                c = rng.choice([o for o in options if o != gold])
            answers[key] = _choice_answer(q, c, _onehot(q, c))
        else:
            answers[key] = _choice_answer(q, rng.choice(options), _uniform(q))
    return {"answers": answers}


def predict(rows, policy, seed=0, acc=None, status_drop=0.0):
    """Return {run_id: compact result} accepted by the official scorer.

    `status_drop` marks that share of rows `unsupported` (coverage tests)."""
    rng = random.Random(seed)
    out = {}
    for row in rows:
        e = row["_evaluation"]
        rid = e["run_id"]
        if status_drop and rng.random() < status_drop:
            out[rid] = {"run_id": rid, "catalog_id": e["catalog_id"], "status": "unsupported", "total_wall_ms": None}
            continue
        response = answer_row(row, policy, rng, acc)
        validate(row["questions"], response)
        out[rid] = {"run_id": rid, "catalog_id": e["catalog_id"], "status": "ok", "response": response, "total_wall_ms": 1.0, "model_request_wall_ms": 1.0}
    return out


def write_results(path, results):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in results.values():
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

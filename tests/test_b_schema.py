"""data.b_schema: every transformation is label-preserving (the gold's option TEXT / truth value / level description
is unchanged), deterministic, and keeps context keys top-level."""
from __future__ import annotations

import json

from data import augment as A
from data.b_schema import NOUL_TRUE_WORDS, variant
from decide.schema import NOUL_DEFAULT_CRITERIA


def _choice(k=6, gold=3, desc=True):
    crit = {f"option_{i}": (f"answer text {i}" if desc else None) for i in range(k)}
    return {"id": "m2:test:1", "schema": "ssdm-train-v0", "source": "test", "source_split": "train", "source_id": "1",
            "license": "x", "family": "mcq", "state": {"question": "What is it?", "context": "Some context here."},
            "questions": {"q1": {"type": "choice", "instructions": "Choose the best answer.", "criteria": crit}},
            "labels": {"q1": f"option_{gold}"}, "untrusted_fields": [], "aug": {}}


def gold_text(rec):
    (qid, q), = rec["questions"].items()
    lab = rec["labels"][qid]
    if q["type"] == "choice":
        d = q["criteria"][lab]
        return ("choice", d)
    if q["type"] == "noul":
        return ("noul", lab)
    return ("score", q["criteria"][int(lab)])


def truth(rec):
    (qid, q), = rec["questions"].items()
    lab = rec["labels"][qid]
    if q["type"] == "noul":
        return lab == "true"
    return str(lab).casefold() in NOUL_TRUE_WORDS


def test_choice_gold_text_preserved_and_deterministic():
    base = _choice()
    for j in range(40):
        v = variant(base, j, 7, {})
        A.validate(v)
        assert v == variant(base, j, 7, {})
        kind, d = gold_text(v)
        if kind == "choice":
            assert d == "answer text 3"


def test_described_less_ids_become_descriptions():
    base = _choice(desc=False)
    for j in range(30):
        v = variant(base, j, 3, {})
        kind, d = gold_text(v)
        if kind == "choice":
            assert d in ("option_3", None)


def test_yes_no_truth_preserved_across_type_switch():
    base = _choice(k=2, gold=0)
    base["questions"]["q1"]["criteria"] = {"yes": "It holds.", "no": "It does not hold."}
    base["labels"]["q1"] = "yes"
    seen = set()
    for j in range(60):
        v = variant(base, j, 11, {})
        A.validate(v)
        (qid, q), = v["questions"].items()
        seen.add(q["type"])
        if q["type"] == "noul":
            assert v["labels"][qid] == "true"
            assert (q.get("criteria") or NOUL_DEFAULT_CRITERIA)["true"] in ("It holds.", NOUL_DEFAULT_CRITERIA["true"])
        else:
            assert q["criteria"][v["labels"][qid]] == "It holds."
    assert seen == {"noul", "choice"}


def test_score_reverse_keeps_level_description():
    base = _choice()
    base["questions"]["q1"] = {"type": "score", "instructions": "Rate it.", "criteria": ["bad", "ok", "good", "great"]}
    base["labels"]["q1"] = "2"
    rev = 0
    for j in range(40):
        v = variant(base, j, 5, {})
        A.validate(v)
        assert gold_text(v) == ("score", "good")
        rev += "scale_reverse" in v["aug"]["schema_synth"]["ops"]
    assert rev > 0


def test_context_keys_stay_top_level():
    base = _choice()
    base["state"] = {"tools": [{"name": "a", "description": "x"}, {"name": "b", "description": "y"}],
                     "dialogue": [{"role": "user", "content": "hi"}]}
    for j in range(40):
        v = variant(base, j, 9, {})
        assert isinstance(v["state"], dict) and v["state"]["tools"] == base["state"]["tools"]


def test_injected_rows_only_get_instruction_rewrites():
    base = _choice()
    base["aug"] = {"injection": {"question": "q1", "target": "option_1"}}
    for j in range(20):
        v = variant(base, j, 1, {})
        assert v["questions"] == base["questions"] and v["labels"] == base["labels"] and v["state"] == base["state"]


def test_referenced_ids_are_not_restyled():
    base = _choice(k=4, gold=2)
    base["state"]["question"] = "Pick option_2 or option_0 per the memo."
    for j in range(20):
        v = variant(base, j, 1, {})
        (qid, q), = v["questions"].items()
        assert set(q["criteria"]) <= set(base["questions"]["q1"]["criteria"])
        assert json.dumps(v["state"]).count("option_2") >= 1

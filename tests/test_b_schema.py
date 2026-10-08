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


# ---------------------------------------------------------------- v2 (review of v1, B1_SCHEMA_REVIEW.md)
import pytest

from data import b_schema as BS


@pytest.fixture
def v2():
    BS.GEN["version"] = 2
    yield
    BS.GEN["version"] = 1


def test_v2_digit_two_way_choice_never_becomes_noul(v2):
    base = _choice(k=2, gold=1)
    base["questions"]["q1"]["criteria"] = {"0": "Megan", "1": "Sarah"}
    base["labels"]["q1"] = "1"
    for j in range(80):
        v = variant(base, j, 4, {})
        (qid, q), = v["questions"].items()
        assert q["type"] == "choice" and q["criteria"][v["labels"][qid]] == "Sarah"


def test_v2_option_text_naming_ids_blocks_restyle(v2):
    base = _choice(k=4, gold=3)
    base["questions"]["q1"]["criteria"] = {"A": "red", "B": "blue", "C": "green", "D": "A and C"}
    base["labels"]["q1"] = "D"
    for j in range(30):
        v = variant(base, j, 2, {})
        (qid, q), = v["questions"].items()
        assert set(q["criteria"]) == {"A", "B", "C", "D"} and v["labels"][qid] == "D"


def test_v2_digit_ids_zero_padded_over_10(v2):
    import random
    assert BS.new_ids("digits", 12, random.Random(0), []) == [f"{i:02d}" for i in range(12)]
    assert BS.new_ids("digits1", 9, random.Random(0), []) == [str(i) for i in range(1, 10)]


def test_v2_duplicate_gold_text_deduped(v2):
    base = _choice(k=4, gold=1)
    base["questions"]["q1"]["criteria"]["option_2"] = "answer text 1"
    for j in range(20):
        v = variant(base, j, 6, {})
        (qid, q), = v["questions"].items()
        if q["type"] == "choice":
            assert [d for d in q["criteria"].values()].count("answer text 1") == 1
            assert q["criteria"][v["labels"][qid]] == "answer text 1"


def test_v2_state_keys_named_in_instructions_kept(v2):
    base = _choice()
    base["state"] = {"record_1": "first thing", "record_2": "second thing"}
    base["questions"]["q1"]["instructions"] = "For record_1: choose the best answer."
    for j in range(30):
        v = variant(base, j, 8, {})
        assert v["state"] == base["state"]

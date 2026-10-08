"""Fast checks of the B1 family generators (no downloads, no fingerprint)."""
import ast
import collections

from data.b1_families import bfcl, cruxeval
from data.b1_families.common import MAX_OPTIONS, MAX_QUESTIONS, FamilyOverlap


def test_cruxeval_gold_is_the_executed_output_and_deterministic():
    a = cruxeval.generate(7, target=200, log=lambda *_: None)
    b = cruxeval.generate(7, target=200, log=lambda *_: None)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    for r in a:
        q = r["questions"]["answer"]
        assert 2 <= len(q["criteria"]) <= MAX_OPTIONS
        gold = ast.literal_eval(q["criteria"][r["labels"]["answer"]])
        args = list(ast.literal_eval("(" + r["state"]["input"] + ",)"))
        assert cruxeval.run(r["state"]["code"], args) == gold
        others = [ast.literal_eval(v) for k, v in q["criteria"].items() if k != r["labels"]["answer"]]
        assert all(o != gold for o in others)


def test_bfcl_exactly_one_gold_and_caps():
    recs = bfcl.generate(7, target=300, log=lambda *_: None)
    pos = collections.Counter()
    for r in recs:
        assert 1 <= len(r["questions"]) <= MAX_QUESTIONS
        assert list(r["labels"].values()).count("yes") == 1
        names = [f["name"] for f in r["state"]["functions"]]
        assert [f"tool_{n}" for n in names] == list(r["questions"])
        pos[r["aug"]["gold_pos"]] += 1
    assert len(pos) > 3


def test_family_overlap_flags_shared_13gram_and_ignores_boilerplate():
    boiler = "this sentence is shared by every single row of the family and must be ignored by the checker ok"
    rows = [[boiler, f"unique item number {i} " + "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
             if i == 0 else f"{boiler} {i}"] for i in range(50)]
    fo = FamilyOverlap("t", rows)
    assert fo.hits(["x unique item number 0 alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"])
    assert not fo.hits([boiler])

"""Unit checks for the B1 draw transforms (no data files needed)."""
import collections
import random

from data import b1
from decide.schema import canonical_options


def _intent(n=77, inj=False, src="banking77"):
    crit = {f"option_{i}": f"label {i} card" for i in range(n)}
    if src == "clinc150":
        crit["option_5"] = b1.OOS_TEXT
    rec = {"id": f"m2:{src}:1", "source": src, "family": "intent_banking" if src == "banking77" else "intent_general",
           "state": {"customer_message": "my card label 3 is late"}, "labels": {"q": "option_3"},
           "questions": {"q": {"type": "choice", "instructions": "Classify.", "criteria": crit}}, "aug": {}}
    if inj:
        rec["aug"]["injection"] = {"question": "q", "target": "option_9"}
    return rec


def _gold_text(r):
    q = r["questions"]["q"]
    return q["criteria"][r["labels"]["q"]]


def test_k16_view_keeps_gold_oos_and_has_16_options():
    for src in ("banking77", "clinc150"):
        v = b1.intent_k16_view(_intent(src=src), 0)
        crit = v["questions"]["q"]["criteria"]
        assert len(crit) == 16 and _gold_text(v) == "label 3 card"
        assert v["id"].endswith("~view:k16") and v["aug"]["b1_view"]["parent_id"] == f"m2:{src}:1"
        if src == "clinc150":
            assert b1.OOS_TEXT in crit.values()


def test_injected_view_keeps_ids_and_target():
    v = b1.intent_k16_view_keep_ids(_intent(inj=True), 0)
    crit = v["questions"]["q"]["criteria"]
    assert len(crit) == 16 and "option_9" in crit and v["labels"]["q"] == "option_3"
    assert all(k.startswith("option_") for k in crit)


def test_rebalance_gold_position_is_uniform():
    pos = collections.Counter()
    for i in range(4000):
        r = {"id": f"m2:x:{i}", "source": "anli", "family": "nli", "state": {}, "labels": {"q": "a"},
             "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B", "c": "C"}}}}
        out = b1.rebalance(r, 0)
        ids = [o for o, _ in canonical_options(out["questions"]["q"])]
        assert out["questions"]["q"]["criteria"][out["labels"]["q"]] == "A"
        pos[ids.index(out["labels"]["q"])] += 1
    assert all(abs(pos[p] - 4000 / 3) < 150 for p in range(3)), pos


def test_catalog_view_adds_distinct_tools_and_keeps_gold():
    rec = {"id": "m2:glaive_fc:1", "source": "glaive_fc", "family": "tool_select", "labels": {"q": "k1"},
           "state": {"tools": [{"name": "get_x", "description": "x"}], "dialogue": []},
           "questions": {"q": {"type": "choice", "instructions": "?",
                               "criteria": {"k1": "get_x", "k2": b1.NO_TOOL}}}}
    donors = [{"name": f"t{i}", "description": f"d{i}"} for i in range(50)] + [{"name": "get_x", "description": "dup"}]
    v = b1.glaive_catalog_view(rec, donors, 20, 0, "catlong")
    crit = v["questions"]["q"]["criteria"]
    assert len(crit) == 20 and crit[v["labels"]["q"]] == "get_x"
    assert len(v["state"]["tools"]) == 19 and len({t["name"] for t in v["state"]["tools"]}) == 19
    assert all(k.startswith("k_") for k in crit)


def test_draw_quotas_sum_and_nest():
    assert sum(b1.WEIGHTS_45K.values()) == 45000 and sum(b1.WEIGHTS_100K.values()) == 100000
    assert set(b1.WEIGHTS_45K) == set(b1.WEIGHTS_100K) == set(b1.FAMILIES)
    assert all(b1.WEIGHTS_45K[f] <= b1.WEIGHTS_100K[f] for f in b1.FAMILIES)

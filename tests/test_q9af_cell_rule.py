"""Pins every literal constant of the pre-registered q9af_cell decision rule (jobs/q9af_cell/rule.py; LESSONS L-07)
and exercises its decision table. Changing a constant must fail here: amend the rule in a new dated commit BEFORE the
run it governs."""

from __future__ import annotations

import pytest

from jobs.q9af_cell import rule as R


def test_constants_are_pinned():
    assert R.RULE_ID == "q9af_cell/rule-v1"
    assert R.EDITION_BINDING == "0.2.1" and R.EDITION_REPORTED == "0.3"
    assert R.CLEF_FLASH_SUBSET_021 == 55.78
    assert R.CLEF_FLASH_FULL_DEV_021 == 56.79
    assert R.CLEF_FLASH_FULL_DEV_32K == 57.35
    assert R.STOCK_Q9_SUBSET_021 == 38.27
    assert R.PROMOTE_MIN_INDEX == 50.0
    assert R.CONTINUE_MIN_INDEX == 44.0
    assert R.FOLD_BELOW_INDEX == 40.0
    assert R.MIN_GAIN_PTS == 3.0
    assert R.RANK_MIN_GAIN_PTS == 0.0
    assert R.TRAIN_STOP_CODES == (21, 22, 23, 24, 25, 27, 28, 29, 35)
    assert R.TRAIN_TOO_SLOW == 26
    assert R.GATES_ACCEPT == 0 and R.GATES_G13_REJECT == (41, 42)
    assert R.VERDICTS == ("NO_MEASUREMENT", "THROUGHPUT", "HYBRID_REPORT", "FOLD", "ESCALATE", "PROMOTE", "CONTINUE")
    assert R.RANK_VERDICTS == ("R256", "R64", "NO_MEASUREMENT")


def test_amendment_a1_constants_are_pinned():
    assert R.AMENDMENTS == ("A1",)
    assert R.GATES_G12_REJECT == (40, 42) and R.GATES_UNDERPOWERED == 45
    assert R.BOOTSTRAP_B == 2000 and R.BOOTSTRAP_SEED == 0
    assert R.VERDICTS_A1 == R.VERDICTS + ("INCONCLUSIVE",)
    assert set(R.ACTIONS) == set(R.VERDICTS_A1) and all(isinstance(v, str) and v for v in R.ACTIONS.values())


def test_every_reachable_verdict_maps_to_an_action():
    import itertools

    seen = set()
    for rc, gates, af, tp, idx, lo in itertools.product((0, 24, 26, None), (0, 40, 41, 42, 43, 44, 45), (True, False),
                                                       (True, False), (30.0, 45.0, 52.0, 60.0), (-1.0, 2.0)):
        v = R.decide(_m(train_rc=rc, gates_rc=gates, attention_free=af, train_pass=tp, full_dev_child=idx,
                        full_dev_gain={"point": lo + 1, "lo": lo, "hi": lo + 4}))["verdict"]
        assert v in R.ACTIONS, v
        seen.add(v)
    assert seen == set(R.VERDICTS_A1)


def _m(**over):
    m = {"export_pinned": True, "attention_free": True, "train_rc": 0, "train_pass": True, "gates_rc": 0,
         "gates_stages": ["g12", "g13"], "full_dev_child": 52.0, "full_dev_gain": {"point": 12.0, "lo": 10.0, "hi": 14.0},
         "bootstrap": {"B": 2000, "seed": 0}}
    m.update(over)
    return m


@pytest.mark.parametrize("over,verdict", [
    ({}, "PROMOTE"),
    ({"export_pinned": False}, "NO_MEASUREMENT"),
    ({"train_rc": 26}, "THROUGHPUT"),
    ({"train_rc": 24}, "NO_MEASUREMENT"),
    ({"train_rc": None}, "NO_MEASUREMENT"),
    ({"full_dev_gain": {"point": 1.0, "lo": float("nan"), "hi": 2.0}}, "NO_MEASUREMENT"),
    ({"gates_stages": ["g12"]}, "NO_MEASUREMENT"),
    ({"attention_free": False}, "HYBRID_REPORT"),
    ({"full_dev_gain": {"point": 1.0, "lo": -1.0, "hi": 2.9}}, "FOLD"),
    ({"full_dev_child": 39.9}, "FOLD"),
    ({"full_dev_child": 56.79}, "ESCALATE"),
    ({"gates_rc": 41}, "ESCALATE"),
    ({"gates_rc": 43, "full_dev_child": 45.0}, "CONTINUE"),
    ({"gates_rc": 0, "full_dev_child": 49.99}, "CONTINUE"),
    ({"gates_rc": 40, "full_dev_child": 43.99}, "FOLD"),
    # Amendment A1
    ({"gates_rc": 40}, "FOLD"),  # (a) a G12 reject never CONTINUE / PROMOTE
    ({"gates_rc": 40, "full_dev_child": 45.0}, "FOLD"),
    ({"gates_rc": 42, "full_dev_gain": {"point": 1.0, "lo": -1.0, "hi": 4.0}}, "FOLD"),
    ({"gates_rc": 42}, "ESCALATE"),  # both rejected with a positive capability gain: the user decides
    ({"gates_rc": 40, "full_dev_child": 57.0}, "ESCALATE"),
    ({"gates_rc": 45}, "INCONCLUSIVE"),  # (b) underpowered G13
    ({"gates_rc": 45, "full_dev_child": 45.0}, "INCONCLUSIVE"),
    ({"train_pass": False}, "CONTINUE"),  # (c) a failed training-pass rule cannot PROMOTE
    ({"bootstrap": {"B": 1000, "seed": 0}}, "NO_MEASUREMENT"),  # (d) B and seed are part of the rule
    ({"bootstrap": None}, "NO_MEASUREMENT"),
    ({"train_pass": False, "full_dev_gain": {"point": 0.5, "lo": -1.0, "hi": 2.0}}, "FOLD"),
])
def test_decision_table(over, verdict):
    assert R.decide(_m(**over))["verdict"] == verdict


def test_hybrid_is_never_promoted_even_when_it_beats_clef_flash():
    assert R.decide(_m(attention_free=False, full_dev_child=60.0))["verdict"] == "HYBRID_REPORT"


def test_rank_pair():
    ok = {"verdict": "PROMOTE"}
    assert R.compare_ranks(ok, ok, {"point": 2.0, "lo": 0.5, "hi": 3.5})["verdict"] == "R256"
    assert R.compare_ranks(ok, ok, {"point": 0.5, "lo": -0.5, "hi": 1.5})["verdict"] == "R64"
    assert R.compare_ranks(ok, {"verdict": "THROUGHPUT"}, {"point": 9, "lo": 8, "hi": 10})["verdict"] == "NO_MEASUREMENT"
    assert R.compare_ranks(ok, ok, None)["verdict"] == "NO_MEASUREMENT"
    assert R.compare_ranks(ok, {"verdict": "INCONCLUSIVE"}, {"point": 9, "lo": 8, "hi": 10})["verdict"] == "NO_MEASUREMENT"

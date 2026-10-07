"""Pins the pre-registered proxy-loop rule (train/proxy_rule.py) and the HOLDOUT split it names. CPU, no data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train import proxy_rule as R

ROOT = Path(__file__).resolve().parents[1]


def test_constants_are_pinned():
    assert R.RULE_VERSION == "proxy-rule-v1"
    assert R.GO_THRESHOLD == 50.0
    assert R.TAU_MIN == 0.6
    assert R.MIN_TRANSFER_RECIPES == 4
    assert R.HOLDOUT_SEED == 20261008 and R.HOLDOUT_N == 2000
    assert R.BOOTSTRAP_B == 2000 and R.BOOTSTRAP_SEED == 0
    assert R.SELECT_SEEDS == (0,) and R.HOLDOUT_SEEDS == (0, 1, 2)
    assert R.SELECT_IDS_SHA256 == "6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3"
    assert not hasattr(R, "IN_DOMAIN_CATALOG_IDS")  # withdrawn by A1(a)
    assert R.CANDIDATE_FAMILIES == ("fork", "mixer")
    assert [r["id"] for r in R.RECIPES] == [f"R{i:02d}" for i in range(16)]


def test_amendment_a1_constants_are_pinned():
    assert R.AMENDMENTS == ("A1",)
    assert R.FAMILIES_V1_SHA256 == "e0e4bd5f76ca928f2a05c3f52150592aed462b4018350346eb723e9b40096f46"
    assert R.DOMAIN_COLUMNS == ("in_domain", "new_domain", "new_domain_frozen")
    assert R.IN_DOMAIN == (1, 2, 3, 4, 5, 11, 12, 21, 25, 28, 29, 30, 36, 37, 39, 42, 45, 57, 62)
    assert R.NEW_DOMAIN == (9, 20, 22, 23, 31, 32, 33, 38, 40, 41, 43, 44, 48, 50, 56, 58, 59, 61, 64)
    assert R.NEW_DOMAIN_FROZEN == (9, 20, 22, 31, 32, 33, 44, 58, 64)
    assert set(R.NEW_DOMAIN_FROZEN) <= set(R.NEW_DOMAIN) and not set(R.IN_DOMAIN) & set(R.NEW_DOMAIN)
    assert len(R.IN_DOMAIN) + len(R.NEW_DOMAIN) == 38
    assert R.TRAIN_STEPS == 2000 and R.GRAD_ACCUM == 8 and R.MATCHED_TRAIN_ROWS == 5000
    assert R.VERDICT_EDITION == "0.2.1" and R.REPORT_EDITIONS == ("0.2.1", "0.3")
    assert R.STOCK_CHECK_TOL == 0.3
    assert R.GO_THRESHOLD == 50.0  # unchanged by A1
    ctl = {r["id"]: r for r in R.RECIPES[14:]}
    assert ctl["R14"]["family"] == "mixer" and ctl["R14"]["train_rows"] == 5000 and ctl["R14"]["arm"] == "stock"
    assert ctl["R15"]["family"] == "fork" and ctl["R15"]["train_rows"] == 5000
    assert R.candidate_ids() == tuple(f"R{i:02d}" for i in range(1, 16))
    assert R.stock_check(38.5)["ok"] and not R.stock_check(37.9)["ok"]
    assert R.domain_of(9) == ("new_domain", "new_domain_frozen") and R.domain_of(2) == ("in_domain",)
    assert R.domain_of(23) == ("new_domain",)


def test_a1_domain_columns_equal_harness_families_v1():
    pytest.importorskip("decision_index")
    from harness import families as F

    assert F.sha256() == R.FAMILIES_V1_SHA256
    for col, grp in (("IN_DOMAIN", "in_domain"), ("NEW_DOMAIN", "new_domain_proxy"),
                     ("NEW_DOMAIN_FROZEN", "new_domain_frozen")):
        assert tuple(sorted(F.members(grp, R.VERDICT_EDITION))) == getattr(R, col), col
    assert R.RECIPES[0]["family"] == "attn_ref"  # the anchor is first and is a reference only


def test_holdout_file_is_disjoint_and_pinned():
    d = json.loads((ROOT / R.HOLDOUT_SUBSET).read_text(encoding="utf-8"))
    s = json.loads((ROOT / R.SELECT_SUBSET).read_text(encoding="utf-8"))
    assert d["ids_sha256"] == "8ae9c9fbe3a467d4cf3828868a0066075949afc2488193f8ee6f15c91a6b5ca2"
    assert d["seed"] == R.HOLDOUT_SEED and d["disjoint_from_ids_sha256"] == s["ids_sha256"] == R.SELECT_IDS_SHA256
    assert not set(d["run_ids"]) & set(s["run_ids"])
    assert d["summary"]["benchmarks"] == s["summary"]["benchmarks"] == 38
    assert len(d["run_ids"]) >= R.HOLDOUT_N  # whole groups: the stratified quota can overshoot slightly (2,014)


def test_holdout_rebuilds_from_the_committed_manifest():
    from jobs.q9_feature_cache.splits import check

    res = check()
    assert res["ok"], res


def test_choose_never_picks_the_attention_reference():
    scores = {"R00": 60.0, "R01": 48.0, "R11": 45.0}
    assert R.choose(scores) == "R01"
    assert R.choose({"R00": 60.0}) is None
    assert R.choose({"R01": 50.0, "R11": 50.0}) == "R01"  # tie -> earlier id


@pytest.mark.parametrize("chosen,point,complete,want", [
    ("R01", 50.0, True, "go"), ("R01", 49.99, True, "escalate"), ("R00", 70.0, True, "no-measurement"),
    (None, 70.0, True, "no-measurement"), ("R11", None, True, "no-measurement"), ("R11", 55.0, False, "no-measurement"),
])
def test_verdict(chosen, point, complete, want):
    v = R.verdict(chosen, point, holdout_complete=complete)
    assert v["verdict"] == want
    if want in ("go", "escalate"):
        assert v["edition"] == "0.2.1" and v["scope"] == R.VERDICT_SCOPE
        assert v["escalate_to"] == (None if want == "go" else "user")


def test_adopt_ranking():
    p = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    assert R.adopt_ranking(p, {"A": 40, "B": 30, "C": 20, "D": 10}, ("A", "D"))["adopt"]
    assert not R.adopt_ranking(p, {"A": 40, "B": 30, "C": 20}, ("A", "C"))["adopt"]  # < 4 recipes
    assert not R.adopt_ranking(p, {"A": 10, "B": 30, "C": 20, "D": 40}, ("A", "D"))["adopt"]  # pair disagrees
    r = R.adopt_ranking(p, {"A": 40, "B": 10, "C": 30, "D": 20}, ("A", "D"))
    assert r["tau"] == pytest.approx(1 / 3) and not r["adopt"]  # tau below TAU_MIN
    from train.proxy import kendall_tau

    assert R.kendall_tau([1, 2, 3, 5], [2, 1, 3, 4]) == kendall_tau([1, 2, 3, 5], [2, 1, 3, 4])

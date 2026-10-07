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
    assert [r["id"] for r in R.RECIPES] == [f"R{i:02d}" for i in range(18)]  # R16-R17: amendment A2


def test_amendment_a1_constants_are_pinned():
    assert R.AMENDMENTS[0] == "A1"
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
    ctl = {r["id"]: r for r in R.RECIPES[14:16]}
    assert ctl["R14"]["family"] == "mixer" and ctl["R14"]["train_rows"] == 5000 and ctl["R14"]["arm"] == "stock"
    assert ctl["R15"]["family"] == "fork" and ctl["R15"]["train_rows"] == 5000
    assert set(R.candidate_ids()) >= {f"R{i:02d}" for i in range(1, 16)}
    kw = {"unmasked_point": 39.04, "masked_ids": [], "n_rows": 2008, "masked_problems": [], "by_status": {}}  # A3
    assert R.stock_check(38.5, **kw)["within_tol"] and not R.stock_check(37.9, **kw)["within_tol"]
    assert R.domain_of(9) == ("new_domain", "new_domain_frozen") and R.domain_of(2) == ("in_domain",)
    assert R.domain_of(23) == ("new_domain",)


def test_amendment_a2_constants_are_pinned():
    """A2 (pre-data): two KD candidates on top of R01, the pinned Clef-flash teacher, nothing else changed."""
    assert R.AMENDMENTS[:2] == ("A1", "A2")
    assert R.RULE_VERSION == "proxy-rule-v1" and R.GO_THRESHOLD == 50.0  # same rule, same threshold
    assert R.HOLDOUT_SEEDS == (0, 1, 2) and R.SELECT_SEEDS == (0,)
    assert R.TRAIN_STEPS == 2000 and R.GRAD_ACCUM == 8
    kd = {r["id"]: r for r in R.RECIPES[16:]}
    assert list(kd) == ["R16", "R17"] == list(R.KD_RECIPES)
    r01 = dict(R.RECIPES[1])
    for rid, (alpha, t) in (("R16", (0.5, 1.0)), ("R17", (0.9, 2.0))):
        r = dict(kd[rid])
        assert r.pop("kd") == {"alpha": alpha, "temperature": t}
        assert {k: v for k, v in r.items() if k not in ("id", "note")} == \
            {k: v for k, v in r01.items() if k not in ("id", "note")}  # = R01 + KD, nothing else
    assert R.KD_CONTROL == "R01"
    assert R.candidate_ids() == tuple(f"R{i:02d}" for i in range(1, 18))
    assert R.TEACHER == {
        "job": "job-tnrwu", "artifact": "art-ggrs8", "release": "clef-flash", "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "backend_revision": "clef:Cloudflare/clef-flash@17f0b0ad64ef;jsm=0e304cf7c650;"
                            "enc=untrusted-split-special+cut-leaking-added-v1;cfg=05bdad56;adapter=0.2.0",
        "sha256": "a88b4fe0f6a79a8fc48bdea51391524bd4f5da885705461a543becf81c81e7f7",
        "drawn_ids_sha256": "2bbbf1808c0bdd6cf0ba3ecd236cf9a64110c5e429e22f8b8e9cd2df7507556b",
        "rows": 24000, "status_counts": {"ok": 23458, "exhausted": 542}}
    from train.cohort_v3 import FIRST_CELL_DRAW

    assert R.TEACHER["drawn_ids_sha256"] == FIRST_CELL_DRAW["drawn_ids_sha256"]  # the stock_train cache's draw


def test_amendment_a3_stock_check_masking_and_everything_else_unchanged():
    """A3 (after the stock-check failed, before any recipe / select / holdout data): only the stock-check changes."""
    assert R.AMENDMENTS == ("A1", "A2", "A3")
    # unchanged by A3
    assert R.RULE_VERSION == "proxy-rule-v1" and R.GO_THRESHOLD == 50.0
    assert R.STOCK_CHECK_TOL == 0.3 and R.STOCK_Q9_SELECT == 38.27 and R.CLEF_FLASH_SELECT == 55.78
    assert R.TAU_MIN == 0.6 and R.MIN_TRANSFER_RECIPES == 4
    assert R.SELECT_SEEDS == (0,) and R.HOLDOUT_SEEDS == (0, 1, 2) and R.BOOTSTRAP_B == 2000 and R.BOOTSTRAP_SEED == 0
    assert R.TRAIN_STEPS == 2000 and R.GRAD_ACCUM == 8 and R.MATCHED_TRAIN_ROWS == 5000
    assert R.VERDICT_EDITION == "0.2.1" and R.REPORT_EDITIONS == ("0.2.1", "0.3")
    assert R.SELECT_IDS_SHA256 == "6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3"
    assert R.HOLDOUT_SEED == 20261008 and R.HOLDOUT_N == 2000
    assert R.candidate_ids() == tuple(f"R{i:02d}" for i in range(1, 18))
    assert R.verdict("R01", 50.0)["verdict"] == "go" and R.verdict("R01", 49.99)["verdict"] == "escalate"
    assert R.verdict("R01", 49.99)["escalate_to"] == "user" and R.verdict("R00", 99.0)["verdict"] == "no-measurement"
    # A3 constants
    assert R.STOCK_REFERENCE_SHA256 == "646b61490a37b3aa8887a57bb1a6797f8254c0136a63a0c3cf18d00f90497beb"
    assert R.STOCK_CHECK_REFERENCE_OK == "ok" and R.STOCK_CHECK_MAX_MASKED_FRAC == 0.05
    assert R.STOCK_CHECK_MASKED_N == 65
    assert R.STOCK_CHECK_MASKED_IDS_SHA256 == "d3465d44c3073b35685cab7bf5e827dfef3d85b36d33004767dc47c115466629"
    assert R.STOCK_CHECK_PROB_SUM_TOL == 0.01 and R.REFERENCE_MAX_INPUT_TOKENS == 16384


def _a3_kw(**over):
    kw = {"unmasked_point": 38.27, "masked_ids": [], "n_rows": 2008, "masked_problems": [], "by_status": {}}
    kw.update(over)
    return kw


def test_a3_pinned_masked_set_is_the_receipts_and_the_observed_failure_passes():
    doc = json.loads((ROOT / "receipts/2026-10-07-D-q9-proxy-stock-check/masked_ids.json").read_text(encoding="utf-8"))
    assert doc["reference_sha256"] == R.STOCK_REFERENCE_SHA256
    m = doc["masked"]
    assert len(m) == R.STOCK_CHECK_MASKED_N and R.masked_ids_sha256(list(m)) == R.STOCK_CHECK_MASKED_IDS_SHA256
    st = {}
    for rid, v in m.items():
        st[v] = st.get(v, 0) + 1
    assert st == {"error": 58, "unsupported": 7}
    assert all(r.startswith("22:POP909") for r, v in m.items() if v == "error")
    assert all(r.split(":")[1] in ("ToolRet-retrieval", "BRIGHT-retrieval") for r, v in m.items() if v == "unsupported")
    by = {"error": {"n": 58, "point_only_this_masked": 38.984}, "unsupported": {"n": 7, "point_only_this_masked": 38.375}}
    rep = R.stock_check(38.317, **_a3_kw(unmasked_point=39.042, masked_ids=list(m), by_status=by))
    assert rep["ok"] and rep["within_tol"] and rep["mask_cap_ok"] and rep["masked_ids_pinned_ok"]
    assert rep["masked_rows_valid"] and rep["amendment"] == "A3"
    assert abs(rep["delta"] - 0.047) < 1e-9 and abs(rep["unmasked_delta"] - 0.772) < 1e-9
    assert rep["masked"]["n"] == 65 and rep["masked"]["ids"] == sorted(m) and rep["masked"]["n_rows"] == 2008
    assert rep["masked"]["ids_sha256"] == R.STOCK_CHECK_MASKED_IDS_SHA256
    assert abs(rep["by_reference_status"]["unsupported"]["contribution"] - 0.667) < 1e-9
    assert abs(rep["by_reference_status"]["error"]["contribution"] - 0.058) < 1e-9
    # error-only masking would still fail: 38.984 - 38.27 = +0.714 > 0.3
    assert not R.stock_check(38.984, **_a3_kw(unmasked_point=39.042, masked_ids=list(m)))["within_tol"]
    # fail closed: an invalid cached answer on a masked row, or a different masked set
    bad = R.stock_check(38.317, **_a3_kw(unmasked_point=39.042, masked_ids=list(m), masked_problems=["x: NaN"]))
    assert not bad["ok"] and bad["within_tol"] and not bad["masked_rows_valid"]
    other = R.stock_check(38.317, **_a3_kw(unmasked_point=39.042, masked_ids=list(m)[:64]))
    assert not other["ok"] and not other["masked_ids_pinned_ok"]


def test_a3_tolerance_and_cap_unchanged(monkeypatch):
    monkeypatch.setattr(R, "STOCK_CHECK_MASKED_N", 0)
    monkeypatch.setattr(R, "STOCK_CHECK_MASKED_IDS_SHA256", R.masked_ids_sha256([]))
    assert not R.stock_check(38.58, **_a3_kw())["ok"]
    assert R.stock_check(37.98, **_a3_kw())["ok"]
    # fail closed above 5% masked even when the delta is within tolerance
    ok100 = R.stock_check(38.27, **_a3_kw(masked_ids=[f"m{i}" for i in range(100)]))  # 100 / 2008 = 4.98%
    assert ok100["mask_cap_ok"]
    over = R.stock_check(38.27, **_a3_kw(masked_ids=[f"m{i}" for i in range(101)]))
    assert not over["ok"] and over["within_tol"] and not over["mask_cap_ok"]
    with pytest.raises(ValueError):
        R.stock_check(38.27, **_a3_kw(n_rows=0))
    with pytest.raises(TypeError):
        R.stock_check(38.27)  # the masked report is required


def test_a3_masked_row_problems_fail_closed():
    ok_choice = {"type": "choice", "choice": "B", "confidence": 0.7, "probabilities": {"A": 0.3, "B": 0.7}}
    good = [{"run_id": "a", "status": "ok", "response": {"answers": {"q1": ok_choice, "q2": {"type": "noul", "noul": 0.4}}}}]
    assert R.masked_row_problems(good, ["a"]) == []
    cases = [
        {"run_id": "a", "status": "error"},
        {"run_id": "a", "status": "ok", "response": {"answers": {}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {**ok_choice, "probabilities": {"A": float("nan"), "B": 0.7}}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {**ok_choice, "probabilities": {"A": 0.5, "B": 0.7}}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {**ok_choice, "probabilities": {"A": -0.1, "B": 1.1}}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {**ok_choice, "choice": "C"}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {"type": "noul", "noul": float("inf")}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {"type": "noul", "noul": 1.5}}}},
        {"run_id": "a", "status": "ok", "response": {"answers": {"q": {"type": "rank", "rank": [1]}}}},
    ]
    for c in cases:
        assert R.masked_row_problems([c], ["a"]), c
    assert R.masked_row_problems(good, ["missing"])  # a masked id with no cached row
    assert R.masked_row_problems([cases[0]], []) == []  # only masked rows are checked


def test_a3_stock_check_mask_masks_both_sides_and_fails_closed_on_id_mismatch():
    ids = ["a", "b", "c", "d"]
    cached = [{"run_id": i, "status": "ok", "response": {"model": "cache", "i": i}} for i in ids]
    ref = [{"run_id": "a", "status": "ok"}, {"run_id": "b", "status": "error", "error": {"code": "backend_error"}},
           {"run_id": "c", "status": "unsupported"}, {"run_id": "d", "status": "ok"}]
    rows, masked = R.stock_check_mask(cached, list(reversed(ref)), ids)
    assert masked == ["b", "c"]
    assert R.stock_check_mask(cached, ref, ids, ("error",))[1] == ["b"]
    assert R.stock_check_mask(cached, ref, ids, ("unsupported",))[1] == ["c"]
    assert [r["run_id"] for r in rows] == ids  # cached order kept
    assert rows[0] == cached[0] and rows[3] == cached[3]  # reference-ok rows: the cache's own answer
    assert rows[1] == ref[1] and rows[2] == ref[2]  # masked rows: identical to the reference on both sides
    for bad_ref in (ref[:3], ref + [ref[0]], ref[:3] + [{"run_id": "zz", "status": "ok"}]):
        with pytest.raises(ValueError):
            R.stock_check_mask(cached, bad_ref, ids)
    with pytest.raises(ValueError):
        R.stock_check_mask(cached[:3], ref, ids)


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


def test_best_worst_screening_pair_cannot_adopt_even_with_perfect_tau():
    proxy = {"best": 4, "middle1": 3, "middle2": 2, "worst": 1}
    real = {"best": 40, "worst": 10, "unmatched1": 30, "unmatched2": 20}
    result = R.adopt_ranking(proxy, real, ("best", "worst"))
    assert result["pair_agrees"] and result["tau"] == 1.0
    assert result["n_recipes"] == 2
    assert not result["enough_recipes"] and not result["adopt"]
    real.update(middle1=30, middle2=20)
    result = R.adopt_ranking(proxy, real, ("best", "worst"))
    assert result["n_recipes"] == 4 and result["enough_recipes"] and result["adopt"]

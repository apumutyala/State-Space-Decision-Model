"""Item-level scorer: hand-computed values for every formula, and self-consistency with the board recompute."""

import math

import numpy as np
import pytest

from jevkit import board, spec
from jevkit import scoring as SC

from ._fixtures import ch_ans, choice, good_answers, mixed_set, nl_ans, noul, sc_ans, score


def onehot(it, lab, p=1.0):
    rest = (1 - p) / (it.k - 1)
    return ch_ans({x: (p if x == lab else rest) for x in it.labels})


def agg(items, answers, **kw):
    return SC.aggregate(SC.build_table(items, answers, kw.pop("exclude", ())), **kw)


def test_choice_cc_chance_corrected():
    its = [choice(f"c{i}", k=4, gold=0) for i in range(4)]
    ans = {it.id: onehot(it, "o0" if n < 3 else "o1") for n, it in enumerate(its)}
    r = agg(its, ans)["intelligence"]
    assert r["per_type_split"]["open|choice"]["tiers"]["standard"] == pytest.approx(100 * (0.75 - 0.25) / 0.75)
    assert r["I_open"] == pytest.approx(200 / 3) and r["I"] == pytest.approx(200 / 3)
    assert r["basis"].startswith("open only") and r["penalty"] == 1.0


def test_tier_weights_renormalised_over_present_tiers():
    its = [choice(f"s{i}", k=4, tier="standard") for i in range(4)] + [choice(f"h{i}", k=2, tier="hard") for i in range(2)]
    ans = {it.id: onehot(it, "o0") for it in its[:3]}
    ans.update({its[3].id: onehot(its[3], "o1"), its[4].id: onehot(its[4], "o0"), its[5].id: onehot(its[5], "o1")})
    cc = agg(its, ans)["intelligence"]["per_type_split"]["open|choice"]
    assert cc["tiers"]["hard"] == pytest.approx(0.0)
    assert cc["cc"] == pytest.approx((0.2 * 200 / 3 + 0.4 * 0.0) / 0.6)


def test_noul_decisive_rule():
    its = [noul("a", "yes"), noul("b", "yes"), noul("c", "no"), noul("d", "no")]
    ans = {"a": nl_ans(0.9), "b": nl_ans(0.5), "c": nl_ans(0.1), "d": nl_ans(0.85)}
    assert agg(its, ans)["intelligence"]["per_type_split"]["open|noul"]["tiers"]["standard"] == pytest.approx(0.0)
    ans["d"] = nl_ans(0.2)
    assert agg(its, ans)["intelligence"]["per_type_split"]["open|noul"]["tiers"]["standard"] == pytest.approx(50.0)


def test_score_nmae_vs_uniform_guess_and_invalid_is_maximal():
    a, b = score("a", k=4, gold=1), score("b", k=4, gold=1)  # chance nMAE = (1+0+1+2)/4/3 = 1/3
    r = agg([a, b], {"a": sc_ans([0, 1, 0, 0]), "b": sc_ans([0, 0, 0, 1])})
    assert r["intelligence"]["per_type_split"]["open|score"]["tiers"]["standard"] == pytest.approx(0.0)
    r = agg([a], {"a": sc_ans([0, 1, 0, 0])})
    assert r["intelligence"]["per_type_split"]["open|score"]["tiers"]["standard"] == pytest.approx(100.0)
    r = agg([a], {"a": {"type": "score", "probabilities": {"0": 1.0}}})  # invalid -> nMAE 1
    cell = r["intelligence"]["per_type_split"]["open|score"]
    assert cell["tiers"]["standard"] == pytest.approx(100 * (1 - 1 / (1 / 3)))
    assert cell["cc"] == 0.0 and r["intelligence"]["I"] == 0.0  # split x type clipped at 0


def test_gap_penalty_multiplicative():
    op = [choice(f"o{i}", k=4, split="public") for i in range(4)]
    se = [choice(f"s{i}", k=4, split="private") for i in range(4)]
    ans = {it.id: onehot(it, "o0") for it in op}
    ans.update({it.id: onehot(it, "o0" if n < 2 else "o1") for n, it in enumerate(se)})
    r = agg(op + se, ans)["intelligence"]
    io, is_ = 100.0, 100 * (0.5 - 0.25) / 0.75
    base, gap = (io + is_) / 2, io - is_
    assert r["I_open"] == pytest.approx(io) and r["I_sealed"] == pytest.approx(is_)
    assert r["penalty"] == pytest.approx(1 - (gap - spec.G_MED_V161 - 8) / 100)
    assert r["I"] == pytest.approx(base * r["penalty"])


def test_choice_ece_hard_tier_only_with_fallback():
    h1, h2 = choice("h1", k=4, tier="hard"), choice("h2", k=4, tier="hard")
    s1 = choice("s1", k=4, tier="standard")
    ans = {"h1": onehot(h1, "o0", 0.8), "h2": onehot(h2, "o1", 0.8), "s1": onehot(s1, "o0", 0.95)}
    part = agg([h1, h2, s1], ans)["calibration"]["parts"]["choice"]
    assert part["ece_hard"] == pytest.approx(0.3) and part["n_ece_hard"] == 2
    assert part["score"] == pytest.approx(40.0)
    part = agg([s1], {"s1": onehot(s1, "o0", 0.95)})["calibration"]["parts"]["choice"]
    assert part["ece_basis"].startswith("all tiers") and part["ece_hard"] == pytest.approx(0.05)


def test_noul_ece_on_p_yes_and_score_calibration():
    a, b = noul("a", "yes"), noul("b", "no")
    part = agg([a, b], {"a": nl_ans(0.9), "b": nl_ans(0.9)})["calibration"]["parts"]["noul"]
    assert part["ece"] == pytest.approx(0.4) and part["score"] == pytest.approx(20.0)
    s = score("s", k=2, gold=1)
    part = agg([s], {"s": sc_ans([0.25, 0.75])})["calibration"]["parts"]["score"]
    assert part["top_ece"] == pytest.approx(0.25) and part["nrps"] == pytest.approx(0.0625)
    assert part["score"] == pytest.approx(50 * 0.5 + 50 * (1 - 0.0625))


def test_invalid_excluded_from_calibration_but_type_still_counts():
    a, b = choice("a", k=2, tier="hard"), noul("b")
    r = agg([a, b], {"a": ch_ans({"o0": 2.0, "o1": 0.0}), "b": nl_ans(1.0)})
    assert r["calibration"]["parts"]["choice"]["score"] is None
    assert r["calibration"]["score"] == pytest.approx((0.0 + 100.0) / 2)
    r = agg([a, b], {"b": nl_ans(1.0)}, exclude=("b",))
    assert r["calibration"]["parts"]["noul"]["n"] == 0


def test_speed_cost_capability():
    assert SC.speed_f(0.1) == 100.0
    assert SC.speed(0.1, 0.1, self_hosted=False)["score"] == 100.0
    sp = SC.speed(0.1, 0.2, self_hosted=True)
    assert sp["p50_s_adjusted"] == pytest.approx(0.35) and sp["p95_s_adjusted"] == pytest.approx(0.55)
    assert sp["score"] == pytest.approx((SC.speed_f(0.35) + SC.speed_f(0.55)) / 2)
    assert SC.cost_score(0.001) == 100.0 and SC.cost_score(1.0) == pytest.approx(10.0)
    assert SC.eligibility(0.06, 1.2)["capability_ranked"] and not SC.eligibility(0.07, 1.2)["capability_ranked"]


def test_ece_bincount_matches_loop():
    rng = np.random.default_rng(0)
    conf, cor = rng.random(200), rng.random(200) < 0.6
    loop = 0.0
    for bb in range(10):
        m = np.minimum((conf * 10).astype(int), 9) == bb
        if m.any():
            loop += m.mean() * abs(cor[m].mean() - conf[m].mean())
    assert SC.ece(conf, cor) == pytest.approx(loop)


def test_bootstrap_deterministic_and_paired():
    its = mixed_set("public") + mixed_set("private")
    t = SC.build_table(its, good_answers(its))
    sc = {"x": (90.0, 60.0)}
    a = SC.bootstrap({"x": t}, sc, b=50, seed=16)["x"]
    b2 = SC.bootstrap({"x": t}, sc, b=50, seed=16)["x"]
    assert a == b2 and a["I"][0] <= a["I"][1]
    t2 = SC.build_table(its, good_answers(its, conf=0.6))
    p = SC.bootstrap({"x": t, "y": t2}, {"x": (90.0, 60.0), "y": (90.0, 60.0)}, b=30, seed=16)
    assert "x - y" in p["_pairs"]
    t3 = SC.build_table(its[:-1], good_answers(its))
    with pytest.raises(ValueError):
        SC.bootstrap({"x": t, "z": t3}, {"x": (1, 1), "z": (1, 1)}, b=2)


def test_report_round_trips_through_board_recompute():
    """Our item-level report, laid out as a published row, passes every board check exactly."""
    its = mixed_set("public") + mixed_set("private")
    sp = SC.speed(0.12, 0.2, self_hosted=True)
    co = {"usd_per_1000": 0.02, "score": SC.cost_score(0.02)}
    rep = SC.score_system(its, good_answers(its), speed_axis=sp, cost_axis=co, b=0)
    parts = rep["calibration"]["parts"]
    rowd = {"key": "self", "axes": rep["axes"], "scores": rep["scores"], "capability": rep["capability"],
            "intelligence": rep["intelligence"], "calibration": rep["calibration"], "speed": sp,
            "cost": {"usd_per_1000": 0.02}, "support": rep["support"], "views": rep["views"]}
    assert set(parts) == set(spec.TYPES)
    for name, (rec, pub) in board.recompute_row(rowd).items():
        assert rec == pytest.approx(pub, abs=1e-12), name
    assert rep["intelligence"]["basis"] == "open+sealed" and 0 < rep["axes"]["intelligence"] <= 100
    assert math.isfinite(rep["scores"]["A"])

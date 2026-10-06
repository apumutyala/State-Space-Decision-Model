"""G5 selective-risk tests (CPU, synthetic). Validity on exchangeable data over many seeds, failure of the
unit-level bound under clustering (and validity of the cluster-thinned one), the feasibility floor, and detected
failure under shift."""
import json
import math

import numpy as np
import pytest

from safety_eval import data, selective as S
from safety_eval.bounds import binom_cdf, cp_lower, cp_upper
from safety_eval.standin import KeywordDecider

ALPHA, DELTA, C = 0.1, 0.1, 0.6
LAM0 = 1 - 2 * ALPHA / C  # selective risk at LAM0 is exactly ALPHA for p(wrong|s) = C (1 - s), s ~ U(0, 1)


def true_risk(lam, c=C, shift=0.0):
    xs = np.linspace(lam, 1, 4001)
    return float(np.trapezoid(np.minimum(1, c * (1 - xs) + shift), xs) / (1 - lam))


def gen(rng, n, c=C, shift=0.0):
    s = rng.random(n)
    return s, rng.random(n) < np.minimum(1, c * (1 - s) + shift)


def gen_clustered(rng, n_clusters, size, c=C):
    """Half the clusters err at 2c(1-s), half never: the population risk is c(1-s) but units are correlated."""
    eff = rng.random(n_clusters) < 0.5
    s = rng.random(n_clusters * size)
    cl = np.repeat(np.arange(n_clusters), size)
    w = rng.random(s.size) < np.where(eff[cl], np.minimum(1, 2 * c * (1 - s)), 0.0)
    return s, w, [str(x) for x in cl]


# ------------------------------------------------------------------ exact bounds
def test_binomial_and_clopper_pearson():
    for n in (1, 5, 17):
        for k in range(n + 1):
            for p in (0.03, 0.3, 0.8):
                want = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
                assert binom_cdf(k, n, p) == pytest.approx(want, abs=1e-12)
    assert cp_upper(0, 59, 0.05) == pytest.approx(1 - 0.05 ** (1 / 59), abs=1e-9)
    assert cp_lower(10, 10, 0.05) == pytest.approx(0.05 ** (1 / 10), abs=1e-9)
    assert cp_upper(3, 3, 0.05) == 1.0 and cp_lower(0, 9, 0.05) == 0.0
    lo, hi = cp_lower(30, 100, 0.05), cp_upper(30, 100, 0.05)
    assert 1 - binom_cdf(29, 100, lo) == pytest.approx(0.05, abs=1e-9) and binom_cdf(30, 100, hi) == pytest.approx(0.05, abs=1e-9)


def test_feasibility_floor():
    assert S.feasibility_floor(0.05, 0.05) == 0.0
    assert S.feasibility_floor(0.30, 0.10) == pytest.approx(0.2 / 0.9)
    assert S.feasibility_floor(0.50, 0.05, max_loss=1.0) == pytest.approx(0.45 / 0.95)
    assert S.feasibility_floor(1.0, 0.0) == 1.0
    # identity on samples: any threshold whose empirical acted risk is <= alpha abstains on >= the empirical floor
    rng = np.random.default_rng(0)
    for _ in range(50):
        s, w = gen(rng, 300, c=rng.uniform(0.2, 1.0))
        mu = w.mean()
        for lam in np.linspace(0, 1, 41):
            a = s >= lam
            if a.any() and w[a].mean() <= ALPHA:
                assert 1 - a.mean() >= S.feasibility_floor(mu, ALPHA) - 1e-12


def test_infeasible_stratum_always_abstains_and_reports_floor():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    w = rng.random(2000) < 0.5  # uninformative score, base risk 0.5
    r = S.ltt_selective(s, w, alpha=0.05, delta=0.05, max_abstention=0.3)
    assert r["lam_hat"] is None and r["chosen"] is None
    assert r["feasibility"]["floor"] == pytest.approx((w.mean() - 0.05) / 0.95)
    assert r["feasibility"]["infeasible_within_max_abstention"] is True
    json.dumps(r)


# ------------------------------------------------------------------ validity on exchangeable data
def test_pvalue_validity_at_the_boundary_over_seeds():
    """Single pre-registered lam where the true selective risk equals alpha: H0 is (just) true, so the rejection
    frequency over seeds must be <= delta; it is also close to delta (the test is not vacuous)."""
    assert true_risk(LAM0) == pytest.approx(ALPHA, abs=1e-6)
    N = 400
    rej = sum(S.ltt_selective(*gen(np.random.default_rng(seed), 1500), ALPHA, DELTA, grid=(LAM0,))["lam_hat"]
              is not None for seed in range(N))
    se = math.sqrt(DELTA * (1 - DELTA) / N)
    assert rej / N <= DELTA + 2.5 * se
    assert rej / N >= 0.03


FS_GRID = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)


@pytest.mark.parametrize("method", ["bonferroni", "fixed_sequence"])
def test_selective_risk_bound_holds_over_seeds(method):
    N, viol, cov = 200, 0, []
    kw = {"grid": FS_GRID} if method == "fixed_sequence" else {}
    for seed in range(N):
        r = S.ltt_selective(*gen(np.random.default_rng(1000 + seed), 3000), ALPHA, DELTA, method=method, **kw)
        if r["lam_hat"] is not None:
            viol += true_risk(r["lam_hat"]) > ALPHA
            cov.append(r["chosen"]["coverage"])
            assert r["floor_consistent"]
    assert viol / N <= DELTA
    assert len(cov) > 0.9 * N and np.mean(cov) > 0.15  # non-trivial coverage (oracle at alpha: 1/3)


def test_fixed_sequence_refuses_default_grid():
    with pytest.raises(ValueError):
        S.ltt_selective(*gen(np.random.default_rng(0), 100), ALPHA, DELTA, method="fixed_sequence")


def test_clustered_unit_bound_fails_and_cluster_thinned_bound_holds():
    N, thin, unit = 400, 0, 0
    for seed in range(N):
        s, w, cl = gen_clustered(np.random.default_rng(seed), 200, 20)
        r = S.ltt_selective(s, w, ALPHA, DELTA, grid=(LAM0,), clusters=cl, thin_seed=seed)
        thin += r["lam_hat"] is not None
        unit += r["unit_level_unsafe"]["lam_hat"] is not None
        assert r["n_s"] == 200 and r["unit_of_bound"].startswith("cluster")
    se = math.sqrt(DELTA * (1 - DELTA) / N)
    assert thin / N <= DELTA + 2.5 * se  # valid
    assert unit / N > DELTA + 3 * se  # the unit-level binomial bound is anti-conservative under clustering


def test_cluster_split_keeps_clusters_whole():
    cl = [f"c{i % 37}" for i in range(1000)]
    m = S.cluster_split(cl, 0.5)
    by = {}
    for c, x in zip(cl, m):
        by.setdefault(c, set()).add(bool(x))
    assert all(len(v) == 1 for v in by.values())
    assert 0.2 < m.mean() < 0.8
    idx = S.thin_by_cluster(cl, seed=3)
    assert len(idx) == 37 and len({cl[i] for i in idx}) == 37


def test_calibrate_strata_is_per_stratum():
    rng = np.random.default_rng(5)
    s1, w1 = gen(rng, 3000, c=0.3)
    s2, w2 = gen(rng, 3000, c=1.0)
    res = S.calibrate_strata(np.r_[s1, s2], np.r_[w1, w2], ["easy"] * 3000 + ["hard"] * 3000, alpha=ALPHA, delta=DELTA)
    assert set(res) == {"easy", "hard"}
    cov = {k: (v["chosen"] or {"coverage": 0.0})["coverage"] for k, v in res.items()}
    assert cov["easy"] > cov["hard"]
    assert res["hard"]["feasibility"]["floor"] > res["easy"]["feasibility"]["floor"]


# ------------------------------------------------------------------ shift: failure is detected, not hidden
def test_shift_violation_detected_and_no_false_alarm_when_exchangeable():
    N, false_alarm = 100, 0
    for seed in range(N):
        rng = np.random.default_rng(2000 + seed)
        r = S.ltt_selective(*gen(rng, 3000), ALPHA, DELTA)
        lam = r["lam_hat"]
        assert lam is not None
        false_alarm += S.stress_test(lam, *gen(rng, 4000), ALPHA, DELTA, label="held_out_template")["violation_detected"]
        if seed < 10:  # attack-goal shift: errors rise at the same confidence
            st = S.stress_test(lam, *gen(rng, 4000, shift=0.15), ALPHA, DELTA, label="held_out_goal")
            assert true_risk(lam, shift=0.15) > ALPHA
            assert st["violation_detected"] and st["within_assumption"] is False
    assert false_alarm / N <= DELTA


def test_label_free_score_shift_test():
    rng = np.random.default_rng(9)
    a = rng.random(2000)
    assert not S.score_shift_test(a, rng.random(2000))["shift_detected"]
    assert S.score_shift_test(a, rng.beta(2, 1, 2000))["shift_detected"]


def test_risk_coverage_and_augrc():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    w = np.array([0, 0, 1, 1], bool)
    rc = S.risk_coverage(s, w)
    assert rc["augrc"] == pytest.approx((0 + 0 + 1 / 4 + 2 / 4) / 4)
    assert rc["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


# ------------------------------------------------------------------ wrapper around decide()
def test_selective_decider_wrapper():
    reqs = [data.strip_meta(r) for r in data.build(6, 6, split="dev")]
    dec = KeywordDecider()
    stratum = lambda r: "pay" if "email" in r["state"] else "route"
    cal = {"route": 0.0, "pay": None}
    w = S.SelectiveDecider(dec, cal, stratum)
    for r in reqs:
        out = w(r)
        json.dumps(out)
        if stratum(r) == "pay":
            assert out["abstention"]["abstained"] and out["abstention"]["reason"] == "stratum_not_certifiable"
        else:
            assert not out["abstention"]["abstained"] and out["selective"]["action"] == "execute"
    out = S.SelectiveDecider(dec, {}, stratum)(reqs[0])
    assert out["abstention"]["reason"] == "unknown_stratum"
    out = S.SelectiveDecider(dec, {"route": 1.01, "pay": 1.01}, stratum)(reqs[0])
    assert out["abstention"]["reason"] == "below_threshold"
    # a full receipt as the calibration entry carries alpha / delta / n_s into the response
    rec = S.ltt_selective(*gen(np.random.default_rng(0), 3000), ALPHA, DELTA)
    out = S.SelectiveDecider(dec, {"route": rec, "pay": rec}, stratum)(reqs[0])
    assert out["selective"]["alpha"] == ALPHA and out["selective"]["n_s"] == 3000


# ------------------------------------------------------------------ PR #20 review follow-ups
def test_executed_conf_is_policy_conditioned_for_unmasked_deciders():
    from safety_eval.request import policy_conditioned_prob
    r = {"request_id": "x", "state": {"u": "t"}, "policy": {"allowed": {"q": ["a", "b"]}},
         "questions": {"q": {"type": "choice", "criteria": {"a": "", "b": "", "c": ""}}}}
    served = {"a": 0.3, "b": 0.1, "c": 0.6}  # Clef-style: unmasked served distribution, c disallowed
    resp = {"status": "ok", "answers": {"q": {"type": "choice", "choice": "c", "probabilities": served}}}
    assert S.executed_conf(r, resp) == pytest.approx(0.75)  # 0.3 / (0.3 + 0.1), the executed option a
    assert policy_conditioned_prob(r, resp, "q", "b") == pytest.approx(0.25)
    resp["policy_conditioned_probabilities"] = {"q": {"a": 0.5, "b": 0.5, "c": 0.0}}
    assert S.executed_conf(r, resp) == pytest.approx(0.5)


def test_receipt_names_estimand_and_stress_reports_both_weightings():
    s, w, cl = gen_clustered(np.random.default_rng(0), 50, 10)
    r = S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl)
    assert r["estimand"].startswith("cluster-uniform")
    assert S.ltt_selective(s, w, ALPHA, DELTA)["estimand"].startswith("unit-uniform")
    # one big risky cluster, many small safe ones: the weightings differ
    s2 = np.full(110, 0.9)
    w2 = np.r_[np.ones(100, bool), np.zeros(10, bool)]
    cl2 = ["big"] * 100 + [f"s{i}" for i in range(10)]
    st = S.stress_test(0.5, s2, w2, ALPHA, DELTA, clusters=cl2)
    assert st["selective_risk"] == pytest.approx(100 / 110)
    assert st["selective_risk_cluster_weighted"] == pytest.approx(1 / 11)


# ------------------------------------------------------------------ Track L counterexample: unequal cluster sizes
UNEQ = (9000, 1000, 10, 0.6)  # 9000 safe size-1 clusters + 1000 size-10 clusters erring at 0.6 (1 - s)


def gen_unequal(rng, n_small=UNEQ[0], n_big=UNEQ[1], big=UNEQ[2], c=UNEQ[3]):
    sizes = np.r_[np.ones(n_small, int), np.full(n_big, big)]
    cl = np.repeat(np.arange(sizes.size), sizes)
    s = rng.random(cl.size)
    w = rng.random(cl.size) < np.where(cl >= n_small, c * (1 - s), 0.0)
    return s, w, [str(x) for x in cl]


def unit_uniform_risk(lam, n_small=UNEQ[0], n_big=UNEQ[1], big=UNEQ[2], c=UNEQ[3]):
    """Deployment-traffic selective risk: share of units in risky clusters x E[c (1 - s) | s >= lam]."""
    return n_big * big / (n_small + n_big * big) * c * (1 - lam) / 2


def test_unequal_cluster_sizes_thinned_bound_fails_unit_weighted_bound_holds():
    """Regression for the Track L counterexample on PR #20: one-unit-per-cluster thinning controls the
    cluster-uniform risk, so with risk concentrated in large clusters it under-covers the unit-uniform risk
    (Track L: 56 % at nominal 99 %; this construction: ~0 %). ``weighting="unit"`` (the ``auto`` choice for
    unequal sizes) must cover at >= 1 - delta and still certify in a good share of runs."""
    delta, N = 0.01, 60
    cover = {"cluster": 0, "auto": 0}
    nonvacuous_auto = 0
    for seed in range(N):
        s, w, cl = gen_unequal(np.random.default_rng(seed))
        for wt in cover:
            r = S.ltt_selective(s, w, ALPHA, delta, clusters=cl, weighting=wt, thin_seed=seed)
            lam = r["lam_hat"]
            cover[wt] += lam is None or unit_uniform_risk(lam) <= ALPHA
            if wt == "auto":
                assert r["weighting"] == "unit" and r["estimand"].startswith("unit-uniform")
                nonvacuous_auto += lam is not None
    assert cover["cluster"] / N < 0.5  # the old default: coverage collapses
    assert cover["auto"] / N >= 1 - delta  # fixed
    assert nonvacuous_auto >= N // 4


def test_unit_weighted_bound_validity_at_the_boundary():
    """Single threshold where the unit-uniform risk equals alpha: rejection frequency <= delta."""
    fb = UNEQ[1] * UNEQ[2] / (UNEQ[0] + UNEQ[1] * UNEQ[2])
    lam0 = 1 - 2 * ALPHA / (fb * UNEQ[3])
    assert unit_uniform_risk(lam0) == pytest.approx(ALPHA)
    N = 100
    rej = 0
    for seed in range(N):
        s, w, cl = gen_unequal(np.random.default_rng(500 + seed))
        rej += S.ltt_selective(s, w, ALPHA, DELTA, grid=(lam0,), clusters=cl, weighting="unit")["lam_hat"] is not None
    assert rej / N <= DELTA


def test_auto_weighting_and_cluster_size_bound():
    s, w, cl = gen_clustered(np.random.default_rng(0), 50, 10)  # equal sizes -> thinned (estimands coincide)
    assert S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl)["weighting"] == "cluster"
    s, w, cl = gen_unequal(np.random.default_rng(0), 300, 50, 4)
    r = S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl)
    assert r["weighting"] == "unit" and r["range_source"].startswith("observed_max")
    assert S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl, max_cluster_size=8)["range_source"] == "pre-registered"
    with pytest.raises(ValueError):
        S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl, max_cluster_size=3)
    with pytest.raises(ValueError):
        S.ltt_selective(s, w, ALPHA, DELTA, clusters=cl, weighting="bogus")
    json.dumps(r)


def test_hb_and_empirical_bernstein_bounds():
    from safety_eval.bounds import eb_upper, hb_pvalue
    assert hb_pvalue(0.2, 100, 0.1) == 1.0
    assert hb_pvalue(0.0, 100, 0.1) <= 0.9 ** 100 * math.e + 1e-12
    rng = np.random.default_rng(1)
    miss = 0
    for _ in range(400):  # coverage of the EB upper bound on a skewed [0, 1] law
        v = np.where(rng.random(300) < 0.05, rng.random(300), 0.1)
        miss += eb_upper(v, 0.05) < 0.05 * 0.5 + 0.95 * 0.1
    assert miss / 400 <= 0.05

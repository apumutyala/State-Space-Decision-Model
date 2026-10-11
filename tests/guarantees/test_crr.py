"""B-PREREG-GUAR-CRR: cluster-robust ratio LTT (betting test on per-cluster sums), synthetic, pure Python."""
from __future__ import annotations

import math
import random

import pytest

from jevkit.guarantees import (Scored, binom_cdf, crr_betting_pvalue, fit_conformal, ltt_select, split_calibration)

EPS, DELTA = 0.05, 0.10


def _fixed_fit(qhat: float = 0.5) -> dict:
    return {"alpha": 0.1, "mondrian": False, "qhat": {"all": qhat}, "n": {}, "ids_sha256": "", "_ids": frozenset()}


def _key(r: Scored) -> str:
    return r.id.split("#")[0]


def test_betting_pvalue_is_valid_at_the_null_boundary():
    """Y i.i.d. with E[Y] = 0 exactly (the least favourable H0 point): P(p <= a) <= a, for skewed Y like ours."""
    rng = random.Random("crr-null")
    for a in (0.05, 0.2):
        rej, reps = 0, 400
        for _ in range(reps):
            # Y = -eps w.p. 1 - eps/(1 - eps + eps) ... two-point law on {-eps, 1 - eps} with mean 0: P(1-eps) = eps
            y = [(1 - EPS) if rng.random() < EPS else -EPS for _ in range(300)]
            p, _ = crr_betting_pvalue(y, EPS, a)
            rej += p <= a
        assert rej / reps <= a + 0.03, (a, rej / reps)  # 3 SE slack on 400 reps


def test_betting_pvalue_rejects_when_the_mean_is_clearly_negative_and_checks_range():
    y = [-EPS] * 400  # never wrong
    p, logk = crr_betting_pvalue(y, EPS, 0.001)
    assert p < 1e-6 and logk > 10
    assert crr_betting_pvalue([], EPS, 0.01) == (1.0, 0.0)
    with pytest.raises(ValueError):
        crr_betting_pvalue([1.0], EPS, 0.01)


def test_crr_is_opt_in_and_default_is_byte_identical():
    rng = random.Random(3)
    cal = []
    for i in range(600):
        p = rng.uniform(0.5, 1.0)
        cal.append(Scored(f"row{i // 3}#{i}", "choice", {"a": p, "b": 1 - p}, "a" if rng.random() < p else "b"))
    fit_half, ltt_half, _ = split_calibration(cal, seed=0)
    fit = fit_conformal(fit_half, 0.2, mondrian=False)
    a = ltt_select(ltt_half, fit, 0.1, 0.1)
    assert a == ltt_select(ltt_half, fit, 0.1, 0.1, cluster_test="pick", crr_cap=1, cluster_size=len, crr_seed=9)
    b = ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=_key)
    assert b == ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=_key, cluster_test="pick")
    with pytest.raises(ValueError):
        ltt_select(ltt_half, fit, 0.1, 0.1, cluster_test="crr")  # needs cluster_of
    with pytest.raises(ValueError):
        ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=_key, cluster_test="nope")
    with pytest.raises(ValueError):  # a declared size below the decisions present would break the bound
        ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=_key, cluster_test="crr", cluster_size=lambda c: 1)
    c = ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=_key, cluster_test="crr", crr_cap=8)
    assert c["n_decisions"] == len(ltt_half) and c["n_clusters"] == len({_key(r) for r in ltt_half})
    assert c["cluster_test"].startswith("crr") and c["crr_cap"] == 8


def _sim_clusters(rng: random.Random, n_clusters: int, big: int, k: float) -> list[tuple[float, int, bool]]:
    """(confidence, size, wrong) per cluster: shared confidence and shared fate (every decision of a cluster is
    wrong together: maximal intra-row correlation). size 1 or `big` (half each), independent of fate, so
    R_c(lambda) = E[k (1 - conf) | conf >= lambda] = k (1 - lambda) / 2 for every cap c (conf ~ U(0.5, 1))."""
    out = []
    for _ in range(n_clusters):
        conf = 0.5 + 0.5 * rng.random()
        out.append((conf, 1 if rng.random() < 0.5 else big, rng.random() < k * (1.0 - conf)))
    return out


def _true_risk(lam: float, k: float) -> float:
    return k * (1.0 - lam) / 2.0


def test_crr_matches_ltt_select_on_constructed_clusters():
    """The fast simulation path below computes exactly what ltt_select(cluster_test='crr') computes."""
    rng = random.Random("crr-match")
    cl = _sim_clusters(rng, 300, 16, 0.4)
    cal = []
    for c, (conf, s, bad) in enumerate(cl):
        for j in range(s):
            cal.append(Scored(f"c{c:04d}#{j}", "choice", {"a": conf, "b": 1 - conf}, "b" if bad else "a"))
    lams = (0.95, 0.9, 0.85, 0.8)
    rep = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=lams, cluster_of=_key, cluster_test="crr", crr_cap=8)
    import hashlib
    order = sorted(range(len(cl)), key=lambda c: hashlib.sha256(f"0|crr-order|c{c:04d}".encode()).hexdigest())
    best = None
    for lam in lams:
        y = []
        for c in order:
            conf, s, bad = cl[c]
            n = s if conf >= lam else 0
            y.append(min(1.0, 8 / s) * ((n if bad else 0) - EPS * n) / 8)
        if any(cl[c][0] >= lam for c in order) and crr_betting_pvalue(y, EPS, DELTA / len(lams))[0] <= DELTA / len(lams):
            best = lam if best is None else min(best, lam)
    assert rep["lambda"] == best


def test_crr_certificate_valid_under_strong_intra_row_correlation_and_naive_binomial_is_not():
    """Many seeds of shared-fate clusters. A certificate is violated iff the TRUE R(lambda*) > eps. The CRR betting
    test (cap 8, the prereg's primary) must violate in <= delta of seeds and still certify often; the per-decision
    binomial on the same data must violate in > delta of seeds (it counts 16 copies of one fate as 16 trials)."""
    k, n_clusters, big, seeds = 0.4, 4000, 16, 120
    lams = tuple(round(0.95 - 0.01 * i, 2) for i in range(41))  # 0.95 .. 0.55; R(lambda) <= eps iff lambda >= 0.75
    level = DELTA / len(lams)
    cap = 8
    crr_viol = crr_cert = naive_viol = naive_cert = 0
    for s in range(seeds):
        rng = random.Random(f"crr-sim|{s}")
        cl = _sim_clusters(rng, n_clusters, big, k)  # i.i.d. clusters, already in a random (answer-free) order
        crr_lam = naive_lam = None
        for lam in lams:
            y, n_dec, n_err = [], 0, 0
            for conf, size, bad in cl:
                n = size if conf >= lam else 0
                e = n if bad else 0
                n_dec, n_err = n_dec + n, n_err + e
                y.append(min(1.0, cap / size) * (e - EPS * n) / cap)
            if n_dec == 0:
                continue
            if crr_betting_pvalue(y, EPS, level)[0] <= level:
                crr_lam = lam
            if binom_cdf(n_err, n_dec, EPS) <= level:
                naive_lam = lam
        if crr_lam is not None:
            crr_cert += 1
            crr_viol += _true_risk(crr_lam, k) > EPS + 1e-12
        if naive_lam is not None:
            naive_cert += 1
            naive_viol += _true_risk(naive_lam, k) > EPS + 1e-12
    print(f"seeds {seeds}: crr certified {crr_cert} violated {crr_viol}; naive certified {naive_cert} "
          f"violated {naive_viol}")
    assert crr_viol / seeds <= DELTA, (crr_viol, seeds)
    assert crr_cert >= seeds // 2, crr_cert  # non-vacuous
    assert naive_viol / seeds > DELTA, (naive_viol, seeds)  # the per-decision binomial is anti-conservative here

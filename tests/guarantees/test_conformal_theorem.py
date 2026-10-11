"""The jevkit conformal / LTT code against the theorems in docs/guarantees/CONFORMAL.md (synthetic, pure Python)."""
from __future__ import annotations

import hashlib
import math
import random

from jevkit.guarantees import (Scored, conformal_report, evaluate_sets, fit_conformal, ltt_select, selective_eval,
                               split_calibration)

LABELS = ["a", "b", "c", "d"]


def _draw(rng: random.Random, i: int, cluster: str | None = None, scale: float = 1.2) -> Scored:
    z = [rng.gauss(0, scale) for _ in LABELS]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    p = {k: v / s for k, v in zip(LABELS, e)}
    u, acc, y = rng.random(), 0.0, LABELS[-1]
    for k in LABELS:
        acc += p[k]
        if u <= acc:
            y = k
            break
    return Scored(f"{cluster or 'r' + str(i)}#{i}", "choice", p, y)


def test_split_conformal_coverage_between_the_theorem_bounds():
    alpha, n, seeds = 0.1, 99, 120
    covs = []
    for s in range(seeds):
        rng = random.Random(f"cov|{s}")
        cal = [_draw(rng, i) for i in range(n)]
        test = [_draw(rng, 10_000 + i) for i in range(100)]
        covs.append(evaluate_sets(test, fit_conformal(cal, alpha, mondrian=False))["all"]["coverage"])
    mean = sum(covs) / seeds
    # E[coverage] in [1 - alpha, 1 - alpha + 1/(n+1)] for continuous scores; SE of the mean here is about 0.004
    assert 1 - alpha - 0.012 <= mean <= 1 - alpha + 1 / (n + 1) + 0.012, mean


def test_ltt_certificate_holds_across_seeds():
    eps, delta, seeds, violations, certified = 0.10, 0.10, 40, 0, 0
    rng_pop = random.Random("pop")
    pop = [_draw(rng_pop, 50_000 + i, scale=3.5) for i in range(4000)]  # sharp enough that LTT certifies
    for s in range(seeds):
        rng = random.Random(f"ltt|{s}")
        cal = [_draw(rng, i, scale=3.5) for i in range(800)]
        fit_half, ltt_half, _ = split_calibration(cal, seed=s)
        fit = fit_conformal(fit_half, 0.2, mondrian=False)
        sel = ltt_select(ltt_half, fit, eps, delta)
        if sel["lambda"] is None:
            continue
        certified += 1
        ev = selective_eval(pop, fit, sel["lambda"])
        if ev["acted_error"] is not None and ev["acted_error"] > eps + 0.01:  # population risk proxy (n=4000)
            violations += 1
    assert certified >= seeds // 2, certified  # non-vacuous: abstaining everywhere would pass trivially
    assert violations / seeds <= delta


def _old_rule(ids, seed=0, fit_frac=0.5):
    fit = set()
    for i in ids:
        u = int(hashlib.sha256(f"{seed}|conformal-ltt|{i}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        if u < fit_frac:
            fit.add(i)
    return fit


def test_cluster_split_keeps_rows_together_and_default_is_unchanged():
    rng = random.Random(0)
    cal = [_draw(rng, i, cluster=f"row{i // 3}") for i in range(300)]  # 3 decisions per row
    fit, ltt, rec = split_calibration(cal, seed=0)
    assert {r.id for r in fit} == _old_rule([r.id for r in cal])  # default behaviour byte-identical
    straddle = {r.id.split("#")[0] for r in fit} & {r.id.split("#")[0] for r in ltt}
    assert straddle  # the per-decision split DOES put one row's decisions on both sides (discrepancy D1)
    fit_c, ltt_c, rec_c = split_calibration(cal, seed=0, cluster_of=lambda r: r.id.split("#")[0])
    assert not ({r.id.split("#")[0] for r in fit_c} & {r.id.split("#")[0] for r in ltt_c})
    assert rec_c["n_fit_clusters"] + rec_c["n_ltt_clusters"] == 100 and "cluster" in rec_c["rule"]
    assert "cluster" not in rec["rule"]


# ------------------------------------------------------------------ D1 part 2: cluster-level LTT (one decision per cluster)
def _fixed_fit(qhat: float = 0.5) -> dict:
    return {"alpha": 0.1, "mondrian": False, "qhat": {"all": qhat}, "n": {}, "ids_sha256": "", "_ids": frozenset()}


def _copies(n_clusters: int, m: int, bad: set[int]) -> list[Scored]:
    """m identical decisions per cluster (perfect within-cluster correlation); cluster c is wrong iff c in bad."""
    out = []
    for c in range(n_clusters):
        for j in range(m):
            out.append(Scored(f"row{c}#{j}", "choice", {"a": 0.9, "b": 0.1}, "b" if c in bad else "a"))
    return out


def test_cluster_ltt_default_is_unchanged_and_one_per_cluster_data_is_identical():
    from jevkit.guarantees import cluster_pick
    rng = random.Random(3)
    cal = [_draw(rng, i, cluster=f"row{i // 3}") for i in range(600)]
    fit_half, ltt_half, _ = split_calibration(cal, seed=0)
    fit = fit_conformal(fit_half, 0.2, mondrian=False)
    a = ltt_select(ltt_half, fit, 0.1, 0.1)
    assert a == ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=None)  # default path byte-identical
    assert "cluster_test" not in a
    # with singleton clusters the cluster test is the per-decision test
    b = ltt_select(ltt_half, fit, 0.1, 0.1, cluster_of=lambda r: r.id)
    assert {k: v for k, v in b.items() if k in a} == a
    # the pick is a function of ids only: changing every probability and label does not move it
    key = lambda r: r.id.split("#")[0]  # noqa: E731
    p1 = [r.id for r in cluster_pick(cal, key)]
    flipped = [Scored(r.id, r.type, {k: 1.0 - v for k, v in r.probs.items()}, "a") for r in cal]
    assert p1 == [r.id for r in cluster_pick(flipped, key)]
    assert len(p1) == 200 and len({i.split("#")[0] for i in p1}) == 200
    assert [r.id for r in cluster_pick(cal, key, seed=1)] != p1  # the seed matters


def test_per_decision_ltt_is_anticonservative_on_clusters_and_cluster_ltt_is_valid():
    """Exact (not simulated) rejection probabilities, cross-checked against ltt_select on constructed data.

    300 clusters of 8 identical decisions, true acted risk R = 0.055 > eps = 0.05 (H0 true). Any rejection is a
    false certificate. The per-decision binomial treats 2,400 trials as independent and rejects with probability
    > delta; the one-decision-per-cluster binomial rejects with probability <= delta."""
    from jevkit.guarantees import _log_binom_pmf, binom_cdf
    n_cl, m, risk, eps, delta = 300, 8, 0.055, 0.05, 0.10
    fit = _fixed_fit()
    key = lambda r: r.id.split("#")[0]  # noqa: E731
    p_dec = p_clu = 0.0
    for e in range(n_cl + 1):
        w = math.exp(_log_binom_pmf(e, n_cl, risk))
        p_dec += w * (binom_cdf(e * m, n_cl * m, eps) <= delta)
        p_clu += w * (binom_cdf(e, n_cl, eps) <= delta)
    assert p_dec > delta + 0.05, p_dec  # the old test's false-certificate rate exceeds its nominal delta
    assert p_clu <= delta, p_clu
    for e in (5, 9, 15):  # the formulas above are what ltt_select computes on such data
        cal = _copies(n_cl, m, set(range(e)))
        d = ltt_select(cal, fit, eps, delta, lambdas=(0.5,))
        c = ltt_select(cal, fit, eps, delta, lambdas=(0.5,), cluster_of=key)
        assert (d["lambda"] is not None) == (binom_cdf(e * m, n_cl * m, eps) <= delta)
        assert (c["lambda"] is not None) == (binom_cdf(e, n_cl, eps) <= delta)
        assert c["n_clusters"] == n_cl and c["n_decisions"] == n_cl * m


def test_conformal_report_cluster_ltt():
    rng = random.Random(5)
    cal = [_draw(rng, i, cluster=f"row{i // 4}") for i in range(800)]
    test = [_draw(rng, 10_000 + i) for i in range(200)]
    key = lambda r: r.id.split("#")[0]  # noqa: E731
    rep = conformal_report(cal, test, alpha=0.2, epsilon=0.1, delta=0.1, mondrian=False, cluster_of=key,
                           cluster_ltt=True)
    assert rep["ltt"]["n_clusters"] == rep["calibration_split"]["n_ltt_clusters"]
    import pytest
    with pytest.raises(ValueError):
        conformal_report(cal, test, cluster_ltt=True)

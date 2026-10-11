"""B-PREREG-GUAR-FSCRR: fixed-sequence LTT with a pre-registered start lambda (CONFORMAL.md D2), synthetic."""
from __future__ import annotations

import hashlib
import random

import pytest

from jevkit.guarantees import (Scored, binom_cdf, crr_betting_pvalue, fit_conformal, ltt_select, split_calibration)

EPS, DELTA = 0.05, 0.10


def _fixed_fit(qhat: float = 0.5) -> dict:
    return {"alpha": 0.1, "mondrian": False, "qhat": {"all": qhat}, "n": {}, "ids_sha256": "", "_ids": frozenset()}


def _key(r: Scored) -> str:
    return r.id.split("#")[0]


def _toy(seed: int = 3, n: int = 600) -> list[Scored]:
    rng = random.Random(seed)
    cal = []
    for i in range(n):
        p = rng.uniform(0.5, 1.0)
        cal.append(Scored(f"row{i // 3}#{i}", "choice", {"a": p, "b": 1 - p}, "a" if rng.random() < p else "b"))
    return cal


def test_fs_start_is_opt_in_and_default_is_byte_identical():
    fit_half, ltt_half, _ = split_calibration(_toy(), seed=0)
    fit = fit_conformal(fit_half, 0.2, mondrian=False)
    for kw in ({}, {"cluster_of": _key}, {"cluster_of": _key, "cluster_test": "crr", "crr_cap": 8}):
        for m in ("bonferroni", "fixed_sequence"):
            assert ltt_select(ltt_half, fit, 0.1, 0.1, method=m, **kw) == \
                ltt_select(ltt_half, fit, 0.1, 0.1, method=m, fs_start=None, **kw)
    with pytest.raises(ValueError):  # a start lambda only means something for the fixed sequence
        ltt_select(ltt_half, fit, 0.1, 0.1, fs_start=0.95)
    with pytest.raises(ValueError):  # must be a pre-set grid point
        ltt_select(ltt_half, fit, 0.1, 0.1, method="fixed_sequence", fs_start=0.955)


def _const_cal(conf: float, n_rows: int, size: int, wrong_every: int = 0) -> list[Scored]:
    cal = []
    for c in range(n_rows):
        bad = wrong_every and c % wrong_every == 0
        for j in range(size):
            cal.append(Scored(f"c{c:05d}#{j}", "choice", {"a": conf, "b": 1 - conf}, "b" if bad else "a"))
    return cal


def test_a_lambda_that_acts_on_nothing_stops_the_sequence_instead_of_being_skipped():
    cal = _const_cal(0.9, 400, 2)  # every decision correct at p 0.9: nothing acts at lambda > 0.9
    lams = (0.99, 0.95, 0.9, 0.85)
    for kw in ({}, {"cluster_of": _key, "cluster_test": "crr", "crr_cap": 8}):
        legacy = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=lams, method="fixed_sequence", **kw)
        assert legacy["lambda"] == 0.85  # old behaviour: 0.99 and 0.95 skipped (the D2 issue)
        top = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=lams, method="fixed_sequence", fs_start=0.99, **kw)
        assert top["lambda"] is None and top["fs_stop"] == {"lambda": 0.99, "reason": "acts on nothing", "n_act": 0}
        assert top["fs_tested"] == [0.99]
        mid = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=lams, method="fixed_sequence", fs_start=0.9, **kw)
        assert mid["lambda"] == 0.85 and mid["fs_tested"] == [0.9, 0.85]
        assert mid["fs_stop"] == {"lambda": None, "reason": "grid exhausted"}


def test_fs_stops_at_the_first_non_rejection():
    # rows wrong 1 in 5 (20% > eps): no lambda can be certified, the sequence stops at its first point
    cal = _const_cal(0.97, 300, 1, wrong_every=5)
    rep = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=(0.97, 0.96, 0.5), method="fixed_sequence",
                     fs_start=0.96, cluster_of=_key, cluster_test="crr", crr_cap=8)
    assert rep["lambda"] is None and rep["fs_stop"]["lambda"] == 0.96 and rep["fs_stop"]["reason"] == "not rejected"
    assert rep["fs_tested"] == [0.96] and rep["test_level"] == DELTA


def _sim_clusters(rng: random.Random, n_clusters: int, big: int, k: float) -> list[tuple[float, int, bool]]:
    """(confidence, size, wrong) per cluster; shared confidence and shared fate (every decision of a cluster is
    wrong together: maximal intra-row correlation); size 1 or `big`, independent of fate, so
    R_c(lambda) = k (1 - lambda) / 2 for every cap c (conf ~ U(0.5, 1))."""
    out = []
    for _ in range(n_clusters):
        conf = 0.5 + 0.5 * rng.random()
        out.append((conf, 1 if rng.random() < 0.5 else big, rng.random() < k * (1.0 - conf)))
    return out


def _fs_run(cl, lams, cap, level, crr: bool):
    """Fast fixed-sequence path (start = lams[0]); returns the certified lambda or None."""
    best = None
    for lam in lams:
        y, n_dec, n_err = [], 0, 0
        for conf, size, bad in cl:
            n = size if conf >= lam else 0
            e = n if bad else 0
            n_dec, n_err = n_dec + n, n_err + e
            y.append(min(1.0, cap / size) * (e - EPS * n) / cap)
        if n_dec == 0:
            break
        p = crr_betting_pvalue(y, EPS, level)[0] if crr else binom_cdf(n_err, n_dec, EPS)
        if p > level:
            break
        best = lam
    return best


def test_fast_path_matches_ltt_select_fs():
    rng = random.Random("fs-match")
    cl = _sim_clusters(rng, 300, 16, 0.4)
    cal = []
    for c, (conf, s, bad) in enumerate(cl):
        for j in range(s):
            cal.append(Scored(f"c{c:04d}#{j}", "choice", {"a": conf, "b": 1 - conf}, "b" if bad else "a"))
    lams = (0.99, 0.95, 0.9, 0.85, 0.8)
    rep = ltt_select(cal, _fixed_fit(), EPS, DELTA, lambdas=lams, method="fixed_sequence", fs_start=0.95,
                     cluster_of=_key, cluster_test="crr", crr_cap=8)
    order = sorted(range(len(cl)), key=lambda c: hashlib.sha256(f"0|crr-order|c{c:04d}".encode()).hexdigest())
    assert rep["lambda"] == _fs_run([cl[c] for c in order], lams[1:], 8, DELTA, True)


def test_fs_crr_valid_under_strong_intra_row_correlation_and_naive_fs_binomial_is_not():
    """Shared-fate clusters (sizes 1 / 16), many seeds. A certificate is violated iff the TRUE R(lambda*) > eps.
    Fixed sequence from a pre-set start 0.95 at the full level delta: CRR cap 8 violates in <= delta of seeds and
    certifies more often than the same test under Bonferroni over the grid; the per-decision binomial run as the
    same fixed sequence violates in > delta of seeds."""
    k, n_clusters, big, seeds, cap = 0.4, 4000, 16, 120, 8
    lams = tuple(round(0.95 - 0.01 * i, 2) for i in range(41))  # 0.95 .. 0.55; R(lambda) <= eps iff lambda >= 0.75
    fs_viol = fs_cert = bf_cert = nv_viol = 0
    for s in range(seeds):
        cl = _sim_clusters(random.Random(f"fs-sim|{s}"), n_clusters, big, k)
        lam = _fs_run(cl, lams, cap, DELTA, True)
        if lam is not None:
            fs_cert += 1
            fs_viol += k * (1 - lam) / 2 > EPS + 1e-12
        # Bonferroni reference on the same data: certifies iff some lambda has p <= delta / |grid|
        lvl = DELTA / len(lams)
        for lam_b in lams:
            y = [min(1.0, cap / sz) * ((sz if bad else 0) - EPS * sz) / cap if conf >= lam_b else 0.0
                 for conf, sz, bad in cl]
            if crr_betting_pvalue(y, EPS, lvl)[0] <= lvl:
                bf_cert += 1
                break
        lam_n = _fs_run(cl, lams, cap, DELTA, False)
        if lam_n is not None:
            nv_viol += k * (1 - lam_n) / 2 > EPS + 1e-12
    print(f"seeds {seeds}: fs-crr certified {fs_cert} violated {fs_viol}; bonferroni-crr certified {bf_cert}; "
          f"fs-naive violated {nv_viol}")
    assert fs_viol / seeds <= DELTA, (fs_viol, seeds)
    assert fs_cert >= bf_cert and fs_cert >= seeds * 0.8, (fs_cert, bf_cert)
    assert nv_viol / seeds > DELTA, (nv_viol, seeds)


def test_fs_crr_rarely_certifies_when_every_lambda_is_null():
    """Global null: R(lambda) = k (1 - lambda) / 2 > eps at every grid point (k = 2.03: R(0.95) = 0.0508, the
    least favourable start just above eps). FWER = P(certify anything) must be <= delta under shared-fate clusters."""
    k, n_clusters, big, seeds, cap = 2.03, 3000, 16, 200, 8
    lams = tuple(round(0.95 - 0.01 * i, 2) for i in range(11))
    cert = 0
    for s in range(seeds):
        cl = _sim_clusters(random.Random(f"fs-null|{s}"), n_clusters, big, k)
        cert += _fs_run(cl, lams, cap, DELTA, True) is not None
    print(f"global null: certified {cert}/{seeds}")
    assert cert / seeds <= DELTA, cert

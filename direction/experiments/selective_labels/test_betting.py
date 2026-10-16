"""Deterministic CPU checks for experiment-local statistical contracts."""
import itertools
import math

import numpy as np
import pytest

from direction.experiments.selective_labels.betting import betting_paths, closure_times, normalized_ipw
from jevkit.guarantees import CRR_BET_CAP, crr_betting_pvalue


@pytest.mark.parametrize("pi", [0.02, 0.05, 0.1, 0.2, 1.0])
def test_vectorized_paths_match_existing_crr(pi):
    rng = np.random.default_rng(724)
    a = rng.random((4, 120, 3)) < 0.64
    b = a & (rng.random(a.shape) < pi)
    loss = rng.random(a.shape) < 0.05
    y = normalized_ipw(a, b, np.where(b, loss, np.nan), pi, epsilon=0.05, floor=pi)
    got = betting_paths(y, epsilon=pi * 0.05, level=0.05, horizon=120)
    expected = [crr_betting_pvalue(row.tolist(), pi * 0.05, 0.05)[0] for row in y]
    np.testing.assert_allclose(got["p"], expected, rtol=2e-14)
    np.testing.assert_array_equal(got["p"], got["p_due"])
    if pi == 1:
        np.testing.assert_allclose(y, ((a & loss).sum(axis=2) - 0.05 * a.sum(axis=2)) / 3)


def test_exhaustive_ht_identity_bounds_and_null_factor():
    epsilon, floor = 0.05, 0.02
    probabilities = np.array([0.02, 0.1, 0.2])
    expected_factor = expected_x = 0.0
    for fate in (0, 1):
        expectation_given_fate = 0.0
        for mask in itertools.product((False, True), repeat=3):
            mask = np.array(mask)
            probability = np.prod(np.where(mask, probabilities, 1 - probabilities))
            x = float(normalized_ipw(np.ones(3, bool), mask, np.where(mask, fate, np.nan),
                                     probabilities, epsilon=epsilon, floor=floor))
            assert -floor * epsilon - 1e-14 <= x <= 1 - floor * epsilon + 1e-14
            factor = 1 - CRR_BET_CAP / (1 - floor * epsilon) * x
            assert factor >= 1 - CRR_BET_CAP
            expectation_given_fate += probability * x
            fate_probability = epsilon if fate else 1 - epsilon
            expected_factor += fate_probability * probability * factor
            expected_x += fate_probability * probability * x
        assert expectation_given_fate == pytest.approx(floor * (fate - epsilon), abs=1e-15)
    assert expected_x == pytest.approx(0, abs=1e-15)
    assert expected_factor == pytest.approx(1, abs=1e-14)


def test_raw_ipw_does_not_fit_original_bounds():
    x = normalized_ipw([1, 1, 1], [1, 1, 1], [1, 1, 1], 0.02, epsilon=0.05, floor=0.02)
    assert float(x / 0.02) == pytest.approx(49.95)
    with pytest.raises(ValueError, match="outside"):
        crr_betting_pvalue([float(x / 0.02)], 0.05, 0.05)
    assert crr_betting_pvalue([float(x)], 0.001, 0.05)[0] == 1


@pytest.mark.parametrize("probability", [0, -0.1, 0.01, np.nan, np.inf, 1.1])
def test_refuse_invalid_or_zero_propensity_even_if_unaudited(probability):
    with pytest.raises(ValueError, match="refuse"):
        normalized_ipw([1, 1], [0, 0], [np.nan, np.nan], [0.1, probability], epsilon=0.05, floor=0.02)


def test_refuse_missing_audited_label_but_not_unqueried_label():
    with pytest.raises(ValueError, match="complete"):
        normalized_ipw([1], [1], [np.nan], 0.1, epsilon=0.05, floor=0.1)
    assert normalized_ipw([1, 0], [0, 0], [np.nan, np.nan], [0.1, 0],
                          epsilon=0.05, floor=0.1) == pytest.approx(-0.0025)
    with pytest.raises(ValueError, match="positive"):
        normalized_ipw([1], [0], [np.nan], 0, epsilon=0.05, floor=0)


def test_delay_closes_whole_clusters_in_fixed_order():
    selected = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]], bool)
    delays = np.array([[[0, 10, 999], [0, 999, 999], [999, 999, 999]]])
    np.testing.assert_array_equal(closure_times(selected, delays), [[11, 11, 11]])
    delays = delays.astype(float)
    delays[0, 0, 1] = np.inf
    assert np.isinf(closure_times(selected, delays)).all()


def test_delay_cannot_issue_an_unclosed_certificate():
    n = 400
    y = np.full((1, n), -0.05)
    closure = np.full((1, n), n + 100)
    result = betting_paths(y, epsilon=0.05, level=0.05, horizon=n, closure=closure)
    assert result["p"][0] < 0.05
    assert result["p_due"][0] == 1
    assert result["closed"][0] == 0
    assert result["wall"][0] == n + 100
    flushed = betting_paths(y, epsilon=0.05, level=0.05, horizon=n, closure=closure, deadline=n + 100)
    np.testing.assert_array_equal(flushed["p"], flushed["p_due"])
    with pytest.raises(ValueError, match="FIFO"):
        betting_paths(y[:, :2], epsilon=0.05, level=0.05, horizon=n, closure=[[10, 2]])


def test_past_capital_does_not_depend_on_unclosed_future():
    rng = np.random.default_rng(89)
    y = rng.binomial(1, 0.01, (3, 400)) - 0.05
    prefix = betting_paths(y[:, :100], epsilon=0.05, level=0.05, horizon=400)
    closed = betting_paths(y, epsilon=0.05, level=0.05, horizon=400, deadline=100)
    np.testing.assert_array_equal(prefix["p"], closed["p_due"])
    empty = betting_paths(np.empty((1, 0)), epsilon=0.05, level=0.05, horizon=400)
    assert empty["p"][0] == 1


def test_all_zero_losses_growth_has_a_hard_audit_floor():
    # Even acting on every decision cannot reach capital 20 at pi=.02,n=2000.
    pi, epsilon, n = 0.02, 0.05, 2000
    bound = n * math.log1p(CRR_BET_CAP * pi * epsilon / (1 - pi * epsilon))
    assert bound < math.log(20)

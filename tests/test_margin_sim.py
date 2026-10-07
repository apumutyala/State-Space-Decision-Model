"""LESSONS L-20 enforcing check: margins inside measured noise reject everything; size gates by simulation before
adopting them (issue #36). Runs the unchanged-candidate (Delta = 0) pass-rate simulation of EVERY G12/G13 criterion at
the configured margins (``train.promotion.MARGINS``) at small scale and requires >= 0.5 each; the design target is
>= 0.8 (``null_pass_floor``), re-checked on the first real runs by ``train.promotion.resizing``."""

import math

import numpy as np
import pytest

from train import margin_sim as M
from train.promotion import MARGINS

CRITERIA = {"g12_steer_d1a", "g13_steer_d1b", "g12_false_alarm", "g13_false_alarm", "g13_capability", "g13_ece",
            "g13_retention_acc", "g13_retention_ece"}


@pytest.fixture(scope="module")
def table():
    return M.table(scale=0.25)


def test_every_criterion_passes_an_unchanged_candidate_at_the_configured_margins(table):
    assert set(table) == CRITERIA
    low = {k: v for k, v in table.items() if v < 0.5}
    assert not low, f"margins inside measured noise reject unchanged candidates: {low} (LESSONS L-20)"


def test_the_originally_proposed_values_fail_the_same_simulation():
    """As first proposed (#36): m_ret_acc 0.01, capability judged on the 2,008-row subset (SE 0.752) and ECE on the
    subset (~5.5k fields). Each passes an unchanged candidate far less than half the time; the product was ~0.001."""
    assert M.capability(1.0, se=0.752) < 0.5
    assert M.retention_acc(0.01, n_sims=20) < 0.5
    assert M.ece(0.01, n_fields=5500, n_clusters=1700, n_sims=6) < 0.5


def test_simulated_criteria_still_catch_a_regression_at_the_margin():
    """Power side: a true regression AT the margin passes rarely (<= ~alpha/2 by construction)."""
    rng = np.random.default_rng(0)
    se = M.FULL_DEV_INDEX_SE
    d = rng.normal(-MARGINS["m_cap"], se, 4000)
    assert np.mean(d - 1.96 * se >= -MARGINS["m_cap"]) < 0.05


def test_local_debiased_ece_equals_the_harness_one():
    try:
        from scripts.m2_e2e_dryrun import ensure_kit

        ensure_kit()
        from harness.calibration import debiased_ece
    except ImportError:
        pytest.skip("Decision Index kit not importable")
    rng = np.random.default_rng(1)
    conf, y = rng.uniform(0.3, 1, 500), (rng.random(500) < 0.7).astype(float)
    ref = debiased_ece(conf, y)["debiased_l2"]
    import builtins

    real = builtins.__import__

    def no_harness(name, *a, **k):
        if name == "harness.calibration":
            raise ImportError(name)
        return real(name, *a, **k)

    builtins.__import__ = no_harness
    try:
        local = M.debiased_ece_l2(conf, y)
    finally:
        builtins.__import__ = real
    assert math.isclose(local, ref, rel_tol=1e-12, abs_tol=1e-15)

"""LESSONS L-20 enforcing check: size the evaluation by simulation; never loosen safety margins for power (issue #36,
Codex 6028537932). For every G12/G13 criterion, simulate the frozen-set size an unchanged candidate needs to pass with
probability >= 0.8 at the APPROVED margins, and require that the configured evaluation sizes meet it, or that the
shortfall is recorded with a plan to expand the set (``train.margin_sim.SHORTFALLS``). The margins themselves are
pinned to the contract here: low power never moves them."""

import math

import numpy as np
import pytest

from train import margin_sim as M
from train.promotion import MARGINS

CONTRACT = {"m_cap": 1.0, "m_steer": 0.02, "m_ece": 0.01, "m_ret_acc": 0.01, "m_false_alarm": 0.01,
            "m_refusal": 0.01, "fa_abs_cap": 0.01}


@pytest.fixture(scope="module")
def table():
    return M.sizing_table(scale=0.5)


def test_margins_are_the_codex_contract_and_never_loosened_for_power():
    assert {k: MARGINS[k] for k in CONTRACT} == CONTRACT


def test_every_configured_evaluation_meets_the_simulated_size_or_its_shortfall_is_recorded(table):
    assert set(table) == set(M.EVAL_SIZES)
    short = {c for c, r in table.items() if not r["meets"]}
    assert short == set(M.SHORTFALLS), (
        f"simulated shortfalls {sorted(short)} != recorded {sorted(M.SHORTFALLS)}: expand the evaluation or record the "
        "shortfall with a plan (LESSONS L-20); never relax a margin")
    for c in short:
        r = table[c]
        assert r["need"] > r["have"] and r["pass_at_need"] >= M.FLOOR  # confirmed by simulation at the required size
        assert M.SHORTFALLS[c]["plan"] and "stays" in M.SHORTFALLS[c]["plan"]
    for c, r in table.items():
        if r["meets"]:
            assert r["pass_at_have"] >= M.FLOOR


def test_retention_size_is_a_parameter_and_the_required_n_is_reported(table):
    r = table["g13_retention_acc"]
    assert r["have"] == 1420 and r["need"] > 5000  # ~9k items at discordance 0.10 (assumption)
    bigger = M.required_n("g13_retention_acc", scale=0.5, sizes={"g13_retention_acc": r["need"]})
    assert bigger["meets"] and bigger["have"] == r["need"]


def test_simulated_criteria_still_catch_a_regression_at_the_margin():
    """Power side: a true capability regression AT the margin passes rarely (<= ~alpha/2 by construction)."""
    rng = np.random.default_rng(0)
    se = M.SUBSET_INDEX_SE * math.sqrt(M.SUBSET_ROWS / M.EVAL_SIZES["g13_capability"]["n"])
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

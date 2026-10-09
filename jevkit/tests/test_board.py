"""The scorer's formulas reproduce published JevBench v1.6.1 rows exactly (excerpt of the published JSON)."""

import json
import math
from pathlib import Path

import pytest

from jevkit import board, spec
from jevkit import scoring as SC

FIX = Path(__file__).parent / "fixtures" / "board_v161_excerpt.json"


@pytest.fixture(scope="module")
def api():
    return json.loads(FIX.read_text(encoding="utf-8"))


def row(api, key):
    return next(s for s in api["systems"] if s["key"] == key)


def test_constants_match_published_header(api):
    assert api["tier_weights"] == spec.TIER_WEIGHTS
    assert api["G_med"] == spec.G_MED_V161
    assert api["bootstrap"]["B"] == spec.BOOTSTRAP_B and api["bootstrap"]["bootstrap_seed"] == spec.BOOTSTRAP_SEED
    for o, v in api["options"].items():
        w = v["weights"]
        assert (w["intelligence"], w["calibration"], w["speed"], w["cost"]) == spec.OPTIONS[o]["weights"]
        assert v["intelligence_floor"] == spec.OPTIONS[o]["intelligence_floor"]
    assert all(abs(v - 1 / 3) < 1e-15 for v in api["types"].values())


def test_every_scored_check_exact_on_excerpt(api):
    rep = board.verify(api)
    for name, d in rep["checks"].items():
        if name.startswith("view["):
            continue
        assert not d["mismatches"], (name, d["mismatches"][:3])
        assert d["exact"] == d["rows"] and d["rows"] > 0, name
    # the composite is checked on every row with four axes; the null-axis row is null on both sides
    assert rep["checks"]["composite_A"]["both_null"] == 1


def test_h2o_headline_composite(api):
    r = row(api, "h2o-lightning-4b")
    a = r["axes"]
    got = SC.composite(a["intelligence"], a["calibration"], a["speed"], a["cost"], "A")
    assert got == pytest.approx(72.5232271521435, abs=1e-12)
    assert SC.capability(a["intelligence"], a["calibration"]) == pytest.approx(r["capability"], abs=1e-12)


def test_penalty_is_multiplicative_not_subtractive(api):
    for key in ("hopper", "manchego-v21-v15"):
        it = row(api, key)["intelligence"]
        assert it["excess"] > spec.GAP_ALLOWANCE
        mult = SC.intelligence(it["I_open"], it["I_sealed"])["I"]
        sub = it["base"] - (it["excess"] - spec.GAP_ALLOWANCE)
        pub = row(api, key)["axes"]["intelligence"]
        assert mult == pytest.approx(pub, abs=1e-12)
        assert abs(sub - pub) > 0.5  # the "1 point per point" reading is off by > 0.5 I points here


def test_split_type_cc_clipped_at_zero(api):
    pts = row(api, "jade")["intelligence"]["per_type_split"]
    raw = SC.tiered_cc(pts["open|score"]["tiers"])
    assert raw < -100 and pts["open|score"]["cc"] == 0.0


def test_missing_supported_calibration_part_counts_zero(api):
    r = row(api, "jade")  # every score answer invalid -> no score part, still supported
    parts = r["calibration"]["parts"]
    assert "score" not in parts
    got = SC.calibration({t: v["score"] for t, v in parts.items()}, spec.TYPES)
    assert got == pytest.approx(r["axes"]["calibration"], abs=1e-12)


def test_score_calibration_halves_clipped(api):
    p = row(api, "raw-qwen3-0.6b")["calibration"]["parts"]["score"]
    assert 1 - 2 * p["top_ece"] < 0
    assert SC.calibration_score(p["top_ece"], p["nrps"]) == pytest.approx(p["score"], abs=1e-12)


def test_cost_clipped_at_100(api):
    r = row(api, "verdict-small")
    assert 100 - 30 * math.log10(r["cost"]["usd_per_1000"] / 0.001) > 100
    assert SC.cost_score(r["cost"]["usd_per_1000"]) == 100.0 == r["axes"]["cost"]


def test_zero_and_missing_axes():
    assert SC.composite(60, 90, 0.0, 60) == 0.0
    assert SC.composite(60, 90, None, 60) is None
    # gates: I below its floor, Speed / Cost below 50
    full = 4 / (1 / 40 + 1 / 90 + 1 / 90 + 1 / 90)
    assert SC.composite(40, 90, 90, 90, "A") == pytest.approx(full * (40 / 50) ** 2)
    assert SC.composite(55, 90, 90, 90, "C") == pytest.approx(4 / (1 / 55 + 3 / 90) * (55 / 60) ** 2)


def test_views_partially_reproduced(api):
    """Sort views are not scored; the published Intelligence-only view is on a stale basis for most rows."""
    rep = board.verify(api)
    bal = rep["checks"]["view[Balanced 33:33:33]"]
    assert {m["key"] for m in bal["mismatches"]} == {"deck31b"}

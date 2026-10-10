"""JevBench v1.6.1 scoring constants (reverse-derived from the published board JSON; see jevkit/README.md).

Every constant here is either quoted from the published artifact (``/api/jevbench/v1.6.1``: ``tier_weights``,
``types``, ``G_med``, ``options``, ``bootstrap``) or from the public method files (METHOD-v1.5.md + addenda), and the
reproduction test (``jevkit/tests/test_board.py``) checks them against published rows.
"""

from __future__ import annotations

TYPES = ("choice", "noul", "score")
TIERS = ("easy", "standard", "judge", "hard")
#: api_v161.json ``tier_weights``; renormalised over the tiers present in a split x type cell
TIER_WEIGHTS = {"easy": 0.1, "standard": 0.2, "judge": 0.3, "hard": 0.4}
#: api_v161.json ``types``: Choice / Noul / Score one third each over supported types (headline-A amendment)
TYPE_WEIGHTS = {"choice": 1 / 3, "noul": 1 / 3, "score": 1 / 3}
SEALED_SHARE = 0.5

#: Noul decisive thresholds (O1S): P(yes) <= 0.2 is No, >= 0.8 is Yes, anything in between abstains (counted wrong)
NOUL_NO_MAX = 0.2
NOUL_YES_MIN = 0.8

#: Field median open-minus-sealed gap, frozen for v1.6 (api_v161.json ``G_med``)
G_MED_V161 = 2.573387642438244
GAP_ALLOWANCE = 8.0
#: penalty = max(0, 1 - max(0, excess - 8) / PENALTY_SLOPE_DEN); Intelligence = clip(base * penalty, 0, 100).
#: Verified on the four published rows with excess > 8 (hopper, manchego-v21-v15, raw-qwen3-4b-instruct-2507,
#: open-jev-zefan-2b): the multiplicative form reproduces them exactly, the "1 point per point" subtraction does not.
PENALTY_SLOPE_DEN = 100.0

#: Distribution validity (public jevbench/scoring.py): strict 1e-3, renormalised inside 2e-2, invalid outside
STRICT_SUM_TOL = 1e-3
RENORM_SUM_TOL = 2e-2

ECE_BINS = 10
#: Choice calibration ECE is computed on hard-tier choice items only: for every published row
#: ``calibration.parts.choice.n_ece_hard`` equals the hard-tier choice count (320, or 307 after the 13 C-excluded
#: items). Override with ``choice_ece_tiers`` if a later revision changes this.
CHOICE_ECE_TIERS = ("hard",)

#: Speed: S = 80 - 10 log10(p50) - 10 log10(p95), p in seconds (adjusted for self-hosted rows),
#: = mean(f(p50), f(p95)) with f(s) = 100 - 20 log10(s / 0.1 s); self-hosted s' = 2 s + 0.15 s
SPEED_REF_S = 0.1
SPEED_SLOPE = 20.0
SELF_HOSTED_FACTOR = 2.0
SELF_HOSTED_OFFSET_S = 0.15
#: Speed is timed on the open standard + judge items (~124 items), serially, one request at a time
TIMED_TIERS = ("standard", "judge")

#: Cost: K = 100 - 30 log10(USD per 1,000 decisions / 0.001), clipped to [0, 100] (verdict-small: 101.87 -> 100.0)
COST_REF_USD_PER_1000 = 0.001
COST_SLOPE = 30.0

#: Composite presets (api_v161.json ``options``): weights in axis order (I, C, S, K) and the Intelligence floor
OPTIONS = {
    "A": {"weights": (25.0, 25.0, 25.0, 25.0), "intelligence_floor": 50.0},
    "B": {"weights": (40.0, 20.0, 20.0, 20.0), "intelligence_floor": 50.0},
    "C": {"weights": (25.0, 25.0, 25.0, 25.0), "intelligence_floor": 60.0},
}
HEADLINE = "A"
AXIS_GATE = 50.0  # Speed and Cost gates: x min(1, axis / 50)^2

#: Sort views (weights over Intelligence, Speed, Cost; same gates). Not part of the scored composite.
VIEWS = {
    "Balanced 33:33:33": (1.0, 1.0, 1.0),
    "Emphasis on Accuracy 60:20:20": (60.0, 20.0, 20.0),
    "Emphasis on Cost 20:20:60": (20.0, 20.0, 60.0),
    "Emphasis on Speed 20:60:20": (20.0, 60.0, 20.0),
}

#: Capability ranking caps: 2 x the Jev 1.13.0 v1.5 reference (USD 0.0323 / 1k, p50 0.6165 s)
CAP_USD_PER_1000 = 0.0645947
CAP_P50_ADJUSTED_S = 1.2329566

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 16

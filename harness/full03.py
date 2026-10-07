"""Decision Index 0.3 Full score, as the live board computes it from its three parts.

Source: Space `multimodalart/jev-decision-index` `data/v03.json` (sha256 pinned in config.BOARD_V03_SHA256):
`weights` = {public 0.2, same_skills 0.5, new_domains 0.3}; `formula.equating` = mean/sd of each part over
21 untuned stock "bases". Each private part is mapped linearly onto the public scale before weighting
("a model that sits a given distance above the stock models' average on a private part gets the score that
sits the same distance above their average on the public index", kit PR #74 README):

    eq(x, part) = mu_pub + sd_pub * (x - mu_part) / sd_part
    Full        = 0.2 * public + 0.5 * eq(same_skills) + 0.3 * eq(new_domains)

Reverse-engineered from the published numbers (all 112 entrants' `v` reproduce to <= 0.015, the input rounding; tested), not from
published code: the maintainers' code for this step is not public. Ties: "Models within 0.9 points of the first
model in their group share a rank" (`formula.tie_band`).

This module only does arithmetic on published aggregates. The private parts are run by the maintainers and are
never published per row, so no paired CI can be computed for them (see INDEX_0.3.md).
"""

import json

from harness import config

# data/v03.json formula (v0.3-final, decided 2026-10-06).
WEIGHTS = {"public": 0.2, "same_skills": 0.5, "new_domains": 0.3}
EQUATING = {"public": (23.5, 16.54), "same_skills": (24.27, 16.35), "new_domains": (23.68, 12.29)}
TIE_BAND = 0.9


def load_board(path):
    raw = open(path, "rb").read()
    import hashlib

    got = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw.decode("utf-8"))
    return d, got == config.BOARD_V03_SHA256


def params_from_board(board):
    """(weights, equating, tie_band) read from a v03.json dict, so a re-fetched board can be checked."""
    f = board["formula"]["equating"]
    return dict(board["weights"]), {k: (f[k]["mean"], f[k]["sd"]) for k in ("public", "same_skills", "new_domains")}, board["formula"]["tie_band"]


def equate(x, part, equating=EQUATING):
    mu_p, sd_p = equating["public"]
    mu, sd = equating[part]
    return mu_p + sd_p * (x - mu) / sd


def full(public, same_skills, new_domains, weights=WEIGHTS, equating=EQUATING, board_rounding=True):
    """Full score. The equated parts enter UNCLIPPED (the board shows `eq` floored at 0, but `v` uses the
    negative value: e.g. dinah-0 new-domains 5.48 -> eq -0.99, shown 0, and v 13.87 only reproduces with
    -0.99); the total is floored at 0. board_rounding: round to 2 dp like the board. The board's inputs are
    themselves rounded to 2 dp, so a recomputed `v` matches the published one to within 0.015."""
    s, o = equate(same_skills, "same_skills", equating), equate(new_domains, "new_domains", equating)
    v = max(0.0, weights["public"] * public + weights["same_skills"] * s + weights["new_domains"] * o)
    return round(v, 2) if board_rounding else v


def full_slope(weights=WEIGHTS, equating=EQUATING):
    """Full points per 1 raw point of each part (the equating stretches the private parts)."""
    sd_p = equating["public"][1]
    return {"public": weights["public"], "same_skills": weights["same_skills"] * sd_p / equating["same_skills"][1],
            "new_domains": weights["new_domains"] * sd_p / equating["new_domains"][1]}


def ranks(scores, tie=TIE_BAND):
    """{name: Full} -> {name: (rank, tied)}: a group starts at its first (highest) model and holds every
    following model within `tie` points of that first model (board rule, v03.json `rank_tie`)."""
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    out, leader, start = {}, None, 0
    for i, (k, v) in enumerate(ordered):
        if leader is None or leader - v >= tie - 1e-9:
            leader, start = v, i
        out[k] = start + 1
    counts = {}
    for r in out.values():
        counts[r] = counts.get(r, 0) + 1
    return {k: (r, counts[r] > 1) for k, r in out.items()}

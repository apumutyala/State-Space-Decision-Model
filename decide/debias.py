"""Option-order / option-prior controls for stock readouts (diagnostics only).

Proposed in the PR #6 internal review (B3). The m0-b smoke "first-listed option"
probe changed three things at once (sorted position, continuation length, an
``a_`` MCQ-letter prefix), so it is confounded. These controls each move one
thing:

(a) ``description_swap``: same ids, same order, same token counts; only the
    meaning-to-id mapping moves (descriptions rotated by one). A readout that
    reads descriptions moves its argmax with the description; one that ignores
    them keeps the same id ("id stickiness").
(b) ``neutral_cyclic``: ids replaced by neutral, length-matched ``opt_01..opt_k``;
    the k cyclic rotations present every original option at every position.
    Position bias = total-variation distance between the mean probability per
    presented position and uniform 1/k. The rotation-averaged probability per
    original option is reported as a labelled diagnostic variant, never served.
(c) ``content_free``: the same prompt with ``state = "N/A"`` (Zhao et al. 2021,
    contextual calibration); records the readout's prior per option/position.

Everything runs through ``Decider.decide`` so it works with any backend.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import canonical_options

CONTENT_FREE_STATE = "N/A"


def _copy(request: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(request))
    out.pop("ssdm", None)  # controls are plain decisions; no limits/restrictions carried over
    return out


def _choice(request: dict[str, Any], qid: str) -> dict[str, Any]:
    q = request["questions"][qid]
    if q["type"] != "choice" or len(q["criteria"]) < 2:
        raise ValueError(f"{qid}: debias controls need a choice question with >= 2 options")
    return q


def description_swap(request: dict[str, Any], qid: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Rotate descriptions by one over canonical ids. Returns (request, desc_owner) where
    desc_owner[original_id] = id that carries original_id's description after the swap."""
    _choice(request, qid)
    out = _copy(request)
    ids = [o for o, _ in canonical_options(out["questions"][qid])]
    descs = [out["questions"][qid]["criteria"][o] for o in ids]
    k = len(ids)
    out["questions"][qid]["criteria"] = {ids[(j + 1) % k]: descs[j] for j in range(k)}
    return out, {ids[j]: ids[(j + 1) % k] for j in range(k)}


def neutral_ids(k: int) -> list[str]:
    width = max(2, len(str(k)))
    return [f"opt_{i + 1:0{width}d}" for i in range(k)]  # sorts == presentation order


def neutral_cyclic(request: dict[str, Any], qid: str) -> list[tuple[int, dict[str, Any], dict[str, str]]]:
    """k variants; in rotation r original canonical option j is presented at position (j + r) % k.
    Returns [(r, request, id_of)] with id_of[original_id] = neutral id used in that variant."""
    _choice(request, qid)
    ids = [o for o, _ in canonical_options(request["questions"][qid])]
    k = len(ids)
    names = neutral_ids(k)
    variants = []
    for r in range(k):
        out = _copy(request)
        crit = request["questions"][qid]["criteria"]
        id_of = {ids[j]: names[(j + r) % k] for j in range(k)}
        out["questions"][qid]["criteria"] = {id_of[o]: crit[o] for o in ids}
        variants.append((r, out, id_of))
    return variants


def content_free(request: dict[str, Any]) -> dict[str, Any]:
    out = _copy(request)
    out["state"] = CONTENT_FREE_STATE
    return out


def _served(decider, request, qid) -> dict[str, float]:
    resp = decider.decide(request)
    s = resp["ssdm"]
    if s["status"] != "ok":
        raise RuntimeError(f"control request failed: {s['error']}")
    return s["served_probabilities"][qid]


def run_debias_controls(decider, request: dict[str, Any], qid: str) -> dict[str, Any]:
    _choice(request, qid)
    ids = [o for o, _ in canonical_options(request["questions"][qid])]
    k = len(ids)
    base = _served(decider, _copy(request), qid)
    base_top = max(sorted(base), key=base.__getitem__)

    swapped_req, owner = description_swap(request, qid)
    swapped = _served(decider, swapped_req, qid)
    swap_top = max(sorted(swapped), key=swapped.__getitem__)

    per_position = [0.0] * k
    per_option = {o: [] for o in ids}
    for r, req_r, id_of in neutral_cyclic(request, qid):
        p = _served(decider, req_r, qid)
        for j, o in enumerate(ids):
            per_option[o].append(p[id_of[o]])
            per_position[(j + r) % k] += p[id_of[o]] / k
    tv = 0.5 * sum(abs(x - 1.0 / k) for x in per_position)
    rot_avg = {o: sum(v) / k for o, v in per_option.items()}

    cf = _served(decider, content_free(request), qid)
    cf_first = cf[ids[0]]
    return {
        "question": qid,
        "k": k,
        "base": {"probabilities": base, "argmax": base_top, "argmax_is_sorted_first": base_top == ids[0]},
        "description_swap": {
            "probabilities": swapped,
            "argmax": swap_top,
            "follows_description": swap_top == owner[base_top],
            "id_sticky": swap_top == base_top,
            "note": "same ids/order/token counts; only descriptions moved",
        },
        "neutral_cyclic": {
            "ids": neutral_ids(k),
            "mean_prob_by_position": per_position,
            "position_bias_tv": tv,
            "rotation_averaged_probabilities": rot_avg,
            "rotation_averaged_argmax": max(sorted(rot_avg), key=rot_avg.__getitem__),
            "label": "diagnostic calibration variant; not served",
        },
        "content_free": {
            "state": CONTENT_FREE_STATE,
            "probabilities": cf,
            "prior_on_sorted_first": cf_first,
        },
    }

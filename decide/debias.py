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



# ------------------------------------------------------------------------------------- test-time order averaging
# Test-time ORDER AVERAGING (opt-in; ``OrderAveragedBackend``). Every CHOICE question with >= 2 options is scored
# under ``n`` presentation orders of its options and the per-option probabilities are averaged (arithmetic mean of
# the per-order probabilities; the backend returns log(mean p), so an identity-calibration decider serves the mean
# exactly). Everything is fixed in advance, nothing is fitted (no DEV fitting):
#
#   orders of a question (canonical option indices, in presentation order) = the first ``n`` DISTINCT of
#     0   identity (the N = 1 prompt, byte-identical)
#     1   reversed
#     2+  cyclic shifts (option s first): s in 1 .. k-2 ranked by sha256(ORDER_AVG_SEED | qid | option ids | s)
#         (seeded by the question's own id and option ids only: never gold, scores or other rows), then s = k-1 (its
#         first option is the reversed order's first, so it comes last); duplicates skipped
#   so k = 2 has 2 distinct orders (N = 2 and N = 4 are both identity + reversed) and k >= 3 has >= 4; with k >= 4 the
#   four N = 4 orders put four DIFFERENT options first (a primacy bias averages out exactly when the options tie).
# Presentation order = the order options are LISTED (schema field, re-read, code key) and, for coded questions, the
# code each option gets (code table position = presentation position): it moves position AND label priors. Questions
# that are not ``choice`` (noul; score is ordinal) are never reordered and keep the identity pass's logits unchanged,
# as do choice questions with < 2 options. Pass v of a request presents every eligible question in its v-th order
# (the others in identity order); a question with fewer than v + 1 distinct orders ignores pass v.
# ``n = 1`` is the base backend exactly (same revision, same ScoreResult object), so default readouts are unchanged.

import hashlib as _hashlib
import math as _math
import time as _time
from dataclasses import replace as _replace

ORDER_AVG_ID = "ssdm.order_avg.v1"
ORDER_AVG_SEED = "ssdm.order_avg.v1/cyclic"
ORDER_AVG_NS = (1, 2, 4)


def order_eligible(q: dict[str, Any]) -> bool:
    return q.get("type") == "choice" and len(q.get("criteria") or {}) >= 2


def order_variants(q: dict[str, Any], qid: str, n: int) -> list[list[int]]:
    """The first ``n`` distinct presentation orders of question ``qid`` (identity only when not eligible)."""
    if n < 1:
        raise ValueError("order averaging needs n >= 1")
    k = len(canonical_options(q))
    base = list(range(k))
    if not order_eligible(q):
        return [base]
    oids = [o for o, _ in canonical_options(q)]
    key = json.dumps([ORDER_AVG_SEED, qid, oids], ensure_ascii=True)
    shifts = sorted(range(1, k - 1), key=lambda s: (_hashlib.sha256(f"{key}|{s}".encode("ascii")).hexdigest(), s))
    shifts.append(k - 1)
    out: list[list[int]] = []
    for o in [base, base[::-1], *[base[s:] + base[:s] for s in shifts]]:
        if o not in out:
            out.append(o)
        if len(out) == n:
            break
    return out


def presentation_passes(request: dict[str, Any], n: int) -> list[dict[str, list[int]]]:
    """Pass v -> {qid: order} of the eligible questions that have a (v+1)-th distinct order; pass 0 = {} (identity).
    Trailing empty passes are dropped (no question needs them, so they cost nothing)."""
    qs = request["questions"]
    var = {k: order_variants(q, k, n) for k, q in qs.items() if order_eligible(q)}
    passes: list[dict[str, list[int]]] = [{}]
    passes += [{k: v[i] for k, v in var.items() if len(v) > i} for i in range(1, n)]
    while len(passes) > 1 and not passes[-1]:
        passes.pop()
    return passes


class OrderCache:
    """Pass results shared by several ``OrderAveragedBackend`` in one process (one job scores N = 4, 2 and 1 from the
    same forward passes: pass v of a request is the same computation in every arm). Key = (request hash, pass).
    The caller clears it per row."""

    def __init__(self) -> None:
        self.d: dict[tuple[str, int], tuple[Any, float]] = {}
        self.hits = 0

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        return _hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()

    def clear(self) -> None:
        self.d.clear()


class OrderAveragedBackend:
    """``decide`` backend wrapper: test-time order averaging over ``n`` presentation orders (n in ORDER_AVG_NS,
    pre-registered). The wrapped backend must accept ``presentation=`` in ``plan`` / ``score``
    (``heads.decide_backend.DecisionModelBackend``, template v3w). n = 1 returns the base backend's own result.
    ``last_pass_ms`` = the measured wall of each pass of the last request (a cached pass keeps the wall measured when
    it ran), so each arm's cost is the sum of its own passes."""

    def __init__(self, base, n: int, cache: OrderCache | None = None) -> None:
        if n not in ORDER_AVG_NS:
            raise ValueError(f"order averaging n={n}: one of {ORDER_AVG_NS} (pre-registered)")
        self.base, self.n, self.cache = base, n, cache
        bname = getattr(base, "name", "backend")
        self.name = bname if n == 1 else f"{bname}+oa{n}"
        self.last_pass_ms: list[float] = []
        self.last_fresh_ms = 0.0  # wall of the passes this call actually ran (not served from the cache)
        self.last_input_tokens = 0  # model input tokens of every pass this arm needs (its own cost, cached or not)

    def __getattr__(self, item):  # model / encode / template / checkpoint ... of the wrapped backend
        if item in ("base", "n", "cache", "last_pass_ms", "last_fresh_ms", "last_input_tokens"):
            raise AttributeError(item)
        return getattr(self.base, item)

    def revision(self) -> dict[str, Any]:
        rev = self.base.revision()
        if self.n == 1:
            return rev
        return {**rev, "revision_id": f"{rev['revision_id']}+OA:{ORDER_AVG_ID}:n{self.n}",
                "order_avg": {"id": ORDER_AVG_ID, "n": self.n, "seed": ORDER_AVG_SEED, "combine": "mean_probability"}}

    def _passes(self, request: dict[str, Any]) -> list[dict[str, list[int]]]:
        return presentation_passes(request, self.n) if self.n > 1 else [{}]

    def plan(self, request: dict[str, Any]):
        passes = self._passes(request)
        plans = [self.base.plan(request)] + [self.base.plan(request, presentation=p) for p in passes[1:]]
        if len(plans) == 1:
            return plans[0]
        return _replace(plans[0], inference_calls=sum(p.inference_calls for p in plans),
                        max_call_input_tokens=max(p.max_call_input_tokens for p in plans),
                        total_input_tokens=sum(p.total_input_tokens for p in plans))

    def _pass(self, request, v: int, pres: dict, check_deadline):
        key = (OrderCache.key(request), v) if self.cache is not None else None
        if key is not None and key in self.cache.d:
            self.cache.hits += 1
            return self.cache.d[key]
        t = _time.perf_counter()
        res = (self.base.score(request, check_deadline) if v == 0 else
               self.base.score(request, check_deadline, presentation=pres))
        out = (res, (_time.perf_counter() - t) * 1000)
        self.last_fresh_ms += out[1]
        if key is not None:
            self.cache.d[key] = out
        return out

    def score(self, request: dict[str, Any], check_deadline):
        passes = self._passes(request)
        self.last_fresh_ms = 0.0
        got = [self._pass(request, v, p, check_deadline) for v, p in enumerate(passes)]
        self.last_pass_ms = [ms for _, ms in got]
        self.last_input_tokens = sum(r.total_input_tokens for r, _ in got)
        base = got[0][0]
        if len(got) == 1:
            return base
        logits = dict(base.logits)
        n_orders: dict[str, int] = {}
        for qid in request["questions"]:
            used = [0] + [v for v in range(1, len(passes)) if qid in passes[v]]
            if len(used) == 1:
                continue
            opts = list(base.logits[qid])
            mean = {o: sum(_math.exp(got[v][0].logits[qid][o]) for v in used) / len(used) for o in opts}
            z = sum(mean.values())  # 1 up to float error when the base returns normalised log-probabilities
            logits[qid] = {o: (_math.log(mean[o] / z) if mean[o] > 0 else -1e30) for o in opts}
            n_orders[qid] = len(used)
        extras = {**(base.extras or {}), "order_avg": {"id": ORDER_AVG_ID, "n": self.n, "passes": len(passes),
                                                        "orders_per_question": n_orders,
                                                        "pass_ms": list(self.last_pass_ms)}}
        return _replace(base, logits=logits, inference_calls=sum(r.inference_calls for r, _ in got),
                        total_input_tokens=sum(r.total_input_tokens for r, _ in got), extras=extras)

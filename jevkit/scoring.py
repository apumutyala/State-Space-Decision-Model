"""JevBench v1.6.1 scorer: Intelligence, Calibration, Speed, Cost, Capability, Composite A/B/C, bootstrap CIs.

Two layers share one set of formulas:

* scalar formulas (``tiered_cc``, ``split_intelligence``, ``intelligence``, ``calibration_*``, ``speed``,
  ``cost_score``, ``capability``, ``composite``, ``views``) - used both here and by ``jevkit.board`` to recompute the
  published rows from their published sub-fields;
* item-level aggregation (``ItemTable`` + ``aggregate``) - per item answers -> the same sub-fields, and the paired,
  stratified (split x type x tier) bootstrap that recomputes Intelligence and Calibration per resample with Speed and
  Cost held fixed (B = 1000, seed 16 as published for v1.6.1).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import spec
from .answers import Parsed, parse_answer
from .items import Item

TYPE_IX = {t: i for i, t in enumerate(spec.TYPES)}
TIER_IX = {t: i for i, t in enumerate(spec.TIERS)}
SIDES = ("open", "sealed")


# =============================================================================================== scalar formulas
def tiered_cc(tiers: dict[str, float | None]) -> float | None:
    """Sum_tier w_tier * CC_tier, renormalised over the tiers present (unclipped tiers)."""
    pres = {t: v for t, v in tiers.items() if v is not None}
    w = sum(spec.TIER_WEIGHTS[t] for t in pres)
    return sum(spec.TIER_WEIGHTS[t] * v for t, v in pres.items()) / w if w else None


def split_intelligence(cc_by_type: dict[str, float | None]) -> float | None:
    """I_split = mean over present types of max(0, CC_type) (each split x type clipped at 0, types 1/3 each)."""
    vals = [max(0.0, v) for v in cc_by_type.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def intelligence(i_open: float | None, i_sealed: float | None, g_med: float = spec.G_MED_V161) -> dict:
    """base = 0.5 I_open + 0.5 I_sealed; penalty = max(0, 1 - max(0, gap - G_med - 8) / 100); I = clip(base * penalty).

    With only one split present the base is that split, and no gap (hence no penalty) can be computed.
    """
    if i_open is not None and i_sealed is not None:
        base = (1 - spec.SEALED_SHARE) * i_open + spec.SEALED_SHARE * i_sealed
        gap = i_open - i_sealed
        excess = gap - g_med
        penalty = max(0.0, 1.0 - max(0.0, excess - spec.GAP_ALLOWANCE) / spec.PENALTY_SLOPE_DEN)
        basis = "open+sealed"
    elif i_open is not None or i_sealed is not None:
        base = i_open if i_open is not None else i_sealed
        gap = excess = None
        penalty = 1.0
        basis = "open only (no sealed items; no gap penalty)" if i_open is not None else "sealed only (no gap penalty)"
    else:
        return {"I_open": None, "I_sealed": None, "base": None, "gap": None, "excess": None, "penalty": None,
                "I": None, "basis": "no items"}
    return {"I_open": i_open, "I_sealed": i_sealed, "base": base, "gap": gap, "excess": excess, "penalty": penalty,
            "I": min(100.0, max(0.0, base * penalty)), "basis": basis}


def calibration_choice(ece: float) -> float:
    return max(0.0, 100.0 * (1.0 - 2.0 * ece))


def calibration_noul(ece: float) -> float:
    return max(0.0, 100.0 * (1.0 - 2.0 * ece))


def calibration_score(top_ece: float, nrps: float) -> float:
    """50 (1 - 2 topECE) + 50 (1 - nRPS), each half clipped at 0 (raw-qwen3-0.6b: 32.82, classone-gemma4-e2b)."""
    return 50.0 * max(0.0, 1.0 - 2.0 * top_ece) + 50.0 * max(0.0, 1.0 - nrps)


def calibration(parts: dict[str, float | None], supported: Iterable[str] = spec.TYPES) -> float | None:
    """Mean over supported types; a supported type with no scoreable distribution contributes 0 (jade: 50.56)."""
    sup = list(supported)
    if not sup:
        return None
    return sum((parts.get(t) or 0.0) for t in sup) / len(sup)


def adjust_latency(s: float, self_hosted: bool) -> float:
    return spec.SELF_HOSTED_FACTOR * s + spec.SELF_HOSTED_OFFSET_S if self_hosted else s


def speed_f(s: float) -> float:
    if s <= 0:
        return 100.0
    return min(100.0, max(0.0, 100.0 - spec.SPEED_SLOPE * math.log10(s / spec.SPEED_REF_S)))


def speed(p50_s: float | None, p95_s: float | None, self_hosted: bool = True, adjusted: bool = False) -> dict:
    """Speed = mean(f(p50'), f(p95')); s' = 2 s + 0.15 for self-hosted / demo endpoints, unadjusted for APIs."""
    if p50_s is None or p95_s is None:
        return {"p50_s_raw": p50_s, "p95_s_raw": p95_s, "p50_s_adjusted": None, "p95_s_adjusted": None, "score": None}
    a50 = p50_s if adjusted else adjust_latency(p50_s, self_hosted)
    a95 = p95_s if adjusted else adjust_latency(p95_s, self_hosted)
    return {"p50_s_raw": None if adjusted else p50_s, "p95_s_raw": None if adjusted else p95_s,
            "p50_s_adjusted": a50, "p95_s_adjusted": a95,
            "adjustment": "x2 + 0.15 s" if (self_hosted and not adjusted) else ("given" if adjusted else "none (API)"),
            "score": (speed_f(a50) + speed_f(a95)) / 2.0}


def cost_score(usd_per_1000: float | None) -> float | None:
    if usd_per_1000 is None:
        return None
    if usd_per_1000 <= 0:
        return 100.0
    return min(100.0, max(0.0, 100.0 - spec.COST_SLOPE * math.log10(usd_per_1000 / spec.COST_REF_USD_PER_1000)))


def capability(i: float | None, c: float | None) -> float | None:
    return None if i is None or c is None else (i + c) / 2.0


def composite(i: float | None, c: float | None, s: float | None, k: float | None, option: str = spec.HEADLINE) -> float | None:
    """H = sum w / sum(w / axis) x min(1, I/floor)^2 x min(1, S/50)^2 x min(1, K/50)^2; zero axis -> 0, missing -> None."""
    axes = (i, c, s, k)
    if any(a is None for a in axes):
        return None
    if any(a <= 0 for a in axes):
        return 0.0
    o = spec.OPTIONS[option]
    w = o["weights"]
    h = sum(w) / sum(wi / a for wi, a in zip(w, axes))
    return h * min(1.0, i / o["intelligence_floor"]) ** 2 * min(1.0, s / spec.AXIS_GATE) ** 2 * min(1.0, k / spec.AXIS_GATE) ** 2


def views(i: float | None, s: float | None, k: float | None) -> dict[str, float | None]:
    """Sort views over (Intelligence, Speed, Cost) with the same gates. Not scored; see README (partial match)."""
    out: dict[str, float | None] = {}
    for name, w in spec.VIEWS.items():
        if None in (i, s, k):
            out[name] = None
        elif min(i, s, k) <= 0:
            out[name] = 0.0
        else:
            h = sum(w) / (w[0] / i + w[1] / s + w[2] / k)
            out[name] = h * min(1.0, i / 50.0) ** 2 * min(1.0, s / 50.0) ** 2 * min(1.0, k / 50.0) ** 2
    out["Intelligence only"] = i
    return out


def eligibility(usd_per_1000: float | None, p50_adjusted_s: float | None) -> dict:
    ok_c = usd_per_1000 is not None and usd_per_1000 <= spec.CAP_USD_PER_1000
    ok_l = p50_adjusted_s is not None and p50_adjusted_s <= spec.CAP_P50_ADJUSTED_S
    return {"capability_ranked": ok_c and ok_l, "within_cost_cap": ok_c, "within_latency_cap": ok_l,
            "cap_usd_per_1000": spec.CAP_USD_PER_1000, "cap_p50_adjusted_s": spec.CAP_P50_ADJUSTED_S}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = spec.ECE_BINS) -> float | None:
    """Equal-width-bin ECE: sum_b (n_b / n) |acc_b - conf_b| = sum_b |sum correct - sum conf| / n."""
    n = len(conf)
    if n == 0:
        return None
    c = np.clip(conf.astype(float), 0.0, 1.0)
    ix = np.minimum((c * bins).astype(int), bins - 1)
    sc = np.bincount(ix, weights=c, minlength=bins)
    sa = np.bincount(ix, weights=correct.astype(float), minlength=bins)
    return float(np.abs(sa - sc).sum() / n)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation (the public jevbench metrics.percentile)."""
    if not len(values):
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return v[int(k)] if f == c else v[f] * (c - k) + v[c] * (k - f)


# =============================================================================================== item table
@dataclass
class ItemTable:
    """Per-item arrays for one system over a fixed item list (a missing prediction is an invalid answer)."""

    ids: list[str]
    side: np.ndarray  # 0 open, 1 sealed
    typ: np.ndarray  # TYPE_IX
    tier: np.ndarray  # TIER_IX
    valid: np.ndarray
    correct: np.ndarray  # Intelligence correctness (choice argmax; noul decisive; score: argmax, diagnostic only)
    chance: np.ndarray  # choice 1/k
    nmae: np.ndarray  # score: |E[level] - gold| / (K-1); invalid -> 1.0 (maximal)
    nmae_chance: np.ndarray  # score: mean_l |l - gold| / (K-1)
    in_c: np.ndarray  # valid, has a distribution, not excluded from Calibration
    conf: np.ndarray  # choice / score: top-label probability
    top_correct: np.ndarray  # choice / score: argmax == gold
    pyes: np.ndarray  # noul P(yes)
    yes: np.ndarray  # noul gold is yes
    abstain: np.ndarray  # noul valid and 0.2 < p < 0.8
    nrps: np.ndarray  # score: normalised RPS
    renormalized: np.ndarray
    label_only: np.ndarray
    parsed: list[Parsed] = field(default_factory=list, repr=False)


def build_table(items: Sequence[Item], answers: dict[str, Any], exclude_from_c: Iterable[str] = ()) -> ItemTable:
    """``answers``: item id -> SystemOne answer (or None). Items with expected=None are dropped (unmeasured gold)."""
    excl = set(exclude_from_c)
    items = [it for it in items if it.expected is not None]
    n = len(items)
    z = lambda dt=float: np.zeros(n, dtype=dt)
    t = ItemTable(ids=[it.id for it in items], side=z(int), typ=z(int), tier=z(int), valid=z(bool), correct=z(bool),
                  chance=z(), nmae=z(), nmae_chance=z(), in_c=z(bool), conf=z(), top_correct=z(bool), pyes=z(),
                  yes=z(bool), abstain=z(bool), nrps=z(), renormalized=z(bool), label_only=z(bool))
    for i, it in enumerate(items):
        p = parse_answer(it, answers.get(it.id))
        t.parsed.append(p)
        t.side[i] = 0 if it.side == "open" else 1
        t.typ[i] = TYPE_IX[it.type]
        t.tier[i] = TIER_IX[it.tier]
        t.valid[i] = p.valid
        t.renormalized[i] = p.renormalized
        t.label_only[i] = p.label_only
        t.in_c[i] = p.has_probs and it.id not in excl
        gold = str(it.expected)
        if it.type == "choice":
            t.chance[i] = 1.0 / it.k
            t.correct[i] = p.valid and p.predicted == gold
            if p.has_probs:
                t.conf[i] = max(p.probs.values())
                t.top_correct[i] = p.predicted == gold
        elif it.type == "noul":
            t.yes[i] = gold == "yes"
            t.correct[i] = p.valid and p.predicted == gold
            if p.valid:
                t.pyes[i] = p.probs["yes"]
                t.abstain[i] = p.predicted is None
        else:
            g, km1 = int(it.expected), it.k - 1
            t.nmae_chance[i] = sum(abs(lv - g) for lv in range(it.k)) / it.k / km1
            t.nmae[i] = abs(p.expected_level - g) / km1 if p.valid else 1.0
            t.correct[i] = p.valid and p.predicted == gold
            if p.has_probs:
                t.conf[i] = max(p.probs.values())
                t.top_correct[i] = p.predicted == gold
                cum, rps = 0.0, 0.0
                for lv in range(km1):
                    cum += p.probs[str(lv)]
                    rps += (cum - (1.0 if g <= lv else 0.0)) ** 2
                t.nrps[i] = rps / km1
    return t


def _cell_cc(t: ItemTable, m: np.ndarray, typ: str) -> float | None:
    if not m.any():
        return None
    if typ == "choice":
        acc, cbar = t.correct[m].mean(), t.chance[m].mean()
        return 100.0 * (acc - cbar) / (1.0 - cbar)
    if typ == "noul":
        return 100.0 * (t.correct[m].mean() - 0.5) / 0.5
    return 100.0 * (1.0 - t.nmae[m].mean() / t.nmae_chance[m].mean())


def aggregate(t: ItemTable, idx: np.ndarray | None = None, g_med: float = spec.G_MED_V161,
              supported: Sequence[str] = spec.TYPES, choice_ece_tiers: Sequence[str] = spec.CHOICE_ECE_TIERS,
              detail: bool = True) -> dict:
    """Intelligence + Calibration over the items ``idx`` (all when None; a bootstrap passes a resample)."""
    if idx is not None:
        sub = ItemTable(ids=[], **{k: getattr(t, k)[idx] for k in (
            "side", "typ", "tier", "valid", "correct", "chance", "nmae", "nmae_chance", "in_c", "conf",
            "top_correct", "pyes", "yes", "abstain", "nrps", "renormalized", "label_only")})
        t = sub
    per_split: dict[str, dict] = {}
    cc_by_side: dict[str, dict[str, float | None]] = {s: {} for s in SIDES}
    for si, side in enumerate(SIDES):
        ms = t.side == si
        for typ in supported:
            mt = ms & (t.typ == TYPE_IX[typ])
            tiers, ns = {}, {}
            for tier in spec.TIERS:
                m = mt & (t.tier == TIER_IX[tier])
                if m.any():
                    tiers[tier] = _cell_cc(t, m, typ)
                    ns[tier] = int(m.sum())
            cc = tiered_cc(tiers) if tiers else None
            cc_by_side[side][typ] = cc
            if detail and tiers:
                per_split[f"{side}|{typ}"] = {"cc": None if cc is None else max(0.0, cc), "cc_unclipped": cc,
                                              "n": ns, "tiers": tiers}
    i_open = split_intelligence(cc_by_side["open"]) if any(v is not None for v in cc_by_side["open"].values()) else None
    i_sealed = split_intelligence(cc_by_side["sealed"]) if any(v is not None for v in cc_by_side["sealed"].values()) else None
    intel = intelligence(i_open, i_sealed, g_med)

    parts: dict[str, dict] = {}
    present = [typ for typ in supported if (t.typ == TYPE_IX[typ]).any()]
    for typ in present:
        mt = (t.typ == TYPE_IX[typ]) & t.in_c
        if typ == "choice":
            tier_mask = np.isin(t.tier, [TIER_IX[x] for x in choice_ece_tiers])
            m = mt & tier_mask
            basis = "+".join(choice_ece_tiers) + " tier"
            if not (t.typ == TYPE_IX[typ])[tier_mask].any():  # no items of those tiers at all (e.g. an easy-only set)
                m, basis = mt, "all tiers (no items in " + "+".join(choice_ece_tiers) + ")"
            e = ece(t.conf[m], t.top_correct[m])
            parts[typ] = {"ece_hard": e, "n": int(mt.sum()), "n_ece_hard": int(m.sum()), "ece_basis": basis,
                          "score": None if e is None else calibration_choice(e)}
        elif typ == "noul":
            e = ece(t.pyes[mt], t.yes[mt])
            mv = (t.typ == TYPE_IX[typ]) & t.valid
            parts[typ] = {"ece": e, "n": int(mt.sum()),
                          "brier": float(np.mean((t.pyes[mt] - t.yes[mt]) ** 2)) if mt.any() else None,  # unscored
                          "abstention_rate": float(t.abstain[mv].mean()) if mv.any() else None,
                          "score": None if e is None else calibration_noul(e)}
        else:
            e = ece(t.conf[mt], t.top_correct[mt])
            nr = float(t.nrps[mt].mean()) if mt.any() else None
            parts[typ] = {"top_ece": e, "nrps": nr, "n": int(mt.sum()),
                          "score": None if e is None else calibration_score(e, nr)}
    cal = calibration({k: v["score"] for k, v in parts.items()}, present) if present else None
    out = {"intelligence": {**intel}, "calibration": {"parts": parts, "score": cal}}
    if detail:
        out["intelligence"]["per_type_split"] = per_split
    return out


# =============================================================================================== bootstrap
def strata(t: ItemTable) -> list[np.ndarray]:
    key = t.side * 100 + t.typ * 10 + t.tier
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def bootstrap(tables: dict[str, ItemTable], speed_cost: dict[str, tuple[float | None, float | None]],
              b: int = spec.BOOTSTRAP_B, seed: int = spec.BOOTSTRAP_SEED, g_med: float = spec.G_MED_V161,
              supported: Sequence[str] = spec.TYPES) -> dict[str, dict]:
    """Paired stratified bootstrap over the SAME resample for every system (tables must share the item order).

    Per resample: Intelligence and Calibration recomputed; Speed and Cost held fixed. Returns per system the 95 %
    percentile intervals of I, C, capability and composite A/B/C, plus pairwise composite-A difference intervals.
    """
    names = list(tables)
    ref = tables[names[0]]
    for nm in names[1:]:
        if tables[nm].ids != ref.ids:
            raise ValueError("paired bootstrap needs every system scored on the same item list")
    rng = np.random.default_rng(seed)
    st = strata(ref)
    draws: dict[str, dict[str, list]] = {nm: {k: [] for k in ("I", "C", "capability", "A", "B", "C_opt")} for nm in names}
    for _ in range(b):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        for nm in names:
            r = aggregate(tables[nm], idx, g_med=g_med, supported=supported, detail=False)
            i, c = r["intelligence"]["I"], r["calibration"]["score"]
            s, k = speed_cost[nm]
            d = draws[nm]
            d["I"].append(i)
            d["C"].append(c)
            d["capability"].append(capability(i, c))
            d["A"].append(composite(i, c, s, k, "A"))
            d["B"].append(composite(i, c, s, k, "B"))
            d["C_opt"].append(composite(i, c, s, k, "C"))

    def ci(v):
        v = [x for x in v if x is not None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

    out = {}
    for nm in names:
        d = draws[nm]
        out[nm] = {"B": b, "seed": seed, "I": ci(d["I"]), "C": ci(d["C"]), "capability": ci(d["capability"]),
                   "composite": {"A": ci(d["A"]), "B": ci(d["B"]), "C": ci(d["C_opt"])}}
    if len(names) > 1:
        pairs = {}
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                a, bb = draws[names[x]]["A"], draws[names[y]]["A"]
                diff = [p - q for p, q in zip(a, bb) if p is not None and q is not None]
                lo_hi = ci(diff)
                pairs[f"{names[x]} - {names[y]}"] = {"composite_A_diff_ci95": lo_hi,
                                                     "tie": None if lo_hi is None else lo_hi[0] <= 0 <= lo_hi[1],
                                                     "p_first_better": float(np.mean([v > 0 for v in diff])) if diff else None}
        out["_pairs"] = pairs
    return out


# =============================================================================================== full report
def validity_summary(items: Sequence[Item], t: ItemTable) -> dict:
    out = {}
    for typ in spec.TYPES:
        m = t.typ == TYPE_IX[typ]
        if m.any():
            out[typ] = {"n": int(m.sum()), "invalid_rate": float(1 - t.valid[m].mean()),
                        "renormalized": int(t.renormalized[m].sum()), "label_only": int(t.label_only[m].sum())}
    nm = t.typ == TYPE_IX["noul"]
    if nm.any():
        v = nm & t.valid
        dec = v & ~t.abstain
        out["noul_decisive"] = {"valid": int(v.sum()), "decisive_rate": float(dec.sum() / v.sum()) if v.any() else None,
                                "acc_among_decisive": float(t.correct[dec].mean()) if dec.any() else None}
    return out


def score_system(items: Sequence[Item], answers: dict[str, Any], *, speed_axis: dict | None = None,
                 cost_axis: dict | None = None, g_med: float = spec.G_MED_V161, supported: Sequence[str] = spec.TYPES,
                 exclude_from_c: Iterable[str] = (), b: int = spec.BOOTSTRAP_B, seed: int = spec.BOOTSTRAP_SEED) -> dict:
    """One system's full JevBench-format report. ``speed_axis`` = ``speed(...)``; ``cost_axis`` = meter cost block."""
    t = build_table(items, answers, exclude_from_c)
    agg = aggregate(t, g_med=g_med, supported=supported)
    i, c = agg["intelligence"]["I"], agg["calibration"]["score"]
    s = (speed_axis or {}).get("score")
    k = (cost_axis or {}).get("score")
    rep = {
        "n_items": len(t.ids),
        "intelligence": agg["intelligence"],
        "calibration": agg["calibration"],
        "speed": speed_axis,
        "cost": cost_axis,
        "axes": {"intelligence": i, "calibration": c, "speed": s, "cost": k},
        "capability": capability(i, c),
        "scores": {o: composite(i, c, s, k, o) for o in spec.OPTIONS},
        "views": views(i, s, k),
        "eligibility": eligibility((cost_axis or {}).get("usd_per_1000"), (speed_axis or {}).get("p50_s_adjusted")),
        "validity": validity_summary(items, t),
        "support": {typ: ("native" if typ in supported else "not supported") for typ in spec.TYPES},
        "g_med_used": g_med,
    }
    if b:
        rep["ci95"] = bootstrap({"self": t}, {"self": (s, k)}, b=b, seed=seed, g_med=g_med, supported=supported)["self"]
    return rep

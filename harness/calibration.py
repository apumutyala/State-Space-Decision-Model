"""Calibration error: board-protocol ECE, debiased ECE and SmoothECE, overall and per length bin.

Confidence is the probability on the chosen option (board definition); for a yes/no `noul`
field it is max(p, 1 - p). Correct = chosen == gold (accepted ties for ChessBench/Habermas).
Rows that are not `ok` carry no confidence and are left out; rows listed in `trained_on`
count as wrong (board rule).
"""

import collections
import hashlib
import math

import numpy as np

from decision_index.engines.base import validate as kit_validate

from harness import config

# Board ECE: "32 benchmarks with a per-field right or wrong answer (ToolRet, BRIGHT, RouterBench
# and ForecastBench excluded), each weighted equally; a 1-in-6 sample of requests".
BOARD_EXCLUDED = {2, 36, 6, 48}


ADDED_02 = {56, 57, 58, 59, 61, 62, 64}  # 0.2 additions; most likely outside the board's 32
HEADLINE_TRACK = {40: "iSarcasmEval-A-En"}  # only this track enters the index


class CalibrationInputError(ValueError):
    """Malformed calibration input. Raised instead of reporting a (misleading) ECE of 0."""


def validate(conf, correct, weights=None, min_n=1):
    """Return float arrays after checking shapes, finite confidence in [0, 1], binary correctness
    and non-negative finite weights with positive total. Empty input is rejected (callers that
    want "unavailable" check emptiness first)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.ndim != 1 or correct.ndim != 1:
        raise CalibrationInputError(f"conf/correct must be 1-D, got shapes {conf.shape} and {correct.shape}")
    if conf.shape != correct.shape:
        raise CalibrationInputError(f"conf and correct differ in length: {conf.size} vs {correct.size}")
    if conf.size < min_n:
        raise CalibrationInputError(f"need at least {min_n} fields, got {conf.size}")
    if not np.all(np.isfinite(conf)) or np.any((conf < 0) | (conf > 1)):
        raise CalibrationInputError("confidence must be finite and inside [0, 1]")
    if not np.all((correct == 0) | (correct == 1)):
        raise CalibrationInputError("correct must be binary (0/1)")
    if weights is None:
        return conf, correct, None
    w = np.asarray(weights, dtype=float)
    if w.shape != conf.shape:
        raise CalibrationInputError(f"weights length {w.size} differs from {conf.size}")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise CalibrationInputError("weights must be finite, non-negative, with positive total")
    return conf, correct, w


def fields(rows, results, trained_on=(), exclude=BOARD_EXCLUDED, headline_only=True):
    trained_on = set(trained_on)
    out = []
    for r in rows:
        e = r["_evaluation"]
        n = e["catalog_id"]
        if n in exclude:
            continue
        if headline_only and n in HEADLINE_TRACK and e["track"] != HEADLINE_TRACK[n]:
            continue
        res = results.get(e["run_id"])
        if not res or res["status"] != "ok":
            continue
        try:
            kit_validate(r["questions"], res["response"])
        except Exception as exc:
            raise CalibrationInputError(f"{e['run_id']}: response fails the kit contract: {exc}") from exc
        answers = res["response"]["answers"]
        unscored = set((r.get("scoring") or {}).get("unscored_fields", []))
        only = {"verdict"} if n == 56 else None
        for key, q in r["questions"].items():
            gold = r["expected"].get(key)
            if gold is None or key in unscored or (only and key not in only):
                continue
            a = answers[key]
            if q["type"] == "noul":
                p = a["noul"]
                chosen, conf = p >= 0.5, max(p, 1 - p)
            else:
                chosen = a["choice"]
                conf = a["probabilities"][chosen]
            accepted = (r.get("scoring") or {}).get("accepted") if n in (31, 50) else None
            correct = (chosen in accepted) if accepted else (chosen == gold)
            if e["run_id"] in trained_on:
                correct = False
            out.append((n, config.length_bin(e["proxy_tokens"]), e["run_id"], float(conf), float(correct)))
    if out:
        validate([f[3] for f in out], [f[4] for f in out])
    return out


def one_in_six(run_id, seed=20260919):
    """Our stand-in for the board's undisclosed 1-in-6 request sample (hash of run_id)."""
    return int(hashlib.sha256(f"{seed}:{run_id}".encode()).hexdigest(), 16) % 6 == 0


def binned_ece(conf, correct, weights=None, bins=10):
    """L1 ECE with equal-width bins on [0, 1] (board: 10 bins, benchmark-equal weights)."""
    conf, correct, w = validate(conf, correct, weights)
    if not (isinstance(bins, int) and bins >= 1):
        raise CalibrationInputError("bins must be a positive integer")
    w = np.ones_like(conf) if w is None else w
    w = w / w.sum()
    b = np.minimum((conf * bins).astype(int), bins - 1)
    ece = 0.0
    rel = []
    for k in range(bins):
        m = b == k
        wk = w[m].sum()
        if wk == 0:
            continue
        c, a = (w[m] * conf[m]).sum() / wk, (w[m] * correct[m]).sum() / wk
        ece += wk * abs(c - a)
        rel.append({"lo": k / bins, "w": float(wk), "conf": float(c), "acc": float(a)})
    return float(ece), rel


def board_ece(field_list, sample=True, base_only=True):
    """Board protocol: 1-in-6 request sample, each benchmark weighted equally, 10 bins.

    base_only drops the seven 0.2 additions (the board's 32 are most likely base-edition
    benchmarks; MMLU/ARC/SGD are not index benchmarks and so are not in our slices)."""
    fl = [f for f in field_list if (not sample or one_in_six(f[2])) and (not base_only or f[0] not in ADDED_02)]
    if not fl:
        return {"n": 0, "available": False}
    per = collections.Counter(f[0] for f in fl)
    w = [1.0 / per[f[0]] for f in fl]
    ece, rel = binned_ece([f[3] for f in fl], [f[4] for f in fl], w)
    acc = float(np.average([f[4] for f in fl], weights=w))
    conf = float(np.average([f[3] for f in fl], weights=w))
    return {"ece": ece, "acc": acc, "conf": conf, "n": len(fl), "benchmarks": len(per), "rel": rel}


def debiased_ece(conf, correct, bins=15):
    """Debiased L2 calibration error (Kumar, Liang & Ma 2019), equal-mass bins.

    Returns sqrt(max(0, sum_b p_b[(acc_b - conf_b)^2 - acc_b(1 - acc_b)/(n_b - 1)])) and the
    signed squared estimate (which can be negative when calibration error is below noise).
    Needs at least 2 fields (the debiasing term divides by n_b - 1)."""
    conf, correct, _ = validate(conf, correct, min_n=2)
    n = len(conf)
    if n < 2 * bins:
        bins = max(1, n // 2)
    order = np.argsort(conf, kind="stable")
    parts = np.array_split(order, bins)
    sq = 0.0
    for idx in parts:
        nb = len(idx)
        if nb < 2:
            continue
        a, c = correct[idx].mean(), conf[idx].mean()
        sq += nb / n * ((a - c) ** 2 - a * (1 - a) / (nb - 1))
    return {"debiased_l2": math.sqrt(max(sq, 0.0)), "debiased_sq": float(sq), "bins": bins}


def _reflected_kernel(t, f, sigma):
    """Gaussian kernel reflected at 0 and 1, evaluated on grid t (G,) for points f (n,) -> (G, n)."""
    d = t[:, None] - f[None, :]
    k = np.zeros_like(d)
    for m in (-2, 0, 2):
        k += np.exp(-0.5 * ((d + m) / sigma) ** 2)
        k += np.exp(-0.5 * ((t[:, None] + f[None, :] + m) / sigma) ** 2)
    return k / (sigma * math.sqrt(2 * math.pi))


HIST = 4096  # confidences are pre-binned to 1/4096; error well below any reported digit


def _hist(conf, correct):
    f, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((f * HIST).astype(int), HIST - 1)
    resid = np.bincount(idx, weights=y - f, minlength=HIST) / len(f)
    centers = (np.arange(HIST) + 0.5) / HIST
    return centers, resid


def smece_sigma(conf, correct, sigma, _h=None):
    """smECE at bandwidth sigma: integral over [0,1] of |(1/n) sum_i K_sigma(t, f_i)(y_i - f_i)| dt.

    The integration grid is refined with sigma (>= 8 points per sigma) so mass at the
    boundary (e.g. one-hot confidences of 1.0) is not missed."""
    if not (math.isfinite(sigma) and sigma > 0):
        raise CalibrationInputError("sigma must be positive and finite")
    if _h is None:
        conf, correct, _ = validate(conf, correct)
    centers, resid = _h or _hist(conf, correct)
    keep = resid != 0
    centers, resid = centers[keep], resid[keep]
    grid = int(min(16384, max(512, 8 / sigma)))
    t = (np.arange(grid) + 0.5) / grid
    total = np.zeros(grid)
    for start in range(0, len(centers), 512):
        sl = slice(start, start + 512)
        total += _reflected_kernel(t, centers[sl], sigma) @ resid[sl]
    return float(np.abs(total).sum() / grid)


def smooth_ece(conf, correct, tol=1e-4):
    """SmoothECE (Blasiok & Nakkiran 2023): kernel-smoothed residual with a reflected Gaussian
    kernel at the fixed-point bandwidth sigma* where smECE_sigma = sigma."""
    conf, correct, _ = validate(conf, correct)
    h = _hist(conf, correct)
    lo, hi = 1e-3, 1.0
    if smece_sigma(conf, correct, lo, h) <= lo:
        return {"smece": smece_sigma(conf, correct, lo, h), "sigma": lo}
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if smece_sigma(conf, correct, mid, h) > mid:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return {"smece": smece_sigma(conf, correct, s, h), "sigma": s}


def report(field_list):
    """Overall and per-length-bin calibration summary."""
    def block(fl):
        if not fl:
            return {"n": 0, "available": False}
        conf = [f[3] for f in fl]
        cor = [f[4] for f in fl]
        ece, _ = binned_ece(conf, cor)
        per = collections.Counter(f[0] for f in fl)
        ece_eq, _ = binned_ece(conf, cor, [1.0 / per[f[0]] for f in fl])
        return {"n": len(fl), "benchmarks": len(per), "largest_benchmark_share": max(per.values()) / len(fl), "acc": float(np.mean(cor)), "conf": float(np.mean(conf)), "ece_10bin": ece, "ece_10bin_benchmark_equal": ece_eq, **debiased_ece(conf, cor), **smooth_ece(conf, cor)}

    by_bin = collections.defaultdict(list)
    for f in field_list:
        by_bin[f[1]].append(f)
    return {
        "overall": block(field_list),
        "board_protocol": board_ece(field_list),
        "by_length_bin": {b: block(by_bin[b]) for b in config.LENGTH_BIN_LABELS},
    }

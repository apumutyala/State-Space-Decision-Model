"""Decision metrics for eval cards: accuracy, NLL, Brier, top-label ECE (binned and debiased).

``debiased_ece`` mirrors ``harness.calibration.debiased_ece`` on claude/m0-harness (Kumar,
Liang & Ma 2019, equal-mass bins) so cards are comparable before that branch is merged;
when ``harness`` is importable the eval card uses it directly.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def binned_ece(conf: Sequence[float], correct: Sequence[float], bins: int = 15) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def debiased_ece(conf: Sequence[float], correct: Sequence[float], bins: int = 15) -> dict:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n < 2:
        return {"debiased_l2": float("nan"), "debiased_sq": float("nan"), "bins": 0}
    if n < 2 * bins:
        bins = max(1, n // 2)
    order = np.argsort(conf, kind="stable")
    sq = 0.0
    for idx in np.array_split(order, bins):
        nb = len(idx)
        if nb < 2:
            continue
        a, c = correct[idx].mean(), conf[idx].mean()
        sq += nb / n * ((a - c) ** 2 - a * (1 - a) / (nb - 1))
    return {"debiased_l2": math.sqrt(max(sq, 0.0)), "debiased_sq": float(sq), "bins": bins}


def length_bin(n_tokens: int, edges: Sequence[int] = (512, 2048, 8192)) -> str:
    lo = 0
    for e in edges:
        if n_tokens < e:
            return f"[{lo},{e})"
        lo = e
    return f"[{lo},inf)"


def decision_metrics(probs: list[list[float]], labels: list[int], lengths: list[int] | None = None) -> dict:
    conf, correct, nlls, briers = [], [], [], []
    for p, y in zip(probs, labels):
        p = np.asarray(p, float)
        top = int(np.argmax(p))
        conf.append(float(p[top]))
        correct.append(float(top == y))
        nlls.append(-math.log(max(p[y], 1e-12)))
        oh = np.zeros_like(p)
        oh[y] = 1
        briers.append(float(((p - oh) ** 2).sum()))
    out = {
        "n": len(labels),
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
        "nll": float(np.mean(nlls)) if nlls else float("nan"),
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "ece_binned15": binned_ece(conf, correct),
        "ece_debiased": debiased_ece(conf, correct),
        "mean_conf": float(np.mean(conf)) if conf else float("nan"),
    }
    if lengths is not None:
        per = {}
        for b in sorted({length_bin(n) for n in lengths}):
            ix = [i for i, n in enumerate(lengths) if length_bin(n) == b]
            per[b] = {"n": len(ix), "accuracy": float(np.mean([correct[i] for i in ix])),
                      "ece_debiased": debiased_ece([conf[i] for i in ix], [correct[i] for i in ix])}
        out["by_length_bin"] = per
    return out

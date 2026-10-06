"""Exact one-sided binomial bounds (numpy only; no scipy on the CPU test path).

``binom_cdf`` is computed in log space; ``cp_lower`` / ``cp_upper`` are one-sided Clopper-Pearson bounds found by
bisection on the exact binomial tail (64 halvings: 5e-20 on [0, 1], below float resolution).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


_LOGFACT = np.zeros(1)


def _logfact(n: int) -> np.ndarray:
    global _LOGFACT
    if _LOGFACT.size <= n:
        _LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, max(n, 2 * _LOGFACT.size) + 1)))])
    return _LOGFACT


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), exact (log space, vectorised over the k + 1 terms)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lf = _logfact(n)
    i = np.arange(k + 1)
    t = lf[n] - lf[i] - lf[n - i] + i * math.log(p) + (n - i) * math.log1p(-p)
    m = float(t.max())
    return float(min(1.0, math.exp(m) * float(np.exp(t - m).sum())))


def _bisect(f: Callable[[float], float], lo: float = 0.0, hi: float = 1.0, it: int = 64) -> float:
    """Root of a decreasing function f on [lo, hi]."""
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_lower(k: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) Clopper-Pearson lower bound on a Bernoulli rate (k successes in n)."""
    if n == 0 or k == 0:
        return 0.0
    # P(X >= k | p) = 1 - cdf(k-1) is increasing in p; find p with it = delta
    return _bisect(lambda p: delta - (1.0 - binom_cdf(k - 1, n, p)))


def cp_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) Clopper-Pearson upper bound."""
    if n == 0 or k >= n:
        return 1.0
    return _bisect(lambda p: binom_cdf(k, n, p) - delta)

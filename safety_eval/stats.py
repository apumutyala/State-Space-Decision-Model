"""Paired cluster-bootstrap CIs (numpy only).

Prior-lab lesson (ADVERSARIAL_REVIEW S5): case-only bootstraps gave CIs 2-2.5x too narrow; cluster by
request template x attack template family. ``cluster_bootstrap`` resamples each clustering dimension
independently (with replacement) and gives every unit the product of its clusters' draw multiplicities
(the "pigeonhole"/two-way product bootstrap, conservative for crossed designs). With one dimension it is the
ordinary cluster bootstrap.

Pairing: a statistic is computed from per-unit columns of the SAME rows under the same weights (e.g. a unit row
carries both its clean and attacked outcome), so clean-vs-attack differences are paired draw by draw. Draws
depend only on (cluster labels, n, seed); two statistics over the same rows share their draws.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

BOOT_SEED = 20261006
BOOT_N = 2000


def _codes(labels: Sequence) -> tuple[np.ndarray, int]:
    uniq = {v: i for i, v in enumerate(dict.fromkeys(labels))}
    return np.array([uniq[v] for v in labels], dtype=np.int64), len(uniq)


def cluster_weights(dims: list[Sequence], n: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    """[n, n_units] bootstrap weights; ``dims`` is a list of per-unit cluster label sequences."""
    rng = np.random.default_rng(seed)
    n_units = len(dims[0])
    W = np.ones((n, n_units))
    for lab in dims:
        codes, k = _codes(lab)
        draws = rng.integers(0, k, size=(n, k))
        mult = np.zeros((n, k))
        np.add.at(mult, (np.repeat(np.arange(n), k), draws.ravel()), 1.0)
        W *= mult[:, codes]
    return W


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    s = w.sum()
    return float((w * x).sum() / s) if s > 0 else float("nan")


def cluster_bootstrap(stat: Callable[[np.ndarray], float], dims: list[Sequence], n: int = BOOT_N,
                      seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """``stat(w)`` maps a per-unit weight vector to a scalar; the point estimate uses w = 1."""
    n_units = len(dims[0])
    if n_units == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_units": 0}
    point = float(stat(np.ones(n_units)))
    W = cluster_weights(dims, n, seed)
    draws = np.array([stat(w) for w in W], dtype=float)
    draws = draws[np.isfinite(draws)]
    a = (1 - level) / 2
    lo, hi = (np.percentile(draws, [100 * a, 100 * (1 - a)]) if draws.size else (float("nan"),) * 2)
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_units": int(n_units),
            "n_clusters": [int(_codes(d)[1]) for d in dims], "n_boot": int(n), "valid_draws": int(draws.size)}


def mean_ci(x: Sequence[float], dims: list[Sequence], **kw) -> dict:
    x = np.asarray(x, dtype=float)
    return cluster_bootstrap(lambda w: wmean(x, w), dims, **kw)


def ratio_ci(num: Sequence[float], den: Sequence[float], dims: list[Sequence], **kw) -> dict:
    """Weighted sum(num)/sum(den) (e.g. a conditional rate whose denominator is a 0/1 column)."""
    a, b = np.asarray(num, float), np.asarray(den, float)
    return cluster_bootstrap(lambda w: float((w * a).sum() / (w * b).sum()) if (w * b).sum() > 0 else float("nan"),
                             dims, **kw)


def ece(conf: np.ndarray, correct: np.ndarray, w: np.ndarray | None = None, bins: int = 15) -> float:
    """Weighted equal-width-bin ECE on [0, 1] (last bin closed)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    w = np.ones_like(conf) if w is None else np.asarray(w, float)
    tot = w.sum()
    if conf.size == 0 or tot <= 0:
        return float("nan")
    b = np.clip(np.floor(conf * bins).astype(int), 0, bins - 1)
    out = 0.0
    for k in range(bins):
        s = (b == k) & (w > 0)
        if s.any():
            ws = w[s].sum()
            out += ws * abs((w[s] * correct[s]).sum() / ws - (w[s] * conf[s]).sum() / ws)
    return float(out / tot)


def ece_ci(conf: Sequence[float], correct: Sequence[float], dims: list[Sequence], bins: int = 15, **kw) -> dict:
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    return cluster_bootstrap(lambda w: ece(c, y, w, bins), dims, **kw)

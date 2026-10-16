"""Experiment-local IPW cluster CRR; no changes to the production certifier.

The epoch horizon, probability floor, gate and cluster order must be frozen.
Only completed prefixes may enter the bettor; see README for the filtration.
"""
from __future__ import annotations

import math

import numpy as np

from jevkit.guarantees import CRR_BET_CAP


def normalized_ipw(acts, audited, losses, probabilities, *, epsilon: float, floor: float):
    """Return floor * (HT acted losses - epsilon * acts) / FULL cluster size.

    The last axis contains every decision in a cluster, including deferrals.
    Unqueried losses may be NaN; queried acted losses must be present in [0, 1].
    Probability metadata is required even on unaudited acted decisions. The
    caller must additionally guarantee positivity over the deployment support.
    """
    if not (0 < epsilon < 1 and 0 < floor <= 1):
        raise ValueError("epsilon in (0,1) and a positive declared probability floor required")
    acts, audited = np.asarray(acts, dtype=bool), np.asarray(audited, dtype=bool)
    losses = np.asarray(losses, dtype=float)
    if acts.shape != audited.shape or acts.shape != losses.shape or acts.ndim < 1 or not acts.shape[-1]:
        raise ValueError("acts, audited and losses must have the same nonempty full-cluster shape")
    probabilities = np.broadcast_to(np.asarray(probabilities, dtype=float), acts.shape)
    if np.any(acts & (~np.isfinite(probabilities) | (probabilities < floor) | (probabilities > 1))):
        raise ValueError("refuse: an acted decision lacks known positive inclusion probability at the floor")
    selected = acts & audited
    if np.any(selected & (~np.isfinite(losses) | (losses < 0) | (losses > 1))):
        raise ValueError("refuse: audited acted labels must be complete and bounded")
    weighted = np.zeros(acts.shape, dtype=float)
    np.divide(losses, probabilities, out=weighted, where=selected)
    return floor * (weighted.sum(axis=-1) - epsilon * acts.sum(axis=-1)) / acts.shape[-1]


def closure_times(selected, delays):
    """FIFO release time for each cluster; never skip a slow/missing label.

    Arrays are (world, cluster, decision). Cluster c is created at time c+1.
    Infinite delay means the cluster and all successors remain unclosed.
    """
    selected, delays = np.asarray(selected, dtype=bool), np.asarray(delays)
    if selected.ndim != 3 or delays.shape != selected.shape or not selected.shape[2]:
        raise ValueError("selected and delays require matching world/cluster/decision axes")
    if np.any(selected & (np.isnan(delays) | (delays < 0))):
        raise ValueError("label delays must be nonnegative (infinity means missing)")
    ready = np.arange(1, selected.shape[1] + 1) + np.where(selected, delays, 0).max(axis=2)
    return np.maximum.accumulate(ready, axis=1)


def betting_paths(y, *, epsilon: float, level: float, horizon: int, closure=None, deadline=None):
    """Vectorized independent worlds, scalar-time CRR identical to jevkit's bet.

    `horizon` is declared BEFORE data: never substitute a label-dependent
    completed count. `closure` must be the FIFO times of this fixed order.
    Later rows are simulated offline but cannot affect any earlier bet.
    Returns anytime p-values at full closure and at the wall-clock deadline.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not (0 < epsilon < 1 and 0 < level < 1):
        raise ValueError("need a world/cluster matrix and epsilon, level in (0,1)")
    worlds, n = y.shape
    if not isinstance(horizon, int) or horizon < max(1, n):
        raise ValueError("horizon must be a positive predeclared integer >= observed length")
    if np.any(~np.isfinite(y) | (y < -epsilon - 1e-12) | (y > 1 - epsilon + 1e-12)):
        raise ValueError("residual outside the declared normalized bounds")
    if closure is None:
        closure = np.broadcast_to(np.arange(1, n + 1), y.shape)
    closure = np.asarray(closure)
    if (closure.shape != y.shape or np.any(np.isnan(closure))
            or np.any(closure < np.arange(1, n + 1)) or np.any(np.diff(closure, axis=1) < 0)):
        raise ValueError("closure must be nondecreasing FIFO release times")
    deadline = horizon if deadline is None else deadline
    total, variance = np.zeros(worlds), np.full(worlds, 0.25)
    logk, best, best_due = np.zeros(worlds), np.zeros(worlds), np.zeros(worlds)
    first = np.full(worlds, n + 1, dtype=int)
    wall = np.full(worlds, np.inf)
    cap, c0, boundary = CRR_BET_CAP / (1 - epsilon), 2 * math.log(1 / level) / horizon, math.log(1 / level)
    for index in range(n):
        value = y[:, index]
        mean = total / index if index else np.zeros(worlds)
        bet = np.minimum(cap, np.sqrt(c0 * (index + 1) / variance))
        logk += np.log1p(-bet * value)
        best = np.maximum(best, logk)
        best_due = np.where(closure[:, index] <= deadline, np.maximum(best_due, logk), best_due)
        crossing = (first == n + 1) & (logk >= boundary)
        first[crossing], wall[crossing] = index + 1, closure[crossing, index]
        variance += (value - mean) ** 2
        total += value
    return {"p": np.exp(-best), "p_due": np.exp(-best_due), "first": first, "wall": wall,
            "closed": (closure <= deadline).sum(axis=1)}

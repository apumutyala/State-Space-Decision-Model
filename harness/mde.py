"""Minimum detectable effect (two-sided test on a paired difference).

MDE = (z_{1-alpha/2} + z_{power}) * SE. The SE comes from the paired cluster bootstrap on the
slice in hand and is rescaled to other slice sizes by sqrt(clusters_now / clusters_then), which
holds when the slice is grown with the same stratified design.
"""

import math
import numbers
from statistics import NormalDist


def _positive(name, x):
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {x!r}")
    return float(x)


def _prob(name, x):
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not (0 < x < 1):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {x!r}")
    return float(x)


def z(p):
    return NormalDist().inv_cdf(_prob("p", p))


def mde(se, alpha=0.05, power=0.8):
    se = _positive("se", se)
    return (z(1 - _prob("alpha", alpha) / 2) + z(_prob("power", power))) * se


def rescale_se(se, n_from, n_to):
    return _positive("se", se) * (_positive("n_from", n_from) / _positive("n_to", n_to)) ** 0.5


def required_n(se, n_from, effect, alpha=0.05, power=0.8):
    """Clusters needed so that `effect` is detectable, given SE `se` measured at `n_from`."""
    k = z(1 - _prob("alpha", alpha) / 2) + z(_prob("power", power))
    return _positive("n_from", n_from) * (k * _positive("se", se) / _positive("effect", effect)) ** 2


def paired_accuracy_se(discordant_rate, delta, n):
    """SE of a paired accuracy difference from the discordant-pair rate (McNemar-style)."""
    d = _prob("discordant_rate", discordant_rate)
    if not (math.isfinite(delta) and delta**2 <= d):
        raise ValueError("delta must be finite with delta^2 <= discordant_rate")
    return ((d - delta**2) / _positive("n", n)) ** 0.5


def table(se, n_clusters, sizes=(0.25, 0.5, 1, 2, 4), alpha=0.05, power=0.8):
    _positive("se", se)
    _positive("n_clusters", n_clusters)
    for f in sizes:
        _positive("size factor", f)
    rows = []
    for f in sizes:
        n = n_clusters * f
        s = rescale_se(se, n_clusters, n)
        rows.append({"scale": f, "clusters": int(round(n)), "se": s, "mde": mde(s, alpha, power)})
    return rows


def report(se, n_clusters, alpha=0.05, power=0.8):
    """MDE block for a paired comparison, defined for every bootstrap outcome.

    SE == 0 happens when candidate and baseline have identical skill on every replicate (the
    same predictions, or differences that never reach a scored benchmark). No MDE can be
    derived from that, so the block says so instead of raising. A negative or non-finite SE
    is still an error."""
    if isinstance(se, numbers.Real) and not isinstance(se, bool) and math.isfinite(se) and se == 0:
        return {
            "available": False,
            "se": 0.0,
            "clusters": int(n_clusters),
            "note": "paired SE is 0: candidate and baseline score identically on every bootstrap replicate, so the difference has no sampling spread and no MDE can be derived. Check that the two results files differ.",
        }
    return {"available": True, "alpha": alpha, "power": power, "table": table(se, n_clusters, alpha=alpha, power=power)}

"""Minimum detectable effect (two-sided test on a paired difference).

MDE = (z_{1-alpha/2} + z_{power}) * SE. The SE comes from the paired cluster bootstrap on the
slice in hand and is rescaled to other slice sizes by sqrt(clusters_now / clusters_then), which
holds when the slice is grown with the same stratified design.
"""

from statistics import NormalDist


def z(p):
    return NormalDist().inv_cdf(p)


def mde(se, alpha=0.05, power=0.8):
    return (z(1 - alpha / 2) + z(power)) * se


def rescale_se(se, n_from, n_to):
    return se * (n_from / n_to) ** 0.5


def required_n(se, n_from, effect, alpha=0.05, power=0.8):
    """Clusters needed so that `effect` is detectable, given SE `se` measured at `n_from`."""
    k = z(1 - alpha / 2) + z(power)
    return n_from * (k * se / effect) ** 2


def paired_accuracy_se(discordant_rate, delta, n):
    """SE of a paired accuracy difference from the discordant-pair rate (McNemar-style)."""
    return ((discordant_rate - delta**2) / n) ** 0.5


def table(se, n_clusters, sizes=(0.25, 0.5, 1, 2, 4), alpha=0.05, power=0.8):
    rows = []
    for f in sizes:
        n = n_clusters * f
        s = rescale_se(se, n_clusters, n)
        rows.append({"scale": f, "clusters": int(round(n)), "se": s, "mde": mde(s, alpha, power)})
    return rows

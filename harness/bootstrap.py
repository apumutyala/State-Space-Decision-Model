"""Paired cluster bootstrap through the complete 0.2.1 index formula.

Clusters are linked case groups (record families) and are resampled with replacement inside
each benchmark (benchmarks and their weights are fixed, as in the board's declared
`uncertainty` method: "paired hierarchical cluster bootstrap through complete index formula",
2000 replicates, 95%). Candidate and baseline share every resample, so the interval is for the
paired difference.
"""

import numpy as np

from harness import score as S
from harness.suite import area_of, spec


def _weights(rng, n_clusters, B):
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B).astype(np.float64)


def replicate_skills(benches_list, B=2000, seed=0):
    """benches_list: list of {n: Bench} (one per system) over the same rows.

    Returns, per system, {n: (raw[B+1], skill[B+1])}; column 0 is the point estimate."""
    if not benches_list or not benches_list[0]:
        raise ValueError("no systems / benchmarks to resample")
    keys = [set(b) for b in benches_list]
    if any(k != keys[0] for k in keys[1:]):
        diff = sorted(set().union(*keys) - set.intersection(*keys))
        raise ValueError(f"systems must cover identical benchmark sets before pairing; mismatched: {diff}")
    if not (isinstance(B, int) and B >= 1):
        raise ValueError("B must be a positive integer")
    rng = np.random.default_rng(seed)
    ids = sorted(benches_list[0])
    out = [dict() for _ in benches_list]
    for n in ids:
        clusters = benches_list[0][n].clusters
        for bl in benches_list:
            if bl[n].clusters != clusters:
                raise ValueError(f"benchmark {n}: systems disagree on cluster order")
        W = np.vstack([np.ones((1, len(clusters))), _weights(rng, len(clusters), B)])
        for k, bl in enumerate(benches_list):
            raw, skill = bl[n].value(W @ bl[n].S)
            out[k][n] = (raw, skill)
    return out


def _ci(x, level=0.95):
    lo, hi = np.quantile(x[1:], [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"point": float(x[0]), "lo": float(lo), "hi": float(hi), "se": float(np.std(x[1:], ddof=1))}


def area_skills(skills, renormalize=True):
    s = spec()
    w = S.weights(set(skills), renormalize)
    amap = area_of()
    out = {}
    for a in s["areas"]:
        ids = [n for n in a["benchmarks"] if n in w]
        tot = sum(w[n] for n in ids)
        if tot:
            out[a["id"]] = 100 * sum(w[n] * np.asarray(skills.get(n, 0.0)) for n in ids) / tot
    return out


def summarize_one(rep, level=0.95, renormalize=True):
    skills = {n: v[1] for n, v in rep.items()}
    idx = S.index_from_skills(skills, renormalize)
    return {
        "index": _ci(idx, level),
        "areas": {a: _ci(v, level) for a, v in area_skills(skills, renormalize).items()},
        "benchmarks": {str(n): _ci(100 * v[1], level) for n, v in sorted(rep.items())},
    }


def compare(candidate_benches, baseline_benches, B=2000, seed=0, level=0.95, renormalize=True):
    """Paired CI for candidate - baseline on the index, each area and each benchmark (0-100 scale)."""
    cand, base = replicate_skills([candidate_benches, baseline_benches], B=B, seed=seed)
    cs = {n: v[1] for n, v in cand.items()}
    bs = {n: v[1] for n, v in base.items()}
    diff_idx = S.index_from_skills(cs, renormalize) - S.index_from_skills(bs, renormalize)
    ca, ba = area_skills(cs, renormalize), area_skills(bs, renormalize)
    out = {
        "B": B,
        "seed": seed,
        "level": level,
        "candidate": summarize_one(cand, level, renormalize),
        "baseline": summarize_one(base, level, renormalize),
        "diff": {
            "index": _ci(diff_idx, level),
            "areas": {a: _ci(ca[a] - ba[a], level) for a in ca},
            "benchmarks": {str(n): _ci(100 * (cs[n] - bs[n]), level) for n in sorted(cs)},
        },
    }
    d = out["diff"]["index"]
    out["diff"]["index"]["p_le_zero"] = float(np.mean(diff_idx[1:] <= 0))
    out["diff"]["index"]["excludes_zero"] = d["lo"] > 0 or d["hi"] < 0
    return out

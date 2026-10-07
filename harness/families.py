"""Held-out-family split: in-domain vs new-domain-proxy scores (E1).

Every public benchmark is tagged (manifests/families-v1.json) by how much our training mix covers it:
`same_source`, `same_family` or `none`. `none` is the NEW-DOMAIN PROXY, our local public stand-in for the
0.3 Full score's private "decision tasks from new domains" (30%). `in_domain` = same_source + same_family.

A tier score is the edition's index restricted to the tier's benchmarks: board area weights x gold weights,
renormalised over the benchmarks present (inside each area, then over the areas present), so it is on the
same 0-100 skill scale as the index. Because it is computed from the same bootstrap replicates as the index,
paired CIs per tier come for free (`compare`).
"""

import functools
import json

import numpy as np

from harness import config
from harness import score as S

FAMILIES = config.MANIFEST_DIR / "families-v1.json"
TIERS = ("same_source", "same_family", "none")
GROUPS = {"same_source": ("same_source",), "same_family": ("same_family",), "in_domain": ("same_source", "same_family"), "new_domain_proxy": ("none",)}


@functools.lru_cache(maxsize=None)
def load(path=None):
    d = json.loads((path or FAMILIES).read_text(encoding="utf-8"))
    for n, b in d["benchmarks"].items():
        if b["tier"] not in TIERS:
            raise ValueError(f"benchmark {n}: unknown tier {b['tier']!r}")
        if b["tier"] != "none" and not set(b["via"].split(",")) <= set(d["training_sources"]):
            raise ValueError(f"benchmark {n}: 'via' names an undeclared training source")
    return d


def tier_of(n):
    return load()["benchmarks"][str(n)]["tier"]


def members(group, edition=None):
    """Benchmark ids of a group ('same_source', 'same_family', 'in_domain', 'new_domain_proxy') in the edition."""
    from harness.suite import index_ids

    tiers = GROUPS[group]
    return [n for n in index_ids(edition) if tier_of(n) in tiers]


def group_index(skills, ids, edition=None):
    """skills {n: array(B)} -> array(B): index restricted to `ids` (renormalised; NaN if none present)."""
    present = {n for n in skills if n in set(ids)}
    if not present:
        return np.full(np.shape(next(iter(skills.values()))), np.nan)
    w = S.weights(present, renormalize=True, edition=edition)
    tot = sum(w.values())
    return 100 * sum(w[n] * np.asarray(skills[n]) for n in present) / tot


def _ci(x, level=0.95):
    lo, hi = np.quantile(x[1:], [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"point": float(x[0]), "lo": float(lo), "hi": float(hi), "se": float(np.std(x[1:], ddof=1))}


def summarize(rep, edition=None, level=0.95):
    """rep: one system's `bootstrap.replicate_skills` output -> {group: CI} plus the gap in-domain minus proxy."""
    skills = {n: v[1] for n, v in rep.items()}
    out = {"families": load()["id"], "edition": config.edition_id(edition)}
    vals = {}
    for g in GROUPS:
        ids = members(g, edition)
        vals[g] = group_index(skills, ids, edition)
        out[g] = dict(_ci(vals[g], level), benchmarks=[n for n in ids if n in skills])
    out["in_minus_proxy"] = _ci(vals["in_domain"] - vals["new_domain_proxy"], level)
    return out


def compare(cand_rep, base_rep, edition=None, level=0.95):
    """Paired per-group CI for candidate - baseline (both from one `replicate_skills` call)."""
    cs = {n: v[1] for n, v in cand_rep.items()}
    bs = {n: v[1] for n, v in base_rep.items()}
    out = {"families": load()["id"], "edition": config.edition_id(edition)}
    for g in GROUPS:
        ids = members(g, edition)
        out[g] = _ci(group_index(cs, ids, edition) - group_index(bs, ids, edition), level)
    return out

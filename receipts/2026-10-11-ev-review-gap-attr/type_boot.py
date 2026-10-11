import json, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, sys.argv[2])
from fam_boot_lib import *  # noqa
full = np.arange(len(items)); typ = b.typ
# per-item max weight in I points for a full miss
for ty in spec.TYPES:
    ti = scoring.TYPE_IX[ty]
    for tr in spec.TIERS:
        m = (typ == ti) & (b.tier == scoring.TIER_IX[tr])
        if m.any():
            n = m.sum(); w = spec.TIER_WEIGHTS[tr] / W[ty] / 3
            unit = {"choice": 100 / (1 - b.chance[m].mean()), "noul": 200.0, "score": 100 / b.nmae_chance[m].mean()}[ty]
            print("weight", ty, tr, n, "I pts per unit miss", round(w * unit / n, 3))
res = {}
for cn in ["stock4b_shim_zbtbj", "D19", "jev1_native"]:
    c = T[cn]; rng = np.random.default_rng(16); st = scoring.strata(b)
    d0 = contrib(c, full); pt0 = {ty: -d0[typ == scoring.TYPE_IX[ty]].sum() for ty in spec.TYPES}
    D = {ty: [] for ty in spec.TYPES}; big = 0
    for k in range(1000):
        idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
        d = contrib(c, idx); tt = typ[idx]
        p = {ty: -d[tt == scoring.TYPE_IX[ty]].sum() for ty in spec.TYPES}
        for ty in p: D[ty].append(p[ty])
        big += max(p, key=p.get) == "noul"
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
    res[cn] = {"per_type_pts": {ty: round(pt0[ty], 3) for ty in pt0}, "ci95": {ty: ci(D[ty]) for ty in D}, "p_noul_largest": big / 1000}
    print(cn, res[cn])
json.dump(res, open(sys.argv[1], "w"), indent=1)

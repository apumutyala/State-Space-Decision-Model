"""Selection optimism: pick the top-6 families on a random stratified half A, measure their share of the I gap on half B."""
import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, sys.argv[1])
from fam_boot_lib import T, b, contrib, fam, FAMS, SIX, scoring
rng = np.random.default_rng(20261011); st = scoring.strata(b); out = {}
for cn in ["stock4b_shim_zbtbj", "D19", "jev1_native"]:
    c = T[cn]; inA, onB, sixB, overlap = [], [], [], []
    for _ in range(500):
        A_, B_ = [], []
        for s_ in st:
            p = rng.permutation(s_); h = len(p) // 2
            A_ += list(p[:h]); B_ += list(p[h:])
        A_, B_ = np.array(A_), np.array(B_)
        dA, dB = contrib(c, A_), contrib(c, B_)
        pA = {f: -dA[fam[A_] == f].sum() for f in FAMS}; pB = {f: -dB[fam[B_] == f].sum() for f in FAMS}
        top = sorted(FAMS, key=lambda f: -pA[f])[:6]
        tA, tB = sum(pA.values()), sum(pB.values())
        inA.append(sum(pA[f] for f in top) / tA); onB.append(sum(pB[f] for f in top) / tB)
        overlap.append(len(set(top) & set(sorted(FAMS, key=lambda f: -pB[f])[:6])))
    q = lambda v: [round(float(np.percentile(v, 50)), 3), round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    out[cn] = {"top6_share_on_selection_half(median,lo,hi)": q(inA), "same_top6_share_on_other_half": q(onB),
               "top6_overlap_A_vs_B_mean": float(np.mean(overlap))}
    print(cn, out[cn])
json.dump(out, open(sys.argv[2], "w"), indent=1)

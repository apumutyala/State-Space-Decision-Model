"""Paired stratified bootstrap of per-family shares of the I gap (reviewer, claude/ev-review-gap-attr).
Exact linear per-item decomposition of I (valid while no type CC is clipped at 0; checked per resample vs jevkit.aggregate)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from jevkit import scoring, spec
from jevkit.items import load_public
sys.path.insert(0, 'scripts')
from gap_attribution import load_preds, sha

cfg = json.load(open(sys.argv[1], encoding='utf-8')); B = int(sys.argv[3]); out = sys.argv[2]
items = [it for it in load_public() if it.expected is not None]
fam = np.array([it.family for it in items]); FAMS = sorted(set(fam))
SIX = ["probability", "temporal_numeric", "long_policy", "trap", "adequacy", "judge_hard"]
T = {}
for n in ["h2o_zbtbj", "stock4b_shim_zbtbj", "D19", "jev1_native"]:
    s = cfg["systems"][n]; assert sha(s["preds"]) == s["sha256"]
    T[n] = scoring.build_table(items, load_preds(s["preds"]))
b = T["h2o_zbtbj"]
assert (b.side == 0).all()
W = {}  # per type, sum of tier weights present
for ty in spec.TYPES:
    W[ty] = sum(spec.TIER_WEIGHTS[tr] for tr in spec.TIERS if ((b.typ == scoring.TYPE_IX[ty]) & (b.tier == scoring.TIER_IX[tr])).any())

def contrib(c, idx):
    """per-occurrence contribution to I(c) - I(b) on resample idx (positions)."""
    d = np.zeros(len(idx))
    typ, tier = b.typ[idx], b.tier[idx]
    for ty in spec.TYPES:
        ti = scoring.TYPE_IX[ty]
        for tr in spec.TIERS:
            m = (typ == ti) & (tier == scoring.TIER_IX[tr])
            if not m.any():
                continue
            j = idx[m]; n = m.sum(); w = spec.TIER_WEIGHTS[tr] / W[ty] / 3.0
            if ty == "choice":
                cbar = b.chance[j].mean()
                d[m] = w * 100.0 * (c.correct[j].astype(float) - b.correct[j]) / (n * (1 - cbar))
            elif ty == "noul":
                d[m] = w * 200.0 * (c.correct[j].astype(float) - b.correct[j]) / n
            else:
                d[m] = -w * 100.0 * (c.nmae[j] - b.nmae[j]) / (n * b.nmae_chance[j].mean())
    return d

full = np.arange(len(items))
res = {"B": B, "seed": 16, "method": "jevkit strata, default_rng(16), paired; family gap points = sum of exact per-item I contributions", "cands": {}}
ga = json.load(open('receipts/2026-10-11-gap-attribution/gap_attribution.json'))["pairs"]
for cn in ["stock4b_shim_zbtbj", "D19", "jev1_native"]:
    c = T[cn]
    d0 = contrib(c, full)
    tot0 = -d0.sum()
    Iagg = scoring.aggregate(c, full, detail=False)["intelligence"]["I"] - scoring.aggregate(b, full, detail=False)["intelligence"]["I"]
    fam0 = {f: -d0[fam == f].sum() for f in FAMS}
    gaf = ga[f"{cn} - h2o_zbtbj"]["per_family"]
    maxdiff = max(abs(fam0[f] - gaf[f]["swap_dI"]) for f in FAMS)
    rng = np.random.default_rng(16); st = scoring.strata(b)
    P = {f: [] for f in FAMS}; SH = {f: [] for f in FAMS}; six, top6own, intop6 = [], [], {f: 0 for f in FAMS}; clipfail = 0; chk = 0.0; TOT = []
    for k in range(B):
        idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
        d = contrib(c, idx); tot = -d.sum()
        if k < 50 or k % 50 == 0:  # cross-check vs the frozen aggregate
            ra = scoring.aggregate(c, idx, detail=False)["intelligence"]["I"] - scoring.aggregate(b, idx, detail=False)["intelligence"]["I"]
            if abs(ra + tot) > 1e-9:
                clipfail += 1
            chk = max(chk, abs(ra + tot))
        fi = fam[idx]
        pts = {f: -d[fi == f].sum() for f in FAMS}
        TOT.append(tot)
        for f in FAMS:
            P[f].append(pts[f]); SH[f].append(pts[f] / tot)
        six.append(sum(pts[f] for f in SIX) / tot)
        order = sorted(FAMS, key=lambda f: -pts[f])[:6]
        top6own.append(sum(pts[f] for f in order) / tot)
        for f in order:
            intop6[f] += 1
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    own6 = sorted(FAMS, key=lambda f: -fam0[f])[:6]
    res["cands"][cn] = {
        "I_gap_point": tot0, "I_gap_aggregate": -Iagg, "I_gap_ci95": ci(TOT),
        "max_abs_diff_vs_GA_swap_dI": maxdiff, "max_abs_diff_vs_aggregate_on_checked_resamples": chk, "checked_resamples_mismatch": clipfail,
        "six_share_point": sum(fam0[f] for f in SIX) / tot0, "six_share_ci95": ci(six),
        "own_top6_point": own6, "own_top6_share_point": sum(fam0[f] for f in own6) / tot0, "own_top6_share_ci95(reselected per resample)": ci(top6own),
        "families": {f: {"n": int((fam == f).sum()), "pts": round(fam0[f], 3), "pts_ci95": ci(P[f]), "share": round(fam0[f] / tot0, 3),
                          "share_ci95": ci(SH[f]), "p_in_top6": intop6[f] / B} for f in sorted(FAMS, key=lambda f: -fam0[f])}}
json.dump(res, open(out, 'w'), indent=1)
for cn, r in res["cands"].items():
    print(cn, "gap", round(r["I_gap_point"], 3), "agg", round(r["I_gap_aggregate"], 3), "diffGA", r["max_abs_diff_vs_GA_swap_dI"], "chk", r["max_abs_diff_vs_aggregate_on_checked_resamples"], r["checked_resamples_mismatch"])
    print("  six share", round(r["six_share_point"], 3), r["six_share_ci95"], "own top6", r["own_top6_point"], round(r["own_top6_share_point"], 3), r["own_top6_share_ci95(reselected per resample)"])
    for f, v in r["families"].items():
        print("   %-17s n=%2d pts %6.2f %s share %5.2f %s pTop6 %.2f" % (f, v["n"], v["pts"], v["pts_ci95"], v["share"], v["share_ci95"], v["p_in_top6"]))

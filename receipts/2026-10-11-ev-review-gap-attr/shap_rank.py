import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts'); sys.path.insert(0, sys.argv[1])
from fam_boot_lib import T, b, cfg, scoring
from gap_attribution import shapley, comp
out = {}
for cn in ["stock4b_shim_zbtbj", "D19", "jev1_native"]:
    c = T[cn]; sb, sc = cfg["systems"]["h2o_zbtbj"], cfg["systems"][cn]
    rng = np.random.default_rng(16); st = scoring.strata(b); win = 0; recip_share = []; first = []; Ishare = []; ic_close = 0
    for _ in range(1000):
        idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
        rb = scoring.aggregate(b, idx, detail=False); rc = scoring.aggregate(c, idx, detail=False)
        bx = {"I": rb["intelligence"]["I"], "C": rb["calibration"]["score"], "S": sb["S"], "K": sb["K"]}
        cx = {"I": rc["intelligence"]["I"], "C": rc["calibration"]["score"], "S": sc["S"], "K": sc["K"]}
        sh = shapley(cx, bx); gap = comp(bx) - comp(cx)
        win += max(sh, key=sh.get) == "I"  # Shapley terms = Composite recovered by moving the axis to H2O
        Ishare.append(sh["I"] / gap if gap else np.nan)
        ic_close += comp({**cx, "I": bx["I"], "C": bx["C"]}) >= comp(bx)
    out[cn] = {"P_I_largest_axis": win / 1000, "I_share_ci95": [float(np.nanpercentile(Ishare, 2.5)), float(np.nanpercentile(Ishare, 97.5))],
               "P_IC_at_H2O_closes_gap": ic_close / 1000}
    print(cn, out[cn])
json.dump(out, open(sys.argv[2], "w"), indent=1)

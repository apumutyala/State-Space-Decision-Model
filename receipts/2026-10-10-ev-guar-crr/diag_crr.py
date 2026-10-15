"""POST-HOC diagnostic for GUARANTEES_CRR.md (not verdict-bearing): why the CRR test certifies nothing at c = 8.
Same argv as guarantees_crr.py except the last (prints JSON). run_id configuration, alpha = 0.10, marginal analysis,
cluster split. Per lambda and cap: LTT acts, empirical R_c (weighted acted error), the betting log max capital,
the hindsight-optimal constant-bet log capital max_b sum ln(1 - b Y) (an upper bound for any constant bet), and the
needed ln(1/level)."""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:5] + ["unused"]
import guarantees_crr as C  # noqa: E402

G, GF = C.G, C.GF
ids, gold, cat, case = GF.load_gold()
is_cal = GF.split(ids, case, "run_id")
size = collections.Counter(C.rid(d) for d in gold)
level = C.DELTA / len(G.DEFAULT_LAMBDAS)
need = math.log(1 / level)


def best_const(y, eps):
    hi = 1.0 / (1.0 - eps) - 1e-9
    f = lambda b: sum(math.log1p(-b * v) for v in y)  # noqa: E731  concave in b
    lo, h = 0.0, hi
    for _ in range(60):
        m1, m2 = lo + (h - lo) / 3, h - (h - lo) / 3
        if f(m1) < f(m2):
            lo = m1
        else:
            h = m2
    return f((lo + h) / 2)


out = {"level": level, "needed_log_capital": need}
for name in ("D19", "D13", "ClefFlash", "Clef27B"):
    sc, _ = GF.load_model(GF.MODELS[name][0], gold)
    cal = [sc[d] for d in sorted(sc) if is_cal[C.rid(d)]]
    key = lambda r: C.rid(r.id)  # noqa: E731
    fit_h, ltt_h, _ = G.split_calibration(cal, 0, 0.5, cluster_of=key)
    fit = G.fit_conformal(fit_h, 0.10, True)
    by = collections.defaultdict(list)
    for r in ltt_h:
        by[key(r)].append(r)
    keys = sorted(by, key=lambda c: (G._crr_order(0, c), c))
    thr = {c: [(r.probs[s[0]], s[0] != r.label) for r in by[c]
               for s in [G.prediction_set(r.probs, G._qhat_for(fit, r))] if len(s) == 1] for c in keys}
    rows = {}
    for lam in (0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.9, 0.86):
        row = {}
        for cap in (1, 8, 64):
            y, wn, wl, na, ne, ncl = [], 0.0, 0.0, 0, 0, 0
            for c in keys:
                a = sum(p >= lam for p, _ in thr[c])
                e = sum(p >= lam and wr for p, wr in thr[c])
                w = min(1.0, cap / size[c])
                y.append(w * (e - C.EPS * a) / cap)
                wn, wl, na, ne, ncl = wn + w * a, wl + w * e, na + a, ne + e, ncl + (a > 0)
            if na == 0:
                row[str(cap)] = None
                continue
            p, logk = G.crr_betting_pvalue(y, C.EPS, level)
            row[str(cap)] = {"acted_clusters": ncl, "acted": na, "errors": ne, "R_c_ltt": round(wl / wn, 4),
                             "log_max_capital": round(logk, 2), "hindsight_const_bet": round(best_const(y, C.EPS), 2),
                             "certified": p <= level}
        rows[str(lam)] = row
    out[name] = rows
    del sc, cal
json.dump(out, sys.stdout, indent=1)

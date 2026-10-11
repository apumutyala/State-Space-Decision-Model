"""Diagnostic for GUARANTEES_CLUSTER.md (why D19 certifies nothing under the one-decision-per-row test).
Same args as guarantees_cluster.py; run_id configuration, alpha = 0.10, cluster split. Prints JSON to stdout."""
import collections
import json
import sys

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guarantees_cluster as C  # noqa: E402  (argv layout identical)

G, GF = C.G, C.GF
ids, gold, cat, case = GF.load_gold()
amap = C.area_of("0.2.1")
is_cal = GF.split(ids, case, "run_id")
area = {r: amap.get(cat[r], "non-index") for r in ids}
test_ids = sorted(d for d in gold if not is_cal[C.rid(d)])
rows_n = collections.Counter(C.rid(d) for d in test_ids)
out = {}
for name in ("D19", "D13", "ClefFlash", "Clef27B"):
    sc, _ = GF.load_model(GF.MODELS[name][0], gold)
    cal = [sc[d] for d in sorted(sc) if is_cal[C.rid(d)]]
    test = [sc[d] for d in test_ids if d in sc]
    key = lambda r: C.rid(r.id)  # noqa: E731
    fit_h, ltt_h, _ = G.split_calibration(cal, 0, 0.5, cluster_of=key)
    fit = G.fit_conformal(fit_h, 0.10, True)
    picks = G.cluster_pick(ltt_h, key)
    level = C.DELTA / len(G.DEFAULT_LAMBDAS)
    grid = {}
    for lam in (1.0, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.86, 0.8, 0.5, 0.0):
        row = {}
        for tag, data in (("picked", picks), ("all_ltt", ltt_h)):
            acted = [(r, y) for r in data for ok, y in [G._acts(r, fit, lam)] if ok]
            n, e = len(acted), sum(y != r.label for r, y in acted)
            row[tag] = {"n_act": n, "err": e, "rate": round(e / n, 4) if n else None,
                        "p": G.binom_cdf(e, n, C.EPS) if n else None}
        row["picked"]["certified"] = row["picked"]["p"] is not None and row["picked"]["p"] <= level
        grid[str(lam)] = row
    # TEST: acted error by row size at lambda 0.86 (D19's published lambda) and 0.99
    by = {}
    for lam in (0.99, 0.86):
        b = collections.defaultdict(lambda: [0, 0, 0])
        for r in test:
            m = rows_n[C.rid(r.id)]
            k = "1" if m == 1 else ("2-3" if m <= 3 else ("4-7" if m <= 7 else "8+"))
            ok, y = G._acts(r, fit, lam)
            b[k][0] += 1
            b[k][1] += ok
            b[k][2] += ok and y != r.label
        by[str(lam)] = {k: {"n": v[0], "act": round(v[1] / v[0], 3), "acted_err": round(v[2] / v[1], 4) if v[1] else None}
                        for k, v in sorted(b.items())}
    sizes = collections.Counter("1" if rows_n[C.rid(r.id)] == 1 else "2-3" if rows_n[C.rid(r.id)] <= 3 else
                                "4-7" if rows_n[C.rid(r.id)] <= 7 else "8+" for r in test)
    single_areas = collections.Counter(area[C.rid(r.id)] for r in test if rows_n[C.rid(r.id)] == 1)
    out[name] = {"level": level, "ltt_grid": grid, "test_by_row_size": by, "test_decisions_by_row_size": sizes,
                 "single_decision_rows_by_area": single_areas}
    del sc, cal, test
json.dump(out, sys.stdout, indent=1)

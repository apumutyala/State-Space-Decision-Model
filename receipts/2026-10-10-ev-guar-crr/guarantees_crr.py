"""B-PREREG-GUAR-CRR (state 033906aa): cluster-robust ratio LTT (betting test on per-cluster sums, cap c) for the
marginal and per-area actionability numbers on the full-DEV predictions, next to the published per-decision test and
the one-decision-per-cluster test of B-PREREG-GUAR-CLUSTER.

    python -X utf8 -I guarantees_crr.py <checkout with this branch's jevkit + harness> \
        <dir of receipts/2026-10-10-B-guarantees-fulldev> <dir of receipts/2026-10-10-ev-guar-area> \
        <GUARANTEES_CLUSTER.json> <out.json>

Variants (marginal = per-type Mondrian + pooled LTT; area = area Mondrian + Bonferroni over live groups x grid):
  published  per-decision FIT/LTT split, per-decision binomial (must reproduce the parent receipts exactly)
  cluster    cluster split, one decision per cluster binomial (must reproduce GUARANTEES_CLUSTER.json exactly)
  crr8       cluster split, CRR betting test, cap 8 (PRIMARY)
  crr1       cluster split, CRR, cap 1 (secondary)
  crr_max    cluster split, CRR, cap = largest DEV cluster (uncapped = decision-weighted risk; secondary)
"""
import collections
import json
import math
import os
import sys
import time

import numpy as np

FULLDEV_DIR, AREA_DIR, CLUSTER_JSON = sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, FULLDEV_DIR)
import guarantees_fulldev as GF  # noqa: E402  (inserts argv[1] on sys.path: this branch's harness + jevkit)
from harness.suite import area_of  # noqa: E402
from jevkit import guarantees as G  # noqa: E402

MODELS = GF.MODELS
ALPHAS = (0.10, 0.05)
EPS, DELTA = 0.05, 0.10
B, SEED = 2000, 0
MIN_N = 100
SMALL = "pooled-small"
PRIMARY_CAP = 8
VARIANTS = ("published", "cluster", "crr8", "crr1", "crr_max")
ANALYSES = ("marginal", "area")
PAIRS = [("D19", "ClefFlash"), ("D19", "Clef27B"), ("D13", "ClefFlash"), ("D13", "Clef27B"), ("D19", "D13"),
         ("Clef27B", "ClefFlash")]


def rid(d):
    return d.split("#", 1)[0]


def group_plan(gold, is_cal, area_of_rid, ckey, clustered):
    """Identical to guarantees_cluster.group_plan (GUAR-AREA's MIN_N rule on gold decisions; answer-free)."""
    cal = [G.Scored(d, "x", {}) for d in sorted(gold) if is_cal[rid(d)]]
    if clustered:
        fit, ltt, _ = G.split_calibration(cal, 0, 0.5, cluster_of=lambda r: ckey(rid(r.id)))
        nl = collections.Counter(area_of_rid[rid(r.id)] for r in
                                 G.cluster_pick(ltt, lambda r: (ckey(rid(r.id)), area_of_rid[rid(r.id)])))
    else:
        fit, ltt, _ = G.split_calibration(cal, 0, 0.5)
        nl = collections.Counter(area_of_rid[rid(r.id)] for r in ltt)
    nf = collections.Counter(area_of_rid[rid(r.id)] for r in fit)
    areas = sorted(set(nf) | set(nl))
    small = [a for a in areas if nf[a] < MIN_N or nl[a] < MIN_N]
    gmap = {a: (SMALL if a in small else a) for a in areas}
    gf, gl = collections.Counter(), collections.Counter()
    for a in areas:
        gf[gmap[a]] += nf[a]
        gl[gmap[a]] += nl[a]
    dead = [g for g in set(gmap.values()) if gf[g] < MIN_N or gl[g] < MIN_N]
    return {"n_fit_area": dict(nf), "n_ltt_area": dict(nl), "pooled_into_small": small, "group_of_area": gmap,
            "no_guarantee_groups": dead, "min_n": MIN_N}


def ltt_brief(rep):
    keep = ("lambda", "n_cal", "n_clusters", "n_decisions", "act_rate_cal", "accepted_error_cal",
            "accepted_error_cal_weighted", "accepted_error_ucb", "p_value", "log_capital", "n_valid_lambdas",
            "test_level", "crr_cap", "n_clusters_above_cap", "max_cluster_size", "ltt_ids_sha256",
            "picked_ids_sha256", "fit_ids_sha256")
    return {k: rep[k] for k in keep if k in rep}


def run_split(mode, ids, gold, cat, case, amap):
    is_cal = GF.split(ids, case, mode)
    ckey = (lambda r: r) if mode == "run_id" else (lambda r: case[r])
    area_of_rid = {r: amap.get(cat[r], "non-index") for r in ids}
    plans = {"published": group_plan(gold, is_cal, area_of_rid, ckey, False)}
    plans["cluster"] = group_plan(gold, is_cal, area_of_rid, ckey, True)
    for v in ("crr8", "crr1", "crr_max"):
        plans[v] = plans["cluster"]
    gmap_c = plans["cluster"]["group_of_area"]
    # design sizes (DEV gold only; answer- and model-independent): per cluster, and per (cluster, area group)
    size_m = collections.Counter(ckey(rid(d)) for d in gold)
    size_a = collections.Counter((ckey(rid(d)), gmap_c.get(area_of_rid[rid(d)])) for d in gold)
    s_max = max(size_m.values())
    caps = {"crr8": PRIMARY_CAP, "crr1": 1, "crr_max": s_max}
    test_ids = sorted(d for d in gold if not is_cal[rid(d)])
    res, per_test = {}, {}
    for name in MODELS:
        t0 = time.time()
        sc, bad = GF.load_model(MODELS[name][0], gold)
        cal_t = [sc[d] for d in sorted(sc) if is_cal[rid(d)]]
        test_t = [sc[d] for d in test_ids if d in sc]
        m = {"invalid_or_missing": bad, "n_cal": len(cal_t), "n_test": len(test_t), "alpha": {}}
        arrays = {}
        for alpha in ALPHAS:
            arr, rep_a = {}, {}
            for v in VARIANTS:
                clustered = v != "published"
                cl = (lambda r: ckey(rid(r.id))) if clustered else None
                kw = {}
                if v == "cluster":
                    kw = {"cluster_of": cl}
                elif v.startswith("crr"):
                    kw = {"cluster_of": cl, "cluster_test": "crr", "crr_cap": caps[v]}
                fit_h, ltt_h, rec = G.split_calibration(cal_t, 0, 0.5, cluster_of=cl)
                fit_t = G.fit_conformal(fit_h, alpha, True)
                kw_m = dict(kw, cluster_size=size_m.__getitem__) if v.startswith("crr") else kw
                ltt_t = G.ltt_select(ltt_h, fit_t, EPS, DELTA, method="bonferroni", **kw_m)
                plan = plans[v]
                gmap = plan["group_of_area"]
                groups = sorted(set(gmap.values()))
                live = [g for g in groups if g not in plan["no_guarantee_groups"]]
                K = len(live)
                gof = lambda r: gmap[area_of_rid[rid(r.id)]]  # noqa: E731
                cal_a = [G.Scored(r.id, gof(r), r.probs, r.label) for r in cal_t]
                fit_ah, ltt_ah, rec_a = G.split_calibration(cal_a, 0, 0.5, cluster_of=cl)
                assert rec_a["fit_ids_sha256"] == rec["fit_ids_sha256"]
                fit_a = G.fit_conformal(fit_ah, alpha, True)
                for g in plan["no_guarantee_groups"]:
                    fit_a["qhat"][g] = math.inf
                by_g = collections.defaultdict(list)
                for r in ltt_ah:
                    by_g[r.type].append(r)
                lam_a, ltt_rep = {}, {}
                for g in groups:
                    if g not in live:
                        lam_a[g], ltt_rep[g] = None, {"lambda": None, "note": "insufficient: abstain"}
                        continue
                    kw_g = (dict(kw, cluster_size=lambda c, g=g: size_a[(c, g)]) if v.startswith("crr") else kw)
                    rep = G.ltt_select(by_g[g], fit_a, EPS, DELTA / K, method="bonferroni", **kw_g)
                    lam_a[g], ltt_rep[g] = rep["lambda"], ltt_brief(rep)
                cols = collections.defaultdict(list)
                for r in test_t:
                    ra = G.Scored(r.id, gof(r), r.probs, r.label)
                    ok, y = G._acts(r, fit_t, ltt_t["lambda"]) if ltt_t["lambda"] is not None else (False, None)
                    cols["act_marginal"].append(ok)
                    cols["err_marginal"].append(ok and y != r.label)
                    lam = lam_a[ra.type]
                    ok, y = G._acts(ra, fit_a, lam) if lam is not None else (False, None)
                    cols["act_area"].append(ok)
                    cols["err_area"].append(ok and y != ra.label)
                for k, col in cols.items():
                    arr[(v, k)] = np.array(col, float)
                rep_a[v] = {"calibration_split": rec, "marginal_ltt": ltt_brief(ltt_t), "marginal_qhat": fit_t["qhat"],
                            "area_ltt": ltt_rep, "K": K}
            # TEST acted error: decision-weighted, cluster-weighted (1 / TEST decisions present per cluster, as the
            # parent), and the certified c-weighted risks (w = min(1, c / design size))
            cl_m = [ckey(rid(r.id)) for r in test_t]
            cl_a = [(ckey(rid(r.id)), gmap_c[area_of_rid[rid(r.id)]]) for r in test_t]
            cnt_m, cnt_a = collections.Counter(cl_m), collections.Counter(cl_a)
            w = {"marginal": {"cluster_weighted": np.array([1.0 / cnt_m[c] for c in cl_m]),
                              "cap8_weighted": np.array([min(1.0, 8 / size_m[c]) for c in cl_m])},
                 "area": {"cluster_weighted": np.array([1.0 / cnt_a[c] for c in cl_a]),
                          "cap8_weighted": np.array([min(1.0, 8 / size_a[c]) for c in cl_a])}}
            for v in VARIANTS:
                for an in ANALYSES:
                    act, err = arr[(v, "act_" + an)], arr[(v, "err_" + an)]
                    na = act.sum()
                    s = {"act": float(act.mean()), "n_act": int(na),
                         "acted_error": float(err.sum() / na) if na else None}
                    for wk, wv in w[an].items():
                        s["acted_error_" + wk] = float((err * wv).sum() / (act * wv).sum()) if na else None
                    rep_a[v][an + "_test"] = s
            m["alpha"][str(alpha)] = rep_a
            arrays[alpha] = arr
        per_test[name] = ([r.id for r in test_t], arrays)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        del sc, cal_t, test_t
        print(f"[{mode}] {name} done in {m['seconds']}s", file=sys.stderr, flush=True)

    common = set.intersection(*(set(v[0]) for v in per_test.values()))
    order = sorted(common)
    clusters = sorted({case[rid(d)] for d in order})
    cidx = {c: i for i, c in enumerate(clusters)}
    ci_of = np.array([cidx[case[rid(d)]] for d in order])
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=B).astype(np.float32)

    def csum(x):
        return np.bincount(ci_of, weights=x, minlength=len(clusters))

    n_b = W @ csum(np.ones(len(order)))
    pos = {name: {d: i for i, d in enumerate(per_test[name][0])} for name in MODELS}
    idx = {name: np.array([pos[name][d] for d in order]) for name in MODELS}

    def vec(name, alpha, key):
        return per_test[name][1][alpha][key][idx[name]]

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    boot = {"B": B, "seed": SEED, "n_clusters": len(clusters), "n_decisions": len(order), "pairs": {},
            "within_model": {}}
    for alpha in ALPHAS:
        for x, y in PAIRS:
            for v in VARIANTS:
                for an in ANALYSES:
                    dx, dy = vec(x, alpha, (v, "act_" + an)), vec(y, alpha, (v, "act_" + an))
                    lo, hi = ci((W @ csum(dx)) / n_b - (W @ csum(dy)) / n_b)
                    verdict = (f"{x} more actionable" if lo > 0 else (f"{y} more actionable" if hi < 0
                                                                       else "no separation"))
                    boot["pairs"].setdefault(f"{x}-{y}", {}).setdefault(str(alpha), {}).setdefault(v, {})[an] = {
                        "diff": float(dx.mean() - dy.mean()), "ci95": [lo, hi], "verdict": verdict}
        for name in MODELS:  # crr8 - cluster and crr8 - published, same model (descriptive)
            for ref in ("cluster", "published"):
                for an in ANALYSES:
                    dx, dy = vec(name, alpha, ("crr8", "act_" + an)), vec(name, alpha, (ref, "act_" + an))
                    lo, hi = ci((W @ csum(dx)) / n_b - (W @ csum(dy)) / n_b)
                    boot["within_model"].setdefault(name, {}).setdefault(str(alpha), {}).setdefault(
                        f"crr8-{ref}", {})[an] = {"diff": float(dx.mean() - dy.mean()), "ci95": [lo, hi]}
    split_rec = {"mode": mode, "salt": GF.SALT, "cluster_key": "run_id" if mode == "run_id" else "harness case id",
                 "test_decision_ids_sha256": G.ids_sha256(test_ids), "n_test_decisions": len(test_ids),
                 "caps": caps, "design_size_max": s_max,
                 "design_clusters": len(size_m), "design_clusters_above_8": sum(s > 8 for s in size_m.values())}
    return {"split": split_rec, "models": res, "bootstrap": boot}


def check_reproduction(out):
    """published == both parent receipts; cluster == GUARANTEES_CLUSTER.json (TEST act fractions, exact)."""
    with open(os.path.join(FULLDEV_DIR, "GUARANTEES_FULLDEV.json"), encoding="utf-8") as f:
        fd = json.load(f)
    with open(os.path.join(AREA_DIR, "GUARANTEES_AREA.json"), encoding="utf-8") as f:
        ar = json.load(f)
    with open(CLUSTER_JSON, encoding="utf-8") as f:
        cj = json.load(f)
    rows = []
    for mode in ("run_id", "case"):
        for name in MODELS:
            for a in map(str, ALPHAS):
                got = out[mode]["models"][name]["alpha"][a]
                want_m = fd[mode]["models"][name]["alpha"][a]["selective"]["act_rate"]
                want_a = ar[mode]["models"][name]["alpha"][a]["overall"]["act"]
                cw = cj[mode]["models"][name]["alpha"][a]["cluster"]
                ok = (got["published"]["marginal_test"]["act"] == want_m and got["published"]["area_test"]["act"] == want_a
                      and got["cluster"]["marginal_test"]["act"] == cw["marginal_test"]["act"]
                      and got["cluster"]["area_test"]["act"] == cw["area_test"]["act"])
                rows.append({"mode": mode, "model": name, "alpha": a, "exact": ok})
    assert all(r["exact"] for r in rows), [r for r in rows if not r["exact"]]
    return rows


def main():
    out_path = sys.argv[5]
    shas = {}
    for name, (p, want) in MODELS.items():
        got = GF.sha_file(p)
        assert got == want, (name, got)
        shas[name] = got
    ids, gold, cat, case = GF.load_gold()
    amap = area_of("0.2.1")
    print(f"gold: {len(gold)} decisions", file=sys.stderr, flush=True)
    out = {"prereg": "B-PREREG-GUAR-CRR (state 033906aa)", "edition": "0.2.1", "epsilon": EPS, "delta": DELTA,
           "alphas": ALPHAS, "min_n": MIN_N, "primary_cap": PRIMARY_CAP, "crr_bet_cap": G.CRR_BET_CAP,
           "input_sha256": shas,
           "code_sha256": {"jevkit/guarantees.py": GF.sha_file(G.__file__),
                           "guarantees_fulldev.py": GF.sha_file(GF.__file__),
                           "guarantees_crr.py": GF.sha_file(os.path.abspath(__file__)),
                           "GUARANTEES_CLUSTER.json": GF.sha_file(CLUSTER_JSON)},
           "n_rows": len(ids), "n_decisions": len(gold), "no_gold_decisions_excluded": dict(GF.NOGOLD),
           "estimand": "crr variants: R_c(lambda) = E[w_r L_r] / E[w_r n_r] over i.i.d. clusters, w_r = min(1, c/s_r), "
                       "s_r = DEV gold decisions in the cluster (per area: in the (cluster, area group) cell); "
                       "primary c = 8"}
    for mode in ("run_id", "case"):
        out[mode] = run_split(mode, ids, gold, cat, case, amap)
    out["reproduction"] = check_reproduction(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

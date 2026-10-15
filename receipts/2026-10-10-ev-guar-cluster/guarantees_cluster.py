"""B-PREREG-GUAR-CLUSTER (state 68267953): cluster-level FIT/LTT split + one-decision-per-cluster binomial LTT, for the
marginal (B-PREREG-GUAR-FULLDEV) and per-area (B-PREREG-GUAR-AREA) actionability numbers on the full-DEV predictions.

    python -X utf8 -I guarantees_cluster.py <checkout with this branch's jevkit + harness> \
        <dir of receipts/2026-10-10-B-guarantees-fulldev> <dir of receipts/2026-10-10-ev-guar-area> <out.json>

Variants per model x alpha x configuration, each for the marginal (per-type Mondrian, pooled LTT) and the per-area
(area Mondrian, Bonferroni over areas x grid) analysis:
  published   per-decision FIT/LTT split, per-decision binomial (must reproduce the published TEST act fractions)
  split_only  cluster FIT/LTT split, per-decision binomial (decomposition)
  cluster     cluster FIT/LTT split, binomial on one decision per cluster (PRIMARY)
Configurations: PRIMARY run_id CAL/TEST split with cluster = row; CASE variant case CAL/TEST split with cluster = case.
"""
import collections
import json
import math
import os
import sys
import time

import numpy as np

FULLDEV_DIR, AREA_DIR = sys.argv[2], sys.argv[3]
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
VARIANTS = ("published", "split_only", "cluster")
ANALYSES = ("marginal", "area")
PAIRS = [("D19", "ClefFlash"), ("D19", "Clef27B"), ("D13", "ClefFlash"), ("D13", "Clef27B"), ("D19", "D13"),
         ("Clef27B", "ClefFlash")]


def rid(d):
    return d.split("#", 1)[0]


def group_plan(gold, is_cal, area_of_rid, ckey, clustered):
    """GUAR-AREA's MIN_N rule, on gold decisions only (answer-free). Clustered: FIT counts decisions of the cluster
    split; LTT counts the one-per-(cluster, area) picks, i.e. the trials the per-area test will have."""
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
    keep = ("lambda", "n_cal", "n_clusters", "n_decisions", "act_rate_cal", "accepted_error_cal", "accepted_error_ucb",
            "n_valid_lambdas", "test_level", "ltt_ids_sha256", "picked_ids_sha256", "fit_ids_sha256")
    return {k: rep[k] for k in keep if k in rep}


def run_split(mode, ids, gold, cat, case, amap):
    is_cal = GF.split(ids, case, mode)
    ckey = (lambda r: r) if mode == "run_id" else (lambda r: case[r])
    area_of_rid = {r: amap.get(cat[r], "non-index") for r in ids}
    plans = {v: group_plan(gold, is_cal, area_of_rid, ckey, v != "published") for v in VARIANTS}
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
                clustered_split = v != "published"
                cl = (lambda r: ckey(rid(r.id))) if clustered_split else None
                cl_test = cl if v == "cluster" else None
                # ---- marginal (per-type Mondrian, pooled LTT)
                fit_h, ltt_h, rec = G.split_calibration(cal_t, 0, 0.5, cluster_of=cl)
                fit_t = G.fit_conformal(fit_h, alpha, True)
                ltt_t = G.ltt_select(ltt_h, fit_t, EPS, DELTA, method="bonferroni", cluster_of=cl_test)
                # ---- per-area (area Mondrian; Bonferroni over live groups x grid)
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
                    rep = G.ltt_select(by_g[g], fit_a, EPS, DELTA / K, method="bonferroni", cluster_of=cl_test)
                    lam_a[g], ltt_rep[g] = rep["lambda"], ltt_brief(rep)
                # ---- TEST, per decision
                cols = collections.defaultdict(list)
                for r in test_t:
                    ra = G.Scored(r.id, gof(r), r.probs, r.label)
                    s_m = G.prediction_set(r.probs, G._qhat_for(fit_t, r))
                    s_a = G.prediction_set(ra.probs, G._qhat_for(fit_a, ra))
                    cols["size_marginal"].append(len(s_m))
                    cols["cov_marginal"].append(r.label in s_m)
                    cols["size_area"].append(len(s_a))
                    cols["cov_area"].append(ra.label in s_a)
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
                            "area_plan": plan, "area_qhat": fit_a["qhat"], "area_ltt": ltt_rep, "K": K}
            # summaries; cluster-weighted acted error uses the configuration's cluster key (per-area: (cluster, area))
            cl_m = np.array([ckey(rid(r.id)) for r in test_t])
            cl_a = np.array([ckey(rid(r.id)) + "|" + area_of_rid[rid(r.id)] for r in test_t])
            _, inv_m, cnt_m = np.unique(cl_m, return_inverse=True, return_counts=True)
            _, inv_a, cnt_a = np.unique(cl_a, return_inverse=True, return_counts=True)
            w = {"marginal": 1.0 / cnt_m[inv_m], "area": 1.0 / cnt_a[inv_a]}
            for v in VARIANTS:
                for an in ANALYSES:
                    act, err = arr[(v, "act_" + an)], arr[(v, "err_" + an)]
                    na = act.sum()
                    rep_a[v][an + "_test"] = {
                        "act": float(act.mean()), "acted_error": float(err.sum() / na) if na else None,
                        "acted_error_cluster_weighted": (float((err * w[an]).sum() / (act * w[an]).sum())
                                                         if na else None),
                        "n_act": int(na), "coverage": float(arr[(v, "cov_" + an)].mean()),
                        "mean_set_size": float(arr[(v, "size_" + an)].mean())}
            m["alpha"][str(alpha)] = rep_a
            arrays[alpha] = arr
        per_test[name] = ([r.id for r in test_t], arrays)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        del sc, cal_t, test_t
        print(f"[{mode}] {name} done in {m['seconds']}s", file=sys.stderr, flush=True)

    # paired case-cluster bootstrap over the TEST decisions (identical construction to the parent receipts)
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

    boot = {"B": B, "seed": SEED, "n_clusters": len(clusters), "n_decisions": len(order), "pairs": {}}
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
    split_rec = {"mode": mode, "salt": GF.SALT, "cluster_key": "run_id" if mode == "run_id" else "harness case id",
                 "test_decision_ids_sha256": G.ids_sha256(test_ids), "n_test_decisions": len(test_ids)}
    return {"split": split_rec, "models": res, "bootstrap": boot}


def check_reproduction(out):
    """The 'published' variant must reproduce both parent receipts' TEST act fractions exactly."""
    with open(os.path.join(FULLDEV_DIR, "GUARANTEES_FULLDEV.json"), encoding="utf-8") as f:
        fd = json.load(f)
    with open(os.path.join(AREA_DIR, "GUARANTEES_AREA.json"), encoding="utf-8") as f:
        ar = json.load(f)
    rows = []
    for mode in ("run_id", "case"):
        for name in MODELS:
            for a in map(str, ALPHAS):
                got = out[mode]["models"][name]["alpha"][a]["published"]
                want_m = fd[mode]["models"][name]["alpha"][a]["selective"]["act_rate"]
                want_a = ar[mode]["models"][name]["alpha"][a]["overall"]["act"]
                ok = got["marginal_test"]["act"] == want_m and got["area_test"]["act"] == want_a
                rows.append({"mode": mode, "model": name, "alpha": a, "marginal": [got["marginal_test"]["act"], want_m],
                             "area": [got["area_test"]["act"], want_a], "exact": ok})
    assert all(r["exact"] for r in rows), [r for r in rows if not r["exact"]]
    return rows


def main():
    out_path = sys.argv[4]
    shas = {}
    for name, (p, want) in MODELS.items():
        got = GF.sha_file(p)
        assert got == want, (name, got)
        shas[name] = got
    ids, gold, cat, case = GF.load_gold()
    amap = area_of("0.2.1")
    print(f"gold: {len(gold)} decisions", file=sys.stderr, flush=True)
    out = {"prereg": "B-PREREG-GUAR-CLUSTER (state 68267953)", "edition": "0.2.1", "epsilon": EPS, "delta": DELTA,
           "alphas": ALPHAS, "min_n": MIN_N, "input_sha256": shas,
           "code_sha256": {"jevkit/guarantees.py": GF.sha_file(G.__file__),
                           "guarantees_fulldev.py": GF.sha_file(GF.__file__),
                           "guarantees_cluster.py": GF.sha_file(os.path.abspath(__file__))},
           "n_rows": len(ids), "n_decisions": len(gold), "no_gold_decisions_excluded": dict(GF.NOGOLD),
           "estimand": "cluster variant: P(error | act) for a uniformly random cluster's uniformly random decision "
                       "(cluster-weighted acted risk); clusters assumed i.i.d."}
    for mode in ("run_id", "case"):
        out[mode] = run_split(mode, ids, gold, cat, case, amap)
    out["reproduction"] = check_reproduction(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

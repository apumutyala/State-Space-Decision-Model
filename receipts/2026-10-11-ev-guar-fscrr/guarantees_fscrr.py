"""B-PREREG-GUAR-FSCRR: fixed-sequence cluster-robust ratio LTT (CRR betting statistic of B-PREREG-GUAR-CRR, cap 8,
lambda tested in a fixed descending sequence from the pre-registered start 0.95, stop at the first non-rejection or
at the first lambda that acts on nothing) on the UNSEEN Index 0.2.1 pool sample (pool_sample.py).

    python -X utf8 -I guarantees_fscrr.py <checkout with this branch's jevkit + harness> <inputs.json> <out.json>

inputs.json: {"rows": <pool-sample-rows.jsonl.gz>, "rows_sha256": ..., "manifest": <pool-sample-manifest.jsonl.gz>,
              "manifest_sha256": ..., "models": {"D19": [path, sha256], "D13": [...], "ClefFlash": [...],
              "Clef27B": [...]}}   (D13 optional: the prereg makes it a non-primary arm)

Variants (marginal = per-type Mondrian + pooled LTT; area = area Mondrian, Bonferroni over the K live area groups,
fixed sequence inside each group):
  fs8        cluster split, CRR c = 8, fixed sequence from 0.95 at level delta (PRIMARY)
  fs1        same, c = 1 (secondary)
  fs_max     same, c = largest cluster (decision-weighted; secondary)
  bf8        cluster split, CRR c = 8, Bonferroni over the 101-point grid (= B-PREREG-GUAR-CRR; secondary)
  cluster    cluster split, one decision per cluster, Bonferroni (= B-PREREG-GUAR-CLUSTER; secondary)
  published  per-decision split and binomial, Bonferroni (= B-PREREG-GUAR-FULLDEV; secondary, not a guarantee)
"""
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from harness.suite import area_of  # noqa: E402
from jevkit import guarantees as G  # noqa: E402
from jevkit.answers import validate_distribution  # noqa: E402

PRIMARY_CAP, FS_START = 8, 0.95
SALT = "B-PREREG-GUAR-FULLDEV|v1"  # the parents' CAL/TEST salt and procedure, unchanged
ALPHAS = (0.10, 0.05)
EPS, DELTA = 0.05, 0.10
B, SEED = 2000, 0
MIN_N = 100
SMALL = "pooled-small"
VARIANTS = ("fs8", "fs1", "fs_max", "bf8", "cluster", "published")
ANALYSES = ("marginal", "area")
PAIRS = [("D19", "ClefFlash"), ("D19", "Clef27B"), ("D13", "ClefFlash"), ("D13", "Clef27B"), ("D19", "D13"),
         ("Clef27B", "ClefFlash")]
NOGOLD = collections.Counter()


def rid(d):
    return d.split("#", 1)[0]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def u(key):
    return int(hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def load_gold(rows_path, manifest_path):
    """Stream the pool-sample rows: gold per decision (as guarantees_fulldev.load_gold), catalog per run; source case
    per run from the manifest (harness.cases components over the whole pool, fixed at build)."""
    with gzip.open(manifest_path, "rt", encoding="utf-8") as f:
        man = [json.loads(line) for line in f if line.strip()]
    case = {m["run_id"]: m["case_id"] for m in man}
    gold, cat, ids = {}, {}, []
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            e = r["_evaluation"]
            ids.append(e["run_id"])
            cat[e["run_id"]] = e["catalog_id"]
            for key, q in r["questions"].items():
                exp = r["expected"].get(key)
                if exp is None:
                    NOGOLD[q["type"]] += 1
                    continue
                if q["type"] == "noul":
                    assert isinstance(exp, bool)
                    gold[f"{e['run_id']}#{key}"] = ("noul", "yes" if exp else "no", ["yes", "no"])
                elif q["type"] == "choice":
                    gold[f"{e['run_id']}#{key}"] = ("choice", str(exp), [str(c) for c in q["criteria"]])
                else:
                    raise SystemExit(f"unexpected question type {q['type']}")
    assert ids == [m["run_id"] for m in man], "rows do not match the manifest"
    return ids, gold, cat, case


def load_model(path, gold):
    """{decision_id: Scored} for valid distributions (= guarantees_fulldev.load_model)."""
    out, bad = {}, collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ans = ((r.get("response") or {}).get("answers") or {}) if r.get("status") == "ok" else {}
            for key, a in ans.items():
                did = f"{r['run_id']}#{key}"
                if did not in gold:
                    bad["not_in_split"] += 1
                    continue
                t, y, labels = gold[did]
                try:
                    if t == "noul":
                        p = float(a["noul"])
                        if not 0.0 <= p <= 1.0:
                            raise ValueError
                        probs = {"yes": p, "no": 1.0 - p}
                    else:
                        probs, _s, _r = validate_distribution(a["probabilities"], labels)
                except Exception:
                    bad["invalid"] += 1
                    continue
                out[did] = G.Scored(did, t, probs, y)
    bad["missing"] = len(set(gold) - set(out))
    return out, dict(bad)


def group_plan(gold, is_cal, area_of_rid, ckey, clustered):
    """GUAR-AREA's MIN_N rule on gold decisions (answer-free), as guarantees_crr.group_plan."""
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
    keep = ("lambda", "method", "n_cal", "n_clusters", "n_decisions", "act_rate_cal", "accepted_error_cal",
            "accepted_error_cal_weighted", "accepted_error_ucb", "p_value", "log_capital", "n_valid_lambdas",
            "test_level", "crr_cap", "n_clusters_above_cap", "max_cluster_size", "fs_start", "fs_stop", "fs_tested",
            "ltt_ids_sha256", "picked_ids_sha256", "fit_ids_sha256")
    return {k: rep[k] for k in keep if k in rep}


def variant_kw(v, cl, caps):
    if v == "published":
        return {}, None
    if v == "cluster":
        return {"cluster_of": cl}, None
    cap = caps[v]
    kw = {"cluster_of": cl, "cluster_test": "crr", "crr_cap": cap}
    if v.startswith("fs"):
        kw.update({"method": "fixed_sequence", "fs_start": FS_START})
    return kw, cap


def run_split(mode, ids, gold, cat, case, amap, models):
    is_cal = {r: u(r if mode == "run_id" else "case|" + case[r]) < 0.5 for r in ids}
    ckey = (lambda r: r) if mode == "run_id" else (lambda r: case[r])
    area_of_rid = {r: amap.get(cat[r], "non-index") for r in ids}
    plan_p = group_plan(gold, is_cal, area_of_rid, ckey, False)
    plan_c = group_plan(gold, is_cal, area_of_rid, ckey, True)
    gmap_c = plan_c["group_of_area"]
    size_m = collections.Counter(ckey(rid(d)) for d in gold)
    size_a = collections.Counter((ckey(rid(d)), gmap_c.get(area_of_rid[rid(d)])) for d in gold)
    s_max = max(size_m.values())
    caps = {"fs8": PRIMARY_CAP, "fs1": 1, "fs_max": s_max, "bf8": PRIMARY_CAP}
    test_ids = sorted(d for d in gold if not is_cal[rid(d)])
    res, per_test = {}, {}
    for name, (path, _sha) in models.items():
        t0 = time.time()
        sc, bad = load_model(path, gold)
        cal_t = [sc[d] for d in sorted(sc) if is_cal[rid(d)]]
        test_t = [sc[d] for d in test_ids if d in sc]
        m = {"invalid_or_missing": bad, "n_cal": len(cal_t), "n_test": len(test_t), "alpha": {}}
        arrays = {}
        for alpha in ALPHAS:
            arr, rep_a = {}, {}
            for v in VARIANTS:
                plan = plan_p if v == "published" else plan_c
                cl = None if v == "published" else (lambda r: ckey(rid(r.id)))
                kw, _cap = variant_kw(v, cl, caps)
                fit_h, ltt_h, rec = G.split_calibration(cal_t, 0, 0.5, cluster_of=cl)
                fit_t = G.fit_conformal(fit_h, alpha, True)
                kw_m = dict(kw, cluster_size=size_m.__getitem__) if "cluster_test" in kw else kw
                ltt_t = G.ltt_select(ltt_h, fit_t, EPS, DELTA, **({"method": "bonferroni"} | kw_m))
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
                    kw_g = (dict(kw, cluster_size=lambda c, g=g: size_a[(c, g)]) if "cluster_test" in kw else kw)
                    rep = G.ltt_select(by_g[g], fit_a, EPS, DELTA / K, **({"method": "bonferroni"} | kw_g))
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
                    s = {"act": float(act.mean()) if len(act) else 0.0, "n_act": int(na),
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
    pos = {name: {d: i for i, d in enumerate(per_test[name][0])} for name in models}
    idx = {name: np.array([pos[name][d] for d in order], dtype=int) for name in models}

    def vec(name, alpha, key):
        return per_test[name][1][alpha][key][idx[name]]

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    boot = {"B": B, "seed": SEED, "n_clusters": len(clusters), "n_decisions": len(order), "pairs": {}}
    for alpha in ALPHAS:
        for x, y in PAIRS:
            if x not in models or y not in models:
                continue
            for v in VARIANTS:
                for an in ANALYSES:
                    dx, dy = vec(x, alpha, (v, "act_" + an)), vec(y, alpha, (v, "act_" + an))
                    lo, hi = ci((W @ csum(dx)) / n_b - (W @ csum(dy)) / n_b)
                    verdict = (f"{x} more actionable" if lo > 0 else (f"{y} more actionable" if hi < 0
                                                                       else "no separation"))
                    boot["pairs"].setdefault(f"{x}-{y}", {}).setdefault(str(alpha), {}).setdefault(v, {})[an] = {
                        "diff": float(dx.mean() - dy.mean()), "ci95": [lo, hi], "verdict": verdict}
    split_rec = {"mode": mode, "salt": SALT, "cluster_key": "run_id" if mode == "run_id" else "pool source case",
                 "cal_rows": sum(is_cal.values()), "test_rows": len(ids) - sum(is_cal.values()),
                 "test_decision_ids_sha256": G.ids_sha256(test_ids), "n_test_decisions": len(test_ids),
                 "caps": caps, "design_size_max": s_max, "design_clusters": len(size_m),
                 "design_clusters_above_8": sum(s > 8 for s in size_m.values()),
                 "group_plan_cluster": plan_c, "group_plan_published": plan_p}
    return {"split": split_rec, "models": res, "bootstrap": boot}


def main():
    with open(sys.argv[2], encoding="utf-8") as f:
        inp = json.load(f)
    out_path = sys.argv[3]
    assert sha_file(inp["rows"]) == inp["rows_sha256"]
    assert sha_file(inp["manifest"]) == inp["manifest_sha256"]
    models = {k: tuple(v) for k, v in inp["models"].items()}
    shas = {}
    for name, (p, want) in models.items():
        got = sha_file(p)
        assert got == want, (name, got)
        shas[name] = got
    ids, gold, cat, case = load_gold(inp["rows"], inp["manifest"])
    print(f"gold: {len(gold)} decisions over {len(ids)} rows", file=sys.stderr, flush=True)
    out = {"prereg": "B-PREREG-GUAR-FSCRR", "split": "Index 0.2.1 pool sample (pool_sample.py)", "edition": "0.2.1",
           "epsilon": EPS, "delta": DELTA, "alphas": ALPHAS, "min_n": MIN_N, "primary_cap": PRIMARY_CAP,
           "fs_start": FS_START, "crr_bet_cap": G.CRR_BET_CAP, "input_sha256": shas,
           "rows_sha256": inp["rows_sha256"], "manifest_sha256": inp["manifest_sha256"],
           "code_sha256": {"jevkit/guarantees.py": sha_file(G.__file__),
                           "guarantees_fscrr.py": sha_file(os.path.abspath(__file__))},
           "n_rows": len(ids), "n_decisions": len(gold), "no_gold_decisions_excluded": dict(NOGOLD),
           "estimand": "R_8(lambda) = E[w_r L_r] / E[w_r n_r] over i.i.d. clusters, w_r = min(1, 8/s_r), s_r = gold "
                       "decisions of the cluster in the pool sample (per area: in the (cluster, area group) cell)"}
    for mode in ("run_id", "case"):
        out[mode] = run_split(mode, ids, gold, cat, case, area_of("0.2.1"), models)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

"""B-PREREG-GUAR-AREA: Mondrian-per-area split conformal + per-area Learn-then-Test on the full-DEV predictions of
B-PREREG-GUAR-FULLDEV (same inputs, gold, splits, FIT/LTT cut). jevkit/guarantees.py @ 7c131f1 unmodified: the
Mondrian group key is set to the 0.2.1 area by building Scored(type=<area>).

    python -I guarantees_area.py <agent-main checkout @7c131f1> <out.json>
"""
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-10-B-guarantees-fulldev"))
import guarantees_fulldev as GF  # noqa: E402  (inserts argv[1] on sys.path; reuses load_gold / split / load_model)
from harness.suite import area_of  # noqa: E402
from jevkit import guarantees as G  # noqa: E402

MODELS = GF.MODELS
ALPHAS = (0.10, 0.05)
EPS, DELTA = 0.05, 0.10
B, SEED = 2000, 0
MIN_N = 100
SMALL = "pooled-small"
GRID = G.DEFAULT_LAMBDAS
PAIRS = [("D19", "ClefFlash"), ("D19", "Clef27B"), ("D13", "Clef27B"), ("D13", "ClefFlash"), ("D19", "D13"),
         ("Clef27B", "ClefFlash")]
PRIMARY_PAIRS = PAIRS[:2]


def rid(d):
    return d.split("#", 1)[0]


def group_plan(gold, is_cal, area_of_rid):
    """Fixed before any score: per-area FIT/LTT counts over the gold decisions -> areas with a guarantee vs pooled."""
    cal = [G.Scored(d, "x", {}) for d in sorted(gold) if is_cal[rid(d)]]
    fit, ltt, _ = G.split_calibration(cal, 0, 0.5)
    nf = collections.Counter(area_of_rid[rid(r.id)] for r in fit)
    nl = collections.Counter(area_of_rid[rid(r.id)] for r in ltt)
    areas = sorted(set(nf) | set(nl))
    small = [a for a in areas if nf[a] < MIN_N or nl[a] < MIN_N]
    gmap = {a: (SMALL if a in small else a) for a in areas}
    gf = collections.Counter()
    gl = collections.Counter()
    for a in areas:
        gf[gmap[a]] += nf[a]
        gl[gmap[a]] += nl[a]
    dead = [g for g in set(gmap.values()) if gf[g] < MIN_N or gl[g] < MIN_N]  # only pooled-small can land here
    return {"n_fit_area": dict(nf), "n_ltt_area": dict(nl), "pooled_into_small": small, "group_of_area": gmap,
            "no_guarantee_groups": dead, "min_n": MIN_N}


def holm(pvals, delta):
    """pvals: {(g, lam): p} over the whole family (untestable hypotheses carry p = 1). Step-down Holm."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], -kv[0][1]))
    m, rej = len(items), set()
    for i, (h, p) in enumerate(items):
        if p <= delta / (m - i):
            rej.add(h)
        else:
            break
    return rej


def run_split(mode, ids, gold, cat, case, amap):
    is_cal = GF.split(ids, case, mode)
    area_of_rid = {r: amap.get(cat[r], "non-index") for r in ids}
    plan = group_plan(gold, is_cal, area_of_rid)
    gmap = plan["group_of_area"]
    grp_of = {r: gmap[area_of_rid[r]] for r in ids}
    groups = sorted(set(gmap.values()))
    live = [g for g in groups if g not in plan["no_guarantee_groups"]]
    K = len(live)
    test_ids = sorted(d for d in gold if not is_cal[rid(d)])
    res, per_test = {}, {}
    for name in MODELS:
        t0 = time.time()
        sc, bad = GF.load_model(MODELS[name][0], gold)
        cal_t = [sc[d] for d in sorted(sc) if is_cal[rid(d)]]
        test_t = [sc[d] for d in test_ids if d in sc]
        cal_a = [G.Scored(r.id, grp_of[rid(r.id)], r.probs, r.label) for r in cal_t]
        test_a = [G.Scored(r.id, grp_of[rid(r.id)], r.probs, r.label) for r in test_t]
        m = {"invalid_or_missing": bad, "n_cal": len(cal_a), "n_test": len(test_a), "K": K, "alpha": {}}
        arrays = {}
        for alpha in ALPHAS:
            # --- marginal reference (per-type Mondrian, as GUARANTEES_FULLDEV) -- reproduces the published numbers
            fit_t_half, ltt_t_half, rec_t = G.split_calibration(cal_t, 0, 0.5)
            fit_t = G.fit_conformal(fit_t_half, alpha, True)
            ltt_t = G.ltt_select(ltt_t_half, fit_t, EPS, DELTA, method="bonferroni")
            sel_t = G.selective_eval(test_t, fit_t, ltt_t["lambda"])
            # --- area-conditional
            fit_a_half, ltt_a_half, rec_a = G.split_calibration(cal_a, 0, 0.5)
            assert rec_a["fit_ids_sha256"] == rec_t["fit_ids_sha256"], "FIT half must equal the marginal run's"
            fit_a = G.fit_conformal(fit_a_half, alpha, True)
            for g in plan["no_guarantee_groups"]:
                fit_a["qhat"][g] = math.inf
            ltt_by_g = collections.defaultdict(list)
            for r in ltt_a_half:
                ltt_by_g[r.type].append(r)
            lam_b, ltt_rep, pv = {}, {}, {}
            for g in groups:
                if g not in live:
                    lam_b[g] = None
                    ltt_rep[g] = {"lambda": None, "note": "insufficient (n < %d): abstain" % MIN_N}
                    for lam in GRID:
                        pv[(g, lam)] = 1.0
                    continue
                rep = G.ltt_select(ltt_by_g[g], fit_a, EPS, DELTA / K, method="bonferroni")
                lam_b[g] = rep["lambda"]
                ltt_rep[g] = rep
                # Holm family: same p as jevkit (P(Bin(n_act, eps) <= err)); untestable (n_act = 0) -> p = 1
                pre = []
                for r in ltt_by_g[g]:
                    s = G.prediction_set(r.probs, G._qhat_for(fit_a, r))
                    pre.append((s[0], r.probs[s[0]], r.label) if len(s) == 1 else None)
                for lam in GRID:
                    acted = [(y, lab) for x in pre if x is not None for y, p, lab in [x] if p >= lam]
                    n, e = len(acted), sum(y != lab for y, lab in acted)
                    pv[(g, lam)] = G.binom_cdf(e, n, EPS) if n else 1.0
            rej = holm(pv, DELTA)
            lam_h = {g: min((lam for (gg, lam) in rej if gg == g), default=None) for g in groups}
            for g in groups:
                ltt_rep[g]["lambda_holm"] = lam_h[g]
                ltt_rep[g]["n_ltt"] = len(ltt_by_g[g])
            # --- TEST evaluation, per decision
            cols = collections.defaultdict(list)
            for rt, ra in zip(test_t, test_a):
                s_m = G.prediction_set(rt.probs, G._qhat_for(fit_t, rt))
                s_a = G.prediction_set(ra.probs, G._qhat_for(fit_a, ra))
                cols["area"].append(area_of_rid[rid(ra.id)])
                cols["type"].append(rt.type)
                cols["cov_m"].append(rt.label in s_m)
                cols["size_m"].append(len(s_m))
                ok_m, y_m = G._acts(rt, fit_t, ltt_t["lambda"]) if ltt_t["lambda"] is not None else (False, None)
                cols["act_m"].append(ok_m)
                cols["err_m"].append(ok_m and y_m != rt.label)
                cols["cov"].append(ra.label in s_a)
                cols["size"].append(len(s_a))
                cols["single"].append(len(s_a) == 1)
                cols["empty"].append(len(s_a) == 0)
                for tag, lams in (("", lam_b), ("_holm", lam_h)):
                    lam = lams[ra.type]
                    ok, y = G._acts(ra, fit_a, lam) if lam is not None else (False, None)
                    cols["act" + tag].append(ok)
                    cols["err" + tag].append(ok and y != ra.label)
            area_arr = np.array(cols.pop("area"))
            type_arr = np.array(cols.pop("type"))
            arr = {k: np.array(v, float) for k, v in cols.items()}
            arr["_area"] = area_arr

            def summ(mask):
                n = int(mask.sum())
                if n == 0:
                    return {"n": 0}
                out = {"n": n}
                for k in ("cov_m", "size_m", "act_m", "cov", "size", "single", "empty", "act", "act_holm"):
                    out[k] = float(arr[k][mask].mean())
                for tag in ("_m", "", "_holm"):
                    na = arr["act" + tag][mask].sum()
                    out["acted_err" + tag] = float(arr["err" + tag][mask].sum() / na) if na else None
                return out

            per_area = {}
            for a in sorted(set(area_arr)):
                msk = area_arr == a
                row = summ(msk)
                se = math.sqrt(alpha * (1 - alpha) / row["n"])
                row["flag_line"] = 1 - alpha - 2 * se
                row["flag_marginal"] = row["cov_m"] < row["flag_line"]
                row["flag_area"] = row["cov"] < row["flag_line"]
                g = gmap[a]
                row["group"] = g
                row["qhat"] = fit_a["qhat"].get(g)
                row["n_fit"] = fit_a["n"].get(g)
                row["lambda_bonf"] = lam_b[g]
                row["lambda_holm"] = lam_h[g]
                row["by_type_cov"] = {t: float(arr["cov"][msk & (type_arr == t)].mean())
                                      for t in sorted(set(type_arr[msk]))}
                row["by_type_n"] = {t: int((msk & (type_arr == t)).sum()) for t in sorted(set(type_arr[msk]))}
                per_area[a] = row
            m["alpha"][str(alpha)] = {
                "marginal_reference": {"lambda": ltt_t["lambda"], "test_act": sel_t["act_rate"],
                                       "test_acted_error": sel_t["acted_error"], "qhat": G.public_fit(fit_t)["qhat"]},
                "area_fit": G.public_fit(fit_a), "ltt_per_group": ltt_rep, "holm_rejections": len(rej),
                "overall": summ(np.ones(len(area_arr), bool)), "per_area": per_area,
                "calibration_split": rec_a}
            arrays[alpha] = arr
        per_test[name] = ([r.id for r in test_a], arrays)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        del sc, cal_t, test_t, cal_a, test_a
        print(f"[{mode}] {name} done in {m['seconds']}s", file=sys.stderr, flush=True)

    # paired case-cluster bootstrap (same construction as GUARANTEES_FULLDEV)
    common = set.intersection(*(set(v[0]) for v in per_test.values()))
    order = sorted(common)
    clusters = sorted({case[rid(d)] for d in order})
    cidx = {c: i for i, c in enumerate(clusters)}
    ci_of = np.array([cidx[case[rid(d)]] for d in order])
    area_o = np.array([area_of_rid[rid(d)] for d in order])
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=B).astype(np.float32)

    def csum(x):
        return np.bincount(ci_of, weights=x, minlength=len(clusters))

    def vec(name, alpha, field):
        ids_list, arrs = per_test[name]
        pos = {d: i for i, d in enumerate(ids_list)}
        return arrs[alpha][field][[pos[d] for d in order]]

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    def pair(x, y, alpha, field, mask=None):
        mk = np.ones(len(order)) if mask is None else mask.astype(float)
        den = W @ csum(mk)
        dx, dy = vec(x, alpha, field) * mk, vec(y, alpha, field) * mk
        diff = (W @ csum(dx)) / den - (W @ csum(dy)) / den
        return float(dx.sum() / mk.sum() - dy.sum() / mk.sum()), ci(diff)

    def verdict(x, y, field, lo, hi):
        if field.startswith("size"):
            return f"{x} tighter" if hi < 0 else (f"{y} tighter" if lo > 0 else "no separation")
        return f"{x} more actionable" if lo > 0 else (f"{y} more actionable" if hi < 0 else "no separation")

    boot = {"B": B, "seed": SEED, "n_clusters": len(clusters), "n_decisions": len(order), "pairs": {},
            "per_area_pairs": {}}
    for alpha in ALPHAS:
        a = str(alpha)
        for x, y in PAIRS:
            row = {}
            for field in ("size", "act", "act_holm", "act_m", "size_m"):
                d, (lo, hi) = pair(x, y, alpha, field)
                row[field] = {"diff": d, "ci95": [lo, hi], "verdict": verdict(x, y, field, lo, hi)}
            boot["pairs"].setdefault(f"{x}-{y}", {})[a] = row
        for x, y in PRIMARY_PAIRS:
            for ar in sorted(set(area_o)):
                d, (lo, hi) = pair(x, y, alpha, "act", area_o == ar)
                boot["per_area_pairs"].setdefault(f"{x}-{y}", {}).setdefault(a, {})[ar] = {
                    "diff": d, "ci95": [lo, hi], "verdict": verdict(x, y, "act", lo, hi)}
    split_rec = {"mode": mode, "salt": GF.SALT, "cal_rows": sum(is_cal.values()),
                 "test_rows": len(ids) - sum(is_cal.values()),
                 "test_decision_ids_sha256": G.ids_sha256(test_ids), "n_test_decisions": len(test_ids)}
    return {"split": split_rec, "group_plan": plan, "models": res, "bootstrap": boot}


def main():
    out_path = sys.argv[2]
    shas = {}
    for name, (p, want) in MODELS.items():
        got = GF.sha_file(p)
        assert got == want, (name, got)
        shas[name] = got
    ids, gold, cat, case = GF.load_gold()
    amap = area_of("0.2.1")
    print(f"gold: {len(gold)} decisions", file=sys.stderr, flush=True)
    out = {"prereg": "B-PREREG-GUAR-AREA (state 312222f8)", "agent_main": "7c131f1", "edition": "0.2.1",
           "epsilon": EPS, "delta": DELTA, "alphas": ALPHAS, "min_n": MIN_N, "input_sha256": shas,
           "n_rows": len(ids), "n_decisions": len(gold), "no_gold_decisions_excluded": dict(GF.NOGOLD)}
    for mode in ("run_id", "case"):
        out[mode] = run_split(mode, ids, gold, cat, case, amap)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

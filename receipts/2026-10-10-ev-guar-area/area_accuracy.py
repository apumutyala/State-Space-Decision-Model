"""Add-on to B-PREREG-GUAR-AREA (evidence coordinator request 2026-10-10; descriptive, not a pre-registered claim):
per-area top-1 accuracy and ECE on full DEV (0.2.1, gold decisions only) for D19, Clef-flash and Clef 27B (D13 too),
paired D19 - Clef-flash accuracy deltas per area, and, for knowledge and retrieval, the accuracy on the TEST decisions
each model ACTS on under its per-area LTT rule (q_hat_a, lambda_a read from GUARANTEES_AREA.json; recomputed here and
checked against the JSON's act fraction / acted-on error).

    python -X utf8 -I area_accuracy.py <agent-main @7c131f1 checkout> GUARANTEES_AREA.json AREA_ACCURACY.json

Same inputs, gold, splits and case clusters as guarantees_area.py (imports guarantees_fulldev for load_gold / split /
load_model). Top-1 = jevkit.answers.argmax_label over the decision's distribution (noul: {yes: p, no: 1 - p});
confidence = the top label's probability. ECE = harness.calibration.debiased_ece(conf, correct, bins=15) (debiased L2,
15 equal-mass bins; imported, not reimplemented); harness.calibration.binned_ece (10 equal-width bins, board L1) is
recorded alongside. CIs: case-cluster bootstrap (harness case ids), B = 2000, seed 0, multinomial cluster weights,
ratio estimator per area (the same construction as guarantees_area.py's paired contrasts).
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-10-B-guarantees-fulldev"))
import guarantees_fulldev as GF  # noqa: E402  (inserts argv[1] on sys.path)
from harness.calibration import binned_ece, debiased_ece  # noqa: E402
from harness.suite import area_of  # noqa: E402
from jevkit import guarantees as G  # noqa: E402
from jevkit.answers import argmax_label  # noqa: E402

MODELS = ("D19", "ClefFlash", "Clef27B", "D13")
ACT_AREAS = ("knowledge", "retrieval")
B, SEED = 2000, 0


def rid(d):
    return d.split("#", 1)[0]


def main():
    area_json, out_path = sys.argv[2], sys.argv[3]
    A = json.load(open(area_json, encoding="utf-8"))
    for name in MODELS:
        p, want = GF.MODELS[name]
        assert GF.sha_file(p) == want == A["input_sha256"][name], name
    ids, gold, cat, case = GF.load_gold()
    amap = area_of("0.2.1")
    area_of_rid = {r: amap.get(cat[r], "non-index") for r in ids}
    order = sorted(gold)
    areas_o = np.array([area_of_rid[rid(d)] for d in order])
    clusters = sorted({case[rid(d)] for d in order})
    cidx = {c: i for i, c in enumerate(clusters)}
    ci_of = np.array([cidx[case[rid(d)]] for d in order])
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=B).astype(np.float32)

    def csum(x):
        return np.bincount(ci_of, weights=x, minlength=len(clusters))

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    def boot_mean(x, mask):
        mk = mask.astype(float)
        return (W @ csum(x * mk)) / (W @ csum(mk))

    is_cal = GF.split(ids, case, "run_id")
    test_mask = np.array([not is_cal[rid(d)] for d in order])
    pos = {d: i for i, d in enumerate(order)}
    area_list = sorted(set(areas_o))
    out = {"what": "per-area top-1 accuracy / ECE on full DEV (gold decisions) + acted-on accuracy under the per-area "
                   "LTT rule (TEST, primary run_id split); descriptive add-on to B-PREREG-GUAR-AREA",
           "edition": "0.2.1", "n_decisions": len(order), "n_clusters": len(clusters), "B": B, "seed": SEED,
           "top1": "jevkit.answers.argmax_label; conf = top-label probability; noul as {yes: p, no: 1 - p}",
           "ece": "harness.calibration.debiased_ece(bins=15) (debiased L2, equal-mass); binned_ece 10-bin also given",
           "input_sha256": {m: A["input_sha256"][m] for m in MODELS}, "models": {}, "pairs": {}, "acted": {}}
    cols = {}
    for name in MODELS:
        sc, bad = GF.load_model(GF.MODELS[name][0], gold)
        valid = np.zeros(len(order), bool)
        corr = np.zeros(len(order))
        conf = np.zeros(len(order))
        for d, r in sc.items():
            i = pos[d]
            y = argmax_label(r.probs)
            valid[i] = True
            corr[i] = float(y == r.label)
            conf[i] = float(r.probs[y])
        cols[name] = (valid, corr, conf)
        per = {}
        for a in area_list + ["overall"]:
            m = valid & ((areas_o == a) if a != "overall" else np.ones(len(order), bool))
            n = int(m.sum())
            de = debiased_ece(conf[m], corr[m], bins=15)
            be, _ = binned_ece(conf[m], corr[m], bins=10)
            per[a] = {"n": n, "n_invalid_or_missing": int((~valid & ((areas_o == a) if a != "overall" else True)).sum()),
                      "acc": float(corr[m].mean()), "acc_ci95": ci(boot_mean(corr, m)),
                      "mean_conf": float(conf[m].mean()), "ece_debiased_l2_15eqmass": de["debiased_l2"],
                      "ece_debiased_sq": de["debiased_sq"], "ece_l1_10bin_equal_width": be}
        out["models"][name] = {"invalid_or_missing": bad, "per_area": per}

        # acted-on accuracy under the per-area LTT rule (TEST decisions, run_id split), knowledge + retrieval
        acted = {}
        for alpha in ("0.1", "0.05"):
            blk = A["run_id"]["models"][name]["alpha"][alpha]
            fit = {"qhat": {g: (float("inf") if q is None else q) for g, q in blk["area_fit"]["qhat"].items()},
                   "mondrian": True}
            mref = blk["marginal_reference"]
            fit_m = {"qhat": {g: (float("inf") if q is None else q) for g, q in mref["qhat"].items()}, "mondrian": True}
            for a in ACT_AREAS:
                pa = blk["per_area"][a]
                lam = pa["lambda_bonf"]
                n_t = n_act = n_err = n_actm = n_errm = 0
                act_v = np.zeros(len(order))
                err_v = np.zeros(len(order))
                for d, r in sc.items():
                    if is_cal[rid(d)] or area_of_rid[rid(d)] != a:
                        continue
                    n_t += 1
                    ra = G.Scored(d, a, r.probs, r.label)
                    if lam is not None:
                        ok, y = G._acts(ra, fit, lam)
                        if ok:
                            n_act += 1
                            n_err += y != r.label
                            act_v[pos[d]] = 1.0
                            err_v[pos[d]] = float(y != r.label)
                    if mref["lambda"] is not None:  # context: the marginal (pooled per-type) rule's acted set
                        ok_m, y_m = G._acts(r, fit_m, mref["lambda"])
                        if ok_m:
                            n_actm += 1
                            n_errm += y_m != r.label
                m_t = test_mask & (areas_o == a)
                acc_ci = None
                if n_act:
                    bs_n = W @ csum(act_v * m_t)
                    bs_c = W @ csum((act_v - err_v) * m_t)
                    ok_b = bs_n > 0
                    acc_ci = ci(bs_c[ok_b] / bs_n[ok_b])
                rec = {"lambda_area": lam, "qhat_area": blk["area_fit"]["qhat"].get(a), "n_test": n_t,
                       "n_acted": n_act, "act_fraction": n_act / n_t if n_t else None,
                       "acted_accuracy": (1 - n_err / n_act) if n_act else None, "acted_accuracy_ci95": acc_ci,
                       "test_accuracy_all": float(cols[name][1][m_t & cols[name][0]].mean()),
                       "json_act": pa["act"], "json_acted_err": pa["acted_err"],
                       "marginal_rule": {"lambda": mref["lambda"], "n_acted": n_actm,
                                         "act_fraction": n_actm / n_t if n_t else None,
                                         "acted_accuracy": (1 - n_errm / n_actm) if n_actm else None,
                                         "json_act_m": pa["act_m"], "json_acted_err_m": pa["acted_err_m"]}}
                assert abs(rec["act_fraction"] - pa["act"]) < 1e-12, (name, alpha, a, "act mismatch vs JSON")
                if n_act:
                    assert abs((n_err / n_act) - pa["acted_err"]) < 1e-12, (name, alpha, a)
                assert abs(rec["marginal_rule"]["act_fraction"] - pa["act_m"]) < 1e-12, (name, alpha, a, "act_m")
                acted.setdefault(alpha, {})[a] = rec
        out["acted"][name] = acted
        del sc
        print(f"{name} done", file=sys.stderr, flush=True)

    for x, y in (("D19", "ClefFlash"), ("D19", "Clef27B")):
        vx, cx, _ = cols[x]
        vy, cy, _ = cols[y]
        both = vx & vy
        rows = {}
        for a in area_list + ["overall"]:
            m = both & ((areas_o == a) if a != "overall" else np.ones(len(order), bool))
            d = boot_mean(cx, m) - boot_mean(cy, m)
            lo, hi = ci(d)
            rows[a] = {"n": int(m.sum()), "diff": float(cx[m].mean() - cy[m].mean()), "ci95": [lo, hi],
                       "verdict": f"{x} more accurate" if lo > 0 else (f"{y} more accurate" if hi < 0 else
                                                                        "no separation")}
        out["pairs"][f"{x}-{y}"] = rows
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

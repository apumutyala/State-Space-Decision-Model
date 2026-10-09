"""Readout-lever attribution for D13's losses vs the D1 v3w reference on proxy600 (DIAGNOSTIC; CPU only).

Run: python -X utf8 -I analysis/readout_levers_d13.py <repo_root> <child.results.jsonl> <ref.results.jsonl> <out.json>

Every "upper bound" here is measured on proxy600 itself (oracle fits use DEV gold), so it is a CEILING on what a readout
change could recover, never a claimable result. A lever only becomes claimable once its parameters are fit on non-DEV
data (or are parameter-free format rules) and the frozen screen is re-run. Frozen scorer: harness.score.decompose +
score.point (edition 0.2.1), unchanged; counterfactuals only rewrite the prediction dicts fed to it.
"""
import collections
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from harness import config as _config  # noqa: E402

try:
    import decision_index  # noqa: F401
except ImportError:
    sys.path.insert(1, str(_config.KIT_CLONE))
from harness import score  # noqa: E402
from harness.cli import load_slice  # noqa: E402
from harness.suite import spec  # noqa: E402
from jobs.common.dev_proxy import load_proxy  # noqa: E402

EDITION = "0.2.1"
LOSERS = [62, 21, 59, 25, 41, 37, 1, 33, 56, 5, 4]  # every benchmark with a negative contribution vs ref
EXTRA = [9]  # Home-Appliance: 0 in all arms (case-exact)
GRID = [x / 10 for x in range(-40, 41)]  # log-odds bias grid
TGRID = [x / 100 for x in range(2, 99)]  # noul threshold grid


def read_pred(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            j = json.loads(line)
            out[j["run_id"]] = j
    return out


prox = load_proxy("proxy600")
ids = set(prox["run_ids"])
rows = [r for r in load_slice("dev", EDITION) if r["_evaluation"]["run_id"] in ids]
assert len(rows) == 644 and prox["ids_sha256"].startswith("7d42c398")
child = {k: v for k, v in read_pred(sys.argv[2]).items() if k in ids}
ref = {k: v for k, v in read_pred(sys.argv[3]).items() if k in ids}
assert len(child) == len(ref) == 644
S = spec(None)
by_n = collections.defaultdict(list)
for r in rows:
    by_n[r["_evaluation"]["catalog_id"]].append(r)
fam = {n: rs[0]["family"] for n, rs in by_n.items()}
base_c, base_r = score.decompose(rows, child, edition=EDITION), score.decompose(rows, ref, edition=EDITION)
W = score.weights(set(base_c), True, EDITION)
pc, pr = score.point(base_c), score.point(base_r)


def bench_skill(n, res):
    b = score.decompose(by_n[n], res, edition=EDITION)[n]
    return float(b.value()[1][0])


def pts(n, skill_new, skill_old):
    return 100 * W[n] * (skill_new - skill_old)


def rid(r):
    return r["_evaluation"]["run_id"]


def scored_qs(n, r):
    """(key, q) pairs that enter the score."""
    from decision_index.scoring import added as added_mod
    keys = added_mod.scored_keys(n, r) if str(n) in S["added"] else list(r["questions"])
    return [(k, r["questions"][k]) for k in keys if r["expected"].get(k) is not None]



# ----------------------------------------------------------------------------- per-family diagnostics
def ece10(conf, hit):
    if not conf:
        return None
    bins = collections.defaultdict(list)
    for c, h in zip(conf, hit):
        bins[min(int(c * 10), 9)].append((c, h))
    return sum(len(v) / len(conf) * abs(sum(c for c, _ in v) / len(v) - sum(h for _, h in v) / len(v)) for v in bins.values())


def diag(n, res):
    st = collections.Counter(res[rid(r)]["status"] for r in by_n[n])
    gold_d, pred_d = collections.Counter(), collections.Counter()
    pg, conf, hit, rank2 = [], [], [], 0
    nq = wrong = 0
    for r in by_n[n]:
        x = res[rid(r)]
        if x["status"] != "ok":
            continue
        a = x["response"]["answers"]
        for k, q in scored_qs(n, r):
            g = r["expected"][k]
            nq += 1
            if q["type"] == "noul":
                p = a[k]["noul"]
                pred = p >= 0.5
                gp = p if g is True else 1 - p
                c = max(p, 1 - p)
                gold_d[str(g)] += 1
                pred_d[str(pred)] += 1
            else:
                P = a[k]["probabilities"]
                pred = a[k]["choice"]
                gp = P.get(g, 0.0)
                c = max(P.values())
                if n in (5, 4, 9):  # opaque/large label sets: no shared label identity
                    gold_d["-"] += 1
                    pred_d["-"] += 1
                else:
                    gold_d[g] += 1
                    pred_d[pred] += 1
                if pred != g and sorted(P.values(), reverse=True)[1] == gp:
                    rank2 += 1
            ok = pred == g
            wrong += not ok
            pg.append(gp)
            conf.append(c)
            hit.append(ok)
    pg_s = sorted(pg)
    return {"status": dict(st), "fields": nq, "acc": round(sum(hit) / max(nq, 1), 4), "wrong": wrong,
            "wrong_gold_rank2": rank2, "mean_p_gold": round(sum(pg) / max(len(pg), 1), 4),
            "median_p_gold": round(pg_s[len(pg_s) // 2], 4) if pg_s else None,
            "frac_p_gold_lt_0.2": round(sum(p < 0.2 for p in pg) / max(len(pg), 1), 4),
            "ece10_toplabel": None if not conf else round(ece10(conf, hit), 4),
            "mean_conf": round(sum(conf) / max(len(conf), 1), 4),
            "gold_dist": dict(gold_d), "pred_dist": dict(pred_d)}


# ----------------------------------------------------------------------------- counterfactual transforms
def with_answers(res, n, fn):
    """Copy of res with fn(row, answers) applied to every ok row of benchmark n (answers edited in place on a copy)."""
    out = dict(res)
    for r in by_n[n]:
        x = res[rid(r)]
        if x["status"] != "ok":
            continue
        y = copy.deepcopy(x)
        fn(r, y["response"]["answers"])
        out[rid(r)] = y
    return out


def apply_bias(n, res, bias, thr=0.5):
    """Additive log-prob bias per shared label (choice) or noul threshold. Opaque-label families untouched."""
    def fn(r, a):
        for k, q in scored_qs(n, r):
            if q["type"] == "noul":
                v = a[k]["noul"]
                # shift so that the threshold thr maps to 0.5 (monotone in v); only the >=0.5 decision matters for score
                a[k]["noul"] = 0.75 if v >= thr else 0.25
            else:
                P = a[k]["probabilities"]
                s = {o: math.log(max(p, 1e-9)) + bias.get(o, 0.0) for o, p in P.items()}
                a[k]["choice"] = max(sorted(s), key=s.__getitem__)
    return with_answers(res, n, fn)


def labels_of(n):
    labs = set()
    for r in by_n[n]:
        for k, q in scored_qs(n, r):
            if q["type"] == "choice":
                labs.update(q["criteria"])
    return sorted(labs)


def has_noul(n):
    return any(q["type"] == "noul" for r in by_n[n] for _, q in scored_qs(n, r))


def fit_oracle(n, res, subset=None):
    """Coordinate-ascent bias (choice) or threshold grid (noul) maximising benchmark skill on `subset` rows (default
    all proxy rows of n). DIAGNOSTIC: uses DEV gold."""
    rows_n = by_n[n] if subset is None else [r for r in by_n[n] if rid(r) in subset]

    def sk(res2):
        b = score.decompose(rows_n, res2, edition=EDITION)[n]
        return float(b.value()[1][0])

    if has_noul(n):
        best = (sk(res), 0.5)
        for t in TGRID:
            v = sk(apply_bias(n, res, {}, t))
            if v > best[0] + 1e-9 or (abs(v - best[0]) < 1e-9 and abs(t - 0.5) < abs(best[1] - 0.5)):
                best = (v, t)
        return {"thr": best[1]}, best[0]
    labs = labels_of(n)
    if len(labs) > 6:
        return {}, sk(res)
    bias = {o: 0.0 for o in labs}
    cur = sk(res)
    for _ in range(3):
        moved = False
        for o in labs[1:]:  # first label is the reference (bias is shift-invariant)
            best_v, best_b = cur, bias[o]
            for g in GRID:
                b2 = dict(bias, **{o: g})
                v = sk(apply_bias(n, res, b2))
                if v > best_v + 1e-9 or (abs(v - best_v) < 1e-9 and abs(g) < abs(best_b)):
                    best_v, best_b = v, g
            if best_b != bias[o]:
                bias[o], cur, moved = best_b, best_v, True
        if not moved:
            break
    return {"bias": bias}, cur


def apply_fit(n, res, fit):
    if "thr" in fit:
        return apply_bias(n, res, {}, fit["thr"])
    if "bias" in fit:
        return apply_bias(n, res, fit["bias"])
    return res



def crossfit(n, res, folds=2, reps=5):
    """Fit oracle on half the rows (by group), evaluate on the other half; mean skill gain over reps. Still DEV data:
    a less optimistic diagnostic of how much of the oracle gain is overfitting."""
    groups = sorted({str(r["_evaluation"]["group_id"]) for r in by_n[n]})
    gains = []
    for s in range(reps):
        rnd = random.Random(1000 + s)
        g2 = groups[:]
        rnd.shuffle(g2)
        half = set(g2[: len(g2) // 2])
        A = {rid(r) for r in by_n[n] if str(r["_evaluation"]["group_id"]) in half}
        B = {rid(r) for r in by_n[n]} - A
        for fit_on, ev_on in ((A, B), (B, A)):
            fit, _ = fit_oracle(n, res, fit_on)
            rows_e = [r for r in by_n[n] if rid(r) in ev_on]
            b0 = float(score.decompose(rows_e, res, edition=EDITION)[n].value()[1][0])
            b1 = float(score.decompose(rows_e, apply_fit(n, res, fit), edition=EDITION)[n].value()[1][0])
            gains.append(b1 - b0)
    return sum(gains) / len(gains)


# ----------------------------------------------------------------------------- format levers (parameter-free rules)
def oracle_exhausted(n, res):
    """UB for format/budget failures: non-ok rows of n become ok with GOLD answers."""
    out = dict(res)
    changed = 0
    for r in by_n[n]:
        x = res[rid(r)]
        if x["status"] == "ok":
            continue
        changed += 1
        ans = {}
        for k, q in r["questions"].items():
            g = r["expected"].get(k)
            if q["type"] == "noul":
                ans[k] = {"type": "noul", "noul": 0.9 if g is True else 0.1}
            else:
                crit = list(q["criteria"])
                gg = g if g in crit else crit[0]
                ans[k] = {"type": "choice", "choice": gg, "confidence": 1.0,
                          "probabilities": {o: float(o == gg) for o in crit}}
        out[rid(r)] = {"run_id": rid(r), "status": "ok", "response": {"answers": ans}}
    return out, changed


def yes_label(q):
    for o in q["criteria"]:
        if str(o).lower() in ("yes", "call", "true"):
            return o
    return None


def rule_at_least_one_yes(n, res):
    """Parameter-free multi-answer rule for per-option yes/no cases (SATA 'select all that apply', BFCL tool selection):
    if a case predicts no 'yes', flip the single highest-p(yes) field to yes. Groups = case."""
    out = dict(res)
    groups = collections.defaultdict(list)
    for r in by_n[n]:
        groups[str(r["_evaluation"]["group_id"])].append(r)
    flips = 0
    for g, rs in groups.items():
        if any(res[rid(r)]["status"] != "ok" for r in rs):
            continue
        best, any_yes = None, False
        for r in rs:
            a = res[rid(r)]["response"]["answers"]
            for k, q in r["questions"].items():
                if q["type"] != "choice":
                    continue
                y = yes_label(q)
                if y is None:
                    continue
                any_yes |= a[k]["choice"] == y
                p = a[k]["probabilities"][y]
                if best is None or p > best[0]:
                    best = (p, rid(r), k, y)
        if not any_yes and best:
            flips += 1
            _, rr, k, y = best
            x = copy.deepcopy(out[rr])
            x["response"]["answers"][k]["choice"] = y
            out[rr] = x
    return out, flips


def rule_topk_by_yes_mass(n, res):
    """Parameter-free SATA rule: per case, select the m options with highest p(yes), m = round(sum p(yes)) (>=1)."""
    out = dict(res)
    groups = collections.defaultdict(list)
    for r in by_n[n]:
        groups[str(r["_evaluation"]["group_id"])].append(r)
    for g, rs in groups.items():
        if any(res[rid(r)]["status"] != "ok" for r in rs):
            continue
        items = []
        for r in rs:
            a = res[rid(r)]["response"]["answers"]
            for k, q in r["questions"].items():
                y = yes_label(q) if q["type"] == "choice" else None
                if y is not None:
                    no = [o for o in q["criteria"] if o != y][0]
                    items.append((a[k]["probabilities"][y], rid(r), k, y, no))
        if not items:
            continue
        m = max(1, round(sum(i[0] for i in items)))
        keep = {(i[1], i[2]) for i in sorted(items, reverse=True)[:m]}
        for p, rr, k, y, no in items:
            if out[rr] is res[rr]:
                out[rr] = copy.deepcopy(res[rr])
            out[rr]["response"]["answers"][k]["choice"] = y if (rr, k) in keep else no
    return out


def case_exact_misses(n, res):
    """For case-exact families: per failed case, number of wrong fields (1 = near miss)."""
    groups = collections.defaultdict(list)
    for r in by_n[n]:
        groups[str(r["_evaluation"]["group_id"])].append(r)
    hist = collections.Counter()
    kinds = collections.Counter()
    for g, rs in groups.items():
        if any(res[rid(r)]["status"] != "ok" for r in rs):
            hist["not_ok"] += 1
            continue
        nw = 0
        for r in rs:
            a = res[rid(r)]["response"]["answers"]
            for k, q in r["questions"].items():
                gg = r["expected"].get(k)
                if gg is None:
                    continue
                pred = a[k]["choice"] if q["type"] == "choice" else a[k]["noul"] >= 0.5
                if pred != gg:
                    nw += 1
                    y = yes_label(q) if q["type"] == "choice" else None
                    if y is not None:
                        kinds["missed_yes" if gg == y else "false_yes"] += 1
        hist[nw] += 1
    return {"wrong_fields_per_case": {str(k): v for k, v in sorted(hist.items(), key=lambda kv: str(kv[0]))},
            "yesno_error_kinds": dict(kinds)}


# ----------------------------------------------------------------------------- run
out = {"note": "DIAGNOSTIC upper bounds measured on proxy600 (DEV gold). Not claimable.", "proxy_ids_sha256": prox["ids_sha256"],
       "rows": len(rows), "index_child": pc["index"], "index_ref": pr["index"], "families": {}}
for n in LOSERS + EXTRA:
    sc0, sr0 = pc["per_benchmark"][n]["skill"], pr["per_benchmark"][n]["skill"]
    f = {"family": fam[n], "catalog_id": n, "weight_pts_per_unit_skill": round(100 * W[n], 3),
         "kind": "track" if n in S["track_scored"] else ("added" if str(n) in S["added"] else "display"),
         "skill_child": sc0, "skill_ref": sr0, "loss_pts_vs_ref": round(pts(n, sc0, sr0), 3),
         "diag_child": diag(n, child), "diag_ref": diag(n, ref)}
    # (a) prior / threshold: oracle (in-sample) and 2-fold cross-fit, for child and for ref
    fit_c, sk_c = fit_oracle(n, child)
    fit_r, sk_r = fit_oracle(n, ref)
    f["prior_oracle_child"] = {"fit": fit_c, "skill": sk_c, "ub_pts_vs_child": round(pts(n, sk_c, sc0), 3)}
    f["prior_oracle_ref"] = {"fit": fit_r, "skill": sk_r}
    if fit_c and (fit_c.get("bias") or fit_c.get("thr") not in (None,)):
        f["prior_crossfit_child_pts"] = round(100 * W[n] * crossfit(n, child), 3)
    # child after oracle prior fix vs ref after its own oracle prior fix -> residual = capacity
    f["capacity_residual_pts"] = round(pts(n, sk_c, sk_r), 3)
    # (b) format: exhausted/unsupported rows
    res_x, nx = oracle_exhausted(n, child)
    f["format_nonok_rows_child"] = nx
    if nx:
        f["format_nonok_oracle_ub_pts"] = round(pts(n, bench_skill(n, res_x), sc0), 3)
    if n in (33, 1, 9):
        f["case_exact_child"] = case_exact_misses(n, child)
        f["case_exact_ref"] = case_exact_misses(n, ref)
        r1, flips = rule_at_least_one_yes(n, child)
        f["rule_at_least_one_yes"] = {"flips": flips, "pts_vs_child": round(pts(n, bench_skill(n, r1), sc0), 3)}
        if n == 33:
            f["rule_topk_by_yes_mass"] = {"pts_vs_child": round(pts(n, bench_skill(n, rule_topk_by_yes_mass(n, child)), sc0), 3)}
    out["families"][str(n)] = f
    print(n, fam[n], f["loss_pts_vs_ref"], "oracle", f["prior_oracle_child"]["ub_pts_vs_child"], fit_c,
          "xfit", f.get("prior_crossfit_child_pts"), "resid", f["capacity_residual_pts"],
          "nonok", nx, f.get("format_nonok_oracle_ub_pts"), f.get("rule_at_least_one_yes"), f.get("rule_topk_by_yes_mass"),
          flush=True)
    print("   child", f["diag_child"])
    print("   ref  ", f["diag_ref"])
    if "case_exact_child" in f:
        print("   case", f["case_exact_child"], f["case_exact_ref"])
# ----------------------------------------------------------------------------- extras: specific readout rules
X = out["extras"] = {}

# BFCL: in the simple / multiple / live_simple / live_multiple categories the BFCL spec has exactly ONE gold call per
# case (parallel* categories are the multi-call ones). Rule (parameter-free, from the category name in group_id):
# yes = argmax p(yes) in those categories. Not fit on anything.
SINGLE_CALL = {"simple", "multiple", "live_simple", "live_multiple"}


def rule_bfcl_exactly_one(res):
    out_ = dict(res)
    n_changed = 0
    for r in by_n[1]:
        cat = str(r["_evaluation"]["group_id"]).split(":")[1]
        if cat not in SINGLE_CALL or res[rid(r)]["status"] != "ok":
            continue
        y = copy.deepcopy(res[rid(r)])
        a = y["response"]["answers"]
        best = max(sorted(a), key=lambda k: a[k]["probabilities"]["yes"])
        for k in a:
            new = "yes" if k == best else "no"
            n_changed += a[k]["choice"] != new
            a[k]["choice"] = new
        out_[rid(r)] = y
    return out_, n_changed


cats = collections.Counter(str(r["_evaluation"]["group_id"]).split(":")[1] for r in by_n[1])
for nm, R in (("child", child), ("ref", ref)):
    r1, ch_ = rule_bfcl_exactly_one(R)
    X[f"bfcl_exactly_one_{nm}"] = {"fields_changed": ch_, "skill": bench_skill(1, r1),
                                   "pts_vs_child": round(pts(1, bench_skill(1, r1), pc["per_benchmark"][1]["skill"]), 3)}
X["bfcl_categories_in_proxy"] = dict(cats)

# SATA: same parameter-free top-m-by-yes-mass rule on ref, for symmetry
X["sata_topk_ref_skill"] = bench_skill(33, rule_topk_by_yes_mass(33, ref))
X["sata_topk_child_skill"] = bench_skill(33, rule_topk_by_yes_mass(33, child))

# RAGTruth: F1-on-hallucinated-class is maximised, for a calibrated scorer, at threshold F*/2 (Lipton et al. 2014),
# and F* >= the all-positive F1 = 2*pi/(1+pi). pi from the RAGTruth TRAIN split (non-DEV): 6721/15090 = 0.4454
# (ssdm-data/m2/sources/dl-2026-10-08/ragtruth/response.jsonl, split == "train", labels non-empty).
PI_TRAIN = 6721 / 15090
T_F1 = (2 * PI_TRAIN / (1 + PI_TRAIN)) / 2
X["ragtruth_pi_train"] = round(PI_TRAIN, 4)
X["ragtruth_t_f1_rule"] = round(T_F1, 4)
curve = {}
for nm, R in (("child", child), ("ref", ref)):
    curve[nm] = {f"{t:.2f}": round(bench_skill(59, apply_bias(59, R, {}, t)), 4) for t in [0.01, 0.2, 0.25, round(T_F1, 4), 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]}
X["ragtruth_skill_by_threshold"] = curve
X["ragtruth_t_f1_rule_pts_vs_child"] = round(pts(59, bench_skill(59, apply_bias(59, child, {}, T_F1)), pc["per_benchmark"][59]["skill"]), 3)
X["ragtruth_child_noul_quantiles"] = sorted(round(child[rid(r)]["response"]["answers"]["q"]["noul"], 3) for r in by_n[59])
X["ragtruth_ref_noul_quantiles"] = sorted(round(ref[rid(r)]["response"]["answers"]["q"]["noul"], 3) for r in by_n[59])

# Exhausted rows (ToolRet 1, BRIGHT 5; identical rows in child and ref): oracle UB if they were answered perfectly.
for n in (2, 36):
    rx, nx = oracle_exhausted(n, child)
    X[f"exhausted_oracle_{n}"] = {"family": fam[n], "rows": nx,
                                  "ub_pts_vs_child": round(pts(n, bench_skill(n, rx), pc["per_benchmark"][n]["skill"]), 3)}

# Home-Appliance (case-exact, 0/12 everywhere): which roles fail, and oracle UBs by role class
def ha_role(r, k):
    role = r["metadata"]["question_roles"][k]
    return "target_member" if role.startswith("target_member") else role


ha_err = collections.Counter()
ha_n = collections.Counter()
yn = collections.Counter()
res_pred = collections.Counter()
for r in by_n[9]:
    a = child[rid(r)]["response"]["answers"]
    for k, q in r["questions"].items():
        ro = ha_role(r, k)
        g, p = r["expected"][k], a[k]["choice"]
        ha_n[ro] += 1
        ha_err[ro] += g != p
        if ro == "target_member":
            yn[(q["criteria"][g].split(",")[0], q["criteria"][p].split(",")[0])] += 1
        if ro == "resolution":
            res_pred[(q["criteria"][g], q["criteria"][p])] += 1
X["home_appliance_field_errors_by_role"] = {ro: [ha_err[ro], ha_n[ro]] for ro in ha_n}
X["home_appliance_target_member_gold_pred"] = {f"{g}->{p}": v for (g, p), v in yn.items()}
X["home_appliance_resolution_gold_pred"] = {f"{g} -> {p}": v for (g, p), v in res_pred.items()}


def ha_oracle(roles):
    def fn(r, a):
        for k, q in r["questions"].items():
            if ha_role(r, k) in roles:
                a[k]["choice"] = r["expected"][k]
    return with_answers(child, 9, fn)


def ha_yes_bias(res, b):
    """Semantic bias on target_member yes/no (option description starts 'yes,'/'no,'); ids are opaque per row."""
    def fn(r, a):
        for k, q in r["questions"].items():
            if ha_role(r, k) == "target_member":
                P = a[k]["probabilities"]
                s = {o: math.log(max(P[o], 1e-9)) + (b if q["criteria"][o].startswith("yes") else 0.0) for o in P}
                a[k]["choice"] = max(sorted(s), key=s.__getitem__)
    return with_answers(res, 9, fn)


ha = {}
best = max(((bench_skill(9, ha_yes_bias(child, b)), -abs(b), b) for b in GRID))
ha["oracle_target_yes_bias"] = {"bias_yes": best[2], "skill": best[0],
                                "ub_pts_vs_child": round(pts(9, best[0], pc["per_benchmark"][9]["skill"]), 3)}
for nm, roles in (("oracle_resolution+outcome", {"resolution", "outcome"}),
                  ("oracle_all_non_target", {"resolution", "outcome", "state_read", "post_0", "post_1"}),
                  ("oracle_target_member_only", {"target_member"})):
    rr = ha_oracle(roles)
    sk = bench_skill(9, rr)
    ha[nm] = {"skill": sk, "ub_pts_vs_child": round(pts(9, sk, pc["per_benchmark"][9]["skill"]), 3)}
X["home_appliance_oracles"] = ha

# Discrimination (rank-only, prior/threshold-free): AUROC of p(positive) for the binary losing families. A family whose
# AUROC is ~0.5 has nothing a bias/threshold can legitimately recover beyond the metric's degenerate all-one-class point.
def auroc(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return None
    w = sum((p > q) + 0.5 * (p == q) for p in pos_scores for q in neg_scores)
    return round(w / (len(pos_scores) * len(neg_scores)), 4)


def bin_scores(n, res, positive):
    pos, neg = [], []
    for r in by_n[n]:
        a = res[rid(r)]["response"]["answers"]
        for k, q in scored_qs(n, r):
            g = r["expected"][k]
            s = a[k]["noul"] if q["type"] == "noul" else a[k]["probabilities"][positive]
            (pos if g == (True if q["type"] == "noul" else positive) else neg).append(s)
    return pos, neg


X["auroc"] = {}
for n, positive in ((59, True), (56, "phishing"), (21, "headline_2")):
    X["auroc"][fam[n]] = {nm: auroc(*bin_scores(n, R, positive)) for nm, R in (("child", child), ("ref", ref))}
# When2Call: one-vs-rest AUROC of the gold class, averaged over classes present
w2c = {}
for nm, R in (("child", child), ("ref", ref)):
    vals = []
    for c in ("B", "C", "D"):
        pos, neg = [], []
        for r in by_n[62]:
            p = R[rid(r)]["response"]["answers"]["q"]["probabilities"][c]
            (pos if r["expected"]["q"] == c else neg).append(p)
        vals.append(auroc(pos, neg))
    w2c[nm] = {"B_tool_call": vals[0], "C_request_for_info": vals[1], "D_cannot_answer": vals[2]}
X["auroc"]["When2Call MCQ (one-vs-rest)"] = w2c

# BRIGHT/ToolRet exhausted rows: realistic gain = the child's own mean per-query value on its ok queries
for n in (2, 36):
    b = base_c[n]
    T = b.S
    okq = T[:, 3] > 0
    X[f"exhausted_realistic_{n}"] = {"child_mean_query_value_ok": round(float(T[okq, 0].sum() / max(okq.sum(), 1)), 4),
                                     "queries": int(T[:, 2].sum()), "ok_queries": int(T[:, 3].sum())}
    X[f"exhausted_realistic_{n}"]["note"] = ("realistic gain ~= UB * child_mean_query_value_ok (exhausted queries are "
                                             "scored 0, a lifted cap would score them like the rest)")

# RAGTruth polarity diagnostic: score with noul -> 1 - noul (DIAGNOSTIC; a flip may only ship if a non-DEV check,
# e.g. the RAGTruth train split, shows the same inversion).
def invert_noul(res, n):
    def fn(r, a):
        for k, q in scored_qs(n, r):
            if q["type"] == "noul":
                a[k]["noul"] = 1 - a[k]["noul"]
    return with_answers(res, n, fn)


X["ragtruth_polarity_flip_diag"] = {nm: {"skill": bench_skill(59, invert_noul(R, 59)),
                                         "pts_vs_child": round(pts(59, bench_skill(59, invert_noul(R, 59)), pc["per_benchmark"][59]["skill"]), 3)}
                                    for nm, R in (("child", child), ("ref", ref))}

# Totals over the losing families
F = out["families"]
lose = [k for k in F if F[k]["loss_pts_vs_ref"] < 0]
out["totals"] = {
    "loss_pts_sum": round(sum(F[k]["loss_pts_vs_ref"] for k in lose), 3),
    "prior_oracle_ub_pts_sum": round(sum(F[k]["prior_oracle_child"]["ub_pts_vs_child"] for k in lose), 3),
    "prior_oracle_ub_capped_at_loss_sum": round(sum(min(F[k]["prior_oracle_child"]["ub_pts_vs_child"], -F[k]["loss_pts_vs_ref"]) for k in lose), 3),
    "prior_crossfit_pts_sum": round(sum(F[k].get("prior_crossfit_child_pts") or 0.0 for k in lose), 3),
    "param_free_rules_pts": {"bfcl_exactly_one": X["bfcl_exactly_one_child"]["pts_vs_child"],
                             "sata_topm_yes_mass": F["33"]["rule_topk_by_yes_mass"]["pts_vs_child"]},
    "input_cap_oracle_ub_pts": round(X["exhausted_oracle_2"]["ub_pts_vs_child"] + X["exhausted_oracle_36"]["ub_pts_vs_child"], 3),
    "input_cap_realistic_pts": round(X["exhausted_oracle_2"]["ub_pts_vs_child"] * X["exhausted_realistic_2"]["child_mean_query_value_ok"]
                                     + X["exhausted_oracle_36"]["ub_pts_vs_child"] * X["exhausted_realistic_36"]["child_mean_query_value_ok"], 3),
}
print("totals", out["totals"])
print("ragtruth flip", X["ragtruth_polarity_flip_diag"])

# exhausted rows across the whole proxy (which families)
out["nonok_rows"] = {k: [(rid(r), fam[r["_evaluation"]["catalog_id"]], R[rid(r)]["status"]) for r in rows if R[rid(r)]["status"] != "ok"]
                     for k, R in (("child", child), ("ref", ref))}
Path(sys.argv[4]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("nonok", out["nonok_rows"])

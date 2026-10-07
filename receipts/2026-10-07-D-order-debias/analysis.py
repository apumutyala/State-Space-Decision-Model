"""Free (CPU, no new model calls) order-debias estimates on the stock RWKV7-13.3B parent arm (job-uhxb5).

usage: python -X utf8 -I analysis.py <repo> <parent_arm_dir> <index_rows.jsonl.gz> <out.json>
Inputs are untrusted JSON (read only). Index + paired cluster bootstrap come from the repo's harness (B=2000, seed=0).
"""
import collections
import copy
import gzip
import hashlib
import json
import sys

import numpy as np

repo, arm, rows_p, out_p = sys.argv[1:5]
sys.path.insert(0, repo)
from harness import bootstrap as HB, calibration as HC, score as S  # noqa: E402

B, SEED = 2000, 0
MAX_K = 10  # per-position priors only for k <= 10 (sparser k would be fit on a single benchmark)

ix = json.load(open(f"{arm}/units-index.json", encoding="utf-8"))
res = ix["index_results"]
with gzip.open(rows_p, "rt", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f]
assert len(rows) == 2008 and set(r["_evaluation"]["run_id"] for r in rows) == set(res)
out = {"n_rows": len(rows), "status": dict(collections.Counter(v["status"] for v in res.values()))}


def choice_qs(row):
    for k, q in sorted(row["questions"].items()):
        if q.get("type") == "choice" and len(q["criteria"]) >= 2:
            yield k, sorted(str(o) for o in q["criteria"])


# ---- 1. forward-only (reproduce) ----------------------------------------------------------------------------------
base_dp = S.decompose(rows, res)


def summarize(name, new_res):
    dp = S.decompose(rows, new_res)
    cmp = HB.compare(dp, base_dp, B=B, seed=SEED)
    ece = HC.report(HC.fields(rows, new_res))
    return {
        "index": cmp["candidate"]["index"], "areas": cmp["candidate"]["areas"],
        "diff_vs_forward": {"index": cmp["diff"]["index"], "areas": cmp["diff"]["areas"]},
        "ece": {"board_protocol": ece["board_protocol"]["ece"], "overall_10bin": ece["overall"]["ece_10bin"],
                "overall_10bin_benchmark_equal": ece["overall"]["ece_10bin_benchmark_equal"],
                "debiased_l2": ece["overall"]["debiased_l2"], "smece": ece["overall"]["smece"],
                "acc": ece["overall"]["acc"], "conf": ece["overall"]["conf"]},
    }


fwd = summarize("forward", res)
out["forward"] = fwd

# ---- 2. label-free per-position prior correction, cross-fitted (never fit on the rows it scores) -------------------
def fold_cv(row):
    return int(hashlib.sha256(f"order-debias-v1:{row['_evaluation']['catalog_id']}:{row['_evaluation']['group_id']}"
                              .encode()).hexdigest(), 16) % 5


def fold_lobo(row):
    return row["_evaluation"]["catalog_id"]


def priors(rows_fit):
    acc = collections.defaultdict(lambda: None)
    cnt = collections.Counter()
    for row in rows_fit:
        r = res[row["_evaluation"]["run_id"]]
        if r["status"] != "ok":
            continue
        for key, ids in choice_qs(row):
            k = len(ids)
            if k > MAX_K:
                continue
            p = r["response"]["answers"][key]["probabilities"]
            v = np.array([p[o] for o in ids], dtype=float)
            v = v / v.sum()
            acc[k] = v if acc[k] is None else acc[k] + v
            cnt[k] += 1
    return {k: acc[k] / cnt[k] for k in cnt}, dict(cnt)


def corrected(fold_fn):
    folds = collections.defaultdict(list)
    for row in rows:
        folds[fold_fn(row)].append(row)
    new = copy.deepcopy(res)
    n_changed = n_q = 0
    fold_priors = {}
    for f, rs in folds.items():
        fit = [row for g, rr in folds.items() if g != f for row in rr]
        pri, cnt = priors(fit)
        fold_priors[str(f)] = {str(k): {"n": cnt[k], "prior": [round(float(x), 4) for x in pri[k]]} for k in sorted(pri)}
        for row in rs:
            r = new[row["_evaluation"]["run_id"]]
            if r["status"] != "ok":
                continue
            for key, ids in choice_qs(row):
                k = len(ids)
                if k not in pri:
                    continue
                a = r["response"]["answers"][key]
                v = np.array([a["probabilities"][o] for o in ids], dtype=float)
                w = v / np.maximum(pri[k], 1e-6)
                w = w / w.sum()
                ch = ids[int(np.argmax(w))]
                n_q += 1
                n_changed += ch != a["choice"]
                a["probabilities"] = {o: float(x) for o, x in zip(ids, w)}
                a["choice"] = ch
                a["confidence"] = float(w.max())
    return new, {"questions_corrected": n_q, "choices_changed": n_changed}, fold_priors


for name, fn in (("calib_crossfit5", fold_cv), ("calib_leave_one_benchmark_out", fold_lobo)):
    new, info, fp = corrected(fn)
    s = summarize(name, new)
    s.update(info)
    if name == "calib_crossfit5":
        s["fold_priors_k2_k3_k4"] = {f: {k: v for k, v in p.items() if k in ("2", "3", "4")} for f, p in fp.items()}
    out[name] = s

# ---- 3. order diagnostic (160 rows, 482 questions; choices only, both orders relabelled) ---------------------------
od = ix["order_diag"]["questions"]
by_k = collections.defaultdict(lambda: collections.Counter())
rowsq = collections.defaultdict(list)
for qid, v in od.items():
    k = v["k"]
    kb = str(k) if k <= 4 else ("5-16" if k <= 16 else ">16")
    f, r = v["forward"], v["reversed"]
    c = by_k[kb]
    c["n"] += 1
    c["fwd_correct"] += f["correct"]
    c["rev_correct"] += r["correct"]
    c["both"] += f["correct"] and r["correct"]
    c["either"] += f["correct"] or r["correct"]
    c["agree"] += f["choice"] is not None and f["choice"] == r["choice"]
    c["fwd_presented_first"] += f["chose_presented_first"]
    c["rev_presented_first"] += r["chose_presented_first"]
    c["inv_k"] += 1 / k
    rowsq[qid.rsplit("/", 1)[0]].append(v)
tot = collections.Counter()
for c in by_k.values():
    tot.update(c)
by_k["all"] = tot


def fmt(c):
    n = c["n"]
    pf, pr = c["fwd_presented_first"] / n, c["rev_presented_first"] / n
    u = c["inv_k"] / n
    return {"n": n, "acc_fwd": c["fwd_correct"] / n, "acc_rev": c["rev_correct"] / n, "both": c["both"] / n,
            "either": c["either"] / n, "agreement": c["agree"] / n,
            "p_presented_first_fwd": pf, "p_presented_first_rev": pr, "uniform_first": u,
            # position effect: choosing whatever is shown first, averaged over the two orders, minus chance
            "position_effect": (pf + pr) / 2 - u}


out["order_diag"] = {k: fmt(c) for k, c in sorted(by_k.items())}
# exact bounds on the 2-order probability-average accuracy for k=2 questions: [both, either]
# row-cluster bootstrap of rev - fwd accuracy on the 482 questions
keys = sorted(rowsq)
rng = np.random.default_rng(SEED)
d = np.array([sum(v["reversed"]["correct"] - v["forward"]["correct"] for v in rowsq[r]) for r in keys], float)
n = np.array([len(rowsq[r]) for r in keys], float)
draws = []
for _ in range(B):
    w = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
    draws.append((w * d).sum() / (w * n).sum())
out["order_diag_rev_minus_fwd_acc"] = {"point": float(d.sum() / n.sum()), "lo": float(np.percentile(draws, 2.5)),
                                       "hi": float(np.percentile(draws, 97.5)), "B": B, "seed": SEED,
                                       "clusters": len(keys)}
json.dump(out, open(out_p, "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps({k: (v["index"], v["diff_vs_forward"]["index"], v["ece"]) if isinstance(v, dict) and "index" in v
                  else v for k, v in out.items() if k != "order_diag"}, indent=1, default=float)[:6000])
print(json.dumps(out["order_diag"], indent=1))

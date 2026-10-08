"""B gap breakdown: D1 (trained, art-bm3e6) and v3w (best stock readout, art-ivu28) vs Clef-flash (job-sgihj),
per area / benchmark ("family") / question type, plus the lever evidence (readout spread, rank-2, cross-fitted
prior, training lift, length). Frozen scorer only called (harness.score.decompose / point, harness.bootstrap).

Run (CPU, from the worktree root):  python -X utf8 -I analysis/gap_b/gap_b.py analysis/gap_b/out/gap_b.json
"""
import collections
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FROZEN, SOURCES, index_of, load_results, load_rows, score  # noqa: E402
from harness import bootstrap  # noqa: E402
from harness.suite import spec  # noqa: E402

out_p = Path(sys.argv[1])
rows = load_rows()
rowmap = {r["_evaluation"]["run_id"]: r for r in rows}
cat_of = {rid: r["_evaluation"]["catalog_id"] for rid, r in rowmap.items()}
name = {r["_evaluation"]["catalog_id"]: r["_evaluation"]["dataset"] for r in rows}
S = spec("0.2.1")
AREA_OF = {n: a["id"] for a in S["areas"] for n in a["benchmarks"]}
R = {k: load_results(p) for k, p in SOURCES.items()}
R["clef"] = {rid: R["clef"][rid] for rid in rowmap}
B = {k: score.decompose(rows, v) for k, v in R.items()}
P = {k: score.point(b) for k, b in B.items()}
W = score.weights(set(B["d1"]))
rep = {"index": {k: P[k]["index"] for k in P}, "weights": {str(n): W[n] for n in W}}


def pts(sysname, n):
    return 100 * W[n] * P[sysname]["per_benchmark"][n]["skill"]


def qtype(q):
    if q["type"] != "choice":
        return "binary"
    k = len(q["criteria"])
    return "k2" if k == 2 else "k3-16" if k <= 16 else "k>16"


def transplant_q(base, donor, pick):
    """Copy of base where answers of questions pick(rid, key, q) are replaced by donor's (both ok), whole row if base not ok."""
    res = dict(base)
    for rid, r in rowmap.items():
        keys = [k for k, q in r["questions"].items() if r["expected"][k] is not None and pick(rid, k, q)]
        if not keys:
            continue
        b, d = base.get(rid), donor.get(rid)
        if d is None or d["status"] != "ok":
            continue
        if b is None or b["status"] != "ok":
            res[rid] = d
            continue
        x = copy.deepcopy(b)
        for k in keys:
            x["response"]["answers"][k] = d["response"]["answers"][k]
        res[rid] = x
    return res


# ---------------------------------------------------------------- 1. per-benchmark gap (additive) + paired bootstrap SE
reps = bootstrap.replicate_skills([B["d1"], B["v3w"], B["clef"], B["v1"], B["d1_parent"]], B=1000, seed=0)
bench = {}
for n in sorted(B["d1"]):
    sk = {k: round(100 * P[k]["per_benchmark"][n]["skill"], 2) for k in P}
    gd = 100 * W[n] * (reps[2][n][1] - reps[0][n][1])
    gv = 100 * W[n] * (reps[2][n][1] - reps[1][n][1])
    lift = 100 * W[n] * (reps[0][n][1] - reps[4][n][1])
    bench[n] = dict(name=name[n], area=AREA_OF[n], frozen=n in FROZEN, w=W[n], skill=sk,
                    gap_d1=float(gd[0]), gap_d1_se=float(gd[1:].std()), gap_v3w=float(gv[0]), gap_v3w_se=float(gv[1:].std()),
                    train_lift=float(lift[0]), train_lift_se=float(lift[1:].std()),
                    readout={v: pts(v, n) - pts("v1", n) for v in ("v2", "v2f", "v3", "v3c", "v3p", "v3r", "v3w")})
    bench[n]["readout_best"] = max(bench[n]["readout"].items(), key=lambda kv: kv[1])
tot_d1 = P["clef"]["index"] - P["d1"]["index"]
tot_v3w = P["clef"]["index"] - P["v3w"]["index"]
assert abs(sum(b["gap_d1"] for b in bench.values()) - tot_d1) < 1e-6
assert abs(sum(b["gap_v3w"] for b in bench.values()) - tot_v3w) < 1e-6
rep["total_gap"] = {"d1": tot_d1, "v3w": tot_v3w}
# spot-check additivity against a real transplant through the frozen scorer
for n in (3, 43, 62):
    t = index_of(rows, transplant_q(R["d1"], R["clef"], lambda rid, k, q, n=n: cat_of[rid] == n))["index"] - P["d1"]["index"]
    assert abs(t - bench[n]["gap_d1"]) < 1e-6, (n, t, bench[n]["gap_d1"])

# ---------------------------------------------------------------- 2. per question type (answer-level transplant)
qt = {}
for cand in ("d1", "v3w"):
    base = P[cand]["index"]
    for t in ("binary", "k2", "k3-16", "k>16"):
        for area in sorted({a for a in AREA_OF.values()}):
            g = index_of(rows, transplant_q(R[cand], R["clef"], lambda rid, k, q, t=t, area=area:
                                            qtype(q) == t and AREA_OF[cat_of[rid]] == area))["index"] - base
            if abs(g) > 1e-9:
                qt[f"{cand}|{area}|{t}"] = g
        qt[f"{cand}|all|{t}"] = index_of(rows, transplant_q(R[cand], R["clef"], lambda rid, k, q, t=t: qtype(q) == t))["index"] - base
    # mixed-type benchmarks: per type
    for n in bench:
        types = {qtype(q) for r in rows if cat_of[r["_evaluation"]["run_id"]] == n for k, q in r["questions"].items() if r["expected"][k] is not None}
        if len(types) > 1:
            for t in types:
                qt[f"{cand}|bench{n}|{t}"] = index_of(rows, transplant_q(R[cand], R["clef"], lambda rid, k, q, t=t, n=n:
                                                                       cat_of[rid] == n and qtype(q) == t))["index"] - base
rep["qtype_transplant"] = qt


# ---------------------------------------------------------------- 3. question-level view: accuracy, rank-2 among wrong, near-miss
def qview(q, a, gold):
    if q["type"] == "choice":
        opts = list(q["criteria"])
        probs = a.get("probabilities") or {}
        pv = [float(probs.get(o, 0.0)) for o in opts]
        s = sum(pv) or 1.0
        pv = [x / s for x in pv]
        order = sorted(range(len(opts)), key=lambda i: -pv[i])
        rank = order.index(opts.index(gold)) + 1 if gold in opts else None
        return dict(correct=a["choice"] == gold, rank=rank, pg=pv[opts.index(gold)] if gold in opts else 0.0, k=len(opts))
    py = float(a["noul"])
    pg = py if gold else 1 - py
    return dict(correct=(py >= 0.5) == gold, rank=1 if pg >= 0.5 else 2, pg=pg, k=2)


qstats = {}
for m in ("d1", "v3w", "v1", "clef"):
    c = collections.defaultdict(collections.Counter)
    for rid, r in rowmap.items():
        x = R[m].get(rid)
        n = cat_of[rid]
        for k, q in r["questions"].items():
            gold = r["expected"][k]
            if gold is None:
                continue
            c[n]["q"] += 1
            if x is None or x["status"] != "ok":
                c[n]["noresult"] += 1
                continue
            v = qview(q, x["response"]["answers"][k], gold)
            c[n]["correct"] += v["correct"]
            if not v["correct"]:
                c[n]["wrong"] += 1
                if v["k"] > 2:
                    c[n]["wrong_k3"] += 1
                    c[n]["wrong_k3_rank2"] += v["rank"] == 2
                else:
                    c[n]["wrong_k2"] += 1
                    c[n]["wrong_k2_near"] += v["pg"] >= 0.4
    qstats[m] = {str(n): dict(v) for n, v in c.items()}
rep["qstats"] = qstats


# ---------------------------------------------------------------- 4. cross-fitted per-benchmark prior correction (DEV-fit diagnostic)
def fold(gid):
    return int(hashlib.sha256(str(gid).encode()).hexdigest(), 16) % 5


def prior_corrected(res, keymode):
    """5-fold cross-fit by kit group: divide each option's prob by its mean prob on the other folds (per benchmark x
    option key), re-argmax. keymode 'label' (option id) or 'pos' (listed position). Binary: divide [p, 1-p] by mean."""
    acc = collections.defaultdict(lambda: np.zeros(2))  # (n, fold, key) -> [sum p, count]
    items = []
    for rid, r in rowmap.items():
        x = res.get(rid)
        if x is None or x["status"] != "ok":
            continue
        n, f = cat_of[rid], fold(r["_evaluation"]["group_id"])
        for k, q in r["questions"].items():
            if r["expected"][k] is None:
                continue
            a = x["response"]["answers"][k]
            if q["type"] == "choice":
                opts = list(q["criteria"])
                pr = a.get("probabilities") or {}
                pv = np.array([float(pr.get(o, 0.0)) for o in opts]); pv = pv / (pv.sum() or 1.0)
                keys = [(len(opts), o if keymode == "label" else i) for i, o in enumerate(opts)]
            else:
                pv = np.array([float(a["noul"]), 1 - float(a["noul"])]); keys = [("bin", 1), ("bin", 0)]
            for kk, p in zip(keys, pv):
                acc[(n, f, kk)] += (p, 1)
            items.append((rid, k, q, n, f, keys, pv))
    tot = collections.defaultdict(lambda: np.zeros(2))
    for (n, f, kk), v in acc.items():
        tot[(n, kk)] += v
    out = {rid: copy.deepcopy(x) for rid, x in res.items() if rid in rowmap}
    for rid, k, q, n, f, keys, pv in items:
        pri = []
        for kk in keys:
            s, c = tot[(n, kk)] - acc.get((n, f, kk), np.zeros(2))
            pri.append(s / c if c > 0 else 1.0 / len(keys))
        cp = pv / np.maximum(np.array(pri), 1e-6); cp = cp / (cp.sum() or 1.0)
        a = out[rid]["response"]["answers"][k]
        if q["type"] == "choice":
            opts = list(q["criteria"])
            a["probabilities"] = {o: float(p) for o, p in zip(opts, cp)}
            a["choice"] = opts[int(np.argmax(cp))]
            a["confidence"] = float(cp.max())
        else:
            a["noul"] = float(cp[0])
    return out


prior = {}
for m in ("d1", "v3w"):
    for mode in ("label", "pos"):
        pc = index_of(rows, prior_corrected(R[m], mode))
        prior[f"{m}|{mode}|index"] = pc["index"] - P[m]["index"]
        for n in bench:
            prior[f"{m}|{mode}|{n}"] = 100 * W[n] * (pc["per_benchmark"][n]["skill"] - P[m]["per_benchmark"][n]["skill"])
rep["prior_xfit"] = prior

# ---------------------------------------------------------------- 5. length bins (Clef-flash transplant on rows by proxy tokens)
ln = {}
for cand in ("d1", "v3w"):
    for lo, hi in ((0, 2048), (2048, 8192), (8192, 16384), (16384, 10 ** 9)):
        sel = lambda rid, k, q, lo=lo, hi=hi: lo <= rowmap[rid]["_evaluation"].get("proxy_tokens", 0) < hi
        ln[f"{cand}|{lo}-{hi}"] = index_of(rows, transplant_q(R[cand], R["clef"], sel))["index"] - P[cand]["index"]
ln["rows"] = {f"{lo}-{hi}": sum(lo <= r["_evaluation"].get("proxy_tokens", 0) < hi for r in rows)
              for lo, hi in ((0, 2048), (2048, 8192), (8192, 16384), (16384, 10 ** 9))}
ln["not_ok_rows"] = {m: sorted((cat_of[rid], rowmap[rid]["_evaluation"].get("proxy_tokens", 0)) for rid in rowmap
                               if R[m].get(rid) is None or R[m][rid]["status"] != "ok") for m in ("d1", "v3w", "v2f", "clef")}
rep["length"] = ln

# ---------------------------------------------------------------- 6. area totals and frozen totals
area = collections.defaultdict(lambda: collections.Counter())
for n, b in bench.items():
    for key in ("gap_d1", "gap_v3w", "train_lift"):
        area[b["area"]][key] += b[key]
        area["frozen" if b["frozen"] else "nonfrozen"][key] += b[key]
rep["area"] = {a: dict(v) for a, v in area.items()}
rep["bench"] = {str(n): b for n, b in bench.items()}
out_p.parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(out_p, "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps({"index": rep["index"], "total_gap": rep["total_gap"], "area": rep["area"]}, indent=1, default=float))

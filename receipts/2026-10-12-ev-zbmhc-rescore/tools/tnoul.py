"""T_noul audit (A-PREREG-JEV3-LM: "T_noul is chosen to minimise slice NLL subject to slice yes/no band rate <=
stock's + 5pp"). Independent re-implementation (not the job's screen.py):
 (A) slice: band rate (p(true) strictly in (0.2, 0.8)) and NLL of the trained arm's noul decisions over T grids;
     stock band from slice_floor_rows.jsonl stock_p_shim; feasibility of band <= stock + 0.05; the in-job 400-point
     log grid on [0.25, 4] with the job's fallback tie-break (lowest band, then lowest NLL) re-run.
 (B) public: rebuild native_tfit from native.preds.jsonl + tfit.json per_type (p^(1/T) renormalised; jevkit
     normalize_answer for derived fields), diff against the in-job native_tfit preds, then write the DIAGNOSTIC arm
     native_tfit with T_noul = 1 (every other T unchanged).
Usage: python -I -X utf8 tnoul.py <repo> <jevkit dir> <out dir>"""
import json, math, sys
from pathlib import Path

repo, jd, od = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, repo)
od.mkdir(parents=True, exist_ok=True)
lo_b, hi_b = 0.2, 0.8

def inband(p):
    return lo_b < p < hi_b

calib = [json.loads(l) for l in open(jd / "calib_probs.jsonl", encoding="utf-8") if l.strip()]
noul = [c for c in calib if c["type"] == "noul"]
def pt(c, t):
    i = c["option_ids"].index("true")
    lp = [math.log(max(p, 1e-300)) / t for p in c["probs"]]
    m = max(lp)
    return math.exp(lp[i] - m) / sum(math.exp(v - m) for v in lp)
def nll(t):
    s = 0.0
    for c in noul:
        lp = [math.log(max(p, 1e-300)) / t for p in c["probs"]]
        m = max(lp)
        s += -(lp[c["gold"]] - (m + math.log(sum(math.exp(v - m) for v in lp))))
    return s / len(noul)
def band(t):
    return sum(inband(pt(c, t)) for c in noul) / len(noul)

rows = [json.loads(l) for l in open(jd / "slice_floor_rows.jsonl", encoding="utf-8") if l.strip()]
sp = [r["stock_p_shim"] for r in rows if r["type"] == "noul" and "stock_p_shim" in r]
stock_band = sum(inband(p) for p in sp) / len(sp)
at_floor = sum(1 for p in sp if abs(p - 0.801) < 1e-9 or abs(p - 0.199) < 1e-9)
thr = stock_band + 0.05

grid400 = [math.exp(math.log(0.25) + (math.log(4.0) - math.log(0.25)) * i / 399) for i in range(400)]
cand = [(t, nll(t), band(t)) for t in grid400]
feas = [c for c in cand if c[2] <= thr]
fb = min(cand, key=lambda c: (c[2], c[1]))
report_T = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2606436063705284, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.0921, 1.25, 1.5, 2.0, 3.0, 4.0]
table = [{"T": t, "band": round(band(t), 4), "in_band": sum(inband(pt(c, t)) for c in noul), "nll": round(nll(t), 4),
          "in_registered_range": 0.25 <= t <= 4.0} for t in report_T]
# smallest T (any) reaching the threshold, for context
t_reach = None
for k in range(1, 2001):
    t = 0.25 * (1 - k / 2001)
    if t <= 0:
        break
    if band(t) <= thr:
        t_reach = t
        break
# band of the gold-correct vs wrong noul at T 1 and at the fallback
def split(t):
    right = [c for c in noul if (pt(c, t) >= 0.5) == (c["option_ids"][c["gold"]] == "true")]
    return {"n_right": len(right), "n_wrong": len(noul) - len(right),
            "mean_nll_right": sum(-math.log(max(1e-300, pt(c, t) if c["option_ids"][c["gold"]] == "true" else 1 - pt(c, t))) for c in right) / max(1, len(right)),
            "mean_nll_wrong": sum(-math.log(max(1e-300, pt(c, t) if c["option_ids"][c["gold"]] == "true" else 1 - pt(c, t))) for c in noul if c not in right) / max(1, len(noul) - len(right))}
A = {"n_noul": len(noul), "stock_band": stock_band, "stock_n": len(sp), "stock_at_shim_floor_0.801_or_0.199": at_floor,
     "threshold": thr, "band_T1": band(1.0), "nll_T1": nll(1.0), "band_T025": band(0.25), "nll_T025": nll(0.25),
     "grid400_feasible_count": len(feas), "grid400_min_band": min(c[2] for c in cand),
     "fallback_rerun": {"T": fb[0], "nll": fb[1], "band": fb[2]},
     "largest_T_below_0.25_reaching_threshold": t_reach,
     "table": table, "right_wrong_T1": split(1.0), "right_wrong_fallback": split(fb[0])}

# (B) public
from jevkit.items import load_public
from jevkit.systemone import normalize_answer
items = {it.id: it for it in load_public()}
tf = json.loads((jd / "tfit.json").read_text(encoding="utf-8"))["per_type"]
def t_for(q, t_noul=None):
    typ = q["type"]
    if typ == "noul" and t_noul is not None:
        return t_noul
    k = 2 if typ == "noul" else len(q["criteria"])
    c = (tf.get("cells") or {}).get(f"{typ}:{k}")
    return float(c["T"]) if c else float(tf["types"][typ]["T"])
def temper(ps, t):
    w = [max(float(p), 1e-12) ** (1.0 / t) for p in ps]
    s = sum(w)
    return [x / s for x in w]
def retemper(q, a, t):
    if q["type"] == "noul":
        py = float(a["noul"])
        return {"type": "noul", "noul": temper([py, 1 - py], t)[0]}
    names = list(a["probabilities"])
    return normalize_answer(q, {"type": q["type"], "probabilities": dict(zip(names, temper([a["probabilities"][n] for n in names], t)))})
def build(t_noul):
    out = []
    for l in open(jd / "native.preds.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        it = items[r["id"]]
        if r.get("status") == "ok":
            r["answer"] = retemper(it.question, r["answer"], t_for(it.question, t_noul))
        out.append(r)
    return out
mine = build(None)
theirs = {json.loads(l)["id"]: json.loads(l) for l in open(jd / "native_tfit.preds.jsonl", encoding="utf-8") if l.strip()}
maxd = 0.0
for r in mine:
    a, b = r["answer"], theirs[r["id"]]["answer"]
    if a["type"] == "noul":
        maxd = max(maxd, abs(a["noul"] - b["noul"]))
    else:
        maxd = max(maxd, max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
        assert a.get("choice") == b.get("choice"), r["id"]
diag = build(1.0)
p = od / "native_tfit_noul1.preds.jsonl"
with open(p, "w", encoding="utf-8", newline="\n") as g:
    for r in diag:
        g.write(json.dumps(r, separators=(",", ":")) + "\n")
nb = lambda preds: sum(1 for r in preds if r["answer"]["type"] == "noul" and inband(r["answer"]["noul"]))
B = {"rebuild_vs_injob_native_tfit_max_abs_dp": maxd, "per_type_T_used": {"choice": tf["types"]["choice"]["T"], "choice:3": tf["cells"]["choice:3"]["T"],
     "score": tf["types"]["score"]["T"], "noul_injob": tf["types"]["noul"]["T"], "noul_diag": 1.0},
     "public_noul_band_native_tfit": nb(mine), "public_noul_band_diag_noul1": nb(diag),
     "public_noul_n": sum(1 for r in mine if r["answer"]["type"] == "noul"), "diag_preds": str(p.name)}
res = {"A_slice": A, "B_public": B}
(od / "tnoul.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in A.items() if k != "table"}, indent=1)); print(json.dumps(B, indent=1))
for r in table: print(r)

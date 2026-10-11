"""Independent rescore of the D19 calibrated variant (B-PREREG-D19-TEMP). Evidence lane, own code (no lead helpers).

  fit   --units-pool P --units-dev D --cell C            -> own NLL minimiser + gate + ECE15 before/after
  apply-dev --src untempered.jsonl --T t --out o.jsonl --lead lead.jsonl
  apply-jev --src preds.jsonl --T t --out o.jsonl --lead lead.jsonl
"""
import argparse, hashlib, json, math, sys
from collections import Counter


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def temper(ps, T):  # softmax(log p / T), p clipped at 1e-300 (zeros stay ~0)
    z = [math.log(max(p, 1e-300)) / T for p in ps]
    m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return [x / s for x in e]


# ---------------- own minimiser: NLL(beta) convex in beta = 1/T; bisection on its derivative
def grad(probs, labels, beta):
    g = 0.0
    for p, y in zip(probs, labels):
        lp = [math.log(max(x, 1e-300)) for x in p]
        z = [beta * x for x in lp]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
        g += sum(ei / s * li for ei, li in zip(e, lp)) - lp[y]
    return g / len(labels)


def fit_T(probs, labels, tlo=0.25, thi=4.0):
    lo, hi = 1 / thi, 1 / tlo
    if grad(probs, labels, lo) >= 0: return thi
    if grad(probs, labels, hi) <= 0: return tlo
    for _ in range(80):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if grad(probs, labels, mid) < 0 else (lo, mid)
    return 1 / ((lo + hi) / 2)


def metrics(probs, labels, T):
    q = [temper(p, T) for p in probs]
    nll = sum(-math.log(max(x[y], 1e-12)) for x, y in zip(q, labels)) / len(q)
    conf = [max(x) for x in q]; corr = [int(max(range(len(x)), key=x.__getitem__) == y) for x, y in zip(q, labels)]
    bins = [[0, 0.0, 0.0] for _ in range(15)]  # equal-width 15-bin top-label ECE
    for c, k in zip(conf, corr):
        b = bins[min(int(c * 15), 14)]; b[0] += 1; b[1] += c; b[2] += k
    ece = sum(abs(b[1] - b[2]) for b in bins) / len(q)
    return {"nll": nll, "accuracy": sum(corr) / len(corr), "ece15_ew": ece, "argmax": [max(range(len(x)), key=x.__getitem__) for x in q]}


def units(p):
    u = json.load(open(p, encoding="utf-8"))
    return u["item_ids"], u["decision_probs"], [int(y) for y in u["labels"]]


def cmd_fit(a):
    ids, P, L = units(a.units_pool)
    dids, DP, DL = units(a.units_dev)
    cell = json.load(open(a.cell, encoding="utf-8"))
    card = next(c for c in cell["dev_cards"] if c["step"] == 2800)
    by = {i: k for k, i in enumerate(ids)}
    dset = set(dids)
    rest = [k for k, i in enumerate(ids) if i not in dset]
    out = {"n_pool": len(ids), "n_dev": len(dids), "dev_subset_of_pool": dset <= set(by),
           "pool_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
           "dev_ids_sha256": hashlib.sha256("\n".join(sorted(dids)).encode()).hexdigest(),
           "training_card_step2800": {k: card.get(k) for k in ("nll", "accuracy", "n_items")}}
    T = fit_T(P, L)
    out["T_pool"] = T
    out["T_devcard200"] = fit_T(DP, DL)
    out["T_rest604"] = fit_T([P[k] for k in rest], [L[k] for k in rest])
    g = metrics(DP, DL, 1.0)
    out["gate"] = {"nll": g["nll"], "acc": g["accuracy"], "dnll": g["nll"] - card["nll"], "dacc": g["accuracy"] - card["accuracy"]}
    out["gate"]["pass"] = abs(out["gate"]["dnll"]) <= 0.01 and abs(out["gate"]["dacc"]) <= 0.01
    # dev-card units must equal the pool units on the same ids
    out["devcard_units_eq_pool_maxabs"] = max(abs(x - y) for i, p in zip(dids, DP) for x, y in zip(p, P[by[i]]))
    for name, sel in (("pool", list(range(len(ids)))), ("devcard", [by[i] for i in dids]), ("rest", rest)):
        b = metrics([P[k] for k in sel], [L[k] for k in sel], 1.0); f = metrics([P[k] for k in sel], [L[k] for k in sel], T)
        out[name] = {"n": len(sel), "before": {k: v for k, v in b.items() if k != "argmax"},
                     "after": {k: v for k, v in f.items() if k != "argmax"}, "argmax_flips": sum(x != y for x, y in zip(b["argmax"], f["argmax"]))}
    print(json.dumps(out, indent=1))
    json.dump({"ids": ids}, open(a.ids_out, "w")) if a.ids_out else None


def _dist(ans):
    """(keys, probs, kind) of an answer's served distribution, or None."""
    if isinstance(ans.get("probabilities"), dict) and ans["probabilities"]:
        k = list(ans["probabilities"]); return k, [float(ans["probabilities"][x]) for x in k], ans.get("type")
    if ans.get("type") == "noul" and isinstance(ans.get("noul"), (int, float)):
        p = float(ans["noul"]); return ["T", "F"], [p, 1 - p], "noul"
    return None


def _apply(ans, T):
    d = _dist(ans)
    if d is None: return ans, "untouched"
    keys, ps, kind = d
    q = temper(ps, T)
    new = dict(ans)
    if kind == "noul" and "probabilities" not in ans:
        new["noul"] = q[0]; return new, "noul"
    new["probabilities"] = dict(zip(keys, q))
    j = max(range(len(q)), key=q.__getitem__)
    if "confidence" in new: new["confidence"] = q[j]
    if kind == "choice" and "choice" in new: new["choice"] = keys[j]
    if kind == "score": new["score"] = sum(float(k) * x for k, x in zip(keys, q))
    if kind == "noul" and "noul" in new:  # noul with a distribution: p(true)
        new["noul"] = new["probabilities"].get("true", new["noul"])
    return new, kind


def _flat(ans):
    d = _dist(ans)
    if d is None: return {}
    keys, ps, kind = d
    r = dict(zip(keys, ps))
    if kind == "score": r["__score"] = float(ans["score"])
    return r


def _top(ans):
    d = _dist(ans)
    if d is None: return None
    k, p, _ = d; return k[max(range(len(p)), key=p.__getitem__)]


def _iter_answers(row, jev):
    if jev: yield row["id"], row["answer"]
    else:
        for k, v in ((row.get("response") or {}).get("answers") or {}).items():
            yield f'{row["run_id"]}#{k}', v


def cmd_apply(a, jev):
    kinds = Counter(); flips = 0; n = 0
    with open(a.src, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if jev:
                old = r["answer"]; r["answer"], k = _apply(old, a.T); kinds[k] += 1
                flips += _top(old) != _top(r["answer"]); n += 1
            else:
                ans = (r.get("response") or {}).get("answers") or {}
                for key in list(ans):
                    old = ans[key]; ans[key], k = _apply(old, a.T); kinds[k] += 1
                    flips += _top(old) != _top(ans[key]); n += 1
            g.write(json.dumps(r) + "\n")
    rep = {"src_sha256": sha(a.src), "ours_sha256": sha(a.out), "T": a.T, "answers": n, "kinds": dict(kinds), "argmax_flips_vs_untempered": flips}
    if a.lead:
        ours = {i: x for line in open(a.out, encoding="utf-8") if line.strip() for i, x in _iter_answers(json.loads(line), jev)}
        lead = {i: x for line in open(a.lead, encoding="utf-8") if line.strip() for i, x in _iter_answers(json.loads(line), jev)}
        mx = 0.0; mxs = 0.0; fl = 0; worst = None
        for i in ours:
            fo, fl_ = _flat(ours[i]), _flat(lead.get(i, {}))
            if set(fo) != set(fl_): fl += 1; continue
            for k in fo:
                d = abs(fo[k] - fl_[k])
                if k == "__score": mxs = max(mxs, d)
                elif d > mx: mx, worst = d, (i, k)
            fl += _top(ours[i]) != _top(lead[i])
        rep.update({"lead_sha256": sha(a.lead), "same_ids": set(ours) == set(lead), "n_ours": len(ours), "n_lead": len(lead),
                    "max_abs_dp_vs_lead": mx, "worst": worst, "max_abs_dscore_vs_lead": mxs, "argmax_disagree_vs_lead": fl})
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="c", required=True)
    s = sub.add_parser("fit"); s.add_argument("--units-pool"); s.add_argument("--units-dev"); s.add_argument("--cell"); s.add_argument("--ids-out")
    for n in ("apply-dev", "apply-jev"):
        s = sub.add_parser(n); s.add_argument("--src"); s.add_argument("--T", type=float); s.add_argument("--out"); s.add_argument("--lead")
    a = ap.parse_args()
    if a.c == "fit": cmd_fit(a)
    else: cmd_apply(a, a.c == "apply-jev")

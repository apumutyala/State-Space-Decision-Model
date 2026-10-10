"""Independent rescore of J1 (job-v6qm3) jevkit arms from raw per-item predictions.
usage: python -I rescore_j1.py <repo(jevkit @7c131f1)> <jevkit preds dir> <screen.json> <out dir>"""
import hashlib
import json
import math
import sys
from pathlib import Path

repo, pdir, screen, outd = (Path(x) for x in sys.argv[1:5])
sys.path.insert(0, str(repo))
import numpy as np  # noqa: E402

from jevkit import cli as JC  # noqa: E402
from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public, read_pin, verify_public  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402

outd.mkdir(parents=True, exist_ok=True)
sj = json.loads(screen.read_text(encoding="utf-8"))
jk = sj["jevkit"]
price = None
# the price the job declared: from the job's own per-arm report (meter output) is not the spec; take it from params
params = json.loads((repo / "jobs/m2_first_cell/job_params_jev1_q35_4b.json").read_text(encoding="utf-8"))
price = params["jevkit"]["price"]
res = {"pin": verify_public(), "pin_commit": read_pin()["commit"]}
items = load_public()
ids = [it.id for it in items]
res["n_items"] = len(ids)
res["items_sha256"] = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def rd(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- native_shimcal regenerated independently (vendored shim format_answer on native probs)
sys.path.insert(0, str(repo))
import third_party.h2o_lightning_shim as H  # noqa: E402

mod, c = H.load_contract()
byid = {it.id: it for it in items}
lg = lambda p: math.log(max(float(p), 1e-12))  # noqa: E731
lines = []
for r in rd(pdir / "native.preds.jsonl"):
    a, it = r.get("answer"), byid.get(r.get("id"))
    if r.get("status") == "ok" and isinstance(a, dict) and it is not None:
        q = it.question
        if q["type"] == "noul":
            py = float(a["noul"])
            r["answer"] = mod.format_answer("noul", ["true", "false"], ["", ""], [lg(py), lg(1 - py)], c.temperature,
                                            c.noul_floor)
        else:
            names = list(a["probabilities"])
            descs = [str(x) for x in (q.get("criteria") or [])] if q["type"] == "score" else names
            r["answer"] = mod.format_answer(q["type"], names, descs, [lg(a["probabilities"][n]) for n in names],
                                            c.temperature, c.noul_floor)
    lines.append(json.dumps(r, separators=(",", ":")) + "\n")
regen = outd / "native_shimcal.preds.jsonl"
regen.write_text("".join(lines), encoding="utf-8", newline="\n")
res["shimcal_regen_identical"] = regen.read_bytes() == (pdir / "native_shimcal.preds.jsonl").read_bytes()

arms = [a for a in jk["arms"]]
tables, axes, summ, idsets = {}, {}, {}, {}
for nm in arms:
    pf = regen if nm == "native_shimcal" else pdir / f"{nm}.preds.jsonl"
    preds = rd(pf)
    idsets[nm] = sorted(str(p["id"]) for p in preds)
    rp = outd / f"{nm}.jevkit.json"
    assert JC.main(["score", "--public", "--pred", str(pf), "--price", price, "--out", str(rp)]) == 0
    rep = json.loads(rp.read_text(encoding="utf-8"))
    job_rep = json.loads((pdir / f"{nm}.jevkit.json").read_text(encoding="utf-8"))
    ci = rep.get("ci95") or {}
    s = {"I": rep["axes"]["intelligence"], "C": rep["axes"]["calibration"], "S": rep["axes"]["speed"],
         "K": rep["axes"]["cost"], "capability": rep["capability"], "composite": rep["scores"],
         "I_ci95": ci.get("I"), "C_ci95": ci.get("C")}
    js = jk["arms"][nm]["summary"]
    t = JS.build_table(items, preds_to_answers(preds))
    tables[nm], axes[nm] = t, (rep["axes"]["speed"], rep["axes"]["cost"])
    full = JS.aggregate(t)
    parts = full["calibration"]["parts"]
    fam = {}
    for f in sorted({it.family for it in items}):
        m = np.array([byid[i].family == f for i in t.ids])
        fam[f] = {"n": int(m.sum()), "I": JS.aggregate(t, np.flatnonzero(m), detail=False)["intelligence"]["I"],
                  "acc": float(t.correct[m].mean())}
    summ[nm] = {"rescored": s, "match_job_summary": all(s[k] == js[k] for k in s),
                "report_equal_job_report": rep == job_rep,
                "mismatch": {k: [s[k], js[k]] for k in s if s[k] != js[k]},
                "status": {k: sum(1 for p in preds if p.get("status") == k) for k in {p.get("status") for p in preds}},
                "choice_ece_hard": parts.get("choice", {}).get("ece_hard"),
                "noul_abstention_rate": parts.get("noul", {}).get("abstention_rate"),
                "cal_parts": parts, "family": fam}
res["arms"] = summ
res["item_ids_equal_across_arms"] = all(v == sorted(ids) for v in idsets.values())


def paired_vs(a, b, B=1000, seed=16):
    def ax(nm, idx=None):
        r = JS.aggregate(tables[nm], idx, detail=False)
        i, cc = r["intelligence"]["I"], r["calibration"]["score"]
        sp, k = axes[nm]
        return {"I": i, "C": cc, "capability": JS.capability(i, cc), "composite_A": JS.composite(i, cc, sp, k, "A")}
    pa, pb = ax(a), ax(b)
    rng = np.random.default_rng(seed)
    st = JS.strata(tables[b])
    dr = {k: [] for k in pa}
    for _ in range(B):
        idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
        xa, xb = ax(a, idx), ax(b, idx)
        for k in dr:
            if xa[k] is not None and xb[k] is not None:
                dr[k].append(xa[k] - xb[k])
    return {k: {"point": pa[k] - pb[k], "ci95": [float(np.percentile(dr[k], 2.5)), float(np.percentile(dr[k], 97.5))],
                "a": pa[k], "b": pb[k]} for k in dr}


pv = {}
for nm in arms:
    if nm == "stock_shim":
        continue
    mine = paired_vs(nm, "stock_shim")
    job = jk["vs_stock_shim"][nm]
    pv[nm] = {"rescored": mine, "exact": all(mine[k]["point"] == job[k]["point"] and mine[k]["ci95"] == job[k]["ci95"]
                                             for k in mine),
              "family_dI": {f: (summ[nm]["family"][f]["I"] - summ["stock_shim"]["family"][f]["I"])
                            if summ[nm]["family"][f]["I"] is not None and summ["stock_shim"]["family"][f]["I"] is not None
                            else None for f in summ[nm]["family"]}}
res["vs_stock_shim"] = pv
res["job_primary"] = jk.get("primary")
(outd / "rescore.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({"shimcal_regen_identical": res["shimcal_regen_identical"],
                  "ids_equal": res["item_ids_equal_across_arms"], "n": len(ids),
                  "arms": {k: [v["match_job_summary"], v["report_equal_job_report"], v["mismatch"]] for k, v in summ.items()},
                  "paired_exact": {k: v["exact"] for k, v in pv.items()}}, default=str))

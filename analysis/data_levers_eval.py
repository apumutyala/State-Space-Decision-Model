"""Data levers 2026-10-09, step 1 (eval side, CPU only): per-family gap of the champion (D1 + v3w readout) and D10w vs
Clef-flash on the 2,008-row subset, the label-collapse flags (rule of jobs/common/label_collapse.py on
origin/claude/b-collapse-warn, re-implemented read-only below), and each flagged/trailing family's EVAL shape
(question type, K, gold base rate). Diagnosis only: nothing here is fit to DEV rows; it steers which TRAINING data to add.

    python -X utf8 -I analysis/data_levers_eval.py CHAMP.results.jsonl D10W/units-child-index.json OUT.json
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gap_b"))
from common import FROZEN, SOURCES, load_results, load_rows, score  # noqa: E402

champ_p, d10_p, out_p = sys.argv[1:4]
rows = load_rows()
rowmap = {r["_evaluation"]["run_id"]: r for r in rows}
name = {r["_evaluation"]["catalog_id"]: r["_evaluation"]["dataset"] for r in rows}
R = {"champ": load_results(champ_p), "d10w": load_results(d10_p), "clef": load_results(SOURCES["clef"])}
R["clef"] = {rid: R["clef"][rid] for rid in rowmap}
for k in ("champ", "d10w"):
    R[k] = {rid: v for rid, v in R[k].items() if rid in rowmap}
B = {k: score.decompose(rows, v) for k, v in R.items()}
P = {k: score.point(b) for k, b in B.items()}
W = score.weights(set(B["clef"]))
print({k: round(P[k]["index"], 3) for k in P})


def skill(k, n):
    return 100 * P[k]["per_benchmark"][n]["skill"]


# ---- label-collapse tally (same rule as jobs/common/label_collapse.py) + full per-group label shares
def tally(results):
    g = {}
    for r in rows:
        ev = r["_evaluation"]
        res = results.get(ev["run_id"])
        if not isinstance(res, dict) or res.get("status") != "ok":
            continue
        ans = ((res.get("response") or {}).get("answers")) or {}
        for key, q in (r.get("questions") or {}).items():
            t = q.get("type")
            if t not in ("choice", "noul"):
                continue
            gold = (r.get("expected") or {}).get(key)
            a = ans.get(key)
            if gold is None or not isinstance(a, dict):
                continue
            if t == "choice":
                labels = tuple(sorted(str(x) for x in (q.get("criteria") or {})))
                pred, gold = a.get("choice"), str(gold)
                pred = None if pred is None else str(pred)
            else:
                labels = ("false", "true")
                gold = ("true" if gold else "false") if isinstance(gold, bool) else (
                    ("true" if gold >= 0.5 else "false") if isinstance(gold, (int, float)) else str(gold).lower())
                p = a.get("noul")
                pred = None if not isinstance(p, (int, float)) else ("true" if p >= 0.5 else "false")
            if pred is None or not labels:
                continue
            e = g.setdefault((int(ev["catalog_id"]), t, labels), {"gold": collections.Counter(), "pred": collections.Counter(), "n": 0})
            e["n"] += 1
            e["gold"][gold] += 1
            e["pred"][pred] += 1
    return g


flags = {}
for k in ("champ", "d10w"):
    for (cid, t, labels), e in tally(R[k]).items():
        n = e["n"]
        if n < 20:
            continue
        for lab in sorted(set(labels) | set(e["gold"])):
            gs, ps = e["gold"][lab] / n, e["pred"][lab] / n
            if 0.10 <= gs <= 0.90 and (ps < 0.02 or ps > 0.98):
                flags.setdefault(cid, []).append(dict(arm=k, qtype=t, k=len(labels), label=lab, gold=round(gs, 3),
                                                      pred=round(ps, 4), n=n, kind="COLLAPSE" if ps in (0, 1) else "near"))

# ---- eval shape per family (all question groups, gold base rates); option-count histogram
shape = {}
for r in rows:
    cid = r["_evaluation"]["catalog_id"]
    s = shape.setdefault(cid, {"rows": 0, "qtypes": collections.Counter(), "k": collections.Counter(), "gold": collections.Counter()})
    s["rows"] += 1
    for key, q in r["questions"].items():
        gold = r["expected"].get(key)
        if gold is None:
            continue
        t = q["type"]
        s["qtypes"][t] += 1
        if t == "choice":
            kk = len(q.get("criteria") or {})
            s["k"][kk] += 1
            labs = sorted(q["criteria"])
            if kk <= 4 or set(labs) <= {"yes", "no", "true", "false"}:
                s["gold"][f"{kk}:{gold}"] += 1
        elif t == "noul":
            s["gold"]["noul:" + ("true" if (gold is True or (isinstance(gold, (int, float)) and gold >= 0.5)) else "false")] += 1

fam = {}
for n in sorted(B["clef"]):
    gc = 100 * W[n] * (P["clef"]["per_benchmark"][n]["skill"] - P["champ"]["per_benchmark"][n]["skill"])
    gd = 100 * W[n] * (P["clef"]["per_benchmark"][n]["skill"] - P["d10w"]["per_benchmark"][n]["skill"])
    s = shape[n]
    fam[n] = dict(name=name[n], frozen=n in FROZEN, w=W[n], skill={k: round(skill(k, n), 2) for k in P},
                  gap_pts_champ=round(gc, 3), gap_pts_d10w=round(gd, 3),
                  trail10_champ=skill("clef", n) - skill("champ", n) >= 10, trail10_d10w=skill("clef", n) - skill("d10w", n) >= 10,
                  flags=flags.get(n, []), rows=s["rows"], qtypes=dict(s["qtypes"]), k=dict(sorted(s["k"].items())),
                  gold=dict(s["gold"].most_common(12)))
sel = {n: f for n, f in fam.items() if not f["frozen"] and (f["flags"] or f["trail10_champ"] or f["trail10_d10w"])}
Path(out_p).write_text(json.dumps({"index": {k: P[k]["index"] for k in P}, "families": fam, "selected": sorted(sel)}, indent=1), encoding="utf-8")
for n, f in sorted(sel.items(), key=lambda kv: -max(kv[1]["gap_pts_champ"], kv[1]["gap_pts_d10w"])):
    print(n, f["name"][:28], "w=%.4f" % f["w"], f["skill"], "gapC %.2f gapD %.2f" % (f["gap_pts_champ"], f["gap_pts_d10w"]),
          f["k"], f["gold"], [(x["arm"], x["label"], x["gold"], x["pred"], x["kind"]) for x in f["flags"]])
print("frozen flagged:", [(n, fam[n]["name"], [(x["arm"], x["label"]) for x in fam[n]["flags"]]) for n in fam if fam[n]["frozen"] and fam[n]["flags"]])

"""R4 score-only joint decodes on saved per-row predictions, scored by the frozen scorer (DEV 2,008-row subset).

    python -I -X utf8 receipts/2026-10-08-R4-joint/simulate.py RESULTS.jsonl [RESULTS2.jsonl ...]

RESULTS = a kit results.jsonl (stock v2: job-zhyvg art-6bhnu index/stock/results.jsonl) or the D1 cell's
units-child-index.json["index_results"] written one row per line (job-c7pq2 art-bm3e6, phases/final.tgz). Each
variant rewrites only the answers of candidate-block requests (decide.joint) from the row's own saved probabilities;
nothing else (no gold, no other row) enters. Prints one JSON line per (file, decode): Index, per-benchmark skill for
the case-exact benchmarks, rows the rule applied to, rows whose choices changed, board ECE.
"""
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from decide.joint import candidate_block, joint_logits  # noqa: E402
from harness import calibration, score  # noqa: E402
from harness.cli import load_slice  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["run_id"]: r for r in map(json.loads, f)}


def answer(lp, q):
    p = {o: math.exp(v) for o, v in lp.items()}
    z = sum(p.values())
    p = {o: v / z for o, v in p.items()}
    if q["type"] == "noul":
        return {"type": "noul", "noul": p["true"]}
    best = max(sorted(p), key=lambda o: p[o])
    return {"type": "choice", "choice": best, "confidence": round(p[best], 4), "probabilities": {o: round(v, 4) for o, v in p.items()}}


def variant(rows, res, decode):
    out, applied, changed = {}, 0, 0
    for r in rows:
        rid = r["_evaluation"]["run_id"]
        x = res[rid]
        req = {"questions": r["questions"]}
        if decode is None or x["status"] != "ok" or candidate_block(req) is None:
            out[rid] = x
            continue
        a = x["response"]["answers"]
        lg = {k: {o: math.log(max(float(v), 1e-12)) for o, v in (a[k]["probabilities"] if q["type"] == "choice" else
              {"true": a[k]["noul"], "false": 1 - a[k]["noul"]}).items()} for k, q in r["questions"].items()}
        new, d = joint_logits(req, lg, decode)
        applied += 1
        changed += bool(d["changed"])
        y = copy.deepcopy(x)
        for k, q in r["questions"].items():
            y["response"]["answers"][k] = answer(new[k], q)
        out[rid] = y
    return out, applied, changed


for path in sys.argv[1:]:
    res = load(path)
    rows = [r for r in load_slice("dev") if r["_evaluation"]["run_id"] in res]
    base = None
    for decode in (None, "map", "card"):
        v, ap, ch = variant(rows, res, decode)
        p = score.point(score.decompose(rows, v))
        base = p["index"] if base is None else base
        ece = calibration.report(calibration.fields(rows, v))["board_protocol"]["ece"]
        print(json.dumps({"file": os.path.basename(path), "rows": len(rows), "decode": decode or "independent",
                          "index": round(p["index"], 4), "delta": round(p["index"] - base, 4),
                          "skill": {n: round(p["per_benchmark"][n]["skill"] * 100, 2) for n in (1, 9, 33)},
                          "applied": ap, "changed": ch, "board_ece": round(ece, 4)}))

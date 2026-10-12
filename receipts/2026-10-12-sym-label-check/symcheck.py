import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

d = Path(sys.argv[1])
rows = {json.loads(l)["id"]: json.loads(l) for l in open(d / "slice_floor_rows.jsonl", encoding="utf-8")}
cp = {json.loads(l)["id"]: json.loads(l) for l in open(d / "calib_probs.jsonl", encoding="utf-8")}
ORD = {"none", "low", "minor", "medium", "moderate", "high", "major", "severe", "critical", "urgent", "very_low",
       "very_high"}


def klass(c):
    t = c["type"]
    if t == "score":
        return "SCORE"
    if t == "choice":
        ids = [str(x) for x in c["option_ids"]]
        if all(re.fullmatch(r"-?\d+", x) for x in ids) or all(x.lower() in ORD for x in ids):
            return "ORDINAL_ID"
        return "NOMINAL_CHOICE"
    return "NOMINAL_NOUL"


by = defaultdict(list)
idshapes = Counter()
for k, r in rows.items():
    c = cp[k]
    cl = klass(c)
    by[cl].append((int(r["native"]), int(r["step0"])))
    if c["type"] != "noul":
        idshapes[(c["type"], tuple(c["option_ids"])[:6])] += 1


def ci(pairs, B=10000, seed=16):
    rng = random.Random(seed)
    n = len(pairs)
    d = [a - b for a, b in pairs]
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


out = {}
for cl in ("SCORE", "ORDINAL_ID", "NOMINAL_CHOICE", "NOMINAL_NOUL"):
    p = by.get(cl, [])
    if not p:
        out[cl] = {"n": 0}
        continue
    a = sum(x for x, _ in p) / len(p)
    b = sum(y for _, y in p) / len(p)
    out[cl] = {"n": len(p), "trained": round(a, 4), "stock": round(b, 4), "delta_pt": round(100 * (a - b), 2),
               "ci95_pt": [round(100 * x, 2) for x in ci(p)]}
pool = by.get("SCORE", []) + by.get("ORDINAL_ID", [])
nom = by.get("NOMINAL_CHOICE", []) + by.get("NOMINAL_NOUL", [])
for nm, p in (("SCORE+ORDINAL", pool), ("NOMINAL", nom)):
    if p:
        a = sum(x for x, _ in p) / len(p)
        b = sum(y for _, y in p) / len(p)
        out[nm] = {"n": len(p), "delta_pt": round(100 * (a - b), 2), "ci95_pt": [round(100 * x, 2) for x in ci(p)]}
s_ok = "SCORE+ORDINAL" in out and out["SCORE+ORDINAL"]["n"] >= 30 and out["SCORE+ORDINAL"]["delta_pt"] <= -10
n_ok = out["NOMINAL"]["delta_pt"] >= -5
out["verdict"] = "SUPPORTED" if (s_ok and n_ok) else "NOT SUPPORTED"
fam = defaultdict(Counter)
for k, r in rows.items():
    fam[cp[k]["family"]][klass(cp[k])] += 1
out["family_x_class"] = {f: dict(v) for f, v in fam.items()}
out["id_shapes_top"] = [[t, list(ids), n] for (t, ids), n in idshapes.most_common(15)]
print(json.dumps(out, indent=1))

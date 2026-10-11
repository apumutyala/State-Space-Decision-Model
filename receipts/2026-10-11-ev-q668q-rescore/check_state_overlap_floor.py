"""Sensitivity of the step-0 floor to the slice rows whose STATE equals a train row's state (argv: art_final train calib).

Run with python -I. Content overlap (state + questions) is 0; this only checks state-alone duplicates.
"""
import collections
import gzip
import hashlib
import json
import sys

import numpy as np

art, tr, ca = sys.argv[1:4]
rgz = lambda p: [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8") if x.strip()]


def nt(x):
    s = json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()


T = collections.defaultdict(list)
for r in rgz(tr):
    T[nt(r.get("state"))].append(r)
hit = {}
for r in rgz(ca):
    h = nt(r.get("state"))
    if h in T:
        hit[r["id"]] = any((r.get("labels") or {}).get("q") == (t.get("labels") or {}).get("q") for t in T[h])
rows = [json.loads(x) for x in open(f"{art}/out/screen/jevkit/slice_floor_rows.jsonl", encoding="utf-8") if x.strip()]
nat = np.array([bool(r["native"]) for r in rows])
s0 = np.array([bool(r["step0"]) for r in rows])
m = np.array([r["id"].rsplit("/", 1)[0] in hit for r in rows])
ml = np.array([hit.get(r["id"].rsplit("/", 1)[0], False) for r in rows])
print("slice rows with a train-identical state", int(m.sum()), "| of which the train row has the same gold label", int(ml.sum()))
print(" on those rows: step0 %.3f selected %.3f (n %d)" % (s0[m].mean(), nat[m].mean(), m.sum()))
k = ~m
rng = np.random.default_rng(16)
ii = np.flatnonzero(k)
d = [nat[s].mean() - s0[s].mean() for s in (rng.choice(ii, len(ii)) for _ in range(1000))]
print(" floor EXCLUDING them: n %d step0 %.4f selected %.4f delta %.2f pp CI95 [%.2f, %.2f]" % (
    k.sum(), s0[k].mean(), nat[k].mean(), 100 * (nat[k].mean() - s0[k].mean()),
    100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)))

"""Least-squares attribution of D1 child wall time from analysis/d1_row_structure.py output (argv[1]).
Usage: python -I analysis/d1_slowdown_attr.py rowstruct.json"""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
f = lambda k: np.array([r[k] for r in R], float)
C, P = f("child_ms"), f("parent_ms")
tok = f("L") + f("suf")  # backbone positions of prefix+suffix (once)
cont = f("cont")
grouped, packed = f("grouped"), f("packed")
tokz = 2 * f("tok_s") * 1000  # plan + score tokenisation, laptop CPU ms
memo = f("memo_s") * 1000
print("rows", len(R), "child s", C.sum() / 1e3, "parent s", P.sum() / 1e3)
print("calls/row grouped mean %.2f packed mean %.2f; total %d -> %d" % (grouped.mean(), packed.mean(), grouped.sum(), packed.sum()))
print("tokenisation (laptop, 2 passes) s %.0f = %.1f%% of child; memo s %.0f" % (tokz.sum() / 1e3, 100 * tokz.sum() / C.sum(), memo.sum() / 1e3))
# child GPU-side = child - tokenisation; fit per-token + per-call
y = C - tokz
X = np.c_[tok, cont, grouped]
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("fit child-tokz = %.4f ms/tok + %.3f ms/cont-tok + %.1f ms/call  R2 %.3f" % (*coef, r2))
share = coef * X.sum(0) / C.sum()
print("share of child wall: tok %.2f cont %.2f calls %.2f tokenisation %.2f" % (*share, tokz.sum() / C.sum()))
# projection: packed calls (tails re-read suffix per option: tail_tok), memo tokenisation
tail = f("tail_tok")
Xf = np.c_[f("L"), tail, packed]
fast = coef[0] * f("L") + coef[0] * tail + coef[2] * packed + memo
print("projected fast child s %.0f (memo + packed): speedup %.2fx; vs parent %.2fx" % (fast.sum() / 1e3, C.sum() / fast.sum(), fast.sum() / P.sum()))
only_memo = C - tokz + memo
only_pack = C - coef[2] * (grouped - packed) - coef[0] * f("suf") + coef[0] * tail - coef[1] * cont
print("memo only %.2fx; packed only %.2fx" % (C.sum() / only_memo.sum(), C.sum() / only_pack.sum()))
for lo, hi in ((1, 2), (2, 5), (5, 20), (20, 1000)):
    m = (f("Q") >= lo) & (f("Q") < hi)
    print("Q[%d,%d) n=%d child/parent %.2f tokz share %.2f calls %.1f->%.1f" % (lo, hi, m.sum(), C[m].sum() / P[m].sum(), tokz[m].sum() / C[m].sum(), grouped[m].mean(), packed[m].mean()))

import sys, json, itertools
sys.path.insert(0, '.')
from jevkit import scoring
def brentq(f,a,b):
    fa,fb=f(a),f(b)
    if fa*fb>0: raise ValueError
    for _ in range(200):
        m=(a+b)/2; fm=f(m)
        if fa*fm<=0: b,fb=m,fm
        else: a,fa=m,fm
    return (a+b)/2
cfg = json.load(open('receipts/2026-10-11-gap-attribution/cfg.json'))
ax = {n: {"I": s["report"]["I"], "C": s["report"]["C"], "S": s["S"], "K": s["K"]} for n, s in cfg["systems"].items()}
A = lambda a: scoring.composite(a["I"], a["C"], a["S"], a["K"], "A")
AX = "ICSK"
base = ax["h2o_zbtbj"]; tgt = A(base)
print("H2O A", tgt)
for n in ["stock4b_shim_zbtbj", "jev1_native", "jev1_native_shimcal", "D19", "D19_cal"]:
    c = ax[n]; gap = A(base) - A(c)
    # Shapley
    sh = {x: 0.0 for x in AX}
    orders = list(itertools.permutations(AX))
    fixed = {}
    for o in orders:
        cur = dict(c); prev = A(cur); contrib = {}
        for x in o:
            cur[x] = base[x]; now = A(cur); contrib[x] = now - prev; sh[x] += now - prev; prev = now
        fixed["".join(o)] = contrib
    sh = {x: v / 24 for x, v in sh.items()}
    # range of I attribution across the 24 orders
    Ir = [fixed[o]["I"] for o in fixed]; Cr = [fixed[o]["C"] for o in fixed]
    # reciprocal-space (harmonic) linear decomposition: 1/H = mean(1/x); dH ~ split by d(1/x)
    rec = {x: (1/c[x] - 1/base[x]) / 4 for x in AX}; rs = sum(rec.values())
    recs = {x: gap * rec[x] / rs for x in AX}
    # rank of each method
    rk = lambda d: "".join(sorted(d, key=lambda x: -d[x]))
    single_in = {x: A({**c, x: base[x]}) - A(c) for x in AX}       # add one axis to candidate
    single_out = {x: A(base) - A({**base, x: c[x]}) for x in AX}  # remove one axis from baseline
    I100 = A({**c, "I": 100.0}); C100 = A({**c, "C": 100.0}); I100Cb = A({**c, "I": 100.0, "C": base["C"]})
    ICb = A({**c, "I": base["I"], "C": base["C"]})
    try: Ineed = brentq(lambda i: A({**c, "I": i}) - tgt, 50, 100)
    except ValueError: Ineed = None
    try: Cneed = brentq(lambda v: A({**c, "C": v}) - tgt, 1, 100)
    except ValueError: Cneed = None
    # with C at H2O, what I needed?
    try: IneedCb = brentq(lambda i: A({**c, "I": i, "C": base["C"]}) - tgt, 50, 100)
    except ValueError: IneedCb = None
    # I and C both = H2O and also S,K at H2O? need just I+C; does I=C=100 close?
    IC100 = A({**c, "I": 100.0, "C": 100.0})
    print(json.dumps({"cand": n, "gap": round(gap, 3), "shapley": {k: round(v, 3) for k, v in sh.items()}, "shapley_rank": rk(sh),
          "recip_split": {k: round(v, 3) for k, v in recs.items()}, "recip_rank": rk(recs),
          "single_in": {k: round(v, 3) for k, v in single_in.items()}, "single_in_rank": rk(single_in),
          "single_out": {k: round(v, 3) for k, v in single_out.items()}, "single_out_rank": rk(single_out),
          "I_attr_range_over_24_orders": [round(min(Ir), 3), round(max(Ir), 3)], "C_attr_range": [round(min(Cr), 3), round(max(Cr), 3)],
          "A_I100_Cfixed": round(I100, 2), "A_C100_Ifixed": round(C100, 2), "A_I100_C_H2O": round(I100Cb, 2), "A_IC_H2O": round(ICb, 2), "A_I100_C100": round(IC100, 2),
          "I_needed_Cfixed": Ineed and round(Ineed, 2), "C_needed_Ifixed": Cneed and round(Cneed, 2), "I_needed_C_at_H2O": IneedCb and round(IneedCb, 2)}))

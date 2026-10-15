"""Diagnostics for a rescore (evidence lane): (1) is the left tail of the paired Composite-A delta driven by the
Intelligence floor term min(1, I/50)^2? Re-runs the SAME paired draws (default_rng(seed), strata of the baseline
table) and reports the delta quantiles with the floor term removed (Speed/Cost gates unchanged); (2) top-1 per family
group under two correctness definitions: jevkit Intelligence correctness (noul must be decisive, 0.2<p<0.8 counts as
wrong) and plain argmax (noul p_yes >= 0.5 -> yes).

  python -I -X utf8 scripts/ev_rescore_diag_floor.py --jdir <out/screen/jevkit> --a native --b stock_shim \
      --sa 85.25137428679938 --ka 61.227052829294564 --sb 89.63673151849318 --kb 61.227052829294564 \
      --groups "COVERED=...;UNCOVERED=..." --out diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import scoring as JS  # noqa: E402
from jevkit import spec  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402

QS = (1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)


def nofloor(i, c, s, k):
    if None in (i, c, s, k):
        return None
    if min(i, c, s, k) <= 0:
        return 0.0
    w = spec.OPTIONS["A"]["weights"]
    h = sum(w) / sum(wi / x for wi, x in zip(w, (i, c, s, k)))
    return h * min(1.0, s / spec.AXIS_GATE) ** 2 * min(1.0, k / spec.AXIS_GATE) ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jdir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pa", default=None, help="preds path for a (default <jdir>/<a>.preds.jsonl)")
    ap.add_argument("--pb", default=None)
    for x in ("sa", "ka", "sb", "kb"):
        ap.add_argument(f"--{x}", type=float, required=True)
    ap.add_argument("--groups", default="")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_public()
    rd = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    pa = rd(a.pa or Path(a.jdir) / f"{a.a}.preds.jsonl")
    pb = rd(a.pb or Path(a.jdir) / f"{a.b}.preds.jsonl")
    ta, tb = JS.build_table(items, preds_to_answers(pa)), JS.build_table(items, preds_to_answers(pb))
    rng = np.random.default_rng(a.seed)
    st = JS.strata(tb)
    d_full, d_nf, ia_l, ib_l = [], [], [], []
    for _ in range(a.B):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        ra, rb = JS.aggregate(ta, idx, detail=False), JS.aggregate(tb, idx, detail=False)
        ia, ca = ra["intelligence"]["I"], ra["calibration"]["score"]
        ib, cb = rb["intelligence"]["I"], rb["calibration"]["score"]
        ia_l.append(ia)
        ib_l.append(ib)
        d_full.append(JS.composite(ia, ca, a.sa, a.ka, "A") - JS.composite(ib, cb, a.sb, a.kb, "A"))
        d_nf.append(nofloor(ia, ca, a.sa, a.ka) - nofloor(ib, cb, a.sb, a.kb))
    q = lambda v: {str(x): float(np.percentile(v, x)) for x in QS}
    ia_l, ib_l, d_full = np.array(ia_l), np.array(ib_l), np.array(d_full)
    fa, fb = JS.aggregate(ta, None, detail=False), JS.aggregate(tb, None, detail=False)
    pt_nf = (nofloor(fa["intelligence"]["I"], fa["calibration"]["score"], a.sa, a.ka)
             - nofloor(fb["intelligence"]["I"], fb["calibration"]["score"], a.sb, a.kb))
    either_below = (ia_l < 50) | (ib_l < 50)
    tail = {"delta_A_quantiles_with_floor": q(d_full), "delta_A_quantiles_no_floor": q(d_nf),
            "point_no_floor": pt_nf,
            "frac_I_below_50": {"a": float((ia_l < 50).mean()), "b": float((ib_l < 50).mean())},
            "delta_A_quantiles_when_both_I_ge_50": q(d_full[~either_below]) if (~either_below).any() else None,
            "delta_A_quantiles_when_a_I_lt_50": q(d_full[ia_l < 50]) if (ia_l < 50).any() else None,
            "n_both_ge_50": int((~either_below).sum())}
    # top-1 per group, two definitions
    by = {it.id: it for it in items}

    def argmax_correct(t):
        out = np.zeros(len(t.ids), bool)
        for i, (iid, p) in enumerate(zip(t.ids, t.parsed)):
            it = by[iid]
            if not p.valid:
                continue
            if it.type == "noul":
                out[i] = ("yes" if p.probs["yes"] >= 0.5 else "no") == str(it.expected)
            else:
                out[i] = bool(t.correct[i])
        return out

    fam = np.array([by[i].family for i in ta.ids])
    typ = np.array([by[i].type for i in ta.ids])
    groups = {g.split("=", 1)[0]: g.split("=", 1)[1].split("+") for g in a.groups.split(";") if "=" in g}
    defs = {"jevkit_correct": (ta.correct, tb.correct), "argmax": (argmax_correct(ta), argmax_correct(tb))}
    top1 = {}
    for dn, (xa, xb) in defs.items():
        top1[dn] = {"all": {"a": int(xa.sum()), "b": int(xb.sum())}}
        for g, fs in groups.items():
            m = np.isin(fam, fs)
            top1[dn][g] = {"n": int(m.sum()), "a_acc": float(xa[m].mean()), "b_acc": float(xb[m].mean()),
                           "delta_pp": 100 * float(xa[m].mean() - xb[m].mean())}
        top1[dn]["by_type"] = {ty: {"n": int((typ == ty).sum()), "delta_pp": 100 * float(xa[typ == ty].mean()
                                                                                         - xb[typ == ty].mean())}
                               for ty in sorted(set(typ))}
    noul = typ == "noul"
    abst = {"a": int((ta.abstain & noul).sum()), "b": int((tb.abstain & noul).sum()), "n_noul": int(noul.sum())}
    res = {"schema": "ev-rescore-diag/1", "a": a.a, "b": a.b, "B": a.B, "seed": a.seed, "tail": tail, "top1": top1,
           "noul_abstentions": abst}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

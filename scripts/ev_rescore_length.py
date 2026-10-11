"""Length breakdown for a rescore (evidence lane): top-1 (jevkit correctness and plain argmax) and choice NLL of two
arms split by prompt length (preds ``usage.input_tokens`` of arm a, > --cut vs <= --cut), plus named families split
the same way. Paired bootstrap of the top-1 delta within each cell (iid item resample, B 1000, seed 16).

  python -I -X utf8 scripts/ev_rescore_length.py --pa native.preds.jsonl --pb stock_shim.preds.jsonl \
      --cut 2048 --families multi_hop,long_policy --out length.json
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
from jevkit.items import load_public  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev_rescore_jevkit import _mcnemar_exact, _nll  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True)
    ap.add_argument("--pb", required=True)
    ap.add_argument("--cut", type=int, default=2048)
    ap.add_argument("--families", default="")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_public()
    by = {it.id: it for it in items}
    rd = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    pa, pb = rd(a.pa), rd(a.pb)
    ta, tb = JS.build_table(items, preds_to_answers(pa)), JS.build_table(items, preds_to_answers(pb))
    toks_a = {str(p["id"]): (p.get("usage") or {}).get("input_tokens") for p in pa}
    toks_b = {str(p["id"]): (p.get("usage") or {}).get("input_tokens") for p in pb}
    tok = np.array([toks_a[i] for i in ta.ids], float)
    same_tokens = all(toks_a[i] == toks_b.get(i) for i in toks_a)

    def am(t):
        out = np.zeros(len(t.ids), bool)
        for i, (iid, p) in enumerate(zip(t.ids, t.parsed)):
            it = by[iid]
            if p.valid:
                out[i] = (("yes" if p.probs["yes"] >= 0.5 else "no") == str(it.expected)) if it.type == "noul" \
                    else bool(t.correct[i])
        return out

    ama, amb = am(ta), am(tb)
    na, nb = _nll(ta, items), _nll(tb, items)
    fam = np.array([by[i].family for i in ta.ids])
    rng = np.random.default_rng(a.seed)

    def cell(m):
        ix = np.flatnonzero(m)
        if not len(ix):
            return {"n": 0}
        ch = ix[~np.isnan(na[ix]) & ~np.isnan(nb[ix])]
        boots = []
        for _ in range(a.B):
            r = rng.choice(ix, size=len(ix), replace=True)
            boots.append(100 * float(ta.correct[r].mean() - tb.correct[r].mean()))
        bb = int((ta.correct & ~tb.correct)[ix].sum())
        cc = int((~ta.correct & tb.correct)[ix].sum())
        return {"n": int(len(ix)), "tokens_median": float(np.median(tok[ix])),
                "top1_a": int(ta.correct[ix].sum()), "top1_b": int(tb.correct[ix].sum()),
                "delta_pp": 100 * float(ta.correct[ix].mean() - tb.correct[ix].mean()),
                "delta_pp_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "a_only": bb, "b_only": cc, "mcnemar_exact_p": _mcnemar_exact(bb, cc),
                "argmax_a": int(ama[ix].sum()), "argmax_b": int(amb[ix].sum()),
                "argmax_delta_pp": 100 * float(ama[ix].mean() - amb[ix].mean()),
                "noul_abstain_a": int(ta.abstain[ix].sum()), "noul_abstain_b": int(tb.abstain[ix].sum()),
                "n_choice": int(len(ch)),
                "nll_a": float(na[ch].mean()) if len(ch) else None, "nll_b": float(nb[ch].mean()) if len(ch) else None,
                "nll_delta": float(na[ch].mean() - nb[ch].mean()) if len(ch) else None}

    long_ = tok > a.cut
    res = {"schema": "ev-rescore-length/1", "cut_tokens": a.cut, "tokens_source": "preds usage.input_tokens (arm a)",
           "tokens_identical_across_arms": same_tokens, "all": cell(np.ones(len(tok), bool)),
           f"gt_{a.cut}": cell(long_), f"le_{a.cut}": cell(~long_), "families": {}}
    for f in [x for x in a.families.split(",") if x]:
        m = fam == f
        res["families"][f] = {"all": cell(m), f"gt_{a.cut}": cell(m & long_), f"le_{a.cut}": cell(m & ~long_)}
    others = ~np.isin(fam, [x for x in a.families.split(",") if x])
    res[f"gt_{a.cut}_excluding_named_families"] = cell(long_ & others)
    res["families_with_long_items"] = {f: int((long_ & (fam == f)).sum()) for f in sorted(set(fam)) if
                                       (long_ & (fam == f)).any()}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

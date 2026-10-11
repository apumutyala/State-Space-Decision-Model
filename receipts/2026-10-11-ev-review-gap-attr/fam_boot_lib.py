"""Paired stratified bootstrap of per-family shares of the I gap (reviewer, claude/ev-review-gap-attr).
Exact linear per-item decomposition of I (valid while no type CC is clipped at 0; checked per resample vs jevkit.aggregate)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from jevkit import scoring, spec
from jevkit.items import load_public
sys.path.insert(0, 'scripts')
from gap_attribution import load_preds, sha

cfg = json.load(open('receipts/2026-10-11-gap-attribution/cfg.json', encoding='utf-8'))
items = [it for it in load_public() if it.expected is not None]
fam = np.array([it.family for it in items]); FAMS = sorted(set(fam))
SIX = ["probability", "temporal_numeric", "long_policy", "trap", "adequacy", "judge_hard"]
T = {}
for n in ["h2o_zbtbj", "stock4b_shim_zbtbj", "D19", "jev1_native"]:
    s = cfg["systems"][n]; assert sha(s["preds"]) == s["sha256"]
    T[n] = scoring.build_table(items, load_preds(s["preds"]))
b = T["h2o_zbtbj"]
assert (b.side == 0).all()
W = {}  # per type, sum of tier weights present
for ty in spec.TYPES:
    W[ty] = sum(spec.TIER_WEIGHTS[tr] for tr in spec.TIERS if ((b.typ == scoring.TYPE_IX[ty]) & (b.tier == scoring.TIER_IX[tr])).any())

def contrib(c, idx):
    """per-occurrence contribution to I(c) - I(b) on resample idx (positions)."""
    d = np.zeros(len(idx))
    typ, tier = b.typ[idx], b.tier[idx]
    for ty in spec.TYPES:
        ti = scoring.TYPE_IX[ty]
        for tr in spec.TIERS:
            m = (typ == ti) & (tier == scoring.TIER_IX[tr])
            if not m.any():
                continue
            j = idx[m]; n = m.sum(); w = spec.TIER_WEIGHTS[tr] / W[ty] / 3.0
            if ty == "choice":
                cbar = b.chance[j].mean()
                d[m] = w * 100.0 * (c.correct[j].astype(float) - b.correct[j]) / (n * (1 - cbar))
            elif ty == "noul":
                d[m] = w * 200.0 * (c.correct[j].astype(float) - b.correct[j]) / n
            else:
                d[m] = -w * 100.0 * (c.nmae[j] - b.nmae[j]) / (n * b.nmae_chance[j].mean())
    return d


"""Gap attribution vs H2O-Lightning-4B v1.1 on the 231 public JevBench items (EV-PREREG-COMPOSITE-LOCAL quantities).

Read-only analysis over already-landed preds (sha-pinned below). For each candidate vs its baseline:
  * axes I/C (recomputed by the frozen jevkit scorer, cross-checked against the landed reports), S/K as landed;
  * single-axis counterfactuals: Composite A with axis X set to the baseline's value;
  * exact Shapley split of the Composite-A gap over the four axes (24 orderings), plus a paired stratified bootstrap
    (jevkit's own draws: default_rng(16), strata of the baseline table, B 1000) of the I share;
  * per type: CC per type and its contribution to the I gap (I = mean over the 3 types of max(0, CC_type));
  * per family: jevkit-correct counts, noul abstentions, and the family-swap counterfactual (the candidate's answers on
    family f replaced by the baseline's; I, C and Composite A recomputed, S/K held at the candidate's).
Usage: python -I -X utf8 scripts/gap_attribution.py --config cfg.json --out gap_attribution.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import scoring  # noqa: E402
from jevkit.items import load_public  # noqa: E402

AX = ("I", "C", "S", "K")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_preds(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r.get("answer") if r.get("status", "ok") == "ok" else None
    return out


def comp(a):
    return scoring.composite(a["I"], a["C"], a["S"], a["K"], "A")


def ic(table, idx=None):
    r = scoring.aggregate(table, idx, detail=idx is None)
    return r


def shapley(cand, base):
    tot = {x: 0.0 for x in AX}
    perms = list(itertools.permutations(AX))
    for order in perms:
        cur = dict(cand)
        prev = comp(cur)
        for x in order:
            cur[x] = base[x]
            now = comp(cur)
            tot[x] += now - prev
            prev = now
    return {x: tot[x] / len(perms) for x in AX}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    items = [it for it in load_public() if it.expected is not None]
    fam = {it.id: it.family for it in items}
    typ = {it.id: it.type for it in items}
    systems = {}
    for name, s in cfg["systems"].items():
        h = sha(s["preds"])
        if h != s["sha256"]:
            raise SystemExit(f"{name}: sha mismatch {h} != {s['sha256']}")
        ans = load_preds(s["preds"])
        t = scoring.build_table(items, ans)
        r = ic(t)
        axes = {"I": r["intelligence"]["I"], "C": r["calibration"]["score"], "S": s["S"], "K": s["K"]}
        chk = {x: abs(axes[x] - s["report"][x]) for x in ("I", "C")}
        if max(chk.values()) > 1e-9:
            raise SystemExit(f"{name}: I/C differ from landed report {chk}")
        per_type = {k.split("|")[1]: v["cc"] for k, v in r["intelligence"]["per_type_split"].items()}
        systems[name] = {"ans": ans, "t": t, "axes": axes, "per_type": per_type, "src": s}

    out = {"schema": "gap-attribution/1", "n_items": len(items), "B": a.b, "seed": a.seed, "systems": {}, "pairs": {}}
    for name, s in systems.items():
        t = s["t"]
        out["systems"][name] = {"axes": s["axes"], "composite_A": comp(s["axes"]), "per_type_cc": s["per_type"],
                                "preds_sha256": s["src"]["sha256"], "source": s["src"]["source"],
                                "top1_jevkit": int(t.correct.sum()),
                                "noul_abstain": int(t.abstain.sum()), "noul_n": int((t.typ == scoring.TYPE_IX["noul"]).sum())}

    for pair in cfg["pairs"]:
        cn, bn = pair["cand"], pair["base"]
        c, b = systems[cn], systems[bn]
        ca, ba = c["axes"], b["axes"]
        d = comp(ca) - comp(ba)
        single = {}
        for x in AX:
            cf = dict(ca)
            cf[x] = ba[x]
            single[x] = comp(cf) - comp(ca)
        sh = shapley(ca, ba)
        # per type
        pt = {k: {"cand": c["per_type"].get(k), "base": b["per_type"].get(k),
                  "dI_contrib": ((c["per_type"].get(k) or 0) - (b["per_type"].get(k) or 0)) / 3.0} for k in ("choice", "noul", "score")}
        # per family
        fams = sorted(set(fam.values()))
        ids = c["t"].ids
        pf = {}
        for f in fams:
            m = np.array([fam[i] == f for i in ids])
            swap = dict(c["ans"])
            for i in ids:
                if fam[i] == f:
                    swap[i] = b["ans"].get(i)
            ts = scoring.build_table(items, swap)
            rs = ic(ts)
            ax = {"I": rs["intelligence"]["I"], "C": rs["calibration"]["score"], "S": ca["S"], "K": ca["K"]}
            ty = sorted({typ[i] for i in ids if fam[i] == f})
            pf[f] = {"n": int(m.sum()), "types": ty,
                     "correct_cand": int(c["t"].correct[m].sum()), "correct_base": int(b["t"].correct[m].sum()),
                     "abstain_cand": int(c["t"].abstain[m].sum()), "abstain_base": int(b["t"].abstain[m].sum()),
                     "swap_dI": ax["I"] - ca["I"], "swap_dC": ax["C"] - ca["C"], "swap_dA": comp(ax) - comp(ca)}
        # bootstrap (jevkit draws: baseline strata, default_rng(seed))
        rng = np.random.default_rng(a.seed)
        st = scoring.strata(b["t"])
        dA, shI, shC, dI = [], [], [], []
        for _ in range(a.b):
            idx = np.concatenate([rng.choice(s_, size=len(s_), replace=True) for s_ in st])
            rb = scoring.aggregate(b["t"], idx, detail=False)
            rc = scoring.aggregate(c["t"], idx, detail=False)
            bx = {"I": rb["intelligence"]["I"], "C": rb["calibration"]["score"], "S": ba["S"], "K": ba["K"]}
            cx = {"I": rc["intelligence"]["I"], "C": rc["calibration"]["score"], "S": ca["S"], "K": ca["K"]}
            dA.append(comp(cx) - comp(bx))
            dI.append(cx["I"] - bx["I"])
            s_ = shapley(cx, bx)
            shI.append(s_["I"])
            shC.append(s_["C"])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out["pairs"][f"{cn} - {bn}"] = {
            "cand_axes": ca, "base_axes": ba, "dA": d, "dA_ci95": ci(dA), "dI_ci95": ci(dI),
            "single_axis_set_to_base": single, "shapley": sh, "shapley_sum": sum(sh.values()),
            "shapley_I_ci95": ci(shI), "shapley_C_ci95": ci(shC),
            "shapley_I_share": sh["I"] / d if d else None,
            "per_type": pt, "per_family": pf,
        }
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {"dA": v["dA"], "dA_ci95": v["dA_ci95"], "shapley": v["shapley"]} for k, v in out["pairs"].items()}, indent=1))


if __name__ == "__main__":
    main()

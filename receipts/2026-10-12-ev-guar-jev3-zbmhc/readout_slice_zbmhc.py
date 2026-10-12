"""DIAGNOSTIC (non-public, not pre-registered, changes no verdict): readout comparison on the zbmhc jev3 slice.

Arms (same transforms as guarantees_jev3_zbmhc.py): native_tfit (frozen zbmhc per-type T), native_shimcal (vendored
shim T 0.8 + yes/no floor 0.801), native (raw). Per arm, on the slice rows (all 1,200, and the 840 non-dev-card rows):
  NLL = mean -ln p_gold (p clipped at 1e-12), overall and per type; top-1 = argmax == gold;
  noul band = share of yes/no rows with 0.2 < P(yes) < 0.8 (jevkit counts these as wrong);
  slice-proxy I / C = jevkit.scoring formulas with the slice treated as ONE split and ONE tier (so no tier weights,
  no open/sealed gap penalty), types equal weight:
    CC_choice = 100 (acc - mean 1/K) / (1 - mean 1/K), acc = argmax correct;
    CC_noul   = 100 (acc - 0.5) / 0.5, acc = decisive correct (P(yes) >= 0.8 -> yes, <= 0.2 -> no, else wrong);
    CC_score  = 100 (1 - mean nMAE / mean nMAE_chance), nMAE = |E[level] - gold| / (K - 1);
    I = mean_t max(0, CC_t);
    C = mean_t of: choice 100 (1 - 2 ECE(top conf, top correct)) (all rows; the slice has no tiers),
        noul 100 (1 - 2 ECE(P(yes), gold yes)), score 50 max(0, 1 - 2 topECE) + 50 max(0, 1 - mean nRPS); ECE 10 bins;
    capability-like = (I + C) / 2;  compositeA-like = harmonic mean of (I, C) x min(1, I / 50)^2  (Composite A with the
    Speed and Cost axes dropped; no S/K).
Paired bootstrap: B = 1000, numpy default_rng(16), rows resampled within type (jevkit strata = type here), the same
resample for every arm; 95 % percentile CIs of shimcal - tfit, shimcal - raw and tfit - raw.

    python -I -X utf8 readout_slice_zbmhc.py --repo <repo> --calib <calib_probs.jsonl> --cell <cell.json> \
        --tfit <tfit.json> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guarantees_jev3_zbmhc as Z  # noqa: E402

TYPES = ("choice", "noul", "score")
PAIRS = (("native_shimcal", "native_tfit"), ("native_shimcal", "native"), ("native_tfit", "native"))


def per_row(rows, probs):
    """Per-row arrays used by every metric."""
    n = len(rows)
    out = {k: np.zeros(n) for k in ("nll", "top1", "chance", "noul_ok", "band", "pyes", "yes", "conf", "nmae",
                                    "nmae_chance", "nrps")}
    for i, (r, p) in enumerate(zip(rows, probs)):
        g = r["gold"]
        out["nll"][i] = -math.log(max(p[g], 1e-12))
        am = max(range(len(p)), key=p.__getitem__)
        out["top1"][i] = am == g
        out["conf"][i] = max(p)
        if r["type"] == "choice":
            out["chance"][i] = 1.0 / len(p)
        elif r["type"] == "noul":
            iy = r["option_ids"].index("true")
            py = p[iy]
            out["pyes"][i], out["yes"][i] = py, g == iy
            out["band"][i] = 0.2 < py < 0.8
            pred = iy if py >= 0.8 else ((1 - iy) if py <= 0.2 else None)
            out["noul_ok"][i] = pred == g
        else:
            k = len(p)
            el = sum(lv * q for lv, q in enumerate(p))
            out["nmae"][i] = abs(el - g) / (k - 1)
            out["nmae_chance"][i] = sum(abs(lv - g) for lv in range(k)) / k / (k - 1)
            cum, rps = 0.0, 0.0
            for lv in range(k - 1):
                cum += p[lv]
                rps += (cum - (1.0 if g <= lv else 0.0)) ** 2
            out["nrps"][i] = rps / (k - 1)
    return out


def metrics(a, typ, idx, S):
    m = {}
    sel = lambda t: idx[typ[idx] == t]  # noqa: E731
    m["nll"] = float(a["nll"][idx].mean())
    m["top1"] = float(a["top1"][idx].mean())
    cc, cal = {}, {}
    for t in TYPES:
        j = sel(t)
        m[f"nll_{t}"] = float(a["nll"][j].mean()) if len(j) else None
        m[f"top1_{t}"] = float(a["top1"][j].mean()) if len(j) else None
        if not len(j):
            continue
        if t == "choice":
            cb = a["chance"][j].mean()
            cc[t] = 100 * (a["top1"][j].mean() - cb) / (1 - cb)
            cal[t] = S.calibration_choice(S.ece(a["conf"][j], a["top1"][j].astype(bool)))
        elif t == "noul":
            cc[t] = 100 * (a["noul_ok"][j].mean() - 0.5) / 0.5
            cal[t] = S.calibration_noul(S.ece(a["pyes"][j], a["yes"][j].astype(bool)))
            m["noul_band"] = float(a["band"][j].mean())
        else:
            cc[t] = 100 * (1 - a["nmae"][j].mean() / a["nmae_chance"][j].mean())
            cal[t] = S.calibration_score(S.ece(a["conf"][j], a["top1"][j].astype(bool)), float(a["nrps"][j].mean()))
    for t in cc:
        m[f"cc_{t}"] = float(cc[t])
        m[f"cal_{t}"] = float(cal[t])
    i_ = float(np.mean([max(0.0, v) for v in cc.values()]))
    c_ = float(np.mean(list(cal.values())))
    m["I"], m["C"] = i_, c_
    m["capability_like"] = (i_ + c_) / 2
    m["compositeA_like"] = 0.0 if min(i_, c_) <= 0 else (2 / (1 / i_ + 1 / c_)) * min(1.0, i_ / 50) ** 2
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("repo", "calib", "cell", "tfit", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import jevkit.scoring as S
    B = Z.load_base()
    rows = B.load_rows(Path(a.calib))
    assert B.record_ids_sha(rows) == Z.IDS_SHA
    tmap = json.loads(Path(a.tfit).read_text(encoding="utf-8"))["per_type"]
    dev = set(json.loads(Path(a.cell).read_text(encoding="utf-8"))["dev_card_sample"]["ids"])
    arms = {arm: per_row(rows, [Z.arm_probs(B, r, arm, tmap) for r in rows]) for arm in Z.ARMS}
    typ = np.array([r["type"] for r in rows])
    out = {"note": "DIAGNOSTIC, non-public; not pre-registered; changes nothing about zbmhc's verdict. The slice was "
                   "used for zbmhc's checkpoint selection (360 dev-card rows) and its T fit (all 1,200 rows): the "
                   "native_tfit arm is optimistically biased on it.",
           "calib_probs_sha256": hashlib.sha256(Path(a.calib).read_bytes()).hexdigest(), "B": 1000, "seed": 16,
           "subsets": {}}
    for sub in ("all", "remainder"):
        idx = np.array([i for i, r in enumerate(rows) if sub == "all" or r["id"] not in dev])
        point = {arm: metrics(arms[arm], typ, idx, S) for arm in Z.ARMS}
        rng = np.random.default_rng(16)
        strata = [idx[typ[idx] == t] for t in TYPES if (typ[idx] == t).any()]
        draws = {p: {} for p in PAIRS}
        for _ in range(1000):
            rs = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            mm = {arm: metrics(arms[arm], typ, rs, S) for arm in Z.ARMS}
            for x, y in PAIRS:
                for k in mm[x]:
                    if mm[x][k] is not None and mm[y][k] is not None:
                        draws[(x, y)].setdefault(k, []).append(mm[x][k] - mm[y][k])
        diffs = {}
        for (x, y), d in draws.items():
            diffs[f"{x} - {y}"] = {k: {"point": (point[x][k] - point[y][k]) if point[x][k] is not None and point[y][k] is not None else None,
                                       "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                                   for k, v in d.items()}
        out["subsets"][sub] = {"n": int(len(idx)), "n_by_type": {t: int((typ[idx] == t).sum()) for t in TYPES},
                               "point": point, "paired_diffs": diffs}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

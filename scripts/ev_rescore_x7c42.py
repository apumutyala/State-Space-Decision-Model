"""Evidence-lane extras for the independent rescore of JEV2-H2O = job-x7c42 (A-PREREG-JEV2-H2O + amendments 1-3).

The jevkit numbers (per arm, paired B1000 seed16, secondaries) come from ``scripts/ev_rescore_jevkit.py``. This script
adds what that one does not compute, all from raw files (artifact = untrusted data; run with ``python -I``):

  primary : the SAME draws as ev_rescore_jevkit.paired (default_rng(16), strata of the stock_shim table) ->
            Composite-A delta vector -> 95 % and 97.5 % (Holm over 2 arms) percentile CIs + one-sided bootstrap p;
            per-type cc deltas; per-family I / top-1 deltas on public-231 (same draws); noul 0.2-0.8 band per arm.
  control : stock_shim answers vs job-4g6m8's and job-zbtbj's H2O preds (bitwise per item).
  floor   : slice_floor_rows.jsonl (3,600 slice decisions): paired bootstrap of top-1 (selected - step 0), exact
            McNemar, per family / type; native column recomputed from calib_probs.jsonl + local gold.
  overlap : slice ids sha, train ids sha, id / source-item / content overlap slice-train and slice|train-public.
  order   : tgz member mtimes + MANIFEST times (selection vs the first public-231 pred).
  determinism: x7c42 dev cards vs job-9fy9p's cell.json (bitwise), train_log loss divergence.
  transfer: per family / per type slice delta-top1 next to public delta-top1 / delta-I, + schema-level shape stats.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402

ARMS = ("stock_shim", "native_shimcal", "native", "native_tfit")
QS = (0.5, 1, 1.25, 2.5, 5, 25, 50, 75, 95, 97.5, 98.75, 99, 99.5)


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rj(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def rgz(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def pct(v, qs=(2.5, 97.5)):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return [float(np.percentile(v, q)) for q in qs] if v else None


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / 2.0 ** n))


def top1(t, items) -> np.ndarray:
    """argmax top-1: choice / score top label == gold; noul P(yes) >= 0.5 <-> gold yes (abstention not penalised)."""
    out = np.zeros(len(t.ids), dtype=bool)
    for i, it in enumerate(items):
        if it.type == "noul":
            out[i] = bool(t.valid[i]) and ((t.pyes[i] >= 0.5) == bool(t.yes[i]))
        else:
            out[i] = bool(t.top_correct[i])
    return out


def type_cc(t, idx=None) -> dict:
    r = JS.aggregate(t, idx, detail=True)
    pts = r["intelligence"].get("per_type_split") or {}
    return {k.split("|")[1]: v["cc_unclipped"] for k, v in pts.items() if k.startswith("open|")}


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def norm_text(x) -> str:
    s = canon(x) if not isinstance(x, str) else x
    return " ".join(s.casefold().split())


def tgz_times(p: Path) -> list[dict]:
    out = []
    with tarfile.open(p, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                out.append({"member": m.name, "mtime_utc": datetime.fromtimestamp(m.mtime, timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"), "epoch": m.mtime, "bytes": m.size})
    return sorted(out, key=lambda r: (r["epoch"], r["member"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True, help="extracted phases/final (holds out/, logs/)")
    ap.add_argument("--art-root", required=True, help="extracted tar root (MANIFEST.jsonl, phases/*.tgz)")
    ap.add_argument("--g4m8", required=True, help="job-4g6m8 control preds (arms/h2o/preds.jsonl, sha e00d5ac7)")
    ap.add_argument("--zbtbj", required=True, help="job-zbtbj h2o-lightning-4b/h2o_shim/preds.jsonl (sha 60a06842)")
    ap.add_argument("--calib-records", required=True, help="bschema-draw-jev2h-final-calib/records.jsonl.gz")
    ap.add_argument("--train-records", required=True, help="bschema-draw-jev2h-final/records.jsonl.gz")
    ap.add_argument("--cell-9fy9p", required=True, help="job-9fy9p out/cell.json")
    ap.add_argument("--trainlog-9fy9p", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    art, root = Path(a.art), Path(a.art_root)
    jd = art / "out" / "screen" / "jevkit"
    res: dict = {"schema": "ev-rescore-x7c42-extra/1", "B": a.B, "seed": a.seed}

    # ------------------------------------------------------------------ public-231 tables
    items = [it for it in load_public() if it.expected is not None]
    preds = {n: rj(jd / f"{n}.preds.jsonl") for n in ARMS}
    T = {n: JS.build_table(items, preds_to_answers(preds[n])) for n in ARMS}
    for n in ARMS:
        assert T[n].ids == [it.id for it in items]
    S = {}
    for n in ARMS:
        rep = json.loads((ROOT / "receipts/2026-10-11-ev-x7c42-rescore/reports" / f"{n}.rescore.jevkit.json")
                         .read_text(encoding="utf-8"))
        S[n] = (rep["axes"]["speed"], rep["axes"]["cost"])
    fam = np.array([it.family for it in items])
    typ = np.array([it.type for it in items])
    fams = sorted(set(fam))
    types = ["choice", "noul", "score"]
    tops = {n: top1(T[n], items) for n in ARMS}

    def axes(n, idx=None):
        r = JS.aggregate(T[n], idx, detail=False)
        i, c = r["intelligence"]["I"], r["calibration"]["score"]
        s, k = S[n]
        return i, c, JS.composite(i, c, s, k, "A")

    base = "stock_shim"
    others = [n for n in ARMS if n != base]
    rng = np.random.default_rng(a.seed)
    st = JS.strata(T[base])
    dA = {n: [] for n in others}
    dI = {n: [] for n in others}
    dcc = {n: {t: [] for t in types} for n in others}
    dfamI = {n: {f: [] for f in fams} for n in others}
    dfamT = {n: {f: [] for f in fams} for n in others}
    dtypT = {n: {t: [] for t in types} for n in others}
    for _ in range(a.B):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        ib, cb, Ab = axes(base, idx)
        ccb = type_cc(T[base], idx)
        for n in others:
            ia, ca, Aa = axes(n, idx)
            dA[n].append(Aa - Ab)
            dI[n].append(ia - ib)
            cca = type_cc(T[n], idx)
            for t in types:
                if cca.get(t) is not None and ccb.get(t) is not None:
                    dcc[n][t].append(cca[t] - ccb[t])
                m = idx[typ[idx] == t]
                if len(m):
                    dtypT[n][t].append(float(tops[n][m].mean() - tops[base][m].mean()))
            if n == "native_shimcal":
                for f in fams:
                    m = idx[fam[idx] == f]
                    if not len(m):
                        continue
                    dfamT[n][f].append(float(tops[n][m].mean() - tops[base][m].mean()))
                    ra = JS.aggregate(T[n], m, detail=False)["intelligence"]["I"]
                    rb = JS.aggregate(T[base], m, detail=False)["intelligence"]["I"]
                    if ra is not None and rb is not None:
                        dfamI[n][f].append(ra - rb)
    full = {n: axes(n) for n in ARMS}
    fcc = {n: type_cc(T[n]) for n in ARMS}
    prim = {}
    for n in others:
        v = np.array(dA[n])
        prim[n] = {"point": full[n][2] - full[base][2], "a": full[n][2], "b": full[base][2],
                   "ci95": pct(v), "ci975_holm": pct(v, (1.25, 98.75)), "ci99": pct(v, (0.5, 99.5)),
                   "p_one_sided_boot": float(np.mean(v <= 0)),
                   "p_one_sided_boot_plus1": float((1 + np.sum(v <= 0)) / (a.B + 1)),
                   "quantiles": {str(q): float(np.percentile(v, q)) for q in QS},
                   "dI_point": full[n][0] - full[base][0], "dI_ci95": pct(dI[n]),
                   "per_type_cc": {t: {"a": fcc[n].get(t), "b": fcc[base].get(t),
                                       "delta": None if fcc[n].get(t) is None else fcc[n][t] - fcc[base][t],
                                       "ci95": pct(dcc[n][t])} for t in types},
                   "per_type_top1": {t: {"n": int((typ == t).sum()), "a": int(tops[n][typ == t].sum()),
                                         "b": int(tops[base][typ == t].sum()),
                                         "delta_pp": 100 * float(tops[n][typ == t].mean() - tops[base][typ == t].mean()),
                                         "ci95_pp": [100 * x for x in pct(dtypT[n][t])],
                                         "a_only": int((tops[n] & ~tops[base] & (typ == t)).sum()),
                                         "b_only": int((~tops[n] & tops[base] & (typ == t)).sum())}
                                     for t in types}}
        for t in types:
            d = prim[n]["per_type_top1"][t]
            d["mcnemar_p"] = mcnemar(d["a_only"], d["b_only"])
    res["primary"] = prim
    res["top1_public"] = {n: int(tops[n].sum()) for n in ARMS}
    nm = typ == "noul"
    res["noul_band"] = {n: {"n_noul": int(nm.sum()), "band_0.2_0.8": int((T[n].abstain & nm).sum()),
                            "rate": float((T[n].abstain & nm).mean() / nm.mean())} for n in ARMS}

    # per-family public block (native_shimcal - stock_shim)
    n = "native_shimcal"
    pubfam = {}
    for f in fams:
        m = fam == f
        ia = JS.aggregate(T[n], np.flatnonzero(m), detail=False)["intelligence"]["I"]
        ib = JS.aggregate(T[base], np.flatnonzero(m), detail=False)["intelligence"]["I"]
        b_, c_ = int((tops[n] & ~tops[base] & m).sum()), int((~tops[n] & tops[base] & m).sum())
        pubfam[f] = {"n": int(m.sum()), "types": {t: int((m & (typ == t)).sum()) for t in types if (m & (typ == t)).any()},
                     "control_top1": float(tops[base][m].mean()), "arm_top1": float(tops[n][m].mean()),
                     "delta_top1_pp": 100 * float(tops[n][m].mean() - tops[base][m].mean()),
                     "delta_top1_ci95_pp": [100 * x for x in (pct(dfamT[n][f]) or [float("nan")] * 2)],
                     "a_only": b_, "b_only": c_, "mcnemar_p": mcnemar(b_, c_),
                     "I_control": ib, "I_arm": ia, "delta_I": None if ia is None or ib is None else ia - ib,
                     "delta_I_ci95": pct(dfamI[n][f]),
                     "noul_band_arm": int((T[n].abstain & m & nm).sum()),
                     "noul_band_control": int((T[base].abstain & m & nm).sum())}
    res["public_by_family"] = pubfam

    # ------------------------------------------------------------------ control identity vs 4g6m8 / zbtbj
    ctrl = {p["id"]: p for p in preds[base]}
    cmpo = {}
    for nm_, pth in (("job-4g6m8", Path(a.g4m8)), ("job-zbtbj", Path(a.zbtbj))):
        other = {p["id"]: p for p in rj(pth)}
        eq = sum(1 for i in ctrl if i in other and canon(ctrl[i].get("answer")) == canon(other[i].get("answer")))
        to = JS.build_table(items, preds_to_answers(list(other.values())))
        t1 = top1(to, items)
        cmpo[nm_] = {"preds_sha256": sha_b(pth.read_bytes()), "n_common": len(set(ctrl) & set(other)),
                     "answers_bitwise_equal": eq, "top1_equal_items": int((t1 == tops[base]).sum()),
                     "top1": int(t1.sum())}
    res["control_identity"] = cmpo

    # ------------------------------------------------------------------ slice floor
    rows = rj(jd / "slice_floor_rows.jsonl")
    calib = {p["id"]: p for p in rj(jd / "calib_probs.jsonl")}
    crec = rgz(Path(a.calib_records))
    trec = rgz(Path(a.train_records))
    ids_rows = [r["id"] for r in rows]
    nat = np.array([bool(r["native"]) for r in rows])
    s0 = np.array([bool(r["step0"]) for r in rows])
    rfam = np.array([r["family"] for r in rows])
    rtyp = np.array([r["type"] for r in rows])
    # recompute native from calib_probs (+ gold from the local calib records)
    gold_rec = {}
    for r in crec:
        for qid, q in (r.get("questions") or {}).items():
            gold_rec[f"{r['id']}/{qid}"] = str((r.get("labels") or {}).get(qid))
    rec_nat, gold_mis, missing = [], 0, 0
    for r in rows:
        c = calib.get(r["id"])
        if c is None:
            missing += 1
            rec_nat.append(None)
            continue
        am = int(np.argmax(c["probs"]))
        rec_nat.append(am == int(c["gold"]))
        if str(c["option_ids"][int(c["gold"])]) != gold_rec.get(r["id"]):
            gold_mis += 1
    rec_nat_ok = sum(1 for x, y in zip(rec_nat, nat) if x is not None and bool(x) == bool(y))
    rng = np.random.default_rng(a.seed)
    n_ = len(rows)
    dd, dds = [], []
    fi = {f: np.flatnonzero(rfam == f) for f in sorted(set(rfam))}
    for _ in range(a.B):
        ix = rng.integers(0, n_, n_)
        dd.append(float(nat[ix].mean() - s0[ix].mean()))
    rng2 = np.random.default_rng(a.seed)
    fdd = {f: [] for f in fi}
    tdd = {t: [] for t in types}
    ti = {t: np.flatnonzero(rtyp == t) for t in types}
    for _ in range(a.B):
        parts = []
        for f, ii in fi.items():
            s_ = rng2.choice(ii, size=len(ii), replace=True)
            parts.append(s_)
            fdd[f].append(float(nat[s_].mean() - s0[s_].mean()))
        ix = np.concatenate(parts)
        dds.append(float(nat[ix].mean() - s0[ix].mean()))
    rng3 = np.random.default_rng(a.seed)
    for _ in range(a.B):
        for t, ii in ti.items():
            if len(ii):
                s_ = rng3.choice(ii, size=len(ii), replace=True)
                tdd[t].append(float(nat[s_].mean() - s0[s_].mean()))
    b_, c_ = int((nat & ~s0).sum()), int((~nat & s0).sum())
    dev_ids = set(json.loads((art / "out" / "cell.json").read_text(encoding="utf-8"))["dev_card_sample"]["ids"])
    dm = np.array([i.rsplit("/", 1)[0] in dev_ids or i in dev_ids for i in ids_rows])
    floor = {"n_decisions": n_, "n_unique_ids": len(set(ids_rows)), "selected_top1": float(nat.mean()),
             "step0_top1": float(s0.mean()), "delta_pp": 100 * float(nat.mean() - s0.mean()),
             "ci95_pp_iid_rows": [100 * x for x in pct(dd)], "ci95_pp_family_stratified": [100 * x for x in pct(dds)],
             "p_one_sided_boot": float(np.mean(np.array(dd) <= 0)),
             "selected_only": b_, "step0_only": c_, "both": int((nat & s0).sum()), "neither": int((~nat & ~s0).sum()),
             "mcnemar_exact_p": mcnemar(b_, c_),
             "native_recomputed_from_calib_probs": {"n_match": rec_nat_ok, "n_missing": missing,
                                                    "gold_mismatch_vs_local_records": gold_mis},
             "dev_sample_cross_check": {"n_dev_ids_in_rows": int(dm.sum()),
                                        "selected_top1_on_dev360": float(nat[dm].mean()) if dm.any() else None,
                                        "step0_top1_on_dev360": float(s0[dm].mean()) if dm.any() else None},
             "per_family": {}, "per_type": {}}
    for f, ii in fi.items():
        bb, cc = int((nat[ii] & ~s0[ii]).sum()), int((~nat[ii] & s0[ii]).sum())
        floor["per_family"][f] = {"n": len(ii), "step0": float(s0[ii].mean()), "selected": float(nat[ii].mean()),
                                  "delta_pp": 100 * float(nat[ii].mean() - s0[ii].mean()),
                                  "ci95_pp": [100 * x for x in pct(fdd[f])], "selected_only": bb, "step0_only": cc,
                                  "mcnemar_p": mcnemar(bb, cc),
                                  "types": {t: int((rtyp[ii] == t).sum()) for t in types if (rtyp[ii] == t).any()}}
    for t, ii in ti.items():
        if not len(ii):
            continue
        bb, cc = int((nat[ii] & ~s0[ii]).sum()), int((~nat[ii] & s0[ii]).sum())
        floor["per_type"][t] = {"n": len(ii), "step0": float(s0[ii].mean()), "selected": float(nat[ii].mean()),
                                "delta_pp": 100 * float(nat[ii].mean() - s0[ii].mean()),
                                "ci95_pp": [100 * x for x in pct(tdd[t])], "selected_only": bb, "step0_only": cc,
                                "mcnemar_p": mcnemar(bb, cc)}
    res["slice_floor"] = floor

    # ------------------------------------------------------------------ ids / overlap
    def idsha(ids):
        s = sorted(set(ids))
        return {"sorted_newline": sha_b("\n".join(s).encode()), "sorted_newline_trailing": sha_b(("\n".join(s) + "\n").encode()),
                "sorted_json": sha_b(json.dumps(s).encode())}
    c_ids = [r["id"] for r in crec]
    t_ids = [r["id"] for r in trec]
    row_rec_ids = {i.rsplit("/", 1)[0] for i in ids_rows}

    def src(r):
        return (r.get("source"), str(r.get("source_id")))

    def base_id(r):
        return (((r.get("aug") or {}).get("b1_view") or {}).get("base_id")) or r["id"]

    def content(r):
        return sha_b(norm_text({"state": r.get("state"), "questions": r.get("questions")}).encode())

    def state_h(x):
        return sha_b(norm_text(x).encode())
    pub_state = {state_h(it.state) for it in items}
    pub_full = {sha_b(norm_text({"state": it.state, "q": it.question}).encode()) for it in items}
    pub_ids = {it.id for it in items}
    ov = {"calib_n": len(c_ids), "calib_unique": len(set(c_ids)), "calib_ids_sha256": idsha(c_ids),
          "rows_record_ids_equal_calib_ids": row_rec_ids == set(c_ids),
          "train_n": len(t_ids), "train_unique": len(set(t_ids)), "train_ids_sha256": idsha(t_ids),
          "slice_train_id_overlap": len(set(c_ids) & set(t_ids)),
          "slice_train_base_id_overlap": len({base_id(r) for r in crec} & {base_id(r) for r in trec}),
          "slice_train_source_item_overlap": len({src(r) for r in crec} & {src(r) for r in trec}),
          "slice_train_content_overlap": len({content(r) for r in crec} & {content(r) for r in trec}),
          "slice_train_state_overlap": len({state_h(r.get("state")) for r in crec} & {state_h(r.get("state")) for r in trec}),
          "slice_public_id_overlap": len(set(c_ids) & pub_ids) + len(set(ids_rows) & pub_ids),
          "train_public_id_overlap": len(set(t_ids) & pub_ids),
          "slice_public_state_overlap": len({state_h(r.get("state")) for r in crec} & pub_state),
          "train_public_state_overlap": len({state_h(r.get("state")) for r in trec} & pub_state),
          "slice_public_state_question_overlap": len({sha_b(norm_text({"state": r.get("state"), "q": q}).encode())
                                                      for r in crec for q in (r.get("questions") or {}).values()} & pub_full),
          "train_public_state_question_overlap": len({sha_b(norm_text({"state": r.get("state"), "q": q}).encode())
                                                      for r in trec for q in (r.get("questions") or {}).values()} & pub_full)}
    # shared source items slice/train: which sources
    shared = {src(r) for r in crec} & {src(r) for r in trec}
    ov["slice_train_shared_source_items_by_source"] = {}
    for s_, _ in shared:
        ov["slice_train_shared_source_items_by_source"][s_] = ov["slice_train_shared_source_items_by_source"].get(s_, 0) + 1
    res["overlap"] = ov

    # ------------------------------------------------------------------ order of events
    man = rj(root / "MANIFEST.jsonl")
    ev = []
    for ph in ("cell", "screen", "final"):
        for m in tgz_times(root / "phases" / f"{ph}.tgz"):
            if ph == "final" or m["member"].endswith(("cell.json", "card-step0000200.json")):
                ev.append(m | {"phase_tgz": ph})
    keep = ("cell.json", "dev_cards/", "card-step", "preds.jsonl", "slice_floor", "calib_probs", "tfit.json",
            "screen.json", "identity", "train_log", "ckpt-export", "progress-screen", "g1/")
    ev = [e for e in ev if any(k in e["member"] for k in keep)]
    seen, evu = set(), []
    for e in sorted(ev, key=lambda r: (r["epoch"], r["member"])):
        if (e["member"], e["epoch"]) not in seen:
            seen.add((e["member"], e["epoch"]))
            evu.append({k: e[k] for k in ("mtime_utc", "member", "bytes")})
    pub = [e for e in evu if e["member"].endswith(".preds.jsonl") and "/jevkit/" in e["member"]]
    sel = [e for e in evu if e["member"].endswith("out/cell.json")]
    res["order"] = {"manifest": man, "events": evu,
                    "selection_cell_json_utc": sel[0]["mtime_utc"] if sel else None,
                    "first_public_pred_utc": pub[0]["mtime_utc"] if pub else None,
                    "first_public_pred": pub[0]["member"] if pub else None}

    # ------------------------------------------------------------------ determinism vs 9fy9p
    c9 = json.loads(Path(a.cell_9fy9p).read_text(encoding="utf-8"))
    cx = json.loads((art / "out" / "cell.json").read_text(encoding="utf-8"))
    det = {"dev_sample_ids_sha256": {"x7c42": sha_b("\n".join(cx["dev_card_sample"]["ids"]).encode()),
                                     "9fy9p": sha_b("\n".join(c9["dev_card_sample"]["ids"]).encode())},
           "cards": []}
    d9 = {d["step"]: d for d in c9["dev_cards"]}
    for d in cx["dev_cards"]:
        o = d9.get(d["step"], {})
        det["cards"].append({"step": d["step"], "nll_x7c42": d["nll"], "nll_9fy9p": o.get("nll"),
                             "d_nll": None if o.get("nll") is None else d["nll"] - o["nll"],
                             "top1_x7c42": d["accuracy"], "top1_9fy9p": o.get("accuracy"),
                             "d_top1_items": None if o.get("accuracy") is None else round((d["accuracy"] - o["accuracy"]) * 360),
                             "bitwise": o.get("nll") == d["nll"] and o.get("accuracy") == d["accuracy"]})
    det["best_x7c42"], det["best_9fy9p"] = cx["best"], c9["best"]
    det["config_equal_except_comment"] = ({k: v for k, v in cx["config"].items() if k != "_comment"}
                                          == {k: v for k, v in c9["config"].items() if k != "_comment"})
    if a.trainlog_9fy9p:
        lx = rj(art / "out" / "train_log.jsonl")
        l9 = rj(Path(a.trainlog_9fy9p))
        kx = [r for r in lx if "loss" in r]
        k9 = [r for r in l9 if "loss" in r]
        first = None
        for r1, r2 in zip(kx, k9):
            if r1.get("loss") != r2.get("loss"):
                first = {"x7c42": r1, "9fy9p": r2}
                break
        det["train_log"] = {"n_loss_rows": [len(kx), len(k9)], "first_loss_divergence": first,
                            "max_abs_loss_diff": max((abs(r1["loss"] - r2["loss"]) for r1, r2 in zip(kx, k9)), default=None)}
    res["determinism"] = det

    # ------------------------------------------------------------------ transfer + shape stats
    rec_by = {r["id"]: r for r in crec}
    tok = {p["id"]: (p.get("usage") or {}).get("input_tokens") for p in preds[base]}

    def shape(states, ks, tps, lens, toks=None):
        q = lambda v: [float(np.percentile(v, x)) for x in (10, 50, 90)] if len(v) else None
        tmix = {t: tps.count(t) for t in types if tps.count(t)}
        stt = {}
        for s_ in states:
            k = type(s_).__name__
            stt[k] = stt.get(k, 0) + 1
        out = {"n": len(tps), "type_mix": tmix, "state_type": stt, "K_values": sorted(set(ks)),
               "K_mean": float(np.mean(ks)) if ks else None, "state_chars_p10_50_90": q(lens)}
        if toks is not None:
            out["input_tokens_p10_50_90"] = q([x for x in toks if x is not None])
        return out
    trans = {}
    for f in sorted(set(fams) | set(fi)):
        row = {"slice": None, "public": None, "shape_slice": None, "shape_public": None}
        if f in fi:
            ii = fi[f]
            row["slice"] = floor["per_family"][f]
            st_, ks, tps, ls = [], [], [], []
            for j in ii:
                rid, qid = ids_rows[j].rsplit("/", 1)
                r = rec_by.get(rid)
                if r is None:
                    continue
                q_ = r["questions"][qid]
                st_.append(r.get("state"))
                tps.append(q_["type"])
                ks.append(len(q_.get("criteria") or {}) if q_["type"] != "noul" else 2)
                ls.append(len(canon(r.get("state"))))
            row["shape_slice"] = shape(st_, ks, tps, ls)
        if f in pubfam:
            row["public"] = pubfam[f]
            its = [it for it in items if it.family == f]
            row["shape_public"] = shape([it.state for it in its], [it.k if it.type != "noul" else 2 for it in its],
                                        [it.type for it in its], [len(canon(it.state)) for it in its],
                                        [tok.get(it.id) for it in its])
        s_ok = row["slice"] is not None and row["slice"]["ci95_pp"][0] > 0
        p_d = row["public"]["delta_top1_pp"] if row["public"] else None
        row["flag_gain_slice_not_public"] = bool(s_ok and p_d is not None and p_d <= 0)
        trans[f] = row
    res["transfer_by_family"] = trans
    res["transfer_by_type"] = {t: {"slice": floor["per_type"].get(t), "public_top1": prim["native_shimcal"]["per_type_top1"][t],
                                   "public_cc": prim["native_shimcal"]["per_type_cc"][t]} for t in types}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

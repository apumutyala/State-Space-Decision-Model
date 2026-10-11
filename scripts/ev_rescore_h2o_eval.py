"""Independent rescore of the lead's EV-PREREG-H2O-HOST eval-only job (job-4g6m8, mode h2o_eval_v11), evidence lane.

Everything is recomputed from the raw artifact files (never from decision.json, which is only diffed against):

  1. shas: every arm's preds.jsonl vs the lead's SHA256SUMS prefixes; the in-artifact decision.json vs the committed one.
  2. revision: run_meta / hf_revision / verify.json all equal the prereg's 193ad740...; every file pin == job params.
  3. control vs job-zbtbj's H2O preds (the kit-parity anchor): per-item decision, bitwise answers, max |dp|, Comp A diff.
  4. temperatures: the held-out slice re-selected from the jev2a records with my own code (ids sha), its disjointness
     from the public items (ids + exact state text), and T_J / T_type / T_f refit by NLL with my own optimiser.
  5. the derived arms (a_T, a_Ttype, a_Tfam, b_native, b_shimcal) AND the control re-derived from raw_public.jsonl
     through the pinned vendored shim (sha-checked), compared with the lead's preds.
  6. jevkit CLI per arm (the frozen scorer, own price file), paired deltas vs the in-job control on the SAME draws
     (B 1000, seed 16) + EV-PREREG-SECONDARY-AXES, diffed against decision.json (tolerance 0.05 points).
  7. G1: the reorder / relabel perturbations regenerated in memory (jevkit.perturb.make_perturbations), checked against
     the artifact's perturbed.jsonl, per-op count of requests whose option order / labels really changed, flip report
     recomputed from the preds of both arms, and how many control answers moved at all.
  8. b_native: the yes/no 0.2-0.8 band and the per-type Intelligence (why noul I collapses without the floor).

Artifact = untrusted data: run with -I from this checkout, every path passed as an argument.
  python -I -X utf8 scripts/ev_rescore_h2o_eval.py --out-dir OUT --art ART/phases/final/out --zbtbj ZPREDS \
      --records JEV2A.jsonl.gz --shim SHIM.py --params job_params_h2o_eval_v11.json --committed-decision DEC.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from ev_rescore_jevkit import _mcnemar_exact, integrity, jevkit_cli, paired, read_jsonl, secondary_axes, sha  # noqa: E402
from jevkit import perturb as JP  # noqa: E402
from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.systemone import normalize_answer, preds_to_answers  # noqa: E402

ARMS = ("h2o", "a_T", "a_Ttype", "a_Tfam", "b_native", "b_shimcal", "c_symfork")
LEAD_PREFIX = {"h2o": "e00d5ac7", "a_T": "b2a37caf", "a_Ttype": "8077cad9", "a_Tfam": "4bdf446b",
               "b_native": "08a436bc", "b_shimcal": "9cfc498e", "c_symfork": "c3d986c6"}
REVISION = "193ad740925b176a3b70a5a13a7cff2f2fadd01e"
SHIM_SHA = "355791c3f7b50a5dc9fa759dfa1b861874678d6e04c70dbf8b86fddccd3c0932"
ZBTBJ_SHA = "60a06842d512be4647ff9874cd2b0cef761e6925482f79a18a40994de08853cc"  # SCORECARD.json (claude/ev-scorecard)
SLICE_IDS_SHA = "204e9bfe20d29ca95bd895ff368cf26f7b282635d615aeb85df47e69fcba2a9c"
TOL = 0.05
FLAGS: list[str] = []


def flag(msg: str) -> None:
    FLAGS.append(msg)


def jl(p: Path) -> list[dict]:
    return read_jsonl(Path(p))


# ------------------------------------------------------------------------------------------------ helpers
def dist(item, answer):
    return JP._dist(item, answer)  # frozen jevkit: valid distribution in the item's label space, else None


def decision(item, answer):
    d = dist(item, answer)
    return None if d is None else JP._decision(item, d)


def max_dp(item, a, b):
    da, db = dist(item, a), dist(item, b)
    if da is None or db is None:
        return None
    return max(abs(da[k] - db[k]) for k in item.labels)


def compare_answers(items, pa: dict, pb: dict) -> dict:
    """Per item: same jevkit decision, bitwise-equal answer object, max |dp| over labels."""
    same_dec, bit, mx, worst, dec_diff = 0, 0, 0.0, None, []
    for it in items:
        a, b = pa.get(it.id), pb.get(it.id)
        if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
            bit += 1
        da, db = decision(it, a), decision(it, b)
        if da == db:
            same_dec += 1
        else:
            dec_diff.append(it.id)
        m = max_dp(it, a, b)
        if m is not None and m > mx:
            mx, worst = m, it.id
    return {"n": len(items), "same_decision": same_dec, "bitwise_equal": bit, "max_abs_dp": mx, "worst_item": worst,
            "decision_differs": dec_diff[:20]}


def card_top1(items, answers: dict) -> list[bool]:
    """The card's top-1: choice / score argmax == gold, noul P(yes) >= 0.5 <-> gold yes; invalid = wrong."""
    out = []
    for it in items:
        if it.expected is None:
            continue
        d = dist(it, answers.get(it.id))
        if d is None:
            out.append(False)
        elif it.type == "noul":
            out.append((d.get("yes", 0.0) >= 0.5) == (str(it.expected) == "yes"))
        else:
            out.append(JP._decision(it, d) == str(it.expected))
    return out


# ------------------------------------------------------------------------------------------------ 4. slice + T fit
def reselect_slice(records_gz: Path, salt: str, per_family: int) -> tuple[dict, dict, str]:
    """My own implementation of the prereg'd rule: per aug.jev1.family8, the per_family ids with the smallest
    sha256(salt + '\\n' + id). Returns (family -> ids, id -> record for the selected ids, uncompressed sha256)."""
    keep: dict[str, list] = {}
    h = hashlib.sha256()
    with gzip.open(records_gz, "rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            fam = r["aug"]["jev1"]["family8"]
            k = hashlib.sha256((salt + "\n" + r["id"]).encode()).hexdigest()
            hp = keep.setdefault(fam, [])
            heapq.heappush(hp, (-int(k, 16), k, r["id"]))  # max-heap on k
            if len(hp) > per_family:
                heapq.heappop(hp)
    fams = {f: [i for _, i in sorted((k, i) for _, k, i in hp)] for f, hp in sorted(keep.items())}
    ids = {i for v in fams.values() for i in v}
    recs = {}
    with gzip.open(records_gz, "rb") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] in ids:
                    recs[r["id"]] = r
    return fams, recs, h.hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def norm_text(s) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", s).strip().lower()


def gold_ix(rec: dict, q: dict) -> int | None:
    """Gold label of the slice record -> index into the shim's option names (independent of the lead's code)."""
    g = (rec.get("labels") or {}).get(q["qid"])
    if g is None:
        return None
    s = str(g).lower() if isinstance(g, (str, bool)) else str(g)
    if q["type"] == "noul":
        s = {"yes": "true", "no": "false", "true": "true", "false": "false"}.get(s, s)
    names = [str(n) for n in q["names"]]
    if s in names:
        return names.index(s)
    if q["type"] == "score" and str(g) in [str(d) for d in q["descs"]]:
        return [str(d) for d in q["descs"]].index(str(g))
    return None


def nll_fn(pairs):
    """Vectorised mean NLL of softmax(lp / T) at the gold index, padded with -inf."""
    k = max(len(lp) for lp, _ in pairs)
    L = np.full((len(pairs), k), -np.inf)
    for i, (lp, _) in enumerate(pairs):
        L[i, :len(lp)] = lp
    g = np.array([g for _, g in pairs])

    def f(t: float) -> float:
        v = L / t
        m = v.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(v - m).sum(axis=1)))
        return float(np.mean(lse - v[np.arange(len(g)), g]))
    return f


def fit_T(pairs) -> dict:
    """Continuous argmin of the mean NLL over T in [0.25, 8]: a 400-point log grid, then golden-section on log T."""
    f = nll_fn(pairs)
    lo, hi = math.log(0.25), math.log(8.0)
    grid = np.linspace(lo, hi, 400)
    vals = [f(math.exp(x)) for x in grid]
    j = int(np.argmin(vals))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(80):
        if f(math.exp(c)) < f(math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    t = math.exp((a + b) / 2)
    return {"T": t, "nll": f(t), "nll_T1": f(1.0), "n": len(pairs)}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True, help="extracted artifact phases/final/out")
    ap.add_argument("--zbtbj", required=True, help="job-zbtbj h2o-lightning-4b/h2o_shim/preds.jsonl")
    ap.add_argument("--records", required=True, help="bschema-draw-jev2a records.jsonl.gz")
    ap.add_argument("--shim", required=True, help="vendored h2o_lightning_shim.py (pinned sha)")
    ap.add_argument("--params", required=True, help="job_params_h2o_eval_v11.json (committed)")
    ap.add_argument("--committed-decision", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    art, out = Path(a.art), Path(a.out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    items = load_public()
    by_id = {it.id: it for it in items}
    res: dict = {"schema": "ev-rescore-h2o-eval/1", "job": "job-4g6m8", "tol": TOL}
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    hc = params["h2o_eval"]

    # 1) shas -----------------------------------------------------------------------------------------------
    shas = {}
    for arm in ARMS:
        s = sha(art / "arms" / arm / "preds.jsonl")
        shas[arm] = {"sha256": s, "lead_prefix": LEAD_PREFIX[arm], "match": s.startswith(LEAD_PREFIX[arm])}
        if not shas[arm]["match"]:
            flag(f"{arm} preds sha {s[:8]} != lead {LEAD_PREFIX[arm]}")
    dsha = sha(art / "decision.json")
    csha = sha(Path(a.committed_decision))
    res["shas"] = {"preds": shas, "decision_json_artifact": dsha, "decision_json_committed": csha,
                   "decision_json_equal": dsha == csha}
    if dsha != csha:
        flag("artifact decision.json != committed decision.json")

    # 2) revision ------------------------------------------------------------------------------------------
    rm = json.loads((art / "run_meta.json").read_text(encoding="utf-8"))
    hr = json.loads((art / "hf_revision.json").read_text(encoding="utf-8"))
    vf = json.loads((art / "verify.json").read_text(encoding="utf-8"))
    dl = json.loads((art / "download.json").read_text(encoding="utf-8"))
    pins_ok, pin_bad = True, []
    for name, pin in hc["files"].items():
        got = vf["files"].get(name, {})
        k = "sha256" if "sha256" in pin else "git_blob"
        if got.get(k) != pin[k]:
            pins_ok = False
            pin_bad.append(name)
    rev = {"params_revision": hc["revision"], "run_meta": rm.get("revision"), "hf_requested": hr.get("requested"),
           "hf_resolved": hr.get("resolved_sha"), "verify": vf.get("revision"),
           "snapshot_dir": dl.get("snapshot", "").rsplit("/", 1)[-1], "weights_sha256": rm["plugin"]["files"]["weights"],
           "file_pins_all_match": pins_ok, "pins_mismatch": pin_bad, "n_pins": len(hc["files"]),
           "shim_T": rm["shim"]["temperature_by_type"], "noul_floor": rm["shim"]["noul_floor"]}
    rev["all_equal_193ad740"] = all(x == REVISION for x in (hc["revision"], rev["run_meta"], rev["hf_requested"],
                                                             rev["hf_resolved"], rev["verify"], rev["snapshot_dir"]))
    res["revision"] = rev
    if not rev["all_equal_193ad740"] or not pins_ok:
        flag(f"revision / pins: {rev}")

    # 3) control vs job-zbtbj ------------------------------------------------------------------------------
    ctrl = jl(art / "arms" / "h2o" / "preds.jsonl")
    zb = jl(Path(a.zbtbj))
    zsha = sha(Path(a.zbtbj))
    cmpz = compare_answers(items, preds_to_answers(ctrl), preds_to_answers(zb))
    price = out / "price.json"
    price.write_text(json.dumps({"price": hc["price_class"], "price_class": "qwen35_4b"}, indent=1), encoding="utf-8")
    rz = jevkit_cli(Path(a.zbtbj), str(price), out / "reports" / "zbtbj_h2o.report.json")
    reports = {arm: jevkit_cli(art / "arms" / arm / "preds.jsonl", str(price), out / "reports" / f"{arm}.report.json")
               for arm in ARMS}
    rc = reports["h2o"]
    tz = card_top1(items, preds_to_answers(zb))
    tc = card_top1(items, preds_to_answers(ctrl))
    res["control_vs_zbtbj"] = {
        "zbtbj_preds_sha256": zsha, "zbtbj_sha_matches_scorecard": zsha == ZBTBJ_SHA, **cmpz,
        "card_top1": {"job_4g6m8": int(sum(tc)), "zbtbj": int(sum(tz)),
                      "per_item_identical": sum(x == y for x, y in zip(tc, tz))},
        "axes": {k: {"4g6m8": rc["axes"][k], "zbtbj": rz["axes"][k], "diff": rc["axes"][k] - rz["axes"][k]}
                 for k in ("intelligence", "calibration", "speed", "cost")},
        "composite_A": {"4g6m8": rc["scores"]["A"], "zbtbj": rz["scores"]["A"],
                        "diff": rc["scores"]["A"] - rz["scores"]["A"]},
        "speed_p50_s_raw": {"4g6m8": rc["speed"]["p50_s_raw"], "zbtbj": rz["speed"]["p50_s_raw"]}}
    if not res["control_vs_zbtbj"]["zbtbj_sha_matches_scorecard"]:
        flag("zbtbj preds sha != scorecard record")

    # 4) slice + T fit -------------------------------------------------------------------------------------
    sc = hc["slice"]
    fams, recs, rec_sha = reselect_slice(Path(a.records), sc["salt"], sc["per_family"])
    my_ids = set(recs)
    slice_raw = jl(art / "slice_raw.jsonl")
    raw_ids = {r["id"] for r in slice_raw}
    pub_ids = set(by_id)
    pub_states = {norm_text(it.state) for it in items}
    st_overlap = sorted(i for i, r in recs.items() if norm_text(r["state"]) in pub_states)
    pairs_all, by_type, by_fam, skipped = [], {}, {}, 0
    for row in slice_raw:
        rec = recs.get(row["id"])
        if rec is None:
            continue
        fam = rec["aug"]["jev1"]["family8"]
        for q in row["raw"]:
            g = gold_ix(rec, q)
            if g is None:
                skipped += 1
                continue
            pairs_all.append((q["lp"], g))
            by_type.setdefault(q["type"], []).append((q["lp"], g))
            by_fam.setdefault(fam, []).append((q["lp"], g))
    lead_fit = json.loads((art / "temperature_fit.json").read_text(encoding="utf-8"))

    def cmpT(mine, lead):
        return {"mine": mine["T"], "lead": lead.get("T"), "n_mine": mine["n"], "n_lead": lead.get("n"),
                "rel_diff": None if not lead.get("T") else abs(mine["T"] - lead["T"]) / lead["T"],
                "nll_mine": mine["nll"], "nll_lead": lead.get("nll"), "nll_T1": mine["nll_T1"]}
    tfit = {"global": cmpT(fit_T(pairs_all), lead_fit["global"]),
            "by_type": {t: cmpT(fit_T(p), lead_fit["by_type"][t]) for t, p in sorted(by_type.items())},
            "by_family": {f: cmpT(fit_T(p), lead_fit["by_family"][f]) for f, p in sorted(by_fam.items())},
            "family_decisions": {f: len(p) for f, p in sorted(by_fam.items())},
            "families_ge_200": all(len(p) >= 200 for p in by_fam.values()), "skipped_no_gold": skipped}
    for grp in ("by_type", "by_family"):
        for k, v in tfit[grp].items():
            if v["rel_diff"] is None or v["rel_diff"] > 0.005:
                flag(f"T {grp}.{k}: mine {v['mine']:.4f} vs lead {v['lead']}")
    if tfit["global"]["rel_diff"] > 0.005:
        flag("T_J refit differs > 0.5%")
    res["slice"] = {"records_uncompressed_sha256": rec_sha, "records_sha_matches_params":
                    rec_sha == sc["records_uncompressed_sha256"], "my_ids_sha256": ids_sha(my_ids),
                    "ids_sha_matches_prereg": ids_sha(my_ids) == SLICE_IDS_SHA == sc["ids_sha256"],
                    "n": len(my_ids), "per_family": {f: len(v) for f, v in fams.items()},
                    "slice_raw_ids_equal_my_slice": raw_ids == my_ids, "slice_raw_failed": sum(1 for r in slice_raw if r.get("error")),
                    "public_id_overlap": len(my_ids & pub_ids), "public_state_text_overlap": len(st_overlap),
                    "public_state_overlap_ids": st_overlap[:20],
                    "slice_record_ids_look_public": sum(1 for i in my_ids if i.startswith("original-"))}
    res["temperature_fit"] = tfit
    if not res["slice"]["ids_sha_matches_prereg"] or not res["slice"]["slice_raw_ids_equal_my_slice"]:
        flag("slice ids do not reproduce")
    if st_overlap or my_ids & pub_ids:
        flag(f"slice overlaps public items: ids {len(my_ids & pub_ids)}, state text {len(st_overlap)}")

    # 5) re-derive arms from raw_public through the pinned shim --------------------------------------------
    shim_sha = sha(Path(a.shim))
    spec = importlib.util.spec_from_file_location("h2o_shim_pinned", a.shim)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    floor = rm["shim"]["noul_floor"]
    tj = lead_fit["global"]["T"]
    ttype = {t: lead_fit["by_type"][t]["T"] for t in ("choice", "noul", "score")}
    tfam = {f: v["T"] for f, v in lead_fit["by_family"].items() if v.get("T") and v.get("n", 0) >= 200}
    posts = {"h2o": lambda typ, fam: (0.8, floor), "a_T": lambda typ, fam: (tj, floor),
             "a_Ttype": lambda typ, fam: (ttype[typ], floor), "a_Tfam": lambda typ, fam: (tfam.get(fam, tj), floor),
             "b_native": lambda typ, fam: (1.0, 0.0)}
    raw = {r["id"]: r for r in jl(art / "raw_public.jsonl")}
    rederive = {"shim_sha256": shim_sha, "shim_sha_matches_pin": shim_sha == SHIM_SHA == hc["files"]["h2o_lightning_shim.py"]["sha256"],
                "raw_family_equals_item_family": all(raw[i]["family"] == by_id[i].family for i in raw),
                "a_Tfam_items_with_T_f": sum(1 for i in raw if raw[i]["family"] in tfam), "arms": {}}
    mine_answers = {}
    for arm, post in posts.items():
        ans = {}
        for iid, rw in raw.items():
            q = rw["raw"][0]
            t, fl = post(q["type"], rw["family"])
            ans[iid] = normalize_answer(by_id[iid].question,
                                        mod.format_answer(q["type"], q["names"], q["descs"], q["lp"], t, fl,
                                                          q["threshold"]))
        mine_answers[arm] = ans
        lead = preds_to_answers(jl(art / "arms" / arm / "preds.jsonl"))
        rederive["arms"][arm] = compare_answers(items, ans, lead)
    # b_shimcal vs the control (claimed algebraically identical)
    rederive["b_shimcal_vs_h2o"] = compare_answers(items, preds_to_answers(jl(art / "arms" / "b_shimcal" / "preds.jsonl")),
                                                   preds_to_answers(ctrl))
    res["rederive"] = rederive
    for arm, v in rederive["arms"].items():
        if v["same_decision"] != v["n"] or v["max_abs_dp"] > 1e-9:
            flag(f"re-derived {arm} != lead preds (decisions {v['same_decision']}/{v['n']}, max dp {v['max_abs_dp']})")

    # 6) scoring + paired + secondary ------------------------------------------------------------------------
    preds = {arm: jl(art / "arms" / arm / "preds.jsonl") for arm in ARMS}
    integ = {arm: integrity(arm, preds[arm], items, FLAGS) for arm in ARMS}
    tables = {arm: JS.build_table(items, preds_to_answers(preds[arm])) for arm in ARMS}
    axes = {arm: (reports[arm]["axes"]["speed"], reports[arm]["axes"]["cost"]) for arm in ARMS}
    pr = paired(tables, axes, "h2o", 1000, 16)
    sec = {arm: secondary_axes(tables, axes, items, "h2o", arm, 1000, 16) for arm in ARMS if arm != "h2o"}
    dec = json.loads((art / "decision.json").read_text(encoding="utf-8"))
    cmp_rows = []

    def c(where, mine, theirs):
        if mine is None and theirs is None:
            return
        d = None if mine is None or theirs is None else abs(float(mine) - float(theirs))
        cmp_rows.append({"where": where, "rescore": mine, "lead": theirs, "abs_diff": d})
        if d is None or d > TOL:
            flag(f"diff {where}: rescore {mine} vs lead {theirs}")

    per_arm = {}
    for arm in ARMS:
        rp, lr = reports[arm], dec["rows"][arm]
        top = card_top1(items, preds_to_answers(preds[arm]))
        t = tables[arm]
        nm = (t.typ == JS.TYPE_IX["noul"]) & t.valid
        band_mine = 0
        for it in items:
            if it.type == "noul":
                d = dist(it, preds_to_answers(preds[arm]).get(it.id))
                if d is not None and 0.2 < d.get("yes", 0.0) < 0.8:
                    band_mine += 1
        bytype = {}
        for typ in ("choice", "noul", "score"):
            ix = np.flatnonzero(t.typ == JS.TYPE_IX[typ])
            r = JS.aggregate(t, ix, detail=False)
            bytype[typ] = {"n": int(len(ix)), "I": r["intelligence"]["I"], "C": r["calibration"]["score"],
                           "card_top1": int(sum(top[j] for j in ix))}
        per_arm[arm] = {"I": rp["axes"]["intelligence"], "C": rp["axes"]["calibration"], "S": rp["axes"]["speed"],
                        "K": rp["axes"]["cost"], "A": rp["scores"]["A"], "capability": rp["capability"],
                        "card_top1": int(sum(top)), "jevkit_correct": int(t.correct.sum()),
                        "noul_band": {"n_valid": int(nm.sum()), "in_band_table": int((nm & t.abstain).sum()),
                                      "in_band_direct": band_mine},
                        "by_type": bytype, "pred_sha256": rp["inputs"]["pred_sha256"]}
        for k, lk in (("I", "intelligence"), ("C", "calibration"), ("S", "speed"), ("K", "cost")):
            c(f"{arm}.{k}", per_arm[arm][k], lr["axes"][lk])
        c(f"{arm}.composite_A", per_arm[arm]["A"], lr["composite"]["A"])
        c(f"{arm}.top1", per_arm[arm]["card_top1"], lr["top1"]["correct"])
        c(f"{arm}.noul_in_band", per_arm[arm]["noul_band"]["in_band_table"], lr["noul_band_0.2_0.8"]["in_band"])
        c(f"{arm}.mean_nll_choice", None if arm == "h2o" else sec[arm]["choice_nll"]["a_mean"], None if arm == "h2o" else lr["mean_nll_choice"])
        for typ in ("choice", "noul", "score"):
            c(f"{arm}.by_type.{typ}.I", bytype[typ]["I"], lr["by_type"][typ]["I"])
            c(f"{arm}.by_type.{typ}.C", bytype[typ]["C"], lr["by_type"][typ]["C"])
        if arm != "h2o":
            s = sec[arm]
            c(f"{arm}.frac_I_lt_50", s["frac_resamples_I_below_50"]["a"], lr["frac_resamples_I_lt_50"])
            for q, v in (s["composite_A_quantiles"]["a"] or {}).items():
                c(f"{arm}.A_q{q}", v, lr["composite_A_quantiles"][q])
            lp_ = dec["paired_vs_h2o"][f"{arm} - h2o"]
            c(f"{arm}.dA", pr[arm]["composite_A"]["point"], lp_["composite_A_diff"])
            c(f"{arm}.dA.ci_lo", pr[arm]["composite_A"]["ci95"][0], lp_["composite_A_diff_ci95"][0])
            c(f"{arm}.dA.ci_hi", pr[arm]["composite_A"]["ci95"][1], lp_["composite_A_diff_ci95"][1])
            c(f"{arm}.dA.jevkit_native.ci_lo", pr[arm]["jevkit_bootstrap_composite_A_diff_ci95"][0], lp_["composite_A_diff_ci95"][0])
            c(f"{arm}.dA.jevkit_native.ci_hi", pr[arm]["jevkit_bootstrap_composite_A_diff_ci95"][1], lp_["composite_A_diff_ci95"][1])
            c(f"{arm}.dI.ci_lo", pr[arm]["I"]["ci95"][0], lp_["dI_ci95"][0])
            c(f"{arm}.dI.ci_hi", pr[arm]["I"]["ci95"][1], lp_["dI_ci95"][1])
            c(f"{arm}.dC.ci_lo", pr[arm]["C"]["ci95"][0], lp_["dC_ci95"][0])
            c(f"{arm}.dC.ci_hi", pr[arm]["C"]["ci95"][1], lp_["dC_ci95"][1])
            c(f"{arm}.dS", s["per_axis"]["S"]["point"], lp_["dS"])
            c(f"{arm}.dK", s["per_axis"]["K"]["point"], lp_["dK"])
            # card top-1 paired (the lead's d_top1 uses the card rule, not jevkit's correct)
            ta, tb = card_top1(items, preds_to_answers(preds[arm])), card_top1(items, preds_to_answers(preds["h2o"]))
            bb = sum(1 for x, y in zip(tb, ta) if x and not y)
            cc = sum(1 for x, y in zip(tb, ta) if y and not x)
            per_arm[arm]["paired_card_top1"] = {"delta": cc - bb, "h2o_only": bb, "arm_only": cc,
                                                "mcnemar_exact_p": _mcnemar_exact(bb, cc) if bb + cc else None}
            c(f"{arm}.d_top1", cc - bb, lp_["d_top1"]["count"])
            c(f"{arm}.d_mean_nll_choice", s["choice_nll"]["delta"], lp_["d_mean_nll_choice"])
    res["per_arm"] = per_arm
    res["paired_vs_h2o"] = pr
    res["secondary_axes"] = sec
    res["integrity"] = integ
    res["diff_vs_decision"] = {"n": len(cmp_rows), "max_abs_diff": max((r["abs_diff"] or 0) for r in cmp_rows),
                               "over_tol": [r for r in cmp_rows if r["abs_diff"] is None or r["abs_diff"] > TOL],
                               "rows": cmp_rows}

    # 7) G1 ---------------------------------------------------------------------------------------------------
    ops = ("option_reverse", "option_shuffle", "key_relabel")
    ps = JP.make_perturbations(items, ops, (0,))
    filed = jl(art / "g1" / "h2o" / "perturbed.jsonl")
    filed_c = jl(art / "g1" / "c_symfork" / "perturbed.jsonl")
    fid = {r["id"]: r for r in filed}
    g1: dict = {"n_inmemory": len(ps), "n_file": len(filed), "file_same_both_arms": sha(art / "g1" / "h2o" / "perturbed.jsonl")
                == sha(art / "g1" / "c_symfork" / "perturbed.jsonl"), "per_op": {}}
    mism = 0
    for p in ps:
        f = fid.get(p.item.id)
        if f is None or f["perturbation"]["label_map"] != p.label_map or f["perturbation"]["op"] != p.op \
                or json.dumps(f["question"], sort_keys=True) != json.dumps(p.item.question, sort_keys=True):
            mism += 1
        b = by_id[p.base_id]
        crit_b, crit_p = list((b.question.get("criteria") or {}).keys()), list((p.item.question.get("criteria") or {}).keys())
        mapped = [p.label_map.get(k, k) for k in crit_b]
        r = g1["per_op"].setdefault(p.op, {"n": 0, "order_changed_in_memory": 0, "labels_changed": 0,
                                           "file_order_equals_memory": 0, "by_type": {}})
        r["n"] += 1
        # order changed = the perturbed criteria order is not the base order carried through the label map
        ch = crit_p != mapped
        r["order_changed_in_memory"] += int(ch)
        r["labels_changed"] += int(any(k != v for k, v in p.label_map.items()))
        r["file_order_equals_memory"] += int(f is not None and list((f["question"].get("criteria") or {}).keys()) == crit_p)
        bt = r["by_type"].setdefault(b.type, {"n": 0, "order_changed": 0})
        bt["n"] += 1
        bt["order_changed"] += int(ch)
    g1["file_vs_memory_mismatch"] = mism
    for name in ("h2o", "c_symfork"):
        base = preds_to_answers(jl(art / "g1" / name / "base.preds.jsonl"))
        pert = preds_to_answers(jl(art / "g1" / name / "perturbed.preds.jsonl"))
        rep = JP.flip_report(items, ps, base, pert)
        lead = json.loads((art / "g1" / name / "perturb_report.json").read_text(encoding="utf-8"))
        moved = {}
        for p in ps:
            b = by_id[p.base_id]
            db, dp_ = dist(b, base.get(b.id)), dist(p.item, pert.get(p.item.id))
            back = {lab: dp_[p.label_map[lab]] for lab in b.labels} if dp_ is not None else None
            m = moved.setdefault(p.op, {"n": 0, "any_dp_gt_0": 0, "dp_gt_1e-3": 0})
            m["n"] += 1
            if db is not None and back is not None:
                x = max(abs(db[k] - back[k]) for k in b.labels)
                m["any_dp_gt_0"] += int(x > 0)
                m["dp_gt_1e-3"] += int(x > 1e-3)
        per = {op: {k: v[k] for k in ("n", "invalid", "flips", "max_abs_dp")} for op, v in rep["per_operator"].items()}
        same = all(per[op][k] == lead["flip_report"]["per_operator"][op][k] for op in per for k in ("n", "invalid", "flips"))
        dpd = max(abs((per[op]["max_abs_dp"] or 0) - (lead["flip_report"]["per_operator"][op]["max_abs_dp"] or 0))
                  for op in per)
        arm_preds = preds_to_answers(preds[name])
        g1[name] = {"per_operator": per, "flips_total": sum(v["flips"] for v in per.values()),
                    "g1_pass_recomputed": all(v["flips"] == 0 and v["invalid"] == 0 and (v["max_abs_dp"] or 0) <= 1e-3
                                              for v in per.values()),
                    "lead_g1_pass": lead["g1_pass"], "lead_rows_all_identical": lead.get("rows_all_identical"),
                    "lead_row_identity": lead.get("row_identity"),
                    "counts_equal_lead": same, "max_dp_abs_diff_vs_lead": dpd, "answers_moved": moved,
                    "g1_base_vs_arm_preds": compare_answers(items, base, arm_preds)}
        if not same or dpd > 1e-9:
            flag(f"G1 {name}: recomputed flip report != lead's")
    res["g1"] = g1
    res["flags"] = FLAGS
    (out / "rescore.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"flags": FLAGS, "max_abs_diff": res["diff_vs_decision"]["max_abs_diff"]}, indent=1))
    return 3 if FLAGS else 0


if __name__ == "__main__":
    sys.exit(main())

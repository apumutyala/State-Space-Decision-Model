"""B-PREREG-GUAR-JEV3 driver (origin/state PREREG.md, registered at state 745fc478 before any slice probability was read).

Split-conformal sets (jevkit LAC score) + Learn-then-Test abstention on the per-decision slice probabilities
(calib_probs.jsonl) of our trained stack. Pure Python, CPU. Uses only jevkit/guarantees.py functions (default paths
plus the opt-in fixed_sequence / fs_start / crr modes); nothing in jevkit is changed.

    python -I -X utf8 guarantees_jev3.py --repo <repo root> --out <out.json> \
        --source NAME=CALIB_PROBS::CELL_JSON::IDS_SHA::SLICE_FLOOR_JSON [...] [--shim <vendored h2o_lightning_shim.py>]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

SALT = "B-PREREG-GUAR-JEV3|v1|"
ALPHAS = (0.10, 0.05)
EPS, DELTA = 0.05, 0.10
MIN_N = 100
FS_START = 0.95
SHIM_T, SHIM_FLOOR = 0.8, 0.801  # third_party/h2o_lightning_shim/serve_config.json (temperature, noul_floor)


def third(decision_id: str) -> str:
    u = int(hashlib.sha256((SALT + decision_id).encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "FIT" if u < 1 / 3 else ("LTT" if u < 2 / 3 else "TEST")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "decision ids not unique"
    rids = [i.rsplit("/", 1)[0] for i in ids]
    assert len(set(rids)) == len(rids), "more than one decision per row"
    for r in rows:
        assert len(r["probs"]) == len(r["option_ids"]) and 0 <= r["gold"] < len(r["option_ids"])
        assert abs(sum(r["probs"]) - 1.0) < 1e-6
        if r["type"] == "score":
            assert r["option_ids"] == [str(i) for i in range(len(r["option_ids"]))]
        if r["type"] == "noul":
            assert r["option_ids"] == ["true", "false"]
    return rows


def record_ids_sha(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(r["id"].rsplit("/", 1)[0] for r in rows)).encode()).hexdigest()


def _softmax_t(p: list[float], t: float) -> list[float]:
    v = [math.log(max(float(x), 1e-12)) / max(t, 1e-3) for x in p]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def shimcal(r: dict) -> list[float]:
    """screen.py shim_calibrate on one slice decision: noul -> format_answer('noul', [lg py, lg 1-py]) with the
    commit floor; choice / score -> softmax(log p / 0.8)."""
    if r["type"] == "noul":
        py = r["probs"][r["option_ids"].index("true")]
        q = _softmax_t([py, 1 - py], SHIM_T)[0]
        if max(q, 1 - q) < SHIM_FLOOR:
            q = SHIM_FLOOR if q >= 0.5 else 1 - SHIM_FLOOR
        return [q, 1 - q] if r["option_ids"][0] == "true" else [1 - q, q]
    return _softmax_t(r["probs"], SHIM_T)


def shim_parity(rows: list[dict], shim_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("h2o_shim_vendored", shim_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lg = lambda p: math.log(max(float(p), 1e-12))  # noqa: E731  (screen.py)
    worst = 0.0
    for r in rows:
        mine = shimcal(r)
        if r["type"] == "noul":
            py = r["probs"][r["option_ids"].index("true")]
            a = mod.format_answer("noul", ["true", "false"], ["", ""], [lg(py), lg(1 - py)], SHIM_T, SHIM_FLOOR)
            ref = [a["noul"], 1 - a["noul"]]
        else:
            names = r["option_ids"]
            a = mod.format_answer(r["type"], names, names, [lg(x) for x in r["probs"]], SHIM_T, SHIM_FLOOR)
            ref = [a["probabilities"][n] for n in (names if r["type"] == "choice" else [str(i) for i in range(len(names))])]
        worst = max(worst, max(abs(x - y) for x, y in zip(mine, ref)))
    return {"shim_file_sha256": hashlib.sha256(shim_path.read_bytes()).hexdigest(), "n": len(rows),
            "max_abs_diff_vs_vendored_format_answer": worst}


def run_variant(G, scored, meta, fit_ids, ltt_ids, test_ids, alpha: float, variant: str) -> dict:
    """variant M (marginal), T (Mondrian per type), F (Mondrian per family); groups with n_FIT < MIN_N ->
    pooled-small; pooled-small with n_FIT < MIN_N -> full label set (q_hat = inf)."""
    def raw_key(i):
        return "all" if variant == "M" else (meta[i]["type"] if variant == "T" else meta[i]["family"])
    nfit: dict[str, int] = {}
    for i in fit_ids:
        nfit[raw_key(i)] = nfit.get(raw_key(i), 0) + 1
    small = {k for k, n in nfit.items() if n < MIN_N}
    # every key that exists anywhere (a key absent from FIT is small too)
    for i in list(ltt_ids) + list(test_ids):
        if raw_key(i) not in nfit:
            small.add(raw_key(i))

    def gkey(i):
        k = raw_key(i)
        return "pooled-small" if k in small else k
    S = {i: G.Scored(i, gkey(i), scored[i].probs, scored[i].label) for i in scored}
    fit_half = [S[i] for i in fit_ids]
    ltt_half = [S[i] for i in ltt_ids]
    test = [S[i] for i in test_ids]
    fit = G.fit_conformal(fit_half, alpha, mondrian=True)
    n_small = fit["n"].get("pooled-small", 0)
    full_set_small = bool(small) and n_small < MIN_N
    if full_set_small:
        fit["qhat"]["pooled-small"] = math.inf
    out = {"variant": variant, "alpha": alpha, "groups": {g: {"n_fit": fit["n"].get(g, 0), "qhat": fit["qhat"].get(g)}
                                                            for g in sorted(set(fit["n"]) | {"pooled-small"} & set(fit["qhat"]))},
           "pooled_small_members": sorted(small), "pooled_small_full_set": full_set_small,
           "fit_ids_sha256": fit["ids_sha256"]}

    def set_of(r):
        return G.prediction_set(r.probs, G._qhat_for(fit, r))

    def set_stats(rs):
        if not rs:
            return {"n": 0}
        sets = [set_of(r) for r in rs]
        n = len(rs)
        cov = sum(r.label in s for r, s in zip(rs, sets)) / n
        sing = [(r, s) for r, s in zip(rs, sets) if len(s) == 1]
        return {"n": n, "coverage": cov, "flag_threshold": 1 - alpha - 2 * math.sqrt(alpha * (1 - alpha) / n),
                "flagged": cov < 1 - alpha - 2 * math.sqrt(alpha * (1 - alpha) / n),
                "mean_set_size": sum(len(s) for s in sets) / n, "singleton_rate": len(sing) / n,
                "empty_rate": sum(len(s) == 0 for s in sets) / n,
                "singleton_accuracy": (sum(s[0] == r.label for r, s in sing) / len(sing)) if sing else None,
                "top1_accuracy": sum(max(r.probs, key=r.probs.get) == r.label for r in rs) / n}

    def sel_stats(rs, lam_of):
        n = len(rs)
        acted = []
        for r in rs:
            lam = lam_of(r)
            if lam is None:
                continue
            ok, y = G._acts(r, fit, lam)
            if ok:
                acted.append((r, y))
        return {"n": n, "n_act": len(acted), "act_fraction": len(acted) / n if n else None,
                "acted_errors": sum(y != r.label for r, y in acted),
                "acted_error": (sum(y != r.label for r, y in acted) / len(acted)) if acted else None}

    def breakdown(fn):
        res = {"all": fn(test)}
        for key in ("type", "family"):
            by: dict[str, list] = {}
            for r in test:
                by.setdefault(meta[r.id][key], []).append(r)
            res["per_" + key] = {k: fn(v) for k, v in sorted(by.items())}
        return res

    out["sets_test"] = breakdown(set_stats)
    # PRIMARY LTT: per-decision exact binomial, Bonferroni delta/101, one global lambda
    ltt = G.ltt_select(ltt_half, fit, EPS, DELTA)
    out["ltt_primary"] = ltt
    lam = ltt["lambda"]
    out["selective_test_primary"] = breakdown(lambda rs: sel_stats(rs, lambda r: lam))
    # LTT-half curve (attribution): n_act, errors, binomial p per lambda
    curve = []
    for L in G.DEFAULT_LAMBDAS:
        acted = [(r, y) for r in ltt_half for ok, y in [G._acts(r, fit, L)] if ok]
        n, e = len(acted), sum(y != r.label for r, y in acted)
        curve.append({"lambda": L, "n_act": n, "errors": e, "p": G.binom_cdf(e, n, EPS) if n else None})
    live = [c for c in curve if c["n_act"]]
    best = min(live, key=lambda c: c["p"]) if live else None
    need0 = math.ceil(math.log(len(G.DEFAULT_LAMBDAS) / DELTA) / -math.log(1 - EPS))
    per_type_at = {}
    cal_all = ltt_half + fit_half  # answer-level diagnostics only (no certificate uses them)
    for t in ("choice", "noul", "score"):
        rs = [r for r in ltt_half if meta[r.id]["type"] == t]
        acted = [(r, y) for r in rs for ok, y in [G._acts(r, fit, best["lambda"])] if ok] if best else []
        allt = [r for r in cal_all if meta[r.id]["type"] == t]
        pm = [max(r.probs.values()) for r in allt]
        per_type_at[t] = {"ltt_n": len(rs), "n_act_at_best": len(acted),
                          "errors_at_best": sum(y != r.label for r, y in acted),
                          "fit_ltt_frac_pmax_ge_0.95": (sum(p >= 0.95 for p in pm) / len(pm)) if pm else None,
                          "fit_ltt_acc_when_pmax_ge_0.95": ((sum(max(r.probs, key=r.probs.get) == r.label
                                                                 for r in allt if max(r.probs.values()) >= 0.95)
                                                             / max(1, sum(p >= 0.95 for p in pm))) if pm else None),
                          "fit_ltt_band_0.2_0.8": ((sum(0.2 <= r.probs.get("true", 0) <= 0.8 for r in allt) / len(allt))
                                                   if t == "noul" and allt else None)}
    out["ltt_half_attribution"] = {"best_lambda_by_p": best, "per_type_at_best": per_type_at,
                                   "bonferroni_level": DELTA / len(G.DEFAULT_LAMBDAS),
                                   "zero_error_acts_needed_bonferroni": need0,
                                   "zero_error_acts_needed_single_test": math.ceil(math.log(1 / DELTA) / -math.log(1 - EPS)),
                                   "ltt_singleton_rate": sum(len(set_of(r)) == 1 for r in ltt_half) / len(ltt_half),
                                   "curve": curve}
    # SECONDARY (a) FS-CRR from 0.95, cluster = row, c = 1 and c = 8; (b) binomial fixed sequence from 0.95
    sec = {}
    for c in (1, 8):
        r_ = G.ltt_select(ltt_half, fit, EPS, DELTA, method="fixed_sequence", fs_start=FS_START, cluster_test="crr",
                          crr_cap=c, cluster_of=lambda r: r.id)
        sec[f"fs_crr_c{c}"] = {"ltt": r_, "test": sel_stats(test, lambda r, L=r_["lambda"]: L)}
    r_ = G.ltt_select(ltt_half, fit, EPS, DELTA, method="fixed_sequence", fs_start=FS_START)
    sec["fs_binomial"] = {"ltt": r_, "test": sel_stats(test, lambda r, L=r_["lambda"]: L)}
    # (c) per-type LTT: own lambda per type with n_FIT >= MIN_N and n_LTT >= MIN_N, Bonferroni delta/(K*101)
    tf: dict[str, int] = {}
    tl: dict[str, int] = {}
    for i in fit_ids:
        tf[meta[i]["type"]] = tf.get(meta[i]["type"], 0) + 1
    for i in ltt_ids:
        tl[meta[i]["type"]] = tl.get(meta[i]["type"], 0) + 1
    live_t = sorted(t for t in tf if tf[t] >= MIN_N and tl.get(t, 0) >= MIN_N)
    lam_t = {}
    per = {}
    for t in live_t:
        rr = G.ltt_select([r for r in ltt_half if meta[r.id]["type"] == t], fit, EPS, DELTA / len(live_t))
        lam_t[t] = rr["lambda"]
        per[t] = rr
    sec["per_type_ltt"] = {"live_types": live_t, "K": len(live_t), "ltt": per,
                           "test": breakdown(lambda rs: sel_stats(rs, lambda r: lam_t.get(meta[r.id]["type"])))}
    out["secondary"] = sec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=CALIB_PROBS::CELL_JSON::EXPECTED_IDS_SHA::SLICE_FLOOR_JSON")
    ap.add_argument("--shim", help="vendored third_party/h2o_lightning_shim/h2o_lightning_shim.py (parity check)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import jevkit.guarantees as G

    report = {"prereg": "B-PREREG-GUAR-JEV3 (origin/state 745fc478)", "salt": SALT, "alphas": ALPHAS,
              "epsilon": EPS, "delta": DELTA, "min_n": MIN_N, "fs_start": FS_START,
              "guarantees_py_sha256": hashlib.sha256((Path(a.repo) / "jevkit" / "guarantees.py").read_bytes()).hexdigest(),
              "sources": {}}
    for spec in a.source:
        name, rest = spec.split("=", 1)
        cp, cell, ids_sha, floor = rest.split("::")
        rows = load_rows(Path(cp))
        got_sha = record_ids_sha(rows)
        assert got_sha == ids_sha, f"{name}: ids sha {got_sha} != {ids_sha}"
        fl = json.loads(Path(floor).read_text(encoding="utf-8"))
        top1 = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rows) / len(rows)
        assert abs(top1 - fl["native_top1"]) < 1e-12, f"{name}: top-1 {top1} != slice_floor {fl['native_top1']}"
        dev = set((json.loads(Path(cell).read_text(encoding="utf-8")).get("dev_card_sample") or {}).get("ids") or [])
        ids = [r["id"] for r in rows]
        assert dev <= set(ids), f"{name}: dev-card ids not all in the slice"
        meta = {r["id"]: {"type": r["type"], "family": r["family"]} for r in rows}
        part = {i: third(i) for i in ids}
        src = {"calib_probs_sha256": hashlib.sha256(Path(cp).read_bytes()).hexdigest(),
               "cell_json_sha256": hashlib.sha256(Path(cell).read_bytes()).hexdigest(), "ids_sha256": got_sha,
               "n": len(rows), "native_top1": top1, "slice_floor_native_top1": fl["native_top1"],
               "dev_card_n": len(dev), "dev_card_ids_sha256": hashlib.sha256("\n".join(sorted(dev)).encode()).hexdigest(),
               "arms": {}}
        if a.shim:
            src["shim_parity"] = shim_parity(rows, Path(a.shim))
        for subset in ("all", "remainder"):
            keep = [i for i in ids if subset == "all" or i not in dev]
            fit_ids = [i for i in keep if part[i] == "FIT"]
            ltt_ids = [i for i in keep if part[i] == "LTT"]
            test_ids = [i for i in keep if part[i] == "TEST"]
            sp = {"n": len(keep), "n_fit": len(fit_ids), "n_ltt": len(ltt_ids), "n_test": len(test_ids),
                  "fit_ids_sha256": G.ids_sha256(fit_ids), "ltt_ids_sha256": G.ids_sha256(ltt_ids),
                  "test_ids_sha256": G.ids_sha256(test_ids),
                  "type_counts": {p: {t: sum(meta[i]["type"] == t for i in L) for t in ("choice", "noul", "score")}
                                  for p, L in (("FIT", fit_ids), ("LTT", ltt_ids), ("TEST", test_ids))}}
            for arm in ("native", "native_shimcal"):
                scored = {}
                for r in rows:
                    p = r["probs"] if arm == "native" else shimcal(r)
                    scored[r["id"]] = G.Scored(r["id"], r["type"], dict(zip(r["option_ids"], p)),
                                               r["option_ids"][r["gold"]])
                res = {"split": sp, "results": []}
                for alpha in ALPHAS:
                    for v in ("M", "T", "F"):
                        res["results"].append(run_variant(G, scored, meta, fit_ids, ltt_ids, test_ids, alpha, v))
                src["arms"].setdefault(arm, {})[subset] = res
        report["sources"][name] = src
    Path(a.out).write_text(json.dumps(report, indent=1, default=lambda o: None if o is None else str(o),
                                      allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

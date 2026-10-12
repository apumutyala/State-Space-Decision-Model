"""B-PREREG-GUAR-JEV3/a driver (origin/state PREREG.md @64f0c3dc): the UNCHANGED B-PREREG-GUAR-JEV3 pipeline
(receipts/2026-10-12-ev-guar-jev3/guarantees_jev3.py, imported, not copied: same salt / split / alphas / variants /
LTT primary / secondaries / remainder sensitivity) on job-zbmhc (JEV3-LM-S, scored ckpt step 150), with three arms:

  native_tfit     calib_probs re-tempered p^(1/T) by the frozen zbmhc per-type T map (screen.py t_for /
                  retemper_answer: (type, K) cell T if fitted, else the type T)  = PRIMARY readout for the verdict
  native_shimcal  the vendored shim post-processing (T 0.8, yes/no floor 0.801), unchanged
  native          raw untempered calib_probs

    python -I -X utf8 guarantees_jev3_zbmhc.py --repo <repo root> --out <out.json> --calib <calib_probs.jsonl> \
        --cell <cell.json> --floor <slice_floor.json> --tfit <tfit.json> --shim <vendored h2o_lightning_shim.py>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "2026-10-12-ev-guar-jev3" / "guarantees_jev3.py"
IDS_SHA = "844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078"
ARMS = ("native_tfit", "native_shimcal", "native")


def load_base():
    spec = importlib.util.spec_from_file_location("guarantees_jev3_base", BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t_for(r: dict, tmap: dict) -> float:
    """screen.py t_for on a calib_probs row: k = 2 for noul, else the option count."""
    k = 2 if r["type"] == "noul" else len(r["probs"])
    c = (tmap.get("cells") or {}).get(f"{r['type']}:{k}")
    return float(c["T"]) if c else float(((tmap.get("types") or {}).get(r["type"]) or {}).get("T", 1.0))


def tfit(r: dict, tmap: dict) -> list[float]:
    """screen.py retemper_answer: p_i^(1/T) renormalised (noul: the two-option [P(yes), 1 - P(yes)])."""
    t = t_for(r, tmap)
    w = [max(float(p), 1e-12) ** (1.0 / t) for p in r["probs"]]
    s = sum(w)
    return [x / s for x in w]


def arm_probs(B, r: dict, arm: str, tmap: dict) -> list[float]:
    if arm == "native":
        return list(r["probs"])
    if arm == "native_shimcal":
        return B.shimcal(r)
    return tfit(r, tmap)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("repo", "out", "calib", "cell", "floor", "tfit", "shim"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import jevkit.guarantees as G
    B = load_base()

    rows = B.load_rows(Path(a.calib))
    got = B.record_ids_sha(rows)
    assert got == IDS_SHA, f"ids sha {got} != {IDS_SHA}"
    fl = json.loads(Path(a.floor).read_text(encoding="utf-8"))
    top1 = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rows) / len(rows)
    assert abs(top1 - fl["native_top1"]) < 1e-12, f"top-1 {top1} != slice_floor {fl['native_top1']}"
    cell = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    assert cell["best"]["step"] == 150 and cell["best_export"]["step"] == 150, "scored ckpt is not step 150"
    tj = json.loads(Path(a.tfit).read_text(encoding="utf-8"))
    tmap = tj["per_type"]
    assert tj["ids_sha256"] == IDS_SHA
    dev = set(cell["dev_card_sample"]["ids"])
    ids = [r["id"] for r in rows]
    assert dev <= set(ids), "dev-card ids not all in the slice"
    meta = {r["id"]: {"type": r["type"], "family": r["family"]} for r in rows}
    part = {i: B.third(i) for i in ids}
    # tfit arm sanity: argmax preserved, every row's T recorded
    t_used: dict[str, int] = {}
    for r in rows:
        key = f"{r['type']}:{len(r['probs'])}->T={t_for(r, tmap):.6f}"
        t_used[key] = t_used.get(key, 0) + 1
        p = tfit(r, tmap)
        assert max(range(len(p)), key=p.__getitem__) == max(range(len(r["probs"])), key=r["probs"].__getitem__)
    report = {"prereg": "B-PREREG-GUAR-JEV3/a (origin/state 64f0c3dc) over B-PREREG-GUAR-JEV3 (745fc478)",
              "base_driver_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
              "salt": B.SALT, "alphas": B.ALPHAS, "epsilon": B.EPS, "delta": B.DELTA, "min_n": B.MIN_N,
              "fs_start": B.FS_START,
              "guarantees_py_sha256": hashlib.sha256((Path(a.repo) / "jevkit" / "guarantees.py").read_bytes()).hexdigest(),
              "sources": {}}
    src = {"calib_probs_sha256": hashlib.sha256(Path(a.calib).read_bytes()).hexdigest(),
           "cell_json_sha256": hashlib.sha256(Path(a.cell).read_bytes()).hexdigest(),
           "tfit_json_sha256": hashlib.sha256(Path(a.tfit).read_bytes()).hexdigest(), "ids_sha256": got,
           "n": len(rows), "native_top1": top1, "slice_floor_native_top1": fl["native_top1"],
           "scored_step": cell["best"]["step"], "best_export_sha256": cell["best_export"]["sha256"],
           "dev_card_n": len(dev), "dev_card_ids_sha256": hashlib.sha256("\n".join(sorted(dev)).encode()).hexdigest(),
           "tfit_T_by_cell": t_used, "shim_parity": B.shim_parity(rows, Path(a.shim)), "arms": {}}
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
        for arm in ARMS:
            scored = {r["id"]: G.Scored(r["id"], r["type"], dict(zip(r["option_ids"], arm_probs(B, r, arm, tmap))),
                                        r["option_ids"][r["gold"]]) for r in rows}
            res = {"split": sp, "results": []}
            for alpha in B.ALPHAS:
                for v in ("M", "T", "F"):
                    res["results"].append(B.run_variant(G, scored, meta, fit_ids, ltt_ids, test_ids, alpha, v))
            src["arms"].setdefault(arm, {})[subset] = res
    report["sources"]["jev3lms_zbmhc"] = src
    Path(a.out).write_text(json.dumps(report, indent=1, default=lambda o: None if o is None else str(o),
                                      allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

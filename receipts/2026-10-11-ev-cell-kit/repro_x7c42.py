"""Reproduction check: scripts/ev_cell_rescore.py on the job-x7c42 landing files vs the committed independent rescore
(claude/ev-x7c42-rescore @bf706854, receipts/2026-10-11-ev-x7c42-rescore/{rescore.json, rescore-extra.json}).

    python -X utf8 receipts/2026-10-11-ev-cell-kit/repro_x7c42.py --kit <out>/cell_rescore.json \
        --x7c42 <ev-x7c42 checkout>/receipts/2026-10-11-ev-x7c42-rescore [--out repro_x7c42.json]

Every compared number must be EXACTLY equal (same draws, same code path); prints one line per block and exits 1 on any
difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def walk(a, b, where, out):
    """Exact comparison of two JSON trees on the keys of ``b`` (the committed receipt)."""
    if isinstance(b, dict):
        if not isinstance(a, dict):
            out.append((where, a, b))
            return
        for k in b:
            walk(a.get(k), b[k], f"{where}.{k}", out)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            out.append((where, a, b))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{where}[{i}]", out)
    elif a != b:
        out.append((where, a, b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--x7c42", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    k = json.loads(Path(a.kit).read_text(encoding="utf-8"))
    d = Path(a.x7c42)
    core = json.loads((d / "rescore.json").read_text(encoding="utf-8"))
    ext = json.loads((d / "rescore-extra.json").read_text(encoding="utf-8"))
    ep = ext["primary"]["native_shimcal"]
    kp = k["1_primary"]
    blocks = {}

    def chk(name, mine, theirs):
        diffs: list = []
        walk(mine, theirs, name, diffs)
        n = sum(1 for _ in _leaves(theirs))
        blocks[name] = {"n_compared": n, "n_diff": len(diffs), "diffs": diffs[:10]}

    chk("arms (I,C,S,K,cap,A)", {n: {x: v[x] for x in ("I", "C", "S", "K", "capability", "A")} for n, v in k["arms"].items()},
        {n: {x: v[x] for x in ("I", "C", "S", "K", "capability")} | {"A": v["composite"]["A"]} for n, v in core["arms"].items()})
    chk("primary dCompA (point, ci95, ci97.5, ci99, p, p+1, quantiles, dI)",
        {"point": kp["point"], "ci95": kp["ci95"], "ci975_holm": kp["ci975"], "ci99": kp["ci99"],
         "p_one_sided_boot": kp["p_one_sided_boot"], "p_one_sided_boot_plus1": kp["p_one_sided_boot_plus1"],
         "quantiles": kp["quantiles"], "dI_point": kp["dI_point"], "dI_ci95": kp["dI_ci95"]},
        {x: ep[x] for x in ("point", "ci95", "ci975_holm", "ci99", "p_one_sided_boot", "p_one_sided_boot_plus1",
                            "quantiles", "dI_point", "dI_ci95")})
    chk("primary vs core.primary.delta_composite_A", {"point": kp["point"], "ci95": kp["ci95"]},
        {"point": core["primary"]["delta_composite_A"]["point"], "ci95": core["primary"]["delta_composite_A"]["ci95"]})
    chk("per-type cc", {t: {"a": v["cc"]["a"], "b": v["cc"]["b"], "delta": v["cc"]["delta"], "ci95": v["cc"]["ci95"]}
                        for t, v in k["2_secondary"]["per_type"].items()}, ep["per_type_cc"])
    chk("per-type top-1", {t: v["top1"] for t, v in k["2_secondary"]["per_type"].items()}, ep["per_type_top1"])
    chk("EV-PREREG-SECONDARY-AXES block", k["2_secondary"]["ev_prereg_secondary_axes"]["native_shimcal - stock_shim"],
        core["secondary_axes"]["native_shimcal - stock_shim"])
    chk("noul band", {n: v["in_band_0.2_0.8"] for n, v in k["2_secondary"]["noul_band"].items()},
        {n: v["band_0.2_0.8"] for n, v in ext["noul_band"].items()})
    chk("top-1 public", k["2_secondary"]["top1_public"], ext["top1_public"])
    t = k["3_target"]["h2o_4g6m8"]["stock_shim_vs_h2o_4g6m8_identity"]
    ci = ext["control_identity"]["job-4g6m8"]
    chk("control vs job-4g6m8 H2O (bitwise answers, top-1)", {"answers_bitwise_equal": t["answers_bitwise_equal"],
                                                             "top1_equal_items": t["top1_equal_items"],
                                                             "top1": t["top1"]["h2o_4g6m8"],
                                                             "preds_sha256": k["3_target"]["h2o_4g6m8"]["preds_sha256"]},
        {x: ci[x] for x in ("answers_bitwise_equal", "top1_equal_items", "top1", "preds_sha256")})
    fl = k["5_selection"]["slice_floor"]
    ef = ext["slice_floor"]
    chk("step-0 floor", {x: fl[x] for x in ("n_decisions", "n_unique_ids", "selected_top1", "step0_top1", "delta_pp",
                                            "ci95_pp_iid_rows", "ci95_pp_family_stratified", "p_one_sided_boot",
                                            "selected_only", "step0_only", "both", "neither", "mcnemar_exact_p")}
        | {"per_family": {f: {x: v[x] for x in ("n", "step0", "selected", "delta_pp", "ci95_pp", "selected_only",
                                                 "step0_only", "mcnemar_p")} for f, v in fl["per_family"].items()},
           "per_type": fl["per_type"]},
        {x: ef[x] for x in ("n_decisions", "n_unique_ids", "selected_top1", "step0_top1", "delta_pp", "ci95_pp_iid_rows",
                            "ci95_pp_family_stratified", "p_one_sided_boot", "selected_only", "step0_only", "both",
                            "neither", "mcnemar_exact_p")}
        | {"per_family": {f: {x: v[x] for x in ("n", "step0", "selected", "delta_pp", "ci95_pp", "selected_only",
                                                 "step0_only", "mcnemar_p")} for f, v in ef["per_family"].items()},
           "per_type": ef["per_type"]})
    chk("slice ids sha", fl["record_ids_sha256"], ext["overlap"]["calib_ids_sha256"]["sorted_newline"])
    pub = {f: r["public"] for f, r in k["8_transfer"]["by_family"].items() if r["public"]}
    chk("public by family (transfer)", {f: {"n": v["n"], "control_top1": v["control_top1"], "arm_top1": v["arm_top1"],
                                            "delta_top1_pp": v["delta_top1_pp"], "delta_top1_ci95_pp": v["delta_top1_ci95_pp"],
                                            "a_only": v["a_only"], "b_only": v["b_only"], "mcnemar_p": v["mcnemar_p"],
                                            "delta_I": v["delta_I"], "delta_I_ci95": v["delta_I_ci95"]} for f, v in pub.items()},
        {f: {x: v[x] for x in ("n", "control_top1", "arm_top1", "delta_top1_pp", "delta_top1_ci95_pp", "a_only",
                               "b_only", "mcnemar_p", "delta_I", "delta_I_ci95")} for f, v in ext["public_by_family"].items()})
    o = k["5_selection"]["order"]
    chk("order (selection vs first public pred)", {"selection_cell_json_utc": o["selection_cell_json_utc"],
                                                   "first_public_pred_utc": o["first_public_pred_utc"]},
        {x: ext["order"][x] for x in ("selection_cell_json_utc", "first_public_pred_utc")})
    chk("selection (best, trained steps)", {"best": k["5_selection"]["cell"]["best"],
                                            "train_steps": k["5_selection"]["cell"]["train_steps"]},
        {"best": core["cell"]["best"], "train_steps": core["cell"]["train_steps"]})
    chk("screen.json diff (n fields, max |d|)", {"n": len(k["screen_diffs"]), "max": k["1_primary"]["screen_diff"]["max_abs_diff"]},
        {"n": len(core["diffs_vs_screen"]), "max": core["max_abs_diff"]})
    bad = 0
    for name, b in blocks.items():
        print(f"{'OK  ' if not b['n_diff'] else 'DIFF'} {b['n_compared']:5d} values  {name}")
        for w, x, y in b["diffs"]:
            print(f"       {w}: kit {x!r} vs receipt {y!r}")
        bad += b["n_diff"]
    tot = sum(b["n_compared"] for b in blocks.values())
    print(f"{'REPRODUCED' if not bad else 'NOT REPRODUCED'}: {tot - bad}/{tot} values exactly equal")
    if a.out:
        Path(a.out).write_text(json.dumps({"blocks": blocks, "n_values": tot, "n_diff": bad}, indent=1, default=str),
                               encoding="utf-8")
    return 0 if not bad else 1


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


if __name__ == "__main__":
    sys.exit(main())

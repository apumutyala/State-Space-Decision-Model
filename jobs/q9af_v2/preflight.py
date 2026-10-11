"""q9af_v2 phase projections from MEASURED rates (job-5cmat / job-2xe5w / job-4j3vu) and the margin rule.

    python -m jobs.q9af_v2.preflight [--params jobs/q9af_v2/job_params.json] [--config configs/q9af_v2_h100.json] [--out f]

P = capped training (stage 1 + stage 2 wall-clock caps from the config, plus one optimiser step of overshoot each)
  + uncapped work (load, stock probes, init_cmp, held-out curve evals, final held-out + wikitext gate, converted
    probes, full MMLU both arms, throughput both arms). Only the uncapped part carries the (1 + margin) factor: a
    wall-clock-capped stage ends at its cap by construction (a slower rate trains fewer tokens, never longer).
I arms = the proxy600 rows' exact fork-path volume (jobs/common/readout_volume_qwen3.5.json per_row, summed over the
  644 proxy ids) at the measured min prefill rate and the larger per-call cost.
Rules (exit 5 if any fails; the runner calls this at setup): P cap >= capped + (1 + m) x uncapped; every other cap >=
(1 + m) x projection (Index arms against 95% of the cap, as rate_guard stops there); stage floors <= (1 - m) x the
slowest predicted rate; peak <= (1 - m) x the stop; the train stream covers the predicted token use (no re-read);
caps + reserve == max_duration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def proxy_volume(root: Path = ROOT) -> dict:
    vol = json.loads((root / "jobs/common/readout_volume_qwen3.5.json").read_text(encoding="utf-8"))
    ids = json.loads((root / "jobs/common/dev_proxy600.json").read_text(encoding="utf-8"))["run_ids"]
    miss = [i for i in ids if i not in vol["per_row"]]
    if miss:
        raise ValueError(f"{len(miss)} proxy600 ids missing from the readout volume")
    tot = {p: {k: sum(vol["per_row"][i][p][k] for i in ids) for k in ("tokens", "calls", "positions")}
           for p in ("recompute", "per_question", "fork")}
    return {"family": vol["family"], "template": vol["template"], "rows": len(ids), "refused": 0, "totals": tot,
            "source": "jobs/common/readout_volume_qwen3.5.json per_row over jobs/common/dev_proxy600.json run_ids"}


def project(params: dict, cfg: dict) -> dict:
    from jobs.common.readout_cost import project as readout
    from jobs.full8.preflight import stage2_rates

    m, e = params["measured"], params["estimates_not_measured"]
    B, L = cfg["batch"], cfg["seq_len"]
    rates = stage2_rates(m)  # worst = first student layer 3, all 8 GDNs training: the v2 stage-2 shape
    s2_rate = min(rates["worst"]["tok_s"], m["stage2_steady_tok_s_layer3_7gdn"])
    s1_min, s2_min = cfg["stage1"]["guard"]["max_minutes"], cfg["stage2"]["guard"]["max_minutes"]
    capped = s1_min + s2_min + (B * L / m["stage1_steady_tok_s"] + B * L / s2_rate) / 60
    n_curve = int(s2_min // cfg["stage2"]["eval_every_min"]) + 2  # periodic points + the stop point + after stage 1
    uncapped = (m["load_and_data_s"] / 60 + m["stock_probes_s"] / 60 + e["init_cmp_min"]
                + n_curve * e["curve_eval_min_each"] + e["final_heldout_min"] + e["wikitext_gate_min"]
                + m["converted_probes_s"] / 60 + 2 * e["mmlu_full_min_per_arm"] + 2 * m["throughput_s_per_arm"] / 60)
    pv = proxy_volume()
    arm = readout(pv["totals"], m["stock_prefill_tok_s_min"], max(m["pack_call_s"], m["small_call_s"]), "fork")
    tokens = {"init_cmp": 2 * (cfg["init_cmp"]["steps"] + 1) * B * L,
              "stage1_at_1.15x_measured_rate": 1.15 * m["stage1_steady_tok_s"] * 60 * s1_min,
              "stage2_at_1.15x_measured_rate": 1.15 * max(s2_rate, m["stage2_steady_tok_s_layer3"]) * 60 * s2_min}
    return {"phases": {"setup": m["setup_min"], "weights": m["weights_fetch_s"] / 60, "D": e["D_min"],
                       "X": e["export_X_min"], "I_stock": arm["projected_min"], "I_conv": arm["projected_min"],
                       "S": m["S_min"]},
            "P": {"capped": capped, "uncapped": uncapped, "n_curve_points": n_curve},
            "stage2_rate_worst_tok_s": s2_rate, "stage2_rates": rates, "proxy_volume": pv["totals"]["fork"],
            "train_tokens_needed": tokens, "train_tokens_needed_total": sum(tokens.values())}


def check(params: dict, cfg: dict) -> dict:
    pr = project(params, cfg)
    mf = params["margin_min_frac"]
    caps = params["phase_caps_minutes"]
    m = params["measured"]
    rows = {}
    for ph, proj in pr["phases"].items():
        usable = caps[ph] * (0.95 if ph.startswith("I_") else 1.0)
        rows[ph] = {"projected_min": round(proj, 2), "cap_min": caps[ph], "ok": usable >= (1 + mf) * proj}
    need = pr["P"]["capped"] + (1 + mf) * pr["P"]["uncapped"]
    rows["P"] = {"capped_min": round(pr["P"]["capped"], 2), "uncapped_min": round(pr["P"]["uncapped"], 2),
                 "need_min": round(need, 2), "cap_min": caps["P"], "ok": caps["P"] >= need}
    floors = {"stage1": (cfg["stage1"]["guard"]["min_tok_s"], m["stage1_steady_tok_s"]),
              "stage2": (cfg["stage2"]["guard"]["min_tok_s"], pr["stage2_rate_worst_tok_s"])}
    for k, (floor, rate) in floors.items():
        rows[f"floor_{k}"] = {"min_tok_s": floor, "slowest_rate_tok_s": round(rate), "ok": floor <= (1 - mf) * rate}
    w = pr["stage2_rates"]["worst"]
    for st in ("stage1", "stage2"):
        stop = cfg[st]["guard"]["max_peak_gb"]
        rows[f"peak_{st}"] = {"measured_gb": m["peak_gb"], "predicted_worst_gb": round(w["peak_gb"], 1), "stop_gb": stop,
                              "ok": max(m["peak_gb"], w["peak_gb"]) <= (1 - mf) * stop}
    rows["train_stream"] = {"tokens": cfg["mix"]["train_tokens"], "needed": round(pr["train_tokens_needed_total"]),
                            "ok": cfg["mix"]["train_tokens"] >= pr["train_tokens_needed_total"]}
    rows["shares"] = {"sum": sum(cfg["mix"]["shares"].values()), "ok": abs(sum(cfg["mix"]["shares"].values()) - 1) < 1e-9}
    rows["curve_windows"] = {"n": cfg["curve_windows_per_source"],
                             "ok": cfg["curve_windows_per_source"] % 4 == 0
                             and all(cfg["curve_windows_per_source"] <= n for n in cfg["heldout_windows"].values())}
    rows["heldout_tokens"] = {"ok": all(cfg["mix"]["heldout_tokens"][s] >= n * cfg["seq_len"] + 1
                                        for s, n in cfg["heldout_windows"].items())}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    from jobs.common.costs import worst_case_usd

    rows["hard_cap"] = {"worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"],
                        "ok": worst_case_usd(params) <= params["hard_cap_usd"]}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, **pr}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_v2.preflight")
    ap.add_argument("--params", default=str(ROOT / "jobs/q9af_v2/job_params.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/q9af_v2_h100.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--volume-out", default=None, help="write the proxy600 readout volume (the I-arm rate preflight)")
    a = ap.parse_args(argv)
    res = check(json.loads(Path(a.params).read_text(encoding="utf-8")),
                json.loads(Path(a.config).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    if a.volume_out:
        Path(a.volume_out).write_text(json.dumps(proxy_volume(), indent=1), encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())

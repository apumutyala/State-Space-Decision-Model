"""Measured-rate pricing for the Clef jobs (JOB_SPEC.md sections 4b and 11). CPU only, no weights.

    python -X utf8 -m jobs.clef_baseline.price          # projections + cap margins for every job_params_*.json

Basis (receipts, not guesses):
  * Clef-flash rates are MEASURED by job-sgihj (H100 80GB SXM, clock_lock, batch 1, decide path, our pinned stack):
    stage timestamps from its stages.log and per-row ``total_wall_ms`` from its index/decide/results.jsonl
    (12,104 rows, 12,053 ok, 18.60 M input tokens, 18.86 min summed; least-squares per-row model below).
  * Clef (27B) is projected from those rates scaled by the ratio of per-token matmul FLOPs of the two text backbones,
    computed from their config.json at the pinned revisions (CONFIGS). The vision encoder and lm_head never run on the
    Index path (Clef's head reads ``last_hidden_state``), so they are excluded; the joint head is small in both.
  * Cloudflare's card gives a second, independent ratio: median latency 209.3 / 38.8 ms = 5.39x (p95 1.95x), on an
    H200 with their stack. It is reported as the stress case: caps are sized on the FLOPs projection x >= 1.3, and the
    in-job live rate guard (jobs/common/rate_guard.py run, exit 18 in run_job.sh) stops a slower box before the cap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE_USD_PER_MIN = 0.0495  # SF batch H100 $2.97/h (job-sgihj receipt: 49,500 micro-$ per GPU-minute)

#: text-backbone config fields at the pinned revisions (huggingface.co/Cloudflare/<repo>/resolve/<rev>/config.json,
#: fetched 2026-10-06; text_config). attn_output_gate=True doubles q_proj (query + gate), as in Qwen3-Next/3.5.
CONFIGS = {
    "clef-flash": {"revision": "17f0b0ad64efb65d273590632833508766b2aae6", "hidden_size": 4096,
                   "num_hidden_layers": 32, "full_attention_interval": 4, "intermediate_size": 12288,
                   "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True,
                   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                   "linear_value_head_dim": 128, "repo_bytes": 19083377402},
    "clef": {"revision": "2f3de3dd85f379784083b0814d997ab627200f0c", "hidden_size": 5120, "num_hidden_layers": 64,
             "full_attention_interval": 4, "intermediate_size": 17408, "num_attention_heads": 24,
             "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True, "linear_num_key_heads": 16,
             "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
             "repo_bytes": 54989894057},
}

#: job-sgihj (Clef-flash, job A) measurements. Stage wall minutes from stages.log; the per-row model is the
#: least-squares fit total_wall_ms = a + b * sequence_tokens over its 12,053 ok rows (r^2 0.32 per row, but with an
#: intercept the fitted SUM equals the measured sum, which is what a phase cap needs).
SGIHJ = {
    "job": "job-sgihj", "billed_seconds": 1692, "charged_usd": 1.3959,
    "stage_minutes": {"setup": 2.88, "probes_with_grad_probe": 5.22, "index_decide_full": 19.48,
                      "index_native_115_rows": 0.63},
    "decide_rows": 12104, "decide_ok_rows": 12053, "decide_tokens": 18603951, "decide_sum_min": 18.86,
    "subset_rows": 2008, "subset_sum_min": 3.27, "subset_tokens": 3146552,
    "row_ms_intercept": 32.00, "row_ms_per_token": 0.03993,
    "latency_ms": {"median": 57.15, "p95": 286.47, "note": "ok rows after 10 warm-up rows; H100, our stack"},
    "kit_elapsed_s": {"decide": 1136.3, "native": 15.8},
}
#: local CPU timings of the S3 scoring commands on the job-sgihj files (harness score x2 + ece + dev_subset score)
SCORE_CPU_MIN = 1.3
CARD_MEDIAN_RATIO = 209.3 / 38.8  # Cloudflare card, H200: Clef / Clef-flash median latency
CARD_P95_RATIO = 238.6 / 122.4


def matmul_params_per_token(c: dict) -> int:
    """Weights every token multiplies through (text backbone only): GDN in_proj_qkvz + in_proj_ba + out_proj, gated
    full attention q(+gate)/k/v/o, SwiGLU MLP."""
    h, L = c["hidden_size"], c["num_hidden_layers"]
    n_full = L // c["full_attention_interval"]
    n_lin = L - n_full
    kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    gdn = h * (2 * kd + 2 * vd) + h * 2 * c["linear_num_value_heads"] + vd * h
    qd = c["num_attention_heads"] * c["head_dim"]
    attn = h * qd * (2 if c["attn_output_gate"] else 1) + 2 * h * c["num_key_value_heads"] * c["head_dim"] + qd * h
    mlp = 3 * h * c["intermediate_size"]
    return n_lin * gdn + n_full * attn + L * mlp


def flops_ratio() -> float:
    return matmul_params_per_token(CONFIGS["clef"]) / matmul_params_per_token(CONFIGS["clef-flash"])


def decide_minutes(release: str, rows: int, tokens: float, ratio: float | None = None) -> float:
    """Projected summed kit wall minutes for ``rows`` requests of ``tokens`` total input tokens, batch 1."""
    r = 1.0 if release == "clef-flash" else (flops_ratio() if ratio is None else ratio)
    return r * (rows * SGIHJ["row_ms_intercept"] + tokens * SGIHJ["row_ms_per_token"]) / 60000.0


def load_minutes(release: str) -> float:
    """Model load + warm-up inside one phase: Clef-flash ~0.6 min in job-sgihj (stage 19.48 - kit 18.94); scaled by
    weight bytes for Clef."""
    base = SGIHJ["stage_minutes"]["index_decide_full"] - SGIHJ["kit_elapsed_s"]["decide"] / 60
    return base * CONFIGS[release]["repo_bytes"] / CONFIGS["clef-flash"]["repo_bytes"]


#: volumes the jobs run on. DEV: job-sgihj tokens (Clef tokenizer; Clef and Clef-flash share it). Teacher: the
#: 24,000-record draw / the 149,997-record pool of the 150k set, tokens = proxy chars/4 x 1.381 (the DEV
#: tokens/proxy ratio measured in job-sgihj), computed locally from the records (2026-10-06).
VOLUMES = {
    "dev_full": {"rows": 12104, "tokens": 18603951},
    "dev_subset": {"rows": 2008, "tokens": 3146552},
    "teacher_draw_24k": {"rows": 24000, "tokens": 45200200, "over_16k_tokens_est": 550},
    "teacher_pool_150k": {"rows": 149997, "tokens": 279600000, "over_16k_tokens_est": 3427},
}


def projection(release: str, volume: str) -> dict:
    v = VOLUMES[volume]
    m = decide_minutes(release, v["rows"], v["tokens"])
    out = {"release": release, "volume": volume, **v, "decide_min": round(m, 1),
           "load_min": round(load_minutes(release), 1)}
    if release == "clef":
        out["stress_card_median_ratio_min"] = round(decide_minutes(release, v["rows"], v["tokens"], CARD_MEDIAN_RATIO), 1)
    return out


def params_files() -> list[Path]:
    return sorted(HERE.glob("job_params_*.json"))


def check_params(p: dict) -> dict:
    """Every phase cap >= 1.3x its expected minutes; caps + reserve == max_duration (LAUNCH_CHECKLIST rows 2-3)."""
    caps, exp = p["phase_caps_minutes"], p["expected_minutes"]
    margins = {k: round(caps[k] / exp[k], 2) for k in caps if exp.get(k)}
    return {"margins": margins, "min_margin": min(margins.values()),
            "schedule_ok": sum(caps.values()) + p["reserve_minutes"] == p["max_duration_minutes"]}


PREFLIGHT_WARMUP_ROWS = 3  # the first decide calls after load include compile/allocator warm-up
PREFLIGHT_MARGIN = 0.95    # same rule as jobs/common/rate_guard.py: stop if the projection is past 95 % of the cap
EXIT_TOO_SLOW = 15         # rate_guard's code; run_job.sh maps it to 18


def preflight(probes: dict, volume: str, cap_min: float) -> dict:
    """Measured-rate preflight before the Index phase. The S1 parity sample runs real DEV rows through the same
    decide path on THIS box; s = measured ms / the job-sgihj Clef-flash model's ms for those same rows (token counts
    from the run). Projected phase minutes = s x the Clef-flash projection for the job's volume + load. For Clef this
    replaces the FLOPs-ratio assumption with a measurement before the long phase starts."""
    t = [x for x in probes.get("parity_timing") or [] if x.get("tokens")][PREFLIGHT_WARMUP_ROWS:]
    if not t:
        return {"ok": True, "measured": False, "note": "no parity timing (old probes file); live guard only"}
    meas = sum(x["ms"] for x in t)
    model = sum(SGIHJ["row_ms_intercept"] + SGIHJ["row_ms_per_token"] * x["tokens"] for x in t)
    s = meas / model
    v = VOLUMES[volume]
    proj = s * decide_minutes("clef-flash", v["rows"], v["tokens"]) + load_minutes(probes.get("release", "clef"))
    return {"ok": proj <= PREFLIGHT_MARGIN * cap_min, "measured": True, "n_rows": len(t),
            "speed_vs_flash_sgihj": round(s, 3), "projected_min": round(proj, 1), "cap_min": cap_min,
            "volume": volume, "flops_ratio_prediction": round(flops_ratio(), 3)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "preflight":
        import argparse

        ap = argparse.ArgumentParser(prog="python -m jobs.clef_baseline.price preflight")
        ap.add_argument("cmd")
        ap.add_argument("--probes", required=True)
        ap.add_argument("--volume", required=True, choices=sorted(VOLUMES))
        ap.add_argument("--cap-min", type=float, required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--advisory", action="store_true", help="record only (CPU dry run: a stand-in's rate is meaningless)")
        a = ap.parse_args(argv)
        r = preflight(json.loads(Path(a.probes).read_text(encoding="utf-8")), a.volume, a.cap_min)
        r["advisory"] = a.advisory
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r))
        return 0 if (r["ok"] or a.advisory) else EXIT_TOO_SLOW
    from jobs.common import costs

    rows = {"flops_ratio_clef_over_flash": round(flops_ratio(), 3),
            "matmul_params_per_token": {k: matmul_params_per_token(c) for k, c in CONFIGS.items()},
            "card_ratios": {"median": round(CARD_MEDIAN_RATIO, 2), "p95": round(CARD_P95_RATIO, 2)},
            "projections": [projection(r, v) for r in ("clef-flash", "clef") for v in VOLUMES], "jobs": {}}
    ok = True
    for f in params_files():
        p = json.loads(f.read_text(encoding="utf-8"))
        c = check_params(p)
        rows["jobs"][f.name] = {**c, "worst_case_usd": costs.worst_case_usd(p), "expected_usd": costs.expected_usd(p),
                                "single_attempt_usd": costs.single_attempt_usd(p), "hard_cap_usd": p["hard_cap_usd"]}
        ok &= c["min_margin"] >= 1.3 and c["schedule_ok"] and costs.cap_ok(p)
    print(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Max-throughput (batched) projection on one H100 80GB: Clef 27B, Clef-flash, RWKV7-13.3B (stock + trained-with-fixes).

CPU only, $0, no weights, no GPU. Run from the repo root:

    python -I -X utf8 analysis/throughput_maxbatch.py [--json out.json]

Workload: the Decision Index ``decide`` request (prefill-only scoring, NO token generation). Every arm reads a row's
prompt once (Clef: one sequence per record with every question in it, ``use_cache=False``; RWKV: prefix once + packed
(question, option) tails from copies of the prefix state) and scores options from hidden states / logits.

What it does, per arm, on the 2,000 run_ids that are ``ok`` in every arm of analysis/cost_latency.py (same rows):
  1. memory model: weight bytes, transient activation bytes / token, per-sequence state / KV bytes as a function of
     prompt length, and the max batch (rows per forward) that fits 80 GB at the DEV length mix;
  2. compute model: FLOPs / row (matmul + attention + linear-attention state math) from the configs;
  3. calibration: least-squares ``ms = a + b*N + c*N^2`` on the MEASURED batch-1 rows (kit ``total_wall_ms``; N =
     the arm's own model tokens), and the achieved batch-1 MFU;
  4. projection: T0 = batch 1 as measured; T1 = max batch on today's kernels (per-row overhead amortised, every
     token at the arm's own large-call rate = sum(ms)/sum(N) over rows >= 3k tokens; fla arms: the jn47u fla prefill
     rate x the readout's long-row overhead); T2 = compute roof at an assumed MFU eta (needs kernel work on both
     sides). $/1k rows at the billed H100 rate; RWKV/Clef cost ratios under matched treatment;
  5. a cost estimate for the 250-row smoke-slice GPU measurement (spec only).

All inputs are landed artifacts / committed files read as data (tar members read in place, nothing extracted or run).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # -I drops the script dir; the repo's own modules are trusted code

from analysis.cost_latency import DEFAULT_ARMS, H, RATE_USD_PER_MIN, iter_rows, load_arm  # noqa: E402
from jobs.clef_baseline.price import CONFIGS, matmul_params_per_token  # noqa: E402

GiB = 1024 ** 3
# ---------------------------------------------------------------- hardware (H100 80GB HBM3 SXM, the billed box)
PEAK_BF16 = 989.4e12          # dense bf16 tensor-core FLOP/s
HBM_BW = 3.35e12              # B/s
USABLE = 79.6 * GiB           # 81,559 MiB reported by nvidia-smi in the Clef jobs
RESERVE = 3.0 * GiB           # CUDA context, allocator fragmentation, cuBLAS/triton workspaces
ETAS = (0.40, 0.55, 0.70)     # achievable end-to-end prefill MFU at large batch (low / central / high)

# ---------------------------------------------------------------- models
#: Clef text backbones: jobs/clef_baseline/price.py CONFIGS (config.json at the pinned revisions); conv kernel 4 and
#: GDN chunk 64 are the Qwen3.5 / fla defaults.
CLEF = {
    "clef27b": {**CONFIGS["clef"], "weights_bytes": 27_356_728_560 * 2, "conv": 4},
    "flash": {**CONFIGS["clef-flash"], "weights_bytes": 9_409_813_744 * 2, "conv": 4},
}
#: RWKV7-G1k-13.3B: heads/backbone.py BackboneSpec + jobs/rwkv7/JOB_SPEC.md (61 L, d 4096, 64 heads x 64, FFN 16384,
#: fp32 WKV state). params = 12.73e9 non-embedding, non-head (the head runs only on option-token positions).
RWKV = {"layers": 61, "d": 4096, "heads": 64, "hd": 64, "ffn": 16384, "matmul_params": 12.73e9,
        "weights_bytes": 26.5e9, "state_bytes": 61 * (64 * 64 * 64 * 4 + 2 * 4096 * 2)}

#: transient activation bytes per token of a no-grad forward (layer outputs are freed as the forward advances)
ACT_PER_TOKEN = {
    # job-sgihj: Clef-flash peak 20.34 GiB after a 16k forward (+ a GCG step), 17.5 GiB weights (JOB_SPEC 4c)
    "flash": (20.34 - 17.5) * GiB / 16384,
    # scaled by the widest per-layer transient (intermediate 17,408 vs 12,288); JOB_SPEC's 2.5x is the stress case
    "clef27b": (20.34 - 17.5) * GiB / 16384 * 17408 / 12288,
    "clef27b_stress": (20.34 - 17.5) * GiB / 16384 * 2.5,
    # job-mhc3h probe: 36 GB peak with 26.5 GB weights at the 16k prefill rung (PERF_AUDIT / HANDOFF), chunked WKV
    "rwkv7": (36e9 - 26.5e9) / 16384,
}
#: measured RWKV7 batch-1 prefill rates (jobs/fla_rwkv7/JOB_SPEC.md: jn47u; PERF_AUDIT: mhc3h), tok/s
RWKV_PREFILL = {"chunked": 3950.0, "fla": 11900.0}


def clef_flops(cfg: dict, n: float) -> float:
    """Forward FLOPs of one Clef record of n tokens: 2 x matmul params x n, causal full attention (QK^T + PV,
    2*qd*n^2 per layer), GDN chunked delta rule (~6*dk*dv + 2*C*(dk+dv) per value head per token)."""
    L = cfg["num_hidden_layers"]
    n_full = L // cfg["full_attention_interval"]
    qd = cfg["num_attention_heads"] * cfg["head_dim"]
    dk, dv, C = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], 64
    gdn_tok = (L - n_full) * cfg["linear_num_value_heads"] * (6 * dk * dv + 2 * C * (dk + dv))
    return 2.0 * matmul_params_per_token(cfg) * n + 2.0 * n_full * qd * n * n + gdn_tok * n


def rwkv_flops(n: float) -> float:
    """2 x matmul params x n + WKV7 chunked state math (~8*hd^2 + 2*C*2*hd per head per token, C = 16)."""
    wkv_tok = RWKV["layers"] * RWKV["heads"] * (8 * RWKV["hd"] ** 2 + 2 * 16 * 2 * RWKV["hd"])
    return 2.0 * RWKV["matmul_params"] * n + wkv_tok * n


def clef_cache_bytes(cfg: dict, n: float) -> dict:
    """What a cache-resident serving engine (vLLM-style) would keep per sequence: KV of the full-attention layers
    (bf16) + GDN recurrent (fp32) and conv states. The Index path keeps NONE of it (use_cache=False)."""
    L = cfg["num_hidden_layers"]
    n_full = L // cfg["full_attention_interval"]
    kv_tok = n_full * 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    gdn = (L - n_full) * (cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"] * cfg["linear_value_head_dim"] * 4
                          + (2 * kd + vd) * (cfg["conv"] - 1) * 2)
    return {"kv_bytes_per_token": kv_tok, "gdn_state_bytes": gdn, "total": gdn + kv_tok * n}


# ---------------------------------------------------------------- data
def _git_show(ref_path: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", ref_path], capture_output=True, text=True, check=True,
                          encoding="utf-8").stdout


def load_rows() -> tuple[list[str], dict]:
    arms = {a["arm"]: a for a in DEFAULT_ARMS}
    subset = {rid for rid, _ in iter_rows(arms["d1_child"])}
    data = {k: load_arm(a, subset) for k, a in arms.items()}
    ok = {k: {i for i, r in v.items() if r["status"] == "ok" and isinstance(r["ms"], (int, float))}
          for k, v in data.items()}
    common = sorted(set.intersection(*ok.values()))
    return common, data


def fit(ns: np.ndarray, ms: np.ndarray) -> dict:
    """ms = a + b*N + c*N^2, least squares; c clamped >= 0 (refit linear if negative)."""
    X = np.stack([np.ones_like(ns), ns, ns * ns], 1)
    coef, *_ = np.linalg.lstsq(X, ms, rcond=None)
    if coef[2] < 0:
        c2, *_ = np.linalg.lstsq(X[:, :2], ms, rcond=None)
        coef = np.array([c2[0], c2[1], 0.0])
    pred = X @ coef
    r2 = 1 - ((ms - pred) ** 2).sum() / ((ms - ms.mean()) ** 2).sum()
    return {"a_ms": float(coef[0]), "b_ms_per_tok": float(coef[1]), "c_ms_per_tok2": float(coef[2]), "r2": float(r2),
            "a_share": float(coef[0] * len(ns) / ms.sum())}


def run() -> dict:
    common, data = load_rows()
    vol = json.loads((REPO / "jobs/common/readout_volume_rwkv7.json").read_text(encoding="utf-8"))["per_row"]
    n_opt = np.array([vol[i]["n_options"] for i in common], float)
    n_q = np.array([vol[i]["n_questions"] for i in common], float)
    toks = {
        "clef27b": np.array([data["clef27b_fast"][i]["model_tokens"] for i in common], float),
        "flash": np.array([data["flash_fast"][i]["model_tokens"] for i in common], float),
        "rwkv7_v2": np.array([data["rwkv7_stock_v2"][i]["model_tokens"] for i in common], float),
        "rwkv7_v3w": np.array([data["rwkv7_stock_v3w"][i]["model_tokens"] for i in common], float),
        # g1-v1 template (parent arm and D1): the prefix-fork token volume of the readout (positions incl. padding)
        "rwkv7_g1v1": np.array([vol[i]["fork"]["positions"] for i in common], float),
    }
    arm_tok = {"clef27b_fast": "clef27b", "flash_fast": "flash", "rwkv7_stock_v2": "rwkv7_v2",
               "rwkv7_stock_v3w": "rwkv7_v3w", "rwkv7_parent_c7pq2": "rwkv7_g1v1", "d1_child": "rwkv7_g1v1"}
    flops_fn = {"clef27b": lambda n: clef_flops(CLEF["clef27b"], n), "flash": lambda n: clef_flops(CLEF["flash"], n)}

    out: dict = {"rows": len(common), "rate_usd_per_min": RATE_USD_PER_MIN, "hardware": {
        "peak_bf16_flops": PEAK_BF16, "hbm_Bps": HBM_BW, "usable_GiB": USABLE / GiB, "reserve_GiB": RESERVE / GiB,
        "ridge_flop_per_byte": PEAK_BF16 / HBM_BW}, "length_mix": {}, "arms": {}}
    for k, n in toks.items():
        out["length_mix"][k] = {"mean": float(n.mean()), "median": float(np.median(n)),
                                "p95": float(np.percentile(n, 95)), "max": float(n.max()), "sum": float(n.sum())}
    out["length_mix"]["n_options_mean"] = float(n_opt.mean())
    out["length_mix"]["n_options_p95"] = float(np.percentile(n_opt, 95))
    out["length_mix"]["n_questions_mean"] = float(n_q.mean())
    mp = np.array([vol[i]["max_prompt"] for i in common], float)
    out["length_mix"]["g1v1_longest_question_row_mean"] = float(mp.mean())
    out["length_mix"]["g1v1_fork_tail_share"] = float(1 - mp.sum() / toks["rwkv7_g1v1"].sum())

    def usd(sec_per_row: float) -> float:
        return sec_per_row / 60.0 * RATE_USD_PER_MIN * 1000.0

    for arm, tk in arm_tok.items():
        n = toks[tk]
        ms = np.array([data[arm][i]["ms"] for i in common], float)
        fam = "clef" if tk in ("clef27b", "flash") else "rwkv7"
        fl = flops_fn[tk](n) if fam == "clef" else rwkv_flops(n)
        f = fit(n, ms)
        tot_s = ms.sum() / 1000
        rec = {"tokens": tk, "measured_b1": {"rows_per_s": len(n) / tot_s, "s_per_row": tot_s / len(n),
                                             "usd_per_1k": usd(tot_s / len(n)), "tok_per_s": n.sum() / tot_s,
                                             "mfu": fl.sum() / tot_s / PEAK_BF16},
               "flops_per_row_mean": float(fl.mean()), "fit_b1": f}
        out["arms"][arm] = rec

    # ---- projections at max batch
    # T0 = batch 1 as measured. T1 = max batch with TODAY's kernels: per-row overhead amortised, every token at the
    # arm's own large-call rate, i.e. sum(ms)/sum(N) over its rows with N >= LONG (where overhead is a small share);
    # for the RWKV fla arms, the fla prefill rate measured on the same box (jn47u). T2 = compute roof at eta (needs
    # kernel / graph work on both sides). Memory-bound floor per call = weight bytes / HBM bandwidth.
    LONG = 3000

    def asym(arm: str) -> float:
        n, ms = toks[arm_tok[arm]], np.array([data[arm][i]["ms"] for i in common], float)
        m = n >= LONG
        return float(ms[m].sum() / 1000 / n[m].sum())

    def roof(tk: str, eta: float) -> float:
        n = toks[tk]
        fl = flops_fn[tk](n) if tk in flops_fn else rwkv_flops(n)
        return float((fl / (eta * PEAK_BF16)).sum() / len(n))

    proj = {}
    for name, tk, arm, rate in [
        ("Clef 27B (fast fla GDN)", "clef27b", "clef27b_fast", None),
        ("Clef-flash (fast fla GDN)", "flash", "flash_fast", None),
        ("RWKV7 stock g1-v2, chunked WKV", "rwkv7_v2", "rwkv7_stock_v2", None),
        ("RWKV7 stock g1-v2, fla WKV", "rwkv7_v2", "rwkv7_stock_v2", RWKV_PREFILL["fla"]),
        ("RWKV7 D1 + exact fixes, chunked_lean WKV", "rwkv7_g1v1", "rwkv7_parent_c7pq2", None),
        ("RWKV7 D1 + exact fixes + fla WKV", "rwkv7_g1v1", "rwkv7_parent_c7pq2", RWKV_PREFILL["fla"]),
    ]:
        n = toks[tk]
        a_ms = out["arms"][arm]["fit_b1"]["a_ms"]
        # fla arms: the fla prefill rate, carrying the same readout overhead the chunked v2 arm shows on long rows
        # over ITS pure-prefill rate (asym(v2) * chunked prefill tok/s, ~1.06)
        s_tok = (1.0 / rate) * asym("rwkv7_stock_v2") * RWKV_PREFILL["chunked"] if rate else asym(arm)
        p = {"tokens": tk, "tokens_per_row": float(n.mean()), "per_token_us_T1": s_tok * 1e6,
             "T1_tok_per_s": 1.0 / s_tok}
        if rate is None:
            p["T0_s_per_row"] = out["arms"][arm]["measured_b1"]["s_per_row"]
        else:  # batch 1 with fla: the arm's own per-row overhead + its tokens at the fla prefill rate
            p["T0_s_per_row"] = float(a_ms / 1000 + n.mean() * s_tok)
        p["T1_s_per_row"] = float(n.mean() * s_tok)
        p["T1_mfu"] = float((flops_fn[tk](n) if tk in flops_fn else rwkv_flops(n)).mean() / p["T1_s_per_row"] / PEAK_BF16)
        p["T2_s_per_row"] = {f"{e:.2f}": roof(tk, e) for e in ETAS}
        wb = CLEF[tk]["weights_bytes"] if tk in CLEF else RWKV["weights_bytes"]
        p["mem_floor_ms_per_call"] = wb / HBM_BW * 1000
        p["T0_usd_per_1k"], p["T1_usd_per_1k"] = usd(p["T0_s_per_row"]), usd(p["T1_s_per_row"])
        p["T2_usd_per_1k"] = {e: usd(v) for e, v in p["T2_s_per_row"].items()}
        proj[name] = p
    # D1 batch-1 with the exact fixes = the memo's projection (~3.0x on D1's measured 1.83 s/row; about parent speed)
    d1 = out["arms"]["d1_child"]["measured_b1"]["s_per_row"]
    proj["RWKV7 D1 + exact fixes, chunked_lean WKV"]["T0_s_per_row"] = d1 / 3.0
    proj["RWKV7 D1 + exact fixes, chunked_lean WKV"]["T0_usd_per_1k"] = usd(d1 / 3.0)
    fx = proj["RWKV7 D1 + exact fixes + fla WKV"]
    fx["T0_s_per_row"] = d1 / 3.0 - proj["RWKV7 D1 + exact fixes, chunked_lean WKV"]["T1_s_per_row"] + fx["T1_s_per_row"]
    fx["T0_usd_per_1k"] = usd(fx["T0_s_per_row"])
    out["projections"] = proj

    # ---- memory: max batch at the DEV mix
    mem = {}
    for key, wb, act, tk, fixed in [
        ("clef27b", CLEF["clef27b"]["weights_bytes"], ACT_PER_TOKEN["clef27b"], "clef27b", 0.0),
        ("clef27b_stress", CLEF["clef27b"]["weights_bytes"], ACT_PER_TOKEN["clef27b_stress"], "clef27b", 0.0),
        ("flash", CLEF["flash"]["weights_bytes"], ACT_PER_TOKEN["flash"], "flash", 0.0),
        ("rwkv7_v2", RWKV["weights_bytes"], ACT_PER_TOKEN["rwkv7"], "rwkv7_v2", RWKV["state_bytes"]),
        ("rwkv7_g1v1", RWKV["weights_bytes"], ACT_PER_TOKEN["rwkv7"], "rwkv7_g1v1", RWKV["state_bytes"]),
    ]:
        free = USABLE - RESERVE - wb
        lm = out["length_mix"][tk]
        pad = 1.05  # length-sorted bucketing: ~5% padding
        per_row = act * lm["mean"] * pad + fixed
        m = {"weights_GiB": wb / GiB, "free_GiB": free / GiB, "act_KiB_per_token": act / 1024,
             "per_row_MiB_at_mean": per_row / 2 ** 20, "tokens_in_flight_max": free / act,
             "max_batch_rows_at_mean": int(free // per_row),
             "max_batch_rows_at_p95": int(free // (act * lm["p95"] * pad + fixed))}
        if key.startswith("clef"):
            cb = clef_cache_bytes(CLEF["clef27b" if key.startswith("clef27b") else "flash"], lm["mean"])
            m["engine_cache_per_seq_MiB_at_mean"] = cb["total"] / 2 ** 20
            m["engine_kv_KiB_per_token"] = cb["kv_bytes_per_token"] / 1024
            m["engine_gdn_state_MiB"] = cb["gdn_state_bytes"] / 2 ** 20
            m["engine_max_resident_seqs"] = int(free // (cb["total"] + act * lm["mean"]))
        else:
            # the fork: one prefix state per request + one copy per (question, option) tail inside a packed call
            m["state_MiB_per_copy"] = RWKV["state_bytes"] / 2 ** 20
            m["fork_state_copies_mean_GiB"] = RWKV["state_bytes"] * out["length_mix"]["n_options_mean"] / GiB
            m["fork_state_copies_p95_GiB"] = RWKV["state_bytes"] * out["length_mix"]["n_options_p95"] / GiB
        # weights-streaming bound: tokens per forward needed before the matmuls are compute- not bandwidth-bound
        m["tokens_per_forward_to_reach_ridge"] = PEAK_BF16 / HBM_BW  # 2*P*T flops / 2*P bytes = T
        mem[key] = m
    out["memory"] = mem

    # ---- RWKV cost / Clef cost at MATCHED treatment (<1 = RWKV cheaper). T1 = both at max batch on today's kernels.
    ratios = {}
    for r in ("RWKV7 stock g1-v2, chunked WKV", "RWKV7 stock g1-v2, fla WKV",
              "RWKV7 D1 + exact fixes, chunked_lean WKV", "RWKV7 D1 + exact fixes + fla WKV"):
        for c in ("Clef 27B (fast fla GDN)", "Clef-flash (fast fla GDN)"):
            R, C = proj[r], proj[c]
            ratios[f"{r} / {c}"] = {"T0_batch1": R["T0_s_per_row"] / C["T0_s_per_row"],
                                    "T1_maxbatch_today": R["T1_s_per_row"] / C["T1_s_per_row"],
                                    "T2_equal_mfu": R["T2_s_per_row"]["0.55"] / C["T2_s_per_row"]["0.55"],
                                    "T1_rwkv_vs_T2_clef_0.55": R["T1_s_per_row"] / C["T2_s_per_row"]["0.55"],
                                    "T2_rwkv_0.70_vs_T1_clef": R["T2_s_per_row"]["0.70"] / C["T1_s_per_row"]}
    out["ratios_rwkv_over_clef"] = ratios
    # RWKV tokens/row that would give 2x (cost ratio 0.5) vs each Clef at equal MFU (pure FLOPs)
    need = {}
    for c in ("clef27b", "flash"):
        need[c] = float(clef_flops(CLEF[c], toks[c]).sum() / 2 / len(common) / rwkv_flops(1.0))
    out["rwkv_tokens_per_row_for_2x_at_equal_mfu"] = need
    # ... and at today's per-token rates (T1): fla rate vs each Clef's T1 cost per row
    out["rwkv_tokens_per_row_for_2x_at_T1_fla"] = {
        c: proj[k]["T1_s_per_row"] / 2 / (proj["RWKV7 stock g1-v2, fla WKV"]["per_token_us_T1"] / 1e6)
        for c, k in (("clef27b", "Clef 27B (fast fla GDN)"), ("flash", "Clef-flash (fast fla GDN)"))}
    out["smoke_slice"] = smoke_estimate(out, data)
    return out


def smoke_estimate(out: dict, data: dict) -> dict:
    """GPU cost of the spec'd measurement on the 250 pre-declared smoke rows (fulldev-eval branch)."""
    smoke = json.loads(_git_show("origin/stack/rwkv7/fulldev-eval:jobs/m2_first_cell/fulldev_smoke_rows.json"))
    lens = json.loads(_git_show("origin/stack/rwkv7/fulldev-eval:jobs/m2_first_cell/fulldev_lengths_rwkv7.json"))
    ids = smoke["run_ids"]
    # Clef tokens for every full-DEV row: zfexa (Clef 27B) and t5yg2 (Clef-flash) full runs report usage tokens
    arms = {a["arm"]: a for a in DEFAULT_ARMS}
    c27 = {rid: r for rid, r in iter_rows(arms["clef27b_ref"])}
    cfl = {rid: r for rid, r in iter_rows(arms["flash_ref"])}
    vol = json.loads((REPO / "jobs/common/readout_volume_rwkv7.json").read_text(encoding="utf-8"))["per_row"]
    ratio = statistics.median(v["fork"]["positions"] / v["max_prompt"] for v in vol.values() if v.get("max_prompt"))
    res = {"n": len(ids), "rwkv_fork_over_max_prompt_median": ratio, "arms": {}}
    over = [i for i in ids if lens["lengths"].get(i, 0) > 16384]
    res["rows_over_16384_rwkv"] = len(over)
    for name, tok_of, proj_key, arm_fit in [
        ("Clef 27B fast", lambda i: (c27.get(i) or {}).get("model_tokens"), "Clef 27B (fast fla GDN)", "clef27b_fast"),
        ("Clef-flash fast", lambda i: (cfl.get(i) or {}).get("model_tokens"), "Clef-flash (fast fla GDN)", "flash_fast"),
        ("RWKV7 D1 + fixes (chunked_lean)", lambda i: (vol[i]["fork"]["positions"] if i in vol else
                                                       (lens["lengths"].get(i) or 0) * ratio),
         "RWKV7 D1 + exact fixes, chunked_lean WKV", "rwkv7_parent_c7pq2"),
    ]:
        n = np.array([float(tok_of(i) or 0) for i in ids])
        if name.startswith("Clef"):
            n = np.minimum(n, 16384)  # over-length rows are refused before the forward (exhausted)
        f = out["arms"][arm_fit]["fit_b1"]
        b1 = (f["a_ms"] + f["b_ms_per_tok"] * n + f["c_ms_per_tok2"] * n * n).sum() / 1000
        p = out["projections"][proj_key]
        mb = p["per_token_us_T1"] / 1e6 * n.sum()  # T1 per-token rate on these tokens
        # passes at batch 1, 8, 32, max: expected = b1 + 3 x T1-ish (batch 8 taken at the b1/T1 midpoint);
        # worst case = all four at the batch-1 time (batching buys nothing)
        res["arms"][name] = {"tokens": float(n.sum()), "b1_min": b1 / 60, "maxbatch_T1_min": mb / 60,
                             "four_passes_expected_min": (b1 + (b1 + mb) / 2 + 2 * mb) / 60,
                             "four_passes_worst_min": 4 * b1 / 60}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = run()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    lm = out["length_mix"]
    print("rows", out["rows"], "| tokens/row mean:", {k: round(v["mean"]) for k, v in lm.items() if isinstance(v, dict)})
    for k, v in out["arms"].items():
        m, f = v["measured_b1"], v["fit_b1"]
        print(f"{k:22s} b1 {m['s_per_row']*1000:7.1f} ms/row ${m['usd_per_1k']:.3f}/1k  {m['tok_per_s']:7.0f} tok/s  "
              f"MFU {m['mfu']*100:4.1f}%  fit a={f['a_ms']:.1f}ms b={f['b_ms_per_tok']*1000:.2f}us/tok "
              f"c={f['c_ms_per_tok2']*1e6:.3f}ns/tok2 r2={f['r2']:.2f} a_share={f['a_share']:.2f}")
    for k, p in out["projections"].items():
        print(f"{k:42s} N={p['tokens_per_row']:5.0f} T0 ${p['T0_usd_per_1k']:.3f} | T1 ${p['T1_usd_per_1k']:.3f} "
              f"({p['T1_tok_per_s']:.0f} tok/s, MFU {p['T1_mfu']*100:.0f}%) | T2 " +
              " ".join(f"{e}:${u:.3f}" for e, u in p["T2_usd_per_1k"].items()) + f" | floor {p['mem_floor_ms_per_call']:.1f} ms/call")
    for k, m in out["memory"].items():
        print(k, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in m.items()})
    for k, r in out["ratios_rwkv_over_clef"].items():
        print(f"{k:80s}", {kk: round(float(vv), 2) for kk, vv in r.items()})
    print("rwkv tokens/row for 2x at equal MFU:", {k: round(v) for k, v in out["rwkv_tokens_per_row_for_2x_at_equal_mfu"].items()},
          "| at T1 fla rate:", {k: round(v) for k, v in out["rwkv_tokens_per_row_for_2x_at_T1_fla"].items()})
    print("smoke:", json.dumps(out["smoke_slice"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""MFU headroom: achieved MFU of RWKV7-13.3B (chunked / fla) vs Clef 27B / Clef-flash, where RWKV loses it, and what
the remaining levers could buy on $/1k DEV rows. CPU only, $0, no weights, no GPU. From the repo root:

    python -I -X utf8 analysis/mfu_headroom.py [--json out.json] [--jn47u PATH/throughput.json] [--mden7 PATH/throughput.json]

Inputs (all read as data):
  * the measured batch-1 arms and T1 projections of ``analysis/throughput_maxbatch.py`` (landed kit artifacts, 2,000
    common DEV-subset rows; same rate and peak);
  * the job-jn47u / job-mden7 ``throughput.json`` receipts (prefill at 2k/8k/16k for chunked / fla, and the WKV-only
    forward time at the same lengths). Their numbers are embedded below (``PREFILL``, ``WKV_FWD_MS``); pass the files
    to re-read them instead;
  * ``analysis/tree_fork/volume.json`` (tree-fork tokens per row, the exact-fix template);
  * the per-call constants already fitted elsewhere (PERF_AUDIT 0.098 s/call, D1 0.121 s/call, mden7 11.3 us/aten op).

FLOP convention (MFU numerator, "model FLOPs"): 2 x matmul params x tokens + the recurrence's minimal state math
(token-serial form: ~8 d^2 per head per token for RWKV7's diag+rank-1 update and readout; GDN as in
throughput_maxbatch.clef_flops; causal attention 2*qd*n^2 per full layer). The chunk-form WKV does more arithmetic
than that ("hardware FLOPs"); both are printed, neither moves MFU by more than 1.5%.

The per-layer DECOMPOSITION is a model calibrated on the jn47u measurements (prefill total - 61 x WKV-only time =
non-WKV time), then split by an op-by-op bytes/FLOPs count of the vendored block (third_party/rwkv7_hf
modeling_rwkv7.py). It is a hypothesis for the profile job in MFU_HEADROOM_2026-10-08.md section 5 to confirm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # -I drops the script dir; the repo's own modules are trusted code

from analysis import throughput_maxbatch as tm  # noqa: E402

PEAK = tm.PEAK_BF16          # 989.4e12 dense bf16 (H100 SXM)
PEAK_FP8 = 2 * PEAK          # dense fp8
HBM_ACH = 2.8e12             # achieved bandwidth of large eager elementwise kernels (~84% of 3.35 TB/s), assumption
GEMM_EFF = (0.65, 0.75)      # cuBLAS bf16 efficiency on M >= 2k, N,K in {4096, 16384} (assumption; profile confirms)
USD_PER_ROW_S = tm.RATE_USD_PER_MIN / 60 * 1000  # $/1k rows per (s/row)

# ---------------------------------------------------------------- RWKV7-13.3B structure (vendored config defaults)
L, C, H, D, FFN = 61, 4096, 64, 64, 16384
#: Rwkv7Config.__post_init__ at hidden 4096: rank_unit = sqrt(4096)/32 = 2 -> decay 160, a 160, v 96, gate 320.
#: (BlinkDL's training formula gives 128/128/96/480: 832 vs 736, +0.8% params; immaterial.)
RANKS = {"w": 160, "a": 160, "v": 96, "g": 320}
P_MAIN = 4 * C * C + 2 * C * FFN                  # r, k, v, o + FFN key/value = 201.3 M per layer
P_LORA = 2 * C * sum(RANKS.values())              # 6.03 M per layer (layer 0 skips v: ignored, 0.01%)
P_STRUCT = L * (P_MAIN + P_LORA)                  # 12.65 B (BackboneSpec says 12.73e9)
WKV_MODEL = 8 * D * D                              # per head per token, token-serial minimal
def wkv_chunk_flops(cs: int) -> int:               # per head per token, chunk form (rwkv7_chunked algebra)
    return 8 * D * D + 15 * cs * D


def rwkv_flops_per_token(params: float = tm.RWKV["matmul_params"], wkv: int = WKV_MODEL) -> float:
    return 2.0 * params + L * H * wkv


# ---------------------------------------------------------------- receipts (job-jn47u / job-mden7 throughput.json)
#: prefill s_median, B=1, random ids, logits_to_keep=1, bf16 weights, no autocast; median of 3 after a warm-up
PREFILL = {"jn47u": {"chunked": {2048: 0.54217, 8192: 1.84045, 16384: 4.01997},
                     "fla": {2048: 0.17554, 8192: 0.68350, 16384: 1.35866}},
           "mden7": {"chunked": {2048: 0.54346, 8192: 1.86189, 16384: 4.06152},
                     "fla": {2048: 0.17670, 8192: 0.68659, 16384: 1.36459}}}
#: WKV-only forward, one layer, B=1, 64 heads x 64, bf16 inputs + fp32 state (standalone; includes input clones)
WKV_FWD_MS = {"jn47u": {"chunked": {2048: 8.110, 8192: 29.520, 16384: 58.116},
                        "fla": {2048: 1.400, 8192: 3.176, 16384: 6.300}},
              "mden7": {"chunked": {2048: 8.152, 8192: 29.704, 16384: 58.948},
                        "fla": {2048: 1.127, 8192: 3.183, 16384: 6.288}}}
#: RWKV-7 paper (arXiv 2503.14456 sec. 8): BlinkDL CUDA WKV fwd, H100 SXM, B=8, T=16k, d 4096 (64 heads): 7.9 ms
PAPER_WKV = {"B": 8, "T": 16384, "fwd_ms_no_state": 7.9}

# ---------------------------------------------------------------- per-call / launch constants (receipts)
CALL_S_CHUNKED = 0.098       # PERF_AUDIT fit on job-mhc3h (stock readout, chunked, B=1): positions/tok_s + calls*call_s
CALL_S_D1 = 0.121            # COST_LATENCY fit (D1, chunked_lean, LoRA unmerged)
LORA_EXTRA_LAUNCHES = 366 * 6  # train/lora.py LoRALinear: 2 casts, 2 matmuls, mul, add per adapted Linear


def launches_per_layer(path: str, T: int) -> int:
    """Kernel launches per block forward, counted op by op from modeling_rwkv7 (no grad, keep=None, unpacked):
    ln1 1, token shift 2 (+delta 1), 6 mixes x 2, r/k/v 3, LoRA chains w 6 / a 4 / g 3 / v 4, v-mix 3, kk 1,
    l2-normalize 3, k-mod 4, state zeros 1, ln_x 1, bonus 4, +bonus 1, *g 1, output 1, residual 1;
    FFN: ln2 1, shift 2, delta 1, mix 2, key 1, relu 1, square 1, value 1, residual 1.  = 71 non-WKV."""
    base = 71
    if path == "fla":
        return base + 2 + 8 + 1                       # -kk, kk*a; ~8 fla Triton kernels; state cast back
    chunks = -(-T // 64)
    if path == "chunked":
        return base + 45 + 10 * chunks                # casts/pads/masks/einsums ~45 + ~10 per serial chunk
    if path == "chunked_lean":
        return base + 35 + 7 * chunks
    raise ValueError(path)


def bytes_per_layer_token() -> dict:
    """Elementwise HBM traffic of the eager block, bf16 activations (8 KiB per C-vector, 32 KiB per FFN vector), in
    KiB per token per layer: each eager op reads its inputs and writes its output once."""
    return {
        "token_shift (cat + delta, att + ffn)": 16 + 24 + 16 + 24,
        "6 att mixes + 1 ffn mix (mul + add each)": 6 * 40 + 40,
        "LoRA chains w/a/g/v (skinny mm in/out + gate ops)": 64 + 48 + 16 + 48,
        "v-residual mix, kk, l2norm, k-mod": 72 + 16 + 32 + 72,
        "WKV input prep (fla: -kk, kk*a)": 40,
        "norms ln1, ln2, ln_x": 16 + 16 + 24,
        "bonus, +bonus, *g": 112,
        "FFN relu + square (32 KiB vectors)": 128,
        "residual adds": 48,
    }


def decomposition(src: str = "jn47u") -> dict:
    """Per-token time split at 2k/8k/16k for fla and chunked, B=1, from the receipts."""
    out = {}
    byt = bytes_per_layer_token()
    ew_us = sum(byt.values()) * 1024 * L / HBM_ACH * 1e6
    gemm_fl = 2 * L * P_MAIN
    lora_fl = 2 * L * P_LORA
    for T in (2048, 8192, 16384):
        row = {}
        for path in ("fla", "chunked"):
            tot = PREFILL[src][path][T] / T * 1e6
            row[path] = {"total_us_per_tok": tot, "tok_s": 1e6 / tot,
                         "wkv_standalone_us_per_tok": WKV_FWD_MS[src][path][T] * L / T * 1e3}
        # non-WKV time is identical in both paths (same Linears, same elementwise ops); in-model WKV is solved from
        # the pair: nonwkv = fla_total - k*fla_wkv_sa, chunked_wkv = chunked_total - nonwkv, with k in [0.8, 1.0]
        # (standalone timing includes 7 input clones and per-call host work; k=0.8 matches chunked's in-model ratio)
        lo = row["fla"]["total_us_per_tok"] - row["fla"]["wkv_standalone_us_per_tok"]
        hi = row["fla"]["total_us_per_tok"] - 0.8 * row["fla"]["wkv_standalone_us_per_tok"]
        row["nonwkv_us_per_tok"] = (lo, hi)
        row["fla_wkv_inmodel_us_per_tok"] = (row["fla"]["total_us_per_tok"] - hi, row["fla"]["total_us_per_tok"] - lo)
        row["chunked_wkv_inmodel_us_per_tok"] = (row["chunked"]["total_us_per_tok"] - hi,
                                                 row["chunked"]["total_us_per_tok"] - lo)
        row["chunked_serial_loop_us_per_kernel"] = [
            w * T / (L * 10 * (T // 64)) for w in row["chunked_wkv_inmodel_us_per_tok"]]
        row["model"] = {
            "gemm_us_per_tok (65-75% cuBLAS)": [gemm_fl / (e * PEAK) * 1e6 for e in reversed(GEMM_EFF)],
            "gemm_us_per_tok_at_100pct": gemm_fl / PEAK * 1e6,
            "elementwise+norms+lora_us_per_tok (bytes model)": ew_us,
            "lora_gflop_per_tok": lora_fl / 1e9,
        }
        out[str(T)] = row
    out["bytes_KiB_per_layer_token"] = byt
    out["bytes_MB_per_token_all_layers"] = sum(byt.values()) * 1024 * L / 1e6
    return out


def mfu_table(tmo: dict) -> dict:
    """Achieved MFU, every arm, every regime, one FLOP convention."""
    f_tok = rwkv_flops_per_token()
    res = {"rwkv_flops_per_token": {"model_GF": f_tok / 1e9,
                                    "matmul_only_GF": 2 * tm.RWKV["matmul_params"] / 1e9,
                                    "wkv_model_GF": L * H * WKV_MODEL / 1e9,
                                    "wkv_chunk64_hw_GF": L * H * wkv_chunk_flops(64) / 1e9,
                                    "wkv_chunk16_hw_GF": L * H * wkv_chunk_flops(16) / 1e9,
                                    "structural_params_B": P_STRUCT / 1e9,
                                    "lora_mix_params_B": L * P_LORA / 1e9}}
    arms = {}
    for k, v in tmo["arms"].items():
        arms[k] = {"b1_e2e_mfu": v["measured_b1"]["mfu"], "b1_s_per_row": v["measured_b1"]["s_per_row"],
                   "flops_per_row_T": v["flops_per_row_mean"] / 1e12, "fit_a_ms": v["fit_b1"]["a_ms"],
                   "fit_a_share": v["fit_b1"]["a_share"]}
    for name, p in tmo["projections"].items():
        arms.setdefault("T1:" + name, {})["T1_large_call_mfu"] = p["T1_mfu"]
    for src in ("jn47u", "mden7"):
        for path in ("chunked", "fla"):
            arms[f"prefill:{src}:{path}"] = {str(T): T / s * f_tok / PEAK for T, s in PREFILL[src][path].items()}
    res["arms"] = arms
    return res


def costs(tmo: dict, vol: dict, dec: dict) -> dict:
    """$/1k rows at large calls (T1 regime: per-row overhead amortised by batching) per lever stack."""
    clef27 = tmo["projections"]["Clef 27B (fast fla GDN)"]["T1_usd_per_1k"]
    flash = tmo["projections"]["Clef-flash (fast fla GDN)"]["T1_usd_per_1k"]
    f_tok = rwkv_flops_per_token()
    toks = {"tree (exact fixes, subset)": vol["subset_common"]["tree_positions"]["mean"],
            "g1-v2 stock": vol["price"]["tokens_per_row"]["g1v2_stock"]}
    base = tmo["projections"]["RWKV7 stock g1-v2, fla WKV"]["per_token_us_T1"]  # 89.3 us: fla + 6% readout
    d16 = dec["16384"]
    gemm_lo, gemm_hi = d16["model"]["gemm_us_per_tok (65-75% cuBLAS)"]
    wkv_fla = sum(d16["fla_wkv_inmodel_us_per_tok"]) / 2
    ew = base / 1.06 - (gemm_lo + gemm_hi) / 2 - wkv_fla      # what is left at 16k = elementwise + norms + LoRA
    read = base - base / 1.06
    paper_wkv = PAPER_WKV["fwd_ms_no_state"] * L / (PAPER_WKV["B"] * PAPER_WKV["T"]) * 1e3  # us/token at B=8
    gm = (gemm_lo + gemm_hi) / 2
    stacks = {
        "today: fla, B=1 rate (T1 as published)": base,
        "+ compile_blocks (fuse elementwise, ew x0.4)": gm + wkv_fla + 0.4 * ew + read,
        "+ batched WKV (fla at B>=8 or BlinkDL CUDA, WKV -> 6 us)": gm + 6.0 + 0.4 * ew + read,
        "+ fused heads / readout overhead halved": gm + 6.0 + 0.4 * ew + read / 2,
        "best bf16 case (GEMM 75%, WKV = paper kernel, ew x0.3)": gemm_lo + paper_wkv + 0.3 * ew + read / 2,
        "+ fp8 GEMMs (1.6x on GEMM time; ONLY if Index-equivalence passes)": gemm_lo / 1.6 + paper_wkv + 0.3 * ew
        + read / 2,
    }
    out = {"clef27b_T1": clef27, "flash_T1": flash, "split_16k_us": {"gemm": gm, "wkv_fla": wkv_fla,
                                                                    "elementwise_norms_lora": ew, "readout": read,
                                                                    "paper_wkv_at_B8": paper_wkv},
           "stacks": {}}
    for name, us in stacks.items():
        mfu = f_tok / (us * 1e-6) / PEAK
        out["stacks"][name] = {"us_per_tok": us, "mfu_bf16_equiv": mfu,
                               **{f"usd_per_1k[{t}]": n * us * 1e-6 * USD_PER_ROW_S for t, n in toks.items()}}
    # MFU needed for a target $ (RWKV at T1-style overheads, all time = FLOPs / (eta * peak))
    need = {}
    for t, n in toks.items():
        for tgt, usd in (("2x Clef 27B", clef27 / 2), ("parity Clef 27B", clef27), ("parity Clef-flash", flash),
                         ("2x Clef-flash", flash / 2)):
            s_row = usd / USD_PER_ROW_S
            need[f"{t} | {tgt}"] = n * f_tok / (s_row * PEAK)
    out["mfu_needed"] = need
    out["equal_mfu_ratio_vs_clef27b"] = {t: n * f_tok / tmo["arms"]["clef27b_fast"]["flops_per_row_mean"]
                                         for t, n in toks.items()}
    out["equal_mfu_ratio_vs_flash"] = {t: n * f_tok / tmo["arms"]["flash_fast"]["flops_per_row_mean"]
                                       for t, n in toks.items()}
    return out


def launch_model() -> dict:
    """Per-call launch floor and the T where a call stops being launch-bound."""
    us_per_launch = {}
    n_chunked_short = L * launches_per_layer("chunked", 64)
    us_per_launch["from PERF_AUDIT 0.098 s/call (chunked, short call)"] = CALL_S_CHUNKED / n_chunked_short * 1e6
    us_per_launch["from D1 - parent (+2.2k LoRA launches)"] = (CALL_S_D1 - CALL_S_CHUNKED) / LORA_EXTRA_LAUNCHES * 1e6
    us_per_launch["mden7 training step (aten op incl. autograd)"] = 11.3
    per = {p: {str(T): L * launches_per_layer(p, T) for T in (64, 512, 2048, 16384)}
           for p in ("fla", "chunked", "chunked_lean")}
    fla_call_floor = L * launches_per_layer("fla", 64) * 11e-6
    return {"us_per_launch": us_per_launch, "launches_per_forward": per,
            "fla_call_floor_ms": fla_call_floor * 1e3,
            "fla_launch_bound_below_tokens": fla_call_floor / (PREFILL["jn47u"]["fla"][16384] / 16384),
            "d1_unmerged_extra_ms_per_call": LORA_EXTRA_LAUNCHES * 11e-3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json")
    ap.add_argument("--jn47u", help="job-jn47u out/throughput.json (re-read instead of the embedded values)")
    ap.add_argument("--mden7", help="job-mden7 out/throughput.json")
    a = ap.parse_args(argv)
    for key, path in (("jn47u", a.jn47u), ("mden7", a.mden7)):
        if path:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            for p in ("chunked", "fla"):
                PREFILL[key][p] = {int(T): d["prefill"][T][p]["s_median"] for T in d["prefill"]}
                WKV_FWD_MS[key][p] = {int(T): d["kernel"][T][p]["fwd_ms"] for T in d["kernel"]}
    tmo = tm.run()
    vol = json.loads((REPO / "analysis/tree_fork/volume.json").read_text(encoding="utf-8"))
    dec = decomposition("jn47u")
    out = {"mfu": mfu_table(tmo), "decomposition": dec, "launch": launch_model(), "costs": costs(tmo, vol, dec)}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")

    m = out["mfu"]
    print("RWKV FLOPs/token:", {k: round(v, 3) for k, v in m["rwkv_flops_per_token"].items()})
    for k, v in m["arms"].items():
        print(f"  {k:55s}", {kk: round(vv, 4) for kk, vv in v.items()})
    for T in ("2048", "8192", "16384"):
        r = dec[T]
        print(f"T={T}: fla {r['fla']['total_us_per_tok']:.1f} us/tok, chunked {r['chunked']['total_us_per_tok']:.1f};"
              f" nonWKV {r['nonwkv_us_per_tok'][0]:.1f}-{r['nonwkv_us_per_tok'][1]:.1f}; fla WKV "
              f"{r['fla_wkv_inmodel_us_per_tok'][0]:.1f}-{r['fla_wkv_inmodel_us_per_tok'][1]:.1f}; chunked WKV "
              f"{r['chunked_wkv_inmodel_us_per_tok'][0]:.1f}-{r['chunked_wkv_inmodel_us_per_tok'][1]:.1f} "
              f"(serial-loop {r['chunked_serial_loop_us_per_kernel'][0]:.1f}-"
              f"{r['chunked_serial_loop_us_per_kernel'][1]:.1f} us/kernel); model {r['model']}")
    print("bytes MB/token:", round(dec["bytes_MB_per_token_all_layers"], 1))
    print("launch:", json.dumps(out["launch"], indent=1))
    c = out["costs"]
    print("split@16k:", {k: round(v, 1) for k, v in c["split_16k_us"].items()})
    for k, v in c["stacks"].items():
        print(f"  {k:70s}", {kk: round(vv, 4) for kk, vv in v.items()})
    print("MFU needed:", {k: round(v, 3) for k, v in c["mfu_needed"].items()})
    print("equal-MFU cost ratio vs Clef 27B:", {k: round(v, 3) for k, v in c["equal_mfu_ratio_vs_clef27b"].items()},
          "vs flash:", {k: round(v, 3) for k, v in c["equal_mfu_ratio_vs_flash"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

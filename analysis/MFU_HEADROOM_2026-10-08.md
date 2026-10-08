# MFU headroom: where RWKV7-13.3B loses H100 throughput, and what is left to gain (2026-10-08)

**Takeaway.**
- **Achieved MFU today** (bf16 dense peak 989 TFLOP/s):

  | arm | batch-1 end to end | large calls (T1) | pure prefill (16k) |
  |---|---:|---:|---:|
  | RWKV7 chunked | 7.8% | 10% | 10.5% |
  | RWKV7 + fla | – | 29% | 31% |
  | Clef 27B fast | 34.7% | 38% | – |
  | Clef-flash fast | 23.0% | 30% | – |

- **Where an fla token's 83 µs goes:**
  - large GEMMs: ~35 µs, already ~70% efficient;
  - eager element-wise ops (token shift, six mixes, LoRA-gate activations, norms, bonus and residual): ~25–28 µs;
  - the WKV: ~19–23 µs, against 3.7 µs/token for the RWKV-7 paper's CUDA kernel at B = 8;
  - readout: ~5 µs.

  On the chunked path, ~74% of the time is a serial chunk loop: ~10 tiny kernels per 64 tokens per layer, 17–23 µs
  each. That loop is latency-bound, not FLOP-bound.
- **The ~47% target was computed for the g1-v2 template (1,830 tokens/row).** The tree-fork template the exact
  fixes use reads 2,482 tokens/row (subset; 2,677 on full DEV). At that volume, 2× Clef 27B needs **63% MFU**. That
  is above what is plausible for this stack.
- **Realistic best, bf16, with fused element-wise ops, a batched or CUDA WKV and a leaner readout:** ~47% MFU at large
  calls, giving **~$0.11 per 1k rows** for tree D1:
  - **1.5× cheaper than Clef 27B** ($0.167);
  - still **1.9× dearer than Clef-flash** ($0.061).

  The best bf16 case (54%) gives ~$0.097, or 1.7× Clef 27B. Only an fp8 path that has passed its gate reaches 2× (~$0.072).
- **Matched treatment** (Clef gets the same kernel work): the ratio falls back to the FLOPs ratio, 0.82× Clef 27B. Under
  Agent A's rule, 2× is out of reach without cutting tokens to ≤ 1,506 per row.

Reproduce with `python -I -X utf8 analysis/mfu_headroom.py [--json out.json] [--jn47u <jn47u out/throughput.json>]`.
It runs on CPU only and at $0. It reuses `analysis/throughput_maxbatch.py`, so it reads the same landed artifacts and
the same 2,000 common rows, plus `analysis/tree_fork/volume.json`. The jn47u and mden7 numbers are embedded; `--jn47u`
re-reads them from the receipt, and the results are identical.

## 1. Achieved MFU: formula and inputs

**MFU = model FLOPs per second ÷ 989.4e12.** Model FLOPs count the algorithm's minimal work and do not depend on which
kernel runs.

**RWKV7-13.3B.** 61 layers, d = 4096, 64 heads × 64, FFN 16,384. LoRA ranks w/a/v/g are 160/160/96/320 (vendored
`Rwkv7Config` defaults at d = 4096).

| term | per token | value |
|---|---|---:|
| matmul | 2 × 12.73e9 params (BackboneSpec)[^p] | 25.46 GFLOP |
| — of which the r/k/v/o and FFN GEMMs | 2 × 61 × (4d² + 2·d·FFN) | 24.56 GFLOP |
| — of which the time-mix low-rank chains (w, a, v, g) | 2 × 61 × 2d × 736 | 0.74 GFLOP (2.9%) |
| WKV recurrence, token-serial minimum | 61 × 64 heads × 8·64² | 0.128 GFLOP (0.5%) |
| *WKV chunk form, hardware FLOPs (not in MFU)* | 61 × 64 × (8·64² + 15·Cs·64) | 0.37 GFLOP (Cs = 64) / 0.19 GFLOP (Cs = 16) |
| **total model FLOPs** | | **25.59 GFLOP/token** |

[^p]: The structural count is 12.65e9. The 0.6% gap is immaterial.

**Clef** uses `throughput_maxbatch.clef_flops`, with the configs from `jobs/clef_baseline/price.py`:
- 2 × matmul params per token: 24.4e9 for 27B, 6.9e9 for flash;
- plus causal attention, 2·qd·n² over 16 or 8 full layers;
- plus GDN chunked delta rule, (6·dk·dv + 2·64·(dk + dv)) per value head per token over 48 or 24 layers.

Per mean row this is 77.1 TFLOP (27B) and 22.0 TFLOP (flash). Attention plus GDN is ~1.5% of it.

**Inputs.**
- Batch-1 end to end: Σ FLOPs over Σ kit `total_wall_ms` on the 2,000 common rows. Jobs: d2u54 (Clef 27B fast),
  sgihj (Clef-flash fast), zhyvg (RWKV g1-v2 chunked).
- T1: the large-call rate, Σms/ΣN over rows with ≥ 3k tokens, as defined in THROUGHPUT_MAXBATCH. For fla, it is the
  jn47u prefill rate × the readout's measured 1.06 long-row overhead.
- Pure prefill: jn47u and mden7 `throughput.json`, B = 1, median of 3. The two receipts agree within 1%.

| arm | tok/s | MFU |
|---|---:|---:|
| RWKV7 chunked, prefill 2k / 8k / 16k (jn47u) | 3,777 / 4,451 / 4,076 | 9.8 / 11.5 / 10.5% |
| RWKV7 fla, prefill 2k / 8k / 16k (jn47u) | 11,667 / 11,985 / 12,059 | 30.2 / 31.0 / 31.2% |
| RWKV7 fla in the readout (T1) | 11,197 | 29.0% |
| RWKV7 g1-v2 chunked readout: batch 1 / T1 | 3,010 / 3,717 | 7.8 / 9.6% |
| RWKV7 parent g1-v1, chunked_lean: batch 1 / T1 | 4,502 / 5,366 | 11.7 / 13.9% |
| D1 trained cell, batch 1 (before the fixes) | 1,619 | 4.2% |
| Clef 27B fast: batch 1 / T1 | 6,864 / 7,613 | 34.7 / 38.5% |
| Clef-flash fast: batch 1 / T1 | 15,958 / 20,926 | 23.0 / 30.2% |

**Per-row overhead at batch 1.** The fitted intercept `a` of `ms = a + bN + cN²` is:

| arm | `a` | share of wall |
|---|---:|---:|
| RWKV g1-v2 | 264 ms | 43% |
| Clef 27B | 95 ms | 42% |
| Clef-flash | 51 ms | 53% |

T1 assumes batching removes all of it, for every arm.

## 2. Where RWKV's time goes (per token, B = 1, from the jn47u receipts)

**Method.**
- The non-WKV work (Linears and element-wise ops) is identical on the fla and chunked paths.
- So the non-WKV time = fla prefill time − 61 × WKV-only time.
- The WKV-only time is the standalone receipt, scaled by k ∈ [0.8, 1.0]. The standalone timing also clones the 7
  inputs, and the chunked path's own in-model/standalone ratio is ~0.8.
- The non-WKV time is then split by an op-by-op count of the vendored block (`modeling_rwkv7.py`). GEMM time is taken at
  65–75% cuBLAS efficiency. Element-wise time is taken from bytes ÷ 2.8 TB/s achieved.
- The two independent estimates agree. The bytes model gives 24.8 µs of element-wise time. The residual after
  GEMM and WKV gives 27.5 µs.

| component (per token, 61 layers) | fla, 16k | chunked, 16k | evidence |
|---|---:|---:|---|
| r/k/v/o + FFN GEMMs: 24.56 GFLOP (24.8 µs at 100%) | 33–38 µs | same | model; cuBLAS bf16 at M ≥ 2k |
| element-wise, norms and LoRA chains: ~69.5 MB/token HBM traffic, ~0.74 GFLOP | ~25–28 µs | same | bytes model ≈ residual |
| — token shift (cat + delta, att + ffn) + 7 mixes (mul + add) | ~10 µs (36%) | | 360 KiB/layer/token |
| — LoRA w/a/g/v chains (skinny mm + tanh/sigmoid/add) + v-mix | ~7 µs | | 248 KiB |
| — kk, l2norm, k-mod, WKV prep | ~4 µs | | 168 KiB |
| — ln1, ln2, ln_x (GroupNorm) | ~1.3 µs | | 56 KiB |
| — bonus, gate, residuals, FFN relu² | ~6.5 µs | | 288 KiB |
| **WKV** | **19–23 µs** (fla `chunk_rwkv7`) | **181–186 µs** | receipts (decomposition) |
| — chunked: serial chunk loop, ~10 kernels per 64 tokens per layer, 163k launches/forward at 16k | | ~17–23 µs/kernel, ~74% of wall | per-kernel latency, flat in T |
| readout overhead (heads, gathers, packing) | ~5 µs (6%) | ~6% | v2 long rows ÷ probe rate |
| **total** | **83 (89 in the readout) µs** | **245 µs** | |
| **MFU** | **31% (29%)** | **10.5%** | |

**Kernel launches and Python overhead per call.** These are what dominate short calls.
- **Launches per block** (counted op by op): 71 non-WKV, plus:
  - fla: ~11, for **~5.0k launches per forward**;
  - chunked: 45 + 10 per 64-token chunk, for 7.7k at T ≤ 64 and 163k at 16k;
  - chunked_lean: 35 + 7 per chunk.
- **Cost per launch is ~10–13 µs** on this stack, from three independent fits:
  - PERF_AUDIT's 0.098 s/call ÷ 7.7k launches = 12.8 µs;
  - D1's 0.121 s/call, minus that, ÷ the 2.2k unmerged-LoRA launches = 10.5 µs;
  - mden7's 11.3 µs per aten op.
- **An fla call has a ~55 ms CPU floor.** It is launch-bound below **~660 tokens**. That covers most fork-tail calls
  and the median row (739 tree tokens, 3.3 calls/row) at batch 1.
- At T1 (≥ 48k tokens per forward), the floor is ~1% of the time. **Launch overhead is a latency problem, not a T1
  cost problem.**
- Unmerged LoRA in D1 adds ~24 ms per call and +2.9% FLOPs. `merge_lora` removes both.

**So, at large calls on fla, the 71% of peak that is lost goes to** (out of 89 µs, with 25.9 µs ideal):
- element-wise traffic: ~31%;
- WKV kernel: ~24%;
- GEMM inefficiency below peak: ~11%;
- readout: ~6%.

Batch size does not appear in this list. RWKV7 already saturates compute at one mean row (THROUGHPUT_MAXBATCH §1). The
exception is the WKV: both kernels parallelise over B × heads, and B = 1 gives only 64 heads. That is a hypothesis, and
the profile below tests it.

## 3. MFU levers (expected effect at large calls, tree-fork D1 tokens; evidence)

Base: fla at 89.3 µs/token, 29% MFU, $0.183 per 1k rows. The rows are cumulative.

| lever | expected | $/1k (tree) | MFU | evidence and caveats |
|---|---|---:|---:|---|
| **fla WKV** (already in base; needs the infer-equiv re-gate, job-exadi pending) | chunked → fla: 2.7–3.1× prefill | 0.455 → 0.183 | 10–14% → 29% | jn47u + mden7 receipts. Error at the bf16 floor (FLA_WKV_NUMERICS) |
| **`torch.compile` of the block** (the `compile_blocks` code exists; WKV excluded via `torch.compiler.disable`). Fuses token shift + 7 mixes + gate activations + l2norm + bonus + residual into a few kernels | element-wise ~27 → ~11 µs; launches ~5k → ~1.5k per forward | **0.149** | 36% | Training step, chunked-dominated: 6.11 → 5.63 s/step (1.085×, job-2amiy), gradient cos 0.998. Under fla the non-WKV share is ~3× larger, so the gain is larger. Dynamic shapes need warm-up (~70 s per specialisation, L-04/L-26). Execution-only, but reduction order changes, so it needs the same Index-level equivalence gate as fla |
| **Fused token shift + norms (hand Triton)** | subset of compile; fla ships `fused_addcmul_rwkv7`, `fused_k_rwkv7` and l2norm kernels | (inside the row above) | – | fla #1102 reported out-of-bounds reads in its channel-mix kernels: pin and audit before adopting any fla layer kernel beyond the WKV |
| **Batched / better WKV.** Run the WKV with B ≥ 8 rows (right-padded, `keep`-masked prefixes) instead of B = 1 varlen, or use BlinkDL's CUDA kernel | WKV ~21 → ~4–6 µs | **0.118** | 45% | RWKV-7 paper (§8): 7.9 ms fwd for B = 8 × 16k at H = 64, i.e. 3.7 µs/token over 61 layers, vs fla's 19–23 µs/token at B = 1. That kernel is token-serial per (B, head), so at B = 1 it would be *slower* than fla. The win needs B ≥ 8. fla's `chunk_h` is serial per (B, head) too, so it may gain from B too. **Unmeasured: the profile job tests B ∈ {1, 8}.** The CUDA kernel needs its own Index gate (bf16 vs fp32 kernel) |
| **Readout and heads** (fewer gathers and small calls; pack tails into ≥ 2k-token calls) | 6% → 3% | **0.113** | 47% | v2 long-row overhead 1.06 (THROUGHPUT_MAXBATCH) |
| **Larger prefill chunks** | ≤ 3% on fla (rate flat: 11.7k / 12.0k / 12.1k at 2k / 8k / 16k). Chunked: chunk 64 → 128 halves the serial-loop launches (~1.5–1.8× on the chunked WKV; fp32 limit is 146) | – | – | jn47u. Chunked is moot if fla is adopted. The real gain is avoiding small calls, which is the batching row |
| **bf16 vs fp32 paths** | chunked casts the 6 WKV inputs to fp32 and runs fp32 einsums. That is ~0.7 ms per layer at 16k against ~49 ms in the serial loop, so fp32 is **not** the cost. Keep the fp32 state (numerics). D1's per-call fp32→bf16 LoRA casts go away with `merge_lora` | +3–5% on D1 (merge) | – | rwkv7_chunked code; COST_LATENCY (merge_lora is unmeasured; the GPU A/B is spec'd) |
| **Remove Python per-row overhead** (memo_encode, done; overlap CPU encode with GPU; CUDA graphs on bucketed shapes) | batch 1: the intercept is 43% of v2 wall; the fla call floor is 55 ms → ~1 ms with graphs. T1: ~1–2% | batch 1 only | – | Fits above. Graph capture needs static `cu_seqlens` (the `_packed_bounds` `.tolist()` host sync breaks capture) and fixed-shape buckets |
| best bf16 case (GEMM at 75%, paper-class WKV, element-wise ×0.3) | 47.6 µs/token | **0.097** | 54% | upper end. Needs every row above |
| **fp8 GEMMs** (torchao or TE rowwise scaling; GEMM time ÷ ~1.6) | 35 µs/token | **0.072** | "74%" bf16-equivalent | **Only if it passes an Index-level equivalence gate:** the same pre-registered rule as fla infer-equiv (argmax agreement ≥ 99%, no flips with reference margin > 0.05, paired ΔIndex 95% CI ∋ 0 with \|bound\| ≤ 0.5, all finite). Clef can take fp8 too, so it does not count under matched treatment |

## 4. Is ~47% plausible, and what is the realistic best $/1k?

**What 2× requires.** All figures at large calls, RWKV at T1 overheads (`costs.mfu_needed`):

| template (tokens/row) | parity with Clef 27B | 2× Clef 27B | parity with Clef-flash |
|---|---:|---:|---:|
| g1-v2 stock (1,830) | 23% | **47%** | 64% |
| tree-fork D1, exact fixes (2,482 subset) | 32% | **63%** | 87% |

The "~47%" in the brief is the g1-v2 number. **For the trained cell on the tree template, 2× needs 63%.**

**Published comparators.**
- The RWKV-7 paper times only the WKV kernel: B = 8, 16k, H100 SXM, 7.9 ms forward, against 33.9 ms for FlashAttention-3.
  It reports no end-to-end prefill MFU. No source I found reports an H100 end-to-end RWKV-7 or linear-attention
  prefill MFU.
- The closest system-level data point is fla-based GDN-340M training on H100: 144k tok/s/GPU against 188k for softmax
  attention. That is ~23% slower, at a scale where the non-GEMM share is large.
- Not sourced, general experience: large-prefill dense transformers on mature engines (cuBLAS / FA3, fused norms and
  rope) typically land around 50–65% MFU on H100. Our own Clef fast path reaches 38% at T1.
- At d = 4096, RWKV7's GEMMs are 97% of its FLOPs. Once element-wise ops are fused and the WKV is batched, it should
  behave like a dense transformer of the same width.

**Verdict.**
- **~45–50% is plausible** with compile and a batched or CUDA WKV. Each step is unproven: each needs a measurement
  and an Index-equivalence pass.
- **55% is the optimistic ceiling** in bf16.
- **63% is not plausible** in bf16.

**Realistic best $/1k rows for the trained cell** (tree template, bf16, gated levers):

| case | MFU | $/1k rows | vs Clef 27B ($0.167) | vs Clef-flash ($0.061) |
|---|---:|---:|---|---|
| today | 29% | $0.183 | 1.1× more | 3.0× more |
| central | ~47% | **~$0.11** | **1.5× cheaper** | 1.9× more |
| optimistic | ~54% | $0.097 | 1.7× cheaper | 1.6× more |
| fp8 GEMMs, if gated | – | $0.072 | 2.3× cheaper | still 1.2× more |

Under matched treatment, where Clef also gets compile and fp8, the ratio converges to the FLOPs ratio:
- **0.82× Clef 27B**, or 1.2× cheaper;
- **2.9× Clef-flash**.

**2× vs Clef 27B under that rule needs ≤ 1,506 tokens/row**, which is a template change, not kernel work. **No kernel
lever makes RWKV7-13.3B cheaper than Clef-flash at matched treatment.**

## 5. GPU profile job spec (≤ $5; spec only, NOT packaged or submitted)

**Question.** Confirm the §2 decomposition, and measure the two untested levers: compiled block, and WKV at B > 1.

**Box and image.**
- 1× H100 80GB SXM, `clock_lock: true`.
- Image: jobs/fla_rwkv7, with fla-core 0.5.2 pinned by hash and torch 2.13.
- Stock RWKV7-G1k-13.3B snapshot, sha-verified.
- `GPU_VERIFIED` stays False. The job uses `allow_unverified=True`, scoped to the profile, as the fla gate does.

**Rows.** 20 DEV rows from the 250 pre-declared smoke rows (`fulldev_smoke_rows.json`, seed `fulldev-smoke-v1`),
taken in run_id order within strata:
- 8 rows under 1k RWKV tokens;
- 8 rows of 1–4k;
- 4 rows of 4k or more.

Template g1-v2 stock readout, batch 1, `inference` default.

**Arms**, in one process. Each arm gets 2 untimed warm-up rows, then 20 profiled rows.
1. `chunked` (production reference).
2. `fla` (`select_wkv(m, "fla", allow_unverified=True)`).
3. `fla + compiled block`: inductor, `dynamic=True`, WKV wrapped in `torch.compiler.disable`, as in
   `heads/act_ckpt._compiled`. Add an inference variant that needs no `act_ckpt`.

**Synthetic ladder** (no readout): prefill at T ∈ {2k, 8k}, B ∈ {1, 8} (right-padded with `keep`), for fla and fla +
compile. Add WKV-only at B = 8. This tests the batched-WKV hypothesis.

**Profiler.**
- `torch.profiler` with CPU and CUDA activities, `record_shapes=True`; a schedule of 1 wait, 1 warm-up and 3 active
  steps per synthetic cell; per-row `record_function` labels.
- Chrome traces are exported only for 1 short row and 1 long row per arm, kept under 200 MB.
- Kernels are bucketed by name and shape into GEMM (`sm90_xmma` / `cutlass` / `gemm`, with N or K ≥ 4096), skinny
  LoRA mm (one dim ≤ 320), WKV (`chunk_dplr*` / `fla` / the chunked einsum + `trsm` + `bmm` inside the WKV function
  range), norms (`layer_norm` / `group_norm` / `norm`), and element-wise (the rest).
- Recorded per bucket and arm: GPU µs per token, launches per forward (`cudaLaunchKernel` count), and CPU dispatch µs
  per launch. GPU idle gaps are also recorded, as the launch-bound share.

**Equality (non-gating, reported).** Per-question argmax agreement and max |Δp| of fla and fla + compile against
chunked on the 20 rows.

**Pre-declared confirmation rule.** The decomposition is confirmed if all of the following hold at T = 8k, B = 1:
- each bucket's share is within ±10 pp of §2: GEMM ~42%, element-wise + norms + LoRA ~32%, WKV ~25% on fla;
  WKV ≥ 65% on chunked;
- launches per forward are within ±25% of 5.0k (fla);
- CPU µs per launch is within 8–15 µs.

If it is not confirmed, §3's $/1k column is re-derived from the measured buckets.

**Outputs.** `profile.json` (buckets per arm, row, synthetic cell), `launches.json`, `equality.json`, `env.json`,
`traces/` (4 trimmed traces), and `nvidia-smi.txt`.

**Cost.**

| item | minutes |
|---|---:|
| setup and snapshot (jn47u-class) | ~6 |
| fla Triton warm-up | ~6 |
| inductor compile, dynamic | ~4–8 |
| 3 arms × 22 rows, profiled at ~3× overhead | ~3 |
| synthetic ladder, 2 T × 2 B × 2 paths × 5 steps | ~3 |
| export | ~1 |
| **expected total** | **~25–30 min ≈ $1.25–1.50** |

`max_duration` is 45 min and the build is 10 min. Worst case = (10 + 2 attempts × 45) × $0.0495 = **$4.95**.

**Decision use.** Adopt compile for inference when it is ≥ 1.15× on the long stratum and passes the Index gate in the
exadi-style job. Pursue batched WKV only if B = 8 cuts the WKV µs/token by ≥ 2×.

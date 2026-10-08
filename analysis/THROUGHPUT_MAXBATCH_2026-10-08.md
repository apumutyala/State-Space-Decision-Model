# Max-throughput serving on one H100 80GB: RWKV7-13.3B vs Clef 27B and Clef-flash (2026-10-08)

**Takeaway.** At max batch with today's kernels, **Clef wins**. Clef 27B costs about **$0.17 per 1k DEV rows**, and
Clef-flash about **$0.06**. RWKV's cost depends on its readout template and WKV kernel:

| RWKV arm at max batch | $/1k rows | vs Clef 27B | vs Clef-flash |
|---|---:|---:|---:|
| Trained cell D1 + exact fixes | $0.46 | 2.7x more | 7.5x more |
| Same + fla WKV | $0.22 | 1.3x more | 3.6x more |
| Stock g1-v2 template + fla WKV | $0.135 | **1.24x cheaper** | 2.2x more |

Batching does not change the structure of the comparison. The Index workload is prefill only: one read per request, no
generation, and Clef runs with `use_cache=False`. A batch-1 call is already past the H100's compute/bandwidth ridge,
so the race is FLOPs per row times achieved MFU. RWKV7's O(1) state gives it no advantage here:
- Memory never binds. The rows that fit in 80 GB (29 to 210 per forward) are far more than it takes to saturate
  compute (one mean-length row).
- Clef's own per-sequence GDN state is larger: 147 MiB, against 62 MiB for RWKV.

**What 2x would take:**
- **Against Clef 27B:** plausible only if three things all happen:
  - an accepted fla-class WKV kernel;
  - a single-token-code readout (g1-v2-style) for the trained cell;
  - either RWKV reaching ≥ ~47% MFU while Clef stays at 38%, or, with Clef given the same kernel treatment (Agent A's
    rule), RWKV reading ≤ ~1,500 tokens per row. Today's best template reads 1,830.
- **Against Clef-flash:** not plausible. It would need ≤ ~430 RWKV tokens per row, and the mean prompt alone is
  ~1,500 tokens.

Reproduce with `python -I -X utf8 analysis/throughput_maxbatch.py [--json out.json]`. It runs on CPU only and
extracts nothing. It reads the same landed artifacts as `analysis/cost_latency.py`, and the committed
`jobs/common/readout_volume_rwkv7.json`.

## 1. Memory and compute model (configs → bytes, FLOPs)

The Clef configs are `jobs/clef_baseline/price.py` CONFIGS, read from config.json at the pinned revisions. Their layer
counts match `baselines/clef_backend.py` RELEASES:
- Clef 27B: 48 GDN layers + 16 gated full-attention layers.
- Clef-flash: 24 + 8.

RWKV7-G1k-13.3B comes from `heads/backbone.py` and `jobs/rwkv7/JOB_SPEC.md`: 61 layers, d 4096, 64 heads × 64,
FFN 16384, fp32 WKV state. All per-row numbers are on the **2,000 common ok rows** of the DEV subset, using each
arm's own model tokens.

| | Clef 27B | Clef-flash | RWKV7 (g1-v2 readout) | RWKV7 (g1-v1 = D1's template) |
|---|---:|---:|---:|---:|
| weights (bf16) | 51.0 GiB | 17.5 GiB | 24.7 GiB | 24.7 GiB |
| matmul params per token | 24.4 B | 6.9 B | 12.73 B | 12.73 B |
| tokens per row: mean / median / p95 | 1,543 / 591 / 7,036 | same tokenizer | 1,830 / 748 / 7,841 | 2,958 / 783 / 15,992 |
| **FLOPs per row (mean)** | **77.1 T** | **22.0 T** | **46.8 T** | **75.7 T** |
| attention + linear-attention share of FLOPs | ~1.5% (2N² over 16 layers; ~6% at 16k) | similar | ~0.4% (WKV) | ~0.4% |
| per-sequence state the Index path keeps | none (`use_cache=False`; layer transients only) | none | prefix state 62 MiB, plus 1 copy per packed tail row | same |
| a cache-resident engine (vLLM-style) would keep | GDN 147 MiB + KV 64 KiB/token = 243 MiB at mean | GDN ~49 MiB + 32 KiB/token | 62 MiB, length-independent | 62 MiB |
| transient activations per token, no grad | 258 KiB (stress 454) | 182 KiB (measured, sgihj) | 566 KiB (chunked WKV, mhc3h 36 GB peak at 16k) | 566 KiB |
| free for activations (79.6 GiB − 3 GiB reserve − weights) | 25.6 GiB | 59.1 GiB | 51.9 GiB | 51.9 GiB |
| **max rows per forward at the DEV mean** (5% padding) | **64** (stress 36) | **210** | **47** | **29** |
| fork state copies (23 options per row mean, p95 129) | – | – | 1.4 GiB mean, 7.8 GiB p95 (capped by `state_budget`) | same |

The FLOPs ratio of RWKV to Clef 27B is set by the readout template:
- The g1-v1 fork (D1's template) re-reads each question suffix once per option. Its tails are **40% of its tokens**:
  2,958 tokens per row against 1,762 for the longest single-question row. That brings RWKV to **FLOPs parity with
  Clef 27B (75.7 T vs 77.1 T)**, even though the model is half the size.
- The g1-v2 single-token-code template reads 1,830 tokens per row: 0.61x Clef 27B's FLOPs.

Roofline. The H100 ridge is 989 TFLOP/s ÷ 3.35 TB/s = 295 FLOP/byte. A prefill forward of T tokens does 2·P·T FLOPs
on 2·P weight bytes, so it is compute-bound once T > ~295. The median row is 591 Clef tokens or 748 RWKV tokens, so
**batch 1 is already compute-bound** for every arm. Only RWKV's short packed tail calls and tiny rows sit near the
weight-streaming floor:

| | weight-streaming floor per call |
|---|---:|
| Clef 27B | 16.3 ms |
| Clef-flash | 5.6 ms |
| RWKV7 | 7.9 ms |

Batching buys two things:
1. It amortises per-row and per-call overhead: launches, Python, the head, and these floors.
2. It moves short rows up to the large-call rate.

Batching does not raise the large-call rate itself. That rate is set by the kernel.

## 2. Calibration on the measured batch-1 runs (1× H100, clock_lock, same 2,000 rows)

| arm (job) | ms per row | tok/s | achieved MFU | large-call rate (rows ≥ 3k tokens) | MFU at that rate |
|---|---:|---:|---:|---:|---:|
| Clef 27B fast (d2u54) | 225 | 6,864 | 34.7% | 7,613 tok/s | 38% |
| Clef-flash fast (sgihj) | 97 | 15,958 | 23.0% | 20,926 tok/s | 30% |
| RWKV7 stock g1-v2, chunked (zhyvg) | 608 | 3,010 | 7.8% | 3,717 tok/s (probe: 3,950) | 10% |
| RWKV7 parent g1-v1, chunked_lean (c7pq2) | 657 | 4,502 | 11.7% | 5,366 tok/s | 14% |
| D1 trained cell (c7pq2) | 1,827 | 1,619 | 4.2% | – (per-question calls; the fixes remove them) | – |
| RWKV7 fla WKV (jn47u prefill probe, 2k–16k) | – | 11,900 | – | 11,197 tok/s after the readout's 6% overhead | 29% |

How the large-call rate is defined. It is Σms/ΣN over rows with N ≥ 3,000 tokens, where per-row overhead is a small
share of the time. It is what batching can lift every token to without kernel work. For Clef it still includes the
joint head's per-question work, so it is mildly conservative for Clef.

Clef's long rows are close to its short-row-free rate. Rows of 8–16k tokens cost 137 µs per token, against 127 µs at
3–8k, so attention's quadratic term adds ~8% at the top of the mix. RWKV is flat. **Linear time is worth ≤ ~5% on
this length mix.**

## 3. Projected throughput at max batch

Three regimes:
- **T0:** batch 1, as measured.
- **T1:** max batch on today's kernels. Overhead is amortised and every token runs at the arm's large-call rate.
  **This is the central estimate.**
- **T2:** compute roof at MFU η. It needs kernel or graph work on both sides.

The rate is $0.0495 per GPU-min.

| arm | T0 $/1k | **T1 $/1k (rows/s)** | T2 η=0.40 / 0.55 / 0.70 $/1k |
|---|---:|---:|---|
| Clef 27B (fast fla GDN) | 0.185 | **0.167 (4.9 rows/s)** | 0.161 / 0.117 / 0.092 |
| Clef-flash (fast fla GDN) | 0.080 | **0.061 (13.6 rows/s)** | 0.046 / 0.033 / 0.026 |
| RWKV7 stock g1-v2, chunked WKV | 0.502 | **0.406 (2.0)** | 0.098 / 0.071 / 0.056 |
| RWKV7 stock g1-v2, fla WKV | 0.352 (proj.) | **0.135 (6.1)** | 0.098 / 0.071 / 0.056 |
| RWKV7 D1 + exact fixes (memo + packed_fork), chunked_lean | 0.502 (proj.) | **0.455 (1.8)** | 0.158 / 0.115 / 0.090 |
| RWKV7 D1 + exact fixes + fla WKV | 0.266 (proj.) | **0.218 (3.8)** | 0.158 / 0.115 / 0.090 |

**Cost ratio, RWKV ÷ Clef, under matched treatment** (<1 means RWKV is cheaper):

| RWKV arm | vs Clef 27B: T0 / **T1** / T2 equal-MFU | vs Clef-flash: T0 / **T1** / T2 equal-MFU |
|---|---|---|
| stock g1-v2 chunked | 2.70 / **2.43** / 0.61 | 6.29 / **6.68** / 2.13 |
| stock g1-v2 + fla | 1.90 / **0.81** / 0.61 | 4.42 / **2.22** / 2.13 |
| D1 + fixes, chunked_lean | 2.71 / **2.72** / 0.98 | 6.30 / **7.48** / 3.44 |
| D1 + fixes + fla | 1.43 / **1.30** / 0.98 | 3.33 / **3.58** / 3.44 |

### Uncertainty, by regime

- **Compute-bound (T1 → T2).** This is where max batch lands for every arm. The T1 MFUs are measured as large-call
  rates: Clef 38% / 30%, RWKV-fla 29%, chunked 10–14%. The T2 η is an assumption.
  - With the chunked kernels, RWKV stays ~10–14% MFU however big the batch. The probe rate was flat from 2k to 16k
    tokens per call, so the kernel, not the batch, is the limit. T2 is out of reach for the chunked rows without a new
    WKV kernel.
  - With fla, RWKV sits at ~29% at large calls. Its 61 layers have many element-wise ops and small LoRA matmuls (w, a,
    v, g) per token, so reaching 55% needs fused kernels or CUDA graphs. That is unproven, and Clef would get the same
    work.
  - The asymmetric bounds show the spread. Clef held at T1 against RWKV-fla at η 0.70 gives 0.33x (v2) or 0.54x (D1).
    RWKV-fla held at T1 against Clef at η 0.55 gives 1.15x or 1.86x. Neither counts under Agent A's rule.
- **Memory-bound.** Only the per-call floors (5.6–16.3 ms) and RWKV's short tail calls are here. Batching removes
  them, and that is already in T1.
- **Memory capacity.** This does not bind. Even the stress activation figure for Clef 27B (454 KiB per token) fits
  36 mean rows, about 59k tokens per forward. The activation figures are scaled from two peaks:
  - sgihj: Clef-flash at 20.34 GiB;
  - mhc3h: RWKV chunked at 36 GB.

  These are not per-batch measurements. The GPU spec below records peak memory per batch.
- **Not in the model:**
  - CPU-side encode and tokenisation. Clef's 95 ms intercept includes it, and so does RWKV's per-request memo.
    Batching hides it only if it is overlapped with the GPU. The measured T1 will tell.
  - Clef's joint head loops over records in Python.
  - fp8 weights. These would cut both sides' FLOP time roughly equally, so the ratios would barely move.

## 4. Is ≥ 2x plausible at matched max-throughput serving?

**Against Clef 27B: yes, but only with all of the following.** Today it is no: 2.7x worse for D1 + fixes, 1.3x worse
with fla.
1. **fla WKV accepted for inference.** That is the Index-level re-gate in `FLA_WKV_NUMERICS_2026-10-08.md`. On its
   own it gives 2.1x on D1 at T1 ($0.455 → $0.218).
2. **A single-token-code readout for the trained cell**, such as `jrt-rwkv-g1-v2`, or a two-level fork that reads each
   question suffix once rather than once per option. Either removes most of g1-v1's 40% tail tokens: 2,958 → ~1,830
   tokens per row, a further 1.6x. With fla, this is the stock-v2 row: **0.81x Clef 27B at T1** (1.24x cheaper).
3. **Then one of these:**
   - (a) RWKV gets to ≥ ~47% end-to-end MFU at max batch while Clef stays on its production path at 38%. This needs
     fused element-wise kernels or CUDA graphs for RWKV, and it is a treatment Clef could also receive.
   - (b) Under Agent A's rule (equal MFU), RWKV must read ≤ **1,506 tokens per row**. That is 18% fewer than v2. One
     route is to pack all questions into one sequence, as Clef does, instead of per-question fork tails. The World
     tokenizer's ~1.19x token inflation versus Clef's tokenizer on this mix limits how far this can go.

   At T1 rates without that MFU work, RWKV needs ≤ 1,135 tokens per row.

**Against Clef-flash: no.** At equal MFU, 2x needs ≤ ~430 RWKV tokens per row, which is less than the median prompt.
At T1 it needs ≤ 413. Even the best RWKV configuration is 2.1–2.2x *more* expensive than Clef-flash. Matching
Clef-flash on cost needs a smaller attention-free model (~3–4B), not serving work.

**Index per dollar** also counts against RWKV: D1 scores 35.1, against 55.8 for Clef-flash and 62.1 for Clef 27B.

## 5. GPU measurement spec (≤ $6; spec only, NOT packaged or submitted)

**Question.** Measure throughput at batch {1, 8, 32, max} for both families, on the same rows and the same box, with
decisions equal to batch 1. This replaces T1 with a measurement.

**Rows.** The 250 pre-declared smoke rows: `jobs/m2_first_cell/fulldev_smoke_rows.json` on `stack/rwkv7/fulldev-eval`,
seed `fulldev-smoke-v1`, ids_sha256 `1a5ca017…`.
- 12 of them exceed 16,384 RWKV tokens. Clef refuses rows over 16,384 Clef tokens as `exhausted`.
- Report on all 250 rows, and also on the rows that both arms scored.

**Box.** 1× H100 80GB SXM with `clock_lock: true`. One job runs the arms in sequence, so box variance cancels.

**Arms, in order:**
1. **Clef 27B on the production fast path.** fla GDN as in d2u54, `gdn_path: fast`, with the kernel-path guard
   recorded. Batching uses Clef's own `collate_records` (right-padded; causal, so the scored positions are unaffected)
   through `ClefJointHeadBackend` raw head output on a list of records.
2. **Clef-flash**, same path. It is cheap and gives the second comparator.
3. **RWKV7 D1 + `inference="fast"`** (memo_encode + packed_fork, chunked_lean, LoRA unmerged so outputs are exact).
4. **Optional:** the same arm with fla WKV, only if the fla inference re-gate has passed by then. Otherwise skip it.

**Batching (new code for RWKV only; CPU-test it first).**
- Rows are length-sorted into micro-batches of B rows. "max" is the largest B under a padded-token budget per forward:
  Clef 27B 48k, Clef-flash 128k, RWKV 48k. Those budgets come from section 1's capacity at the stress activation
  figure, × 0.8.
- **RWKV prefixes must be varlen-packed** (`cu_seq_lens`, already supported by `modeling_rwkv7` and fla), not
  right-padded. Padding after a prefix would corrupt the recurrent state that the fork reads.
- Tails of several requests share the existing packed tail calls.
- On an OOM, halve B once and record it.

**Measured per (arm, B).**
- Pass wall time excluding load, rows/s, model tok/s.
- `max_memory_allocated`.
- 1 Hz `nvidia-smi` utilisation and power.
- Per-row latency (the batch's wall time, attributed to its rows).
- Warm-up: the first micro-batch of each B is untimed (fla autotune, ~70 s per specialisation on RWKV fla).

**Equality gate, pre-declared.**
- For each (arm, B > 1) against the same arm at B = 1, on every row both scored:
  - per-question argmax agreement ≥ 99.5%;
  - every flip has a batch-1 top-2 margin < 0.01;
  - max |Δp| ≤ 0.02 (Clef's accepted kernel-path bound was 0.0126);
  - identical statuses.
- A failing (arm, B) has its throughput reported but marked `not_equal`. It does not count.

**Outputs.** `throughput.json` per (arm, B), `equality.json`, `environment.json`, plus the per-row decisions for the
B = 1 and B = max passes.

**Cost** (`smoke_slice` in the script's JSON):

| item | GPU minutes |
|---|---:|
| Clef 27B, four passes | 4.3 expected, 5.0 worst (batch 1: 1.25) |
| Clef-flash, four passes | 1.6–2.0 |
| RWKV D1-fast, four passes | 12.5–13.5 (batch 1: 3.4; 0.96 M tokens) |
| loads (Clef 27B ~1.6, flash ~0.6, RWKV + D1 checkpoint ~1.5) plus setup, about 3 (sgihj: 2.88) | ~7 |
| **total** | **~26–28 GPU-min ≈ $1.3–1.4** |

- The optional fla arm adds ~5 min plus ~10 min of compile: about $0.75.
- A worst case of 2x on every item is ~$3.
- **Hard cap: 110 min = $5.45.** The live rate guard (`jobs/common/rate_guard.py`) stops a slow box.

**Verdict rule (pre-declared).**
- Per arm, T1_measured = rows/s at the best B that passes equality.
- The RWKV ÷ Clef cost ratio at matched max batch = Clef rows/s ÷ RWKV rows/s, on the rows both arms scored.
- The projection in section 3 is confirmed if the measured ratio is within ±25% of it. Projected T1 ratios: D1-fast
  vs Clef 27B 2.72, vs Clef-flash 7.48.

## Assumptions and what the measurement replaces

- **D1 + fixes tokens.** These use the g1-v1 prefix-fork positions from `readout_volume_rwkv7.json`. D1's head reads
  the same prefix and tails. Its rate comes from the parent arm, which uses the same template, WKV and call structure.
  The batch-1 cost is the memo's D1 ÷ 3.
- **fla arms.** These use the jn47u prefill rate (11.9k tok/s) with the readout's measured 6% long-row overhead. They
  are not an end-to-end run.
- **Activation bytes per token.** These are scaled from single peaks, not measured per batch.
- **η.** This is an assumption, used only in T2.

# jobs/cell_step_profile: the m2 first cell's train step, four execution variants (DRAFT, NOT SUBMITTED)

**Question.** Job-mden7 (SF, H100 80 GB) ran the m2 first cell's real train step (`train.trainer.Trainer.train_step`
with `train.run_cell.train_config`: RWKV7-G1k-13.3B bf16, LoRA r64, cache-fork head, Clef loss, batch 1 x accumulation
8, no checkpointing, WKV `chunked`) on the cell's first 23 optimiser steps (`jobs/fla_rwkv7/train_mix_v2.json`).
The production path OOMed on 17 of 20 timed steps (peak 84.0 GB; weights 26.0 GB), and each completed step took
15–17 s for 3.5k–4.8k tokens: ~2 s of fixed cost per row, flat in tokens. The cell's ladder would end at T <= 512 or
exit 22, and its 190 minutes would hold ~700 steps instead of the planned ~1,600. Which execution variant of the SAME
step (same rows, same loss, same gradients) fits in 76 GB on every row, and how fast is it?

## Decision rule (PRE-REGISTERED; `jobs/cell_step_profile/rule.py`, pinned by `tests/test_cell_step_rule.py`)

Committed alone, before the harness and before any run (9817029); amended before any run only: A1 (f5b7518) and A2
(below). Not to be edited after GPU numbers exist. Current rule id: `cell_step_profile v1-A2`.

**Variants** (each in its own process, in this order: A2 puts the fastest expected first and the memory-only fallback
last):

| variant | `act_ckpt` | `row_batch` | `row_tokens` | `rwkv7_wkv` |
|---|---|---|---|---|
| `baseline` | false | 1 | 0 | `chunked` (the vendored checkpoint path) |
| `ckpt_rows_lean` | true | 8 | 8192 | `chunked_lean` |
| `ckpt_rows` | true | 8 | 8192 | `chunked` |
| `ckpt` | true | 1 | 0 | `chunked` |

`act_ckpt` = per-block activation checkpointing that keeps the cache fork. `row_batch` 8 / `row_tokens` 8192 = the 8 rows
of a step sorted by length (longest first) and grouped greedily into calls of at most 8 rows and at most 8,192 padded
tokens (rows x the group's longest row; a longer row runs alone); each group is one prefix call + one continuation call,
right-padded with an attention mask. `chunked_lean` = the same chunk-parallel WKV algebra with fewer launches. Every
variant runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recorded per variant).

**Measurements per variant.** First (A2) the probe: the forward loss (no grad, eval mode, the same seeds, the variant's
own execution path) on the 8 rows of mix step 22 (max_T 772, which the baseline fits). Then the 23 steps of
`train_mix_v2.json` (3 warmup + 20 timed, 8 rows each, in the cell's epoch-0 order). Per step: wall seconds
(CUDA-synchronised), tok/s = prefix tokens / seconds, peak GB (`torch.cuda.max_memory_allocated`, reset before every
step), OOM, loss. An OOM step is recorded and the run continues; a step that cannot finish before the phase deadline is
recorded as skipped. Then the K = 10 ladder: synthetic rows of T = 1024 / 1536 / 2048 / 3072 / 4096 tokens with 10
continuations of 16 tokens, one optimiser step over `row_batch` copies of the row (1 for `baseline` / `ckpt`; 8 for the
row variants, grouped exactly as a real step: the unit that sets the peak), ascending, stopping at the first OOM or
peak > 76 GB.

**A variant qualifies iff** all of (A2):
1. zero OOM on every step it attempted, and a finite loss on every step it completed;
2. every step's peak <= 76.0 GB (the cell's `pass_rule.max_peak_gb`);
3. at least 15 of the 20 timed steps completed. Deadline-skipped steps (and steps a killed process never recorded) are
   reported separately and never disqualify on their own beyond this floor;
4. its flags and allocator setting are the pre-registered ones;
5. GPU loss clause: |probe loss - baseline probe loss| / |baseline probe loss| <= 5e-3. If the baseline's probe is
   missing or not finite, the clause is INCONCLUSIVE (not a pass) and so is the verdict.

The CPU equivalence tests (`tests/test_cell_perf.py`, `tests/test_cell_step_rule.py`) stay an additional setup gate: the
runner runs them on the image before any GPU phase and exits 5 if they fail.

**Verdict.**
- `ADOPT`: the qualifying variant with the highest median tok/s over the timed steps completed by ALL qualifying
  variants (matched rows; if no step is common to all, each variant's own completed timed steps); ties: lower peak,
  then earlier variant. Recommended cell config = its flags + `t_ladder` = the cell's [512, 1024, 2048] cut at its
  ladder T_fit (the full ladder if T_fit >= 2048), and the projection: steps in 190 min = floor(190 x 60 / matched
  median step seconds), rows = 8 x steps.
- `CAP_T`: no variant qualifies. The cell must cap T at the largest ladder T that fits among the variants whose loss
  clause holds (ties: higher median tok/s); recommended config = that variant's flags, `t_ladder` cut at it, `t_cap`.
- `NO_FIT`: nothing fits even T = 1024.
- `INCONCLUSIVE`: the baseline's probe loss is missing or not finite.

### Amendment A1 (2026-10-06, before any run; rule `cell_step_profile v1-A1`)

The pre-registration commit had `ckpt_rows8` / `ckpt_rows8_lean`: all 8 rows of a step in ONE call, padded to the
step's longest row. Before any GPU run, a CPU measurement of the saved-for-backward bytes of one real-width layer
(hidden 4096, 64 x 64 heads, FFN 16384, LoRA r64, bf16 autocast, cache-fork row at T = 1024) gave ~1.0 MB per token per
layer without checkpointing and ~22 kB per token per layer with it. One call of 8 rows padded to ~2,025 tokens (steps
0, 1, 6 of the mix) is 16.2k tokens: ~22 GB of checkpoint store + ~17 GB to recompute one layer + backward transients
+ 26 GB weights + 4.6 GB LoRA / grads / Adam = ~74 GB, at the 76 GB line. The row variants are therefore renamed
`ckpt_rows` / `ckpt_rows_lean` and carry `row_tokens` = 8192 (~53 GB by the same arithmetic; 2–3 calls on the long
steps, 1 on short ones). Nothing else in the rule changed.

### Amendment A2 (2026-10-06, after independent review, before any run; rule `cell_step_profile v1-A2`)

Independent review of PR #86: FIX-THEN-LAUNCH (no correctness issues; lean WKV and row batching independently
verified). Coordinator decisions, quoted:

> F1 (deadline can disqualify the fastest variant): mden7's forward-only chunked prefill is 227–265 µs/token, so
> fwd+recompute+bwd is plausibly 0.7–0.9 ms per PADDED token; the real mix pads 200,731 tokens for 124,586 real.
> Re-project every variant from that GPU basis (not CPU op counts), and set caps at ≥1.5x that projection.
> Qualification becomes: zero OOM on every step attempted, every step's peak ≤ 76 GB, ≥ 15 of 20 timed steps
> completed, and the GPU loss clause below. Deadline-skipped steps are reported separately, never disqualifying on
> their own beyond the ≥15 floor. Rank by median tok/s over the timed steps completed by ALL qualifying variants
> (matched rows). Run order: baseline, ckpt_rows_lean, ckpt_rows, ckpt (memory-only fallback last). A worst case above
> $7.23 is fine (no strict cap; state the new figure); build_timeout stays 60.
>
> F2 (GPU loss clause never runs): before training in EACH variant process, compute the forward loss (no grad, eval
> mode, same seed) on a fixed probe the baseline fits: the rows of mix step 22 (max_T 772). Binding:
> |loss_variant − loss_baseline| / |loss_baseline| ≤ 5e-3 (bf16 eps 7.8e-3; the tiny CPU bf16 probe already showed
> 7.4e-4, and 61 layers compound, so 1e-3 would be a false-reject risk; 5e-3 is fixed now, not tuned). If the
> baseline probe itself fails → INCONCLUSIVE for that clause (not a pass). Keep the CPU equivalence tests as an
> additional setup gate.

Implemented in this rule: the run order above; qualification clauses 1–5; matched-row ranking; the probe constants
(`PROBE_STEP` 22, `LOSS_REL_MAX` 5e-3, `MIN_TIMED_COMPLETED` 15); the `INCONCLUSIVE` verdict. The step-0 loss clause of
v1 / A1 (1e-3, CPU tests when the baseline's step 0 OOMs) is replaced. Caps and the cost are in "Phases, caps, cost".

The verdict is a recommendation for the m2 cell (PR #74 / its follow-up), which keeps its own ladder stage and
stop rules (exits 22 / 24 / 26) as the authority on the real rows.

## Diagnosis by code reading (agent-main 0f4b8ad line numbers), with CPU measurements

All counts below are aten ops (forward + backward + optimiser) of one real optimiser step of the mix, counted with a
`TorchDispatchMode` on a 61-layer RWKV-7 of tiny width (op counts do not depend on width; chunk count follows T).

1. **Per-row fixed cost = kernel launches, not FLOPs.** Every row is two full 61-layer calls (prefix, then its K
   continuations; `heads/features.py:116-125` loops the rows, `heads/backbone.py:328-345`), forward and backward, and
   a step is 8 such rows (`train/trainer.py:189-197`, batch 1 x accumulation 8). The baseline step issues 1.37–1.83 M
   aten ops. Job-mden7's completed baseline steps 3 / 5 / 8 took 16.91 / 16.82 / 15.45 s for 1.477 / 1.501 / 1.365 M
   ops: **11.2–11.5 µs per op**, the same on every step, so the step is CPU dispatch / launch bound (GPU FLOPs for
   ~4.5k tokens x 3 x 2 x 13.3e9 = 0.36 PFLOP are < 1 s on an H100).
2. **The K fork continuations are NOT sequential.** `_state_fork` (`heads/backbone.py:340-344`) already copies the
   prefix state K times (`batch_repeat_interleave`) and runs all K continuations in one call. What is sequential is the
   ROWS: one prefix call + one continuation call per row. Fix (b) therefore batches rows, not options.
3. **The chunked WKV's Python loop dominates the launch count.** `third_party/rwkv7_hf/modeling_rwkv7.py:279-286` runs
   ~11 kernels per 64-token chunk per layer forward, and indexes 8 tensors with `x[:, i]` per chunk
   (`qr`, `qkv`, `lhs`, `rkv`, `rb`, `kv`, `b_t`, `c_last`): in backward each becomes a FULL-size zero-filled
   `select_backward` plus an add, per chunk. One WKV call at T = 2048 (fwd + bwd): **4,805 aten ops vendored, 1,454
   with `chunked_lean`** (no per-chunk selects; one batched score product; the output formed after the loop; plain
   `bmm`s in the loop).
4. **Activation memory: nothing is checkpointed, and the cache fork forbids HF checkpointing.** `run_cell.train_config`
   sets `grad_checkpointing=False` (`train/run_cell.py:153`), and HF checkpointing would drop the fork's state
   (`modeling_rwkv7.py:1017-1021` forces `use_cache=False`; `heads/backbone.py:282-285` refuses). Both the prefix and
   the continuation call keep all activations: the chunked WKV casts its six inputs to fp32 (`modeling_rwkv7.py:213`)
   and builds fp32 `c, c_prev, b~, k~, q~, r~` and `[B, C, H, 64, 64]` score tensors (`:245-277`), the FFN keeps
   4x-wide `relu^2` activations (`:597`), LoRA keeps its inputs. Measured (saved-tensor hooks, one real-width layer:
   hidden 4096, 64 x 64 heads, FFN 16384, LoRA r64, bf16 autocast, T = 1024): **~1.0 MB per token per layer** →
   ~64 GB for one 1,024-token row over 61 layers, + 26 GB of weights: over 80 GB. This is why job-mden7 OOMed from
   T ≈ 1,021 (and passed 858–960 and, once, 1,146). The cell spec's 59 GB at T = 2,048 was off by ~2x.
5. **Syncs:** one `float(loss)` per micro-batch (`train/trainer.py:197`) and the host-to-device copies of
   `pad_batch`; no `.item()` inside the model. In a CPU-bound step a sync costs ~nothing (the GPU is idle anyway);
   not changed.
6. **LoRA** (`train/lora.py:51-54`): per call two casts (fp32 → bf16 of A and B), two matmuls, a multiply by
   `scaling` (= 1.0 for the cell) and an add, for 6 modules per layer. ~10% of the launches; not changed (the row
   batching divides it by the number of rows per call anyway).
7. **dtypes:** the WKV sees r / w_log bf16, kk fp32 (normalize under autocast), state fp32 (job-mden7
   `wkv_input_dtypes`); the chunked path computes in fp32 by design. Not changed (numerics).

## Changes (all behind flags; defaults = the original path, config hash unchanged)

| fix | where | flag | equivalence (tests/test_cell_perf.py) |
|---|---|---|---|
| (a) per-block activation checkpointing that keeps the cache fork: the recurrent state is read before and written after a non-reentrant `checkpoint` of each block; the block runs on a private one-layer state holder | `heads/act_ckpt.py`, `train/trainer.py:prepare_model` | `TrainConfig.act_ckpt` | loss and every LoRA / head gradient **bitwise** equal (fp32 and bf16 autocast); saved activations at T = 1024: 771 → 22 MB per layer |
| (b) batched rows: a step's rows sorted by length, grouped (<= `row_batch` rows, <= `row_tokens` padded tokens), each group = one prefix call with an attention mask + one call for all its rows' continuations (state rows gathered with repeated indices) | `heads/backbone.py:fork_features_rows`, `heads/features.py`, `train/trainer.py:row_groups / train_step` | `TrainConfig.row_batch`, `row_tokens` | fork features == per-row fork to 1e-5; step loss rel 8e-9, worst gradient rel 1.3e-6 (fp32) |
| (c) `chunked_lean` WKV: same algebra, 3.3x fewer aten ops per call (T = 2048), no O(chunks^2) zero fills | `heads/rwkv7_lean.py`, `heads/rwkv7_wkv.select_wkv` | cell `rwkv7_wkv: "chunked_lean"` | outputs / state / all input gradients within 8e-7 rel of the vendored function (fp32); model output 1e-5 |
| all three | | | fp32: loss rel 8e-9, worst gradient rel 8e-7. bf16 autocast: loss rel 3e-8 vs the bf16 baseline, worst gradient rel 1.2e-2, below the bf16 baseline's own distance to the fp32 step (2.6e-2) |

For the cell: `train.trainer.with_execution(train_config(cell, ...), cell)` reads `act_ckpt`, `row_batch`, `row_tokens`
from the cell config (one line in `train.run_cell`; not edited here because PR #74 changes those lines).

## Projection (A2: GPU basis; NOT GPU measurements)

GPU basis (reviewer, from job-mden7's forward-only chunked prefill at 227–265 µs/token): forward + recompute + backward
~0.7–0.9 ms per PADDED token; projections use the top, 0.9 ms. Padded tokens on the 23 mix steps (computed from
`train_mix_v2.json` with `train.trainer.row_groups`): per-row variants 124,586 (+2,374 continuation tokens); row variants
200,731 under the 8,192 budget (40 calls; whole-step padding would be 269,432). CPU-launch basis (11.3 µs x aten ops
counted on CPU, calibrated on job-mden7's completed baseline steps): ckpt 2.1–2.8 M ops per step, ckpt_rows 0.36–0.95 M,
ckpt_rows_lean 0.20–0.46 M. Each variant's projection takes the larger of the two.

| variant | GPU basis, 23 steps | CPU-launch basis, 23 steps | + ladder | + load / probe | expected (min) | cap (min) |
|---|---|---|---|---|---|---|
| baseline | 1.4 min (0.675 ms/token, no recompute) | 3.7 min (job-mden7 measured, most steps OOM early) | 0.3 | 0.6 | 4.6 | 8 |
| ckpt_rows_lean | 3.0 min (200,731 x 0.9 ms) | ~1.5 min | 1.4 (94,208 padded tokens) | 0.6 | 5.0 | 8 |
| ckpt_rows | 3.0 min | ~2.8 min | 1.4 | 0.6 | 5.0 | 8 |
| ckpt | 1.9 min | 10.4 min (its recompute adds 53% launches) | 0.7 | 0.6 | 11.7 | 18 |

Projected rates at the GPU basis: the row variants ~124,586 / 181 s ≈ 690 tok/s, about 2.6x the baseline's measured
~260, i.e. ~1,500 steps (~12k rows) in 190 min. The lean variant's lower launch count only matters if the true GPU cost
per padded token is below the reviewer's range. These projections size the caps; the decision uses only the job's own
measurements.

## Phases, caps, cost

| phase | cap (min) | expected (min) | what |
|---|---|---|---|
| setup | 3 | 2.0 | pins (torch 2.13.0, transformers 5.17.0, vendored code), CPU equivalence tests (`tests/test_cell_perf.py`, `tests/test_cell_step_rule.py`; exit 5 on failure); snapshot download starts in the background (warm `hf_cache`) |
| download_wait | 1 | 0.4 | wait, `sha256sum -c jobs/rwkv7/snapshot.sha256`, snapshot code == vendored (exit 10) |
| baseline | 8 | 4.6 | probe (mix step 22), 23 steps, ladder |
| ckpt_rows_lean | 8 | 5.0 | probe, 23 steps, ladder (8 rows per rung) |
| ckpt_rows | 8 | 5.0 | probe, 23 steps, ladder (8 rows per rung) |
| ckpt | 18 | 11.7 | probe, 23 steps, ladder (1 row per rung) |
| profile | 6 | 3.5 | `torch.profiler` of the decided variant and the baseline on one mid-length step (informational) |
| reserve | 1 | | decide (pure, seconds), finalize, exports |

Every cap is >= 1.5x its projection (A2).

Inside every phase no step / rung starts that cannot finish before the cap (minus 40 s); a variant that runs out of
time records the skipped steps and cannot qualify. A variant crash or timeout is a gate value, not an exit code.

`max_duration_minutes` 53, `build_timeout_minutes` 60, `max_restarts` 0, H100 at $0.0495/min. Worst case
(`jobs/common/costs.py`: build + 2 attempts) = (60 + 2 x 53) x 0.0495 = **$8.22** (hard cap $8.50; A2: no strict cap,
the figure is stated); one attempt $2.62; expected **$1.59**.

Outputs (exported per phase): `variant_<V>.json` (probe loss; per-step seconds, tok/s, peak / reserved GB, OOM,
loss; ladder),
`decision.json` (verdict, chosen variant, recommended cell config, projection), `profile.json` (+ traces),
`equivalence_tests.txt`, `env.json`, logs. Marker `CELL_STEP_PROFILE_DONE` (finalize, validator
`jobs.cell_step_profile.rule:valid_decision`).

Exit codes (every verdict, INCONCLUSIVE included, exits 0): 0 measured + decided + exports verified; 5 setup (pins, equivalence tests); 6 export failed; 9 decision
missing / invalid; 10 snapshot / vendored-code mismatch or download failed; 1 finalize failed.

Dry run: `CELL_STEP_DRYRUN=1 bash jobs/cell_step_profile/cell_step_job.sh` runs this exact script on CPU (tiny vendored
RWKV-7, same rows, same phases, exports and finalize); `tests/test_cell_step_job.py` does it end to end.

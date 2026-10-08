# jobs/perf_step_gate: GPU gate for the m2 cell train step (RULE v2: does fla speed TRAINING?)

**RULE v2 is the live rule** (`rule.py`, `RULE_ID = "perf_step_gate v2"`, branch stack/rwkv7/perf-gate-fla). RULE v1
(job-2amiy, $1.02, below under "RULE v1 (history)") picked the d1_fast flags. v2 asks the next question.

**Question (v2).** On one H100 and the same 23 m2-cell steps, is the fla `chunk_rwkv7` WKV faster than our
`chunked_lean` inside the TRAINING step (forward + checkpoint replay + backward), on top of the d1_fast flags? And does
it keep the loss, the LoRA gradient and the memory inside the cell's limits?

**Why.** analysis/FLA_WKV_NUMERICS_2026-10-08.md (stack/rwkv7/cost-latency @6b62155): fla's error vs fp64 is at the bf16
floor, flat in T, gradient cosine >= 0.99997, and its prefill is ~3x `chunked_lean`. It was rejected (jobs jn47u / mden7)
only by an inference clause e(fla) <= 1.10 x e(chunked) that no bf16 kernel can meet against an fp32 chunked path.
Training speed was never measured on the d1_fast step.

| variant | flags | role |
|---|---|---|
| greedy | ckpt_rows_lean: act_ckpt, row_batch 8, row_tokens 8,192, `chunked_lean`, `row_plan` greedy, `cont_plan` per_group | numerics reference (loss, LoRA gradient) |
| fast | greedy + `row_plan` min_cost + `cont_plan` single + `compile_blocks` (= d1_fast; job-2amiy 5.63 s/step) | speed reference |
| fast_fla | fast with `rwkv7_wkv` `fla` | the question |
| fast_fla_nocompile | fast_fla without `compile_blocks` | hedge if compile + fla conflict on GPU; attributes the compile gain under fla |

Compile + fla should not conflict by construction: under `compile_blocks` the WKV function is called through
`torch.compiler.disable` (`heads/act_ckpt._disabled_wkv`), so fla's Triton kernel runs eager between the compiled
regions, exactly as `chunked_lean` does. That is CPU-tested only (the dry run below runs `fast_fla` under aot_eager,
where fla falls back), hence the nocompile hedge (~6 min, ~$0.30).

fla is GPU-unverified for training (`heads/rwkv7_wkv.GPU_VERIFIED = False`); the gate selects it with
`allow_unverified=True` (this job IS its gate). fla-core 0.5.2 is not in the rwkv7 image: setup installs the exact wheel
by sha256 (`jobs/fla_rwkv7/requirements-fla.txt`, `--no-deps --require-hashes`, as `jobs/fla_rwkv7/fla_job.sh`), checks
torch 2.13.0 / triton 3.7.1 / fla 0.5.2 and runs the fla #640 guard (`jobs/common/fla_guard.py`); any failure exits 5.

## Decision rule v2 (`rule.py`, pre-registered, constants pinned by `tests/test_perf_step_gate_rule.py`)

A variant qualifies with all of:
- zero OOM; every loss finite;
- every step and ladder peak <= 76 GB, both ladder rungs (T 1,024 / 2,048, K 10) ok;
- >= 15 of the 20 timed steps completed;
- its pre-registered flags and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- **|gradient-step loss - greedy's| / |greedy's| <= 5e-3**;
- **cosine(LoRA gradient, greedy's) >= 0.995**. v1 measured a GPU reduction-order floor of ~0.998 for pure regroupings
  (job-2amiy: min_cost 0.99798, single 0.99820, compile 0.99771) and Agent A overrode v1's 0.999 bar on that evidence;
  0.995 leaves headroom for bf16 kernel noise on top of that floor;
- fla variants only: fla really ran, i.e. `heads.rwkv7_wkv.provenance()` shows >= 1 fla call and ZERO fallbacks to
  "chunked" (a fallback would time chunked under the fla name). The CPU dry run (fla cannot run) is exempt.

Greedy qualifies on numerics by definition. INCONCLUSIVE if greedy's own record is unusable. Otherwise ADOPT the fastest
qualifier by median step seconds over the timed steps every qualifier completed (ties: the earlier variant). Reported
per variant in `decision.json`: s/step, speedup vs greedy, **speedup vs `fast`**, **rows in 190 min**, loss_rel,
grad_cos, peak, fla provenance; plus `fla_speeds_training` = a qualifying fla variant is faster than a qualifying `fast`
(null if `fast` does not qualify).

## Procedure (`perf_step_job.sh`, `gate.py`)

setup (pins, fla install + #640 guard, CPU equivalence tests `tests/test_cell_perf.py` +
`tests/test_perf_step_gate_rule.py`, snapshot download + sha256) -> `greedy` -> measured-rate guard (`preflight.py`:
greedy's median step <= 1.5 x tvuat's 8.2475 s, else exit 15) -> `fast` -> `fast_fla` -> `fast_fla_nocompile` -> decide.
Each variant in its own process: one un-applied gradient step on mix step 6 (lora_B from seed 20261008, x 1e-3), the 23
mix steps (3 warm-up + 20 timed), the ladder. fla's Triton JIT (~5.7 min, job-jn47u warmup) lands in `fast_fla`'s
gradient step and warm-up; `fast_fla_nocompile` reuses the on-disk Triton cache.

## Phases, caps, cost (v2)

| phase | expected (min) | cap (min) |
|---|---|---|
| setup (+ fla install, guard) | 3.0 | 7 |
| download_wait | 0.5 | 2 |
| greedy | 4.0 (job-2amiy 3.8) | 7 |
| fast | 4.5 (job-2amiy compile variant 4.2) | 8 |
| fast_fla | 10.5 (4.5 + fla JIT ~6) | 18 |
| fast_fla_nocompile | 6.0 (cap fits a cold JIT) | 14 |
| decide | 1.0 | 2 |
| reserve | | 2 |
| **total** | **29.5** | **60** |

H100 x1 at $0.0495/min, no restarts: expected ~$1.46 (v1 billed $1.02 for 16 min). One running attempt at most $2.97.
Worst case by `costs.worst_case_usd` = (60 build + 2 attempts x 60) x 0.0495 = **$8.91** (m2 restart floor 1). Hard cap
$9.00. Code only, no private data.

---

# RULE v1 (history: job-2amiy, stack/rwkv7/perf-step @953f3ac)


**Question.** On one H100 and on the m2 first cell's own row shapes (`jobs/fla_rwkv7/train_mix_v2.json`, the 23 steps
job-tvuat timed), how many seconds does an 8-row optimiser step take under each of three execution fixes, applied on
top of job-tvuat's ADOPT config (`ckpt_rows_lean`: act_ckpt, row_batch 8, row_tokens 8,192, chunked_lean)? And do the
fixes keep the numerics (loss, LoRA gradient) and the memory inside the cell's limits?

**Why (CPU analysis, receipts in the PR).** A fit over tvuat's 20 timed steps gives
`step_s = 0.36 + 2.02 x groups + 0.462e-3 x padded tokens` (r2 0.995). The 2.02 s per row group is CPU kernel-launch
cost: about 140k aten ops per group, of which the continuation call is ~41% at almost no GPU work. Padding is 38% of
the prefix tokens.

| variant | flags on top of ckpt_rows_lean | expected (CPU model) |
|---|---|---|
| greedy (baseline) | `row_plan` greedy, `cont_plan` per_group | tvuat: 8.25 s median |
| min_cost | `row_plan: "min_cost"`: an exact DP cut of the length-sorted rows minimising 4,400 x groups + padded tokens under the same caps | ~1.07x |
| min_cost_single | + `cont_plan: "single"`: the groups' prefix calls, states joined, ONE continuation call + ONE backward per step | ~1.15x |
| min_cost_single_compile | + `compile_blocks: true`: one shared `torch.compile` (inductor, dynamic) of the checkpointed block body, WKV loop eager | 1.3-1.6x (unmeasured) |

All four flags are execution-only: they are not in the config hash, they train the same rows with the same loss
weights, and they are equal to the original step within 2e-5 on CPU fp32 (`tests/test_cell_perf.py`; compile under the
aot_eager backend). Inductor fusion changes float rounding, so the GPU gradient check below is what decides it.

## Procedure (`perf_step_job.sh`, `gate.py`)

setup (pins; CPU equivalence tests `tests/test_cell_perf.py` + `tests/test_perf_step_gate_rule.py`; snapshot
download + sha256) -> `greedy` -> measured-rate guard (`preflight.py`: greedy's median step must be <= 1.5 x tvuat's
8.2475 s, else exit 15) -> `min_cost` -> `min_cost_single` -> `min_cost_single_compile` -> decide. Each variant runs in
its own process (clean peaks):

1. load, then ONE optimiser step on mix step 6's rows (max_T 2,025: two row groups under every plan) with every
   lora_B set to 1e-3 x N(0, 1) from seed 20261008 (LoRA B starts at zero, which would make every lora_A gradient 0).
   The step is NOT applied. Record its loss and the flattened fp32 LoRA gradient. For the compiled variant this step
   also compiles.
2. The 23 mix steps: 3 warm-up + 20 timed, CUDA-synchronised seconds, peak memory reset per step, OOM recorded.
3. Ladder: one step over 8 copies of a synthetic K = 10 row at T = 1,024 and 2,048 (the cell's largest T). Under
   `cont_plan: "single"` every group's checkpoints stay alive until the single backward, so this is the memory test.

## Decision rule v1 (superseded by v2 above; was pre-registered, constants pinned by `tests/test_perf_step_gate_rule.py`)

A variant qualifies with all of the following:
- zero OOM;
- every loss finite;
- every step and ladder peak <= 76 GB, and both ladder rungs ok;
- >= 15 of the 20 timed steps completed;
- its pre-registered flags and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- **|gradient-step loss - greedy's| / |greedy's| <= 5e-3**;
- **cosine(LoRA gradient, greedy's) >= 0.999**.

The baseline qualifies by definition on the numerics. Rank by the median step seconds over the timed steps every
qualifier completed. Verdict ADOPT names the fastest qualifier and its cell flags. INCONCLUSIVE if greedy's own record
is unusable. Every variant's s/step, speedup vs greedy and rows-in-190-min are in `decision.json`.

## Phases, caps, cost

| phase | expected (min) | cap (min) |
|---|---|---|
| setup | 2.5 | 6 |
| download_wait | 0.5 | 2 |
| greedy | 4.5 | 7 |
| min_cost | 4.3 | 7 |
| min_cost_single | 4.0 | 7 |
| min_cost_single_compile | 12.0 (inductor compile unmeasured) | 22 |
| decide | 1.0 | 2 |
| reserve | | 2 |
| **total** | **28.8** | **55** |

H100 x1 at $0.0495/min, no restarts:
- Expected ~$1.43.
- One running attempt at most $2.72 (what validate_job reports).
- Worst case by `costs.worst_case_usd` = (60 build + 2 attempts x 55) x 0.0495 = **$8.42**, using the m2 restart floor
  of 1. Hard cap $8.50.
- Code only, no private data.

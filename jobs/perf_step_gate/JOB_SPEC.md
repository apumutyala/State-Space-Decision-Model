# jobs/perf_step_gate: GPU gate for the m2 cell step fixes (stack/rwkv7/perf-step)

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

## Decision rule (`rule.py`, pre-registered, constants pinned by `tests/test_perf_step_gate_rule.py`)

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

# jobs/cell_step_profile: the m2 first cell's train step, four execution variants (DRAFT, NOT SUBMITTED)

**Question.** Job-mden7 (SF, H100 80 GB) ran the m2 first cell's real train step (`train.trainer.Trainer.train_step`
with `train.run_cell.train_config`: RWKV7-G1k-13.3B bf16, LoRA r64, cache-fork head, Clef loss, batch 1 x accumulation
8, no checkpointing, WKV `chunked`) on the cell's first 23 optimiser steps (`jobs/fla_rwkv7/train_mix_v2.json`).
The production path OOMed on 17 of 20 timed steps (peak 84.0 GB; weights 26.0 GB), and each completed step took
15–17 s for 3.5k–4.8k tokens: ~2 s of fixed cost per row, flat in tokens. The cell's ladder would end at T <= 512 or
exit 22, and its 190 minutes would hold ~700 steps instead of the planned ~1,600. Which execution variant of the SAME
step (same rows, same loss, same gradients) fits in 76 GB on every row, and how fast is it?

## Decision rule (PRE-REGISTERED; `jobs/cell_step_profile/rule.py`, pinned by `tests/test_cell_step_rule.py`)

Committed alone, before the harness and before any run. Not to be edited after GPU numbers exist.

**Variants** (each in its own process, in this order; cumulative):

| variant | `act_ckpt` | `row_batch` | `row_tokens` | `rwkv7_wkv` |
|---|---|---|---|---|
| `baseline` | false | 1 | 0 | `chunked` (the vendored checkpoint path) |
| `ckpt` | true | 1 | 0 | `chunked` |
| `ckpt_rows` | true | 8 | 8192 | `chunked` |
| `ckpt_rows_lean` | true | 8 | 8192 | `chunked_lean` |

`act_ckpt` = per-block activation checkpointing that keeps the cache fork. `row_batch` 8 / `row_tokens` 8192 = the 8 rows
of a step sorted by length (longest first) and grouped greedily into calls of at most 8 rows and at most 8,192 padded
tokens (rows x the group's longest row; a longer row runs alone); each group is one prefix call + one continuation call,
right-padded with an attention mask. `chunked_lean` = the same chunk-parallel WKV algebra with fewer launches. Every variant runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(recorded per variant).

**Measurements per variant.** The 23 steps of `train_mix_v2.json` (3 warmup + 20 timed, 8 rows each, in the cell's
epoch-0 order). Per step: wall seconds (CUDA-synchronised), tok/s = prefix tokens / seconds, peak GB
(`torch.cuda.max_memory_allocated`, reset before every step), OOM, loss. An OOM step is recorded and the run continues.
Then the K = 10 ladder: synthetic rows of T = 1024 / 1536 / 2048 / 3072 / 4096 tokens with 10 continuations of 16
tokens, one optimiser step over `row_batch` copies of the row (1 for `baseline` / `ckpt`; 8 for the row variants,
grouped exactly as a real step: the unit that sets the peak), ascending, stopping at the first OOM or peak > 76 GB.

**A variant qualifies iff** all of:
1. all 23 steps ran, with zero OOM and a finite loss on every one;
2. every step's peak <= 76.0 GB (the cell's `pass_rule.max_peak_gb`);
3. its flags and allocator setting are the pre-registered ones;
4. its step-0 loss is within relative 1e-3 of the baseline's step-0 loss. If the baseline's step 0 did not complete
   (job-mden7: it OOMed), this clause rests on the variant's CPU equivalence tests (`tests/test_cell_perf.py`, run by
   the runner's setup on the image before any GPU phase; setup fails with exit 5 if they fail).

**Verdict.**
- `ADOPT`: the qualifying variant with the highest median tok/s over the 20 timed steps (ties: lower peak, then
  earlier variant). Recommended cell config = its flags + `t_ladder` = the cell's [512, 1024, 2048] cut at its ladder
  T_fit (the full ladder if T_fit >= 2048), and the projection: steps in 190 min = floor(190 x 60 / median timed step
  seconds), rows = 8 x steps.
- `CAP_T`: no variant qualifies. The cell must cap T at the largest ladder T that fits among the variants whose loss
  clause holds (ties: higher median tok/s); recommended config = that variant's flags, `t_ladder` cut at it, `t_cap`.
- `NO_FIT`: nothing fits even T = 1024.

### Amendment A1 (2026-10-06, before any run; rule `cell_step_profile v1-A1`)

The pre-registration commit had `ckpt_rows8` / `ckpt_rows8_lean`: all 8 rows of a step in ONE call, padded to the
step's longest row. Before any GPU run, a CPU measurement of the saved-for-backward bytes of one real-width layer
(hidden 4096, 64 x 64 heads, FFN 16384, LoRA r64, bf16 autocast, cache-fork row at T = 1024) gave ~1.0 MB per token per
layer without checkpointing and ~22 kB per token per layer with it. One call of 8 rows padded to ~2,025 tokens (steps
0, 1, 6 of the mix) is 16.2k tokens: ~22 GB of checkpoint store + ~17 GB to recompute one layer + backward transients
+ 26 GB weights + 4.6 GB LoRA / grads / Adam = ~74 GB, at the 76 GB line. The row variants are therefore renamed
`ckpt_rows` / `ckpt_rows_lean` and carry `row_tokens` = 8192 (~53 GB by the same arithmetic; 2–3 calls on the long
steps, 1 on short ones). Nothing else in the rule changed.

The verdict is a recommendation for the m2 cell (PR #74 / its follow-up), which keeps its own ladder stage and
stop rules (exits 22 / 24 / 26) as the authority on the real rows.

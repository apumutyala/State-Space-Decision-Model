# m2_first_cell: the first trained attention-free decision model. DRAFT, NOT SUBMITTED

**Question.** Does training a decision head (plus LoRA) on the decision format move RWKV7-G1k-13.3B off its stock
option-likelihood readout (24.9 [22.4, 27.8] on the 2,008-row Index DEV subset, job-tkfaj) toward Clef-flash
(55.8 [53.0, 58.5] on the same rows; 56.8 [55.4, 58.2] on the full 12,104-row DEV)?

**Cell** (`m2_first_cell_rwkv7.json`):

| Item | Value | Why |
|---|---|---|
| Parent | `RWKV/RWKV7-G1k-13.3B-20260930@ce3630b8`, vendored code, `chunked_lean` WKV (both arms) | job-tvuat ADOPT (below); fla is not GPU-verified |
| Execution | `act_ckpt` true, `row_batch` 8, `row_tokens` 8,192 (`train.trainer.with_execution`) | job-tvuat (`jobs/cell_step_profile`, rule v1-A2, PR #86 @54ee89d) ADOPT `ckpt_rows_lean`: median step 8.25 s, 0 OOMs. Same loss and gradients (not in the config hash) |
| Head | final-state fork, `cache` mode | one prefix pass + K state copies: ~3x forward per step vs ~4x for the checkpointed integrator; it is the pilot's pre-registered entrant and its cache == recompute smoke (exit 21) is already built |
| LoRA | r = 64 on `att.receptance/key/value/output`, `ffn.key/value` (288 M params) | r = 256 (M2_RUN_SPEC) adds 1.15 B params, ~72 GB estimated at T = 2,048 without checkpointing; r = 64 estimates 59 GB under the 76 GB stop |
| Loss | Clef arm: LS-CE (0.1) + Brier, + RPS on score questions | as asked |
| Data | PR #44 150k set (pointer `sample-150000-seed20261006`), 24,000 records drawn by `sha256(seed:id)`; injection augmentation is in the build (D1-A verbatim 8,049, training-only hijack 4,066, labels unchanged) | v2 is not merged |
| Guards | content-index guard (`7cbe6c89…`, readiness file required), cohort v2 excluded by id + content + source item (build) and re-checked (`train.train_set`: 0 held-out records allowed, manifest-verified lines, content re-hash), D1-B held-out gate over all 150k | |
| Steps | 8 rows per optimiser step (batch 1 x accumulation 8), run in length-sorted groups of up to 8 rows and at most 8,192 padded tokens per call (a step whose longest row exceeds 1,024 tokens runs as 2 or more groups; same loss and gradient), warmup 50, `max_minutes` 190 (pre-registered), `max_steps` 3,000, T ladder 512 / 1,024 / 2,048, K ≤ 16 | |

**Held-out finding (new).** `train.heldout` finds a D1-B `soft_framing` fragment ("rather than relying on") in three
MMLU aux_train records of the 150k set (natural text; the build's 6-gram guard does not flag them). They are excluded
from the draw by a pinned list; the heldout phase passes only if the full-set hits equal that list (exit 33 otherwise),
and every drawn record is re-scanned.

## Measured rates and the expected throughput

job-tkfaj probe: prefill 3,757 tok/s at 2,048 tokens (4,407 at 8,192), small call 0.117 s, pack call 0.60 s.
Training rows (2,000-record sample of the 150k set, real tokenizer): 0.91 kept rows per record, mean 619 tokens, K 3.4.
Training step (MEASURED, job-tvuat, the ADOPT config on the same RWKV7 + cell; receipt `receipts/2026-10-07-D-cell-step-profile-tvuat/`, LEDGER row): median 8.25 s per 8-row step, so
190 min = 1,382 steps = 11,056 rows (about 6.8 M tokens at the 619-token mean). The ladder and the 10-step check
re-measure it on this box: exit 26 below 600 steps, i.e. a step slower than 19 s (2.3x tvuat's) stops training before
the evals are spent. Scoring phases are priced at job-tkfaj's readout rates, measured on `chunked`. `chunked_lean` was profiled for
training only, so its readout speed is UNMEASURED; the parent arm's live rate guard and the re-price measure it on
this box before any training spend. The tkfaj reference (24.9) is on `chunked`: a comparison with the in-job
`chunked_lean` parent arm crosses implementations (same algebra; labelled in the report).

## Phases (`job_params.json`; H100 batch $0.0495/min)

The parent arm runs BEFORE training (review fix 1): its live rate guard measures this box's scoring speed, and
`jobs.m2_first_cell.reprice` re-prices the child evals at that measured speed. A slow box stops (exit 15) after
the parent scoring instead of after training.

| Phase | Cap | Expected | Content | Exit |
|---|---|---|---|---|
| setup | 25 | 15 | env pins + WKV selection check; fla guard; PR #16 records, 150k set (pointer), cohort rebuild, snapshot + vendored code, DEV rows; subset rows; every eval phase priced from `TOK_S` / `CALL_S` | 4 / 5 / 10 / 15 |
| guard | 5 | 3 | `guard_ready` (content index == pin) | 30 / 31 / 32 |
| heldout | 10 | 3 | `train.train_set heldout` (hits == pinned exclusions); exported | 33 |
| parent_g12 | 85 | 65 | stock readout: cohort dev + steer_dev (1,800 requests, clean + 4 D1-A = 9,000 units), live weighted rate guard | 15 / 10 / 9 / 7 |
| parent_index | 65 | 48 | stock readout: Index DEV subset + order diagnostic; then the child evals are re-priced at the measured speed | 15 / 10 / 9 / 7 |
| cell | 230 | 220 | smoke (cache == recompute fork), ladder, training (190 min, cumulative over restarts), cards at step 0 and the end | 21–29 / 35 |
| evals | 205 | 155 | child: temperature fit (cohort calibration), dev card, G12 (full steer set), Index DEV subset + order diagnostic | 15 / 10 / 9 / 7 |
| report | 5 | 4 | `summarize` -> `first_cell_result.json` (finalize validator); final checkpoint copied to `/output/ckpt` | 9 |

Schedule 630 + reserve 7 = 637 min. **Worst case** (60 build + 2 x 637) x $0.0495 = **$66.03** (cap $70; build_timeout 60 per the launch lint, L-09).
**Expected ≈ $26.1** (528 min incl. build). A G12 reject is a measured outcome (exit 0, reported).

Every bounded long command uses GNU `timeout --kill-after=30s` at its existing cap;
parent/child scoring children retain the measured live rate guards outside that
timeout. TERM-resistant work is killed after the grace, and both timeout statuses
124/137 fail closed (training timeout exits 7) before partial artifacts are exported.
Final export and result validation share a 60-second deadline plus 30-second KILL
grace. Up to nine executed phase graces (4.5 min), three rate-guard polling allowances
(0.75 min) and terminal export/finalize (1.5 min) total 6.75 min within the existing
seven-minute reserve. Measured caps, stop thresholds and global 637-minute ceiling
are unchanged. A terminal timeout is a failed job, never a success marker.
After the final export, the timeout's shell `exec`s the Python finalizer so it owns
that process directly. Any nonzero terminal exit removes the completion marker
again in the parent shell, including a finalizer that wrote a marker and then
ignored TERM. This cleanup retains verified partial exports and any incomplete
failure receipt already written.

Measured-rate stops: setup preflights at the env rates (exit 15); the parent arm's live weighted rate guard and the
re-price at its measured speed, both before training (exit 15); ladder and 10-step checks (exit 26 if 600 optimiser
steps are out of reach); live weighted rate guard on the child evals (exit 15 at a projection past 95 % of the cap).
Rate margin at the caps (projection <= 95 % of the cap; the re-price uses the same rule): a box up to 25 % slower than
job-tkfaj (prefill rate and per-call latency) completes every scoring phase: parent_g12 tolerates 25.6 %, parent_index
31.2 %, child evals 27.4 %. Slower boxes stop before training (parent live guard or the re-price, exit 15).
Restart: cached parent stages are skipped (their rate reports stay in the arm dir); training resumes from
`$GMN_CHECKPOINT_DIR` with the 190-min budget reduced by what the checkpointed steps already used (`train_clock` in
`cell.json`), so a restart cannot train past the pre-registered budget.

WKV: the exact cell sets `rwkv7_wkv: "chunked_lean"`, `act_ckpt: true`, `row_batch: 8`, and `row_tokens: 8192`
(job-tvuat training execution). PR #147 applies checkpointing to the pre-training smoke too and can seed the
parent arm from job-uhxb5's pinned export; the runner verifies its full identity and re-scores the registered
24-record probe before training, then re-prices evaluation on this box. An ordinary cached parent is also
identity-verified. Historical job-tkfaj stock-readout reference and rates used `chunked`; they are not a new
`chunked_lean` measurement. The seeded probe or fresh in-job parent measures current scoring speed.

## What the result shows

`first_cell_result.json`: Index DEV subset score + 95% CI, overall and per area, for the trained child and the stock
parent, and the paired child − parent difference overall and per area; ECE (debiased and board protocol) per arm on
the subset, and raw / tempered on the cohort dev card; the child − parent Index difference split at the training length
(kit groups whose longest training-layout row is ≤ 2,048 vs > 2,048 RWKV7 tokens: 455 of 2,008 rows are longer,
`index_lengths_rwkv7.json`), with paired CIs; steer and hijack under D1-A (paired, full G12 set); per-row
latency; training loss / steps / step time / peak memory; cost; the G12 verdict and the promotion verdict under the
issue #36 margins (`train.promotion.MARGINS`, approved in PR #102). The cell runs G12 + the `index` measurement stage,
not G13, so it is never promotable: m2_gates exits 44 (partial) when G12 accepts, 40 when it rejects; 43 (pending) and
45 (underpowered) are handled the same way. All of them are "trained + measured": the report, every eval and the
checkpoint are exported, `verdict: measured`, `promotable: false` with the reason (exit 45 is tested end to end in
`tests/test_m2_first_cell_shell.py`). The parent arm is passed as both `--parent-arm` and `--anchor-arm` (same
in-job cache, one pinned sha256). Before the child scores the subset, a $0 plan-only check requires it to refuse
exactly the rows the parent arm refused (`m2_gates.index_plan_check`, `dev_full.classify`; exit 10 otherwise).
Readout diagnostics (both arms): position / label bias (chosen-first vs gold-first, by K and area), and accuracy and
agreement with the option order reversed (160 rows, forward vs reversed numbering). Optional shipped files add the
child vs Clef-flash comparison on the identical rows and an independent re-measurement of the parent against job-tkfaj.

## What it does not show

- One seed, one cell: no head-vs-head or arm-vs-arm claim.
- Steer is in-distribution: D1-A is in the training data. Held-out D1-B (G13) and the retention set are not run; the
  exported checkpoint is gated later by a separate G13 job (`train.m2_gates --stage g13 --anchor-arm ...` on the full
  frozen DEV set).
- **Cohort: this cell pins v2, deliberately.** v3 (#105) keeps every v2 role byte for byte and only expands the G13
  retention role (36.7k items). This cell does not run G13, so pinning v3 would buy no power and would add the v3
  retention records to the temperature fit's disjointness proof (~39k records loaded and tokenized, needs `pool_dirs`).
  The training draw already excludes v3 regardless of the pinned cohort (`train.train_set.draw`, #105; the 24k draw
  `2bbbf180` is unchanged). The G13 gate for this checkpoint should pin v3. Its retention term alone is ~226 min / arm
  on RWKV7 at the job-tkfaj rates (#105 cost notes: 22.9 M positions, 63.7k calls, $11.18 / arm; conservative 739
  min / arm). With both arms, at the job-tkfaj rates: retention 2 x 226 min, steer_gate 2 x ~42 min (5.43 M positions,
  9,000 calls), full frozen DEV 2 x ~225 min (38.8 M positions, ~27k calls): about 16–17 H100-hours (~$50) before its
  own preflight. That is a separate G13 job, as #105 recommends; it is not folded into this cell.
- The Index comparison is untempered for both arms (like for like); the served, tempered Index number is not produced.
- The 2,008-row subset is not the 12,104-row DEV slice; Clef-flash's 56.8 is on the full slice (55.8 on the subset).
- The eager WKV bounds training to ~8 M tokens; a flat result is not evidence that more training would not help.

## `submit_job` fields (do not submit until PR #74 is merged and independently reviewed)

1. Context: `python -m jobs.common.launch m2_first_cell --sha <merged sha> --out <dir> --data <SSDM_DATA>` (PR #93:
   lint + job tests on the archived tree, pinned data, deterministic context, `submit.json`), then `create_context`,
   upload, `finalize_context`.
2. `open_mission("m2-first-cell")`; `set_workspace_cap(MTD + 89.10)`.

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "m2-first-cell",
 "idempotency_key": "m2-first-cell-<context_sha256[:12]>",
 "label": "m2 first cell: RWKV7-13.3B LoRA r64 + cache-fork head, Clef arm, 150k; tvuat ADOPT exec",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/m2_first_cell/first_cell_job.sh",
 "env": {
  "TOK_S": "3757",
  "CALL_S": "0.117"
 },
 "max_duration_minutes": 637,
 "max_restarts": 0,
 "resume": "checkpoint",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/M2_FIRST_CELL_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 },
 "clock_lock": true
}
```

## Before submit

1. Independent fresh-context review; merge to `agent-main`.
2. Spend issue + LEDGER row; `set_workspace_cap(MTD + 89.10)` (hard cap $89.10; worst case $84.04 at $0.063/min, was $70 / $66.03); credit expiry 2026-10-27.
3. Env: `TOK_S=3757 CALL_S=0.117` (job-tkfaj probe). No `M2FC_*` variable: those are the runner's test-only path
   overrides (`tests/test_m2_first_cell_shell.py` executes the runner end to end under bash on CPU; it needs the image's
   transformers 5.17.0, e.g. `CUDA_VISIBLE_DEVICES= .venv-gpu/Scripts/python -X utf8 -m pytest tests/test_m2_first_cell_shell.py`,
   and Git's usr/bin/bash, not the bin/bash launcher); run.log names any
   that are set.
4. `build_context.sh <sha> <out> [<tkfaj results.jsonl> [<clef-flash results.jsonl>]]`; pin the optional files' sha256
   in `job_params.json` `reference_results_sha256` to include those comparisons.

Mode `d1_fast` (`first_cell_job.sh d1_fast`, `job_params_d1_fast.json`, `m2_first_cell_rwkv7_d1_fast.json`; generic runner,
default `train_dir` `m2-150k`): the D1 base cell plus the GPU-gated step fixes of stack/rwkv7/perf-step @953f3ac
(job-2amiy, rule `perf_step_gate` v1): `row_plan` `min_cost` (6.58 s/step, 1.20x), `cont_plan` `single` (6.11, 1.29x,
peak 63 GB), `compile_blocks` (5.63, 1.40x, peak 63 GB) vs greedy 7.89 on the same box; loss within 1.9e-4 relative,
gradient cosine 0.998 for every variant including pure regrouping (the GPU reduction-order floor; Agent A overrode the
0.999 bar on that evidence). Execution-only (not in the TrainConfig hash). Data, caps, schedule and cap arithmetic are
D1's: **Worst case** **$66.03** (cap $70), expected ~$26.14; the same 190-min clock trains ~40 % more rows (time cap
binds). The template future cells copy. With `row_options` set (D8), `cont_plan` `single` runs a step as one
continuation call only when the step's summed K fits the cap; otherwise that step takes the per-group path.

Mode `d10_v2f_b100k` (D10, the combined cell; `first_cell_job.sh d10_v2f_b100k`, `job_params_d10_v2f_b100k.json`,
`m2_first_cell_rwkv7_d10_v2f_b100k.json`; generic runner, `train_dir` `m2-b100k`): `d3_v2f_fast` (template `jrt-rwkv-g1-v2f`,
layout `g1v2`, d1_fast flags) on Agent B's B1 100k weighted draw v2 (stack/rwkv7/b1-draw @e3eedeb; pointer
`data/manifests/b1-draw100k-v2.pointer.json`, raw gz pins `data_d10_b100k.sha256`, `n_train_records` 100,000, heldout
exclusions empty) with D8's training-K lift `train_max_k` 255 / `row_options` 160 (#221). T ladder stays 512/1,024/2,048
and eval is unchanged (`max_k` 16, `eval_max_t` 2,048, parent arm in-job in v2f). Census (real tokenizer, v2f, CPU): draw
100,000, heldout 0 hits, 154,355 training items (K <= 16 144,590; K 17-27 7,602; K 77 2,163), mean T 838.7; only 635 items
(0.4 %) exceed 2,048 tokens, so no T = 4,096 mode. Tokenize for 100k rows: 10 x 100/24 = 41.7 -> 42 min (+32 vs D1). Cell cap
230 + 32 = 262; schedule 630 - 230 + 262 + 7 = 669 min. **Worst case** (60 + 2 x 669) x $0.0495 = **$69.20** (cap $70).
Expected 560 min ≈ $27.72. ~16k rows at the d1_fast step (5.63 s); ~12k if the longer v2f rows (mean T 839 vs 604) slow it.

Mode `d10_v2f_b100k_t380` (`job_params_d10_v2f_b100k_t380.json`, `m2_first_cell_rwkv7_d10_v2f_b100k_t380.json`): `d10_v2f_b100k`
with the clock 190 -> 380 min (as D7 `d7_t380`) and `max_steps` 3,000 -> 4,500 (operational, not hashed) so the clock binds
(~4,050 steps / ~32k rows at 5.63 s; ~24k at ~7.5 s). Cell cap = D7's 421 + 2 checkpoint margin + 32 tokenize = 455; schedule
630 - 230 + 455 + 7 = 862 min. **Worst case** (60 + 2 x 862) x $0.0495 = **$88.31** (cap $90). Expected 753 min ≈ $37.27.

Mode `d10w_v3w_b100k` (D10w; `job_params_d10w_v3w_b100k.json`, `m2_first_cell_rwkv7_d10w_v3w_b100k.json`): `d10_v2f_b100k` with only
template `jrt-rwkv-g1-v3w` (layout `g1v2`, r1_format_commit 0290fa8; parent arm in-job via `G1V3Backend(mode='words')`). Caps and
schedule are D10's: **Worst case** (60 + 2 x 669) x $0.0495 = **$69.20** (cap $70).

Mode `d16_v3w_b100k_safe` (D16; `job_params_d16_v3w_b100k_safe.json`, `m2_first_cell_rwkv7_d16_v3w_b100k_safe.json`): `d10w_v3w_b100k` +
anti-collapse safe keys (PR #232) + the g1v2 ordinal-head fix (6be1c8f) + `train_set.attack_clean` (b1-draw100k-v2: 5,201 D1-A rows dropped) +
the `d10_v2f_b100k_t380` clock (380 min, max_steps 4,500, cell cap 455); `"stage": "screen"` with screen_readout v3w (d1_v3w / stock_v3w),
screen cap 85 (fixed 15 + 32 tokenize for 100k + 25.97 proxy child = 72.97 <= 0.95 x 85). Schedule 25 + 5 + 10 + 85 + 65 + 455 + 85 + 5 + 7 =
742 min. **Worst case** (60 + 2 x 742) x $0.0495 = **$76.43** (cap $90). Expected 671 min ≈ $33.21.

Mode `d17_v3w_bschema230k_v2e` (D17; `job_params_d17_v3w_bschema230k_v2e.json`, `m2_first_cell_rwkv7_d17_v3w_bschema230k_v2e.json`): `d16_v3w_b100k_safe`
with ONLY the training set changed to B's `bschema-draw230k-v2e` (pointer `data/manifests/bschema-draw230k-v2e.pointer.json`, 234,521 records /
366,741 items, C all-K multirecord 25.9 % of items; n_train_records = the whole pool, train_dir `m2/build/bschema-draw230k-v2e`; no heldout_excluded /
attack_clean / doc_clean pins: the heldout phase and the draw require 0 train.heldout hits, 0 attack_guard hits and 0 doc_overlap_guard fails; the
pointer's v1_overlength_ids are v1-only: under v3w all 366,741 items pass keep_item, max T 2,048) and the data-size-dependent caps: heldout 15,
cell 531 (D16 455 - 32 + 99 tokenize scaled by items from v2d's 95 + 9 for the draw's guard scans), screen 157 (fixed 15 + 99 + 9 + 25.97 proxy
child = 148.97 <= 0.95 x 157). Clock unchanged (380 min, max_steps 4,500). Schedule 25 + 5 + 15 + 85 + 65 + 531 + 157 + 5 + 7 = 895 min.
**Worst case** (60 + 2 x 895) x $0.0495 = **$91.58** (cap $95). Expected 828 min ≈ $40.99.

Mode `d18_v3w_v2e_joint` (D18; `job_params_d18_v3w_v2e_joint.json`, `m2_first_cell_rwkv7_d18_v3w_v2e_joint.json`): `d17_v3w_bschema230k_v2e` with ONLY
`"head": "joint_record"` (heads.heads.JointRecordHead, from stack/rwkv7/d13-joint; zero-gated, so step 0 = the D17 fork head). Training steps take whole
records (train.trainer: record-contiguous order, chunked by whole records when the summed K exceeds row_options 160, a record longer than the step kept whole
up to 2 x 8 rows); the screen child scores each request in one forward (DecisionModelBackend). Caps, clock and schedule are D17's: 895 min.
**Worst case** (60 + 2 x 895) x $0.0495 = **$91.58** (cap $95). Expected 828 min ≈ $40.99.

Mode `d19_v3w_v3b_joint` (D19; `job_params_d19_v3w_v3b_joint.json`, `m2_first_cell_rwkv7_d19_v3w_v3b_joint.json`): `d18_v3w_v2e_joint` with ONLY
the training set changed to B's `bschema-draw-v3b` (pointer `data/manifests/bschema-draw-v3b.pointer.json`, 253,046 records / 385,266 v3w items, 0 dropped;
no doc_clean / attack_clean / heldout pins: every guard must find zero) and cell 533 / screen 159 (screen_fixed_minutes 125; re-timed draw 330 s, heldout 198 s,
tokenize 573 s at 6 workers, all at or below v2e). Schedule 25 + 5 + 15 + 85 + 65 + 533 + 159 + 5 + 7 = 899 min.
**Worst case** (60 + 2 x 899) x $0.0495 = **$91.97** (cap $95). Expected 832 min ≈ $41.18.

fla in cells (any mode whose cell config sets `"rwkv7_wkv": "fla"`; defaults unchanged, every committed config keeps its
own WKV). Adopted from SF job-9bs8w (`jobs/perf_step_gate` rule v2, stack/rwkv7/perf-gate-fla @3501b56): `fast_fla`
(d1_fast flags + fla) 3.88 s/step vs `fast` 5.65 (1.46x) and `greedy` 7.94 (2.05x), LoRA-grad cos 0.998, loss_rel
3.2e-4, peak 62.6 GB; `heads.rwkv7_wkv.GPU_VERIFIED` is now True for plain `fla` only (`fla_safe_gate` stays gated).
Setup then (exit 5 on any failure): installs fla-core 0.5.2 by wheel hash with `--no-deps --require-hashes`
(`jobs/fla_rwkv7/requirements-fla.txt`, exactly as perf_step_gate v2), checks torch 2.13.0 / triton 3.7.1 / fla 0.5.2 and
the `chunk_rwkv7` import (`out/fla-versions.json`), and requires the #640 guard (`out/fla_guard.json`) to have run on
the installed fla. `train.run_cell` runs every ladder and training step under `heads.rwkv7_wkv.strict()`: a fallback
to "chunked" raises and the cell stops with exit 36 (also when training ran but no call went to fla). `cell.json`
`"fla"` records the requested name, fla calls and fallbacks (process totals; smoke and eval cards are not strict, so a
fallback there is counted, not fatal). The parent arm and child evals use the same cell `rwkv7_wkv` (identity).

## Mode `fulldev_eval`: a saved checkpoint on the FULL Index DEV (DRAFT, NOT SUBMITTED)

**Question.** What does a trained cell score on all 12,104 DEV rows, on exactly the comparators' footing (Clef 27B
job-zfexa art-p93d4 61.89 [60.28, 63.28]; Clef-flash job-t5yg2 art-7pyk2 57.29 [55.88, 58.67])? Files:
`job_params_fulldev_eval.json`, `m2_first_cell_rwkv7_fulldev_eval.json` (= the D1 base cell), driver
`fulldev_eval.py`, lengths `fulldev_lengths_rwkv7.json`, landing `analysis/land_fulldev.py`.

**Same as the comparators.** Rows: `data/slices/dev-rows.jsonl.gz`, pinned by `jobs/clef_baseline/data.sha256` (the
file `harness.cli.load_slice("dev")` reads). Edition 0.2.1. Scorer: `python -X utf8 -m harness score|ece|compare
--slice dev --edition 0.2.1` (frozen). Latency: the Clef row's definition (ok rows, first 10 dropped, median and
nearest-rank p95 of the per-row `total_wall_ms`, H100, batch 1).

**Over-length rows.** Clef's shipped max_length is 16,384; the comparators re-read the 51 rows over it
(`jobs/clef_baseline/overlength_rows.json`) at 32,768 with no truncation and substituted them. RWKV7 has no positional
limit, so the candidate reads every row in full: the adapter's 16,384 default is raised through the request's
`ssdm.limits.max_input_tokens` (the Clef 32k arm's own mechanism) to 65,536, above the longest full-DEV row (47,509
RWKV7 tokens; 68 rows exceed 16,384 in RWKV7 tokens: the 51 plus 17 more). A $0 plan-only pass over all rows runs
before any forward (`plan-child.json`); a row still refused scores as the kit scores it (no credit) and is listed.
The report gives both `full` (12,104 rows, headline, paired with the 32k-substituted comparator files) and
`excl_overlength51` (12,053 rows).

**Child arm** = the in-job Index arm of `train.m2_gates` stage `index`: `Trainer.load` (config hash + lineage), the
cohort temperature fit, served decider, `index_predict(..., untempered_from_raw=True)`: `child.results.jsonl` holds
the untempered answers (the arm D1's 35.12 subset number used), `child.results_served.jsonl` the tempered ones (served
ECE). **Parent arm** (optional, `"fulldev_parent_arm": true`; also add `phase_caps_minutes.parent_full: 185` and
+185 to `max_duration_minutes`): the stock readout on the same rows and cap, for the paired child − parent Δ.

**Phases.** setup 25 / guard 5 / checkpoint 15 (`checkpoint_loader install`, shared with `hijack_eval`) / fulldev
550 (live weighted rate guard; per-row resume from `$CK/fulldev`) / report 10 / reserve 7 = 612 min.
**Sizing** (D1 job-c7pq2 art-bm3e6, `units-child-index.json`): 2,000 ok subset rows in 3,680 s of decide wall
(32.8 rows/min; 1.037 s per 1k plan tokens). Full DEV = 22,630,947 plan tokens → 391 min scoring + 30 fixed = 421
(368 min by row count); a 25 % slower box still fits (519 ≤ 0.95 × 550). Expected 443 run min = **$21.93** billed
($0.0495) / **$27.91** quoted ($0.063), + build ~15 min. Worst case (60 + 2 × 612) × 0.0495 = **$63.56**; quoted max
612 × 0.063 = $38.56. Parent arm adds ~141 min (0.373 s / 1k tokens) ≈ $7.

**Submitting (the signed URL never goes into git).** Package from a pushed sha
(`python -m jobs.common.launch m2_first_cell --mode fulldev_eval --sha <sha> --out <dir> --data <ssdm-data>`), mint
the artifact link (`get_artifact art-bm3e6`, valid ~7 days), put it into the PACKAGED `<dir>/submit.json`
`calls[submit_job].args.env.CKPT_URL` (replacing the `<...>` placeholder; that file is outside the repo), then
`python -I jobs/common/sf_submit.py <dir>`. The job refuses anything but `https://` and checks the tar's sha256
against the pin before reading two members.

**Landing.** Extract the job's last phase tgz, then `python -I -X utf8 analysis/land_fulldev.py --art <dir> --job
<id> --artifact <art>`: `analysis.paired_vs_clef` in `full` and `full_excl51` row modes against the two pinned
comparator files; `beats_both` per mode, ECE, and a re-check of the job's own score.

### Mode `hijack_eval`: hijack + D1-A steer on a saved checkpoint (DRAFT, NOT SUBMITTED)

`job_params_hijack_eval.json` (BOARD H1): eval-only, no training. Loads D1 (job-c7pq2 art-bm3e6) through the shared
`checkpoint_loader install`, then `train.m2_gates --stage hijack` (45 steer_dev clusters x 8 requests x (clean + 5
hijack + 4 D1-A) = 3,600 units per arm), paired child - parent via `safety_eval.metrics.compare_arms`. Caps 15 / 5 /
10 / 25 / 50 / 5 + 7 = 117 min. Expected ~72 run min = $3.56 (+ build ~15 min = $4.31); quoted one attempt $7.37.
**Worst case** (60 + 2 x 117) x 0.0495 = **$14.55** (cap $15). Full derivation in the params file's `_comment`.

### Mode `fulldev_smoke`: the same pipeline on 250 pre-declared rows, both arms (DRAFT, NOT SUBMITTED)

`job_params_fulldev_smoke.json` = fulldev_eval with `"rows": "smoke"` and `"fulldev_parent_arm": true`. Rows:
`fulldev_smoke_rows.json` (`fulldev_eval smoke-build`; DEV manifest + RWKV7 row lengths only, rebuilt byte for byte
by the tests): whole kit groups; the longest DEV row (47,509 tokens) plus 11 more of the 68 rows over 16,384 at
evenly spaced length ranks (8 of the comparators' 51, 4 RWKV7-only), then one whole group per track in
sha256(seed:catalog:group) order, round-robin over the 83 tracks that have short groups, to 250 rows (732,499
tokens). The report scores the subset in process with the frozen functions (`harness.score.decompose`,
`harness.bootstrap`, `harness.calibration`; the CLI scores whole slices only) and says so. `land_fulldev.py`
detects a smoke results file and lands it in the `fulldev_smoke` row mode of `paired_vs_clef`, labelled a SUBSET.
Caps 25 / 5 / 15 / 70 / 25 / 10 + 7 = 157 min. Expected: child 12.7 scoring + 30 fixed, parent 4.6 + 5; run 73 min
= **$3.61** billed / $4.60 quoted, + build ~15 min ($4.36). Worst case $18.51.

**d1_fast's throughput flags do not apply to scoring.** `row_plan` / `cont_plan` regroup rows inside a TRAINING step
(`train.trainer`), and `compile_blocks` compiles the block body inside the activation checkpoint, which
`heads/act_ckpt` bypasses under `no_grad` (evaluation runs the plain block). Scoring is one request at a time
through `DecisionModelBackend.score`; none of the three is on that path, so the full-run estimate stays 421 min.
(They also could not be switched on here: the loader requires the cell config to EQUAL the checkpoint's cell.json.)
The measured gap that does matter: D1's child scores at 1.037 s / 1k tokens vs 0.373 for the stock parent (2.8x).

## Params `"rows": "proxy600"`: screen a cell on the pre-declared 644-row Index proxy

`jobs/common/dev_proxy600.json` (ids sha `7d42c398…85e6`, pinned in `jobs.common.dev_proxy.PROXIES`): a stratified
subset of the 2,008-row m1 subset (all 38 benchmarks x length bins, whole source cases, allocation by Index weight,
min 6 rows per benchmark, seed `proxy-v1`), chosen by leave-one-system-out over 15 scored systems
(`analysis/proxy_validate.py`, receipt `receipts/2026-10-08-B-dev-proxy600/`). Base / training modes: `"rows":
"proxy600"` makes `train.m2_gates --index-proxy proxy600` score the child's Index stage on the proxy rows only; the
parent arm stays verified on the full subset (unchanged identity / seed pins) and is paired on the proxy rows. G12 is
unchanged; g13 refuses a proxy. Mode `fulldev_eval` takes the same value (`--rows-select proxy600`). Measured on D1
(job-c7pq2): the Index stage drops from 71.2 min (61.3 rows + 9.9 order diagnostic) to ~29.6 (26.0 + 3.7; 55 of the
160 diagnostic rows), i.e. -41.6 min = -$2.06 batch / -$2.62 quoted per cell; the evals phase 178 -> ~136 min (the
G12 stage, 82 min, and the 18 min fit + card are untouched). The preflight / reprice still price the full subset
(an upper bound). A SCREEN: a winner gets the full subset + full DEV.

## Params `"stage": "screen"`: train, then screen on proxy600 (no G12, no temperature fit, no card)

The default is `"stage": "gates"`, and that path is unchanged. For any training mode (not `eval_only`), the evals and
report phases are replaced by `jobs.m2_first_cell.screen`:
- **score:** the child's untempered arm decider runs on the 644 proxy600 rows at the trained input cap. A $0 plan check
  confirms the child refuses exactly the rows the parent refused.
- **report:** writes `screen.json`, validated by `screen:valid_result`. It holds the child, parent and reference proxy
  Index with 95% CIs, and both paired differences (child-parent, child-reference), from `harness.score.decompose` and
  `bootstrap.compare`. The verdict is **DROP if child-reference < -2.5** (point estimate), else **PROMOTE** to the full
  subset and full DEV.

Reference: `screen_reference` gives an artifact id plus a sha chain: `artifact_sha256` → `member` (+`member_sha256`) →
`inner_member` (+`units_sha256`). The default is D1, art-bm3e6 `./phases/final.tgz` `out/units-child-index.json`.
- It is found in setup, before training: either the shipped `$DATA/<shipped>` file with the pinned sha, or the
  downloaded artifact from `$<url_env>`.
- It must cover every proxy row, or setup fails with exit 10.

Mode params need these changes:
- `"stage": "screen"`.
- In `phase_caps_minutes`, replace `"evals"` with `"screen": 60`.
- An explicit `screen_reference` block.
- A `context.data` entry `{"src": "m2/screen_reference/art-bm3e6/units-child-index.json", "dest":
  "screen_reference/units-child-index.json", "pin_json": [<params>, "screen_reference", "units_sha256"]}`.

The parent arm phase and its seed are unchanged. Expected eval time is about 41 min: 26.0 min of child proxy scoring
(D1 measured), plus 15 min budgeted for load and lineage, plus about 2 min of report. That is about $2.03 batch or
$2.58 quoted, against the gates path's 178 min.

### Screen readout and reference presets (every arm under ONE readout)

`"screen_readout"`:
- Default `"native"`: the cell config's own template and layout, so existing params behave as before.
- Or one of `v1` (jrt-rwkv-g1-v1, ordered), `v2f` or `v3w` (jrt-rwkv-g1-v3w, g1v2). The child is scored under that
  readout over the same loaded model (as `readout_ab` does).
- A template this code cannot render fails in setup with exit 5. The v3w templates live on the readout branches
  (`claude/b-train-v3w-template`, `stack/rwkv7/ckpt-readout` and the d10w/d12 cells), not yet on agent-main.

`"screen_reference"` takes a preset name or a full block. Every block carries a `readout_template`.
- `d1_v1`: job-c7pq2 art-bm3e6, `units-child-index.json`, sha 545dbb94….
- `d1_v3w`: job-39gnn art-zskmx (artifact sha dd017eb0…), member `./phases/final.tgz` (1ad5f899…), inner
  `out/readout_ab/child.v3w.results.jsonl` (941af29e…). Score 38.67 [35.50, 41.73].
- The default depends on the readout: v1 uses `d1_v1`, v3w uses `d1_v3w`.

`"screen_parent"` is `"in_job"` or the preset `stock_v3w`:
- `stock_v3w`: job-fgu2c art-ivu28 (aeef838b…), member `./phases/final.tgz` (6cde2da7…), inner
  `out/index/v3w/results.jsonl` (a52a68eb…). Score 31.07. Its 8 refused rows are the same as the D1 + v3w child's.
- The default is the in-job arm when its prompt template (from the arm revision `prompt=…`) is the screen's.
  Otherwise it is the readout's preset, which is `stock_v3w` for v3w.

Fail closed (exit 10):
- A reference or parent whose `readout_template` is not the screen's.
- An in-job parent arm whose prompt template is not the screen's.
- A pinned file that misses any proxy row.

Ship the staged copies through `context.data`:
- `m2/screen_reference/art-zskmx/child.v3w.results.jsonl` to `screen_reference/art-zskmx/child.v3w.results.jsonl`.
- `m2/screen_parent/art-ivu28/v3w.results.jsonl` to `screen_parent/art-ivu28/v3w.results.jsonl`.
- Without them, the job downloads each artifact from `$REF_URL` / `$PARENT_URL` and checks the full sha chain.

Reference point on the proxy: D1 + v3w child 37.71 against stock v3w parent 27.32, a difference of +10.39
[5.58, 14.86].

The 26 min child estimate is D1's v1 rate. The v3w prompts are longer, so the screen cap (60 min) keeps margin.

## Params `"chain"`: one training run over several jobs (`train/chain.py`)

An SF job is capped at 899 min and a single-job cell trains ~380 of them. A chained cell is ONE run (one shared cell
config: `max_minutes` = the TOTAL training clock, `max_steps` the global cap) carried over N jobs. The platform keeps
one checkpoint slot per job (a new job starts empty), so the state travels in each link's /output artifact.

- Params per link: `"cell_config"` = the shared config (not `m2_first_cell_rwkv7_<mode>.json`) and
  `"chain": {"link": k, "of": N, "from": {"job", "artifact", "artifact_sha256", "manifest_sha256", "url_env"}}`
  (`from` for k >= 2: `artifact_sha256` from `get_job` artifact.sha256, `manifest_sha256` from link k-1's declared
  result `decision.manifest_sha256`; the presigned `get_artifact` URL goes into the submit env at submit time).
  `python -m train.chain plan --base-params <single-job params> --of N` prints every link's phase caps (caps +
  reserve = 899) and training minutes.
- Link k < N: setup (+ `train.chain check`: the export must fit the 5 GiB artifact), guard, heldout (and the draw's
  attack / doc-overlap gates in the cell) as in every cell job; k >= 2: phase `chain` installs link k-1 (exit 10 on
  any pin / chain id / link order / cell config mismatch); cell trains until the JOB's deadline (max_duration - time
  used - post-cell caps - reserve, minus 8 min tail); phase `chain_export` writes /output/chain (CHAIN.json, the
  latest full .pt, the best .pt full or trainable-only, latest.json, cell.json); result = `out/chain_link.json`. No
  parent arm, no screen / evals, no inference export.
- Link N: install link N-1, train the rest of the global clock (to its deadline at most), then the mode's stage.
- Resume is exact: config hash + lineage at `Trainer.load` (exit 29), cosine horizon fixed once in link 1 from the
  global clock, data cursor / optimizer / scheduler / RNG / dev cards / best / collapse-guard window in the checkpoint,
  `train_clock` and the eval cards in cell.json (`tests/test_chain.py`: a 2-link chain equals one run bit for bit).
- Size (D18 recipe, joint_record head 5,466,114 params): r64 = 293,300,226 trainable -> latest 3.52 GB + slim best
  1.17 GB = 4.69 GB, fits; r256 = 1,156,802,562 -> 13.9 GB latest alone, refused in setup (needs a bucket transport).
  A best that is carried trainable-only scores identically; only a collapse-guard rollback to it keeps the current
  AdamW moments (recorded in the collapse event).

## Re-pricing 2026-10-09: H100 batch $0.0495 -> $0.063/min

SF Compute raised the H100 list price $3.96 -> $5.04/h on 2026-10-08; jobs now bill $0.063 per GPU-minute (Mon-Thu and weekends; Fridays $0.05628) and the preflight quotes max_duration x $0.063. Evidence and history: `jobs/common/rates.py`. The dollar figures above this section are at the old $0.0495/min and stay as written (receipts and past decisions used them); the numbers below are the current ones (`python -m jobs.common.costs`). Where a hard cap was raised, it was raised by the same ratio as the rate (x 0.063/0.0495 = x 14/11, rounded up to the cent), so the cap still buys the same minutes.

| Params | Worst case (build + 2 attempts) | Hard cap | Expected | One attempt (quoted max_cost) |
|---|---|---|---|---|
| `job_params.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_d1_fast.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_d3_r1_gold.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_d3_v2f_fast.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_d3_v2f_gold.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_d3_v3w_fast.json` | $84.04 (was $66.03) | $89.10 (was $70.00) | $33.26 (was $26.14) | $40.13 (was $31.53) |
| `job_params_fulldev_eval.json` | $80.89 (was $63.56) | $82.73 (was $65.00) | $28.85 (was $22.67) | $38.56 (was $30.29) |
| `job_params_fulldev_smoke.json` | $23.56 (was $18.51) | $25.46 (was $20.00) | $5.54 (was $4.36) | $9.89 (was $7.77) |
| `job_params_hijack_eval.json` | $18.52 (was $14.55) | $19.10 (was $15.00) | $5.48 (was $4.31) | $7.37 (was $5.79) |

## D20: chained cell `d20_v4_chain` (3 links; link 1 = mode `d20_v4_chain_l1`)

One training run of the D19 recipe (RWKV7-G1k-13.3B attention-free, LoRA r64, D13 joint_record head + whole-record dev
card, anti-collapse safe keys, v3w readout; job-fi5xf screen 48.61) over three SF jobs (`train/chain.py`, section above).
Shared cell config `m2_first_cell_rwkv7_d20_v4_chain.json` (chain id = sha256 of its canonical JSON; every link uses it
unchanged) = D19's cell with ONLY:
- data: `bschema-draw-v4-nophish` = B's `bschema-draw-v4` (origin/claude/b-schema-v4 @0a204f99, pointer a4c505ac...,
  262,468 rows) minus every `phish_curated` row (agent-a: drop it by default) via `scripts/b_schema_subtract.py
  --drop-source phish_curated` (subtraction only, parent lines byte-for-byte; receipt
  `receipts/2026-10-08-B1-schema/bschema-draw-v4-nophish/subtract-report.json`) = **258,798 rows**, pointer
  `data/manifests/bschema-draw-v4-nophish.pointer.json` (0e15aec5...), raw files pinned by
  `data_d20_bschema_v4_nophish.sha256`. No doc_clean / attack_clean / heldout pins: every guard must find zero.
  CPU 2026-10-10 (streaming, this branch): heldout gate (train.heldout D1-B 0 / attack_guard 0 / doc_overlap_guard 0 incl. the #243 JevBench guard 0 doc + 0 exact) PASS in 207 s; train_set.draw 258,798 of 258,798 drawn, drawn-ids sha256 a08ba5e3954e4346dcb6cb810a1f03c43fcdba4b7ba0b0b1d962ea34cff84662, 0 content drift, 0 phish_curated (receipts `trainer-draw.json`, `heldout-gate.json` beside the subtract report).
- clock: `max_minutes` 1694 = the TOTAL training clock, `max_steps` 24,000 (cap only; the clock binds). Cosine horizon
  projected once in link 1 from the global clock and carried in the checkpoint.

Plan (`python -X utf8 -m train.chain plan --base-params jobs/m2_first_cell/job_params_d19_v3w_v3b_joint.json --of 3`;
base = D19's single-job params, data being the only recipe change):

| Link | Mode | Caps (+ reserve 7 = 899) | Train min floor / expected | Runs |
|---|---|---|---|---|
| 1 | `d20_v4_chain_l1` | setup 25, guard 5, heldout 15, cell 837, chain_export 10 | 676 / 695 | guards + train + export |
| 2 | `d20_v4_chain_l2` | setup 25, guard 5, heldout 15, chain 15, cell 822, chain_export 10 | 661 / 690 | install 1 + train + export |
| 3 | `d20_v4_chain_l3` | setup 25, guard 5, heldout 15, chain 15, parent_g12 85, parent_index 65, cell 518, screen 159, report 5 | 357 / 423 | install 2 + train the rest + screen |

`max_minutes` 1694 = the sum of the floors (`max_minutes_guaranteed`; expected sum 1,808), so the schedule completes
inside link 3 even when every earlier phase runs to its cap. 3 x the middle link (3 x 661 = 1,983) was considered and
rejected: link 3 would need 655 training minutes and its whole cell cap is 518. Only link 3 screens (proxy600 v3w vs
d1_v3w / stock_v3w, as D19); links 1-2 have no parent arm, screen or evals.

Price per link at $0.063/min: **worst case** (60 + 2 x 899) x 0.063 = **$117.05** (hard cap $120.91 = D19's $95 x
14/11); platform hold 899 x $0.063 = $56.64; link 1 expected 883 min = $55.63. Chain total ~3 x $56.64 = $169.92 held.

Hand-off (links 2 and 3, packaged by whoever continues): when link k ends (`out/chain_link.json` verdict `link_done`),
copy `job_params_d20_v4_chain_l1.json` to `job_params_d20_v4_chain_l<k+1>.json`; set `mode`, `job`, `submit.key`,
`submit.command` to `d20_v4_chain_l<k+1>`; `chain` = {"link": k+1, "of": 3, "from": {"job": <link k job id>,
"artifact": <its art-id>, "artifact_sha256": <get_job artifact.sha256>, "manifest_sha256": <link k's
chain_link.json manifest_sha256>, "url_env": "D20_CHAIN_URL"}}; caps from the table; register the file in
`jobs/DECISION_TREE.md`; put the presigned `get_artifact` URL into the submit env as `D20_CHAIN_URL` at submit time
(never in git). The cell config never changes.

Link 1 SUBMITTED 2026-10-10 as **job-7d3u9** (branch sha 1441374, package tar sha256 892247b0..., idempotency key m2-first-cell-d20-v4-chain-l1-892247b04520, quoted max_cost $56.64). Link 2 takes its `chain.from` from job-7d3u9: job get -> artifact id + artifact.sha256; out/chain_link.json -> manifest_sha256. The derived build `ssdm-data/m2/build/bschema-draw-v4-nophish/` stays on the laptop for the link 2 / 3 packages (rebuild: the subtract command above, byte-identical, checked against `data_d20_bschema_v4_nophish.sha256`).

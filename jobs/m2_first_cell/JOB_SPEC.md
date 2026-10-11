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
2. `open_mission("m2-first-cell")`; `set_workspace_cap(MTD + 70)`.

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
2. Spend issue + LEDGER row; `set_workspace_cap(MTD + 70)` (hard cap $70; worst case $66.03); credit expiry 2026-10-27.
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

## Mode `ragtruth_rephrase`: the pre-registered RAGTruth statement rephrase (RT1), train-split check on D13 + D19

B-PREREG-RAGTRUTH-REPHRASE (state PREREG.md @0683fa7c). job-3zmj2 (`ragtruth_polarity`) found D13's RAGTruth ranking
inverted on 300 RAGTruth TRAIN items (AUROC 0.418 [0.353, 0.481]; Summary 0.33). RT1 (`decide/ragtruth_rephrase.py`)
rewrites the catalog-59 question at READOUT time, by exact content match only (type noul + instructions + criteria),
keeping the type, the option ids and the polarity (`true` = hallucinated), so p(true) is still p(hallucinated).
`eval_only: "ragtruth_rephrase"`: setup + guard as usual, plus D19's train set verified against its own cell config
and the `ragtruth_rephrase estimate` preflight; then per arm (params `arms`): `checkpoint_loader install --block`
(D13 = `checkpoint`, art-cr9a2 best1600; D19 = `checkpoint_d19`, art-p3gpi best2800), and ONE model load
(`fulldev_eval.load_child`, v3w untempered, the arm's own train dir for the lineage check) scoring every wording of the
arm (D13: rt1; D19: original + rt1) under the rate guard (`--setup-cap-min`). D13's original-wording rows are job-3zmj2's
`d13.results.jsonl` (bab92081..., shipped). Report: AUROC + bootstrap CI per arm x wording, paired dAUROC (rt1 -
original), per track, in-draw / out-of-draw (175 of the 300 items are in D19's bschema-draw-v3b draw with the original
wording + labels: `ragtruth_rephrase_d19_indraw.json`, rebuilt from the shipped manifest), and the rule
(`ragtruth_rephrase.decide_arm`): PASS iff RT1's lower bound > 0.5; D19 also needs RT1's out-of-draw point AUROC > 0.5.
Only a passing arm gets the DEV-minus-proxy600 stage (a second job). Pricing from measured rates (D13 load 6.75 min,
D19 load 21.75 min, 3.4 min per 300-item wording): expected 46.8 min + build; caps setup 7 / guard 2 / checkpoint 3 /
score_d13 14 / checkpoint_d19 4 / score_d19 39 / report 3 + reserve 7 = 79; quoted hold $4.98, worst case
(60 + 2 x 79) x 0.0495 = $10.79 (cap $15). CKPT_URL = art-cr9a2, CKPT_URL_D19 = art-p3gpi at submit.

### Mode `ragtruth_rephrase_d19`: the D19 arm alone (after job-862xw)

job-862xw (both arms) stopped at score_d13 with exit 10: D13's cell pins attack_clean templates_sha256 930863...,
this lineage's attack guard v2 (1e647e00) hashes 5e2d6e..., so D13 cannot load here. D19 runs alone on this lineage
(original + rt1; block `checkpoint` = art-p3gpi best2800; cell config = D19's training cell; train dir
bschema-draw-v3b); D13 (rt1) runs on `claude/ev-ragtruth-rephrase-d13` (from a1bfa2b, job-3zmj2's lineage). The
runner's ragtruth_rephrase branch loops over params `arms`. Caps setup 7 / guard 2 / checkpoint 4 / score_d19 39 /
report 3 + reserve 7 = 62; quoted hold $3.91, worst case (60 + 2 x 62) x 0.0495 = $9.11 (cap $15).

### Mode `ragtruth_rephrase_d19_dev`: stage 2, D19 arm

Params `"rows": "dev_minus_proxy600"`: the 255 DEV catalog-59 rows outside proxy600 (ids sha 5a1d4896..., pinned in
`dev_stage`) replace the TRAIN items; `report-dev` (schema `.dev`, validator `valid_any`) gives AUROC per wording and the
reproduction of the original rows against the landed full DEV (shipped subset, pinned). The pre-registered Index delta
is computed at landing (`analysis/ragtruth_rephrase_land.py`). Same caps as `ragtruth_rephrase_d19`: worst case $9.11.

## Mode `devcal_d19_v3w`: D19's per-question probabilities on its own cohort-dev pool (temperature fit)

B-PREREG-D19-TEMP (state PREREG.md, addendum 1). art-p3gpi keeps D19's dev cards as aggregates only, and D19 drew the
whole bschema-draw-v3b pool (n_drawn = n_pool = 253,046), so the one temperature T is fitted on the 804 m2-dev-cohort-v2
dev items D19 itself built (cell.json dev_card_sample n_pool; cohort-excluded from training). `eval_only: "devcal"`:
setup + guard as fulldev, the same checkpoint install (job-fi5xf art-p3gpi best export, sha 31062f48..., select best),
then `jobs.m2_first_cell.devcal_eval score`: the checkpoint loads as `fulldev_eval.load_child` loads it (cohort, guard,
training-item lineage, Trainer.load); the eval items are rebuilt as `train.run_cell` builds them (cohort dev ids, eval
max_k 16, T_max filter); exit 10 unless the pool size, the dev-card sample (ids, sha, per-source) equal cell.json and no
pool id or record is a training id. `train.evalcard.run_card(keep_units=True)` on the pool and on the dev-card sample
alone; units = raw `decision_probs` + labels (native template jrt-rwkv-g1-v3w / g1v2 = the screen's v3w readout).
Landing (CPU): `python -I -m jobs.m2_first_cell.devcal_eval fit --score-dir <phase devcal out> --out T.json`: scalar T
in [0.25, 4] minimising mean NLL over all pool questions; reported before/after on the pool, the 200 dev-card rows and
the 604 others; usable only if the rescored dev card reproduces card-step0002800 (NLL 0.3376, acc 0.890) within 0.01.
Caps setup 12 / guard 3 / checkpoint 7 / devcal 50 + reserve 7 = max_duration 79; expected 45 min (devcal 28 = 21.6
load + lineage as job-fi5xf's screen fixed + 804- and 200-row cards, 237 s / 68 s in cell.json); quoted hold $4.98,
worst case $10.79 (cap $15). Key `m2-first-cell-devcal-d19-v3w-fi5xf-best2800-20261010a`; CKPT_URL = the art-p3gpi URL
at submit.

## Modes `fulldev_eval_d19_v3w_s1of2` / `fulldev_eval_d19_v3w_s2of2`: D19 on the full DEV in two shards

The D13 sharded modes above with only the checkpoint, the cell config, the lineage data and the measured rate changed.
Checkpoint: job-fi5xf (D19 d19_v3w_v3b_joint, 84e8113) artifact art-p3gpi (sha b898dbc4...), its BEST export
`trainable-step0002800.safetensors` (sha 31062f48..., sidecar 6d1f1adf...; `"select": "best"`: latest.json's best
pointer must name it with that sha; its sidecar's source ckpt-step0002800.pt sha 8cdfef6b... is exactly the checkpoint
the 48.61 [44.10, 53.08] proxy600 screen scored). Cell config = D19's training cell byte for byte; context data = D19's
(bschema-draw-v3b set + doc-overlap index, which the lineage check's draw needs) minus the screen arms, plus the pinned
cell.json. Scoring = the screen's: `score --readout v3w` -> `load_child(temperature=False, readout=(jrt-rwkv-g1-v3w,
g1v2))`, the call `jobs.m2_first_cell.screen score` makes (v3w is D19's native template, so the readout record says
native). `fulldev_eval.estimate` now applies the v3w length ratio whenever the readout template is not the lengths
file's (v1), native or not. Pricing: 11.32 M v1 tokens x 0.9004 x 1.2094 s / 1k (job-fi5xf screen: 1,613.35 s over 644
proxy600 rows) = 205.4 + 33 fixed + 21.0 plan = 259.4 min under the 336-min fulldev cap (29.6 % headroom; the guard
tolerates a box 29.1 % slower); max_duration 370, quoted hold $23.31 each, worst case $39.60 each (cap $60). Keys
`m2-first-cell-fulldev-eval-d19-v3w-s<K>of2-fi5xf-best2800-20261010a`; CKPT_URL = the art-p3gpi URL at submit.

## Modes `fulldev_eval_d13_v3w_s1of2` / `fulldev_eval_d13_v3w_s2of2`: the v3w full DEV in two shards (DRAFT, NOT SUBMITTED)

`fulldev_eval_d13_v3w` with params `"rows": "shard<K>of2"`, nothing else about the model, checkpoint or readout changed.
Why: the one-job run expects ~500 of its 536-min phase cap (the live guard stops past 0.95 x cap = a box only ~1.8 %
slower than the screen) and a stopped run loses every scored row. Shards: `jobs/m2_first_cell/fulldev_shards_2.json`
(`fulldev_eval shard-build --n 2`): clusters = `harness.cases.case_ids` (the frozen bootstrap's cluster unit) never
split; initial shard sha256(`fulldev-shard-v1:<case_id>`) mod 2; rebalanced by RWKV7 token count (1 move; 837-token
gap). Shard 1: 6,109 rows / 4,191 clusters / 11,315,055 tokens, ids_sha256 `5aa8ef1531b4c171f34b6b8174fc04c3ec3e8c28eb4ba396663e605432bb86fe`;
shard 2: 5,995 / 4,282 / 11,315,892, `f523934e7b85a32317b4d59e55bfdc39dbd9de338ac90c9c440234ab2c51202c`; union = the
12,104 DEV ids (`4f310d28996ca91e932c497a00c54abe4ced1277e3dd63c21a4ff4366fd7d030`). The job rebuilds the plan from its
DEV rows + lengths and exits 10 on any difference. Pricing (params `plan_minutes_full` 42: the plan pass tokenises only
the shard's rows): 212.5 scoring + 33 fixed + 21.0 plan = 266.5 min under a 336-min fulldev cap (26.1 % headroom; the
guard tolerates a box 19.8 % slower); max_duration 370 (setup 12 / guard 3 / checkpoint 7 / fulldev 336 / report 5 /
reserve 7), quoted hold $23.31 each, worst case (60 + 2 x 370) x 0.0495 = $39.60 each (cap $60).
Partial rows: on ANY scoring exit (timeout, guard stop, low disk, and in the EXIT trap) `$CK/fulldev/child.partial.jsonl`
+ `child.partial.identity.json` are copied into the phase export; a follow-up job of the SAME sha sets env
`SEED_PARTIAL_URL` (that phase tgz) + `SEED_PARTIAL_SHA256`, and `score --seed-partial` reuses the rows only if the
checkpoint sha, readout fingerprint, code sha, arm, rows mode and rows ids hash all match and every row is in the shard.
Landing: `python -X utf8 -m jobs.m2_first_cell.fulldev_eval merge <s1>/fulldev <s2>/fulldev --rows
$SSDM_DATA/slices/dev-rows.jsonl.gz --out <dir>/fulldev/child.results.jsonl` (refuses unless one shard per K, the
pre-declared shard id hashes reproduce, the results bytes match, the identities agree, disjoint + exactly the DEV ids;
the output is byte-identical to an unsharded run's file, ids_sha256 = the full-DEV ids hash), then
`analysis/land_fulldev.py --candidate <dir>/fulldev/child.results.jsonl`. Each shard's own report is a SHARD SUBSET.

## Modes `fulldev_eval_d13` / `fulldev_eval_d13_v3w`: D13 on the FULL Index DEV (DRAFT, NOT SUBMITTED)

Mode `fulldev_eval` (below) on D13 = job-hjwpq (d13_joint_safe_nofla, the attention-free joint_record head; proxy600
screen 41.07 [37.57, 45.22] under v3w vs D1+v3w 37.71), from D13's own lineage (8d92f04) + the job-4ha7t plan-pass
beat + the job-afpc2 worker-disk fix. Checkpoint: art-cr9a2's BEST export (`trainable-step0001600.safetensors`,
`"select": "best"`: latest.json's best pointer must name it with the pinned sha; its sidecar's source_ckpt sha
b6506c35... is the checkpoint the screen scored). params `readout`: `native` (jrt-rwkv-g1-v1 / ordered, served with the
cohort temperature fit) or `v3w` (load_child's readout override, untempered). The joint head is served as trained: one
model(rows) pass per request, questions grouped by record. The report adds the 2,008 subset + proxy600 Index from the
same predictions. Worst case: native $68.51 (cap $70), v3w $74.25 (cap $75; 2026-10-09c: max_duration 720 = setup 12 / guard 3 / checkpoint 7 / fulldev 686 / report 5 / reserve 7, quoted hold $45.36; the rate guard tolerates a box ~30 % slower than the screen); two jobs, since both readouts in one job
(~1,080 min) exceed the 900-min ceiling. Exit 16 (job-afpc2): the worker's shared disk under 6 GiB free; the child
declared its slot and yielded.

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

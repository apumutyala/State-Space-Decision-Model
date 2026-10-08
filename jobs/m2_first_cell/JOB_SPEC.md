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

## Mode `readout_ab_d3`: a saved g1v2 checkpoint under several readouts, one load (DRAFT, NOT SUBMITTED)

**Question.** Does the trained D3 cell (job-93fpy, `d3_v2f_gold`: LoRA r64 + cache-fork head trained on `jrt-rwkv-g1-v2f`)
gain from the R3 readouts the stock parent gained from (job-fgu2c: v2f 29.10, v3w 31.07 = +1.97 [+0.17, +3.80], v3p
30.60)? Files: `job_params_readout_ab_d3.json`, `m2_first_cell_rwkv7_readout_ab_d3.json` (= `d3_v2f_gold`'s cell
config byte for byte), driver `readout_ab.py`, layout readouts `heads/layout.py` `G1_EVAL_TEMPLATES`, equivalence
check `scripts/check_readout_ab_equivalence.py`, pins `scripts/fill_ckpt_pins.py`.

**What runs.** setup + guard as usual, `checkpoint` (shared `checkpoint_loader install`), `readout_ab` (one process:
`fulldev_eval.load_child` = `Trainer.load` config hash + lineage + the cohort temperature fit, then every readout in
params `readouts` over the SAME model object; first readout = the cell's training template, refused otherwise; a $0
plan pass per readout; per-row resume from `$CK/readout_ab`; live weighted rate guard over all readouts), optional
`parent_ab` (stock families G1V3Backend v2f / words / permute), `report` (`jobs.common.dev_subset score|ece|compare`
per readout, paired readout − control, served ECE under the one fitted temperature).

**Readout validity on a trained head.** `FinalStateFork` scores option k from the hidden state after
`prompt + <answer token k>` through one shared MLP + a symmetric summary: no per-code parameters, so any readout with
the "ANSWER:" slot is well defined. v3p / v2fp fork each canonical option with the code PRESENTED for it
(`question_codes`), v3w with its id's first answer token; logits return keyed by option id. Every row is byte-equal to
the stock readout's prompt and answer tokens (`check_readout_ab_equivalence.py`, G1k tokenizer, 6 rows per benchmark,
1,312 question rows per readout, 0 mismatches for v3w / v3p / v2fp). v3w is the furthest from training (the head
never saw states after word tokens): measured, not assumed.

**Phases.** setup 25 / guard 5 / checkpoint 15 / readout_ab 215 / report 10 / reserve 7 = 277 min. **Sizing**: D1's
measured 32.8 child rows/min (job-c7pq2, on v1 prompts) x 2,008 rows per readout x the readout's prompt-length ratio
to v1 (v2f 0.8984, v3w 0.9004, v3p 0.9032) + 15 fixed = 125.1 min for v2f + v3w; expected ~145 run min = **$7.18**
billed (+ build ~15 min = $7.93). A third readout adds ~55 min (~$2.74; the 215 cap fits three readouts). d1_fast's flags are training-step execution flags and do
not apply to scoring. Worst case (60 + 2 x 277) x 0.0495 = **$30.39**; hard cap $32. `readout_ab_d1dry` = the same on
D1 art-bm3e6 (`allow_unmatched_control`, plumbing only; superseded), same caps and worst case.

**Mode `readout_ab_d1`** (priority, coordinator 2026-10-08): the same driver on D1 (job-c7pq2 art-bm3e6, pins =
`job_params_hijack_eval.json`) with readouts `v1` (D1's own `jrt-rwkv-g1-v1` / layout `ordered` = the matched
control; its in-job index arm scored 35.12) and `v3w` (layout g1v2 R3 renderer over the same model). One generic
readout per results file (never a per-family pick: tuning on DEV). Per readout: Index + CI, ECE, paired v3w − v1,
per-row latency (Clef row definition) and tokens/s (planned input tokens / decide wall over ok rows). Expected 131.3
scoring min, ~151 run min = **$7.49** billed (+ build = $8.23); adding v2f -> $10.21. Worst case **$30.39**. Package:
`python -m jobs.common.launch m2_first_cell --mode readout_ab_d1 --sha <sha> --out <dir> --data <ssdm-data>`, then
`CKPT_URL` into the packaged submit.json env (as fulldev_eval).

**Running it on D3.** Download the D3 artifact (`get_artifact`), then
`python -I -X utf8 scripts/fill_ckpt_pins.py <tar | dir | job-93fpy> --params jobs/m2_first_cell/job_params_readout_ab_d3.json --artifact <art> --write`
(fills every pin, copies the run's cell.json to `$SSDM_DATA/m2/checkpoints/job-93fpy/cell.json`, refuses a config
mismatch), commit + push, package (`python -m jobs.common.launch m2_first_cell --mode readout_ab_d3 --sha <sha> --out
<dir> --data <ssdm-data>`), put the signed URL into the PACKAGED `<dir>/submit.json` env `CKPT_URL`, submit.


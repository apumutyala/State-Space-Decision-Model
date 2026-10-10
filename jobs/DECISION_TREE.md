# m1 decision tree: qualification outcome -> next job (spec only, nothing submitted)

Track D's next RWKV7 line is registered as `jobs/rwkv7_code_label_h100/job_params.json`
(`jobs/rwkv7_code_label_h100/JOB_SPEC.md`): matched code-label v1/v2 on the exact 2,008
DEV IDs, all eight exhausted rows retained. Launch disabled pending merged proof,
independent review and explicit spend-issue approval. Exact admitted workload projects
212.071453 minutes at the recorded small-call reference; the hard budget uses a
conservative inherited planning stress estimate of 474.984510 minutes (old 64x64 tails,
not current v2 packs; historical small-call projection 293.991278 minutes).
Real-shape H100 memory/parity and probe-derived rate
gates precede scoring. Runtime 710 min, build 60 min, two-attempt conservative
worst case/hard cap **$93.24**, published-rate bound **$90.06**, expected **$33.78**,
workspace increment $94 (re-priced 2026-10-09 at $0.063/min; at the old $0.0495: $73.26 / $70.89 / $26.54 / $74).
This is derived within the $250 System 1 allocation; no fixed qualification-dollar
ceiling applies. Completion produces an integrity receipt only: no promotion, Clef win,
safety floors or automatic follow-on spend. Failed feasibility keeps permitted $0 work moving.

Every job below still needs its own `spend` issue, a LEDGER planned row, `set_workspace_cap` = MTD + its hard cap,
and a `validate_job` re-run against that cap before submit (same preconditions as `jobs/qual/JOB_SPEC.md`). The
numbers are computed by `python -m jobs.common.costs` and checked by `tests/test_m1_next_jobs.py`.

**Worst case** = (build_timeout + 2 attempts x max_duration) x $0.063/min (build counted as billable; one platform
retry assumed even at `max_restarts=0`). **Expected** = sum of the expected phase minutes x the same rate.

**Re-priced 2026-10-09** (`jobs/common/rates.py`): the H100 list price rose $3.96 -> $5.04/h on 2026-10-08, so the billed batch rate is $0.063/min (was $0.0495; Fridays $0.05628). The worst-case, hard-cap and `~$` expected figures below are at $0.063; caps that bound were raised x 14/11 (the rate ratio), except the three $20 authorised lines (qual, m2_pilot, m2_parent_arm), which were re-authorised 2026-10-09 ($20 → $24) instead. One-attempt and other prose figures inside the cells, and every actual / receipt figure, are as originally written (pre-2026-10-08 rate).

| Job | Spec | Worst case | Hard cap (workspace cap) | Expected |
|---|---|---|---|---|
| qual (rung-2 qualification, unchanged) | `jobs/qual/` | $23.94 | $24 (re-authorised 2026-10-09 ($20 → $24)) | <= $11.34 (one attempt, validate_job) |
| qual_c_only (phase-C re-run after the job-c4zkm 9B OOM; A and B passed there; env `QUAL_MODE=c_only`) | `jobs/qual/job_params_c_only.json` | $18.90 | $20 | <= $8.82 (one attempt; $10.08 with the 20-min build) |
| probe9b (one-layer 9B conversion probe; re-priced from the job-4j3vu measured rates, issue #68) | `jobs/probe9b/` | $35.91 | $40 | ~$8.38 |
| rwkv7 stage A (stock readout + steer smoke) | `jobs/rwkv7/` `JOB=readout` | $20.79 | $25.46 | ~$5.99 |
| rwkv7 stage B (LoRA pilot, cache fork) | `jobs/rwkv7/` `JOB=lora` | $24.57 | $25.46 | ~$9.83 |
| rwkv7 C1 (stock readout through the vendor-compliant jrt-rwkv-g1-v2 template, DEV subset only) | `jobs/rwkv7/job_params_readout_g1v2.json` `JOB=readout_g1v2` | $18.27 | $19.10 | ~$2.50 |
| rwkv7 R3 (stock readout, v2 / v2f / v3 / v3w / v3r / v3p / v1 / v3c in one process, DEV subset only) | `jobs/rwkv7/job_params_readout_g1v3.json` `JOB=readout_g1v3` | $39.31 | $39.46 | ~$13.67 |
| fla_rwkv7 (fla chunk_rwkv7 gate, v1 rule; ran as job-jn47u) | `jobs/fla_rwkv7/` | $5.04 | $5.73 | ~$1.39 |
| fla_rwkv7 truth_v2 (RULE v2 + A1: fp64-truth numerics, m2 cell train speed; `fla_job.sh truth_v2`) | `jobs/fla_rwkv7/job_params_truth_v2.json` | $7.31 | $7.64 | ~$1.51 |
| cell_step_profile (m2 cell train step: baseline / ckpt_rows_lean / ckpt_rows / ckpt; pre-registered rule v1-A2) | `jobs/cell_step_profile/` | $10.46 | $10.82 | ~$2.03 |
| perf_step_gate v2 (m2 cell train step: greedy (numerics ref) / fast = d1_fast (speed ref) / fast_fla / fast_fla_nocompile: does fla chunk_rwkv7 speed TRAINING; pre-registered rule v2; v1 = job-2amiy $1.02; stack/rwkv7/perf-gate-fla) | `jobs/perf_step_gate/` | $11.34 | $11.46 | ~$1.86 |
| full8 (all 8 layers -> GDN, Q9-AF; priced from job-5cmat's measured phases; rule pre-registered at be9eed8; issue #87) | `jobs/full8/` | $107.73 | $108.19 | ~$35.47 |
| clef_index_subset (Clef-27B Index on the 2,008-row m1 DEV subset; canary; `RELEASE=clef JOB=I ROWS=subset`; planned at 5.39x) | `jobs/clef_baseline/job_params_index_subset.json` | $12.98 | $13.37 | ~$4.28 |
| clef_index_full (Clef-27B Index on all 12,104 DEV rows; `depends_on` index_subset `s1_ok and ok`; caps re-derived from the subset with `price.py rederive`; DEV only) | `jobs/clef_baseline/job_params_index_full.json` | $29.86 | $30.55 | ~$10.52 |
| clef_index_full_reference (Clef-27B full DEV with every Index phase on the torch reference GDN, JOB_SPEC 4d: job-jxevr's kernel_path 0.027 > 0.02, tolerance unchanged; caps = d2u54 rate x 4) | `jobs/clef_baseline/job_params_index_full_reference.json` | $49.90 | $50.91 | ~$17.58 |
| clef_overlength_flash (Clef-flash 32k sensitivity on the 51 over-length DEV rows, substituted into job-sgihj; `ROWS=overlength`) | `jobs/clef_baseline/job_params_overlength_flash.json` | $6.68 | $7.64 | ~$1.89 |
| clef_teacher_flash (optional m2 teacher: Clef-flash probabilities on the 24k TRAINING draw; guard exit 20) | `jobs/clef_baseline/job_params_teacher_clef_flash.json` | $14.87 | $15.28 | ~$4.54 |
| clef_teacher_clef (optional m2 teacher: Clef-27B, same draw; `depends_on` index_subset `s1_ok`; planned at 5.39x) | `jobs/clef_baseline/job_params_teacher_clef.json` | $45.49 | $45.82 | ~$16.82 |
| q9_feature_cache stock_dev (proxy head loop, PREMORTEM blocker #1; stock Q9 SELECT + HOLDOUT DEV features; independent of full-8; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_stock_dev.json` | $19.53 | $25.46 | ~$4.98 |
| q9_feature_cache stock_train (proxy head loop, PREMORTEM blocker #1; stock Q9 20,000 training rows; independent of full-8; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_stock_train.json` | $22.05 | $30 | ~$5.92 |
| q9_feature_cache conv1_dev (proxy head loop, PREMORTEM blocker #1; 1-layer-converted (L23 GDN) DEV features; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_conv1_dev.json` | $19.53 | $30 | ~$4.98 |
| q9_feature_cache conv1_train (proxy head loop, PREMORTEM blocker #1; 1-layer-converted, first 5,000 training rows; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_conv1_train.json` | $16.38 | $30 | ~$3.65 |
| q9_feature_cache e5_dev (proxy head loop, PREMORTEM blocker #1; E5 short-code DEV features; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_e5_dev.json` | $19.53 | $30 | ~$4.91 |
| q9_feature_cache e5_train (proxy head loop, PREMORTEM blocker #1; E5 short-code, first 5,000 training rows; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_e5_train.json` | $16.38 | $30 | ~$3.59 |
| m2_first_cell (first trained cell: RWKV7-13.3B LoRA r64 + cache-fork head, Clef arm, 150k set; job-tvuat ADOPT execution config; PR #74) | `jobs/m2_first_cell/` | $84.04 | $89.10 | ~$33.26 |
| m2_first_cell d1_fast (D1 base cell + the job-2amiy perf_step_gate step fixes: row_plan min_cost, cont_plan single, compile_blocks; 1.40x step, same caps, ~40 % more rows in the same 190-min clock; template for future cells) | `jobs/m2_first_cell/job_params_d1_fast.json` | $84.04 | $89.10 | ~$33.26 |
| m2_first_cell fulldev_eval (EVAL-ONLY: a SAVED cell checkpoint, D1 job-c7pq2 art-bm3e6 via the shared checkpoint_loader, scored on the FULL 12,104-row Index DEV, edition 0.2.1, every row read in full (cap 65,536 > longest row 47,509); full + 51-excluded Index, ECE, latency, cost; optional parent arm; vs-Clef paired CI landed with analysis/land_fulldev.py; 421 expected scoring min = 22.6 M tokens x D1's measured 1.037 s / 1k) | `jobs/m2_first_cell/job_params_fulldev_eval.json` | $80.89 | $82.73 | ~$28.85 (quoted $0.063: ~$28.85) |
| m2_first_cell fulldev_smoke (fulldev_eval pipeline proof: D1 art-bm3e6 + the stock parent arm on a pre-declared 250-row DEV slice, 83 tracks, 12 rows > 16,384 tokens incl. the 47,509 one; report + land_fulldev in subset mode; 73 run min) | `jobs/m2_first_cell/job_params_fulldev_smoke.json` | $23.56 | $25.46 | ~$5.54 (run only $3.61) |
| m2_first_cell hijack_eval (EVAL-ONLY, BOARD H1: D1 art-bm3e6 via the shared checkpoint_loader; hijack (5 HIJACK_TEMPLATES) + D1-A steer re-measure, paired child - parent, 3,600 units per arm; 72 run min) | `jobs/m2_first_cell/job_params_hijack_eval.json` | $18.52 | $19.10 | ~$5.48 (run only $3.56) |
| q9af_cell parent (the Q9-AF PARENT ARM once: stock readout of the converted model on cohort v3 + full DEV; shared comparator + lineage anchor of both cells; rule `jobs/q9af_cell/rule.py` 2d18afa; blocked on the job-2xe5w export pin; G13 re-priced at T <= 4,096) | `jobs/q9af_cell/job_params_parent.json` | $48.51 | $50.91 | ~$15.99 |
| q9af_cell cell_r64 (first Q9-AF decision cell: LoRA r64 + fork head, clef arm, 150k draw, act_ckpt, G12/G13 vs the parent arm; `depends_on` parent; E9 pair) | `jobs/q9af_cell/job_params_cell_r64.json` | $94.50 | $95.46 | ~$29.46 |
| q9af_cell cell_r256 (same cell, LoRA r256; E9 pair) | `jobs/q9af_cell/job_params_cell_r256.json` | $94.50 | $95.46 | ~$29.46 |

**Workspace cap when jobs overlap.** `set_workspace_cap` is workspace-wide, so jobs that can bill at the same time
need one cap covering all of them: MTD + `costs.concurrent_cap_usd([...])` = the ceiling of the sum of their worst
cases. probe9b + rwkv7 A (qual pass branch): MTD + 57 (MTD + 45 before the 2026-10-09 re-pricing). rwkv7 A + B chained with `depends_on`: MTD + 46 (was 36; B is
submitted while A can still bill). A job submitted alone: MTD + its hard cap. All six q9_feature_cache parts in parallel: MTD + 114 (was 90); the two stock parts alone: MTD + 42 (was 33).

## After the qualification (`scripts/decide_9b.py` verdict, `jobs/qual` exit code)

| Qual outcome | Next | Why |
|---|---|---|
| `open-$40-one-layer-probe-spend-issue` (stage-2 MFU >= 25%, peak <= 60 GB) | **probe9b, `PROBE_CONFIG=full`**; rwkv7 stage A as the parent comparator (PLAN m1: "steer baseline on all parents") | the 9B pipeline runs at the planned rate |
| `re-price-lean-plan` (MFU 12.5-25%, or peak 60-76 GB) | **probe9b, `PROBE_CONFIG=lean`** after re-pricing: `costs.reprice_probe(decision.json, ...)` gives the stage tokens the measured rate buys and whether the lean stop thresholds (50% of the 12.5%-MFU prediction) pass; quote it in the spend issue. rwkv7 stage A as comparator | same minutes and $, fewer tokens; batch 4 for memory |
| `fold` (MFU < 12.5%, peak > 76 GB) or qual exit 3/4 | **rwkv7 stage A, then stage B** (`depends_on` A `result.ok == true`) | no 9B conversion spend at all |
| qual exit 2/5/6/7/8/9/1 or a `no-measurement` verdict | nothing new; diagnose from the exported phases; at most one qual re-run from the reserve (<= $20) | no measurement, no decision |

## After the probe (`scripts/decide_probe9b.py`, rule in `configs/probe9b_h100.json: success_rule`)

| Probe outcome | Next |
|---|---|
| `proceed-all-8` | full 8-layer conversion: **`jobs/full8/`** (issue #87). Budgeted ~$150 in PLAN m1; priced from job-5cmat's measured phases at **$84.65 worst case / ~$27.87 expected**. KL order, accept/rollback, binding rule `jobs/full8/rule.json` (pre-registered at be9eed8; Amendment A1 2cb8cbc before any run). Layer 23 is re-converted, not warm-started from `gdn_converted.pt` (JOB_SPEC: no warm-start path in the pipeline). The distillation-mix recall lever is not applied: open text only, as in the probe |
| `retrieval-fail` | pre-registered fallback: no full-8 spend; report the cost-of-attention-freedom curve; rwkv7 stage B; shipping a hybrid is `decision-needed` for the user |
| `quality-fail` (gate rejected the layer, or ppl / MMLU / Index past its bound) | rwkv7 stage B (stage A already ran as comparator) |
| probe exit 3/4 (its own stop rule) | rwkv7 stage B |
| any other non-zero exit | diagnose; at most one probe re-run from the reserve |

The verdict is computed from the 511e8dd `success_rule` only. `decision.json: diagnostic_bands_posthoc` (gate dNLL / KL /
cosine bands and a suggested recipe change) was written after seeing the job-4j3vu qualification gate numbers; it is
reported for the reviewer and never changes the verdict or authorises spend.

Parent choice at m1 (PLAN gate "parent selected and frozen") compares the Q9-AF probe arms and the RWKV7 stage-A
readout on the **same 2,008-row DEV subset** (`jobs/common/dev_subset_m1.json`), plus steer and memory. It is a
recorded decision, not an automatic one.

## After the full-8 conversion (`scripts/decide_full8.py`, rule `jobs/full8/rule.json` @ be9eed8)

| Full-8 outcome | Next |
|---|---|
| `q9af-backbone` (8/8 layers accepted, every bound passes) | pin `gdn_converted.pt`'s sha256 in `jobs/m2_pilot/m2_pilot_cell_q9af.json` (own reviewed PR), then m2 parent arm + pilot cell with `PARENT=q9af` |
| `retrieval-fail` | PLAN fallback: no attention added back, cost-of-attention-freedom curve reported, Q9-AF stays the pure-recurrent entrant; the m2 backbone is `decision-needed` |
| `quality-fail` | m2 on RWKV7 |
| `partial-conversion` (1-7 layers accepted: a hybrid, not attention-free) | m2 on RWKV7; the `Q9-H<k>` hybrid and the curve are reported; shipping it is `decision-needed` |
| `no-conversion` | m2 on RWKV7 |
| exit 2 / 7 (crash, preemption, timeout) | finished layers are kept (checkpoint dir + salvage export): one resubmission from the reserve resumes |
| any other non-zero exit (incl. 9 = no-measurement) | diagnose; at most one re-run from the reserve |

## The $525 envelope (October MTD actual spend $16.11 as of 2026-10-07 incl. job-mden7 $1.06, job-5cmat $5.01 and job-tvuat $1.62, `costs.SPENT_USD`; $500 promo expires 2026-10-27T06:53Z)

**2026-10-10: the fixed $525 envelope is retired.** The operator lifted the $525 cap, and credit was topped up 2026-10-09 ($250 + $10). `costs.branch_ledger` now checks October spend + every committed worst case against the total credit granted for the period, read from the newest committed billing receipt (`costs.GRANTED_USD`); spend is the newest usage receipt (`costs.SPENT_USD`); `python -m jobs.common.lint` WARNs when either receipt is older than 2 days. Live balance: `autor billing` (no top-ups or payment links).

At $0.063/min with `receipts/2026-10-10-D-sf-usage-billing/usage.json` (MTD $530.03) and `receipts/2026-10-10-D-sf-usage-billing/billing.json` (granted $785.00 = promo $500 + top-ups $25 + $250 + $10): joint gate $61.74 of $75; Clef-27B Index line $49.52; optional teachers $60.36; spend + worst case per branch A $866.08 (**over** granted credit) / B $782.92 (fits) / C $747.01 (fits). These worst cases still count jobs that have already run and are inside the MTD (e.g. probe9b job-5cmat $5.01, full8 job-2xe5w $26.34, qual job-jbw22 / job-c4zkm), so the branch totals double-count them. The table and paragraph below keep their 2026-10-07 figures ($525 envelope, $16.11 MTD, $0.0495/min).

| Line (PLAN.md) | Budget | Worst case committed by this tree |
|---|---|---|
| Joint gate | $75 | qual $18.81 + Clef-flash GCG B $29.70 = $48.51 (Clef A ran as job-sgihj, $1.40 in MTD; C is superseded for the Index by the Clef-27B Index line below) |
| Clef-27B Index (authorised by the coordinator 2026-10-07, PR #98 review) | - | index_subset $10.20 + index_full $23.46 + overlength_flash $5.25 = $38.91 (re-priced at the card's 5.39x latency ratio; it would overrun the $75 joint line, so it is its own line, counted in the envelope) |
| System 1, branch A: probe proceeds | $250 | probe $28.21 + full-8 $84.65 (jobs/full8 worst case; was a ~$150 estimate) + rwkv7 A $16.34 = $129.20 -> $120.80 left for m2 (the cache-fork plan needs ~$51) |
| System 1, branch B: probe fails | $250 | probe $28.21 + rwkv7 A $16.34 + rwkv7 B $19.30 = $63.85 -> $186.15 left for m2 on RWKV7 (~$12-17 per cell, M2_RUN_SPEC) |
| System 1, branch C: qual folds | $250 | rwkv7 A + B = $35.64 -> $214.36 left for m2 |
| Language / Reserve | $100 / $100 | untouched; re-runs come from the reserve |
| Optional m2 teachers (System 1) | - | clef_teacher_flash $11.68 + clef_teacher_clef $35.74 = $47.42, only if a distillation cell is approved; `costs.branch_ledger` counts it in every branch (`system1_left_after_teachers`) |

Worst-case envelope total, including the $16.11 MTD spend, the Clef-27B Index line and both optional teachers, is at most $280.15 (branch A), so every branch fits $525 with the Language and Reserve
lines intact. Branch A leaves System 1 ~$70 above the cache-fork m2 plan before the optional teachers (it left ~$4 at the old ~$150 full-8
estimate); the full-8 worst case already counts a 60-min build and two full attempts.

D3 candidate `jobs/m2_first_cell/job_params_d3_r1_gold.json` (mode `d3_r1_gold`): D1 (11fb140, gold labels) trained and evaluated in the R1 readout format (C1 format v2 @46c0788: template `jrt-rwkv-g1-v2`, layout `g1v2`); the parent arm is scored in-job in the same format (no v1 seed). Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

D3 candidate `jobs/m2_first_cell/job_params_d3_v2f_gold.json` (mode `d3_v2f_gold`, preferred over `d3_r1_gold`): the same cell with template `jrt-rwkv-g1-v2f` (v2 + the dict-instruction fix, r3-catalog 0290fa8; v2 drops dict-valued instructions), layout `g1v2`; parent arm in-job via `G1V3Backend(mode="v2f")`. Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

D3 fast candidate `jobs/m2_first_cell/job_params_d3_v2f_fast.json` (mode `d3_v2f_fast`): `d3_v2f_gold` unchanged except the cell gains exactly the flags `d1_fast` adds over D1 (`row_plan` min_cost, `cont_plan` single, `compile_blocks` true; job-2amiy 1.40x step, execution-only, not in the TrainConfig hash); `d3_v2f_gold` itself stays as D3 ran it (job-93fpy @30d5239). Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

D3 v3w candidate `jobs/m2_first_cell/job_params_d3_v3w_fast.json` (mode `d3_v3w_fast`): `d3_v2f_fast` with only template `jrt-rwkv-g1-v3w` (layout `g1v2`; the best stock readout, r3-catalog 0290fa8: v3 codes + code key for k>=9 + word-like option ids scored by the first token of " <id>"; stock 31.07 vs v2f 29.10); train ids and gold target token == the v3w readout's (`scripts/check_g1v2_train_equivalence.py --template jrt-rwkv-g1-v3w`); parent arm in-job via `G1V3Backend(mode="words")`. Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

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
worst case/hard cap **$73.26**, published-rate bound **$70.89**, expected **$26.54**,
workspace increment $74.
This is derived within the $250 System 1 allocation; no fixed qualification-dollar
ceiling applies. Completion produces an integrity receipt only: no promotion, Clef win,
safety floors or automatic follow-on spend. Failed feasibility keeps permitted $0 work moving.

Every job below still needs its own `spend` issue, a LEDGER planned row, `set_workspace_cap` = MTD + its hard cap,
and a `validate_job` re-run against that cap before submit (same preconditions as `jobs/qual/JOB_SPEC.md`). The
numbers are computed by `python -m jobs.common.costs` and checked by `tests/test_m1_next_jobs.py`.

**Worst case** = (build_timeout + 2 attempts x max_duration) x $0.0495/min (build counted as billable; one platform
retry assumed even at `max_restarts=0`). **Expected** = sum of the expected phase minutes x the same rate.

| Job | Spec | Worst case | Hard cap (workspace cap) | Expected |
|---|---|---|---|---|
| qual (rung-2 qualification, unchanged) | `jobs/qual/` | $18.81 | $20 | <= $8.91 (one attempt, validate_job) |
| qual_c_only (phase-C re-run after the job-c4zkm 9B OOM; A and B passed there; env `QUAL_MODE=c_only`) | `jobs/qual/job_params_c_only.json` | $14.85 | $20 | <= $6.93 (one attempt; $7.92 with the 20-min build) |
| probe9b (one-layer 9B conversion probe; re-priced from the job-4j3vu measured rates, issue #68) | `jobs/probe9b/` | $28.21 | $40 | ~$6.58 |
| rwkv7 stage A (stock readout + steer smoke) | `jobs/rwkv7/` `JOB=readout` | $16.34 | $20 | ~$4.70 |
| rwkv7 stage B (LoRA pilot, cache fork) | `jobs/rwkv7/` `JOB=lora` | $19.30 | $20 | ~$7.72 |
| rwkv7 C1 (stock readout through the vendor-compliant jrt-rwkv-g1-v2 template, DEV subset only) | `jobs/rwkv7/job_params_readout_g1v2.json` `JOB=readout_g1v2` | $14.36 | $15 | ~$2.50 |
| rwkv7 R3 (stock readout, v2 / v2f / v3 / v3w / v3r / v3p / v1 / v3c in one process, DEV subset only) | `jobs/rwkv7/job_params_readout_g1v3.json` `JOB=readout_g1v3` | $30.89 | $31 | ~$10.74 |
| fla_rwkv7 (fla chunk_rwkv7 gate, v1 rule; ran as job-jn47u) | `jobs/fla_rwkv7/` | $3.96 | $4.50 | ~$1.09 |
| fla_rwkv7 truth_v2 (RULE v2 + A1: fp64-truth numerics, m2 cell train speed; `fla_job.sh truth_v2`) | `jobs/fla_rwkv7/job_params_truth_v2.json` | $5.74 | $6 | ~$1.19 |
| cell_step_profile (m2 cell train step: baseline / ckpt_rows_lean / ckpt_rows / ckpt; pre-registered rule v1-A2) | `jobs/cell_step_profile/` | $8.22 | $8.50 | ~$1.59 |
| perf_step_gate v2 (m2 cell train step: greedy (numerics ref) / fast = d1_fast (speed ref) / fast_fla / fast_fla_nocompile: does fla chunk_rwkv7 speed TRAINING; pre-registered rule v2; v1 = job-2amiy $1.02; stack/rwkv7/perf-gate-fla) | `jobs/perf_step_gate/` | $8.91 | $9.00 | ~$1.46 |
| full8 (all 8 layers -> GDN, Q9-AF; priced from job-5cmat's measured phases; rule pre-registered at be9eed8; issue #87) | `jobs/full8/` | $84.65 | $85 | ~$27.87 |
| clef_index_subset (Clef-27B Index on the 2,008-row m1 DEV subset; canary; `RELEASE=clef JOB=I ROWS=subset`; planned at 5.39x) | `jobs/clef_baseline/job_params_index_subset.json` | $10.20 | $10.50 | ~$3.37 |
| clef_index_full (Clef-27B Index on all 12,104 DEV rows; `depends_on` index_subset `s1_ok and ok`; caps re-derived from the subset with `price.py rederive`; DEV only) | `jobs/clef_baseline/job_params_index_full.json` | $23.46 | $24 | ~$8.27 |
| clef_index_full_reference (Clef-27B full DEV with every Index phase on the torch reference GDN, JOB_SPEC 4d: job-jxevr's kernel_path 0.027 > 0.02, tolerance unchanged; caps = d2u54 rate x 4) | `jobs/clef_baseline/job_params_index_full_reference.json` | $39.20 | $40 | ~$13.81 |
| clef_overlength_flash (Clef-flash 32k sensitivity on the 51 over-length DEV rows, substituted into job-sgihj; `ROWS=overlength`) | `jobs/clef_baseline/job_params_overlength_flash.json` | $5.25 | $6 | ~$1.49 |
| clef_teacher_flash (optional m2 teacher: Clef-flash probabilities on the 24k TRAINING draw; guard exit 20) | `jobs/clef_baseline/job_params_teacher_clef_flash.json` | $11.68 | $12 | ~$3.56 |
| clef_teacher_clef (optional m2 teacher: Clef-27B, same draw; `depends_on` index_subset `s1_ok`; planned at 5.39x) | `jobs/clef_baseline/job_params_teacher_clef.json` | $35.74 | $36 | ~$13.22 |
| q9_feature_cache stock_dev (proxy head loop, PREMORTEM blocker #1; stock Q9 SELECT + HOLDOUT DEV features; independent of full-8; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_stock_dev.json` | $15.35 | $20 | ~$3.91 |
| q9_feature_cache stock_train (proxy head loop, PREMORTEM blocker #1; stock Q9 20,000 training rows; independent of full-8; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_stock_train.json` | $17.32 | $30 | ~$4.65 |
| q9_feature_cache conv1_dev (proxy head loop, PREMORTEM blocker #1; 1-layer-converted (L23 GDN) DEV features; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_conv1_dev.json` | $15.35 | $30 | ~$3.91 |
| q9_feature_cache conv1_train (proxy head loop, PREMORTEM blocker #1; 1-layer-converted, first 5,000 training rows; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_conv1_train.json` | $12.87 | $30 | ~$2.87 |
| q9_feature_cache e5_dev (proxy head loop, PREMORTEM blocker #1; E5 short-code DEV features; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_e5_dev.json` | $15.35 | $30 | ~$3.86 |
| q9_feature_cache e5_train (proxy head loop, PREMORTEM blocker #1; E5 short-code, first 5,000 training rows; one <= 4.5 GiB artifact; rule `train/proxy_rule.py`) | `jobs/q9_feature_cache/job_params_e5_train.json` | $12.87 | $30 | ~$2.82 |
| m2_first_cell (first trained cell: RWKV7-13.3B LoRA r64 + cache-fork head, Clef arm, 150k set; job-tvuat ADOPT execution config; PR #74) | `jobs/m2_first_cell/` | $66.03 | $70 | ~$26.14 |
| m2_first_cell d1_fast (D1 base cell + the job-2amiy perf_step_gate step fixes: row_plan min_cost, cont_plan single, compile_blocks; 1.40x step, same caps, ~40 % more rows in the same 190-min clock; template for future cells) | `jobs/m2_first_cell/job_params_d1_fast.json` | $66.03 | $70 | ~$26.14 |
| m2_first_cell fulldev_eval (EVAL-ONLY: a SAVED cell checkpoint, D1 job-c7pq2 art-bm3e6 via the shared checkpoint_loader, scored on the FULL 12,104-row Index DEV, edition 0.2.1, every row read in full (cap 65,536 > longest row 47,509); full + 51-excluded Index, ECE, latency, cost; optional parent arm; vs-Clef paired CI landed with analysis/land_fulldev.py; 421 expected scoring min = 22.6 M tokens x D1's measured 1.037 s / 1k) | `jobs/m2_first_cell/job_params_fulldev_eval.json` | $63.56 | $65 | ~$22.67 (quoted $0.063: ~$28.85) |
| m2_first_cell fulldev_eval_d13 (EVAL-ONLY full DEV of D13 job-hjwpq art-cr9a2, BEST step 1600 export (select best), joint_record head, NATIVE readout jrt-rwkv-g1-v1 + cohort temperature fit; 12,104 rows, cap 65,536; full + 51-excluded + 2,008-subset + proxy600 Index, ECE served/untempered, latency; 547 expected min = 22.6 M tokens x the D13 screen's measured 1.2513 s / 1k + 75 fixed; vs-Clef landed with analysis/land_fulldev.py) | `jobs/m2_first_cell/job_params_fulldev_eval_d13.json` | $68.51 | $70 | ~$28.21 |
| m2_first_cell fulldev_eval_d13_v3w (the same D13 best checkpoint under the v3w readout = the 41.07 proxy600 screen's readout, UNTEMPERED; 500 expected min = 22.6 M x 0.9004 x 1.2513 s / 1k + 75 fixed; run beside fulldev_eval_d13: both together exceed the 900-min job ceiling) | `jobs/m2_first_cell/job_params_fulldev_eval_d13_v3w.json` | $74.25 | $75 | ~$25.88 |
| m2_first_cell fulldev_eval_d13_v3w_s1of2 / fulldev_eval_d13_v3w_s2of2 (fulldev_eval_d13_v3w SHARDED in two: shard K of the pre-declared jobs/m2_first_cell/fulldev_shards_2.json (harness.cases clusters never split, sha256(seed:case_id) mod 2, token-balanced: 6,109 / 5,995 rows, 11.32 M v1 tokens each, ids 5aa8ef15... / f523934e...); 266.5 expected min each (212.5 scoring + 33 fixed + 21 plan on the shard's rows) under a 336-min fulldev cap (26.1 % headroom); partial rows exported on any exit, SEED_PARTIAL_URL resumes them; merged on CPU by fulldev_eval merge into the unsharded child.results.jsonl, then analysis/land_fulldev.py) | `jobs/m2_first_cell/job_params_fulldev_eval_d13_v3w_s1of2.json`, `jobs/m2_first_cell/job_params_fulldev_eval_d13_v3w_s2of2.json` | $39.60 each | $60 | ~$13.2 + ~$13.2 (quoted hold $23.31 each) |
| m2_first_cell fulldev_eval_d19_v3w_s1of2 / fulldev_eval_d19_v3w_s2of2 (EVAL-ONLY full DEV of D19 job-fi5xf art-p3gpi, BEST step 2800 export trainable-step0002800.safetensors (select best; sidecar source ckpt-step0002800.pt 8cdfef6b... = the checkpoint the 48.61 proxy600 screen scored), joint_record head, readout v3w UNTEMPERED as the screen; the SAME pre-declared jobs/m2_first_cell/fulldev_shards_2.json shards as D13 (6,109 / 5,995 rows); 259.4 expected min each (205.4 scoring at the D19 screen's measured 1.2094 s / 1k v3w tokens + 33 fixed + 21 plan) under a 336-min fulldev cap (29.6 % headroom); partial rows exported on any exit, SEED_PARTIAL_URL resumes them; merged on CPU by fulldev_eval merge, then analysis/land_fulldev.py) | `jobs/m2_first_cell/job_params_fulldev_eval_d19_v3w_s1of2.json`, `jobs/m2_first_cell/job_params_fulldev_eval_d19_v3w_s2of2.json` | $39.60 each | $60 | ~$12.8 + ~$12.8 (quoted hold $23.31 each) |
| m2_first_cell devcal_d19_v3w (EVAL-ONLY, B-PREREG-D19-TEMP addendum 1: D19 job-fi5xf art-p3gpi BEST step 2800 export (as the full-DEV shards) scored on its OWN 804-item m2-dev-cohort-v2 dev pool + the 200-row dev-card sample with train.evalcard.run_card(keep_units=True): per-question raw probs for the CPU temperature fit (jobs.m2_first_cell.devcal_eval fit -> T.json); exit 10 unless pool size / dev-card ids + sha / per-source equal cell.json and no pool id is a training id; no Index DEV / proxy600 / JevBench row; 45 expected min + build) | `jobs/m2_first_cell/job_params_devcal_d19_v3w.json` | $10.79 | $15 | ~$2.97 (quoted hold $4.98) |
| m2_first_cell ragtruth_rephrase (EVAL-ONLY, B-PREREG-RAGTRUTH-REPHRASE stage 1: the one pre-registered RAGTruth statement rephrase RT1 (decide/ragtruth_rephrase.py, readout rewrite by exact content match) vs the original wording on the 300 ragtruth_polarity TRAIN items, D13 art-cr9a2 best1600 (rt1; original = job-3zmj2's pinned rows) + D19 art-p3gpi best2800 (original + rt1), v3w untempered; PASS iff RT1 AUROC lower bound > 0.5 (D19 also out-of-draw point > 0.5); a pass unlocks the DEV-minus-proxy600 stage; 46.8 expected min + build) | `jobs/m2_first_cell/job_params_ragtruth_rephrase.json` | $10.79 | $15 | ~$2.3 (quoted hold $4.98) |
| m2_first_cell ragtruth_rephrase_d19 (the D19 arm of ragtruth_rephrase alone: job-862xw's D13 arm cannot load on this lineage (attack guard v2 templates sha); original + rt1 on the 300 TRAIN items; 34.6 expected min + build) | `jobs/m2_first_cell/job_params_ragtruth_rephrase_d19.json` | $9.11 | $15 | ~$1.7 (quoted hold $3.91) |
| m2_first_cell fulldev_smoke (fulldev_eval pipeline proof: D1 art-bm3e6 + the stock parent arm on a pre-declared 250-row DEV slice, 83 tracks, 12 rows > 16,384 tokens incl. the 47,509 one; report + land_fulldev in subset mode; 73 run min) | `jobs/m2_first_cell/job_params_fulldev_smoke.json` | $18.51 | $20 | ~$4.36 (run only $3.61) |
| m2_first_cell hijack_eval (EVAL-ONLY, BOARD H1: D1 art-bm3e6 via the shared checkpoint_loader; hijack (5 HIJACK_TEMPLATES) + D1-A steer re-measure, paired child - parent, 3,600 units per arm; 72 run min) | `jobs/m2_first_cell/job_params_hijack_eval.json` | $14.55 | $15 | ~$4.31 (run only $3.56) |
| q9af_cell parent (the Q9-AF PARENT ARM once: stock readout of the converted model on cohort v3 + full DEV; shared comparator + lineage anchor of both cells; rule `jobs/q9af_cell/rule.py` 2d18afa; blocked on the job-2xe5w export pin; G13 re-priced at T <= 4,096) | `jobs/q9af_cell/job_params_parent.json` | $38.12 | $40 | ~$12.56 |
| q9af_cell cell_r64 (first Q9-AF decision cell: LoRA r64 + fork head, clef arm, 150k draw, act_ckpt, G12/G13 vs the parent arm; `depends_on` parent; E9 pair) | `jobs/q9af_cell/job_params_cell_r64.json` | $74.25 | $75 | ~$23.15 |
| q9af_cell cell_r256 (same cell, LoRA r256; E9 pair) | `jobs/q9af_cell/job_params_cell_r256.json` | $74.25 | $75 | ~$23.15 |

**Workspace cap when jobs overlap.** `set_workspace_cap` is workspace-wide, so jobs that can bill at the same time
need one cap covering all of them: MTD + `costs.concurrent_cap_usd([...])` = the ceiling of the sum of their worst
cases. probe9b + rwkv7 A (qual pass branch): MTD + 45. rwkv7 A + B chained with `depends_on`: MTD + 36 (B is
submitted while A can still bill). A job submitted alone: MTD + its hard cap. All six q9_feature_cache parts in parallel: MTD + 90; the two stock parts alone: MTD + 33.

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

D10 combined cell `jobs/m2_first_cell/job_params_d10_v2f_b100k.json` (mode `d10_v2f_b100k`): `d3_v2f_fast` on the B1 100k-v2 weighted draw (stack/rwkv7/b1-draw @e3eedeb, train_dir `m2-b100k`) with `train_max_k` 255 / `row_options` 160; T <= 2,048, eval unchanged; 190-min clock, cell cap 262 (tokenize 42 for 100k rows); worst (60 + 2 x 669) x 0.0495 = $69.20 ($70 hard stop), expected ~$27.72. Scaled clock `jobs/m2_first_cell/job_params_d10_v2f_b100k_t380.json` (mode `d10_v2f_b100k_t380`): 380 min, max_steps 4,500, cell cap 455; worst (60 + 2 x 862) x 0.0495 = $88.31 ($90 hard stop), expected ~$37.27.

D10w cell `jobs/m2_first_cell/job_params_d10w_v3w_b100k.json` (mode `d10w_v3w_b100k`): `d10_v2f_b100k` with only the readout format switched to v3w (template `jrt-rwkv-g1-v3w`, layout `g1v2`, r1_format_commit 0290fa8; as `d3_v3w_fast`). Data, flags, `train_max_k` 255 / `row_options` 160, 190-min clock and caps are D10's: worst $69.20 ($70 hard stop). CPU 2026-10-08: train==readout check on 234 B1 100k records / 345 rows (80 with K 17-255) 0 mismatches; census on a seeded 10k-record sample: v3w keeps 14,484 items vs v2f 15,225 (-4.9 %: the k>=9 code key pushes 797 items over T 2,048 vs 56), K > 16 items 282 vs 995, mean T 799.6 vs 833.4 (-4.1 %), so caps stay.

D16 cell `jobs/m2_first_cell/job_params_d16_v3w_b100k_safe.json` (mode `d16_v3w_b100k_safe`): `d10w_v3w_b100k` (job-hgwsq 40.14) + the anti-collapse safe keys (PR #232: lr_head 3e-4, cosine to 0.1x, dev card every 200 on 200 rows, best_ckpt, collapse_guard) + the g1v2 ordinal-head fix (6be1c8f) + `train_set.attack_clean` pinned to `data/manifests/attack_clean/b1-draw100k-v2.json` (5,201 D1-A rows dropped, 94,799 kept) + the 2x clock (380 min, max_steps 4,500, cell cap 455 as `d10_v2f_b100k_t380`); no fla training; stage screen (proxy600, screen_readout v3w vs d1_v3w / stock_v3w, cap 85, screen_fixed_minutes 47); verdict DROP if child-reference < -2.5. Schedule 742 min. **Worst case** (60 + 2 x 742) x $0.0495 = **$76.43** (cap $90). Expected 671 min ≈ $33.21.

D17 cell `jobs/m2_first_cell/job_params_d17_v3w_bschema230k_v2e.json` (mode `d17_v3w_bschema230k_v2e`): the D16 recipe unchanged (v3w, d1_fast flags, chunked_lean, `train_max_k` 255 / `row_options` 160, anti-collapse safe keys incl. dev_card_seed 0, ordinal fix, 380-min clock, max_steps 4,500) on Agent B's clean synthetic draw `bschema-draw230k-v2e` (stack/rwkv7/b-schema-synth @f4772fc; v2d with C's all-K multirecord pool, C 25.9 % of items; 234,521 records / 366,741 items, the whole pool; train_dir `m2/build/bschema-draw230k-v2e`, pinned by `jobs/m2_first_cell/data_d17_bschema230k_v2e.sha256`); no heldout_excluded / attack_clean / doc_clean pins (0 attack_guard hits, 0 doc_overlap_guard fails, 0 train.heldout hits: every guard must find zero); the pointer's v1_overlength_ids do not apply under v3w (CPU cell_items: 366,741 of 366,741 items kept, max T 2,048); the doc-overlap index ships as D16. Caps: heldout 15, cell 531 (tokenize 109 scaled by items + 9 for the draw's guard scans), screen 157 (screen_fixed_minutes 123); stage screen (proxy600 v3w vs d1_v3w / stock_v3w). Schedule 895 min. **Worst case** (60 + 2 x 895) x $0.0495 = **$91.58** (cap $95); platform quote 895 x $0.063 = $56.38. Expected 828 min ≈ $40.99.

D18 cell `jobs/m2_first_cell/job_params_d18_v3w_v2e_joint.json` (mode `d18_v3w_v2e_joint`): `d17_v3w_bschema230k_v2e` with ONLY `"head": "joint_record"` (D13's attention-free joint record head, stack/rwkv7/d13-joint 87d98a1 + 44e40bf record-id grouping; job-hjwpq `d13_joint_safe_nofla` screened +3.36 [-1.47, 7.06] over D1+v3w on proxy600), ported with three fixes for v3w / v2e: g1v2 lexical prior from the option's schema span, joint steps chunked by whole records under `row_options` 160, records longer than the 8-row step kept whole (up to 16 rows). Data, caps, clock and screen stage = D17. Schedule 895 min. **Worst case** (60 + 2 x 895) x $0.0495 = **$91.58** (cap $95); platform quote 895 x $0.063 = $56.39. Expected 828 min ≈ $40.99.

D19 cell `jobs/m2_first_cell/job_params_d19_v3w_v3b_joint.json` (mode `d19_v3w_v3b_joint`): `d18_v3w_v2e_joint` (joint record head + whole-record dev card, @1a75caf) with ONLY the training set changed to B's `bschema-draw-v3b` (stack/rwkv7/b-schema-synth @cddbe75; 253,046 records = v2e + format levers API-Bank K40-60, MMLU K10, BoolQ yes/no, BANKING77 K77, Index-fingerprinted; train_dir `m2/build/bschema-draw-v3b`, pinned by `jobs/m2_first_cell/data_d19_bschema_v3b.sha256`); no doc_clean / attack_clean / heldout pins (doc_overlap_guard 0 fails, attack_guard 0, train.heldout 0); CPU census 385,266 items, 0 dropped under v3w (max T 2,048). Caps: cell 533, screen 159 (screen_fixed_minutes 125), rest = D18. Schedule 899 min. **Worst case** (60 + 2 x 899) x $0.0495 = **$91.97** (cap $95); platform quote 899 x $0.063 = $56.64. Expected 832 min ≈ $41.18.

D3 v3w candidate `jobs/m2_first_cell/job_params_d3_v3w_fast.json` (mode `d3_v3w_fast`): `d3_v2f_fast` with only template `jrt-rwkv-g1-v3w` (layout `g1v2`; the best stock readout, r3-catalog 0290fa8: v3 codes + code key for k>=9 + word-like option ids scored by the first token of " <id>"; stock 31.07 vs v2f 29.10); train ids and gold target token == the v3w readout's (`scripts/check_g1v2_train_equivalence.py --template jrt-rwkv-g1-v3w`); parent arm in-job via `G1V3Backend(mode="words")`. Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

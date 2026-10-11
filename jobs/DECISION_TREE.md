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

At $0.063/min with `receipts/2026-10-10-D-sf-usage-billing/usage.json` (MTD $530.03) and `receipts/2026-10-10-D-sf-usage-billing/billing.json` (granted $785.00 = promo $500 + top-ups $25 + $250 + $10): joint gate $61.74 of $75; Clef-27B Index line $69.56 (its full-DEV slot is clef_index_full_reference at $49.90 since PR #241 review: job-jxevr's fast-GDN index_full failed at S1 and job-zfexa ran the reference); optional teachers $60.36. Entries whose listed jobs (`costs.LEDGER_JOB_IDS`) are all terminal in committed job receipts count $0 toward the credit check, because their actual cost is already in the MTD: qual job-jbw22, job-c4zkm, job-4j3vu (actual $3.29); clef_index_subset job-5mw9h, job-d2u54 (actual $1.47); clef_index_full_reference job-jxevr, job-zfexa (actual $8.28); probe9b job-5cmat (actual $5.01); full8 job-2xe5w (actual $26.34); rwkv7_readout job-mhc3h, job-tkfaj (actual $2.35) ($46.74 in total). Still committed (not run): joint gate $37.80, Clef-27B Index $6.68, teachers $60.36; MTD + committed per branch A $634.87 / B $659.44 / C $659.44, all within the $785.00 granted. The table and paragraph below keep their 2026-10-07 figures ($525 envelope, $16.11 MTD, $0.0495/min).

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

D3 v3w candidate `jobs/m2_first_cell/job_params_d3_v3w_fast.json` (mode `d3_v3w_fast`): `d3_v2f_fast` with only template `jrt-rwkv-g1-v3w` (layout `g1v2`; the best stock readout, r3-catalog 0290fa8: v3 codes + code key for k>=9 + word-like option ids scored by the first token of " <id>"; stock 31.07 vs v2f 29.10); train ids and gold target token == the v3w readout's (`scripts/check_g1v2_train_equivalence.py --template jrt-rwkv-g1-v3w`); parent arm in-job via `G1V3Backend(mode="words")`. Caps unchanged: inherits the D1 $66.03 ceiling ($70 hard stop).

Q9-AF v2 `jobs/q9af_v2/job_params.json` (stack/q9af/v2-full-gdn; option (b) of #140 reopened by its own stall trigger: champion 38.67 is > 15 pts below Clef-flash): all 8 attention layers -> GDN jointly (reuse_attn init, 70 min layer-wise + 430 min end-to-end KD on a generic DEV-decontaminated mix), end-to-end rule `jobs/q9af_v2/rule.py` (held-out KL + stock Index proxy600; STRONG / ADVANCE -> pin for `jobs/q9af_cell`, MARGINAL -> decision-needed, FOLD / FOLD_KL -> archive). One attempt: $37.37 worst case, ~$28 expected.

Q9-AF v2 first cell `jobs/m2_first_cell/job_params_q9af_v2_v3w.json` (mode `q9af_v2_v3w`; stack/q9af/v2-v3w; launch ONLY on a `q9af-v2-rule-v1` STRONG / ADVANCE verdict of job-w3c2e, after `scripts/fill_conversion_pins.py` in its own commit): the D16 recipe on the all-GDN Qwen3.5-9B. Parent `q9af` (converted export, every attention layer GDN; hybrid refused), template `jrt-v3w-generic:qwen3.5` (generic v3w on the Qwen tokenizer + chat template, spec identity pinned; train == readout 0 mismatches on 1,919 B1 rows), row_batch 1 (HF GDN padding is not exact), probe9b image (fla 0.5.2 GDN kernel), in-job parent arm under the same v3w readout, stage screen (proxy600) vs D1+v3w (`screen_reference_match` family). Schedule, caps and worst case are D16's: **$76.43** (cap $90). Verdict DROP if child - reference < -2.5.

A-PREREG-SIZECLIFF size-cliff probe `jobs/m2_first_cell/job_params_sizecliff_q35_0p8b.json`, `jobs/m2_first_cell/job_params_sizecliff_q35_2b.json`, `jobs/m2_first_cell/job_params_sizecliff_q35_4b.json` (modes `sizecliff_q35_0p8b`, `sizecliff_q35_2b`, `sizecliff_q35_4b`; stack/qwen35/sizecliff): the D16 recipe on the STOCK Qwen3.5-0.8B / 2B / 4B (parent `qwen35`), trained and served in the replicated H2O-Lightning shim format (`jrt-h2o-shim:*`, PREREG Amendment 1), same draw (94,786), LoRA r64, 100-min clock on each; screen = proxy600 (cross-readout vs d1_v3w + the Clef-flash / Clef 27B units) + jevkit on the 231 public JevBench items (shim primary; v3w and the shim post-processing score-only; the 4B also with the lean shim prompt). Decision rule = PREREG.md A-PREREG-SIZECLIFF (>= 0.92x the 4B on proxy and jevkit-I -> small-backbone route; < 0.80x on both -> close the <= 2B line; between -> one longer-clock probe; lean adopted iff I drops <= 1 point while tokens drop >= 25%). Worst case $24.95 each, quoted hold $13.99 each.

A-PREREG-JEV1-4B `jobs/m2_first_cell/job_params_jev1_q35_4b.json` (mode `jev1_q35_4b`; stack/qwen35/jev1-4b): the size-cliff 4B r4 recipe on bschema-draw-jev1 (96,000 JevBench-family rows), 1,121 steps; PRIMARY = jevkit delta Composite A vs stock Qwen3.5-4B + the H2O shim in the same job (success iff the 95% CI excludes 0). Worst case $42.46, quoted hold $19.34.

## After JEV1-4B r2 lands (2026-10-10): WIN -> JEV2-4B, FLAT/LOSE -> JEV1-DIAG

Read r2's `screen.json` `jevkit.primary` (A-PREREG-JEV1-4B: delta Composite A vs stock + shim, 95% CI).

- **Always, $0, CPU:** `scripts/check_jev1_vs_jevkit_render.py --out <f>` (tokenizer only; needs >= 4 GB free). It diffs
  one jev1 train row vs one jevkit item per family and qtype through the SAME `heads.layout.build_row`: state type, question
  keys, answer-slot suffix, option continuations. It covers only the 8 jev1 families (120 items). The other 111 items
  (multi_hop, judge_hard, temporal_numeric, fact, tool_selection, probability, trap, ambiguous, tradeoff, adversarial,
  routing_hard) have no train row at all. Also run the per-family breakdown of r2's own preds (covered vs uncovered).
- **WIN (CI > 0) -> JEV2-4B** (`stack/qwen35/jev2-4b`, mode `jev2_q35_4b`, A-PREREG-JEV2-4B). Blocked on the jev2 draw.
  When it lands:
  `scripts/jev12_balanced_draw.py --src .../bschema-draw-jev1 --src .../bschema-draw-jev2 --train-per-family 1000
  --calib-per-family 200 --name bschema-draw-jev12-bal1000`. Then copy its pointer.json into
  `data/manifests/bschema-draw-jev12-bal1000.pointer.json`, its shas into `jobs/m2_first_cell/data_jev2.sha256`, and
  `calib_ids_sha256` into the cell's `calib_slice.ids_sha256` and the params' `jevkit.tfit.ids_sha256`. If the jev2
  family count is not 11, re-run the scratch generator (FAMS) so steps = rows / 8. Run `tests/test_jev2_4b.py`, then
  package with `jobs.common.launch`.
  Budget: 1,000/family = worst $94.12 (expected ~$34). If the free balance is below ~$95, use 800/family instead:
  15,200 rows, 1,900 steps, clock 445, max 607 min, worst $80.26, expected ~$29.
- **FLAT / LOSE -> JEV1-DIAG** (`stack/qwen35/jev1-diag`, mode `jevdiag_q35_4b`, A-PREREG-JEV1-DIAG). It is eval-only
  on r2's checkpoint, so it can launch as soon as r2's artifact shas are in the `checkpoint` block, the shipped cell.json
  pin is set and `jevdiag.r2_delta_I_ci95` is filled. Worst case $17.89. Its `diagnosis` picks the next cell:
  READOUT -> LM-head label readout; TRANSFER -> coverage + format first (jev2); UNDERFIT -> JEV2-4B budget + LR x2;
  mixed -> JEV2-4B.

**Update (A-PREREG-JEV2-4B Amendment 2, 2026-10-10):** r2 was FLAT, so JEV1-DIAG is folded (the jev1 audit explains r2) and JEV2-4B is the next cell. It is registered as `jobs/m2_first_cell/job_params_jev2_q35_4b.json` (mode `jev2_q35_4b`, stack/qwen35/jev2-4b). Shape: 586 rows per family (adequacy 293) x 18 families = 10,255 rows, 1,281 steps, clock 462, max_duration 604 min, worst case $79.88 (operator hard cap $80), expected ~$28.7. It waits on jev2b + jev2c: rebuild the mix with them and refill the PROVISIONAL pins.

**JEV2-4B SMALL fallback (2026-10-11):** registered as `jobs/m2_first_cell/job_params_jev2s_q35_4b.json` (mode `jev2s_q35_4b`): 163 rows per family (adequacy 81) x 18 = 2852 rows, 356 steps, clock 129, max_duration 271 min, worst case $37.93 (<= $38). Use it if the balance cannot hold the full variant's $79.88.
**JEV2-H2O (A-PREREG-JEV2-H2O, 2026-10-11, NOT submitted; the NEXT training cell, ahead of JEV2-4B-stock):** H2O-Lightning-4B v1.1 @193ad740 + SSDM jev2 data (EV-PREREG-H2O-HOST), `stack/qwen35/jev2-h2o`; the FINAL cleared mix (bschema-mix-jev1-jev2 71b503eb components, two exclusion lists asserted, jev2b prob_indep dropped), draw bschema-draw-jev2h-final (2,664 rows, 148 per family x 18). Two registered modes, hard cap $38 each (worst case $37.42, max_duration 267 min, expected ~$11.65): `jobs/m2_first_cell/job_params_jev2_h2o_shim.json` (mode `jev2_h2o_shim`, DEFAULT: H2O shim readout, primary native_shimcal, 333 steps) and `jobs/m2_first_cell/job_params_jev2_h2o_sym.json` (mode `jev2_h2o_sym`, sym-fork + G1 gate, 320 steps). LoRA r16, LR 4e-5; selection + step-0 floor on the held-out slice re-derived from the mix (3,600 rows, 142a593a). WIN (delta Composite A vs the in-job H2O v1.1 control CI > 0 AND the floor passes) -> host line continues; FLAT / floor fails -> the LM-prior residual head (step 0 == the control) next.

**JEV2-H2O-L (A-PREREG-JEV2-H2O-L, 2026-10-11, NOT submitted; after job-x7c42 lands):** the same cell for ~1 epoch of the 1,000-per-family cleared draw (17,959 rows, 2,244 steps), justified by the measured 2.55 s/step (job-9fy9p); `jobs/m2_first_cell/job_params_jev2_h2o_shim_l.json` (mode `jev2_h2o_shim_l`), max_duration 237 min, worst case $33.64 (hard cap $34), expected ~$9; the same selection slice 142a593a; Holm over the two JEV2-H2O arms.

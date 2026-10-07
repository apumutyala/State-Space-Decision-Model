# m1 decision tree: qualification outcome -> next job (spec only, nothing submitted)

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
| fla_rwkv7 (fla chunk_rwkv7 gate, v1 rule; ran as job-jn47u) | `jobs/fla_rwkv7/` | $3.96 | $4.50 | ~$1.09 |
| fla_rwkv7 truth_v2 (RULE v2 + A1: fp64-truth numerics, m2 cell train speed; `fla_job.sh truth_v2`) | `jobs/fla_rwkv7/job_params_truth_v2.json` | $5.74 | $6 | ~$1.19 |
| cell_step_profile (m2 cell train step: baseline / ckpt_rows_lean / ckpt_rows / ckpt; pre-registered rule v1-A2) | `jobs/cell_step_profile/` | $8.22 | $8.50 | ~$1.59 |
| full8 (all 8 layers -> GDN, Q9-AF; priced from job-5cmat's measured phases; rule pre-registered at be9eed8; issue #87) | `jobs/full8/` | $84.65 | $85 | ~$27.87 |
| m2_first_cell (first trained cell: RWKV7-13.3B LoRA r64 + cache-fork head, Clef arm, 150k set; job-tvuat ADOPT execution config; PR #74) | `jobs/m2_first_cell/` | $66.03 | $70 | ~$26.14 |

**Workspace cap when jobs overlap.** `set_workspace_cap` is workspace-wide, so jobs that can bill at the same time
need one cap covering all of them: MTD + `costs.concurrent_cap_usd([...])` = the ceiling of the sum of their worst
cases. probe9b + rwkv7 A (qual pass branch): MTD + 45. rwkv7 A + B chained with `depends_on`: MTD + 36 (B is
submitted while A can still bill). A job submitted alone: MTD + its hard cap.

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
| Joint gate | $75 | qual $18.81 + Clef baseline A+B+C $47.52 = $66.33 |
| System 1, branch A: probe proceeds | $250 | probe $28.21 + full-8 $84.65 (jobs/full8 worst case; was a ~$150 estimate) + rwkv7 A $16.34 = $129.20 -> $120.80 left for m2 (the cache-fork plan needs ~$51) |
| System 1, branch B: probe fails | $250 | probe $28.21 + rwkv7 A $16.34 + rwkv7 B $19.30 = $63.85 -> $186.15 left for m2 on RWKV7 (~$12-17 per cell, M2_RUN_SPEC) |
| System 1, branch C: qual folds | $250 | rwkv7 A + B = $35.64 -> $214.36 left for m2 |
| Language / Reserve | $100 / $100 | untouched; re-runs come from the reserve |

Worst-case envelope total, including the $16.11 MTD spend, is at most $211.64 (branch A), so every branch fits $525 with the Language and Reserve
lines intact. Branch A now leaves System 1 ~$70 above the cache-fork m2 plan (it left ~$4 at the old ~$150 full-8
estimate); the full-8 worst case already counts a 60-min build and two full attempts.

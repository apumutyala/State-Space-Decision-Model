# Distill v0: held-out train-source pilot

## Artifact and scope

- **Checkpoint:** `/results/students/distill-v0-20261015/merged` on Modal volume `keel-results`.
- Base: Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; teacher: pinned Qwen3.5-9B, trust mode, two thinking samples, max_think 768.
- One epoch of bf16 LoRA, rank 16, over **8,105 soft-labelled variants** from **3,107 train-source items**. Objective: label-token-restricted KL(teacher || student) + 0.2 CE(gold).
- Sources: existing repo synthetic access-policy, severity and event-state generators. Total draw 3,299; held-out pilot 192, disjoint by item/group/content. No public JevBench item was scored or used for training by this slice.
- Final training-content scans: **zero public-item overlap and zero evaluation-attack-template hits**. Decision Index document-overlap guard was unavailable, not passed.
- Full recipe, source provenance, template definitions and limitations: [distill package README](../../../keel/distill/README.md). Only code, configurations and aggregates are committed; records and per-item outputs remain under `$SSDM_PRIVATE/distill_v0/` (default `~/ssdm/private`).

## Pilot results

Same 192 held-out source groups for all arms; clean/inject/authority/policy each have 192 units. Evidence has 14 strictly screened counterfactuals. Student and stock 4B use **one-pass trust**; 9B uses **thinking trust**, two samples, with a fresh seed independent of label validation. Baseline scores already cached on a larger evidence superset were restricted to exactly these 14 units. The final comparison was assembled from cached scores without additional GPU calls.

Cells are **percent [95% bootstrap CI] (successes / trials)**. Intervals use 10,000 percentile resamples of entire source groups, seed 61015. Lower ASR is better; higher other rates is better.

| Metric | Student 4B one-pass | Stock 4B one-pass | Teacher 9B think |
|---|---:|---:|---:|
| Clean accuracy | 82.81 [77.60, 88.02] (159/192) | 61.46 [54.69, 68.23] (118/192) | 78.12 [71.88, 83.85] (150/192) |
| Inject ASR | 0.00 [0.00, 0.00] (0/159) | 38.14 [29.46, 47.20] (45/118) | 8.67 [4.49, 13.42] (13/150) |
| Authority ASR | 0.00 [0.00, 0.00] (0/159) | 51.69 [42.74, 60.68] (61/118) | 7.33 [3.38, 11.72] (11/150) |
| Combined ASR | 0.00 [0.00, 0.00] (0/318) | 44.92 [37.39, 52.75] (106/236) | 8.00 [4.25, 12.17] (24/300) |
| Robust accuracy | 82.81 [77.60, 88.02] (159/192) | 24.48 [18.75, 30.73] (47/192) | 68.23 [61.46, 75.00] (131/192) |
| Evidence retention | 100.00 [100.00, 100.00] (14/14) | 92.86 [78.57, 100.00] (13/14) | 100.00 [100.00, 100.00] (14/14) |
| Paired evidence retention | 100.00 [100.00, 100.00] (14/14) | 92.86 [78.57, 100.00] (13/14) | 100.00 [100.00, 100.00] (14/14) |
| Policy compliance | 100.00 [100.00, 100.00] (192/192) | 33.33 [27.07, 40.10] (64/192) | 68.23 [61.98, 75.00] (131/192) |

Definitions:

- ASR counts targeted wrong-option success only on attack trials where the **same model** answered clean correctly. Inject/authority each have one trial per eligible parent; combined ASR has two. It is not an unconditional error rate.
- Robust accuracy requires clean **and both attacks** correct for a parent.
- Evidence retention requires the edited decision to equal the validated changed gold. Paired retention additionally requires correct clean judgment.
- Policy compliance tests an explicitly trusted **category-reporting remap**, on all policy units without teacher-compliance filtering.

**Boundary warning:** percentile bootstrap gives degenerate [0, 0] or [100, 100] intervals when every observed resample is identical. The student's zero observed ASR, 192/192 policy compliance, and 14/14 evidence retention do **not** establish zero population risk or perfect population performance. Evidence is especially small and selected; it supports no general evidence-retention claim.

The student's point estimate improves over stock 4B on this within-family synthetic pilot. This is **not** evidence of unseen-attack-family generalization: the six authored training attack templates are also used with new source instances in this pilot. Groups are individual generated scenarios, not held-out generator families. No hyperparameter search or student selection used these pilot results. Teacher/student validators are related Qwen models. Original event builders can omit initial queried state, so overall clean/robust metrics use generator gold rather than independently adjudicated clean labels; the evidence screen excludes unknown queried initial state. Public Card/gate evaluation is owned by integration and is not included here.

## Evidence quality and label yield

Training labels by kind: clean 2,511; inject 2,108; authority 2,008; policy 1,241; evidence 237. Total 8,105. Clean filtering excluded 596 parents and all descendants. On 2,511 clean-correct parents, teacher mean target steering was 215/2,511 inject and 282/2,511 authority; any-sample decision change was 403/2,511 and 503/2,511 respectively, and these attack units were excluded.

The initial worker audit of 20 accepted evidence edits found 12 valid, 0 definitely invalid and 8 ambiguous. Ambiguities were treated as failures. Evidence was tightened to one explicit later categorical event with known prior queried state, exact replayable changed gold, no instruction/authority language, two 9B-thinking validation samples, and fresh 4B-thinking clean+edited agreement.

- Training evidence: **1,207 teacher accepted → 248 executable-event candidates → 237 cross-model validated**.
- Pilot evidence: **82 → 14 → 14**.
- Fresh random 20 of the final 237 training edits: **20 valid, 0 invalid, 0 ambiguous** on full-record worker review. This is not independent human annotation.

See `label-yield.json`, `train-evidence-screen.json`, `pilot-evidence-screen.json`, and `evidence-audit.json` for counts and seeds.

## Checkpoint execution evidence

Real H100 training completed 2,027 microbatches and 507 optimizer updates; mean objective 0.10745, finite gradient norms, 10.75 GB peak allocation. An initial export omitted processor assets; vLLM initialization repeatedly failed. The exporter now copies pinned image/video processor configurations and chat template, with CPU filesystem regression tests. A CPU-only backfill repaired the same checkpoint without retraining or modifying weights. `AutoProcessor.from_pretrained(..., local_files_only=True)` loaded it successfully.

A separate bounded `keel.runtime.vllm_judge.judge(...)` call then loaded the merged checkpoint and returned finite normalized distributions on **four TRAIN-source rows**: 116.26 s load, 0.485 s warm scoring, 143.99 s local wall. The bounded student pilot also completed, including its own four-row train-source load check. See `train.json`, `processor-assets.json`, `checkpoint-smoke.json`, and `pilot.json`. No validation result here depends on a mock model.

## Spend log

H100 accounting rate: $3.95/hour; cap: $45. **Total upper estimate: $20.72**, including failed runs. These are conservative wall-time estimates, not invoices. CPU-only rows have zero GPU minutes but conservatively use the H100 wall-time rate for cost accounting. The original failed call's **$11.60 exposure is retained**, with a separate $0.13 shutdown reserve derived from its observed app lifetime. The known-bad retry was cancelled and explicitly stopped; later load/pilot runs have 900/1,800-second local and remote limits.

| Date | Function | H100-minutes upper estimate | USD upper estimate |
|---|---|---:|---:|
| 2026-10-15 | train_remote smoke: torch2.10 | 7.284 | $0.480 |
| 2026-10-15 | training image build: CUDA13 | 0.000 | $0.227 |
| 2026-10-15 | train_remote corrected smoke | 12.712 | $0.837 |
| 2026-10-15 | 9B pilot authorship and labels | 6.748 | $0.444 |
| 2026-10-15 | 9B training authorship and labels | 66.015 | $4.346 |
| 2026-10-15 | 4B thinking pilot evidence screen | 3.243 | $0.214 |
| 2026-10-15 | 4B thinking training evidence screen | 3.260 | $0.215 |
| 2026-10-15 | train_remote full epoch | 18.348 | $1.208 |
| 2026-10-15 | student pilot failed: missing processor assets | 176.247 | $11.603 |
| 2026-10-15 | failed app shutdown reserve | 1.953 | $0.129 |
| 2026-10-15 | cancelled pre-repair retry | 1.500 | $0.099 |
| 2026-10-15 | CPU checkpoint processor repair | 0.000 | $0.025 |
| 2026-10-15 | repaired checkpoint four-row smoke | 2.400 | $0.158 |
| 2026-10-15 | runtime.judge: stock_4b pilot | 2.575 | $0.170 |
| 2026-10-15 | runtime.judge: student_4b pilot | 3.069 | $0.202 |
| 2026-10-15 | runtime.judge: teacher_9b pilot | 5.604 | $0.369 |

Machine-readable ledger: `costs.json`; failed-call details: `pilot-failed.json`. The early FLA correctness refusal was fixed by upgrading the training image, never by disabling the guard. The image-build compiler failure was fixed by selecting installed GCC/G++. Processor repair was CPU-only. No billing totals or broader model capabilities are inferred from these measurements.

## CPU acceptance

Final command from the distill worktree:

```sh
python -X utf8 -m pytest -q keel/tests
```

**46 passed in 1.75 s.** Coverage includes attack-template disjointness,
wrong-option targets, a planted public-item overlap failing closed, hand-checked
restricted KL/CE values and real tiny-model gradients, nonlabel exclusion,
teacher filtering, executable evidence semantics, group bootstrap denominators,
and processor export with all required files present and trained-weight bytes
unchanged. The processor regression also checks that the exact pinned base
revision is requested and missing assets fail before partial export.

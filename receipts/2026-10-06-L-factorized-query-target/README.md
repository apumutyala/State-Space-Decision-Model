# Factorized operation and structured query-target CPU condition

The single preregistered factorized architecture improves pooled exact target selection over PR #75's matched all-task-diversity arm, concentrated in the new wording family and explanations. It does not establish that the read/compare tradeoff is solved: read remains slightly below the raw baseline overall, and comparison changes against the matched diversity arm are uncertain. PR #75 worlds/templates are reused iterative-development data, not fresh confirmatory validation. This is synthetic development evidence, not held-out or production qualification; it supports no broad-transfer claim. No follow-up variant was fit.

Source/config frozen at `d2d0a3006ef8702bdf81fb0af08642c2daeb5f57` before fitting. Fresh worktree `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-factorized-query-target`, branch `codex/l0-factorized-query-target`, based directly on `origin/agent-main` commit `4733f785e5284ea959d4d640fff884136487bd05`. The unpushed review branch was subsequently rebased onto `24f164e9cfff1abe8f7eecef68a1771d357ca465`; the original freeze/result commits are preserved on `codex/l0-factorized-query-target-pre-rebase-20261006`, and no training was repeated. No experimental branch was imported or merged. PR #75 source and data were read-only references.

The shared byte diagonal encoder is width 32 and retains PR #75's terminal/mean recurrence. A separate 64-to-3 linear head predicts read/explain/compare. The candidate scorer takes the query plus separate byte encodings for field, left entity, right entity (empty for read/explain), and temporal scope, and their four query products; a 576-to-32 tanh layer and scalar output score the structured target. The scorer has no operation feature. Complete target syntax is enumerated from request-state permitted records and matched to every candidate description; it never parses the task text or consults evaluator bindings. Operation argmax is followed by target argmax within that operation. Training sums operation cross-entropy and structured target cross-entropy conditioned on the supervised operation, with equal weight. This changes both factorization and structured candidate representation, so it does not isolate either component. Temporal scope is constant in these fixtures.

28,004 trainable parameters versus PR #75's 27,937 (+67, +0.24%). Exact PR #75 frozen all_task_diversity rows are reused: 600 training worlds, seed 20261006, six templates per operation, 200 worlds per operation. The same 180 validation task IDs at seed 20261009 are rendered in both frozen old_failing and new_disjoint families (60 tasks per operation). All train/validation hashes match PR #75. Paired initialization/sampling seeds are 20261006, 20261016, 20261026; 300 final-checkpoint Adam steps, batch 32, learning rate 0.003, clip norm 1.0. No validation selection or tuning. Only the factorized architecture was trained; PR #75 predictions are reused without refitting.

Exact target requires operation, field, entity/order and scope. Every row below is raw baseline / matched all_task_diversity / factorized. Operation counts are correct operation regardless of exact target.

| Family | Seed | Operation /180 | Read /60 | Explain /60 | Compare /60 | Exact all /180 |
|---|---|---|---|---|---|---|
| old_failing | 20261006 | 119 / 149 / 180 | 38 / 36 / 42 | 40 / 37 / 34 | 0 / 15 / 34 | 78 / 88 / 110 |
| old_failing | 20261016 | 159 / 175 / 180 | 51 / 40 / 39 | 30 / 40 / 34 | 25 / 31 / 20 | 106 / 111 / 93 |
| old_failing | 20261026 | 70 / 166 / 180 | 50 / 44 / 46 | 0 / 49 / 41 | 1 / 25 / 28 | 51 / 118 / 115 |
| new_disjoint | 20261006 | 28 / 116 / 139 | 22 / 32 / 34 | 0 / 14 / 26 | 0 / 23 / 24 | 22 / 69 / 84 |
| new_disjoint | 20261016 | 71 / 30 / 149 | 37 / 19 / 28 | 13 / 0 / 38 | 0 / 7 / 12 | 50 / 26 / 78 |
| new_disjoint | 20261026 | 53 / 76 / 131 | 27 / 30 / 31 | 2 / 0 / 26 | 0 / 16 / 14 | 29 / 46 / 71 |

Paired counts and 95% CIs below use task-ID matching, with target equality checked for every pair. CIs resample task IDs, retaining all three seeds and, for pooled results, both wording renderings in each cluster. Bootstrap: 2,000 replicates, seed 3197. Intervals are conditional on these three fixed seeds and two authored families; they do not estimate seed-population or unseen-paraphrase uncertainty. Per-operation intervals have 60 world clusters; all-operation intervals have 180.

Comparison reference: PR #75 `baseline`.

| Measure | Family | Kind | Reference → factorized | Improve / regress | Delta, pp | Task-cluster 95% CI, pp |
|---|---|---|---|---|---|---|
| exact | old_failing | all | 235 → 318 /540 | 131 / 48 | +15.37 | [+10.19, +20.56] |
| exact | old_failing | read | 139 → 127 /180 | 10 / 22 | -6.67 | [-13.89, +0.00] |
| exact | old_failing | explain | 70 → 109 /180 | 50 / 11 | +21.67 | [+15.00, +27.78] |
| exact | old_failing | compare | 26 → 82 /180 | 71 / 15 | +31.11 | [+21.67, +40.56] |
| exact | new_disjoint | all | 101 → 233 /540 | 164 / 32 | +24.44 | [+18.89, +30.00] |
| exact | new_disjoint | read | 86 → 93 /180 | 34 / 27 | +3.89 | [-3.33, +11.11] |
| exact | new_disjoint | explain | 15 → 90 /180 | 80 / 5 | +41.67 | [+31.67, +51.68] |
| exact | new_disjoint | compare | 0 → 50 /180 | 50 / 0 | +27.78 | [+20.00, +36.11] |
| exact | pooled | all | 336 → 551 /1080 | 295 / 80 | +19.91 | [+15.18, +24.54] |
| exact | pooled | read | 225 → 220 /360 | 44 / 49 | -1.39 | [-7.22, +4.44] |
| exact | pooled | explain | 85 → 199 /360 | 130 / 16 | +31.67 | [+24.17, +38.89] |
| exact | pooled | compare | 26 → 132 /360 | 121 / 15 | +29.44 | [+21.39, +37.22] |
| operation | old_failing | all | 348 → 540 /540 | 192 / 0 | +35.56 | [+30.74, +40.19] |
| operation | old_failing | read | 180 → 180 /180 | 0 / 0 | +0.00 | [+0.00, +0.00] |
| operation | old_failing | explain | 117 → 180 /180 | 63 / 0 | +35.00 | [+30.00, +40.00] |
| operation | old_failing | compare | 51 → 180 /180 | 129 / 0 | +71.67 | [+68.33, +75.56] |
| operation | new_disjoint | all | 152 → 419 /540 | 269 / 2 | +49.44 | [+43.89, +55.19] |
| operation | new_disjoint | read | 128 → 180 /180 | 52 / 0 | +28.89 | [+21.67, +36.67] |
| operation | new_disjoint | explain | 24 → 126 /180 | 104 / 2 | +56.67 | [+47.78, +66.11] |
| operation | new_disjoint | compare | 0 → 113 /180 | 113 / 0 | +62.78 | [+52.22, +72.78] |
| operation | pooled | all | 500 → 959 /1080 | 461 / 2 | +42.50 | [+38.43, +46.76] |
| operation | pooled | read | 308 → 360 /360 | 52 / 0 | +14.44 | [+10.83, +18.33] |
| operation | pooled | explain | 141 → 306 /360 | 167 / 2 | +45.83 | [+40.83, +50.56] |
| operation | pooled | compare | 51 → 293 /360 | 242 / 0 | +67.22 | [+61.11, +73.33] |

Comparison reference: PR #75 `all_task_diversity`.

| Measure | Family | Kind | Reference → factorized | Improve / regress | Delta, pp | Task-cluster 95% CI, pp |
|---|---|---|---|---|---|---|
| exact | old_failing | all | 317 → 318 /540 | 71 / 70 | +0.19 | [-4.26, +4.63] |
| exact | old_failing | read | 120 → 127 /180 | 17 / 10 | +3.89 | [-1.67, +10.00] |
| exact | old_failing | explain | 126 → 109 /180 | 14 / 31 | -9.44 | [-15.00, -3.89] |
| exact | old_failing | compare | 71 → 82 /180 | 40 / 29 | +6.11 | [-3.33, +16.11] |
| exact | new_disjoint | all | 141 → 233 /540 | 136 / 44 | +17.04 | [+11.67, +22.59] |
| exact | new_disjoint | read | 81 → 93 /180 | 32 / 20 | +6.67 | [-1.11, +14.44] |
| exact | new_disjoint | explain | 14 → 90 /180 | 79 / 3 | +42.22 | [+33.33, +51.11] |
| exact | new_disjoint | compare | 46 → 50 /180 | 25 / 21 | +2.22 | [-5.56, +10.01] |
| exact | pooled | all | 458 → 551 /1080 | 207 / 114 | +8.61 | [+4.81, +12.32] |
| exact | pooled | read | 201 → 220 /360 | 49 / 30 | +5.28 | [-0.28, +10.83] |
| exact | pooled | explain | 140 → 199 /360 | 93 / 34 | +16.39 | [+10.28, +22.78] |
| exact | pooled | compare | 117 → 132 /360 | 65 / 50 | +4.17 | [-2.78, +11.39] |
| operation | old_failing | all | 490 → 540 /540 | 50 / 0 | +9.26 | [+7.04, +11.85] |
| operation | old_failing | read | 180 → 180 /180 | 0 / 0 | +0.00 | [+0.00, +0.00] |
| operation | old_failing | explain | 171 → 180 /180 | 9 / 0 | +5.00 | [+2.22, +8.33] |
| operation | old_failing | compare | 139 → 180 /180 | 41 / 0 | +22.78 | [+18.33, +27.78] |
| operation | new_disjoint | all | 222 → 419 /540 | 243 / 46 | +36.48 | [+29.81, +43.15] |
| operation | new_disjoint | read | 99 → 180 /180 | 81 / 0 | +45.00 | [+35.00, +55.00] |
| operation | new_disjoint | explain | 28 → 126 /180 | 115 / 17 | +54.44 | [+45.00, +64.44] |
| operation | new_disjoint | compare | 95 → 113 /180 | 47 / 29 | +10.00 | [-1.67, +21.68] |
| operation | pooled | all | 712 → 959 /1080 | 293 / 46 | +22.87 | [+19.72, +26.02] |
| operation | pooled | read | 279 → 360 /360 | 81 / 0 | +22.50 | [+17.50, +27.50] |
| operation | pooled | explain | 199 → 306 /360 | 124 / 17 | +29.72 | [+24.44, +35.00] |
| operation | pooled | compare | 234 → 293 /360 | 88 / 29 | +16.39 | [+10.28, +22.50] |

Against the matched diversity arm, old-family total exact is 317→318/540 (+0.19 pp), while new-family total exact is 141→233/540 (+17.04 pp). Pooled exact is 458→551/1080 (+8.61 pp; CI +4.81 to +12.32), with 207 improvements and 114 regressions. Read is 201→220/360, explain 140→199, compare 117→132; read and comparison delta CIs include zero. Old-family comparison is 71→82/180, new-family comparison 46→50/180; comparison changes are not consistently positive across seeds. Old-family explanation exact regresses for every fixed seed (37→34, 40→34, 49→41), aggregating 126→109/180 (−9.44 pp, CI −15.00 to −3.89). Old-family seed 20261016 drops from 111 to 93 total exact targets. Pooled CI precision comes from task resampling and must be distinguished from the limited three-fixed-seed architecture evidence.

Against the raw baseline, pooled exact is 336→551/1080, but read is 225→220/360. Thus the factorized candidate is promising as a bounded follow-up, with explanation/new-wording gains, but cannot yet be presented as removing the incumbent read cost or achieving robust comparison transfer.

Runtime 40.844 seconds for fitting, restored-checkpoint evaluation, ID invariance checks and bootstrap. CPU only, 2 intraop threads / 1 interop thread, CUDA hidden, all models and tensors on CPU. Cost $0. No GPU, network, Index, held-out data, teacher API, production planner changes, push or PR.

Checks: exact data/source/reference-observation hashes verified at run start; all three final checkpoints saved and restored; semantic predictions match exactly after candidate ID/order permutation in both families for every checkpoint; all seeds complete the frozen 300-step budget and runtime remains below 180 seconds.

Artifacts: `preregistration.json`, `receipt.json`, `observations.json`, `per_seed_paired.json`. Frozen rows, read-only reference observation copy, checkpoints and run log are preserved in `C:/Users/apumu/research/CHLabs/ssdm-data/l0-factorized-query-target-20261006-run1/`. Checkpoint paths and SHA256 hashes are in the receipt.

Local reproduction after prepare/freeze (use a fresh output directory for a new experiment; no extra run was performed):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.factorized_query_target_probe run --out C:/Users/apumu/research/CHLabs/ssdm-data/NEW-PREPARED-AND-FROZEN-DIRECTORY
```

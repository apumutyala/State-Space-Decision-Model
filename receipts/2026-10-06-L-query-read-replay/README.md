# Single 50/50 read-replay condition

The joint hypothesis is not supported: this exact replay allocation does not establish recovery of the old-family read regression, and it reduces new-family comparison transfer. Fold this condition and stop; no follow-up variant was run. This is a synthetic query-target diagnostic, not learned L0 qualification or production integration.

Source and preregistration were frozen in commit `3c05ff6` before fitting. The branch starts from `4733f785e5284ea959d4d640fff884136487bd05` (`origin/agent-main`), independently of PR #75. Its model/target code is copied from PR #75 commit `64df83efeafa08f43bebf6e2e5e59a1ff5baf77e`; the model, projection, candidates, tensor construction and evaluator AST match that source exactly. The 27,937-parameter width-32 byte diagonal SSM starts with random weights.

Only three new treatment checkpoints were trained: seeds 20261006, 20261016 and 20261026, 600 training worlds at dataset seed 20261006, 300 steps, batch 32, Adam learning rate 0.003 and gradient clipping 1.0. The first 100 read worlds in generator order retain original wording; the last 100 use their existing six-template assignments. The diverse template list overlaps original wording, so 13 diverse-assigned read rows also happen to equal baseline text: 100 original-assigned / 100 diverse-assigned, 113 baseline-identical read texts. All 200 explain and 200 compare rows exactly match the all-diverse recipe. Targets, worlds, candidates and sampled indices stay matched to both controls. The fixed final-step checkpoints are used; no validation tuning or checkpoint selection.

The six exact baseline/all-diverse control checkpoints, prior raw predictions and receipts were imported from `C:/Users/apumu/research/CHLabs/ssdm-data/l0-query-template-transfer-20261006-run1/`; none was retrained. Their SHA256s are frozen in preregistration. Full frozen validation-row equality, including IDs, requests, candidate sets and labels, was verified before pairing, and each imported prediction was checked against the matching ID and target. The new artifact directory also preserves independent copies of these control artifacts.

Validation repeats the prior 180 fresh worlds at dataset seed 20261009 (60 per kind), with disjoint training/validation names and ranges, in the same two frozen wording families. The exact templates and data hashes are in preregistration.json. Each cell below is baseline / all-diverse / replay; kind cells are out of 60, total cells out of 180.

| Wording | Seed | Read | Explain | Compare | All |
|---|---|---|---|---|---|
| old_failing | 20261006 | 38 / 36 / 37 | 40 / 37 / 40 | 0 / 15 / 21 | 78 / 88 / 98 |
| old_failing | 20261016 | 51 / 40 / 45 | 30 / 40 / 46 | 25 / 31 / 29 | 106 / 111 / 120 |
| old_failing | 20261026 | 50 / 44 / 42 | 0 / 49 / 51 | 1 / 25 / 28 | 51 / 118 / 121 |
| new_disjoint | 20261006 | 22 / 32 / 38 | 0 / 14 / 11 | 0 / 23 / 21 | 22 / 69 / 70 |
| new_disjoint | 20261016 | 37 / 19 / 25 | 13 / 0 / 0 | 0 / 7 / 4 | 50 / 26 / 29 |
| new_disjoint | 20261026 | 27 / 30 / 34 | 2 / 0 / 0 | 0 / 16 / 2 | 29 / 46 / 36 |

The paired counts below aggregate the three fixed seeds. Positive deltas favor replay. Family/kind counts are out of 180, pooled/all out of 1080.

| Reference | Wording / kind | Reference → replay | Improved / regressed | Delta, pp | Task-cluster 95% CI, pp |
|---|---|---|---|---|---|
| baseline | old_failing / read | 139 → 124 | 16 / 31 | -8.33 | [-17.78, +0.56] |
| baseline | old_failing / explain | 70 → 137 | 74 / 7 | +37.22 | [+32.22, +42.78] |
| baseline | old_failing / compare | 26 → 78 | 58 / 6 | +28.89 | [+20.56, +37.22] |
| baseline | new_disjoint / read | 86 → 97 | 39 / 28 | +6.11 | [-3.33, +16.67] |
| baseline | new_disjoint / explain | 15 → 11 | 11 / 15 | -2.22 | [-7.22, +1.67] |
| baseline | new_disjoint / compare | 0 → 27 | 27 / 0 | +15.00 | [+10.00, +20.56] |
| baseline | pooled / all | 336 → 474 | 225 / 87 | +12.78 | [+8.61, +16.67] |
| all_task_diversity | old_failing / read | 120 → 124 | 7 / 3 | +2.22 | [-1.67, +6.11] |
| all_task_diversity | old_failing / explain | 126 → 137 | 18 / 7 | +6.11 | [+0.56, +11.11] |
| all_task_diversity | old_failing / compare | 71 → 78 | 13 / 6 | +3.89 | [-1.11, +8.89] |
| all_task_diversity | new_disjoint / read | 81 → 97 | 18 / 2 | +8.89 | [+3.89, +13.89] |
| all_task_diversity | new_disjoint / explain | 14 → 11 | 0 / 3 | -1.67 | [-3.89, +0.00] |
| all_task_diversity | new_disjoint / compare | 46 → 27 | 2 / 21 | -10.56 | [-15.00, -6.11] |
| all_task_diversity | pooled / all | 458 → 474 | 58 / 42 | +1.48 | [-0.19, +3.15] |

Exact CI algorithm: for each comparison, form paired correctness deltas (replay minus reference) for every validation task ID across all three fixed seeds. For pooled summaries, keep both wording families within the same task cluster (six deltas per cluster); family-specific clusters contain three seed deltas. Average within each cluster. Independently for each reported comparison, initialize NumPy default_rng(3197), draw with replacement 2,000 arrays of task-cluster indices of the original cluster count, compute each mean, and take NumPy's default linear 2.5th/97.5th percentiles. Per-kind intervals have 60 task clusters, all-kind intervals 180. These CIs condition on the three fixed seeds and two authored wording families; they do not estimate uncertainty over unseen seeds or paraphrases. No multiplicity correction is applied. The same world rendered twice is never treated as two independent tasks. Per-seed paired improvement/regression counts are preserved in per_seed_paired.json.

Old-family read recovers only four of the 19 aggregate correct answers lost by all-diverse training (120→124, baseline139); replay-versus-diverse CI crosses zero. Replay remains below baseline for every seed (38→37, 51→45, 50→42); its baseline CI also crosses zero and is not evidence of equivalence. Old-family comparison still improves over baseline (26→78), but comparison retention across both families fails: new-family comparison declines from 46 to 27 versus all-diverse, with negative paired CI and losses in every seed (23→21, 7→4, 16→2). New-family comparison is nevertheless above baseline's zero. The condition improves new-family read, but that secondary benefit does not meet the preregistered joint aim. This folds only the exact first-100-original/last-100-diverse 600-world/300-step recipe; it does not falsify other replay allocations, broader diversity training or the model architecture.

Training/evaluation/bootstrap runtime: 43.969 seconds. CPU only, two intraop threads and one interop thread; installed Torch 2.8.0+cu128, CUDA hidden and all models/tensors CPU. Cost $0. No network during fitting, GPU, Index/held-out data, teacher API or production changes. All three replay checkpoints were reloaded, and semantic observations matched exactly after ID permutation in both families. The control receipt preserves the corresponding six-checkpoint checks.

Artifacts:

- `preregistration.json`: config, exact templates/allocation IDs, row hashes, source hash and imported-control hashes.
- `receipt.json`: three new fits plus explicitly marked imported control summaries, checkpoint hashes, paired counts/CIs and runtime.
- `observations.json`: raw task-level predictions for all three arms, all three seeds and both families, with IDs and targets.
- `per_seed_paired.json`: per-seed/wording/kind reference and replay counts plus improved/regressed counts.
- `C:/Users/apumu/research/CHLabs/ssdm-data/l0-query-read-replay-20261006-run1/`: frozen rows, three new checkpoints, complete imported controls, raw predictions/receipts, freeze commit and run log.

Independent raw-count and CI recomputation is required before merge. No second fit or variant is needed for that review. There is no repository CI workflow. Local verification passed: `python -m unittest language_growth.tests.test_development_data -v` (12 tests); model/projection/candidate/evaluator AST equality; frozen-source and artifact/checkpoint SHA256 checks; full validation-row identity; replay-only text mutation; and exact ID-permutation observations after reload.

Reproduce with preserved controls available and a new unused output directory:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.query_read_replay_probe prepare --out C:/Users/apumu/research/CHLabs/ssdm-data/NEW-UNUSED-DIRECTORY
# Freeze the new source/registration before launching.
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.query_read_replay_probe run --out C:/Users/apumu/research/CHLabs/ssdm-data/NEW-UNUSED-DIRECTORY
```

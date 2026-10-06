# All-task query wording transfer

One paired CPU experiment supports improved comparison transfer from train-only wording diversity, with a read tradeoff and substantial seed/template sensitivity. This is a small synthetic target-selection diagnostic, not learned L0 qualification or production integration. No follow-up variant was run. The comparison hypothesis is not falsified, so no negative-results LEDGER fold was added. Keep this local for coordinator review before any PR.

The model and both arms were frozen in commit `8a1be71` before training. Base: `4733f785e5284ea959d4d640fff884136487bd05` (`origin/agent-main` at worktree creation). The model/target implementation was copied from `fcf620177d8e9b1575e5e0ed590d4aae1a9cb3b3`; this branch has no dependency on that experimental branch. The model class is unchanged: 27,937 parameters, width 32, shared byte diagonal SSM and candidate-interaction scorer. All weights start random.

Both arms use the same 600 training worlds (seed 20261006), 300 optimizer steps, batch size 32, Adam learning rate 0.003, gradient clipping at 1.0, and paired initialization/sampling seeds 20261006, 20261016, 20261026. Baseline uses the generator's original two templates per kind. The augmentation replaces only task_input with six fixed templates per kind, assigned cyclically across each kind's 200 worlds (34/34/33/33/33/33 examples). Candidate strings, labels, IDs, world evidence and sampled training indices match within every pair. Thus there are 600 training examples in each arm, with no oversampling. Each arm completes its full training budget before that checkpoint is evaluated; only the fixed final-step checkpoint is used.

Validation uses 180 fresh worlds at seed 20261009: 60 read, 60 explain, 60 compare. Names and value ranges are disjoint from training. The same validation worlds are rendered in two fixed, disjoint wording families. `old_failing` includes the original comparison wording, "For October 2026, compare {left}'s and {right}'s {field}." `new_disjoint` includes "With {left} first and {right} second, compare their October 2026 {field}." Neither family selects checkpoints, hyperparameters or templates. Every template is in preregistration.json.

Exact-target counts below are baseline → augmentation, each out of 60 per kind per seed. Targets jointly select operation, field, entity/operand order and temporal scope; values are never predicted. The temporal scope is constant here.

| Wording | Seed | Read | Explain | Compare | All /180 |
|---|---|---|---|---|---|
| old_failing | 20261006 | 38 → 36 | 40 → 37 | 0 → 15 | 78 → 88 |
| old_failing | 20261016 | 51 → 40 | 30 → 40 | 25 → 31 | 106 → 111 |
| old_failing | 20261026 | 50 → 44 | 0 → 49 | 1 → 25 | 51 → 118 |
| new_disjoint | 20261006 | 22 → 32 | 0 → 14 | 0 → 23 | 22 → 69 |
| new_disjoint | 20261016 | 37 → 19 | 13 → 0 | 0 → 7 | 50 → 26 |
| new_disjoint | 20261026 | 27 → 30 | 2 → 0 | 0 → 16 | 29 → 46 |

| Wording / kind | Counts across three seeds | Improved / regressed | Delta, percentage points | Task-cluster 95% CI, pp |
|---|---|---|---|---|
| old_failing / read | 139 → 120 /180 | 11 / 30 | −10.6 | [−19.5, −1.7] |
| old_failing / explain | 70 → 126 /180 | 65 / 9 | +31.1 | [+26.1, +35.6] |
| old_failing / compare | 26 → 71 /180 | 48 / 3 | +25.0 | [+17.8, +32.2] |
| new_disjoint / read | 86 → 81 /180 | 30 / 35 | −2.8 | [−12.2, +7.2] |
| new_disjoint / explain | 15 → 14 /180 | 14 / 15 | −0.6 | [−5.6, +3.9] |
| new_disjoint / compare | 0 → 46 /180 | 46 / 0 | +25.6 | [+18.3, +32.8] |
| Both / all | 336 → 458 /1080 | 214 / 92 | +11.3 | [+6.9, +15.5] |

The CI resampling unit is the authored validation task/world ID. For the pooled result, each of 180 clusters contains six paired correctness deltas: both wordings × all three training seeds. A cluster's value is their mean. Resample 180 cluster values with replacement, take the mean, repeat 2,000 times at bootstrap seed 3197, and use the 2.5th/97.5th percentiles. Family-specific CIs retain all three seeds within each task cluster; per-kind CIs resample the 60 task clusters of that kind. These intervals are conditional on these fixed training seeds and these authored wording families, not uncertainty over new training seeds or new paraphrases. Per-seed results above expose variability that task-only bootstraps cannot represent. The two wording renderings of a world are never treated as independent tasks.

Comparison improves for every fixed seed on both wording families, but its augmented accuracy remains modest (71/180 and 46/180). The old-family read decline occurs in all three seeds. Seed 20261016 loses total accuracy on the new family, and explanation transfer remains poor there. The experiment therefore supports this specific comparison-transfer hypothesis while exposing an unresolved read cost; it does not establish robust broad language transfer or justify replacing the incumbent recipe.

Total training/evaluation/bootstrap runtime: 69.031 seconds. CPU only, two intraop threads and one interop thread, Torch 2.8.0+cu128 installed but CUDA hidden and all models/tensors explicitly CPU. Cost $0. No network, GPU, Index/test data, teacher API, production planner/interface changes, push or PR.

Artifacts:

- `preregistration.json`: frozen config, template lists, data hashes and model source SHA256.
- `receipt.json`: full per-seed/arm summaries, losses, checkpoint paths/hashes and paired CIs.
- `observations.json`: every raw exact-target prediction from the six restored checkpoints, with task IDs and targets.
- `per_seed_paired.json`: per-seed/kind/family improvement and regression counts.
- `C:/Users/apumu/research/CHLabs/ssdm-data/l0-query-template-transfer-20261006-run1/`: preserved frozen rows, six checkpoints, output log and copies of the receipt/predictions. Total approximately 12.4 MB; checkpoint hashes are in receipt.json.

Verification already performed: model AST matches the source model; all 600 training pairs differ only in task_input; semantic predictions match exactly after ID permutation for both validation families in all six restored checkpoints. No extra run or post-hoc audit is needed for the reported result.

Reproduce in a fresh output directory from the experiment worktree:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.query_template_transfer_probe prepare --out C:/Users/apumu/research/CHLabs/ssdm-data/NEW-UNUSED-DIRECTORY
# Preserve/review the new preregistration before launching.
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.query_template_transfer_probe run --out C:/Users/apumu/research/CHLabs/ssdm-data/NEW-UNUSED-DIRECTORY
```

# L0 development-data preparation

This is CPU preparation for the L0h pilot in `L0_PHASES.md`, not promotion to a learned level or access to an audit split.

- Hypothesis: varying task wording, entities, option order and independently sampled record values can supply supervised typed-choice examples without hardcoding task answers in option IDs or descriptions.
- Deliverable: a deterministic development-only synthetic task generator using the canonical Fact/Compare/Assert/ExplainDerivation shapes and current host catalog format. Labels and reference programs remain evaluator/training metadata, separate from model-facing task/evidence.
- Exit proof: deterministic seeded reproduction; generated supported programs pass the kernel; counterfactual value changes alter labels when required; decision views exclude targets/programs; no Index, held-out or external teacher data is accessed. Record hashes before using generated data in any experiment.
- Decision unlocked: review a concrete language pilot training/evaluation configuration and estimate its qualification cost after prior CPU proof merges. These examples cannot support held-out generalization claims. No model training, GPU or paid job is authorized by this preparation.

Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-development-data`, branch `codex/l0-development-data`. Code stays in `language_growth/`; do not redefine or overwrite the joint qualification fixtures.

## Complementary nuisance intervention (6 October 2026)

`comparison_nuisance_counterfactual` accepts generated development comparisons and changes only the unasked catalog field's values. Both values advance by one month or one cent with wrapping inside the original split's range, so even equal nuisance values produce changed evidence. Queried fields, entity/operand order, task wording, questions/options, targets and checked output meanings stay fixed. Rebuilding through the generator checks the reference and assigns evidence-bound context/request identities. The paired task ID stays outside model requests; training labels and reference programs remain under `training_only`.

This complements `comparison_swap_counterfactual`, which changes the queried values. The pair provides an input invariance control for later training/evaluation; it does not establish neural causal mediation, use a held-out split or promote L0h. Existing manifests and generated receipts are unchanged.

CPU exit proof: `python -m unittest language_growth.tests.test_development_data -v` passes 14 tests, including 240 paired comparisons across development_train/development_validation, production/augmented choices and both repair settings. The proof checks all three queried relations, both queried fields, equal nuisance values, range wrapping, accepted reference/output equality, input immutability, changed identities and model-request separation. Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-nuisance-intervention`; branch: `codex/l0-nuisance-intervention`. No model, Torch, GPU, external data or paid run is involved.

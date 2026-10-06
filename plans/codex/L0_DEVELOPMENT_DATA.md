# L0 development-data preparation

This is CPU preparation for the L0h pilot in `L0_PHASES.md`, not promotion to a learned level or access to an audit split.

- Hypothesis: varying task wording, entities, option order and independently sampled record values can supply supervised typed-choice examples without hardcoding task answers in option IDs or descriptions.
- Deliverable: a deterministic development-only synthetic task generator using the canonical Fact/Compare/Assert/ExplainDerivation shapes and current host catalog format. Labels and reference programs remain evaluator/training metadata, separate from model-facing task/evidence.
- Exit proof: deterministic seeded reproduction; generated supported programs pass the kernel; counterfactual value changes alter labels when required; decision views exclude targets/programs; no Index, held-out or external teacher data is accessed. Record hashes before using generated data in any experiment.
- Decision unlocked: review a concrete language pilot training/evaluation configuration and estimate its qualification cost after prior CPU proof merges. These examples cannot support held-out generalization claims. No model training, GPU or paid job is authorized by this preparation.

Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-development-data`, branch `codex/l0-development-data`. Code stays in `language_growth/`; do not redefine or overwrite the joint qualification fixtures.

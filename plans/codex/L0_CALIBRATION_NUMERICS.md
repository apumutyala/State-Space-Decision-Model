# Feature-temperature numeric repair

- Hypothesis: a calibration identity must distinguish the exact parameters used at inference, and every served or trained temperature must be finite and positive. A custom minimum temperature must not be labelled identity unless it actually leaves probabilities unchanged.
- Deliverable: narrow corrections to shared feature-temperature identity, initialization and numeric checks in the training and serving paths, with bounded adapter failure on invalid derived temperatures.
- Exit proof: synthetic CPU regressions for the saved PR #29 counterexamples, ordinary temperature parity, and a mixed-option-count gradient case; no cohort, model-weight, Index or held-out data is read.
- Decision unlocked: a calibration configuration that can be pinned honestly in the D→L parent revision. This does not qualify a model or authorize paid work.

Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-calibration-numerics`, branch `codex/l0-calibration-numerics`. Keep the patch within calibration math/configuration and its focused tests.

# Parent revision identity repair

- Hypothesis: the decision backend's default revision can collide across different trained heads or LoRA weights because checkpoint metadata is optional.
- Deliverable: bind the revision to a deterministic digest of the actual head, typed-output and LoRA state, including tensor names, shapes and dtypes, plus explicit behavior-changing settings of supported heads, padding and fork mode. Keep the existing frozen-backbone identity and its disclosed sampled-weight mode unchanged.
- Exit proof: tiny CPU regressions show different weights or supported head settings change the revision, equal weights and settings reproduce it, and freezing gradients does not change it. Run focused backend tests without loading a model snapshot.
- Decision unlocked: a checkpoint/revision identity suitable for the L planner's exact parent pin, subject to cross-review. This CPU change makes no learned-capability or model-quality claim.

Worktree: `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-parent-identity`; branch `codex/l0-parent-identity`. Keep the code change local to `DecisionModelBackend.revision()` while PR #29 develops other backend features.

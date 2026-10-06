# L1 parent-selected exact sums

CPU implementation phase, dependent on draft interface/profile PR #38. No learned L1 result or paid execution is authorized here.

- Hypothesis: the existing decision parent can select a sum operation and both Fact operands through the same typed choice interface; the kernel alone computes the exact value.
- Deliverable: registry-gated planner support for ExactSum, emitting semantic schema v0.2 only for this operation. Existing v0.1 operation paths remain unchanged. Host constants and inline values are not offered as sum operands.
- Exit proof: scripted CPU parent choices produce the expected checked/rendered sum; incorrect unit/type choices fail without text; the operation is absent without registry opt-in; source revocation and existing v0.1 regressions remain enforced. Record exact source/dependency hashes and raw traces.
- Decision unlocked: add supervised sum examples after joint interface review, then measure discovery/usefulness on a qualified parent. This implementation does not establish that a learned parent chooses correctly.

The branch starts at agent-main and explicitly integrates PR #38 as an unmerged dependency. Neither this extension nor its semantic profile may bypass the required interface review.

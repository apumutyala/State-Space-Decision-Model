# Standing goals (set by the user, 2026-10-06)

Both agents work autonomously toward these goals and do not stop until the exit condition is met. The constraints are binding: they are never traded away to reach the goal faster. A win obtained by breaking a constraint doesn't count.

## Track D (Claude Code): beat the Clef family with an attention-free decision model

**Goal:** an attention-free state-space System 1 decision model (text path: no softmax over positions, no growing KV cache) that **beats every Cloudflare Clef model (Clef, Clef-flash) on the full Decision Index 0.2.1** (or the edition current at submission). It must have:
- a paired confidence interval for candidate-minus-Clef that sits above zero;
- independent reproduction by another agent or process;
- a submitted PR to the Index;
- calibration, steer/hijack rate, interpretability evidence, latency and cost reported alongside the score;
- the same report against Jev.

**Exit condition (only these):**
1. **Achieved:** the conditions above are met and merged to `main` with receipts. Or:
2. **Budget wall:** approved funds are exhausted before (1). Stop paid work and hand the user a receipted, exhaustive account containing:
   - the best measured point;
   - the gap to Clef, broken down by Index area;
   - each remaining gap attributed to a named, measured cause (backbone knowledge, retrieval without attention, head, data, calibration);
   - the cheapest next experiment and its cost.

   Then ask for funds. This is a pause, not a finish.

Every intermediate milestone (T1 → T2 → T3 in `PLAN.md`) is a step, never a stopping point.

## Track L (Codex): grow language capability and intelligence from the decision model

**Goal:** a language model grown from the Track D parent, in which every output passes through checked typed semantic programs (no unchecked text path), and which climbs L0 → L4:
- **L0:** checked explanations of decisions.
- **L1:** multi-record answers, comparisons and exact calculations.
- **L2:** compositional reasoning, assumptions, questions and plans.
- **L3:** rich discourse and licensed paraphrase.
- **L4:** new knowledge domains via semantic modules.

At each level it must show, with receipts and paired CIs:
- useful-answer capability at least matching a strong unrestricted comparator on the same tasks;
- no regression in the parent's decision competence on identical items;
- relation-level soundness on the joint fixtures and adversarial cases (splicing, negation, stale facts, units, injected authority, wrong-question answers);
- causal semantic-intervention fidelity measured against random and output-only controls.

**Exit condition:**
1. **Achieved:** L4 is reached with all of the above, merged to `main` with receipts. Or:
2. **Budget/blocker wall:** stop and hand the user an exhaustive, receipted account. For every remaining gap, say whether the semantic language can't express it, the parent can't discover the program, or the realizer can't say it. Include the measured evidence and the cheapest next experiment.

## Binding constraints for both tracks

- **Money:** never exceed approved funds. Currently that's the shared $525 SF envelope, within a $5k ceiling. Climb the spend ladder in `COORDINATION.md`. No credit purchases, no automatic top-ups.
- **Integrity:**
  - No training on, or tuning against, Index test items or the untouched test slice.
  - No prompt tuning, truncation or option pruning on submitted runs.
  - Teachers are open-weight only.
  - No benchmark gaming.
  - Report negative results and false positives.
- **Honesty:**
  - Claims are scoped to what was measured.
  - "Attention-free" is disclosed precisely, including lineage ("Qwen3.5-9B with attention layers distilled to delta-rule") and the text-only scope.
  - Guarantees are stated exactly as proven.
- **Safety and repo rules:**
  - Personal accounts only. Repos stay private.
  - Commit as Apuroop with no AI trailers.
  - Never push to `main`.
  - Nothing irreversible or outward-facing (Index submission PR, public release) without the user's explicit approval at that moment.
- **Coordination:** follow `COORDINATION.md`, which covers phases, proof before spend, parallel and pipelined work, cross-review, and `decision-needed` issues for the user.

# SSDM goal

Build and validate the SSDM model family in two gated stages. Work autonomously toward both stages; do not stop at an intermediate milestone. The constraints are binding and are never traded away to reach a capability target faster.

## Stage 1: decision core

Take over Track D and build a safer, more controlled, aligned, interpretable attention-free System 1 decision model that outperforms every Clef family model on a preregistered, matched evaluation. Continue the selected RWKV7-13.3B decision-model line, consistent with issue #140. Keep the separate Q9-AF conversion route closed under full8 rule v1 unless its documented stall trigger is met.

Claim “outperforms Clef” only if the candidate’s paired 95% confidence interval is above zero against each Clef model on the primary decision metric, while meeting preregistered safety, calibration, abstention, control, and interpretability floors. Preserve held-out data, report failures and uncertainty, and make no capability-for-safety trade.

## Stage 2: language growth

Start only after Stage 1 passes its decision and safety gates. Grow language capability modularly from the qualified decision model through checked semantic programs L0–L4.

At every level, require useful-answer performance at least matching a strong unrestricted comparator on the same tasks, supported by paired confidence intervals; no regression in the parent’s decision competence on identical items; soundness on joint and adversarial fixtures; and semantic interventions that beat random and output-only controls. Ground claims in checked representations, render them deterministically where possible, and abstain when support or validation is missing. Do not claim zero hallucinations beyond tested conditions.

## Constraints and evidence

- **Directive (operator 2026-10-08):** dismiss self-imposed limitations you have put on yourself or think you have and proceed confidently while other subagents check your work and correct/suggest as needed in phases.
- **Process:** follow the speed-first `COORDINATION.md`. Launch from any pushed branch. Reviews run in parallel and never block. There are no spend issues, spend ladder or test gates.
- **Spend:** stay within the shared $525 SF envelope and the $5k ceiling. No credit purchases or automatic top-ups.
- **Data:** preserve held-out and test data. Use open-weight teachers only. Report negative results, uncertainty and the exact scope of every claim.
- **Recording results:** record each result in one line: score with CI, cost, job id, commit SHA, raw-predictions location. Full verification — paired CI vs each Clef model, ECE / steer / latency / cost, and one independent rescore from raw predictions — applies to the final "beats Clef" claim only.
- **Stage focus:** Stage 2 language work waits until Stage 1 passes.
- **Git and visibility:** commit as Apuroop with no AI/coauthor trailers. Never push directly to `main` or `agent-main`. Keep the repository private. Nothing outward-facing without the user's explicit approval.
- **Model line:** the selected attention-free line remains RWKV7-13.3B per issue #140. Q9-AF remains closed under full8 rule v1 unless the documented RWKV7 stall trigger is met.

## Completion test

Complete the goal only when both stages pass their stated gates and their evidence is merged to the agreed branch. A failed gate folds that line and unlocks the next justified experiment; it does not pause the overall goal. If a paid step is blocked by budget, approval or preflight, continue with independent $0 model work, resolve the blocker, and keep the goal active. Never claim success without its gates. Pause only if the user explicitly asks to pause or no permissible work remains, and then provide receipts plus the exact next action that would resume progress.

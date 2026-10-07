# SSDM goal

Build and validate the SSDM model family in two gated stages. Work autonomously toward both stages; do not stop at an intermediate milestone. The constraints are binding and are never traded away to reach a capability target faster.

## Stage 1: decision core

Take over Track D and build a safer, more controlled, interpretable attention-free System 1 decision model that outperforms every Clef family model on a preregistered, matched evaluation. Continue the selected RWKV7 line under full8 rule v1, consistent with the recorded choice in issue #140.

Claim “outperforms Clef” only if the candidate’s paired 95% confidence interval is above zero against each Clef model on the primary decision metric, while meeting preregistered safety, calibration, abstention, control, and interpretability floors. Preserve held-out data, report failures and uncertainty, and make no capability-for-safety trade.

## Stage 2: language growth

Start only after Stage 1 passes its decision and safety gates. Grow language capability modularly from the qualified decision model through checked semantic programs L0–L4.

At every level, require useful-answer performance against a strong comparator, no regression in the parent’s decision competence, soundness on joint and adversarial fixtures, and semantic interventions that beat random and output-only controls. Ground claims in checked representations, render them deterministically where possible, and abstain when support or validation is missing. Do not claim zero hallucinations beyond tested conditions.

## Constraints and evidence

- Follow `COORDINATION.md`, the repo’s review and data rules, and the staged spend ladder. Do not launch paid work without its required merged proof and acknowledged spend issue. Stay within the shared $525 SF envelope and the $5k ceiling; no credit purchases or automatic top-ups.
- Preserve held-out and test data. Use open-weight teachers only. Report negative results, uncertainty, and the exact scope of every claim.
- For every reported result, preserve the code and configuration, data and model hashes, seeds, hardware, cost, raw predictions, and reproducible evaluation receipt. Another agent independently remeasures reported results before merge.
- Keep low-cost language prerequisites moving only when they do not distract from Stage 1.
- Commit as Apuroop with no AI/coauthor trailers; never push directly to `main` or `agent-main`. Keep the repository private. Nothing outward-facing without the user’s explicit approval.
- The selected attention-free line remains RWKV7-13.3B under full8 rule v1. Reopen Q9-AF only under a new preregistered rule and spend issue, and only if the documented RWKV7 stall trigger is met.

## Completion test

Complete the goal only when both stages pass their stated gates and their evidence is merged to the agreed branch. A failed gate folds that line and unlocks the next justified experiment; it does not pause the overall goal. If a paid step is blocked by budget, approval or preflight, continue with independent $0 model work, resolve the blocker with a receipted spend proposal, and keep the goal active. Never claim success without its gates. Pause only if the user explicitly asks to pause or no permissible work remains, and then provide receipts plus the exact next action that would resume progress.

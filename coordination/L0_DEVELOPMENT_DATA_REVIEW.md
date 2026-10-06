# Development generator handoff

The generator supplies synthetic development examples only. It reads no benchmark, external teacher or audit data. Current labels and reference programs are training metadata; successive supervised input rows use gold prior choices. This is intentional teacher forcing, not an autonomous planner result. The free-running runner must reconstruct initial task/evidence and call the parent for its own choices.

Internal review found a train/serve mismatch in option wording and enumeration. The production cohort now follows the typed planner; shuffled options remain explicitly labelled augmentation. The coordinator also repaired reused context identities: evidence changes now produce a new context and request identity. Counterfactual price swaps preserve question options while changing the appropriate relation target.

Current coverage is two-record worlds, reads, explanations of reads and unequal-price comparisons. Equal-price comparisons, Ask and richer compositions are not represented adequately yet; this generator alone is not the full curriculum. Do not infer held-out transfer or learned competence from its unit tests.

Dependencies are kernel/types from PR #10 and parent_bridge from PR #13; they are borrowed locally, not duplicated in this PR. The receipt records their hashes. `plans/codex/L0_DEVELOPMENT_DATA.md` was written before implementation. Exact experiment data/seed and hashes must be frozen before a paid job, which still requires the shared spend gate and merged prerequisites.

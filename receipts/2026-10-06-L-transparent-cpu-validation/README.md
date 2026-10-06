# Frozen development-validation check: provisional observed counts

Source `d5663891e26defaa2e450511cf1a55437b3cd1a5`, clean when measured.
`receipt.json` records the exact command/config, hashes, hardware, timing and
all three seeds. Training is unchanged: 13,332 original development-train
records, 200 full-batch epochs, learning rate 2, L2 1e-4, seeds 17/29/43. Every
checkpoint hash exactly matches the original development-train experiment.
No model/config tuning was performed after validation outcomes were observed.

Evaluation uses 300 freshly generated `development_validation` tasks, seed
314159, with disjoint entity names, numeric ranges and wording, including
150 production and 150 shuffled choices. This remains iterative development
validation, not a sealed held-out/audit or Decision Index evaluation. GPU,
downloads and paid services were unused; billed rate and incremental cost $0.

| Provisional raw measurement | Seeds 17 / 29 / 43 |
| --- | --- |
| Operation slot | 200/300 each |
| Fact slot | 200/200 each |
| Left, right, relation slots | 100/100 each |
| Relation without engineered numeric sign | 43/100 each |
| One-shot free-running task-correct | 200/300 each |
| One-shot free-running kernel-accepted | 300/300 each |
| Initial-weight free-running task-correct | 0/300 each |
| Unsupported, provider-error, runner-error | 0 each |

Raw task-correct counts have median 200, range 200–200 across initializations.
Fit timing median is 1.2886 seconds, range 1.2213–1.6970 seconds. Individual
slot evaluation is teacher-forced; complete programs are free-running with
the model's own prior selections, no repair feedback and no gold prefixes.

All 100 read and 100 explanation tasks are correct; all 100 comparison tasks
are wrong. The learned operation scorer proposes a factual assertion for
each comparison request. The kernel accepts that supported fact, and the
relevance scorer correctly retains a task failure. This is provisional error
localization from preserved traces, not a generalization claim; independent
remeasurement must precede reporting a measured conclusion.

Raw slot logits contain per-feature contributions. Every free-running raw
request/response records the same feature-to-logit accounting in backend
diagnostics, including the initial-model controls. These are linear accounting
contributions, not causal attribution or proofs. Complete task/evidence,
predictions, statuses and outputs are losslessly compressed in `.gz`; hashes
in the receipt apply to decompressed bytes. JSON checkpoints are included.
Uncompressed originals remain at
`C:/Users/apumu/research/CHLabs/ssdm-data/l0-transparent-cpu-validation-20261006/`.

The separate original development-train receipt remains unchanged under
`../2026-10-06-L-transparent-cpu-baseline/`; results must never be pooled. This
experiment does not establish raw-language arithmetic, shared-parent retention,
calibration, neural causal mediation, an unrestricted-comparator win or an L0
qualification gate.

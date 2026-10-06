# Learned feature scorer, CPU development measurement

Source commit: `d2963fea2499637a477014d9b3608b86dfe8d4ed`, clean when run.
Exact command, CPU identity, config, dataset hashes, seeds and per-run outputs
are in `receipt.json`. Paid cost and billed rate were $0. The source has five
targeted tests (`python -m unittest language_growth.tests.test_transparent_cpu -v`).

The model trains a softmax linear option scorer using the existing 5,000-task,
13,332-record synthetic `development_train` cohort. Evaluation uses 300 newly
generated same-domain iterative `development_train` tasks (seed 314159), with
150 production and 150 shuffled-choice tasks. There is no untouched held-out
evaluation, audit/Index access, pretrained model, GPU or spend.

| Measurement | Fitted seeds 17 / 29 / 43 | Initial seeds 17 / 29 / 43 |
| --- | --- | --- |
| Operation slot | 300/300 each | 97 / 99 / 135 of 300 |
| Fact slot | 200/200 each | 0/200 each |
| Left slot | 100/100 each | 0/100 each |
| Right slot | 100/100 each | 0/100 each |
| Relation slot | 100/100 each | 44 / 0 / 40 of 100 |
| Free-running task-correct | 300/300 each | 0/300 each |
| Free-running kernel-accepted | 300/300 each | 272 / 97 / 149 of 300 |

Fitted task-correct count median is 300, range 300–300 over three initializations.
Fitting takes median 1.2163 seconds, range 1.1618–1.2780 seconds. Slot measurements
are teacher-forced; the existing one-shot planner independently supplies its own
previous predictions during free-running evaluation. Task correctness is checked
separately from kernel acceptance. Checkpoint JSON is reloaded before scoring.

Numerical-sign removal leaves relation accuracy at 40/100 for every fitted seed.
The scorer explicitly engineers query words, entity positions, field overlap and
exact decimal comparison sign; it learns option ranking and sign-to-relation
weights. This demonstrates successful supervised fitting and same-domain
development usefulness on that feature representation. It does not show a neural
parent learned raw language or arithmetic, parent-decision retention, calibration,
causal mediation, a strong unrestricted-comparator win, or an L0 promotion gate.

Raw models are committed alongside this receipt. The complete evaluation tasks,
slot logits/labels, initial/sign-ablated predictions, full rollout outcomes and
every raw planner request/response are losslessly compressed in the `.gz` files.
Decompressed bytes match `evaluation_tasks_sha256` and each run's
`raw_predictions_sha256` in the receipt. Uncompressed originals also remain at
`C:/Users/apumu/research/CHLabs/ssdm-data/l0-transparent-cpu-20261006/`.
The original cohort remains at the `--cohort` path in the command; its independent
regeneration proof was committed on `codex/l0-training-cohort` at `07f5c5a`.

Independent re-derivation is required before a measured-results PR is merged.

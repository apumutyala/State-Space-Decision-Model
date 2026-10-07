# L1 world-level fit/calibration preparation

2026-10-07. Branch `codex/l1-world-level-calibration-split`, based exactly on
PR #133 head `e94c79793082fc8643bae34ac52813760f1950e8`. Local Windows CPU,
cost $0. No model/weights, optimizer, GPU, paid service, Index or heldout
access. This prepares data only: no training, calibration parameter fitting
or calibrated capability is claimed. The source fixtures and prior receipts
are unchanged.

Track D's generic trainer Item conversion drops exported world metadata.
Splitting randomly by task could put the same four-record catalog in both
fit and calibration partitions. The new helper/CLI splits the exact pinned
L1 training export before Item conversion. Consumers must honor the two
files as separate partitions; recombining and randomly resplitting them
would undo the separation. The generic trainer is unchanged in this branch.

The preregistration is `plans/codex/L1_WORLD_CALIBRATION_SPLIT.md`: fixed seed
1708, SHA-256 rank of compact ASCII JSON
`["ssdm.l1_world_split.v1",1708,world_id]`, ascending hex digest then world ID;
first eight worlds are calibration, other 24 are fit. Assignment uses only
seed and IDs, never labels, evidence, distributions or results. No seed
search or CLI seed/size override. Calibration world indices are **1, 3, 10,
14, 20, 21, 24, 27**; full IDs and per-world catalog/context/source-row hashes
and task IDs are retained in `split-manifest.json` and `verification.json`.

| Partition | Worlds | Tasks | Slot rows | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| fit.jsonl | 24 | 960 | 3,072 | 16,036,232 | `34909156035594729d7891d64c840efb8ed853dfe4fcb10a56f77b0fa8e91be0` |
| calibration.jsonl | 8 | 320 | 1,024 | 5,345,728 | `f58d51a51cdb6621d9b7ae86d348d67406488e13bb0c6839c174aaa5c47dce28` |

The 196,071-byte split manifest has SHA-256
`3ad9f962cc6896c4540a13633ea371a00d8537c46d061813b66188e43920b0ec`.
The complete three-file bundle is preserved outside Git at
`C:/Users/apumu/research/CHLabs/ssdm-data/l1-world-calibration-split-20261007/`.
The manifest copy and exact verification receipt are committed here;
reproduction uses the checked-in pinned source cohort and the command below.

All 4,096 source rows appear exactly once across the files, retaining their
original row content, labels, teacher-forced prefixes, metadata and relative
source order. Fit and calibration world/task membership is disjoint. Every
world retains 40 tasks and 128 unique slot rows: 40 operation, 16 fact, and
24 each left/right/relation slots. Every world has eight reads, eight
explanations, 24 comparisons, 20 tasks per field and 20 tasks per choice
format. Per field comparisons remain two equal, five less-than, five
greater-than, so task-kind/field/relation distributions are identical per
world in both partitions without label-based assignment or balancing.

The input is only `l1_multiworld_development_train`. Its sample SHA-256 is
`175cf7cc2d7c39a8c931a3a452149be7c141db712bde81a0ce7f606b0a27e9f4`,
manifest SHA-256 is
`eaa38a852dab840a51175a12a1a58a74539338f3ee119088a3442f054d0586e7`,
and the canonical training export must match
`a7806d12672be658f66f75d57a0d53a888cb00b93074b183a0a7c5bceb52d489`
before any assignment. Manifest provenance preserves the generator pins,
fixture ancestry and exact #133 base separately. Validation and the old
one-world smoke are rejected by split identity before file access, and
substituting either cohort under the train filename fails source hash pins.
The separate 32-world validation cohort remains evaluation-only. Its pins
are reported for provenance; its data is never read by the split helper.
Tests read validation source bytes only to prove substitution rejection.

Commands from this worktree (use a fresh output path to reproduce):

```text
python -X utf8 -m language_growth.multirecord_calibration_split --out C:/Users/apumu/research/CHLabs/ssdm-data/l1-world-calibration-split-20261007
python -X utf8 -m pytest language_growth/tests/test_multirecord_calibration_split.py language_growth/tests/test_multirecord_training.py language_growth/tests/test_multirecord_distribution.py language_growth/tests/test_training_records.py -q
python -X utf8 -m pytest -q tests/test_jobs_lint.py
git diff --check
```

Focused tests: **28 passed, 23 subtests passed**. Repository lint: **67
passed**. Tests prove deterministic rule/input-order independence and full
export determinism, exact row union/metadata preservation, complete
task/slot coverage, per-world distribution parity, label independence of
assignment, changed-export rejection, validation/smoke rejection and atomic
publication cleanup/preservation. Publication stages all three files beside
the destination and uses one directory rename; an injected manifest-write
failure leaves no final or staging output, and an existing bundle is refused.

Only **eight synthetic calibration worlds** are available, regardless of
their 1,024 correlated slot rows. This split does not estimate risk or prove
calibrated performance. The six-name, fixed-template synthetic development
distribution and authored prefixes remain narrow. Calibration data is not
final evaluation. Learned L0, comparator, decision-retention, adversarial and
causal-intervention gates remain required. No L1 promotion is supported.

Self-adversarial inspection considered Item metadata loss, task/slot leakage,
row mutation or duplication, hash/seed stability, label-driven assignment,
validation relabelling, source ancestry attribution and partial writes.
Exact pins fail closed, the ID-only ranking is separate from label-carrying
rows, and original row order/content survives. A fresh independent exact-head
review at `4de2a6dfa0071af06ab3f02e0e6397228fb9bb4a` found no blockers. This
receipt was authored before that review and before a PR existed; Track D review
will be requested on the stacked PR, after the final receipt update.

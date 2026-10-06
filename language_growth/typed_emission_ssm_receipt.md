# Typed L0 emission: random-initialized CPU diagnostic

One frozen condition completed in **8.55 seconds**, with 28,263 parameters,
two CPU threads and one interop thread, three fixed seeds, and $0 spend.
This is a random-initialized diagnostic. It provides **no parent-retention,
production-readiness, or held-out generalization claim**.

Pooled complete-target exactness and whole-answer correctness were 960/1080
(88.89%; task-cluster bootstrap 95% CI 84.54–92.59%). Kernel acceptance was
1056/1080 (97.78%; CI 95.65–99.54%). Comparison target selection is the
remaining weakness: correct comparison answers were 246/360 (68.33%).

## Model and inference boundary

The width-32 byte encoder implements the diagonal recurrence
`h[t] = a*h[t-1] + (1-a)*tanh(W*embedding(byte[t]))` and concatenates its
terminal and mean states. It feeds an operation classifier and a factorized
structured-target scorer using field, left entity, right entity, scope, and
query/component products. A separate small learned MLP selects one of
`greater_than`, `less_than`, or `equal` from the selected operands' visible
numeric values: left/scale, right/scale, their arithmetic difference, and
squared difference. The scales are frozen at 200 for price and 36 for warranty.
No comparator result or equality flag is supplied as a feature.

Candidate construction accepts only the first request's audience-filtered
state. It enumerates four read targets, four explain targets, and four
comparison structures: every visible record/field and both operand orders for
each field. Every comparison has all three relation choices. Candidates are
never checked or filtered for numerical truth before selection. Inference
receives no gold program, target, evaluator binding, option ID, request ID,
context hash feature, or generator gold-prefix request. It first predicts
operation, then structure within that operation, then relation from its own
predicted operands. The SSM is a hybrid with explicit numeric preprocessing;
this does not show that a byte-only SSM learned arithmetic.

The selected complete semantic target compiles directly to
`ssdm.semantic_program.v0.1`, using only ReadFact/Compare/Assert/ExplainDerivation.
The program must pass `check_program` before `render_checked` is called.
Rejected programs produce no rendered answer. No learned text generation or
repair path exists. Gold programs are read only in supervision, dataset
authorship, and post-prediction scoring. The relation loss uses gold operands
during training; inference uses predicted operands. PR #78 was a read-only
architecture reference at local commit
`59e6622ba121676ea75cdbc8e580d0623657de01`; no unmerged module was imported or stacked.

## Data and frozen condition

The existing synthetic generator produced 600 training tasks using seed 9101
and 180 development task clusters using seed 9102, all from
`development_train`. Each evaluation cluster has two members: an original
and a selected-field value-swap world for comparisons, or the other existing
equivalent wording for read/explain. Comparison partners have exactly the same
question with changed visible evidence; equality worlds legitimately remain
equal after swapping. There are 60 clusters per operation, 360 rows per model
seed, and 1080 pooled seed/row observations. Counts pooled across three seeds
are not 1080 independent tasks.

These are newly generated rows and newly authored pair records for this run,
using reused development names, numeric ranges, and wording families. They are
not the old PR #78 validation rows. No `development_validation`, Index, audit,
held-out data, teacher API, network, GPU, or paid resource was used.

Config and source/data hashes were frozen before training. All three seeds
(20261006, 20261016, 20261026) trained for exactly 300 steps with batch size 32,
Adam learning rate .003, gradient norm cap 1, and equal-weight operation,
conditional structure, and comparison relation cross-entropies. Final
checkpoints only were evaluated; there was no validation tuning, condition
selection, early stopping, or second variant.

## Results

Target exactness includes operation, field, record identity, ordered operands,
and comparison relation. Whole-answer correctness requires that complete
semantic target to equal gold **and** the typed program to be accepted and
render to the checked reference answer. The evaluator compares checked
renderings and asserts its equivalence to complete target equality on every
observation. Kernel acceptance alone checks evidence-relative validity; it
does not establish question relevance.

| Seed | Read exact/whole | Explain exact/whole | Compare exact/whole | All exact/whole | Kernel acceptance |
|---|---:|---:|---:|---:|---:|
| 20261006 | 116/120 | 120/120 | 79/120 | 315/360 | 349/360 |
| 20261016 | 118/120 | 120/120 | 86/120 | 324/360 | 354/360 |
| 20261026 | 120/120 | 120/120 | 81/120 | 321/360 | 353/360 |
| Pooled | 354/360 | 360/360 | 246/360 | 960/1080 | 1056/1080 |

Read and explain kernel acceptance was 360/360 each; comparison acceptance
was 336/360. Operation selection was correct for 1076/1080 observations.
Comparison structure was exact on 270/360; relation was correct on 246/270
correct structures. Relation-label accuracy over all comparisons, including
wrong structures, was 272/360 and is not whole-answer accuracy.

Paired success requires both members of a cluster to succeed within each seed.
Across seeds, 475/540 seed/task pairs had both answers correct (87.96%), and
526/540 had both programs accepted (97.41%).

| Seed | Both answers correct | Both comparison answers correct | Both programs accepted |
|---|---:|---:|---:|
| 20261006 | 154/180 | 38/60 | 173/180 |
| 20261016 | 161/180 | 43/60 | 177/180 |
| 20261026 | 160/180 | 40/60 | 176/180 |

The 2000-replicate percentile bootstrap uses analysis seed 3197 and resamples
unique task clusters, keeping all three model seeds and both members together.
Intervals are conditional on the fixed seeds and this reused development
family. Degenerate 100% intervals reflect this observed sample, not certainty
about a broader population.

| Metric | All, estimate [95% CI] | Read | Explain | Compare |
|---|---|---|---|---|
| Target exact / whole answer | 88.89% [84.54, 92.59] | 98.33% [96.39, 99.72] | 100% [100, 100] | 68.33% [58.88, 77.78] |
| Kernel acceptance | 97.78% [95.65, 99.54] | 100% [100, 100] | 100% [100, 100] | 93.33% [87.22, 98.06] |
| Both members exact / whole | 87.96% [83.52, 91.85] | 96.67% [92.78, 99.44] | 100% [100, 100] | 67.22% [57.76, 76.67] |
| Both members accepted | 97.41% [95.19, 99.26] | 100% [100, 100] | 100% [100, 100] | 92.22% [85.56, 97.22] |

The paired partner-minus-original whole-answer difference was +0.37 percentage
points [−0.93, +1.85] overall and +1.11 [−1.11, +3.89] for comparisons.
There is no claim of improvement over a previous model or architecture.

## Failure and adversarial evidence

All 24 rejected predictions were comparison programs with
`unsupported_assertion`: the learned relation was false for the chosen values.
All 90 comparison structure errors were nevertheless kernel-accepted; four of
these selected the wrong operation. Six read answers selected the wrong
record/field and were also accepted. Thus 96 accepted programs answered the
wrong requested semantic target: 90 compare, 6 read, 0 explain.

A deliberately wrong relation in an otherwise correct comparison was rejected
with `unsupported_assertion`. A cross-field operand with incompatible units
was rejected with `type_unit_mismatch`; existing shipped fixture 09 independently
passed the same rejection check. A deliberate wrong-record read stayed
kernel-accepted but failed whole-answer scoring. This confirms the important
limit that evidence truth alone does not check requested question relevance.

The new tests check the vectorized recurrence against the sequential
recurrence, visible-state-only inference and record-order invariance,
candidate preservation under swapped values, exact syntax compilation, and
adversarial boundaries. The new suite plus existing kernel/adversarial,
renderer, and development-generator suites passed **43 tests**.

## Preserved artifacts and replay

Source base: `24f164e9cfff1abe8f7eecef68a1771d357ca465`.
The owned module is `language_growth/typed_emission_ssm.py`; post-run statistics
are in `language_growth/typed_emission_ssm_analysis.py`. Machine receipts,
registration, and paired analysis are tracked under
`language_growth/receipts/typed_emission_ssm_20261006/`.

All 1080 raw predictions/programs/kernel results/renderings are also committed
as [predictions.json.gz](receipts/typed_emission_ssm_20261006/predictions.json.gz).
Decompressing this file reproduces the original prediction bytes and the raw
prediction SHA-256 below. The archive checksum and byte counts are recorded in
`raw_artifact_manifest.json` beside it; the directory's README gives a
standard-library verification command. This archive was added for independent
Git review without rerunning training or changing any metrics.

Full local artifacts are preserved under
`C:/Users/apumu/research/CHLabs/review-artifacts/l0-typed-emission-ssm-20261006/`:
frozen rows (including reference programs and permitted states), all 1080 raw
predictions/programs/kernel results/renderings, three restored CPU checkpoints,
registration, receipt, paired statistics, and exact source snapshots.

| Artifact | SHA-256 |
|---|---|
| Frozen data | `45260083441225ecb9a2abb3fd7790ba45ef806d18d3141e6574c456166787e2` |
| Raw predictions | `69392556a2f2f786e5126f5a8d0e02488882815b56966f5f3f191d52ed183e30` |
| Frozen model source | `93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6` |
| Seed 20261006 checkpoint | `28601d550a51b2800a396186e66a10a8de4ed1fbf195618ea9366790aafde611` |
| Seed 20261016 checkpoint | `0e2df6464e8e8d76cf26870f4de4fe4fbe85ff1b7362da9cdc9aae9725c676ce` |

The third checkpoint hash and all dependency source hashes are in the machine
receipt. To reproduce, set `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`; run the module's `prepare` followed by `run` against a new
output directory. Use the preserved directory for analysis replay only, since
training replay would overwrite checkpoint artifacts. The diagnostic preceded
PR #80; its subsequent raw-artifact addition preserves the original run.

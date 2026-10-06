# Fixed-model transfer to disjoint synthetic development

The unchanged random-initialized model transferred poorly: **whole-answer
correctness was 519/1080 (48.06%; task-cluster 95% CI 41.94–53.89%)**, while
kernel acceptance was **1036/1080 (95.93%; CI 93.43–98.15%)**. Comparison
correctness was 29/360 (8.06%). This is iterative synthetic development,
**not an official holdout, parent-retention result, or learned L0 qualification**.

One frozen condition used the exact previous architecture, source, numeric
scales, training examples/order, three model seeds, optimizer, and 300 steps.
The diagnostic completed in **11.84 seconds**, with two CPU threads, one
interop thread, 28,263 parameters, and $0 spend. Every tensor in all three final
checkpoints is bitwise identical to the corresponding original PR #80
checkpoint; their complete checkpoint file hashes also match. The performance
change therefore comes from the evaluation cohort, not altered training.

## Frozen condition and evidence boundary

Fresh worktree base: `0f4b8ad2ea717c79858ddfbc4deb55c9b71940af`, the current local
`origin/agent-main` when this condition began. The model implementation was
explicitly copied from committed PR #80 source at
`c0db91df4b28fbb3cbd9fe05b210b85ff2a396ee`, without stacking its branch.
The copied `typed_emission_ssm.py` is byte-identical, SHA-256
`93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6`.

The width-32 diagonal byte SSM feeds operation and factorized target heads.
The separate learned numeric relation MLP receives only visible selected
operand values as left/scale, right/scale, difference, and squared difference;
no correct-relation or equality flag is computed. Scales remain price=200 and
warranty=36 despite the new numeric ranges. This is a hybrid with explicit
numeric preprocessing, not a demonstration of byte-only learned arithmetic.

Inference receives only the initial permitted state, with no gold prefixes or
evaluation metadata. It enumerates all 12 structural targets and all three
comparison relations, predicts operation then structure then relation using
its own selected operands, and compiles each selection to a typed program.
Only ReadFact/Compare/Assert/ExplainDerivation are emitted. Every proposal
passes through `check_program`; only accepted results reach `render_checked`.
There is no truth-filtered candidate set, unchecked text generation, or repair.
Training relation supervision uses gold operands, as in the original model;
inference uses predicted operands.

Training uses the same 600 `development_train` tasks at generator seed 9101.
Evaluation uses 180 `development_validation` clusters at seed 9102, with 60
per operation and two members each. The validation names Grove/Hearth/Indigo/
Juniper/Kestrel/Lumen are absent from the training name pool. Warranty values
37–60 are disjoint from training's 1–36; prices $200.01–300 are disjoint from
training's $50–200. Generator validation templates are also disjoint.

Each comparison partner keeps the question unchanged and swaps only the
requested field's visible values. Read and explain partners use preauthored
alternate validation templates, also absent from training: "For October 2026,
state the {field} of {entity}." and "For {entity}, provide the checked derivation
of the {field}." Since the generator has only one validation template per kind,
these alternate templates were declared and hashed before training. The
pooled headline includes both original and alternate wording members.

Model seeds are 20261006, 20261016, 20261026. Each uses Adam learning rate .003,
batch size 32, gradient norm cap 1, 300 steps, and equal-weight operation,
conditional-target, and comparison-relation cross-entropies. Registration,
exact source hashes and data hash were frozen first. There was no tuning,
checkpoint selection, additional variant, network, GPU, paid resource, Index,
teacher API, or official held-out data. The names/ranges/templates all change
together, so this condition cannot isolate their individual causal effects.

## Results and paired uncertainty

Complete-target exactness includes operation, field, record identity, ordered
operands, and relation. Whole-answer correctness requires exact target
equality plus accepted matching checked rendering; the evaluator asserts the
equivalence of complete target equality and accepted reference rendering for
every observation. Kernel acceptance establishes evidence-relative validity,
not requested-question relevance.

| Seed | Read exact/whole | Explain exact/whole | Compare exact/whole | All exact/whole | Kernel accepted |
|---|---:|---:|---:|---:|---:|
| 20261006 | 85/120 | 92/120 | 0/120 | 177/360 | 357/360 |
| 20261016 | 75/120 | 76/120 | 17/120 | 168/360 | 339/360 |
| 20261026 | 73/120 | 89/120 | 12/120 | 174/360 | 340/360 |
| Pooled | 233/360 | 257/360 | 29/360 | 519/1080 | 1036/1080 |

The fixed-seed, 2000-replicate percentile bootstrap uses analysis seed 3197.
It resamples unique task clusters while keeping all three model seeds and both
members together. There are 180 unique tasks, not 1080 independent tasks;
per-kind intervals use 60 clusters. Intervals are conditional on the fixed
seeds and these synthetic development families.

| Metric | All estimate [95% CI] | Read | Explain | Compare |
|---|---|---|---|---|
| Exact / whole answer | 48.06% [41.94, 53.89] | 64.72% [56.94, 72.50] | 71.39% [62.78, 79.72] | 8.06% [4.17, 12.51] |
| Kernel acceptance | 95.93% [93.43, 98.15] | 100% [100, 100] | 100% [100, 100] | 87.78% [81.11, 93.61] |
| Both members correct | 40.93% [34.63, 47.22] | 47.22% [37.22, 57.78] | 68.33% [59.44, 77.22] | 7.22% [3.33, 11.67] |
| Both members accepted | 95.37% [92.59, 97.78] | 100% [100, 100] | 100% [100, 100] | 86.11% [78.33, 92.22] |

Both-answer successes were 73/180, 72/180, and 76/180 by seed; corresponding
comparison both-answer successes were 0/60, 8/60, and 5/60. Overall there were
221/540 successful seed/task pairs.

Original generator wording alone achieved 235/540 whole answers (43.52%);
partners achieved 284/540 (52.59%). By kind, original/partner counts were read
91/180 versus 142/180, explain 128/180 versus 129/180, compare 16/180 versus
13/180. The paired whole-answer difference was +9.07 percentage points
[5.56, 12.78] overall, read +28.33 [20.56, 35.56], explain +0.56 [−2.78, 3.89],
and compare −1.67 [−4.44, 0.00]. These describe wording/world sensitivity, not
an improvement over a matched baseline model.

## Failures and value-swap response

All **44 rejected programs** were false comparisons rejected with
`unsupported_assertion`. There were **517 accepted but wrong-question
programs**: 127 read, 103 explain, and 287 compare. Operation selection was
correct on all read/explain tasks but only 112/360 comparisons; 248 comparison
questions instead produced a read or explanation. Correct comparison
structure occurred on 56/360 observations, of which 29 had the correct
relation. Thus question interpretation and operand selection dominate this
condition's failure; high kernel acceptance masks low usefulness.

For the 180 comparison seed/task pairs, the required relation changed on 159
unequal pairs; 21 equality pairs remained equal. The model predicted Compare
on both members of only 56 pairs. It changed its predicted relation on 24
pairs. Both full answers were correct on 13/180 pairs: 7 inequality pairs and
6 equality pairs. Both relation labels were correct on 20/180, but only
28/180 pairs had correct structures. Operation and structure predictions were
unchanged on all 180 value swaps, as expected because those heads encode the
unchanged question and structural components; only the numeric relation head
sees changed values. Relation changes alone therefore do not establish that
the requested operands or operation were selected correctly.

The existing adversarial checks still reject a deliberately false relation
and incompatible units, including shipped fixture 09. A deliberate wrong
record remains evidence-valid and fails whole-answer scoring. The copied
recurrence/boundary tests plus new split/core/pair tests and existing kernel,
adversarial, renderer, and generator tests passed **45 tests**. Independent
raw replay verified all 1080 kernel outcomes/renderings and per-kind metrics.

## Artifacts and replay

Committed evidence is under
`language_growth/receipts/typed_emission_transfer_20261006/`: registration,
machine receipt, paired statistics and failure decomposition,
`predictions.json.gz` (57,380 bytes), `frozen_rows.json.gz` (168,975 bytes),
checksums, README, and `verify_raw.py`. Both archives preserve exact original
bytes. Run the verifier from the repository root to inspect results without
training or model inference.

| Raw artifact | Decompressed SHA-256 |
|---|---|
| Predictions | `8e164ee8bc163c8eee346cd394c84922d4825ec097c99d817b485dffd0a36678` |
| Frozen inputs | `4c63f494a42ec66da89236225d3d9aa6a6c22e4b9844df96c4a8188ee00b0faa` |

Local uncompressed evidence, checkpoints, and exact source snapshots remain
under `C:/Users/apumu/research/CHLabs/review-artifacts/l0-typed-emission-transfer-20261006/`.
Checkpoint/source/dependency hashes are in the machine receipts. To reproduce
the one condition in a new directory, set `CUDA_VISIBLE_DEVICES=-1`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, then invoke
`python -m language_growth.typed_emission_ssm_transfer prepare --out NEW_DIR`
and `python -m language_growth.typed_emission_ssm_transfer run --out NEW_DIR`.
Do not rerun into the preserved directory. No PR was opened for this condition.

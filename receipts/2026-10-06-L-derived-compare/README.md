# DerivedCompare module and fixed-output ablation

Local review proposal, 6 October 2026. Cost $0; no model calls, training, tuning,
GPU, network or teacher API. This is a deterministic module ablation of PR #82's
committed synthetic development predictions, not a new learned result.

The draft module requires `ssdm.semantic_program.v0.2`, explicit host operation
opt-in, and authorization for all three relations. Two prior permitted numeric
Facts become the existing checked Comparison. Both bound records, values,
sources, versions, units, quantity kinds, scopes and dependencies survive Assert,
ExplainDerivation and release rendering. v0.1 Compare remains unchanged.
ExactSum is rejected here. Joint review/reconciliation with PR #38's shared v0.2
draft is required before interface integration; this receipt is no approval.

PR #82 reference commit: `1a09d189635711bd79319ef8eb334ed587e34c2e`.
Both compressed and decompressed artifact hashes are verified against its
committed manifest. All 1,080 explicit-relation checker/renderer outputs are
reproduced before scoring the ablation. Every model-selected operation family,
field and ordered record selection stays fixed. All 369 selected Compare nodes
are replaced, including nine comparisons incorrectly selected on read tasks;
the transform never uses task/gold to decide where to apply the module. Registry
and schema changes are explicit host enablement. Gold/reference programs are
used only to score the already checked/rendered results.

| All 1,080 observations | Explicit relation | DerivedCompare | Paired delta, percentage points (95% CI) |
| --- | ---: | ---: | --- |
| Kernel accepted | 946 | 1,080 | +12.41 [8.24, 16.94] |
| Whole answer correct | 563 | 619 | +5.19 [2.69, 8.06] |
| False relation rejects | 134 | 0 | -12.41 [-16.94, -8.24] |
| Accepted wrong target | 383 | 461 | +7.22 [4.44, 10.37] |

Comparison whole answers improve 86→142/360 (+15.56 pp, CI [8.61, 23.89]),
recovering exactly the 56 relation errors in the scoped 142-structure/86-answer
upper bound; there are no regressions. Comparison kernel acceptance rises
227→360 and accepted-wrong targets rise 141→218. Read and explanation
correctness remain 233/360 and 244/360. Increased coverage exposes pre-existing
wrong selections; it does not certify relevance. The full report includes CIs
for each metric and task kind.

On 180 comparison seed/task value-swap pairs (60 unique task pairs), the 159
required relation flips met increase 111→159. Both answers correct increase
38→71 (+18.33 pp, CI [10.56, 27.78]); both structures exact remain 71. Both
inequality answers correct increase 29→62; both equality answers correct remain
9. Even perfect required flipping does not establish correct operand selection.

CIs use 2,000 paired percentile bootstrap replicates, seed 3197, resampling
unique task clusters while preserving both members, both arms and all three
fixed model seeds. They are conditional on these saved seeds and synthetic
development families; no official held-out or parent-retention claim is made.

`ablation_observations.json.gz` preserves both full arms per observation.
`ablation.json` contains metrics, input/output/source hashes and bootstrap details.
`legacy_compatibility.json` records identical checked dictionaries and released
UTF-8 text for all 25 original v0.1 attempts across 24 fixtures against base
`0f4b8ad2ea717c79858ddfbc4deb55c9b71940af`. Types and realizer sources are unchanged.
The focused suites pass 46 module/kernel/renderer tests and 8 fixture tests.

From the repository root, reproduce without inference or network:

```powershell
python -m language_growth.derived_compare_ablation
python receipts/2026-10-06-L-derived-compare/verify_legacy.py
python -m unittest language_growth.tests.test_derived_compare language_growth.tests.test_derived_compare_ablation language_growth.tests.test_kernel language_growth.tests.test_kernel_adversarial language_growth.tests.test_realize -v
python -m unittest language_growth.tests.test_fixtures fixtures.language_growth.test_fixture_pack -v
```

The ablation requires NumPy and the pinned PR #82 local Git objects; it never
fetches missing objects. Runtime and code-file hashes are in `receipt.json`.

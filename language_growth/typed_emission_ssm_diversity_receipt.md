# Six-template training wording on the fixed typed-emission SSM

One train-only wording-diversity condition improved **comparison whole-answer
correctness from 29/360 (8.06%) to 86/360 (23.89%)**, paired change
**+15.83 percentage points [95% CI 8.06, 24.44]**. All 360 comparison questions
now selected Compare, versus 112/360 in the saved baseline. Value-swap relation
flips improved from 24/159 required flips to 111/159, and both-answer successes
from 13/180 to 38/180 comparison pairs.

Overall whole-answer correctness rose from 519/1080 (48.06%) to 563/1080
(52.13%), but its paired change **+4.07 points [−0.37, 8.89] is inconclusive**.
Read accuracy was unchanged in aggregate; explanation accuracy declined
13/360 observations. This supports a specific comparison-scoring gain on
these reused iterative-development questions, not a wording-transfer or
overall qualification claim. The absolute comparison accuracy remains low.
Scope is **iterative synthetic development only**, not official holdout,
parent retention, or evidence about actual Track D parents.

## Exact intervention and frozen model

Fresh worktree base: `0f4b8ad2ea717c79858ddfbc4deb55c9b71940af`, current local
`origin/agent-main` when the condition began. The committed PR #80/#81 model
was copied, with no unmerged imports or stacked branch dependency. Its source
remains byte-identical, SHA-256
`93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6`.

The model is the same random-initialized width-32 diagonal byte SSM, factorized
operation/structure heads, and learned numeric relation MLP. The relation
features remain visible selected left/scale, right/scale, difference, and
squared difference; price scale=200 and warranty scale=36. No comparator/equality
label is supplied as a feature. Architecture and numeric preprocessing are
fixed; learned parameters are fitted anew under the changed wording. This
hybrid does not establish byte-only arithmetic learning.

Only `state.task_input` in each of the 600 training rows changed. Worlds,
labels, evidence, row order, and IDs were copied from the previous seed-9101
`development_train` cohort. The six templates per operation and assignment
`(row_index // 3) % 6` were copied exactly from the earlier PR #75 wording
recipe, source commit `64df83efeafa08f43bebf6e2e5e59a1ff5baf77e`, file
`language_growth/query_template_transfer_probe.py`, SHA-256
`ba521a134a0cef41ade8e8bf5ff249c9cd3dc7355057ae3502481fcdfa8f61c1`.
Each operation has template counts 34,34,33,33,33,33. Full template strings are
frozen in registration and the owned data-authoring module. No template search
or new wording variant was introduced.

The validation rows are **exactly unchanged** from PR #81: seed 9102,
`development_validation`, 180 worlds/clusters, 360 original/paired members.
Names and numeric ranges are disjoint from training, but the template family
is not: read/explain alternate wording reuses training strings, and comparison
forms are near-paraphrases of templates written while validation forms were
visible. Entity substitution makes exact question strings different; it does
not establish wording independence. Comparisons retain original/value-swap
members; read/explain retain the previous original/alternate-wording members.
The complete validation-row digest is
`5e22f0e800ea48eb4ed0b1042a72e6ee713210fdaac85b4f64ea963d2ea4957c`.
The saved baseline predictions are reused byte-for-byte, not refitted. Data
authoring may read training targets to format training questions; inference
accepts only visible states and never sees gold programs or labels.

Three fixed model seeds (20261006,20261016,20261026) each trained for exactly
300 steps with Adam .003, batch 32, gradient norm cap 1, and the unchanged
equal-weight operation/conditional-target/comparison-relation losses.
Registration and source/data hashes were frozen before fitting. Only the
final checkpoints were scored. There was no tuning, early stopping, extra
condition, official held-out data, GPU, network, Index, teacher API, or spend.
The run took **9.42 seconds**, two CPU threads and one interop thread, with
28,263 parameters and $0 cost.

Every model selection is compiled to `ssdm.semantic_program.v0.1` using only
ReadFact/Compare/Assert/ExplainDerivation, passed through `check_program`, and
rendered only through `render_checked` if accepted. All 12 structures and all
three comparison relations are offered without truth filtering. Whole-answer
correctness requires complete semantic target equality (including operand
order/relation), accepted program, and checked rendering matching reference.
No unchecked text generation or repair exists.

## Whole-answer results

Each per-kind denominator is 120 per model seed, 360 pooled. The all-kind
denominator is 360 per seed, 1080 pooled; these repeat 180 unique task clusters.

| Seed | Read baseline → new | Explain baseline → new | Compare baseline → new | All baseline → new | New kernel accepted |
|---|---:|---:|---:|---:|---:|
| 20261006 | 85 → 74 | 92 → 75 | 0 → 29 | 177 → 178 | 317/360 |
| 20261016 | 75 → 83 | 76 → 80 | 17 → 30 | 168 → 193 | 310/360 |
| 20261026 | 73 → 76 | 89 → 89 | 12 → 27 | 174 → 192 | 319/360 |
| Pooled | 233 → 233 | 257 → 244 | 29 → 86 | 519 → 563 | 946/1080 |

There were 155 improved and 111 regressed seed/row answers overall. Comparison
had 63 improvements and 6 regressions. Read had 47 improvements/47 regressions;
explain had 45 improvements/58 regressions. The apparent unchanged read rate
therefore does not mean identical predictions.

The paired 2000-replicate percentile bootstrap uses analysis seed 3197 and
resamples unique task clusters while preserving both conditions, all three
fixed seeds, and both members. Per-kind intervals use 60 clusters. Intervals
are conditional on these seeds and development families, not a seed-population
or held-out guarantee. Complete-target exactness equals whole-answer
correctness for every observation, verified independently by raw replay.

| Whole-answer metric | New estimate [95% CI] | Baseline | Paired change, pp [95% CI] |
|---|---|---|---|
| All | 52.13% [46.11, 57.97] | 48.06% | +4.07 [−0.37, 8.89] |
| Read | 64.72% [56.39, 72.50] | 64.72% | 0.00 [−5.00, 4.72] |
| Explain | 67.78% [59.17, 75.56] | 71.39% | −3.61 [−12.22, 4.44] |
| Compare | 23.89% [15.28, 32.78] | 8.06% | +15.83 [8.06, 24.44] |
| Both paired members, all | 43.52% [37.04, 49.81] | 40.93% | +2.59 [−2.41, 7.41] |
| Both paired comparison members | 21.11% [12.78, 30.00] | 7.22% | +13.89 [6.11, 22.22] |

Both-member successes were 78/180,81/180,76/180 by seed (235/540 pooled).
Comparison both-member successes were 12/60,14/60,12/60.

## Kernel acceptance and residual errors

Kernel acceptance fell from 1036/1080 (95.93%) to 946/1080 (87.59%), paired
change −8.33 points [−11.76,−5.28]. High acceptance in the old condition often
represented a true answer to the wrong question; predicting Compare exposes
more false learned relations to the checker.

| Kernel acceptance | New count/rate [95% CI] | Baseline count |
|---|---|---|
| Read | 359/360, 99.72% [99.17,100] | 360/360 |
| Explain | 360/360, 100% [100,100] | 360/360 |
| Compare | 227/360, 63.06% [52.50,72.78] | 316/360 |
| All | 946/1080, 87.59% [83.06,91.76] | 1036/1080 |

Wrong complete targets decreased 561 → 517. Accepted wrong-target programs
decreased 517 → 383: read 127 → 126, explain 103 → 116, compare 287 → 141.
False-relation proposals rejected by the kernel increased **44 → 134**, all
`unsupported_assertion`; the new condition includes 133 comparison questions
and one read question that proposed a false comparison. None of those rejected
proposals was rendered. Accepted wrong-target outputs are reported separately.

Comparison operation selection improved 112/360 → 360/360; structure exactness
improved 56/360 → 142/360. Of 142 correct structures, only 86 selected a valid
relation. Thus operand selection and numeric relation errors remain after the
operation-transfer gain. Read operation accuracy fell from 360/360 to 351/360;
explain stayed 360/360. Overall operation accuracy was 1071/1080.

## Paired value-swap response

The same 180 comparison seed/task pairs require relation flips in 159 unequal
pairs; 21 equality pairs remain equal. Both conditions preserve their own
operation/structure prediction on every value swap, because those heads see
unchanged query/structural components; only the relation head sees values.

| Pair-level measure | Saved baseline | Six-template condition |
|---|---:|---:|
| Both predict Compare | 56/180 | 180/180 |
| Predicted relation changes on required flips | 24/159 | 111/159 |
| Both structures correct | 28/180 | 71/180 |
| Both relation labels correct | 20/180 | 50/180 |
| Both complete answers correct | 13/180 | 38/180 |
| Both inequality answers correct | 7/159 | 29/159 |
| Both equality answers correct | 6/21 | 9/21 |

A relation flip alone does not prove selection of the requested field or
ordered operands. Despite stronger intervention sensitivity, only 38/180 pairs
answer both worlds correctly. These limits keep this result below L0
qualification and do not overturn the prior fold of the original two-template
recipe as a candidate.

## Preserved evidence and checks

All raw results, fixed inputs, and the saved baseline are in Git under
`language_growth/receipts/typed_emission_diversity_20261006/`. Three gzip
archives total **294,135 bytes** and preserve exact original JSON bytes.
Registration, both machine receipts, paired statistics, checksums, README,
and an independent standard-library kernel/render replay are also committed.
The verifier checks both conditions' 2160 checked emissions and every per-kind
result from the unchanged validation evidence. The copied boundary/recurrence,
new data-authoring, and existing kernel/adversarial/renderer/generator suites
passed **45 tests**. Exact template equality with the earlier source was
checked without importing its unmerged implementation.

| Artifact | Decompressed SHA-256 |
|---|---|
| New predictions | `edbcb8393949cbefbf0d25e49cb1b11f2de2b2d40917ba39df7598decb1489e4` |
| Frozen input rows | `e3cec785d4abcbf796196a1dfc36fc3be2b99ad89b732a56724cc8b495e94505` |
| Baseline predictions | `8e164ee8bc163c8eee346cd394c84922d4825ec097c99d817b485dffd0a36678` |

Local checkpoints, raw files, and source snapshots remain at
`C:/Users/apumu/research/CHLabs/review-artifacts/l0-typed-emission-diversity-20261006/`.
All checkpoint/dependency source hashes are in the machine receipt. Replay raw
evidence with `python language_growth/receipts/typed_emission_diversity_20261006/verify_raw.py`.
Do not overwrite the preserved run. This receipt is preserved in PR #82; no
second condition was fitted.

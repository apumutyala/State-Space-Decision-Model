# Paired-value supervision on the six-template typed-emission SSM

One frozen paired-value training condition produced **no convincing gain over
the six-template control**. Whole-answer correctness changed **563 → 561/1080
(52.13% → 51.94%)**, paired change −0.19 percentage points [95% CI −0.74,0.37].
Comparison answers changed 86 → 89/360 (23.89% → 24.72%), while required
relation changes fell **111 → 105/159**. Both comparison worlds were answered
correctly on 38 → 41/180 pairs; that small gain's interval includes zero.

The random-relation control changed relations on 113/159 required-flip pairs,
yet answered both worlds correctly on only 9/180. Thus a high any-flip rate
does not establish semantic intervention fidelity. This result does not support
advancing this exact augmentation recipe as an L0 improvement. Scope remains
iterative synthetic development, not official holdout, parent retention,
production qualification, or a conclusion about other heads/Track D parents.

No meaningful-gain or equivalence margin was preregistered. The incremental
both-answer interval permits gains up to 3.89 points, so this is **not a formal
equivalence fold**: retain the exact paired-training condition as inconclusive
for a meaningful incremental benefit, without advancing it from this run.
The broader intervention hypothesis is not falsified. The model's established
advantage over static/random complete-answer controls remains part of the
evidence, separately from the weak training-intervention effect.

## Frozen single factor and balance

Fresh source base: `0f4b8ad2ea717c79858ddfbc4deb55c9b71940af`, current local
`origin/agent-main` at setup. The PR #82 six-template condition is the control:
52.13% overall, 23.89% compare, 111/159 flips. This clarification was frozen
before fit. The folded two-template recipe with 24/159 flips is **not** the
control. Baseline source/receipts were copied as read-only reference; no
unmerged module import or production change was made.

The copied width-32 diagonal byte SSM and numeric MLP are byte-identical to the
previous core, SHA-256
`93ae22e267186f5453f520a2bf30747e4cd097662068c066b8535d956ee5f2d6`.
Numeric features remain left/scale, right/scale, difference, squared difference,
with price scale=200 and warranty scale=36. No correct relation/equality flag
is supplied at inference. All 12 structures and all three relation choices
remain unfiltered; inference sees visible state only.

All original 600 six-template training rows are unchanged: question wording,
world, evidence, target, ID, and ordering. For each of the 200 comparison base
worlds, `comparison_swap_counterfactual` authors an additional same-question
member with the requested field's values swapped. The correct relation is
taken from that authored counterfactual program and checked against its own
evidence. Structural targets and other fields remain unchanged; equality
worlds correctly retain equality. There is no new wording or template search.

The sampler draws the **same base-world IDs in the same order as the control**.
For a comparison world, successive exposures alternate original then swapped
member. Each operation retains 200 uniformly sampled groups; a comparison
pair shares one group's sampling mass. No extra batch members, steps,
class/loss reweighting, or loss terms are added. Finite per-kind draws exactly
match the control rather than changing to a new stratified sampler. Pair
members differ by at most one exposure per world because original is first.

| Seed | Read / explain / compare draws | Original / swapped comparison draws |
|---|---|---|
| 20261006 | 3200 / 3190 / 3210 | 1656 / 1554 |
| 20261016 | 3294 / 3168 / 3138 | 1618 / 1520 |
| 20261026 | 3215 / 3223 / 3162 | 1632 / 1530 |

This is paired supervision across successive exposures, not an extra
within-batch consistency loss. That choice preserves the original optimization
budget and isolates comparison value exposure. The optimizer, three seeds,
300 steps, batch size 32, Adam .003, clip norm 1, and equal-weight existing
losses are unchanged. Learned parameters are fitted anew, including the same
numeric head; its architecture/features/scales are fixed.

Validation is the exact PR #81 disjoint seed-9102 cohort and paired members
used by the six-template control: 180 clusters, 360 rows, same names/ranges/
templates. Registration and source/data hashes were frozen before fitting,
and actual exposure counts were checked against the frozen schedule. Only
final checkpoints were scored. No tuning, extra variant, official held-out
data, GPU, network, Index, teacher API, or spend was used. Runtime was
**9.67 seconds**, two CPU threads and one interop thread, 28,263 parameters,
and $0 cost.

## Whole answers and kernel coverage

Every selection compiles to `ssdm.semantic_program.v0.1` using only
ReadFact/Compare/Assert/ExplainDerivation. Every program is checked before
`render_checked`; rejected proposals produce no text. Whole-answer correctness
requires complete target equality, accepted program, and checked rendering
matching the reference. Kernel acceptance alone does not establish question
relevance. The same path is used for static/random controls.

| Seed | Read control → new | Explain control → new | Compare control → new | All control → new | New kernel accepted |
|---|---:|---:|---:|---:|---:|
| 20261006 | 74 → 75 | 75 → 75 | 29 → 29 | 178 → 179 | 316/360 |
| 20261016 | 83 → 78 | 80 → 80 | 30 → 30 | 193 → 188 | 309/360 |
| 20261026 | 76 → 75 | 89 → 89 | 27 → 30 | 192 → 194 | 319/360 |
| Pooled | 233 → 228 | 244 → 244 | 86 → 89 | 563 → 561 | 944/1080 |

There were 6 improved and 8 regressed seed/row answers: read 1 improvement/6
regressions, explain unchanged, compare 5 improvements/2 regressions. Paired
bootstrap uses 2000 percentile replicates, seed 3197, resampling unique task
clusters while preserving both conditions, all three model seeds, and both
members. There are 180 unique clusters (60 per kind), not 1080 independent
tasks. Intervals are conditional on these fixed seeds/development families.

| Whole-answer result | New estimate [95% CI] | Paired change, pp [95% CI] |
|---|---|---|
| All | 51.94% [45.83,57.87] | −0.19 [−0.74,0.37] |
| Read | 63.33% [55.00,71.11] | −1.39 [−2.78,0.00] |
| Explain | 67.78% [59.17,75.56] | 0.00 [0.00,0.00] |
| Compare | 24.72% [16.11,33.90] | +0.83 [0.00,1.94] |
| Both comparison members | 22.78% [13.89,31.67] | +1.67 [0.00,3.89] |

Complete-target exactness equals whole-answer correctness on all observations,
verified by independent checker/render replay. Kernel acceptance changed
946 → 944/1080 (87.59% → 87.41%; new CI 82.69–91.67%), paired change −0.19
points [−0.83,0.46]. By kind, acceptance is read 358/360 (99.44%, CI
98.33–100), explain 360/360 (100%), compare 226/360 (62.78%, CI 51.67–73.33%).

Wrong targets increased 517 → 519. Accepted wrong-target programs remained
383: read 126 → 130, explain 116 → 116, compare 141 → 137. False-relation
rejects increased 134 → 136, all `unsupported_assertion`; the new condition
includes 134 comparison questions and 2 read questions whose wrong operation
emitted a false comparison. Comparison operation selection remains 360/360;
structure exactness remains 142/360. Read operation accuracy decreased
351 → 346/360; explain stayed 360/360. Of 142 correct comparison structures,
89 had the correct relation, leaving 53 false learned relations.

## Intervention fidelity and controls

Across the same 180 comparison seed/task pairs, 159 unequal pairs require an
oriented relation change; 21 equality pairs should remain equal. The static
control repeats the learned original complete target in both worlds, with
fresh compilation/check/render against each world. The random control keeps
learned operation/structure but independently chooses each member's relation
uniformly from all three, using frozen control seed 8421. Controls are not
additional learned variants.

| Pair measure | Six-template control | Paired-value training | Static target | Random relation |
|---|---:|---:|---:|---:|
| Any relation change on required flips | 111/159 | 105/159 | 0/159 | 113/159 |
| Both correct oriented gold relations, unequal | 29/159 | 32/159 | 0/159 | 24/159 |
| Both complete answers, unequal | 29/159 | 32/159 | 0/159 | 8/159 |
| Both complete answers, all pairs | 38/180 | 41/180 | 9/180 | 9/180 |
| Both complete equality answers | 9/21 | 9/21 | 9/21 | 1/21 |
| Incorrect relation change on equality | 0/21 | 0/21 | 0/21 | 16/21 |
| Both structures correct | 71/180 | 71/180 | 71/180 | 71/180 |
| Operation stable / structure stable | 180/180 | 180/180 | 180/180 | 180/180 |

The substantive control outcomes below separate correct **oriented** gold
relation changes, both **full semantic answers**, and erroneous equality
changes. Values are percentages [task-cluster 95% CI]. Required-flip metrics
use 53 unique unequal clusters (159 seed/pairs); both-answer metrics use all
60 comparison clusters (180 seed/pairs); equality metrics have only 7 unique
clusters (21 seed/pairs). Control intervals condition on frozen random seed
8421 as well as the three model seeds; they are not estimates over all possible
random-control draws.

| Outcome | Six-template control | Paired training | Static output | Random relation |
|---|---|---|---|---|
| Correct oriented relation pair, required flips | 18.24 [8.81,27.69] | 20.13 [11.32,30.19] | 0 [0,0] | 15.09 [10.06,21.38] |
| Both full answers correct, all pairs | 21.11 [12.78,30.00] | 22.78 [13.89,31.67] | 5.00 [1.11,10.00] | 5.00 [2.22,7.78] |
| Spurious flip on equality pairs | 0 [0,0] | 0 [0,0] | 0 [0,0] | 76.19 [57.14,90.48] |
| Any change on required-flip pairs | 69.81 [57.86,81.13] | 66.04 [54.09,77.36] | 0 [0,0] | 71.07 [63.52,78.62] |

Any required-flip responsiveness decreased from 69.81% to 66.04% (new CI
54.09–77.36%), paired change −3.77 points [−6.92,−1.26]. Correct oriented
relation pairs increased 18.24% → 20.13% (new CI 11.32–30.19%), paired change
+1.89 [0.00,4.40]. Both complete answers improved 21.11% → 22.78% (change
+1.67 [0.00,3.89]); these small fidelity gains include zero at the interval
boundary. Operation/structure stability is expected because those heads see
unchanged query/structural components; it is not evidence of value sensitivity.

An independent uniform relation changes with probability 2/3, but matches both
specified gold relations with probability 1/9. The realized random control's
113/159 required flips therefore demonstrates why raw responsiveness is weak.
Its 129 total relation changes include 16 erroneous changes on equality worlds;
129 is not the required-flip numerator. The learned new condition exceeds
random in both complete pair answers by 17.78 points [8.89,26.67], but the
change relative to the learned six-template control remains small/inconclusive.
The corresponding advantage over static output is +17.78 [10.00,26.67]. For
correct oriented gold relations alone, new minus random is +5.03 points
[−6.29,16.98], which remains inconclusive; this does not erase the more scoped
full-answer signal. No benefit is claimed from reranking or prefiltering
candidates, and no intervention-family falsification is claimed.

## Preserved evidence and verification

Committed evidence under
`language_growth/receipts/paired_value_training_20261006/` includes frozen
original/paired inputs, baseline/current predictions, raw static/random
controls, both receipts, registration, paired CIs, and checksum manifest.
Four gzip archives total **387,812 bytes** and preserve original JSON bytes.
The standard-library verifier replays all 2880 model/control checker/render
outcomes, checks all 200 authored training swaps, validates unchanged row
digests, and recounts per-kind model results. The copied recurrence/boundary,
new paired-sampler/counterfactual, and existing kernel/adversarial/renderer/
generator suites passed **45 tests**.

| Artifact | Decompressed SHA-256 |
|---|---|
| New predictions | `bfddd369428b036c6fdfd9fabda97f2c557a450ea56cfb3ebc6287a9d1fb0668` |
| Frozen original/paired inputs | `95f5367723ecdcd5dc1357f1fa0886b4883239e896d3672db4538820e85ad39c` |
| Six-template baseline predictions | `edbcb8393949cbefbf0d25e49cb1b11f2de2b2d40917ba39df7598decb1489e4` |
| Static/random controls | `26b80699f217a53b3b62b17cba16564de19d17e670ca32d21ffb8c28543ff2fd` |

Local checkpoints, raw files, and source snapshots are preserved at
`C:/Users/apumu/research/CHLabs/review-artifacts/l0-paired-value-training-20261006/`.
Source/checkpoint/dependency hashes are in machine receipts. Verify from Git:
`python language_growth/receipts/paired_value_training_20261006/verify_raw.py`.
No further fit, variant, paid run, push, or PR was performed.

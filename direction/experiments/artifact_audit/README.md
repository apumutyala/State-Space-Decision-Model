# Proposition: audit evaluation artifacts before buying monitor scores

## Thesis and scope

A dataset can reward recognizing its authoring process rather than making the intended decision. We test whether
code-only static features separate the frozen monitor labels and whether JevBench can be answered without its
record. This is a CPU-only descriptive audit, not a new monitor, a benchmark leaderboard entry, or an evaluation of
previously scored models. No dataset code is executed and no model/API is called.

## Prior work (links opened and verified, 2026-10-16)

- [Gururangan et al. (2018), *Annotation Artifacts in Natural Language Inference Data*](https://aclanthology.org/N18-2017/):
  hypothesis-only text classification reaches about 67% on SNLI and 53% on MultiNLI; annotation protocols leave
  label-predictive linguistic clues.
- [Poliak et al. (2018), *Hypothesis Only Baselines in Natural Language Inference*](https://aclanthology.org/S18-2023/):
  context-free baselines across ten NLI datasets can exceed majority-class performance. Removing the evidence is
  a diagnostic, not an alternative implementation of inference.
- [Geirhos et al. (2020), *Shortcut Learning in Deep Neural Networks*](https://arxiv.org/abs/2004.07780):
  benchmark-effective decision rules need not transfer to the conditions the benchmark is intended to represent.
- [Koran et al. (2026), *Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute*, §4.1](https://arxiv.org/html/2605.15377v2):
  directly relevant recent evidence: their ControlArena APPS backdoors frequently contain comments explicitly
  identifying backdoor logic; they strip comments before evaluation. This is a documented dataset artifact, not
  a claim that all their monitor performance is spurious.
- [JevBench upstream](https://github.com/fstandhartinger/jevbench) describes typed decisions and warns that the
  public split can be selected against. Here its pinned public-231 remains **EVAL-ONLY**: no fitting, tuning,
  calibration or predictor selection on these items. Modal-label statistics below are in-sample descriptions,
  never held-out predictive results.

## Monitor results and verdict

**Verdict: do not fund further raw-dataset monitor comparisons as evidence of covert sabotage detection.**
The eight-feature **style-only** model reaches AUROC **0.772 [0.698, 0.837]** on unseen problems. All 41 cheap
features reach **0.951 [0.918, 0.974]**. Thus matched model identity and prompts did **not** match the realized
styles. High AST performance can also reflect genuinely suspicious behavior; the style-only result is the
cleaner evidence of an unwanted authoring channel.

Training: 129 problems, 219 honest/65 backdoors; calibration: 65 problems, 119/37; evaluation: 130 problems,
235/60. Brackets below are 95% problem-bootstrap CIs. TPR columns are **empirical test-ROC points**, not
deployment thresholds. Classifier rows operate at 4/235 = 1.70% and 11/235 = 4.68% actual empirical FPR.

| Static method | AUROC | TPR at ≤2% FPR | TPR at ≤5% FPR |
|---|---:|---:|---:|
| Style-only logistic, 8 features | .772 [.698, .837] | 6.7% [0, 27.9] | 28.3% [8.0, 45.5] |
| AST-only logistic, 30 features | .909 [.861, .949] | 46.7% [23.9, 71.9] | 63.3% [44.7, 81.7] |
| All-feature logistic, 41 features | .951 [.918, .974] | 60.0% [19.6, 79.7] | 75.0% [58.3, 87.3] |
| Equality to numeric literal of magnitude >9 | .692 [.621, .764] | 38.3% [24.1, 52.9] | 38.3% [24.1, 52.9] |
| Explicit tell substrings | .533 [.508, .567] | 6.7% [1.5, 13.4] | 6.7% [1.5, 13.4] |

Both last two rows have 0/235 observed false positives, not proof of zero population FPR. Complete prespecified
results, including comment/identifier tells, special branches, string/container comparisons, length ratios and
reference/honest-anchor distances, are in `monitor.json`. Honest-anchor distance is near chance:
AUROC .517 [.434, .604] on 105 non-anchor honest programs and 60 backdoors. No identifier tell is found.

**Prospective calibration is much worse than the empirical ROC suggests.** For the all-feature model, the
calibration-2% threshold yields held-out TPR 85.0% [57.4, 94.6] but **FPR 11.9% [1.7, 17.9]**; the calibration-5%
threshold yields TPR 86.7% [77.6, 97.7] and **FPR 13.6% [8.4, 23.2]**. These are not valid 2%/5% deployments.

### What actually carries the label

Whole-corpus descriptive counts/medians (not independent confirmation):

| Feature | Honest, n=573 | Backdoor, n=162 |
|---|---:|---:|
| Median characters / lines | 538 / 24 | 245 / 13 |
| Median blank-line fraction | .100 | .000 |
| Median identifier length / underscore fraction | 3.36 / .070 | 2.75 / .020 |
| Median indentation | 5.62 | 2.54 |
| Any function definition | 343 | 47 |
| Numeric magic equality | 0 | 53 |
| Literal-container equality | 0 | 7 |
| Literal-gated output/return branch | 81 | 97 |
| Explicit `backdoor`/`sabotage`/`intentional` substring | 1 | 23 |
| Broader tell in source comments | 3 | 27 |

The fitted style model combines shorter names, fewer underscores/blank lines, and source-layout/comment ratios;
all coefficients are recorded. Correlated coefficients are **not causal attributions**, but these eight features
collectively predict the label without understanding code. This is not only a problem-difficulty effect:
within the 42 evaluation problems containing both labels, style-only pairwise AUROC is **.891 [.813, .952]**
(AST .958 [.899, 1.000]; all features .988 [.964, 1.000]).

There are no exact duplicate sources or cross-problem AST duplicates, but **19 extra AST-equivalent rows** and
**one AST-equivalent group with both labels**. Comments/layout can differ while syntax is identical; behavior
alone cannot identify intended sabotage versus an honest bug in that group. Finite-test honest labels and
unvalidated semantic trigger constraints remain separate label-validity limitations.

## JevBench results and verdict

**Verdict: a modest option-length artifact, not wholesale record-free solvability.** The fixed longest-option
rule answers **89/231 = 38.5% [32.1, 46.0]** correctly, versus **31.8%** random-label chance: excess
**6.8 percentage points [2.4, 12.5]**. It gets 51/111 hard items right (45.9% [39.1, 56.5], chance 33.6%).
The intervals are descriptive and unadjusted for examining several rules. No new rule was chosen or tuned.

| Fixed rule | Overall accuracy, 95% family CI | Open-only Intelligence, 95% family CI |
|---|---:|---:|
| First option | 30.3% [22.7, 38.9] | 2.0 [0, 22.9] |
| Last option | 29.0% [22.5, 35.1] | 0.0 [0, 5.6] |
| Longest option | 38.5% [32.1, 46.0] | 11.1 [4.0, 35.6] |
| Shortest option | 30.7% [24.5, 37.5] | 11.9 [0, 28.2] |
| Question–option lexical overlap | 31.2% [25.0, 37.4] | 0.5 [0, 16.7] |

Accuracy percentages by type and tier (full CIs and type×tier cells in `jevbench.json`):

| Rule | Choice (139) | Noul (74) | Score (18) | Easy (48) | Standard (72) | Hard (111) |
|---|---:|---:|---:|---:|---:|---:|
| Uniform chance | 23.0 | 50.0 | 24.7 | 28.4 | 31.1 | 33.6 |
| First | 18.7 | 52.7 | 27.8 | 31.2 | 33.3 | 27.9 |
| Last | 21.6 | 47.3 | 11.1 | 29.2 | 30.6 | 27.9 |
| Longest | 32.4 | 54.1 | 22.2 | 33.3 | 30.6 | 45.9 |
| Shortest | 21.6 | 45.9 | 38.9 | 27.1 | 33.3 | 30.6 |
| Lexical overlap | 23.7 | 44.6 | 33.3 | 27.1 | 33.3 | 31.5 |

Always-yes gets 35/74 = **47.3% [39.4, 55.0]**, always-no 39/74 = **52.7% [45.0, 60.6]**. Both get 50% on
easy/standard noul; hard noul is 44.7% yes versus 55.3% no. There is no demonstrated strong exploitable yes prior
in the labels themselves. This does not explain away the separately observed model yes bias.

The requested **in-sample** majority descriptions are separate from fixed predictors:

| Oracle description | Accuracy, 95% family CI | Valid-label coverage | Open-only Intelligence |
|---|---:|---:|---:|
| One literal mode per type | 21.2% [11.8, 31.3] | 45.0% | 9.1 |
| One literal mode per family | 26.8% [19.3, 35.4] | 56.7% | 4.0 |
| One modal position per type | 35.9% [27.9, 45.1] | 100% | 11.8 |
| One modal position per family | 43.3% [36.9, 50.5] | 99.6% | 25.1 |

There are **112 distinct gold label strings among 139 choice items; 92 occur only once**. Thus a single literal
mode is mostly unavailable, while restricting modes to each item's available labels would leak its answer key.
The discarded initial implementation did exactly that; its artificially high results are not evidence of
benchmark solvability. Position-mode figures are still optimistically in-sample and must not be promoted to a
trained policy. Only 18 families and 18 ordinal items make the uncertainty large; some tier cells have one family,
where a degenerate bootstrap interval supplies no between-family uncertainty estimate.

**How much is solvable without the record?** One fixed rule reproduces 38.5% of answers, about 16 more answers
than the uniform-chance expectation of 73.4/231. That is an empirical record-free score, **not** proof that 38.5%
of items are intrinsically solvable without evidence. None of these fixed methods approaches reliable decision
performance; open-only Intelligence is at most 11.9 among them, and ordinal distance explains why its ranking
differs from exact accuracy. The modest length effect merits correction, not dismissal of the entire benchmark.

## Frozen method

`audit.py` was committed as `4fab95e3` before running audit scores. Monitor features, classifier penalties, splits,
fixed record-free rules and bootstrap settings were frozen there; none was selected on evaluation performance.
One methodological correction followed the first run: an eligible-label frequency lookup is **not** a single
majority label, and can memorize unique answer strings. It was replaced with one literal mode per type/family,
with position-mode companions and aggregate-only concentration statistics. No fixed predictor or monitor result
was changed. The earlier monitor README already disclosed 23 sources containing `backdoor`, `sabotage` or
`intentional`, so this is not a blind replication of that particular diagnostic.

### Monitor v0

Input: `$SSDM_PRIVATE/monitor_v0/dataset.json`, SHA-256
`bdff0e9aec67ca848bdd763e3821d09799f34e1f3aeb94250a44428a6b5b6106`.
The builder reports 162 execution-verified backdoors and 573 honest programs, from 324 APPS problems. The audit
uses the **clean code**, not the separately generated attack comments, triggers, IDs containing author mode,
execution outcomes, existing monitor scores, or problem statements as classifier features. Prior execution labels
are accepted as supplied; their semantic validity is not re-established here.

- **Label-blind feature extraction:** source length; comment fraction; explicit tell substrings; comment and
  identifier tells (`backdoor`, `sabotage`, `intentional`, `trigger`, `special case`, `edge case`); equality against
  numeric literals with magnitude >9, strings of length ≥3 other than `__main__`, or literal containers;
  literal-gated branches containing output/return; and 26 fixed AST node-type counts.
- **Style-only classifier:** eight features—characters, lines, blank/comment/comment-line fractions, mean
  identifier length, underscore frequency and mean indentation. It cannot inspect token meanings, constants,
  the specification, or execution. **AST-only** and **all-feature** classifiers are separate diagnostics.
- **Tiny classifier:** logistic regression with summed log-loss plus `0.5 * ||weights||²` (unpenalized intercept),
  `log1p` features and training-only standardization; SciPy L-BFGS-B, no cross-validation or hyperparameter search.
  Sort problems by SHA-256 of `20261016:problem_id`; first 40% train, next 20% calibrate, remaining 40% evaluate.
  Every submission for a problem remains together. These audit splits are distinct from the paid monitor splits;
  monitor training here is authorized solely for this artifact diagnostic, not a reusable training corpus.
- **Extra-information diagnostics:** length divided by the median length of all programs for that problem is
  label-blind but transductive; token edit similarity to the independent APPS reference uses unavailable-at-monitor
  reference information. Diff-from-honest necessarily needs honest labels: choose a deterministic honest anchor
  per problem, exclude that anchor from scoring, and compare remaining programs to it. This avoids the degenerate
  zero self-distance for honest examples. It is explicitly privileged, never included in a fitted classifier.
- **Metrics:** held-out AUROC (ties count half); maximum empirical held-out TPR with FPR ≤2%/5%, no interpolation
  or tie-breaking; also thresholds calibrated on honest calibration programs, with their actual held-out FPR.
  Flag only scores **strictly greater** than the cutoff. The empirical-ROC columns use test negatives to locate a
  curve point, not to claim a deployable FPR guarantee. The calibrated columns do not guarantee population FPR.
- **Uncertainty:** 2,000 seeded, 95% percentile bootstrap draws of entire problems; calibration and evaluation
  resampled independently and cutoffs recomputed. Classifiers stay frozen, so CIs exclude training variability.
  Within-problem AUROC additionally averages positive–negative comparisons over evaluation problems containing
  both labels, then bootstraps those problems. This checks whether problem-composition differences alone explain
  separation. All intervals are marginal, unadjusted, and conditional on this corpus; degenerate zero intervals
  do not establish zero population risk.

### JevBench public-231

`jevkit.items.load_public()` verifies all three pinned file hashes at revision
`b6b8fff7e345b98c060ad26c13308860ddc67004`. The prediction function receives only question instructions, option
criteria and source label order—never `state` or gold. Rules: first, last, longest/shortest option-criterion text
in characters, and maximum word-set Jaccard overlap between instructions and each option. Ties use the first
source-order option. Binary `false`/`true` criteria map to `no`/`yes`. Always-yes and always-no apply only to `noul`.

The requested majority statistics count gold labels within type or family and choose **one** literal mode for
the entire group (lexicographic ties). If that label is absent from an item's options, the answer is invalid and
counts wrong; coverage is reported. Because semantic label vocabularies differ across items, we also report
the modal source-order position by type/family (ties favor the earlier position). These are **oracle/in-sample
modal agreement**, not trained models, held-out predictive baselines, or evidence of generalization. Their
bootstrap keeps the descriptive mode fixed; it does not remove selection optimism. No majority-derived policy
is exported for deployment.

Accuracy uses exact labels, compared with uniform random-label chance `mean(1/K)`, overall, by type, tier, and
type×tier. CIs resample entire families within each reported subset, retaining related tiers/paraphrases. Intelligence
uses one-hot answers through `jevkit.scoring`, including ordinal distance rather than ordinal exact accuracy and
renormalized present-tier weights. It is **open-only**, with no sealed items or public–sealed gap penalty; it is not
a full leaderboard score. Intelligence CIs also resample families. No best-performing rule is fitted or selected.

## Reproduction and artifacts

From this worktree, with the existing CPU environment:

```sh
env SSDM_PRIVATE="$HOME/ssdm/private" SSDM_DATA="$HOME/ssdm/data" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  nice -n 10 /usr/bin/time -v "$HOME/ssdm/.venv/bin/python" \
  direction/experiments/artifact_audit/audit.py
```

`monitor.json`, `jevbench.json`, and `run.json` contain aggregates only. Private feature rows and predictions go to
`$SSDM_PRIVATE/research-audit/`, never git. No additional packages, hosted compute, model weights, dataset-code
execution, or spend approval are needed. This one-off analysis changes no reusable runtime or evaluation code.

## Experiment actually run

Final command above, code `b27f0c11`: **exit 0**, 10.24 seconds process wall time (9.61 seconds inside the script),
124,272 KiB peak RSS (about 121 MiB), one CPU thread, Python 3.12.13 / NumPy 2.5.4 / SciPy 1.18.1. All three
monitor optimizations converged; every reported interval has 2,000 valid bootstrap draws. The initial pass before
the majority-definition correction took 8.61 seconds; total measured wall time was under 19 seconds. **Paid cost:
$0.** No test suite was run: this is one standalone experiment, not reusable runtime code. The run itself checked
the monitor corpus hash, pinned JevBench hashes, optimizer convergence and validity of every fixed baseline answer.

## What this means, and concrete next steps

The proposition is worth pursuing **as dataset repair and an evaluation prerequisite**, not as a claim to have
built a general sabotage detector. This audit neither identifies which features an LLM monitor actually uses nor
establishes transfer to a new generator, realistic repository, adaptive attacker or independently validated label
set. High AST AUROC may partly be legitimate backdoor detection; it must not all be labeled spurious. Style-only
separation nevertheless establishes a substantial unwanted channel in this particular dataset.

1. **Monitor: keep v0 frozen and version the repair.** Strip source comments and standardize formatting in both
   arms; inspect safe identifier normalization. Do not silently scrub the frozen evaluation and keep its name.
   AST preservation is a static check, not permission to run submissions outside a no-network sandbox.
2. **Match structure, not just the generator.** Construct same-problem honest/backdoor pairs from the same base
   rewrite with a minimal behavior-changing edit. Match layout, identifier and function-wrapper conventions;
   include legitimate rare branches/magic equalities in honest controls. Balance accepted replicates per problem
   and retain problem-disjoint confirmation. Removing only the 23 overt tells cannot remove the style-only signal.
3. **Deduplicate and adjudicate.** Collapse AST-equivalent same-label replicates for weighting; investigate the
   mixed-label AST group and separate intentionality from ordinary wrong-answer behavior. Independently check
   trigger constraints and corrected matched labels before any paid monitor comparison.
4. **JevBench: retain public-231 as evaluation-only.** Report these fixed baselines alongside model scores. For an
   independently authored successor, counterbalance option positions and criterion lengths within type×family×tier,
   balance yes/no labels, and create matched record-counterfactual pairs with unchanged questions/options but
   changed correct answers. Keep paraphrase groups together; never learn those repairs or select predictors on
   the public items. A sealed confirmation set is needed for a new performance claim.

**Next-step cost:** $0 paid compute to design/manual-review a small matched-pair repair, perform static normalization
and rerun these frozen diagnostics. Human authorship/adjudication time is not priced here. Any later generation or
sandbox execution needing hosted resources requires a separate owner-approved budget; no such run is authorized
or necessary to act on this audit's stop-spend recommendation.

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

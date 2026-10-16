# Proposition: normalize presentation, then test behaviour with matched repairs

## Thesis

Canonical layout and deduplication should remove obvious authoring cues, but they cannot make unrelated honest
and backdoored implementations a causal test of behavioural monitoring. B2 remains a within-program
behaviour × presentation experiment, not a contest to drive every static classifier to chance. A magic-constant
branch may **be the defect**: erasing it during presentation normalization would corrupt the positive label.
The paid Monitor Card remains paused throughout this experiment.

## Prior work (primary links opened 2026-10-16)

- Our [artifact audit](../artifact_audit/README.md) measured style AUROC .772 [.698, .837], all-feature AUROC
  .951 [.918, .974], 19 extra AST-equivalent rows and one mixed-label AST group.
- [Najt et al., SLEIGHT-Bench (2026)](https://alignment.anthropic.com/2026/sleight-bench/) uses near-identical
  benign variants of malicious transcripts to isolate monitor sensitivity to harmful actions.
- [Štorek et al., ACL 2026](https://aclanthology.org/2026.acl-long.19/) distinguishes lexical from semantic
  recall and introduces counterfactual semantic-sensitivity measurements.
- [Ye et al. (2025)](https://arxiv.org/abs/2512.19215) studies semantics-preserving transformations as triggers
  against neural code models; that is not the same threat as a defective APPS program, but motivates controlling
  presentation. Their normalization countermeasures are only partial mitigation.

The contribution is this corpus repair and a feasibility measurement for the
[B2 crossover](../proposition_scan/README.md#b2-backdoor-behavior--presentation-crossover--rank-2), not the invention
of matched counterfactual evaluation.

## Frozen method

The protocol and implementation are committed before the single held-out comparison. The v0 counts/results were
already known; this is a controlled follow-up, not an untouched confirmation set. No monitor scores or model calls
are used. No submitted program is compiled or executed; only AST/token inspection is permitted.

1. Read the SHA-256-pinned 735-row v0 corpus. Normalize every source independently of its label with Python 3.12
   `ast.parse` → remove leading docstring expressions → `ast.unparse` plus a final newline. Comments disappear
   through parsing. Remove runs of leading string expressions so a second string cannot become a new docstring;
   insert `pass` only for an otherwise empty function/class body. Do **not** rename identifiers: binding, reflection,
   keyword arguments and imported names make that a separate safety problem. Preserve branches and constants.
2. Require exact round-trip AST equality after the declared docstring deletion, including identifier spellings.
   This checks syntax-tree preservation, **not universal semantic equivalence**: source/line introspection and
   `__doc__` can observe changes. Record conservative reflection flags rather than claim proof of their absence.
   Normalize references for reference-distance diagnostics too. V2 is clean-source data; the old attack-comment
   payloads are not carried into it. IDs, labels, triggers and generator metadata never enter classifier features.
3. Group globally by normalized AST hash. Drop **every row** in a mixed-label group. Otherwise retain the minimum
   `(original source SHA-256, submission ID)` representative. Log every exclusion and its reason/representative.
   Save the full manifest privately and its hash in git. Also save original code for precisely the retained rows:
   this matched-original control separates normalization from changes in corpus composition.
4. Import `artifact_audit.audit`, rather than reimplementing its feature extraction, logistic optimizer, scores,
   metrics or bootstrap. Its small API extension accepts an alternate corpus and fixed problem assignments;
   its private export additionally includes privileged-anchor scores for paired comparisons. Statistical routines
   are unchanged. Refit each frozen recipe on that corpus's training rows; never tune penalties/signs/features.
   Preserve v0's problem assignments even if deduplication empties a problem: SHA-256 ordering of
   `20261016:problem_id`, 40% train / 20% calibration / 40% evaluation.
5. Run all 15 inherited baselines on original, matched-original and v2. Add six **prespecified** family-only
   logistic ablations with the identical frozen fitting recipe: size (2), layout (4), identifiers (2), AST node
   counts (26), branch/equality features (4), and tells (3); these partition the 41 features. The existing
   eight-feature style and 30-feature AST models remain unchanged.
6. Use the inherited 2,000 problem-cluster bootstrap draws and seed, independently sampling calibration and
   evaluation problems. Models remain fixed in bootstrap draws; uncertainty excludes training/generation variation.
   Report marginal 95% intervals and paired v2−original / v2−matched-original AUROC changes using the **same**
   original problem draws for both sides, retaining multiplicities. No AUROC sign flips. The empirical low-FPR
   points are descriptions, not deployment guarantees. All intervals are unadjusted exploratory diagnostics.
7. Draw 20 of the original 162 backdoors without replacement using NumPy `default_rng(20261017)` on sorted
   submission IDs, before any audit scores. Generate single `If`/`IfExp` branch-bypass candidates, excluding
   main-entry guards. Try both arms; compare each resulting AST with every honest counterpart for that problem.
   Retain only patches whose source-preserving and canonical renderings parse to the same modified AST.
   Exact AST agreement with an honest counterpart is strong static evidence, not proof of correct behaviour.
   AST-token edit distance ranks inspection order only, never the sampled programs or acceptance threshold.
8. Inspect **all 20**, recording acceptance and rejection rationales privately. A plausibly minimal repair must be
   one local bypass that removes an isolated exceptional output/result and leaves the apparent intended path,
   supported by the honest/reference implementation and statement/trigger. Merely parsing, shortening code, or
   reducing AST distance is insufficient. Ambiguous domain validity or collateral changes are failures of this
   feasibility screen. Report exact-honest matches separately from assistant-adjudicated plausible repairs.
   Four candidate cells (defective/repaired × original/canonical presentation) stay private. A source patch may
   also remove comments attached to a deleted branch; that is a remaining crossover-design confound, not silently
   counted as a pure behaviour intervention. Dynamic validation and independent adjudication are required next.

## Interpretation fixed in advance

Identifier spelling and literal tell/comment channels are presentation. Size, function organization, indentation
and AST counts mix presentation/implementation choices with genuine algorithmic structure; they are not clean
causal evidence either way. Literal equality and exceptional output branches are behaviour-linked and can be
legitimate defect signal. Strong separation there does not imply a failed artifact repair; only matched repairs
and specification/trigger checks can establish whether those branches are wrong.

## Reproduction and cost

From the worktree root, using the existing environment and no new packages:

```sh
env SSDM_PRIVATE="$HOME/ssdm/private" PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 timeout 3300 prlimit --as=2147483648 --cpu=3300 \
  "$HOME/ssdm/.venv/bin/python" -X utf8 -m direction.experiments.monitor_repair.run
# After the complete static adjudication is recorded privately:
env SSDM_PRIVATE="$HOME/ssdm/private" PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 10 "$HOME/ssdm/.venv/bin/python" -X utf8 \
  -m direction.experiments.monitor_repair.run --summarize-review
```

Private outputs live under `$SSDM_PRIVATE/research-monitor-repair/`: v2 and matched-original corpora,
normalization/exclusion manifests, audit feature/score rows, all candidate code, and adjudication. Only code,
this proposition and compact aggregates belong in git. This experiment costs **$0** paid compute.

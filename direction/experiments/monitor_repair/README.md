# Proposition: normalize presentation, then test behaviour with matched repairs

## Thesis

Canonical layout and deduplication should remove obvious authoring cues, but they cannot make unrelated honest
and backdoored implementations a causal test of behavioural monitoring. B2 remains a within-program
behaviour × presentation experiment, not a contest to drive every static classifier to chance. A magic-constant
branch may **be the defect**: erasing it during presentation normalization would corrupt the positive label.
The paid Monitor Card remains paused throughout this experiment.

**Measured verdict: v2 is not fit to resume the Monitor Card.** Style-only AUROC remains **.768 [.699, .831]**,
versus .772 [.698, .837] before repair; the paired change is **−.004 [−.044, .034]**. Identifier spelling features
alone retain AUROC **.748 [.678, .811]**. B2 is worth pursuing instead: **19/20** seeded candidates yield a
plausibly minimal static repair with an acceptable supplied witness, but none is dynamically validated here.

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

The protocol and implementation were frozen in **`8eee6170`** before the single held-out comparison. The v0
counts/results were already known; this is a controlled follow-up, not an untouched confirmation set. No prior
model-monitor scores or model calls are used. No submitted program is compiled or executed; only AST/token
inspection is permitted.

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

## Experiment actually run

The frozen command completed on 2026-10-16 in **53.17 seconds**, with **127.04 MiB peak RSS**, CPU only under
`nice -n 10`, a 55-minute deadline and 2 GiB virtual-memory cap. Python **3.12.13**, NumPy **2.5.4**, SciPy
**1.18.1**; no packages installed, no paid APIs, no program execution. All 15 original-baseline AUROCs reproduced
within `1e-10`. This run, not a simulation, includes the normalization round-trip checks and all requested
problem-disjoint audits. No separate test suite was run.

### Corpus and exclusions

V2 has **715 programs: 554 inherited honest labels and 161 inherited backdoor labels, across the same 324
problems**. Removed: 18 same-label duplicate rows plus both rows of the single mixed-label AST group
(19 honest / 1 backdoor total). No AST duplicates or mixed-label groups remain. All 735 normalized sources
round-trip to the declared AST; comments occurred in 213, docstrings in **zero**. Thus no docstring-dependent
behaviour was actually altered in this corpus. Nine sources contain `eval`; these are flagged, not executed or
silently claimed safe. Identifier canonicalisation was deliberately skipped.

| Partition | Problems | Original honest / backdoor | V2 honest / backdoor |
|---|---:|---:|---:|
| Training | 129 | 219 / 65 | 214 / 65 |
| Calibration | 65 | 119 / 37 | 116 / 37 |
| Evaluation | 130 | 235 / 60 | 224 / 59 |

Private v2: `$SSDM_PRIVATE/research-monitor-repair/v2/dataset.json`, SHA-256
`618e6912e9aff7daeb47c43285ea037540f5d0152b18cf5c15401c69970414f1`.
Every exclusion is recorded in `exclusions.json`, SHA-256
**`503cbb3b47beadec5bd997c7b258b0c2abcf98b26251ac6ecf29dfeb76ead7f4`**.
Source, normalization, matched-original, candidate and review hashes are in `results.json`.
These labels are inherited from v0, **not re-execution-verified labels for changed source**.

### Residual-separation table

Held-out AUROC, 95% problem-cluster CI; Δ is paired **v2 minus all original rows**, so it includes deduplication.
The family ablations use the frozen logistic recipe, not selected or tuned classifiers.

| Method / feature family | Original | V2 | Paired Δ | Residual interpretation |
|---|---:|---:|---:|---|
| Style-only, 8 | .772 [.698, .837] | **.768 [.699, .831]** | −.004 [−.044, .034] | Presentation plus structural proxies |
| All features, 41 | .951 [.918, .974] | .936 [.892, .969] | −.015 [−.045, .006] | Mixed; not an artifact-only score |
| Frozen AST model, 30 | .909 [.861, .949] | .917 [.873, .954] | +.008 [−.002, .028] | Behaviour-linked plus implementation structure |
| Size, 2 | .716 [.656, .779] | .681 [.615, .743] | −.035 [−.073, −.003] | Mixed: shorter implementations persist |
| Layout, 4 | .759 [.682, .824] | .647 [.576, .712] | −.112 [−.163, −.054] | Residual canonical layout reflects nesting/functions |
| Identifier spelling, 2 | .741 [.673, .804] | **.748 [.678, .811]** | +.007 [−.001, .015] | Presentation-dominated lexical channel |
| AST node counts, 26 | .828 [.761, .886] | .836 [.773, .895] | +.008 [−.004, .029] | Mixed structure, not demonstrated behavioural understanding |
| Branch/equality features, 4 | .838 [.765, .900] | .845 [.775, .907] | +.007 [−.002, .025] | Legitimate behaviour-linked signal can remain |
| Tell counts, 3 | .550 [.517, .589] | .500 [.500, .500] | −.050 [−.089, −.017] | Held-out lexical tells removed |

The matched-original control holds the 715 surviving rows fixed. Its style AUROC is .781 [.709, .846], versus
v2 .768: normalization-only Δ **−.013 [−.054, .025]**. All-feature Δ is −.018 [−.048, .002]; layout-family Δ is
**−.119 [−.169, −.063]**. Identifier, AST-count and branch/equality scores are **identical** before/after
normalization on matched rows (Δ = 0 for every bootstrap draw). Thus a real layout cleanup did not establish
a reduction in overall style separation; this is not evidence that the two overall AUROCs are equivalent.

The frozen tell-substring rules, with no fitted signs:

| Tell rule | Original AUROC | V2 AUROC | Paired Δ |
|---|---:|---:|---:|
| Explicit substrings anywhere | .533 [.508, .567] | .500 [.500, .500] | −.033 [−.067, −.008] |
| Broader substrings in comments | .550 [.517, .589] | .500 [.500, .500] | −.050 [−.089, −.017] |
| Broader substrings in identifiers | .500 [.500, .500] | .500 [.500, .500] | .000 [.000, .000] |

The zero-width .500 intervals reflect constant held-out scores, not proof of no lexical channel in the population.
All 15 inherited methods, all six families, both paired comparisons, feature diagnostics and denominators are in
`results.json`; complete low-FPR metrics and fitted coefficients stay in private audit summaries. Public aggregates
also retain the style/all-feature low-FPR metrics. For example, the v2 all-feature nominal calibration-2% cutoff
actually produces **9.38% held-out FPR [.45%, 16.67%]**, not a deployable 2% guarantee.

### Which residual signal is presentation, and which is behaviour?

- **Presentation-dominated:** mean identifier length and underscore frequency are unchanged by layout
  normalization. Whole-corpus v2 medians are 3.41 vs 2.75 characters and .071 vs .020 underscore fractions
  (honest vs backdoor). The feature extractor includes builtin/import/API names as well as local variables:
  its .748 AUROC is not a causal estimate for *local renaming alone*.
- **Mixed structure:** canonical indentation follows syntactic nesting; `unparse` inserts blank lines around
  functions. Honest/backdoor sources still differ in size (median 506/305 characters) and function use
  (339/554 vs 46/161). Their residual separation cannot honestly be assigned entirely to artifacts or entirely
  to sabotage without matched interventions.
- **Behaviour-linked:** 53/161 backdoors and 0/554 honest programs contain the frozen magic-number equality
  feature; its held-out AUROC is .695 [.625, .769]. Literal-gated output/return branches yield .725 [.655, .794].
  Such a branch often implements the witnessed defect. We preserve it; removing it would be a behavioural repair,
  not presentation normalization. Neither heuristic alone proves a specification violation or intent.
- **Important qualification to the prior tell classification:** one retained positive outside evaluation still
  contains an explicit tell **inside an input-comparison string literal**. The literal participates in its trigger,
  so deleting/changing it is not semantics-preserving. Comments and identifier tells are absent throughout v2,
  but blanket claims that “all tells are removed” or “every tell is merely presentation” would be false.

### B2 static feasibility

The seeded draw contained **20 programs from 19 problems**. All 20 have parseable four-cell candidates with
source-patch/canonical AST agreement; none of the generated source patches failed that check. **Three** sampled
programs have a one-branch repair exactly AST-equal to a same-problem honest program. After examining every
sample, its specification, supplied trigger and reference, **19/20 (95.0%, problem-bootstrap CI [84.2%, 100%])**
pass the frozen plausibly-minimal screen: three exact-honest matches and 16 reference-supported local bypasses,
covering **18 distinct problems**. This is an assistant's static adjudication, not independent human ground truth;
the interval excludes adjudication error and semantic uncertainty.

The remaining program has a clear parseable local repair but a malformed supplied witness with extra test-case
tokens after the declared case count, so it fails the frozen validity rule. It is **not silently replaced** by a
cleaned witness. Separately, static reading identified an allowed-domain corner case where a supplied reference
formula itself disagrees with its problem statement. Removing the planted branch does not repair every inherited
bug. These observations reinforce why “execution-verified divergence from a reference” is weaker than a valid
semantic label. No invalid-trigger example was retrospectively removed from the already frozen v2 audit.

## Meaning, nonclaims and next step

**Do not resume the paid Monitor Card on v2.** This is a useful deterministic cleanup, not a repaired causal
benchmark: identifier/presentation separation survives, the entire dataset's trigger domains are unchecked,
changed sources have not been re-executed, and imperfect references remain. Higher AST performance is not itself
a reason to erase behaviour. No model-monitor ability, universal program equivalence or real-repository
generalization is measured here.

**Pursue B2, not another raw-corpus monitor comparison.** The next stage costs **$0 paid compute**, proposed as a
CPU-only batch capped at **50 minutes / about 2 GiB**. In a no-network, unprivileged sandbox with read-only runtime,
no secrets/project mounts and strict time/process/output limits, validate the 19 candidate repairs on all supplied
tests, domain-checked triggers, independent references and fresh domain-valid inputs. Check both defective
presentations retain the witnessed defect and both repaired presentations remove it without off-trigger
regressions. Log any replacement witness as new evidence. Passing finite tests still does not prove equivalence.

Before any model scoring, independently adjudicate repairs and reference/trigger validity; ensure behaviour edits
do not also remove distinctive comments, and freeze presentation transformations with lexical-binding safeguards
and balanced variants. Extend the score-blind procedure to all available backdoors, preserving problem clusters,
and assess B2's previously declared minimum of 60 feasible independent problems. This 18-problem static pilot
does **not** meet that scale requirement. Paid scoring needs a later explicit owner-approved budget and a corpus
that passes those validity gates.


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

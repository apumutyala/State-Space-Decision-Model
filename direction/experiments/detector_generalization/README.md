# Proposition: a cheap, judge-independent injection signal

## Thesis

A detector trained only on permitted synthetic records might recognise injected authority even when a judge
resists it. The useful target is **high detection at low legitimate-evidence false-positive rates under shift**,
not synthetic accuracy. This experiment tests that proposition with no hosted inference or paid compute.
The protocol below is fixed before held-out scoring; it does not certify a gate.

**Verdict: no-go for this six-template sparse recipe as a deployable gate or attack-prevalence certification
signal.** Synthetic-dev AUROC is 1.000, but both frozen operating points flag **every held-out benign input**.
Even the descriptive Card ROC catches only 1.5% of static attacks at <=1% FPR. AgentDojo shows some useful ranking
on its single standard attack family, not a transferable operating point. This does not rule out better cheap detectors.

## Prior work (primary links opened 2026-10-16)

- [Meta Prompt Guard](https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Prompt-Guard/MODEL_CARD.md)
  classifies benign/injection/jailbreak text and recommends application-specific fine-tuning and layered defenses.
  [Llama Prompt Guard 2](https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md)
  instead uses a binary explicit-override objective, offers 22M/86M-backbone variants, and explicitly warns about
  adaptive attacks. Neither is evaluated here. The original “86M” model additionally has 192M embedding parameters;
  its name is not a <=100M total-parameter claim.
- [DataSentinel, Liu et al., 2025](https://arxiv.org/abs/2504.11358) trains a detector against adaptive injections
  using minimax optimization. Its known-answer detector is deliberately made more susceptible to injected tasks;
  the paper notes weaker detection when injected and target tasks coincide, especially relevant to same-task
  answer steering here. Our linear model has neither that mechanism nor adversarial training.
- [PromptShield, Jacob et al., 2025](https://arxiv.org/abs/2501.15145) emphasizes deployability at low FPR and
  conversational/application-structured curation. Its §3.2.2 explicitly recommends future threshold calibration
  on validation rather than evaluation data; we separate frozen-dev points from descriptive ROC points.
- [AlignSentinel, Jia et al., 2026](https://arxiv.org/abs/2602.13597) distinguishes misaligned instructions, aligned
  instructions and non-instruction inputs using attention features. Legitimate instructions are not attack labels.
- [Hackett et al., 2025](https://arxiv.org/abs/2504.11168) demonstrate character/AML detector evasion, including
  Prompt Guard. [Nasr et al., 2025](https://arxiv.org/abs/2510.09023) argue for defense-aware, resourceful attacks
  rather than static-string testing. Our small grammar search is a lower bound on adversarial capability.

## Frozen method

**Training.** Only `$SSDM_PRIVATE/distill_v0/teacher-train-screened/units.jsonl.gz`: synthetic clean, inject,
authority, evidence and policy variants. Labels come from the construction, not teacher decisions: detecting an
attempt must not depend on whether the judge obeyed it. The file contains 3,107 source groups, each with clean,
inject, authority and policy variants, plus 237 screened evidence variants. Split 80/20 by shuffled source group
(seed 20261016); fit vocabulary/IDF and coefficients on train only. All 3,107 clean content hashes are distinct.
No public item, Card template or workflow output is used to fit, choose, or tune the model.

**One final detector.** Concatenate only host-labelled untrusted segment contents using the existing protocol
JSON serializer. Do not expose IDs, source names, segment kinds, labels, question text or trusted policy. Use
word 1–2 grams (30k cap) and character-within-word 3–5 grams (60k cap), TF-IDF with sublinear TF and min_df=2,
then balanced logistic regression, C=1, liblinear, max_iter=1000. No hyperparameter search, neural encoder, downloads
of model weights, or evaluation-driven revisions. `SparseDetector` implements the existing gate signal interface;
its score is **not** a calibrated probability of an attack. Policy controls deliberately share clean input: the
host's trust boundary prevents scanning authorized instructions, not learned semantic policy understanding.

**Operating points and uncertainty.** Freeze two thresholds on synthetic-dev negatives at empirical FPR <=1%
and <=5%, with whole-score ties rejected conservatively. Report their actual held-out FPR/TPR separately from
descriptive empirical ROC TPR at <=1%/<=5% FPR (no interpolation or deployment threshold selection on evaluation).
Report 95% percentile bootstrap CIs, 1,000 whole-cluster resamples: source groups for dev, 18 item families for
Card, suite/user-task for workflows. Metrics are unit-weighted within a resampled population. Degenerate 0/100%
intervals are not guarantees; small-family and small-evidence samples cannot establish population 1% FPR.

**Held-out evaluations, once after freezing.**

1. Card T1 inject/authority variants from `card/units.json`, against one clean per public item plus the 26
   `card/evidence-units.json` validated-evidence controls. Report evidence and policy FPR separately. “Validated”
   means the existing AI-screened cohort, not independent human adjudication. Fabricated evidence is not relabelled
   as an authority attack or a benign control.
2. T3/T3L final candidates from `card*/{units,v01-units}.json`, linked by exact record digest to `card*/search-*.json`.
   Deduplicate copied runs; report source-model/tier cohorts and the judge-search-success subset. The 257 clean +
   evidence controls are shared across these comparisons. These attacks optimized the **judge**, not this detector.
3. Actual AgentDojo tool messages from `workflow/workflow-*/traces/**/*.json`, excluding parser pilots, empty/error
   responses and host-held-call boilerplate. Positive labels require the recorded injection payload in a message
   (whitespace normalized; decode YAML for label matching, including spotlighting delimiters). Only benign episodes
   supply negatives; unmatched messages in attacked episodes remain unlabelled. Score the original message, not
   the decoded label-matching view. Deduplicate identical text/label within suite/user-task across runs and arms.
   Report banking, Slack and combined sets; this is one standard attack family, not real production traffic.

**Detector-aware probe.** On 60 hash-selected public items (EVAL-ONLY), reuse the Card grammar's held-out attack
names, four target phrasings, four positions and six wrappers. Use its seeded coverage followed by greedy
single-coordinate search (beam=1), maximizing 1 minus the detector score, 64 queries/item. Report cumulative
first evasion at 1/2/4/8/16/32/64 queries, evading-query count and recovery among initially detected candidates.
No judge is called: detector evasion is not demonstrated end-to-end attack success.

## Experiment actually run

Protocol commit `70736dca`; fitted identity and thresholds committed at `ad8cd776` before held-out scoring
([frozen.json](frozen.json)). One model, one final evaluation, no retuning. Fit: **10,120 units / 2,485 groups**;
dev: **2,545 units / 622 groups** (1,244 attacks, 1,301 negatives). Dev AUROC and both ROC recalls were
100% [100, 100]; these are familiar-template results. Frozen score thresholds were **0.04965258** (1%-dev)
and **0.03925091** (5%-dev); actual dev FPR was 0.92% and 5.00%.

Both commands ran on one CPU thread with `nice -n 10`: fitting 6.57 s; full fit command 10.09 s; full evaluation
command 17.63 s. Held-out scoring processed 1,495 distinct texts in 1.43 s (about 1,045 texts/s, batched—not
online tail latency); probe scoring/search took 5.18 s. Evaluation peak RSS was 1,006 MiB. **Paid compute: $0.**
Versions: Python 3.12, scikit-learn 1.8.0, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0, PyYAML 6.0.3.
CPU regression command `python -m pytest -q keel/tests/test_detector.py`: **11 passed in 1.34 s**.

The workflow extraction covered 648 archived episodes: 1,233 empty/error/held messages excluded, 492 unmatched
messages from attacked episodes left unlabelled, 761 provenance-labelled messages reduced to **234** after
within-task deduplication (90 injected, 144 benign). There were no unresolved standard-attack markers.

## Results with uncertainty

All brackets are 95% whole-cluster bootstrap CIs. ROC TPR columns are **descriptive**, not the frozen deployed
thresholds; percentages are not interpolated. Every Card attack row below uses the same 257 negatives. Full
precision, cohort counts, frozen-threshold rates and all CDF points are in [results.json](results.json).

| Evaluation | Attacks / benign | AUROC | TPR at <=1% FPR (%) | TPR at <=5% FPR (%) |
|---|---:|---:|---:|---:|
| Card T1 | 1,386 / 257 | .748 [.664, .836] | 1.5 [1.0, 8.6] | 9.7 [6.5, 27.2] |
| Judge-optimized T3/T3L, pooled | 270 / 257 | .692 [.607, .771] | 1.9 [0.0, 7.0] | 4.8 [0.3, 23.7] |
| Stock one-pass T3 | 60 / 257 | .703 [.610, .781] | 1.7 [0.0, 10.3] | 6.7 [0.0, 28.6] |
| Stock one-pass T3L | 60 / 257 | .711 [.621, .800] | 3.3 [0.0, 8.7] | 3.3 [0.0, 27.4] |
| Stock thinking T3 | 30 / 257 | .728 [.638, .812] | 0.0 [0.0, 11.8] | 6.7 [0.0, 33.4] |
| Student T3 | 60 / 257 | .678 [.581, .770] | 1.7 [0.0, 10.0] | 5.0 [0.0, 20.0] |
| Student T3L | 60 / 257 | .658 [.572, .746] | 1.7 [0.0, 5.1] | 3.3 [0.0, 20.7] |
| Judge-search-success subset | 258 / 257 | .693 [.608, .769] | 1.9 [0.0, 6.3] | 4.7 [0.0, 23.4] |
| AgentDojo, combined | 90 / 144 | .971 [.941, 1.000] | 17.8 [8.9, 100.0] | 100.0 [30.4, 100.0] |
| AgentDojo, banking | 46 / 50 | 1.000 [1.000, 1.000] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| AgentDojo, Slack | 44 / 94 | .965 [.936, 1.000] | 18.2 [9.1, 100.0] | 77.3 [27.3, 100.0] |

The search-success subset means the saved best candidate reached the target decision; it is **not** conditioned
on clean-judge correctness and is not the Card's clean-eligible ASR denominator. The workflow intervals condition
on this one attack family and its two injection tasks; task resampling does not create new attack families.
With only 50/94/144 negatives, the 1% ROC point allows 0/0/1 false positives, explaining its instability.

**Frozen thresholds fail under distribution shift.** Both thresholds produce 100% TPR **and 100% FPR**
on every held-out mixed set. Legitimate-control false positives:

| Legitimate controls | Units / clusters | FPR at 1%-dev threshold (%) | FPR at 5%-dev threshold (%) |
|---|---:|---:|---:|
| Synthetic-dev clean | 622 / 622 | 0.8 [0.2, 1.6] | 4.3 [2.9, 5.9] |
| Synthetic-dev evidence | 57 / 57 | 3.5 [0.0, 8.8] | 19.3 [10.5, 29.8] |
| Synthetic-dev policy | 622 / 622 | 0.8 [0.2, 1.6] | 4.3 [2.9, 5.9] |
| Card clean | 231 / 18 | 100 [100, 100] | 100 [100, 100] |
| Card validated evidence | 26 / 14 | 100 [100, 100] | 100 [100, 100] |
| Card trusted policy | 231 / 18 | 100 [100, 100] | 100 [100, 100] |
| AgentDojo benign | 144 / 24 | 100 [100, 100] | 100 [100, 100] |

For scale, held-out benign score minima were .187 (Card) and .209 (workflows), already above both frozen
thresholds. The clean/policy equivalence is structural: excluding trusted text does not rescue unfamiliar
untrusted evidence. Overall dev FPR also hides substantially worse false positives on its scarce evidence controls.

**Detector-aware search:** 0/60 items evaded either frozen threshold in 64 queries; **0/3,840 queries** evaded.
The cumulative evasion rates at 1, 2, 4, 8, 16, 32 and 64 queries were all 0% [0, 0]. All 60 first candidates
were flagged. Thus no finite first-evasion time was observed within this budget, but **this is vacuous resistance:
the same detector rejects every held-out clean request**. We did not choose a new public-data threshold to make
the probe look useful. These results provide no evasion-time bound for a usable gate and no joint judge/detector
attack success claim.

## What it means—and does not show

The intended signal must recognise attempted authority transfer independently of judge compliance. The reusable
implementation does that architecturally, but this training recipe learns a familiar-template discrimination
problem rather than a transferable low-FPR boundary. No useful operating point was demonstrated. A gate adding
this frozen signal would defer all observed clean traffic; it cannot support useful attack-inclusive certification.
Detector scores or observed alarm fractions are not ground-truth attack prevalence: calibration, sensitivity and
specificity must be validated for the target population, and adaptive attackers invalidate an iid prevalence story.

The single-family AgentDojo ranking is enough to motivate better data, not to deploy this detector. We did not
compare neural encoders, Prompt Guard, attention features, novel human attacks, production traffic, or end-to-end
judge-plus-detector utility. Narrow synthetic templates, one source-generator collection, 26 AI-screened evidence
controls, small family counts and repeated benchmark task structure limit generalization. No claim about the best
achievable cheap detector follows from this negative baseline.

**Next step and cost:** keep the detector interface, not these thresholds. Author allowed synthetic aligned/
misaligned instruction controls with task context, reserve whole domains and attack families for dev, and compare
this baseline with a <=100M-parameter CPU encoder on a **fresh sealed** evaluation cohort. Proposed cap: **$0 paid
compute and one CPU-hour**, with annotation effort separately required; this is a proposed budget, not a measured
runtime. Do not spend on certification before a low-FPR operating point survives that test.

## Reproduction and artifacts

```bash
source ~/ssdm/env.sh
uv pip install --python "$VIRTUAL_ENV/bin/python" scikit-learn==1.8.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 python -m direction.experiments.detector_generalization.run fit \
  --private "$SSDM_PRIVATE" --output "$SSDM_PRIVATE/research-detector"
nice -n 10 python -m direction.experiments.detector_generalization.run evaluate \
  --private "$SSDM_PRIVATE" --output "$SSDM_PRIVATE/research-detector"
python -m pytest -q keel/tests/test_detector.py
```

`fit` refuses to overwrite its directory; `evaluate` refuses a second evaluation. The fitted artifact, per-item
scores, full detector-aware search traces and input hash manifest stay private. Git contains code, this proposition
and compact aggregate results only. Never load a joblib artifact from an untrusted source.

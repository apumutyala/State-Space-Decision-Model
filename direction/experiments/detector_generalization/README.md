# Proposition: a cheap, judge-independent injection signal

## Thesis

A detector trained only on permitted synthetic records might recognise injected authority even when a judge
resists it. The useful target is **high detection at low legitimate-evidence false-positive rates under shift**,
not synthetic accuracy. This experiment tests that proposition with no hosted inference or paid compute.
The protocol below is fixed before held-out scoring; it does not certify a gate.

## Prior work (primary links opened 2026-10-16)

- [Meta Prompt Guard](https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Prompt-Guard/MODEL_CARD.md)
  classifies benign/injection/jailbreak text and recommends application-specific fine-tuning and layered defenses.
  [Llama Prompt Guard 2](https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md)
  instead uses a binary explicit-override objective, offers 22M/86M-backbone variants, and explicitly warns about
  adaptive attacks. Neither is evaluated here. The original “86M” model additionally has 192M embedding parameters;
  its name is not a <=100M total-parameter claim.
- [DataSentinel, Liu et al., 2025](https://arxiv.org/abs/2504.11358) trains a detector against adaptive injections
  using minimax optimization. Our linear model lacks that adversarial training; detector-aware evaluation is essential.
- [PromptShield, Jacob et al., 2025](https://arxiv.org/abs/2501.15145) emphasizes deployability at low FPR and
  conversational/application-structured curation. AUROC alone is insufficient.
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

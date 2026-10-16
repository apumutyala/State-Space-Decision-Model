# Readout calibration from synthetic data does not transfer to public items (2026-10-16, CPU, cached outputs)

## Thesis tested

The gap analysis found a one-pass "yes" prior: the model says yes 59% of the time against a 47% gold rate, and hard
yes/no carries most of the gap to H2O. If that prior is a fixed readout bias of the model, a correction estimated on
our own **non-public synthetic items** should recover public accuracy for free: no model calls and no public data
used for fitting.

## Prior work (links checked 2026-10-16)

- [Zhao et al. 2021, *Calibrate Before Use*](https://arxiv.org/abs/2102.09690): label bias estimated from a
  content-free input ("N/A") and divided out.
- [Fei et al. 2023, *Mitigating Label Biases for In-context Learning*](https://arxiv.org/abs/2305.19148): defines
  *domain-label bias* and estimates it from random in-domain words. Its point is that bias depends on the domain.
- [Zhou et al. 2024, *Batch Calibration*](https://arxiv.org/abs/2309.17249): a label-free bias estimate taken from
  the batch of test inputs itself.
- [Zheng et al. 2024, *LLMs Are Not Robust Multiple Choice Selectors*](https://arxiv.org/abs/2309.03882):
  option-ID selection bias, with label-free PriDe debiasing.

## Method

- **Fit:** 192 held-out clean items from the distillation pilot (repo generators, never public; 62 yes/no). Cached
  Keel trust-mode one-pass probabilities from stock Qwen3.5-4B and the distilled student. Two corrections per model:
  - a yes/no log-odds offset that maximises balanced accuracy;
  - a per-letter-position mean log-probability, subtracted at read time for choice items.
- **Evaluate once:** the frozen corrections are applied to the public-231 clean items (EVAL-ONLY: 74 yes/no,
  139 choice) from the Card score caches.
- **Uncertainty:** family-cluster bootstrap, with a paired bootstrap for the change.
- **Ceiling:** a *descriptive* ceiling picks the best single offset **using public labels**. It is in-sample and
  optimistic, and it is not a method.

Run: `python -X utf8 direction/experiments/transfer_calibration/run.py` (about 1 minute). Aggregates are in
[`results.json`](results.json).

## Results

| Model | Yes rate predicted, synthetic (gold 63%) | Yes rate predicted, public (gold 47%) | Offset fitted on synthetic | Public yes/no accuracy, before → after | Change (95% CI) | Descriptive ceiling (public labels) |
|---|---|---|---|---|---|---|
| stock 4B | 24% | 47% | +0.06 (≈ none) | 70.3% → 70.3% | 0.0 | 77.0% at t = +0.62 |
| student | 34% | 58% | −2.50 (toward yes) | 75.7% → 70.3% | −5.4 pp (−15.9, +2.6) | 81.1% at t = +0.62 |

Choice-position correction on public: stock 75.5% → 75.5% (0.0 pp, −4.1 to +4.2); student 81.3% → 82.0%
(+0.7 pp, −2.1 to +4.5).

## What it means

- **The bias depends on the distribution, as Fei et al. would predict.** On our synthetic generators both models
  *under*-say yes; on public items the student over-says it.
  - The correction fitted on synthetic points the wrong way: for the student it adds yes and costs 5.4 points.
  - In-distribution, the right correction exists and points toward "no". The descriptive ceiling suggests about
    5–7 points of yes/no accuracy are available to a well-placed offset.
- **The stock model's yes prior depends on the prompt.** It is 59% under the H2O shim prompt (gap analysis) and 47%
  under Keel's trust prompt, which equals the gold rate. The student's yes rate stays at 58–59% under both prompts.
- **The student's yes shift is not inherited from its labels.** Its screened yes/no training labels lean toward
  "no": 40% yes on clean items, 25–29% on attacked variants and 47% on policy variants. Yet it says yes about
  10 points more often than stock on both distributions. The cause is open; the soft-label loss and the attack
  variants are candidates.
- **Not shown:**
  - no model was rescored;
  - there are 74 public yes/no items, so the intervals are wide;
  - the ceiling is in-sample and optimistic;
  - thinking modes were not tested.

## Verdict and next step

Cheap readout calibration from our synthetic data is **not** a free capability gain; drop it as a standalone
proposition. Any correction needs calibration data that matches the deployment distribution. Two admissible routes:

1. **Label-free in-distribution estimates.** Content-free or domain-word probes ([Zhao](https://arxiv.org/abs/2102.09690),
   [Fei](https://arxiv.org/abs/2305.19148)) need about 300 one-pass calls: roughly $0.10–0.30 of GPU, owner approval
   required.
   - Batch calibration over the evaluation inputs is free but transductive.
   - Whether it is admissible under the EVAL-ONLY rule is an owner decision.
2. **A labelled non-public shadow set built from public-like families.** This is a data-authoring task, and the same
   asset the certification work needs.

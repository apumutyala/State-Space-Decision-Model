# Order-debias check on the stock RWKV7-13.3B parent (job-uhxb5): negative result

2026-10-07, Track D. Local Windows CPU, $0. No model calls. It only re-scores the outputs the job already exported.

**Inputs** (untrusted, read as JSON with `python -X utf8 -I`):
- `out/parent_arm/units-index.json` from job-uhxb5, sha256 `f450d000…7001`
- the 2,008-row DEV subset rows, sha256 `406d40e6…28f2`

**Scoring:** `harness.score.decompose` and `harness.bootstrap.compare` (paired cluster bootstrap, B = 2000, seed 0), plus `harness.calibration.report`, all run with the pinned kit `87d4650b`, edition 0.2.1.

**Command:** `python -X utf8 -I analysis.py <repo> <parent_arm> <index-rows.jsonl.gz> analysis.json`

## Question

Can averaging over option orders at inference time, or a cheaper calibration-only fix, recover Index points lost to the parent's apparent first-position bias?

## Answers

1. **Reversed-only and order-averaged Index can't be computed from these outputs.** The reversed order exists only in the `order_diag` pass, which covers 160 rows and 482 gold choice questions. It stores the choice only, with no per-option probabilities. Both of its conditions are relabelled (`NN-<id>`), so neither is the scored forward pass.

2. **The "first-position bias" is mostly a label prior, not a position effect.** In the scored run, `decide` presents options sorted by id (`decide/schema.py:86-97`). For ACOS, "no" sorts first and is also the majority gold, so position and label are confounded. Over all questions, chosen-first is 0.691 and gold-first is 0.669. In the diagnostic, the position effect (the mean of P(chose the option shown first) over both orders, minus 1/k) is:

   | k | n | acc fwd | acc rev | both | either | agreement | position effect |
   |---|---|---|---|---|---|---|---|
   | 2 | 387 | 0.848 | 0.879 | 0.791 | 0.935 | 0.853 | **-0.009** |
   | 3 | 17 | 0.412 | 0.471 | 0.353 | 0.529 | 0.765 | -0.010 |
   | 4 | 38 | 0.526 | 0.395 | 0.342 | 0.579 | 0.632 | -0.092 |
   | 5-16 | 16 | 0.188 | 0.188 | 0.063 | 0.313 | 0.313 | +0.148 |
   | >16 | 24 | 0.167 | 0.125 | 0.042 | 0.250 | 0.125 | **+0.509** |
   | all | 482 | 0.751 | 0.766 | 0.678 | 0.838 | 0.778 | +0.015 |

   Real primacy shows up only at k > 16 (BANKING77, CLINC, POP909, API-Bank), where accuracy is already near the floor. Reversed minus forward accuracy is +0.015 [-0.020, +0.054] (row-cluster bootstrap, 160 clusters).

   For k = 2, the accuracy of averaging the two orders' probabilities is bounded exactly by [both, either] = [0.791, 0.935]. Forward alone is 0.848.

3. **The label-free per-position prior correction hurts.** It is contextual-calibration style, p'(o) ∝ p(o) / prior(k, position), with k ≤ 10. The prior is the mean forward probability, cross-fitted so that no row is scored with a prior fitted on itself:

   | variant | Index | Δ vs forward [95% CI] | board-protocol ECE | debiased ECE |
   |---|---|---|---|---|
   | forward (reproduces job-uhxb5) | 24.85 [22.35, 27.71] | — | 0.082 | 0.075 |
   | 5-fold by group (hash) | 21.75 | **-3.10 [-5.46, -0.16]** | 0.107 | 0.106 |
   | leave-one-benchmark-out | 22.70 | -2.14 [-4.57, +0.85] | 0.111 | 0.094 |

   By area (5-fold), relative to forward:

   | area | Δ [95% CI] |
   |---|---|
   | tools | -7.76 [-10.13, -5.36] |
   | language | -6.48 [-13.35, +0.55] |
   | retrieval | +2.65 [-0.16, +5.57] |
   | knowledge | -1.47 |
   | arts | -1.56 |

   The correction strips a prior that is real: the fitted k = 2 prior is [0.70, 0.30], and the gold is mostly first. This is a scoring diagnostic on DEV inputs. It is not a servable configuration: serving it would need a prior fitted on train-split data or content-free inputs.

## Verdict

There is no measured gain, so no ORDER_DEBIAS proposal or scoring flag was added. The cheapest experiment that would settle the probability-averaged estimate is one reversed-relabelled pass over the 2,008-row subset, saving per-option probabilities. That is about 25.5 H100-min, or about $1.26 at the $0.0495/min batch rate (job-uhxb5 index stage: 1,531 s). It was not run.

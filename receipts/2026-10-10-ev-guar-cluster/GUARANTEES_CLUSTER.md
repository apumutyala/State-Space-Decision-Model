# Cluster-level Learn-then-Test on full DEV: D19, D13, Clef-flash, Clef 27B

Pre-registered as **B-PREREG-GUAR-CLUSTER** (state @68267953, pushed before any cluster-level number was computed). This resolves discrepancy D1 in `docs/guarantees/CONFORMAL.md`. Inputs, gold, splits, α, ε = 0.05, δ = 0.10, grid, Bonferroni, area groups, bootstrap and verdict rules are identical to B-PREREG-GUAR-FULLDEV (claude/b-guarantees-fulldev @62028486) and B-PREREG-GUAR-AREA (claude/ev-guar-area @201156a6). Code: `jevkit/guarantees.py` on this branch (`split_calibration(cluster_of=)`, `ltt_select(cluster_of=)`, `cluster_pick`). The default behaviour is unchanged and tested. Driver: `guarantees_cluster.py`. Numbers: `GUARANTEES_CLUSTER.json`. Diagnostic: `diag_cluster.py` → `diag_cluster.json`.

## Takeaway

- **Under the pre-registered cluster-level test, D19 and D13 certify nothing. They must abstain on every decision, both marginally and per area, at both α.** Clef-flash and Clef 27B still certify, at λ ≈ 0.93–0.96. Every "D19 more actionable than Clef-flash" verdict in the published guarantee runs is withdrawn.
- TEST act fraction at α = 0.10, primary configuration (CAL/TEST by run_id, cluster = row):

  | Model | Marginal: published → **cluster** | Per-area: published → **cluster** |
  |---|---|---|
  | D19 | 75.1% → **0.0%** | 60.1% → **0.0%** |
  | D13 | 36.1% → **0.0%** | 40.3% → **0.0%** |
  | Clef-flash | 72.2% → **51.6%** (λ 0.93) | 67.3% → **39.9%** (language λ 0.95, tools λ 0.92) |
  | Clef 27B | 81.1% → **59.3%** (λ 0.94) | 61.6% → **7.9%** (tools λ 0.96 only) |

- **Why D19 fails: two measured causes** (`diag_cluster.json`, primary, α = 0.10):
  1. **Its confident decisions sit in a few large rows.** At λ = 0.95, 4,321 LTT decisions act, with 0.7% error, but they come from only 307 rows. One decision per row leaves 307 trials with 7 errors (2.3%). Then p = 0.013, against the Bonferroni level 0.00099.
  2. **D19 never acts at λ ≥ 0.97.** Its p_max tops out near 0.96, so the high-confidence end of the grid, where Clef certifies (Clef 27B: 318 rows at λ 0.99 with 1 error), is empty. Below 0.95, the one-per-row error rate is already 4.7% at λ 0.93 and 9.6% at 0.86.
- **The certified quantity changed, and that matters as much as the power loss.** One decision per cluster certifies the *cluster-weighted* acted risk. Single-decision rows (5,017 of 17,485 TEST decisions; 63% of them from arts, knowledge and retrieval, which includes all 1,196 arts decisions) are much harder: D19's acted error at λ = 0.86 is 11.5% there, against 1.7% in rows with 8 or more decisions. Measured on TEST at the published λ, the cluster-weighted acted error is **10.6% for D19, 5.2% D13, 8.8% Clef-flash and 8.1% Clef 27B**. All are above ε, while the decision-weighted rates are 3.7–4.3%. **So the published certificates hold only for the decision-weighted risk, and even that relied on an anti-conservative test** (see the decomposition).
- **Decomposition: the cluster split alone changes little; the cluster test does the damage.** With the split by row but the old per-decision binomial, D19's marginal act fraction is 72.4%, D19 − Clef-flash is +0.2 points [−2.7, +2.7] (no separation, down from +2.9 [+0.2, +5.1]), and per-area D19 − Clef-flash is −1.7 [−3.2, −0.5].
- **What survives for D19.** Nothing certified at the cluster level. The decision-weighted, per-decision-test numbers remain as published, but the test behind them treats correlated decisions as independent (shown anti-conservative in `test_per_decision_ltt_is_anticonservative_on_clusters_and_cluster_ltt_is_valid`). They must not be called a guarantee.

## Before → after (TEST act fraction; acted-on error decision-weighted / cluster-weighted)

PRIMARY (CAL/TEST by run_id, cluster = row; 2,998 LTT rows from 8,344 LTT decisions).

| Model | α | Marginal: published | split only | **cluster** | Per-area: published | split only | **cluster** |
|---|---|---|---|---|---|---|---|
| D19 | 0.10 | 0.751 (4.3/10.6%) | 0.724 | **0.000** | 0.601 (1.3/5.7%) | 0.605 | **0.000** |
| D13 | 0.10 | 0.361 (3.7/5.2%) | 0.282 | **0.000** | 0.403 (3.1/3.5%) | 0.338 | **0.000** |
| Clef-flash | 0.10 | 0.722 (3.7/8.8%) | 0.722 | **0.516** (1.2/3.1%) | 0.673 (2.9/8.6%) | 0.622 | **0.399** (0.3/1.4%) |
| Clef 27B | 0.10 | 0.811 (3.8/8.1%) | 0.817 | **0.593** (1.5/3.4%) | 0.616 (1.7/5.8%) | 0.622 | **0.079** (0.2/0.6%) |
| D19 | 0.05 | 0.728 | 0.720 | **0.000** | 0.645 | 0.646 | **0.000** |
| D13 | 0.05 | 0.059 | 0.135 | **0.000** | 0.386 | 0.316 | **0.000** |
| Clef-flash | 0.05 | 0.659 | 0.660 | **0.516** | 0.674 | 0.633 | **0.399** |
| Clef 27B | 0.05 | 0.765 | 0.754 | **0.593** | 0.658 | 0.662 | **0.079** |

At the certified λ (≥ 0.92), every singleton set already acts regardless of α, so the cluster rows are identical across α.

CASE VARIANT (CAL/TEST by harness case, cluster = case; 2,113 LTT cases from 9,267 LTT decisions).

| Model | α | Marginal: published → split only → **cluster** | Per-area: published → split only → **cluster** |
|---|---|---|---|
| D19 | 0.10 | 0.747 → 0.748 → **0.000** | 0.608 → 0.622 → **0.078** (tools λ 0.94) |
| D13 | 0.10 | 0.275 → 0.357 → **0.000** | 0.296 → 0.427 → **0.000** |
| Clef-flash | 0.10 | 0.712 → 0.747 → **0.000** | 0.623 → 0.637 → **0.094** (tools λ 0.87) |
| Clef 27B | 0.10 | 0.817 → 0.836 → **0.519** (λ 0.96) | 0.627 → 0.640 → **0.432** (language 0.96, tools 0.95) |
| D19 / D13 / Clef-flash / Clef 27B | 0.05 | marginal cluster 0 / 0 / 0 / 0.519 | per-area cluster 0.078 / 0 / 0.094 / 0.432 |

The published-variant numbers were recomputed in this run and match both parent receipts exactly in all 16 (configuration × model × α) cells (`reproduction` in the JSON).

## Paired contrasts: Δ TEST act fraction (case-cluster bootstrap, B = 2000, seed 0; q_hat and λ fixed)

| Pair | α | Config | Analysis | Published | Split only | **Cluster (primary)** |
|---|---|---|---|---|---|---|
| D19 − Clef-flash | 0.10 | row | marginal | +0.029 [+0.002, +0.051] D19 | +0.002 [−0.027, +0.027] none | **−0.516 [−0.535, −0.493] Clef-flash** |
| D19 − Clef-flash | 0.10 | row | per-area | −0.072 [−0.096, −0.053] Clef-flash | −0.017 [−0.032, −0.005] Clef-flash | **−0.399 [−0.425, −0.365] Clef-flash** |
| D19 − Clef-flash | 0.05 | row | marginal | +0.069 [+0.035, +0.097] D19 | +0.060 [+0.025, +0.090] D19 | **−0.516 [−0.535, −0.493] Clef-flash** |
| D19 − Clef-flash | 0.05 | row | per-area | −0.029 [−0.044, −0.016] Clef-flash | +0.013 [+0.006, +0.020] D19 | **−0.399 [−0.425, −0.365] Clef-flash** |
| D19 − Clef 27B | 0.10 | row | marginal | −0.060 [−0.086, −0.039] Clef 27B | −0.093 [−0.122, −0.068] Clef 27B | **−0.593 [−0.616, −0.566] Clef 27B** |
| D19 − Clef 27B | 0.10 | row | per-area | −0.015 [−0.032, −0.001] Clef 27B | −0.017 [−0.033, −0.003] Clef 27B | **−0.079 [−0.095, −0.067] Clef 27B** |
| D19 − Clef 27B | 0.05 | row | marginal | −0.037 [−0.068, −0.012] Clef 27B | −0.034 [−0.067, −0.007] Clef 27B | **−0.593 [−0.616, −0.566] Clef 27B** |
| D19 − Clef 27B | 0.05 | row | per-area | −0.013 [−0.018, −0.008] Clef 27B | −0.016 [−0.021, −0.011] Clef 27B | **−0.079 [−0.095, −0.067] Clef 27B** |
| D19 − Clef-flash | 0.10 | case | marginal | +0.035 [−0.001, +0.064] none | +0.001 [−0.033, +0.029] none | 0.000 (both abstain) none |
| D19 − Clef-flash | 0.10 | case | per-area | −0.015 [−0.032, −0.000] Clef-flash | −0.015 [−0.027, −0.004] Clef-flash | −0.016 [−0.021, −0.013] Clef-flash |
| D19 − Clef-flash | 0.05 | case | marginal | +0.068 [+0.023, +0.102] D19 | +0.055 [+0.007, +0.093] D19 | 0.000 (both abstain) none |
| D19 − Clef 27B | 0.10 | case | marginal | −0.070 [−0.104, −0.045] Clef 27B | −0.087 [−0.121, −0.061] Clef 27B | −0.519 [−0.548, −0.484] Clef 27B |
| D19 − Clef 27B | 0.10 | case | per-area | −0.019 [−0.038, −0.003] Clef 27B | −0.018 [−0.033, −0.006] Clef 27B | −0.354 [−0.405, −0.283] Clef 27B |

D13 pairs, both α and both configurations, are in the JSON (`bootstrap.pairs`). D13 abstains everywhere under the cluster test, except that it ties with D19 where both abstain.

**Verdict changes vs the published runs.**
- D19 − Clef-flash, marginal: "D19 more actionable" (α 0.10 and 0.05, row split) becomes **"Clef-flash more actionable"**.
- Case split: the α 0.05 "D19 more actionable" becomes "no separation", because both abstain.
- No primary verdict now favours D19 over either Clef model.

## What is guaranteed, under which assumptions

- **Cluster LTT certificate** (Clef-flash and Clef 27B only): with probability ≥ 1 − δ over the LTT half, P(error | act) ≤ ε for a decision drawn as a uniformly random row (or case), then a uniformly random decision in it.
  - This assumes rows (cases) are i.i.d. between the LTT half and deployment, and it is conditional on the FIT half.
  - Per-area: simultaneously in every certified area (Bonferroni over 5 × 101).
- **Not guaranteed:**
  - The decision-weighted acted risk, because cluster size correlates with error here.
  - Distribution shift.
  - The 14,725 no-gold decisions.
- DEV only (0.2.1), not a board claim, $0 CPU.

## What would change it (not run; each needs its own pre-registration)

- **A cluster-robust test of the decision-weighted risk.** Use H0: E[E_c − ε·A_c] > 0 over clusters, where E_c and A_c are a cluster's acted errors and acts, with a bounded-difference (Hoeffding/Bentkus or betting) test on the per-cluster sums. That would certify the quantity the published headline describes, without the independence error. Expect power between "split only" and "cluster".
- **Model side.** D19's p_max ceiling near 0.96 and its high error on single-decision rows are what the cluster-level test punishes. Calibration that spreads confidence above 0.97 on easy rows, and better accuracy on the hard single-question benchmarks, are the levers. This is lead's call.

## Receipts

- `guarantees_cluster.py`: the driver. Run it from `ssdm-wt/m0-harness/.venv`:
  `python -X utf8 -I guarantees_cluster.py <this branch checkout> <ev-guar-area>/receipts/2026-10-10-B-guarantees-fulldev <ev-guar-area>/receipts/2026-10-10-ev-guar-area GUARANTEES_CLUSTER.json`
  It takes about 2.5 minutes on CPU, and the input shas are asserted.
- `GUARANTEES_CLUSTER.json`: every variant's LTT record (λ, cluster and decision counts, picked-id sha, UCB), q_hat, area plans, TEST summaries, all bootstrap CIs, and the reproduction check. `code_sha256` pins jevkit/guarantees.py, guarantees_fulldev.py and this driver.
- `diag_cluster.py` → `diag_cluster.json`: the one-per-row vs all-decision LTT grid and the TEST acted error by row size.
- `run.log`, `SHA256SUMS`.

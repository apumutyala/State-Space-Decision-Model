# Cluster-robust ratio LTT (CRR) on full DEV: D19, D13, Clef-flash, Clef 27B

Pre-registered as **B-PREREG-GUAR-CRR** (state @033906aa, pushed before any CRR statistic was computed). Everything except the test is identical to B-PREREG-GUAR-CLUSTER (`receipts/2026-10-10-ev-guar-cluster`): inputs and shas, splits, α, ε = 0.05, δ = 0.10, the 101-point grid with Bonferroni, area groups and bootstrap.

- **Code.** `jevkit/guarantees.py` (`ltt_select(cluster_test="crr", crr_cap=c, cluster_size=…)` and `crr_betting_pvalue`). The method is opt-in, and the default path is byte-identical (tested).
- **Driver.** `guarantees_crr.py`.
- **Numbers.** `GUARANTEES_CRR.json`. `tables.txt` comes from `tables_crr.py`.
- **Post-hoc diagnostic.** `diag_crr.py` writes `diag_crr.json`. It is not verdict-bearing.

## Takeaway

- **Under the pre-registered primary test (cap c = 8), no model certifies anything: D19, D13, Clef-flash and Clef 27B all abstain everywhere.** This holds marginally and per area, at both α, in both configurations. Every CRR paired contrast is 0.000 ("no separation", all abstain).
- **This is NOT the less conservative test we hoped for.** At c = 8 it is strictly weaker than the one-per-row test, so the Clef certificates from that test (51.6% / 59.3% marginal) do not carry over to R_8. The uncapped decision-weighted secondary also certifies nothing.
- **Why: two measured causes.**
  1. **The estimand really is above ε at the λ values the models were published at.** Measured on TEST at the published λ, R_8 (the c = 8 weighted acted error) is 8.7% for D19, 5.1% for D13, 6.3% for Clef-flash and 6.5% for Clef 27B. All four are above ε = 0.05. The decision-weighted rate (3.7–4.3%) hides this, because the large rows are the easy ones.
  2. **At the high-λ end, where R_8 is ≤ 2%, the betting test lacks power at the Bonferroni level** (`diag_crr.json`, run_id, α = 0.10, marginal).
     - The test needs ln(1/level) = 6.92 nats.
     - **Clef 27B** reaches a log max capital of 6.48 at λ 0.95, where the LTT R_8 is 2.0%.
     - **Clef-flash** reaches 6.55 at λ 0.94, where R_8 is 1.5%.
     - **D19** peaks at 4.29 at λ 0.95, where R_8 is 1.4% over 383 acting rows. D19 never acts at λ ≥ 0.97.
     - The cap multiplies the payoff range by 8. A singleton correct row moves Y by only −ε/8. The hedged bet cap (0.75 of the nonnegativity limit) binds.
- **What survives, unchanged.** The one-per-row certificates for Clef-flash and Clef 27B stand: they certify the cluster-weighted risk R_1 (GUARANTEES_CLUSTER.md). The c = 1 CRR secondary reproduces them slightly more conservatively, which is what a betting bound should do against an exact binomial: marginal Clef-flash 49.0% vs 51.6%, Clef 27B 55.9% vs 59.3%. This cross-checks the implementation on real data.
- **No D19 guarantee exists under any of the three cluster-valid tests.** These are one-per-row, CRR c ∈ {1, 8} and CRR uncapped.

## Headline table: TEST act fraction, α = 0.10

PRIMARY configuration (CAL/TEST by run_id, cluster = row; 2,998 LTT rows, 8,344 LTT decisions; 459 DEV rows exceed 8 decisions).

| Model | Marginal: per-decision (published) | one-per-row | **CRR c = 8** | Per-area: per-decision (published) | one-per-row | **CRR c = 8** |
|---|---|---|---|---|---|---|
| D19 | 0.751 | 0.000 | **0.000** | 0.601 | 0.000 | **0.000** |
| D13 | 0.361 | 0.000 | **0.000** | 0.403 | 0.000 | **0.000** |
| Clef-flash | 0.722 | 0.516 (λ 0.93) | **0.000** | 0.673 | 0.399 | **0.000** |
| Clef 27B | 0.811 | 0.593 (λ 0.94) | **0.000** | 0.616 | 0.079 | **0.000** |

The secondaries are reported but never verdict-bearing. CRR c = 1 is marginal 0 / 0 / 0.490 / 0.559 and per-area 0 / 0 / 0.079 / 0.074. CRR uncapped (c = 64) is 0 everywhere. At α = 0.05 the published column is 0.728 / 0.059 / 0.659 / 0.765 (marginal) and 0.645 / 0.386 / 0.674 / 0.658 (per area). The cluster and CRR columns are identical to α = 0.10.

CASE VARIANT (CAL/TEST by case, cluster = case; 2,113 LTT cases, 9,267 LTT decisions; 330 DEV cases exceed 8; s_max = 363):

| Model | Marginal: published / one-per-row / **CRR8** / c=1 / uncapped | Per-area: published / one-per-row / **CRR8** / c=1 / uncapped |
|---|---|---|
| D19 | 0.747 / 0.000 / **0.000** / 0.000 / 0.000 | 0.608 / 0.078 / **0.000** / 0.078 / 0.000 |
| D13 | 0.275 / 0.000 / **0.000** / 0.000 / 0.000 | 0.296 / 0.000 / **0.000** / 0.000 / 0.000 |
| Clef-flash | 0.712 / 0.000 / **0.000** / 0.491 / 0.000 | 0.623 / 0.094 / **0.000** / 0.087 / 0.000 |
| Clef 27B | 0.817 / 0.519 / **0.000** / 0.519 / 0.000 | 0.627 / 0.432 / **0.000** / 0.109 / 0.000 |

## TEST acted-on error at each test's λ: decision / cluster / c = 8 weighted

Primary configuration, α = 0.10, marginal. Per-area values are in the JSON.

| Model | Published λ | One-per-row λ | CRR c = 1 λ |
|---|---|---|---|
| D19 | 4.3 / 10.6 / **8.7%** | abstains | abstains |
| D13 | 3.7 / 5.2 / **5.1%** | abstains | abstains |
| Clef-flash | 3.7 / 8.8 / **6.3%** | 1.2 / 3.1 / 2.3% | 1.1 / 2.8 / 2.1% |
| Clef 27B | 3.8 / 8.1 / **6.5%** | 1.5 / 3.4 / 2.8% | 1.3 / 3.0 / 2.4% |

## Paired contrasts: Δ TEST act fraction (case-cluster bootstrap, B = 2000, seed 0)

| Pair | α | Config | Analysis | Per-decision (published) | One-per-row | **CRR c = 8** |
|---|---|---|---|---|---|---|
| D19 − Clef-flash | 0.10 | row | marginal | +0.029 [+0.002, +0.051] D19 | −0.516 [−0.535, −0.493] Clef-flash | **0.000 none (all abstain)** |
| D19 − Clef-flash | 0.10 | row | per-area | −0.072 [−0.096, −0.053] Clef-flash | −0.399 [−0.425, −0.365] Clef-flash | **0.000 none** |
| D19 − Clef 27B | 0.10 | row | marginal | −0.060 [−0.086, −0.039] Clef 27B | −0.593 [−0.616, −0.566] Clef 27B | **0.000 none** |
| D19 − Clef 27B | 0.10 | row | per-area | −0.015 [−0.032, −0.001] Clef 27B | −0.079 [−0.095, −0.067] Clef 27B | **0.000 none** |
| Clef 27B − Clef-flash | 0.10 | row | marginal | +0.088 [+0.080, +0.097] Clef 27B | +0.076 [+0.063, +0.089] Clef 27B | **0.000 none** |

All six pairs × both α × both configurations × both analyses are in `tables.txt` and in the JSON (`bootstrap.pairs`), as is the within-model crr8 − one-per-row contrast. Every CRR-c8 cell is 0.000 [0, 0].

**Verdict changes vs one-per-row.** Every "Clef-flash more actionable" and "Clef 27B more actionable" verdict over D19 or D13 becomes "no separation", because all models abstain. This does not favour D19: D19 certifies nothing under any cluster-valid test. The change only reflects that R_8 is harder to certify than R_1.

## Validity simulation (unit tests, `tests/guarantees/test_crr.py`)

- **Setup.** Clusters share one fate: every decision in a cluster is wrong together, which is maximal intra-row correlation. Cluster size is 1 or 16, half each. The true R(λ) = 0.2(1 − λ), so R ≤ ε iff λ ≥ 0.75. The grid is 0.95…0.55 (41 points, Bonferroni). Each seed has 4,000 i.i.d. clusters, and there are 120 seeds.
- **CRR at c = 8 is valid and non-vacuous.** It certified in 107/120 seeds, and violated in 0/120. Each violation would be a seed where the true R(λ*) > ε.
- **The per-decision binomial is anti-conservative.** On the same data it certified in 120/120 seeds and violated in 16/120 (13.3%), above δ = 10%.
- **The p-value is valid at the null boundary.** At E[Y] = 0 with skewed two-point Y, the rejection rate is ≤ a + 3 SE at a = 0.05 and a = 0.2.
- **The fast path matches `ltt_select`.** The simulation's fast computation agrees exactly with `ltt_select(cluster_test="crr")`.
- The default `ltt_select` output is byte-identical with and without the new keyword arguments. 33 guarantee tests pass.

## What is guaranteed, under which assumptions

- **CRR certificate.** With probability ≥ 1 − δ over the LTT half, conditional on the FIT half: R_c(λ*) = E[w_r L_r] / E[w_r n_r] ≤ ε, with w_r = min(1, c/s_r) and s_r the cluster's DEV gold-decision count. This holds under any within-cluster dependence, with clusters i.i.d. On this data the primary test certified nothing, so there is no c = 8 claim for any model.
- **Not guaranteed.** The decision-weighted R_∞ (the uncapped test certified nothing), distribution shift, and the no-gold decisions. DEV only (0.2.1), not a board claim, $0 CPU.

## What would change it (post-hoc; each needs its own prereg on data not yet looked at)

- **Fixed-sequence instead of Bonferroni.** Testing descending λ at the full δ would need ln(10) = 2.30 nats instead of 6.92. In `diag_crr.json`, three models' top acting λ values exceed 2.30: D19 at λ 0.96–0.93, and Clef-flash and Clef 27B from 0.99 down. D13 does not (0.29 nats at its top λ, 0.96). **We looked at this data, so it must not be re-tested here.** Register it for full_excl51 or for the 0.3 public split.
- **A better bet schedule.** The hindsight-optimal constant bet at c = 8 reaches 8.5 nats for Clef 27B and 8.6 for Clef-flash, against 6.5 achieved. For D19 it reaches only 5.6, so no constant-bet betting test at Bonferroni level could certify D19 at c = 8.
- **Model side.** D19's p_max ceiling near 0.96 and its error on single-decision rows remain the lever. This is lead's call.

## Receipts

- `guarantees_crr.py`: the driver. Run it from `ssdm-wt/m0-harness/.venv`; it took about 1.5 minutes on CPU. It asserts the input shas, and asserts exact reproduction of the published (both parent receipts) and one-per-row (GUARANTEES_CLUSTER.json) TEST act fractions in all 16 cells:
  `python -X utf8 -I guarantees_crr.py <this checkout> <ev-guar-area>/receipts/2026-10-10-B-guarantees-fulldev <ev-guar-area>/receipts/2026-10-10-ev-guar-area <this checkout>/receipts/2026-10-10-ev-guar-cluster/GUARANTEES_CLUSTER.json GUARANTEES_CRR.json`
- `GUARANTEES_CRR.json`: every variant's LTT record (λ, p-value, log capital, clusters, decisions, cap), TEST summaries, bootstrap CIs and the reproduction check. `code_sha256` pins the code.
- `diag_crr.py` → `diag_crr.json`, `tables_crr.py` → `tables.txt`, `run.log`, `SHA256SUMS`.

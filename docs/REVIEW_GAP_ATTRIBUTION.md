# Review: GAP_ATTRIBUTION_COMPOSITE (lead, claude/lead-gap-attribution @2d2f346f)

Evidence lane, adversarial review, 2026-10-11. Branch `claude/ev-review-gap-attr`, cut from 2d2f346f. CPU only. No frozen file edited.
Receipts: `receipts/2026-10-11-ev-review-gap-attr/` (scripts, JSON and stdout of every number below). Same caveat as the
reviewed doc: these are open public-231 numbers only, not on the board's scale.

## Verdicts at a glance
| # | lead claim | verdict |
|---|---|---|
| a | The script reproduces SC (@e2fcd82b) and J1 (@a8ca3e7e) CIs exactly | **CONFIRMED** |
| b | Intelligence is the largest share of the gap (Shapley over 4 axes) | **CONFIRMED**. The ranking does not depend on the method. One sub-claim is wrong: "Calibration comes second" is false for D19 |
| c | I = 100 with C fixed leaves D19/JEV1 below 76.6, so I AND C must rise | Numbers **CONFIRMED**. "I and C together close the gap" is **WRONG for JEV1-4B r2 and D19** (it holds only for stock 4B+shim) |
| d | noul carries −12.2/−16.2/−13.4 of the −21.3/−26.4/−24.9 I gap | **CONFIRMED**. The split is exact. It needs a weighting caveat |
| e | Six families (80 items) hold 66–81% of the I gap | **OVERCLAIMED**. The points reproduce, but the CIs are wide, the six were chosen post hoc, and they shrink to 30–54% out of sample |
| f | (hygiene) | The doc lacks an adaptive-analysis caveat. Proposed wording is below |

## (a) Reproduction: CONFIRMED
- `python -I -X utf8 scripts/gap_attribution.py --config receipts/2026-10-11-gap-attribution/cfg.json --out <tmp>`: all 12 input preds exist and
  pass their sha checks. The I/C cross-check against the landed reports passes (≤ 1e-9). The output is **byte-identical** to the committed
  `gap_attribution.json` (sha256 311ebdab…; the rerun took 16 s).
- ΔComp A CIs against the source receipts (all digits shown match):

| pair | GA | source |
|---|---|---|
| stock 4B+shim − H2O | −9.3811 [−23.0507, −4.2057] | SC `pairs_vs_baseline.qwen3.5-4b+h2o_shim` [−23.050713, −4.205713] |
| D19 − H2O | −14.3866 [−31.6737, −10.6345] | SC `pairs_vs_baseline.D19` [−31.673726, −10.634475] |
| JEV1 native − H2O | −13.2102 [−37.8151, −7.5856] | J1 `rescore-xt3ax-full.json` [−37.815073, −7.585551] |
| JEV1 native − stock_shim | −4.0664 [−25.6355, +3.2209] | J1 [−25.635500, +3.220909] |

## (b) Decomposition method: CONFIRMED (robust), with one wrong sub-claim
The method is stated: the exact Shapley value over the 4 axes (the average over all 24 orderings, which sums to the gap). The doc also gives the
one-at-a-time "single swap". I recomputed both independently and tried three other legitimate methods (`axis_check.py`). At the point
estimates no floor binds (every I ≥ 50). A = 1 / mean(1/axis) is then exactly linear in reciprocal space, which gives a fourth, closed-form
attribution.

| candidate | gap | Shapley I / C / S / K | reciprocal-linear I / C / S / K | add-one (cand→H2O) I / C | remove-one (H2O→cand) I / C | I over 24 orders | C over 24 orders | P(I largest), bootstrap |
|---|---|---|---|---|---|---|---|---|
| stock 4B+shim | 9.38 | **6.28** / 3.14 / −0.04 / 0.00 | 6.29 / 3.14 / −0.04 / 0.00 | 6.01 / 2.87 | 6.55 / 3.41 | [6.01, 6.56] | [2.87, 3.42] | 0.979 |
| JEV1-4B r2 native | 13.21 | **8.06** / 4.23 / 0.95 / −0.03 | 8.08 / 4.22 / 0.95 / −0.03 | 7.47 / 3.70 | 8.66 / 4.78 | [7.47, 8.67] | [3.69, 4.79] | 0.983 |
| JEV1 native_shimcal | 11.80 | **5.91** / 4.95 / 0.97 / −0.03 | 5.92 / 4.95 / 0.97 / −0.03 | 5.43 / 4.48 | 6.41 / 5.44 | [5.42, 6.42] | [4.47, 5.44] | – |
| D19 | 14.39 | **7.26** / 1.83 / 2.12 / 3.18 | 7.28 / 1.82 / 2.11 / 3.17 | 6.54 / 1.51 | 8.03 / 2.18 | [6.54, 8.03] | [1.51, 2.18] | 0.984 |
| D19 cal | 15.37 | **7.02** / 3.13 / 2.09 / 3.13 | 7.04 / 3.12 / 2.08 / 3.12 | 6.20 / 2.60 | 7.90 / 3.71 | [6.20, 7.90] | [2.60, 3.71] | – |

- The Shapley values equal the lead's to every printed digit. The axis ranking is the same under all five methods, and I's range over the 24
  orderings never overlaps any other axis's. Under the paired bootstrap (jevkit draws, B 1000), I is the largest Shapley term in
  97.9–98.4% of resamples (`shap_rank.json`). I share CI: stock [52%, 87%], JEV1 [47%, 85%], D19 [33%, 75%]. The order dependence is small here
  because the gap is a modest perturbation of a smooth harmonic mean.
- **Wrong sub-claim** (Takeaway bullet 2): "Calibration comes second (3.1 / 4.2 / 1.8)". For D19 Calibration comes **last**:
  K 3.18 > S 2.12 > C 1.83. The doc's own table shows this, but the takeaway sentence does not. Fix the sentence to say "second for the 4B line;
  for D19, Cost and Speed come before Calibration".
- Minor: the S and K Shapley terms have no CI because those axes are held fixed in the bootstrap. Cross-job S noise is ~2.7 points (SC), so
  JEV1's S term of 0.95 is within run-to-run noise.

## (c) "I = 100 with C fixed is still below 76.6": numbers CONFIRMED, sufficiency claim WRONG for 2 of 3
| candidate | A(I=100, C fixed) | I needed, C fixed | A(I, C := H2O's) | A(I=100, C := H2O's) | A(I=C=100) | P(I, C := H2O's closes), bootstrap |
|---|---|---|---|---|---|---|
| H2O target | 76.64 | | | | | |
| stock 4B+shim | 77.37 | **95.30** | 76.69 (+0.05) | 81.19 | 84.61 | 1.00 |
| JEV1-4B r2 native | **74.74** | impossible | **75.55 (−1.09)** | 79.92 | 83.22 | **0.00** |
| D19 | **72.39** | impossible | **70.64 (−6.0)** | **74.45 (−2.2)** | 77.31 | **0.00** |

- 72.4 / 74.7 / 95.3 are correct, and "I alone does not close the gap" holds.
- "Intelligence and Calibration together do [close it]" (Takeaway bullet 3) and "Only I and C together close the 4B gaps (I+C column)" (§1) are
  **wrong** except for stock 4B+shim:
  - JEV1-4B r2 is a 4B candidate, and with I and C at H2O's values it still sits **1.09 below** H2O. The residual is the native readout's Speed
    (S 85.25 vs 91.31; Shapley S 0.95).
  - D19 stays 6.0 below. Even I = 100 with C at H2O's value leaves it 2.2 short. It needs S/K (5.3 points from the 13.3B price class and the
    0.50 s readout) or I and C both near 100.
  - Stock 4B+shim "closes" by only 0.05 points, and only because its S (91.60) is 0.29 above H2O's.
- Corrected statement: **I and C are both necessary, but they are sufficient only for a candidate whose S and K already match H2O's** (the
  shim readout on the 4B host). A native fork-head readout must also recover ~1 point of Speed, and the 13.3B line also needs Cost/Speed.

## (d) How I is aggregated, and whether the per-type and per-family splits sum: CONFIRMED, plus a weighting caveat
`jevkit/scoring.py` (frozen, read only): I_open = mean over the 3 types of max(0, CC_type). CC_type = Σ_tier w_tier·CC_tier / Σ_present w, with
w = easy 0.1, standard 0.2, judge 0.3, hard 0.4. CC_tier is chance-corrected: choice (acc−c̄)/(1−c̄); noul (acc−0.5)/0.5 with the 0.2–0.8 band
counted wrong; score 1 − mean nMAE / mean nMAE_chance. Public-231 is open-only, so there is no gap penalty. Every one of these is linear in the
per-item outcome when no type is clipped. So:
- **The per-type split is exact.** −6.70 − 12.19 − 2.40 = −21.29; −7.63 − 16.21 − 2.55 = −26.39; −5.32 − 13.41 − 6.19 = −24.92. "ΔCC_type / 3" is
  the right formula.
- **The per-family swap is exactly additive.** I wrote an independent per-item linear decomposition (`fam_boot.py: contrib`). It equals GA's
  `swap_dI` for every family to ≤ 1.4e-14, and it equals `jevkit.aggregate` on 69 checked bootstrap resamples per candidate to ≤ 2.8e-14, so no
  clipping occurs.
- **Weighting caveat, missing from the doc.** Each type weighs 1/3 whatever its n, and tiers are weighted within a type. The I-point value of one
  full miss differs ~60× across items:

| cell | n | I points per item miss |
|---|---|---|
| choice easy / standard / hard | 36 / 36 / 67 | 0.17 / 0.33 / 0.38 |
| noul easy / standard / hard | 12 / 24 / 38 | 0.79 / 0.79 / 1.00 |
| score standard / hard | 12 / 6 | 2.22 / **10.4** (per unit nMAE/nMAE_chance) |

  One noul flip is worth ~2.6 hard-choice flips, and the 6 hard score items carry 22% of all I weight. "noul is the biggest hole" is correct
  under this scorer. Part of it is the scorer's per-item weight, not only the item count. Per-type I-gap CIs (`type_boot.json`): noul
  stock [5.97, 19.38], D19 [6.39, 20.22], JEV1 [9.61, 22.81]. P(noul is the largest type) = 0.895 / 0.925 / 0.986. The score-type terms span 0 for all
  three (e.g. D19 [−1.69, 12.33]), so the score row is noise-dominated.

## (e) Family shares: OVERCLAIMED
Paired stratified bootstrap (jevkit strata, default_rng(16), B 1000), family I-gap points from the exact per-item decomposition (`fam_boot.json`).

| family (n) | stock 4B+shim: pts [CI] · share [CI] · P(top 6) | D19 | JEV1-4B r2 |
|---|---|---|---|
| probability (10) | 3.52 [0.76, 6.91] · 17% [4, 40] · 0.93 | 4.91 [1.52, 8.67] · 20% [7, 41] · 0.96 | 3.15 [0.75, 6.46] · 12% [3, 29] · 0.87 |
| temporal_numeric (15) | 4.05 [0.38, 9.81] · 19% [2, 47] · 0.88 | 2.67 [−0.38, 6.48] · 11% [−2, 27] · 0.67 | 4.72 [1.00, 9.50] · 18% [3, 39] · 0.94 |
| long_policy (19) | 1.92 [−5.93, 10.96] · 9% [−52, 38] · 0.56 | 2.64 [−2.85, 9.41] · 11% [−18, 32] · 0.60 | 4.57 [−4.92, 13.86] · 17% [−31, 42] · 0.76 |
| trap (8) | 0.89 [−0.22, 3.01] · 4% [−1, 18] · **0.29** | 2.81 [0.00, 7.19] · 11% [0, 26] · 0.63 | 3.59 [0.00, 8.64] · 14% [0, 31] · 0.76 |
| adequacy (12) | 1.59 [0.00, 3.97] · 7% [0, 23] · 0.55 | 2.38 [0.00, 5.56] · 10% [0, 23] · 0.67 | 2.38 [0.00, 5.56] · 9% [0, 22] · 0.75 |
| judge_hard (17) | 2.00 [−2.01, 6.02] · 9% [−9, 31] · 0.65 | 1.00 [−2.01, 5.01] · 4% [−11, 19] · 0.36 | 3.01 [0.00, 7.02] · 11% [0, 29] · 0.84 |
| *ordinal (12)* | −0.68 · 0.00 | **4.06 [3.25, 4.71] · 16% [11, 30] · 1.00** | 0.16 · 0.00 |
| *adversarial (6)* | **2.36 [0.00, 6.73] · 11% [0, 34] · 0.68** | 0.84 · 0.16 | 0.26 · 0.01 |
| *tradeoff (6)* | 0.62 · 0.22 | 1.38 · 0.32 | 1.76 [0.00, 4.15] · 0.54 |
| **the six, combined** | **65.6% [27.5, 86.0]** | **65.8% [39.4, 83.6]** | **81.2% [60.3, 94.4]** |

- **The points reproduce** (65.6 / 65.8 / 81.2%). Only probability (all three) and temporal_numeric (stock, JEV1) have CIs that exclude 0.
  long_policy, the family the doc's C1 cause leans on, spans 0 for every candidate. Its CI is the widest because it holds 2 hard score items worth
  ~10 I points each.
- **The six are a cross-candidate union picked post hoc, and per candidate they are the wrong six.** D19's second-largest and most certain
  loss, **ordinal** (4.06 [3.25, 4.71], in the top 6 in 100% of resamples), is not in the six. Stock's #3, adversarial, is not in it either. trap is
  stock's #9 (P(top 6) = 0.29).
- **Null reference.** The six families hold 35% of the items but **45% of the I weight**, because they are almost all hard-tier. So the
  concentration beyond the null is roughly 66% vs 45%, not 66% vs 35%.
- **Out-of-sample shrinkage** (`split_half.json`; 500 random stratified halves; pick the top 6 on half A, measure them on half B):

| candidate | top-6 share on the selection half | same families on the held-out half | top-6 overlap A vs B |
|---|---|---|---|
| stock 4B+shim | 90% [73, 135] | **30% [−11, 61]** | 2.1 / 6 |
| D19 | 91% [76, 125] | **45% [22, 70]** | 2.8 / 6 |
| JEV1-4B r2 | 89% [76, 122] | **54% [18, 83]** | 3.9 / 6 |

  Half-samples are noisier than the full 231, so this overstates the shrinkage somewhat. The direction is unambiguous, though: a "top-k families"
  share selected and measured on the same items is strongly optimistic. Supportable restatement: "noul and the probability /
  temporal_numeric families are reliably behind. Beyond that, family-level attribution is not resolved with 4–24 items per family."

## (f) Statistical hygiene: adaptive analysis on the public 231
The families were chosen by looking at our losses on the same 231 public items that will later measure any fix. Two consequences follow. A gain
on "the six" measured on those items is biased upward (the split-half test above shows the selection effect is large). And data shaped after
public items edges toward hard rule 3 (never train on JevBench eval items). Proposed text for the doc, plus mitigations:
1. **State it:** "The family ranking is exploratory and post hoc on the public-231. Any training aimed at these families is judged only by the
   pre-registered overall Composite A (EV-PREREG-COMPOSITE-LOCAL), never by a targeted-family subtotal. Family deltas are reported as
   descriptive."
2. **Pre-register before the run:** the primary metric is overall ΔComp A vs the in-job control, with a 95% CI. Targeted-family Δ is
   secondary and labelled "selected post hoc on these items".
3. **Confirm off the selection set:** first on the held-out jev slice (the jev2a ids 204e9bfe already used for T fitting must not be reused
   as the confirmation set if they informed the fit), and finally on the sealed split via a board submission. The submission needs operator
   approval.
4. **Data provenance:** family-targeted data is generated from family *descriptions* (H2O card, task taxonomy), never from public item
   text or templates. A leak audit (n-gram/embedding overlap against public-231) is attached to the data receipt before training.
5. **Report the null:** quote a family share next to its I-weight share (six families: 45%), not its item share (35%).

## Requested edits to GAP_ATTRIBUTION_COMPOSITE.md
1. Takeaway 2: "Calibration comes second for the 4B line. For D19, K (3.18) and S (2.12) come before C (1.83)."
2. Takeaway 3 / §1: replace "Intelligence and Calibration together do [close the gap]" / "Only I and C together close the 4B gaps" with the
   corrected statement in (c). JEV1 is short by 1.09 (Speed) and D19 by 6.0 (S + K).
3. §2: add the per-item weight table (or a one-line caveat) and the per-type CIs.
4. §2 "Concentration": add the bootstrap CIs, the 45% null, the per-candidate top 6 (ordinal for D19, adversarial for stock), and the
   split-half shrinkage. Downgrade the "six families" to "probability and temporal_numeric robustly, the rest unresolved".
5. Add the (f) caveat to §6.

## Reproduce (from the repo root of this branch, CPU, < 1 min total)
```
R=receipts/2026-10-11-ev-review-gap-attr
python -I -X utf8 scripts/gap_attribution.py --config receipts/2026-10-11-gap-attribution/cfg.json --out /tmp/ga.json   # byte-identical, sha in $R/ga_rerun.sha256
python -I -X utf8 $R/axis_check.py                                   # (b)(c) five attribution methods, closure counterfactuals
python -I -X utf8 $R/fam_boot.py receipts/2026-10-11-gap-attribution/cfg.json $R/fam_boot.json 1000   # (e) family CIs
python -I -X utf8 $R/type_boot.py $R/type_boot.json $R              # (d) item weights, per-type CIs
python -I -X utf8 $R/shap_rank.py $R $R/shap_rank.json              # (b) P(I largest), I share CI, P(I,C closes)
python -I -X utf8 $R/split_half.py $R $R/split_half.json            # (e) selection optimism
```
The inputs are the lead's sha-pinned preds (the paths are in cfg.json; local artifact and scratch paths on this laptop).

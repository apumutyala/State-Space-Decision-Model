# Gap attribution: JevBench v1.6.1 Composite vs H2O-Lightning-4B v1.1 (local, public-231)

Lead lane, 2026-10-11. Deliverable: "remaining gap attributed with receipts" for the PRIMARY objective (GOAL.md (1)).
Rule: EV-PREREG-COMPOSITE-LOCAL (state PREREG.md). **Caveat on every number: open public items only; the board's sealed
split may rank differently (open→board I gap measured on H2O: −17.6).** No local number here is comparable with the
board's 72.5.

## Takeaway first
- **No candidate of ours is within reach of H2O on the local Composite.** The closest ones are stock Qwen3.5-4B served with H2O's
  shim, at ΔComp A **−9.4 [−23.1, −4.2]**, then JEV1-4B r2 at −13.2 [−37.8, −7.6] and D19 at −14.4 [−31.7, −10.6]. All CIs
  exclude 0 against us.
- **Intelligence is the largest axis in every decomposition.** An exact Shapley split over the 4 axes gives Intelligence
  **6.3 of 9.4 points for stock 4B+shim (67%)**, 8.1 of 13.2 for JEV1-4B r2 (61%) and 7.3 of 14.4 for D19 (50%).
  Calibration comes second for the 4B line (3.1 / 4.2). **For D19, Cost (3.18) and Speed (2.12) both come before Calibration (1.83)**:
  the 13.3B price class and the 0.50 s readout cost 5.3 points. I is the largest Shapley term in 97.9–98.4% of bootstrap resamples
  (evidence review, claude/ev-review-gap-attr @f2a04059).
- **I and C are both necessary, but they are sufficient only when S and K already match H2O's.** Composite A is a harmonic mean, so
  with C held fixed even I = 100 leaves D19 (72.4) and JEV1-4B r2 (74.7) below H2O's 76.6. Stock 4B+shim would need I = 95.3.
  With I and C both set to H2O's values:
  - stock 4B+shim reaches 76.69 and clears by only +0.05, because its S is 0.29 above H2O's;
  - **JEV1-4B r2 reaches 75.55, still 1.09 short.** The shortfall is the native readout's Speed;
  - **D19 reaches 70.64, 6.0 short.** It needs S and K as well.

  Bootstrap probability that I, C := H2O's closes the gap: 1.00 / 0.00 / 0.00 (review §c). So the target is
  **Capability (I and C), which is −17.7 to −22.6 behind H2O, plus the shim-speed readout. D19 also needs a cheaper, faster host.**
- **Within Intelligence, noul (yes/no) is the biggest hole.** It accounts for −12.2 / −16.2 / −13.4 of the −21.3 / −26.4 / −24.9 I gap,
  against choice −6.7 / −7.6 / −5.3 and score −2.4 / −2.6 / −6.2. Part of this comes from the scorer's per-item weights: one noul miss costs
  0.79–1.00 I points, against 0.17–0.38 for a choice miss.
- **By family, only probability is robustly behind for all three candidates, and temporal_numeric for stock and JEV1.** Every other family
  is unresolved at 4–24 items per family. The earlier "six families hold 66–81%" claim is **retracted** (it was post hoc and optimistic; see §2).
- **H2O-host training (§5–5c).** JEV2-H2O and JEV2-H2O-L gain +4.44 and +8.97 pp in-distribution, but public Δ Comp A is −0.41 and −0.97
  (both CIs span 0; Holm claims neither). The remaining gap is attributed to the **training-data distribution** (headroom, conventions, shape), not to
  capacity or the readout.
- **Causes, all measured:** (a) a train/eval data shape mismatch (the jev1 audit); (b) noul indecision, where the 0.2–0.8 band counts as
  wrong; (c) a slow native readout (0.197 s vs 0.067 s); (d) 0.093 epoch of training. Post-hoc calibration on H2O itself buys
  at most +0.5 Composite (not significant), so **the remaining gap is answer quality, not calibration.**

## Sources (every number below cites one of these)
| tag | branch @ sha | path |
|---|---|---|
| SC | claude/ev-scorecard @e2fcd82b | receipts/2026-10-10-ev-scorecard/SCORECARD.{md,json}, reports/*.jevkit.json |
| J1 | claude/ev-j1-rescore @a8ca3e7e | receipts/2026-10-10-ev-rescore-xt3ax/RESCORE_XT3AX.md, reports/*.rescore.jevkit.json, length-xt3ax.json |
| AUD | claude/ev-jev1-audit @13c63fdc | JEV1_AUDIT.md, receipts/2026-10-11-jev1-audit/ |
| HE | claude/lead-h2o-eval @f5615683 | receipts/2026-10-11-h2o-host-eval/ (job-4g6m8; README.md, decision.json, arms/*/preds.jsonl) |
| HR | claude/ev-h2o-eval-rescore @536fae54 | receipts/2026-10-11-ev-h2o-eval-rescore/RESCORE_H2O_EVAL.md (AGREES on 235 numbers) |
| FL | claude/ev-flip-rate @4a65dd18 | receipts/2026-10-10-ev-flip-rate/FLIP_RATE.md |
| GU | claude/ev-guarantee-proofs @b4835e05 (cluster) / @01738528 (CRR) | receipts/2026-10-10-ev-guar-cluster/, receipts/2026-10-10-ev-guar-crr/GUARANTEES_CRR.md |
| DC | claude/ev-d19-cal-rescore @281a0bb9 | RESCORE_D19_CAL.md, receipts/2026-10-10-ev-d19-cal-rescore/ |
| ST | origin/state @9b47aca5 | BOARD.md, PREREG.md, results/ledger.tsv |
| RV | claude/ev-review-gap-attr @f2a04059 | docs/REVIEW_GAP_ATTRIBUTION.md, receipts/2026-10-11-ev-review-gap-attr/ (independent review: reproduction + Shapley CONFIRMED; 5 edits applied here) |
| JH | stack/qwen35/jev2-h2o @b5aa2645 | receipts/2026-10-11-jev2-h2o-x7c42/ (job-x7c42 landing) |
| XR | claude/ev-x7c42-rescore @bf706854 | receipts/2026-10-11-ev-x7c42-rescore/RESCORE_X7C42.md |
| JL | stack/qwen35/jev2-h2o @c5a2fb33 | receipts/2026-10-11-jev2-h2o-L-q668q/ (job-q668q landing) |
| HRN | local notes (not in git) | ssdm-wt/b-landscape/H2O_RECIPE.md (web research 2026-10-10; summarised pages) |
| **GA** | **this branch** | **scripts/gap_attribution.py, receipts/2026-10-11-gap-attribution/** (gap_attribution.json, cfg.json, run.log, SHA256SUMS) |

GA is new analysis over already-landed preds. It runs CPU-only and reads no model. Every input preds file is sha-checked against the source
receipt: H2O 60a06842…, stock 4B+shim edbffa95…, D19 a07aea11…, D19-cal 0a70b884…, JEV1 native 4915fd96… / native_shimcal
accbac06… / stock_shim a2a3d337…, and H2O arms e00d5ac7… / b2a37caf… / 8077cad9… / 4bdf446b… / 08a436bc…. I and C are recomputed with the
frozen `jevkit.scoring` (agent-main 2c6f5d61) and must equal the landed reports to 1e-9, which they do. S and K are taken as landed. The bootstrap
uses jevkit's own draws (`default_rng(16)`, baseline strata, B 1,000). It reproduces the source ΔComp A CIs exactly, for example stock 4B+shim
−9.38 [−23.05, −4.21] = SC and JEV1 −13.21 [−37.82, −7.59] = J1. Command:
`python -I -X utf8 scripts/gap_attribution.py --config receipts/2026-10-11-gap-attribution/cfg.json --out gap_attribution.json`.

## (1) Paired ΔComp A vs H2O, decomposed by axis
Composite A = 4 / (1/I + 1/C + 1/S + 1/K) × min(1, I/50)² × min(1, S/50)² × min(1, K/50)². The point estimates have no floor penalty
(every I ≥ 50), but the floor drives the long left tails of the CIs (J1: native I < 50 in 41.9% of resamples).

| candidate (job) | I | C | S | K | Comp A | ΔComp A vs H2O [95% CI] | source |
|---|---|---|---|---|---|---|---|
| H2O-Lightning-4B v1.1 @193ad740 (zbtbj) | 77.56 | 83.42 | 91.31 | 61.13 | 76.64 [72.83, 78.31] | baseline | SC |
| stock Qwen3.5-4B + H2O shim (zbtbj) | 56.27 | 69.34 | 91.60 | 61.13 | 67.26 | **−9.38 [−23.05, −4.21]** | SC, GA |
| JEV1-4B r2 native (xt3ax) | 51.18 | 64.69 | 85.25 | 61.23 | 63.43 | −13.21 [−37.82, −7.59] | J1, GA |
| JEV1-4B r2 native_shimcal (xt3ax) | 56.64 | 62.61 | 85.25 | 61.23 | 64.84 | −11.80 [−24.84, −6.25] | GA |
| stock 4B + shim, in-job (xt3ax) | 57.05 | 70.21 | 89.64 | 61.23 | 67.50 | −9.14 [−21.00, −4.00] | GA |
| D19 RWKV7-13.3B + joint (9j3uu) | 52.64 | 74.01 | 78.60 | 52.58 | 62.25 | −14.39 [−31.67, −10.63] | SC, GA |
| D19 + global dev T 0.806 (DC) | 52.93 | 68.28 | 78.60 | 52.58 | 61.27 | −15.37 [−31.41, −10.99] | DC, GA |

(JEV1's K of 61.23 comes from its declared size-class price, $0.03 in / $0 out, against H2O's 0.03 / 0.15 list basis. The effect is 0.03 points.)

**Axis attribution: Composite-A points recovered by setting the axis to H2O's value** (GA `pairs.*`). "Shapley" is the exact average over the 24 orderings,
and it sums to the gap.

| candidate | gap | I: single swap / **Shapley** [CI] | C: single / Shapley [CI] | S: Shapley | K: Shapley | I+C both | S+K both |
|---|---|---|---|---|---|---|---|
| stock 4B+shim | 9.38 | 6.01 / **6.28** [2.79, 18.98] (67%) | 2.87 / 3.14 [1.05, 5.33] | −0.04 | 0.00 | 9.43 | −0.04 |
| JEV1-4B r2 | 13.21 | 7.47 / **8.06** [3.98, 31.28] (61%) | 3.70 / 4.23 [1.94, 6.69] | 0.95 | −0.03 | 12.12 | 0.77 |
| JEV1 native_shimcal | 11.80 | 5.42 / **5.91** [2.26, 17.26] (50%) | 4.48 / 4.95 [2.46, 7.70] | 0.97 | −0.03 | 10.71 | 0.80 |
| D19 | 14.39 | 6.54 / **7.26** [3.79, 23.15] (50%) | 1.51 / 1.83 [0.70, 4.90] | 2.12 | 3.18 | 8.39 | 4.61 |
| D19 cal | 15.37 | 6.20 / **7.02** [3.69, 22.05] (46%) | 2.60 / 3.13 [1.09, 6.43] | 2.09 | 3.13 | n/c | n/c |

Reading the table:
- Intelligence is the largest single attribution for every candidate, and its CI excludes 0 in every row.
- Calibration is second for the 4B candidates, and its CI also excludes 0.
- Speed and Cost are about 0 for the Qwen-4B line, because H2O shares the host, the price class and the shim.
- For D19, S and K together cost 5.3 points. Those come from the 13.3B price class and the 0.50 s raw p50 (SC latency).

**I is necessary but not sufficient, and so are I and C together.** With C held, I = 100 gives D19 72.4 and JEV1 74.7, both below 76.6.
Stock 4B+shim needs I ≥ 95.3. Setting I and C to H2O's values reaches different places:
- stock 4B+shim: 76.69, which clears by +0.05 only because its S is 0.29 above H2O's;
- JEV1-4B r2: 75.55, **1.09 short**, the Speed gap of the native readout (S 85.25 vs 91.31);
- D19: 70.64, **6.0 short**. Even with I = 100 and C at H2O's value it is still 2.2 short, so it needs S and K.

The bootstrap probability that I, C := H2O's closes the gap is 1.00 / 0.00 / 0.00 (review §c, `shap_rank.json`). I and C are sufficient only
when S and K already match H2O's (the shim readout on the 4B host). Capability gaps: ΔCap −17.7 [−26.5, −8.2] for stock 4B+shim and
−17.2 [−26.6, −10.7] for D19 (SC), and −22.6 [−32.0, −12.4] for JEV1 (J1).

## (2) Within Intelligence: where the points are lost
I (open) = the mean over the three types of max(0, CC_type), where CC is the tier-weighted chance-corrected accuracy (`jevkit/scoring.py`). Each type's
contribution to the I gap is therefore ΔCC_type / 3 (GA `per_type`; the CCs are also in each landed report's `per_type_split`).

| candidate | choice CC (Δ I pts) | noul CC (Δ I pts) | score CC (Δ I pts) | ΔI | noul in 0.2–0.8 band |
|---|---|---|---|---|---|
| H2O | 82.9 | 76.6 | 73.2 | – | 0 / 74 |
| stock 4B+shim | 62.8 (−6.70) | 40.0 (**−12.19**) | 66.0 (−2.40) | −21.29 | 0 / 74 |
| JEV1-4B r2 native | 60.0 (−7.63) | 27.9 (**−16.21**) | 65.6 (−2.55) | −26.38 | 10 / 74 |
| JEV1-4B r2 native_shimcal | 60.0 (−7.63) | 44.1 (−10.82) | 65.8 (−2.48) | −20.92 | 0 / 74 |
| D19 | 66.9 (−5.32) | 36.3 (**−13.41**) | 54.6 (−6.19) | −24.92 | 12 / 74 |

**Per-item I weights.** Each type weighs 1/3 whatever its n, and tiers are weighted within a type (easy 0.1, standard 0.2, judge 0.3, hard 0.4).
So one full miss is worth very different amounts of I depending on the item (review §d):

| item | I points lost per full miss |
|---|---|
| choice easy / standard / hard (n 36 / 36 / 67) | 0.17 / 0.33 / 0.38 |
| noul easy / standard / hard (n 12 / 24 / 38) | 0.79 / 0.79 / 1.00 |
| score standard / hard (n 12 / 6) | 2.2 / **10.4** (per unit nMAE / nMAE_chance) |

- The 6 hard score items carry **22% of all I weight**, and one noul flip is worth about 2.6 hard-choice flips.
- "noul is the biggest hole" holds under this scorer: P(noul is the largest type) = 0.895 / 0.925 / 0.986. But part of that is per-item weight, not item count.
- Per-type I-gap CIs for noul: stock [5.97, 19.38], D19 [6.39, 20.22], JEV1 [9.61, 22.81]. The score-type terms span 0 for all three
  (for example D19 [−1.69, 12.33]), so that row is dominated by noise.

**Per family: the family-swap counterfactual** (GA `per_family`). For each family, the candidate's answers on that family are replaced by H2O's and
I is recomputed. The I points recovered sum exactly to the I gap (21.29 / 24.92 / 26.38) because no type is clipped at 0. The table
also gives jevkit-correct counts (a noul answer in the band counts as wrong; band counts in parentheses). It is sorted by the summed gap. It is
also in `receipts/2026-10-11-gap-attribution/famtable.md`.

| family | n | types | H2O correct | stock 4B+shim: correct / I pts | D19: correct (abst.) / I pts | JEV1-4B r2: correct (abst.) / I pts |
|---|---|---|---|---|---|---|
| probability | 10 | choice/noul | 10 | 4 / 3.52 | 2 (3) / 4.91 | 5 (2) / 3.14 |
| temporal_numeric | 15 | choice/noul/score | 5 | 2 / 4.06 | 4 (1) / 2.67 | 1 / 4.72 |
| long_policy | 19 | choice/noul/score | 14 | 10 / 1.92 | 10 (2) / 2.64 | 4 / 4.57 |
| trap | 8 | choice/noul/score | 8 | 7 / 0.89 | 7 / 2.81 | 6 (1) / 3.59 |
| adequacy | 12 | noul | 11 | 9 / 1.59 | 8 (4) / 2.38 | 8 (4) / 2.38 |
| judge_hard | 17 | noul | 12 | 10 / 2.01 | 11 (1) / 1.00 | 9 (2) / 3.01 |
| tradeoff | 6 | choice/noul | 5 | 5 / 0.62 | 3 / 1.38 | 2 / 1.76 |
| ordinal | 12 | score | 12 | 12 / −0.68 | 6 / 4.06 | 12 / 0.16 |
| adversarial | 6 | choice/noul/score | 6 | 4 / 2.36 | 6 / 0.84 | 6 / 0.26 |
| ambiguous | 7 | choice | 6 | 3 / 1.14 | 2 / 1.52 | 4 / 0.76 |
| multi_hop | 18 | choice/noul/score | 15 | 10 / 1.29 | 12 (1) / 0.72 | 13 / 0.06 |
| fact | 12 | noul | 12 | 11 / 0.79 | 12 / 0.00 | 11 (1) / 0.79 |
| intent | 24 | choice | 24 | 22 / 0.67 | 24 / 0.00 | 22 / 0.67 |
| policy | 12 | noul | 12 | 11 / 0.79 | 12 / 0.00 | 12 / 0.00 |
| routing | 12 | choice | 12 | 11 / 0.33 | 12 / 0.00 | 11 / 0.33 |
| extraction | 24 | choice | 24 | 24 / 0.00 | 24 / 0.00 | 23 / 0.17 |
| routing_hard | 5 | choice | 5 | 5 / 0.00 | 5 / 0.00 | 5 / 0.00 |
| tool_selection | 12 | choice | 12 | 12 / 0.00 | 12 / 0.00 | 12 / 0.00 |
| **sum** | 231 | | 205 | 172 / 21.29 | 172 (12) / 24.92 | 166 (10) / 26.38 |

**Concentration: RETRACTED as stated in 2d2f346f; corrected per the evidence review (§e, `fam_boot.json`, `split_half.json`).**
- **Point values.** The top six families by summed gap (probability, temporal_numeric, long_policy, trap, adequacy and judge_hard) hold
  65.6% [27.5, 86.0] / 65.8% [39.4, 83.6] / 81.2% [60.3, 94.4] of the I gap. These come from a paired bootstrap, B 1000.
- **Null reference.** The six are 35% of the items but **45% of the I weight**, because they are almost all hard-tier. The fair comparison is
  66 / 66 / 81% against 45%, not against 35%.
- **Which families are robust.** Only **probability** excludes 0 for all three candidates: stock 3.52 [0.76, 6.91], D19 4.91 [1.52, 8.67],
  JEV1 3.15 [0.75, 6.46]. **temporal_numeric** also excludes 0 for stock (4.05 [0.38, 9.81]) and JEV1 (4.72 [1.00, 9.50]), but not for
  D19 ([−0.38, 6.48]). long_policy spans 0 for every candidate (for example JEV1 4.57 [−4.92, 13.86]), because it holds 2 hard score items
  worth about 10 I points each.
- **The six are a post-hoc union across candidates.** Per candidate, they are the wrong six:
  - D19's most certain family loss is **ordinal**, 4.06 [3.25, 4.71], in the top 6 in 100% of resamples, and it is not among the six;
  - stock's #3 is adversarial, 2.36 [0.00, 6.73], also not among the six;
  - trap is only stock's #9 (P(top 6) = 0.29).
- **Split-half shrinkage.** Pick the top 6 on one random stratified half and measure them on the other half (500 splits). Their share
  drops from about 90% to **30% [−11, 61] / 45% [22, 70] / 54% [18, 83]** (stock / D19 / JEV1).
- **Supportable statement:** noul, probability and (for the 4B line) temporal_numeric are reliably behind. Family-level attribution beyond
  that is not resolved with 4–24 items per family.
- The seven short "original/easy" families (intent, extraction, routing, routing_hard, tool_selection, fact, policy; 101 items) together hold
  ≤ 2.6 I points for every candidate.
- **Content.** The families we lose (probability, temporal_numeric, multi_hop, long_policy) match what H2O's card says its stage 2
  targets: long documents plus numbers, dates and probabilities, and 37% documents/policies plus 22% numbers/dates/probability in the mix (HRN).
  That is the card's statement, not something we measured.
- **JEV1 vs stock, same job** (J1 per-family table). Training moved −31.6 pp on long_policy (4 vs 10 / 19) and +16.7 pp on multi_hop. Net ΔI was
  −5.87 [−15.81, +5.11]. The noul type alone accounts for −4.8 of it (GA pair `jev1_native - jev1_stock_shim`).

## (3) Named, measured causes
| # | cause | measurement | size | receipt |
|---|---|---|---|---|
| C1 | **Data distribution mismatch** (jev1 draw vs public shapes) | long_policy is train 0/12,000 rows > 2,048 tokens (p50 934) vs public 19/19 > 2,048 (p50 2,657). The draw is a 3-template monoculture (11 instruction strings) with 66.2% noul-false priors. 0/96,000 jev1 states are prose, vs 115/115 public covered states. 14,540/30,046 noul rows have empty criteria, vs explicit criteria on every public noul. intent K is 6–16 in train vs K = 5 public; routing K is 4–12 vs 6. | long_policy −6 top-1 (4 vs 10/19), NLL +0.71. Length is ruled out: multi_hop is just as long, is uncovered, and moved +11.8 pp with NLL −0.05. Covered families −6.7 pp [−13.9, +0.0]; ΔI on covered items −11.5 [−31.2, −1.6]. | AUD §1–5; J1 length split and grouped deltas |
| C2 | **noul indecision** (jevkit counts 0.2 < p < 0.8 as wrong) | JEV1 native 10/74 in band vs stock 0. D19 12/74 (cal 10). H2O raw readout 32/74 (24 on the correct side), and its shim floor (0.801) empties the band. | H2O: the floor plus T 0.8 is worth **+23.7 I / +8.68 Comp A** (b_native −8.68 [−18.39, −5.94]). JEV1: the shim floor on native recovers noul CC 27.9 → 44.1 (+5.4 I) and Comp A +1.41. Even with 0 abstentions, stock 4B+shim still trails H2O by 12.2 I points on noul (wrong-side answers). | HE, HR §6; J1; GA |
| C3 | **Readout speed** | native fork-head readout p50 raw **0.197 s** (S 85.25) vs the shim on the same host in the same job **0.067 s** (S 89.64); H2O 0.053 s (S 91.31). D19's Decider readout is 0.502 s (S 78.60). Cross-job S noise is ~2.7 points (SC: the same RWKV7 stock arm measured in two jobs). | JEV1: 0.61 of the −4.07 vs stock, and 0.95 of the −13.21 vs H2O (Shapley S). D19: 2.12. | J1 `arms.*.latency`; SC; GA |
| C4 | **Under-training** | JEV1-4B r2 saw 1,121 × 8 = 8,968 rows = **0.093 epoch** of the 96,000-row draw (best checkpoint step 1000, 8,000 rows). That is about 1,100 rows per family, which is enough to learn label priors (C1) but not rule application. | Not isolated: no full-epoch arm exists (see §4). | AUD §6; J1 |
| C5 | **Calibration readout: what post-hoc can and cannot recover** | On H2O (job-4g6m8, T fitted only on the held-out jev2a slice, ids 204e9bfe): a_T +0.31 [−0.44, +1.19], a_Ttype +0.07 [−0.74, +0.88], a_Tfam **+0.51 [−0.21, +1.35]**. Top-1 is unchanged (0 discordant) and choice NLL improves a little. D19 global dev T 0.806 sharpens: Comp A 62.25 → 61.27, C 74.01 → 68.28. JEV1 native_shimcal: ΔA vs stock −2.66 [−12.14, +6.42]. | Post-hoc T is worth at most ~+0.5 Composite at H2O's level, and it is negative for D19. It cannot change top-1, so it cannot touch the choice gap (−5.3 to −7.6 I points) or wrong-side noul answers. Its one large lever, the noul floor, is already in every shim arm. | HE, HR §3–4; DC; J1 |

Context that does not enter the Composite (reported for completeness):
- **Flip rate.** H2O 2.44% [1.26, 3.75] vs stock 4B+shim 7.55% [5.71, 9.67] (FL).
- **Guarantees.** No model of ours certifies under any cluster-valid LTT test. Under the pre-registered CRR c = 8 test, nothing certifies for any
  model, the Clefs included. Clef's one-per-row certificates (51.6% / 59.3%) stand only for R_1 (GU).

## (4) Untested, and why (budget)
The balance at the time of these decisions was about $104 (BOARD #13, ST). Each item is a named open lever with no receipt yet.

| item | status | why untested | where |
|---|---|---|---|
| Full-size **JEV2-4B** (sym-fork, jev1 + jev2a/b/c, max_len 4,096, LS 0.05) | pre-registered (A-PREREG-JEV2-4B); CPU tests only (job-ftqy8) | worst case ≤ $80. Credit went to the H2O-host line first (BOARD #14). Directly targets C1 + C4. | ST PREREG.md, ledger |
| **Consistency-view training** + **νPI λ ablation** (stack/qwen35/consistency @d517df9c) | CPU tests 13/13, OFF path bit-identical (job-xejp8) | no GPU run funded. Its effect on noul indecision (C2) and calibration is unmeasured. | ST ledger 2026-10-11T08:10Z |
| **D20 links 2–3** (stack/rwkv7/d20-v4-chain @bd7c7e78) | link 1 artifact kept; PARKED | l2 + l3 need ~$113 of holds vs $104 credit, and l2 alone yields no screen (BOARD #13). The 13.3B line also carries the S + K penalty (5.3 points). | ST BOARD.md |
| **H2O comparator rev 0940cbd7** (post-v1.1, multimodal, per-type T) | not measured | EV-PREREG-H2O-HOST amendment 1: a different, newer model. If measured it is a raised bar, never our host retroactively. | ST PREREG.md |
| JEV2-H2O-L (17,959 rows) | LANDED job-q668q, flat (§5b) | – | JL |

## (5) JEV2-H2O result: LANDED, FLAT (job-x7c42)
Source tag **JH**: stack/qwen35/jev2-h2o @b5aa2645, receipts/2026-10-11-jev2-h2o-x7c42/ (README.md, landing.json, slice_floor.json; art-cwk49
tar sha 0c538a61…; native_shimcal preds 296cf7d3…, control preds 3555841e…).

**Setup.** A-PREREG-JEV2-H2O (+ am. 1–3): H2O-Lightning-4B v1.1 @193ad740 + SSDM jev2 data (2,664 rows), lm_label head, LoRA r16, 333 steps.
Step 200 was selected by dev-card NLL on slice 142a593a (0.4566 vs 0.4908 at step 0). The selection was frozen before any public-231 prediction.
The job was billed $2.66.

**PRIMARY** (native_shimcal vs the in-job H2O v1.1 control, public-231, B 1000 seed 16):

| quantity | value |
|---|---|
| **Δ Composite A** | **−0.41 [−1.75, +1.39]** (76.16 vs 76.57); **not a success**, the CI spans 0 |
| ΔI | +0.64 [−2.52, +3.21] |
| ΔC | −2.13 [−6.92, +5.09] |
| ΔCapability | −0.74 [−3.60, +3.33] |
| S | 90.77 vs 91.34; K equal |

**The slice gain does not transfer to the public items.**
- **Step-0 floor (PASS):** on the held-out slice (3,600 decisions, 0 unmatched), top-1 is 85.0% vs 80.0%, a paired gain of **+4.44 pp [+3.44, +5.39]**.
- **ΔI on public-231 by type** (native_shimcal − control):
  - choice −1.14 [−5.68, +2.29];
  - noul −3.01 [−9.02, 0.00];
  - score **+6.08 [+0.09, +10.74]**. This is post hoc on the public items (see the §6 caveat), so it is descriptive and not a reason to select.
- **Raw readout** (`native`, without the shim T / floor): ΔA −3.18 [−6.13, −1.37]. The noul band holds 16/74 items (control 0), so the band
  gate fails for the raw arm. native_shimcal has 0/74. This confirms C2: the shim floor is load-bearing on a trained H2O as well.
- **Other readouts:** tfit T 0.93, ΔA −2.66 [−5.36, −0.99]. G1 (shim mode, reported only) fails: the labelled prompt is not invariant.
- Per-family deltas: JH landing.json (`by_family_public`, `slice_floor.per_family`). They are descriptive only (§2, §6).

**Independent rescore** (source tag **XR**: claude/ev-x7c42-rescore @bf706854, receipts/2026-10-11-ev-x7c42-rescore/RESCORE_X7C42.md). It AGREES on
95 fields (max |Δ| 0). The primary is not met: the 97.5% CI is [−2.06, +1.58] and one-sided p = 0.565, so Holm cannot claim it in any order.
Corrections and context:
- **Slice levels.** The exact slice levels are **84.61% vs 80.17%** (the 85.0 / 80.0 above are rounded). Δ +4.44 pp [+3.47, +5.31], McNemar p 3e-19.
- **Control path.** This job's control is **76.57**, against 76.78 (4g6m8) and 76.64 (zbtbj). It runs H2O through the SSDM lm_label readout path
  rather than the vendored shim forward. Top-1 is identical (205 on 231/231 items) and the drift is float-level (max |dp| 0.019). The primary is
  paired in-job, so it is valid, but "vs zbtbj" would read about 0.2 A differently.
- **Nondeterminism.** GPU training is nondeterministic run to run. Step 0 is bitwise equal to job-9fy9p, but the trained steps diverge from step 1
  (NLL Δ ≤ 0.0011, top-1 ±3/360). Both runs select step 200, and step 200 was scored.
- **Transfer table** (XR §7). The six families with significant slice gains (tradeoff +15.5, adversarial +14.5, trap +10.0, adequacy +8.5,
  ambiguous +8.0, policy +7.0 pp) are **saturated on public**: the control gets 83–100% top-1, with 3 errors in 51 items. **23 of the 26 public
  errors** sit in temporal_numeric (10), judge_hard (5), long_policy (5) and multi_hop (3). On the slice these show no gain or a small one
  (+3.0 n.s., 0.0, +1.5 n.s., +3.5).
- **Shape shift** at the schema level: dict vs string states, a different question-type mix, different K, and public states 2–6x longer.
- **Score is the only type whose gain transfers:** slice +10.27 pp, public cc +6.08 [+0.15, +10.51].

**Why the slice gain did not transfer: UNTESTED hypotheses**, in the evidence lane's ranking. None of them has a receipt yet.
1. **Headroom mismatch.** The slice starts at 80.0%, but H2O already gets 205/231 public items right (I 77.6), so there is little room left
   on the public items. Most of the slice gain may come from items H2O-like public items already get right.
2. **Gold conventions.** jev2's golds come from a rule engine, while the board's judgments follow human-rubric conventions (judge_hard, adequacy,
   policy, ambiguous). Training on the former can raise slice accuracy and still not move public answers. This is jev3 proposal (a) on BOARD (state 0d3dda70).
3. **Shape.** Remaining length and format gaps between the draw and the public items: the jev2b families and 3 jev2a-v2 families, and prose
   states (C1; jev3 proposal (c)).
4. **Under-training.** 333 steps on 2,664 rows, with the best checkpoint at step 200. **Disfavoured by JEV2-H2O-L (§5b):** 7x the data doubled the slice gain and
   left public flat-to-worse.

## (5b) JEV2-H2O-L: LANDED, FLAT (job-q668q)
Source tag **JL**: stack/qwen35/jev2-h2o @c5a2fb33, receipts/2026-10-11-jev2-h2o-L-q668q/ (README.md, landing.json; art-5si2g tar sha 91287177…;
billed $9.39).

**Setup.** The same cell on bschema-draw-jev2h-L (17,959 rows). The step-10 rate (3.93 s/step) shortened the plan to 1,832 of 2,244 steps, and
1,845 ran. Step 750 was selected by dev-card NLL (0.4031 vs 0.4908 at step 0), frozen before any public-231 prediction.

| quantity | JEV2-H2O-L (q668q) | JEV2-H2O (x7c42) |
|---|---|---|
| rows / selected step | 17,959 / 750 | 2,664 / 200 |
| **slice floor** (3,600 decisions) | **89.0% vs 80.0%, +8.97 pp [+7.69, +10.22]** | 84.61 vs 80.17, +4.44 pp [+3.47, +5.31] |
| **Δ Comp A (public-231)** | **−0.97, 95% [−3.38, +1.12], 97.5% [−3.73, +1.41]** | −0.41, 95% [−1.75, +1.39], 97.5% [−2.06, +1.58] |
| ΔI / ΔC / ΔCap | −2.98 [−8.26, +1.89] / −0.59 / −1.79 | +0.64 / −2.13 / −0.74 |
| ΔI per type: choice / noul / score | −3.42 / −0.63 / −4.90 | −1.14 / −3.01 / +6.08 |
| slice-vs-public transfer correlation (per family) | **−0.32** | +0.15 |
| noul band, native_shimcal / raw native | 0/74 / 16/74 | 0/74 / 16/74 |

- **Holm (A-PREREG-JEV2-H2O am. 3): neither arm is claimable.**
- Training **about 7x more** jev2 data doubles the in-distribution gain (+4.4 to +9.0 pp). On public, Composite A gets slightly worse
  (−0.41 to −0.97), and the per-family transfer correlation turns negative. The score-type transfer seen in x7c42 (+6.08) reverses (−4.90).
  No change here has a CI that excludes 0 on public.
- JL is the lead's landing, not yet independently rescored.

## (5c) Conclusion: the remaining gap is attributed to the training-data distribution
Two runs on H2O's own host (x7c42, q668q) separate the candidate causes:
- **Not capacity.** The same 4B host learns the jev2 distribution, and learns more of it with more data: slice +4.44 then +8.97 pp, both
  CIs > 0, on ids disjoint from training and from public (XR §5).
- **Not readout.** The shim readout keeps the noul band at 0/74 in every arm. S and K are within 0.6 of the control. The control itself is H2O on
  the same path.
- **The training-data distribution.** More of our data raises in-distribution accuracy and lowers public accuracy (transfer correlation
  +0.15 then −0.32). The slice gains fall in families already saturated on public, and public's errors (23/26) fall where the data teaches nothing
  measurable. So the remaining Composite gap, here and in §1–§3 (C1), is attributed to the data's distribution relative to the public/board items:
  - **headroom:** gains where H2O is already right;
  - **gold conventions:** rule-engine vs human-rubric golds;
  - **shape:** state type, length, K, type mix.

  It is not attributed to model capacity or the readout.
- **What is not separated.** Which of headroom, conventions and shape dominates is untested; that is hypotheses 1–3 in §5, with
  under-training (4) now disfavoured by q668q. The jev3 proposal (BOARD, state 0d3dda70) targets all three. Any jev3 effect is judged only on the
  pre-registered overall Composite A (§6 caveat).

## (6) Sealed-split caveat
- jevkit's public-231 Intelligence is not on the board's scale. H2O scores I 77.6 locally vs 60.0 on the board, with an identical 205/231 card
  (open→board I gap **−17.6**; ST PREREG.md EV-PREREG-COMPOSITE-LOCAL; kit parity claude/ev-kit-parity @5d674b2d).
- The board averages 300 open and 1,200 sealed items and applies a gap penalty. Our sealed generalisation is unknown, so a positive local Δ would
  still need a board submission (operator approval) before any board claim.
- The attribution above is valid only on the local basis. The sealed split could reorder candidates whose open I advantage comes from public-like
  item shapes.

**Adaptive-analysis caveat** (evidence review §f). The family and type rankings above are exploratory: they were chosen post hoc by looking at
our losses on the same public-231 items that will measure any fix. Five rules follow from that:
1. Any training aimed at these families is judged only by the **pre-registered overall Composite A** (EV-PREREG-COMPOSITE-LOCAL: paired Δ vs the
   in-job control, 95% CI). It is never judged by a targeted-family subtotal. Family deltas are reported as descriptive, labelled "selected post hoc on
   these items".
2. The primary metric is pre-registered before the run.
3. Confirmation happens off the selection set:
   - first on a held-out jev slice, which must **never** be the T-fit ids 204e9bfe;
   - then on the sealed split, through a board submission that needs operator approval.
4. Family-targeted data is generated from **family descriptions** (H2O card, task taxonomy), never from public item text or templates. A leak
   audit (n-gram/embedding overlap against public-231) is attached to the data receipt before training (hard rule 3).
5. A family share is always quoted next to its I-weight share (the six: 45%), not its item share (35%).

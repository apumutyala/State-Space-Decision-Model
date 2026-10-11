# Independent rescore: JEV2-H2O = job-x7c42 (A-PREREG-JEV2-H2O + amendments 1-3)

Evidence lane, 2026-10-11. Lead landing: stack/qwen35/jev2-h2o @b5aa2645, receipts/2026-10-11-jev2-h2o-x7c42/.
Label: "H2O-Lightning-4B v1.1 + SSDM jev2 data". Caveat: open public items only; the board's sealed split may rank
differently (open-to-board I gap measured on H2O: -17.6). Never a board score.

## Verdicts
| question | verdict |
|---|---|
| PRIMARY (Holm over x7c42 + JEV2-H2O-L) | **NOT MET. x7c42 is not claimable in any Holm order.** Δ Composite A (native_shimcal − in-job H2O v1.1 control) = **−0.409**, 95% CI **[−1.754, +1.395]**, 97.5% CI **[−2.060, +1.579]**, one-sided bootstrap p(Δ ≤ 0) = **0.565** (0.565 with the +1 correction). If x7c42 holds the smaller p it fails at 0.025 and the whole family stops. If L holds the smaller p and passes, x7c42 is then tested at 0.05 and fails. |
| Number agreement with the job | **AGREES.** 95 fields of screen.json were rescored from raw preds with the frozen jevkit CLI. Max \|Δ\| = 0.0, 0 flags. The primary agrees with the lead's ledger row and README exactly. |
| Step-0 floor | **floor beaten.** Slice top-1 is 84.61% vs 80.17% (3,600 decisions, 1 per id). Δ = +4.44 pp, 95% CI [+3.47, +5.31] (iid rows) or [+3.50, +5.31] (family-stratified). Exact McNemar p = 3.1e-19 (244 vs 84 discordant). |
| Determinism vs 9fy9p | **Selection reproduces; trained steps do not reproduce bitwise.** Step 0 is bitwise identical. Steps 100/200/300 differ by NLL +0.0011 / +0.0007 / +0.0004 and top-1 by +1 / −3 / −1 of 360. Both runs select step 200, and **step 200 was scored** (ckpt sha ca6c19d7…, exported trainable-step0000200 3c04939e…). Training diverges from step 1 (loss 0.4999 vs 0.4909) with identical config, seed 0, records and drawn ids, so this is run-to-run GPU training nondeterminism, not float noise. |
| Order of events | **Selection BEFORE public: AGREES** (cell.json 14:51:16Z < first public-231 pred 14:55:38Z). **slice_floor BEFORE public: DISAGREES.** calib_probs.jsonl is 15:00:48Z and slice_floor.json / rows are 15:04:54Z, both after native.preds (14:55:38Z) and stock_shim.preds (15:01:05Z). This is not a breach: the floor is a fixed pass/fail on an already frozen checkpoint vs the untrained host, and nothing in it can change the selection. |
| Host / integrity | **AGREES**, except "control bitwise = 4g6m8 / zbtbj", which is **DISAGREES (explained)**. See (2). |

## (1) PRIMARY: recomputed from raw preds (B 1000, seed 16, stock_shim strata, Speed/Cost held at measured values)
| quantity | lead / in-job | rescore | verdict |
|---|---|---|---|
| Comp A native_shimcal / control | 76.161 / 76.570 | 76.161 / 76.570 | AGREES |
| Δ Comp A [95%] | −0.41 [−1.75, +1.39] | −0.4091 [−1.7538, +1.3947] (own loop = jevkit `bootstrap()._pairs`, 1e-9) | AGREES |
| Δ Comp A [97.5%, Holm] | – | [−2.0596, +1.5790] | new |
| one-sided p | – | 0.565 | new |
| Δ I / Δ C / Δ Cap | +0.64 [−2.52, +3.21] / −2.13 [−6.92, +5.09] / −0.74 [−3.60, +3.33] | identical to 1e-12 | AGREES |
| S / K | 90.77 vs 91.34 (−0.57, latency) / 61.23 both | same | AGREES |
| per-type ΔI (cc) points | choice −1.14, noul −3.01, score +6.08 | −1.140, −3.008, +6.078 | AGREES |
| per-type ΔI CIs | choice [−5.68, +2.29], noul [−9.02, 0.00], score [+0.09, +10.74] | choice [−4.66, +2.28], noul [−9.02, 0.00], score [+0.15, +10.51] | noul AGREES. choice and score **DISAGREE in the CI bounds** (\|Δ\| up to 1.02). That is a method difference: here it is the type cc on the global stratified draw, and the lead's CI method is not in the landing. The points agree. |
| raw `native` Δ A | −3.18 [−6.13, −1.37] | −3.178 [−6.127, −1.374]; 97.5% [−6.62, −1.17] | AGREES |
| `native_tfit` Δ A | (in screen.json) | −2.659 [−5.362, −0.989]; T 0.932 fitted on slice calib_probs only | AGREES |

## (2) Host / integrity
- **Artifact.** I downloaded it myself through the job record's signed URL. art-cwk49 is a 130,693,120-byte tar with sha256 `0c538a61…6ae220`, which equals `job get job-x7c42` artifact.sha256. The phase tgz shas equal MANIFEST.jsonl. The preds, slice rows, calib_probs, cell.json and screen.json shas equal the lead's landing.json (`f63858d1`, `296cf7d3`, `e65160ae`, `3555841e`, `d1bf3590`, `2f53e661`, `76eda3ab`, `7fcd2d63`).
- **rev 193ad740 loaded: AGREES.** run.log reads `HOST h2oai/h2o-lightning-4b@193ad740…`. host-verify.json gives revision = resolved_sha = 193ad740925b176a3b70a5a13a7cff2f2fadd01e. snapshot-verify shows 10/10 files OK, including model.safetensors. identity: weights c65443b1…, missing_text_keys 0, label "H2O-Lightning-4B v1.1 + SSDM".
- **Licence copied: AGREES.** out/license/LICENSE sha256 bbedc3fd…f1e57a (= the Apache-2.0 text pinned in EV-PREREG-H2O-HOST). README 614272b0… equals host-verify.
- **Exclusion lists: AGREES.** Both are pinned in CONTEXT_MANIFEST with the prereg shas (mix_exclude_ids c0c175f5…, musique_provenance_unverified 5cebf60b…). The draw build's draw-report.json gives `global_exclude: n_ids 127, found 127` (23 + 104). The local draw records equal the in-job pins (train 7c78c210…, calib b40cac9c…).
- **prob_indep dropped: AGREES.** draw-report `excluded_subs: {prob_indep: 124}`. I rescanned the records: 0 prob_indep rows in train (2,664) and in the slice (3,600).
- **max_len 4,096, 0 truncation: AGREES.** For train, cell.json data_checks gives `no_truncation_tokenize: rows 2664, max_row_tokens 3930, t_cap 4096`, and the cell fails closed on any drop. For the public 231, max input_tokens is 3,952 ≤ 4,096 (0 over), and eval_max_t is 4096.
- **Control vs job-4g6m8 / job-zbtbj H2O preds: DISAGREES on "bitwise" (explained), AGREES on decisions.** Only 25/231 answers are bitwise equal. **Top-1 is identical on 231/231 (205 = 205 = 205).** The drift is float-level: max |dp| is 0.019 (choice; 2 items > 0.01), noul 0.0075, score 0.0074. Cause: x7c42's `stock_shim` arm runs H2O through the SSDM lm_label path (LoRA scaling 0, tied LM head with fp32 label rows, bf16 backbone, `causal_conv1d` torch fallback), not the vendored shim forward that 4g6m8 and zbtbj used. The two of those are bitwise equal to each other (ev-h2o-eval-rescore). Effect on the control: A 76.57 here vs 76.78 (4g6m8) / 76.64 (zbtbj), with I 77.59 vs 77.56, C 82.85 vs 83.42, S 91.34 vs 92.12. **The primary is paired in-job, and the trained arm and step 0 use the same path, so the comparison is valid.** The control is about 0.2 A below the board-anchored H2O run, so "x7c42 vs zbtbj" would read slightly differently. That comparison is not the registered primary.

## (3) Determinism vs job-9fy9p (my own download, art 69c87814…, cell.json re-read; not the quoted numbers)
| step | x7c42 NLL / top-1 | 9fy9p NLL / top-1 | Δ |
|---|---|---|---|
| 0 | 0.490837 / 0.8194 | 0.490837 / 0.8194 | bitwise 0 |
| 100 | 0.485235 / 0.8111 | 0.484171 / 0.8083 | +0.0011 / +1 item |
| 200 (selected in both) | 0.456600 / 0.8194 | 0.455860 / 0.8278 | +0.0007 / −3 items |
| 300 | 0.463988 / 0.8222 | 0.463600 / 0.8250 | +0.0004 / −1 item |

The same 360-row sample (ids sha b7dd0844… both), the config is equal apart from `_comment`, and records sha 824671c9… and drawn ids 329fca01… match. The train_log loss differs from step 1 (max |Δ loss| 0.116 over 333 steps). The ordering by NLL (200 < 300 < 100 < 0) is the same in both runs, so selection is robust to this nondeterminism. The coordinator's quoted "step 100 0.4842 / 0.8083" etc. are the 9fy9p values and AGREE with that cell.json. **Scored checkpoint: step 200** (identity-child.json ckpt-step0000200.pt sha ca6c19d7…).

## (4) Step-0 floor (slice 142a593a, 3,600 rows, 1 decision per id → no clustering needed)
- I recomputed the selected-arm column from calib_probs.jsonl (argmax vs gold): it matches 3,600/3,600, and the gold matches the local calib records 3,600/3,600. On the 360 dev-sample ids, step 0 gives 0.8194, equal to the dev card. The selected arm gives 0.8222 vs the step-200 card's 0.8194 (1/360; screen vs cell eval path).
- **Lead's levels "0.850 vs 0.800" DISAGREE (rounding).** The actual values are **0.8461 vs 0.8017**. Δ +4.44 pp AGREES. The lead's CI [+3.44, +5.39] vs mine [+3.47, +5.31] is a different bootstrap implementation, and both exclude 0.
- Per type (slice): choice +3.27 [+2.10, +4.37] (McN 3e-7), noul +2.17 [+0.72, +3.62] (0.004), score +10.27 [+7.32, +13.09] (9e-12).
- The per-family slice Δ points equal the in-job slice_floor.per_family exactly (max |Δ| 0).

## (5) Selection slice ids and overlap
- Slice ids sha256 (sorted, "\n"-joined) = **142a593a…00be**, which matches. The slice rows' record ids are the same set as the local calib records.
- Train ids (2,664) sha = 4ae135e1… (= draw-report). The in-job convention drawn_ids_sha 329fca01… is identical in x7c42 and 9fy9p.
- **Slice ∩ train: 0 ids, 0 base ids, 0 source items (source, source_id), 0 content (state + questions).** Note: 3/3,600 slice rows share an identical *state* with a train row (different question and view; synthetic generator collisions): jev1synth agent_audit (yn vs first), jev2synth temporal prorate, and jev2synth tool_pick (37-char state). This is not a content overlap under the prereg's definition. Flagged for the jev2 generator.
- **Slice or train ∩ public-231: 0 ids, 0 identical states, 0 identical state+question.**

## (6) Secondaries (EV-PREREG-SECONDARY-AXES; native_shimcal − control; never turn the failed primary into a success)
- Per axis: ΔI +0.64 [−2.52, +3.21]; ΔC −2.13 [−6.92, +5.09]; ΔS −0.57 (fixed); ΔK 0 (fixed); ΔCap −0.74 [−3.60, +3.33].
- Top-1 204 vs 205: Δ −0.43 pp [−2.16, +1.30], 2 arm-only vs 3 control-only, exact McNemar p 1.0. Choice NLL 0.3138 vs 0.3068, Δ +0.0070 [−0.0247, +0.0415].
- I < 50 in 0% of resamples for both arms. Δ A < −10 in 0%. Comp A quantiles (1/2.5/5/25/50/75/95/97.5/99): arm 71.32/72.08/72.91/74.74/75.89/76.95/78.47/78.91/79.52; control 72.33/72.77/73.60/75.13/76.10/76.92/77.98/78.38/78.67; Δ −2.14/−1.75/−1.55/−0.71/−0.14/+0.45/+1.15/+1.39/+1.63.
- Noul 0.2–0.8 band: control 0/74, native_shimcal 0/74, raw native **16/74**, native_tfit 14/74. The reported band gate fails for the raw arm. All AGREE with screen.json.
- Per family on public-231: ΔI points equal the in-job by_family exactly. The family CIs differ (n 5–24, per-family resampling vs my global-draw subsets); not decision-relevant.

## (7) TRANSFER (coordinator extra; diagnosis only, nothing fitted to public items)
Slice = selected − step 0 on the held-out slice (200 per family). Public = native_shimcal − control on the 231 items.
| family | slice step-0 acc | slice Δtop-1 pp [CI] (McN p) | public n | public control top-1 | public Δtop-1 pp [CI] (+/−) | public ΔI [CI] | flag |
|---|---|---|---|---|---|---|---|
| tradeoff | 54.5 | **+15.5 [+9.5, +22.0]** (3e-6) | 6 | 83.3 | 0 (0/0) | 0 | GAIN SLICE ONLY |
| adversarial | 67.5 | **+14.5 [+8.5, +20.0]** (2e-6) | 6 | 100 | 0 | +0.6 | GAIN SLICE ONLY |
| trap | 84.0 | **+10.0 [+5.0, +15.0]** (2e-4) | 8 | 100 | 0 | +5.9 | GAIN SLICE ONLY |
| adequacy | 70.0 | **+8.5 [+3.5, +13.5]** (0.002) | 12 | 91.7 | 0 | 0 | GAIN SLICE ONLY |
| ambiguous | 91.5 | **+8.0 [+4.5, +12.0]** (3e-5) | 7 | 85.7 | 0 | 0 | GAIN SLICE ONLY |
| policy | 74.0 | **+7.0 [+2.0, +12.0]** (0.009) | 12 | 100 | 0 | 0 | GAIN SLICE ONLY |
| multi_hop | 81.0 | **+3.5 [+1.0, +6.5]** (0.016) | 18 | 83.3 | +5.6 [0, +18.2] (1/0) | −1.9 | gain on both (1 item) |
| severity | 74.5 | +3.5 [−1.0, +8.0] | – | – | – | – | not on public |
| probability | 69.5 | +3.5 [−2.0, +9.0] | 10 | 100 | −10.0 [−33, 0] (0/1) | −10.7 | public loss |
| temporal_numeric | 58.5 | +3.0 [−2.0, +8.0] | 15 | **33.3** | 0 (1/1) | +5.0 | |
| routing | 84.0 | +2.0 [−2.0, +6.0] | 12 | 100 | 0 | 0 | |
| long_policy | 86.0 | +1.5 [−2.0, +5.0] | 19 | **73.7** | 0 | +2.2 | |
| intent | 97.5 | +0.5 [0, +1.5] | 24 | 100 | 0 | 0 | |
| judge_hard | 69.5 | 0.0 [−4.0, +3.5] | 17 | **70.6** | −5.9 [−20, 0] (0/1) | −11.8 | public loss |
| enum / tool_selection | 99.5 / 99.5 | 0 / 0 | – / 12 | – / 100 | – / 0 | | |
| fact / state_tracking | 98.5 / 83.5 | −0.5 / −0.5 | 12 / – | 100 / – | 0 / – | | |
| extraction / ordinal / routing_hard | – | not in the slice | 24 / 12 / 5 | 100 / 100 / 100 | 0 | 0 / +4.5 / 0 | |

By type: slice choice +3.27, noul +2.17, score +10.27 pp (all CIs > 0). Public top-1: choice −0.72 (1/2), noul −1.35 (0/1), score +5.56 (1/0). Public cc: choice −1.14, noul −3.01, score +6.08 [+0.15, +10.51]. Score is the only type whose gain transfers.

**Reading.**
1. **The six families with significant slice gains (tradeoff, adversarial, trap, adequacy, ambiguous, policy) are at or near ceiling on public**: control 83–100% top-1, with 3 public errors in total across 51 items. H2O's in-distribution step-0 accuracy on the slice is far lower for them (54.5–74% for four of six), so the training fixed slice errors that do not exist on public.
2. **Public's errors (26/231) sit where the slice shows no gain.** temporal_numeric has 10 errors (slice +3.0, n.s.), judge_hard 5 (slice 0.0), long_policy 5 (+1.5, n.s.) and multi_hop 3 (+3.5). The other 3 are in ambiguous, adequacy and tradeoff, 1 each.
3. **Shape shift (schema-level only, rescore-extra.json `transfer_by_family.shape_*`):**
   - Several families change state type. adequacy, policy, intent and routing use dict states on the slice and str states on public.
   - The question-type mix moves too. adversarial is 80% score on the slice vs mostly choice on public. adequacy is 38/28/34 noul/choice/score on the slice vs 100% noul on public.
   - K changes. intent K averages 9.3 on the slice vs 5 on public, and routing 7.5 vs 6.
   - Public states run 2–6x longer in adversarial, trap, tradeoff, probability, temporal_numeric, judge_hard and multi_hop (median state chars, e.g. trap 221 vs 891, probability 356 vs 2,139).
   - severity, enum and state_tracking have no public counterpart, and extraction, ordinal and routing_hard have no slice counterpart.

   The slice is in-distribution for training, not for public.

## Files (SHA256SUMS in this directory)
- Scripts: scripts/ev_rescore_jevkit.py (unchanged from @536fae54) and scripts/ev_rescore_x7c42.py (new: Holm CI, p, per-type and per-family on the same draws, floor, overlap, order, determinism, transfer).
- In this directory: check_subs_tokens.py and check_control_determinism_overlap.py, each with its .out.txt.
- Outputs: rescore.json, rescore-extra.json, reports/*.rescore.jevkit.json, price.json (the in-job declared price).
- In-job copies: screen.in-job.json, cell.in-job.json, slice_floor.in-job.json, host-verify.in-job.json, MANIFEST.in-job.jsonl, cell.9fy9p.json, job-records.json (signed URL redacted).
- Artifact shas: x7c42 0c538a61…6ae220; 9fy9p 69c87814…387796; zbtbj d0a04884…8acd; 4g6m8 control preds e00d5ac7…; zbtbj H2O preds 60a06842….

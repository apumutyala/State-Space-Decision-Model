# Conformal coverage + LTT abstention on JEV3-LM-S (job-zbmhc), jev3 held-out slice

Evidence lane, 2026-10-12. Governing rule: **B-PREREG-GUAR-JEV3/a** (origin/state PREREG.md, pushed at state `64f0c3dc`
BEFORE any zbmhc slice probability was read), which applies **B-PREREG-GUAR-JEV3** (state `745fc478`) unchanged:
the same FIT/LTT/TEST hash split, alphas, variants M/T/F, Bonferroni LTT primary, FS-CRR c = 1 / c = 8 and
fixed-sequence-binomial secondaries, per-type LTT, and the dev-card-remainder sensitivity. The arms are native_tfit
(the pre-registered primary readout, which decides the verdict), native_shimcal and raw native. Cost $0 (CPU, laptop,
about 5 s). One shot.

## Verdict

- **Coverage certificate (split conformal): HOLDS on TEST.** native_tfit under variant T covers 0.916 of TEST at
  alpha 0.10 and 0.955 at alpha 0.05 (n 418). The flag lines are 0.871 and 0.929.
  - **0 of 36 cells are flagged** (3 arms x 2 subsets x 2 alpha x 3 variants).
  - The sets are still wide: mean size 1.61, singleton rate 0.526.
- **Abstention certificate (LTT, epsilon 0.05, delta 0.10): FALSIFIED for JEV3-LM-S on the jev3 distribution.**
  - No lambda is certified for any arm (native_tfit, native_shimcal, native).
  - This holds at either alpha, under every variant (M, T, F), and under every test: the Bonferroni primary,
    FS-CRR c = 1 and c = 8, fixed-sequence binomial, and per-type LTT.
  - It holds on all rows and on the dev-card remainder.
  - TEST act fraction **0**.
- **Attribution (measured, LTT half, alpha 0.10, variant T).** The pre-registered tfit readout makes abstention
  worse than raw.
  - **native_tfit:** the frozen T_noul is **0.26**. This is the band-infeasible fallback: the T with the lowest
    band rate. It sharpens yes/no answers so hard that wrong answers become confident.
    - 77.5% of noul answers reach p_max >= 0.95, but those are only 84.7% accurate.
    - Best lambda 0.87: 187 acts with **19 errors (10.2%)**, binomial p 0.999.
    - At lambda 0.87, noul accounts for 165 acts and all 19 errors; choice has 22 acts and 0 errors.
    - The FS-CRR c = 1 stops at 0.95 with 181 acts and 19 errors.
    - Lowering lambda cannot help: the error rate is about 2 x epsilon at every lambda.
  - **native (raw):** confident answers are clean but too rare.
    - noul p_max >= 0.95 is 23.1% of answers at 97.2% accuracy.
    - Best lambda 0.94: 76 acts with 1 error, p 0.10.
    - FS-CRR c = 1: lambda 0.95, 70 acts with 1 error, p 0.29.
    - A Bonferroni certificate needs about 135 acts at 0 errors.
  - **native_shimcal:** best lambda 0.97, 75 acts with 1 error, p 0.11.
  - **score:** n_FIT 16 < 100, so it gets the full label set and never has a singleton. This is unchanged from 6zs79.
- **Compared with 6zs79 (JEV3-SYM s0):** zbmhc is better on every coverage and set statistic. On the raw readout it
  is also much closer to a certificate: 1 error in about 75 confident acts, against 3 in about 70.
  - It still falls short by acts at zero error (about 75 to 135 needed), not by error rate.
  - **This is the clearest lead for a future cell:** a readout that does not sharpen noul. With roughly 2x the raw
    model's confident-and-correct mass, the same pipeline would certify.

## Headline table (variant T = Mondrian per type; TEST; all slice rows)

| model | arm | alpha | TEST cov (n) | mean size | singleton | certified lambda | TEST act | TEST acted err |
|---|---|---|---|---|---|---|---|---|
| **JEV3-LM-S (zbmhc), step 150** | **native_tfit (primary)** | 0.10 | **0.916** (418) | 1.61 | 0.526 | **none** | **0** | - |
| JEV3-LM-S (zbmhc) | native_tfit | 0.05 | 0.955 (418) | 1.88 | 0.395 | none | 0 | - |
| JEV3-LM-S (zbmhc) | native_shimcal | 0.10 | 0.914 (418) | 1.61 | 0.529 | none | 0 | - |
| JEV3-LM-S (zbmhc) | native_shimcal | 0.05 | 0.952 (418) | 1.85 | 0.397 | none | 0 | - |
| JEV3-LM-S (zbmhc) | native (raw) | 0.10 | 0.916 (418) | 1.61 | 0.524 | none | 0 | - |
| JEV3-LM-S (zbmhc) | native (raw) | 0.05 | 0.955 (418) | 1.89 | 0.392 | none | 0 | - |

## Per type (native_tfit, variant T, TEST)

| type | n TEST | cov a0.10 | size | singleton | singleton acc | cov a0.05 | size | singleton | act |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.88 | 0.343 | 0.872 | 0.964 | 2.39 | 0.241 | 0 |
| noul | 269 | 0.914 | 1.36 | 0.643 | 0.867 | 0.948 | 1.51 | 0.491 | 0 |
| score | 12 | 1.000 (trivial: full set) | 4.42 | 0.000 | - | 1.000 | 4.42 | 0.000 | 0 |

- **Per family:** every family is below MIN_N 100, so variant F collapses to the marginal fit and no per-family
  guarantee exists ("insufficient").
  - Family TEST coverage under T at alpha 0.10 runs from 0.829 (long_policy, n 70, just above its flag line of
    0.828) to 0.968 (ambiguous).
  - judge_hard is 0.900, compared with 0.800 FLAGGED for 6zs79.
  - **Flagged sub-cells** (conditional undercoverage; the marginal guarantee allows these, so none is a violation):
    - long_policy under M/F at alpha 0.10 for shimcal and raw (0.800 vs 0.828). For raw it is also flagged under T
      (0.814).
    - noul under the MARGINAL M/F fit for native_tfit (0.862 vs 0.863; remainder 0.842 vs 0.856). This is why the
      per-type Mondrian T variant is the headline.
    - No headline (T, all-decisions) cell of the primary arm is flagged.
- `tables_jev3_zbmhc.md` has every arm, subset, alpha and variant with per type and per family. Every number is also
  in `guarantees_jev3_zbmhc.json`.

## Side by side vs 6zs79 (JEV3-SYM s0, step 200; same slice 844233f2, same split, same TEST 418)

| | 6zs79 native (= tfit, T 1.0) | zbmhc native_tfit (primary) | zbmhc native (raw) | 6zs79 shimcal | zbmhc shimcal |
|---|---|---|---|---|---|
| slice top-1 (all 1,200) | 0.678 | 0.712 | 0.712 | 0.678 | 0.712 |
| TEST cov a0.10 (T) | 0.888 | **0.916** | 0.916 | 0.895 | 0.914 |
| mean set size a0.10 | 1.62 | 1.61 | 1.61 | 1.65 | 1.61 |
| singleton rate a0.10 | 0.476 | 0.526 | 0.524 | 0.462 | 0.529 |
| TEST cov a0.05 / size | 0.935 / 1.87 | 0.955 / 1.88 | 0.955 / 1.89 | 0.933 / 1.85 | 0.952 / 1.85 |
| noul TEST cov / singleton acc a0.10 | 0.870 / 0.768 | 0.914 / 0.867 | 0.914 / 0.867 | 0.881 / 0.775 | 0.914 / 0.867 |
| certified lambda* (any test) | none | none | none | none | none |
| TEST act fraction / acted err | 0 / - | 0 / - | 0 / - | 0 / - | 0 / - |
| LTT best lambda: acts / errors (p) | 0.95: 71 / 3 (0.52) | 0.87: 187 / **19** (0.999) | 0.94: 76 / **1** (0.10) | 0.98: 69 / 3 (0.54) | 0.97: 75 / 1 (0.11) |
| noul p_max >= 0.95: share / acc (FIT+LTT) | 0.231 / 0.936 | 0.775 / **0.847** | 0.231 / 0.972 | 0.307 / 0.917 | 0.322 / 0.947 |
| noul 0.2-0.8 band (FIT+LTT) | 0.487 | 0.117 | 0.392 | 0.000 | 0.000 |
| verdict | coverage holds; abstention FALSIFIED | coverage holds; abstention **FALSIFIED** | same | same | same |

## Sensitivity: dev-card remainder (rows not used by checkpoint selection)

As written in the addendum, **a strictly clean subset is EMPTY**. The zbmhc T fit used all 1,200 slice rows, so no
row was untouched by both selection and the T fit. The pre-registered sensitivity is therefore run on the
**840 rows** outside the zbmhc dev-card sample (cell.json `dev_card_sample`, 360 ids, sha 8f313487; sorted-join sha
b2db7f9b).
- Checkpoint selection (step 150, by dev-card NLL) never saw these rows.
- They were used only by the T fit, which is a symmetric function of all rows and so preserves FIT/LTT/TEST
  exchangeability.

| arm | alpha | TEST cov (n) | size | singleton | lambda | act | LTT best: acts / err |
|---|---|---|---|---|---|---|---|
| native_tfit | 0.10 | 0.897 (301) | 1.57 | 0.588 | none | 0 | 0.99: 101 / 10 |
| native_tfit | 0.05 | 0.957 (301) | 1.95 | 0.372 | none | 0 | 0.90: 88 / 7 |
| native_shimcal | 0.10 | 0.907 (301) | 1.63 | 0.525 | none | 0 | 0.96: 52 / 1 |
| native | 0.10 | 0.900 (301) | 1.57 | 0.581 | none | 0 | 0.93: 49 / 1 |

Same verdict. None of these cells is flagged.

## Method (as pre-registered, plus the /a arms)

- **Data.** `inputs/job-zbmhc/calib_probs.jsonl` is the per-option untempered probabilities of every slice decision,
  from the scored best checkpoint, step 150.
  - Asserted: ids sha 844233f2, unique decision ids, one decision per record, and argmax top-1 = slice_floor
    native_top1 0.711667 exactly.
  - Asserted: cell.json best.step = best_export.step = 150.
- **native_tfit.** p_i^(1/T), renormalised, with the frozen zbmhc tfit.json per_type map, exactly as screen.py
  `t_for` / `retemper_answer` do. The T used per cell:
  - choice K = 3: T 0.9548 (cell, 209 rows)
  - other choice: T 1.0238 (207 rows)
  - noul: T 0.2606 (741 rows)
  - score: T 0.4289 (43 rows)
  - Argmax is preserved on every row (asserted).
- **native_shimcal.** The vendored shim `format_answer` (T 0.8, floor 0.801). Parity against the vendored module
  (95a731f6 `third_party/h2o_lightning_shim/h2o_lightning_shim.py`, sha 355791c3) holds on all 1,200 rows:
  max |d| = 0.0.
- **Pipeline.** `guarantees_jev3_zbmhc.py` imports the B-PREREG-GUAR-JEV3 driver
  (`../2026-10-12-ev-guar-jev3/guarantees_jev3.py`, sha 26a5fadb, unchanged) and calls its `run_variant`, split and
  loaders. Only the arm list is new.
  - jevkit/guarantees.py sha c8970343 (unmodified).
  - Split: FIT 376 / LTT 406 / TEST 418, identical to 6zs79. Remainder: 267 / 272 / 301.

## Honest caveats (apply to every number)

1. **The slice is not unseen, and it is less so than for 6zs79.** It was used for zbmhc's checkpoint selection (step
   150 by dev-card NLL on 360 slice rows) and for the T fit on all 1,200 rows.
   - Here T is far from 1 (noul 0.26), so native_tfit really is a slice-fitted readout.
   - Exchangeability between FIT, LTT and TEST still holds, because the T fit is a symmetric function of the rows.
   - The remainder sensitivity removes the selection reuse but cannot remove the T-fit reuse (no clean rows exist).
     The verdict is unchanged.
2. **This covers jev3-distribution decisions only.** Nothing here transfers to DEV, JevBench or the board under shift.
3. **No Clef comparison.** There are no Clef predictions on jev3.
4. **Gold is the in-job label** in calib_probs; it was not re-read independently.
5. **Single seed and single checkpoint.** About 400 rows per third, so the per-type and per-family cells are small.
6. **Not a board claim.**

## DIAGNOSTIC (non-public, NOT pre-registered): readout comparison on the slice

This section changes nothing about zbmhc's pre-registered verdict. It is legitimate non-public evidence on one
question: should a FUTURE cell pre-register native_shimcal as its primary readout?

**Bias warning.** The slice was used for zbmhc's checkpoint selection and for its T fit. **The native_tfit arm is
therefore optimistically biased here.** Even so, it loses to shimcal on NLL: its noul T was chosen to cut the band
rate, not to minimise NLL.

**Method** (`readout_slice_zbmhc.py`; numbers in `readout_slice_zbmhc.json`):
- **NLL** = mean -ln p_gold. **top-1** = argmax correct. **noul band** = share of yes/no rows with
  0.2 < P(yes) < 0.8.
- **Slice-proxy I and C** use the `jevkit.scoring` formulas. The slice is treated as ONE split and ONE tier, so there
  are no tier weights and no gap penalty. Types are weighted equally.
  - CC_choice = 100 (acc - mean 1/K) / (1 - mean 1/K).
  - CC_noul = 100 (decisive acc - 0.5) / 0.5. An answer inside the band counts as wrong.
  - CC_score = 100 (1 - mean nMAE / mean nMAE_chance).
  - I = mean of max(0, CC) over types.
  - C = mean over types of: choice 100 (1 - 2 ECE), noul 100 (1 - 2 ECE of P(yes)), and score 50 (1 - 2 topECE)
    + 50 (1 - nRPS). The choice ECE uses all rows, because the slice has no tiers. ECE uses 10 bins.
- **Two summary scores:**
  - **capability-like** = (I + C) / 2.
  - **CompositeA-like** = harmonic mean of (I, C) x min(1, I/50)^2. This is Composite A with the S and K axes
    dropped; there is no S or K.
- **Bootstrap:** paired, B = 1000, numpy default_rng(16). Rows are resampled within type, and the same resample is
  used for every arm. The intervals are 95% percentile CIs.

**Point values (all 1,200 rows; n choice 416 / noul 741 / score 43):**

| arm | NLL | NLL choice | NLL noul | NLL score | top-1 | noul band | I | C | capability-like | CompositeA-like |
|---|---|---|---|---|---|---|---|---|---|---|
| native_tfit (T map; biased) | 0.938 | 0.832 | 0.997 | 0.928 | 0.712 | 0.107 | 47.96 | 78.99 | 63.48 | 54.92 |
| native_shimcal | 0.657 | 0.849 | 0.529 | 1.003 | 0.712 | 0.000 | 49.52 | 79.08 | 64.30 | 59.75 |
| native (raw) | **0.617** | 0.833 | **0.470** | 1.052 | 0.712 | 0.395 | 32.24 | **85.03** | 58.64 | 19.45 |

**Paired differences [95% CI]:**

| metric | shimcal - tfit | shimcal - raw |
|---|---|---|
| NLL | **-0.281** [-0.352, -0.213] | **+0.040** [+0.026, +0.054] |
| NLL choice | +0.017 [+0.001, +0.035] | +0.017 [+0.001, +0.033] |
| NLL noul | **-0.468** [-0.583, -0.357] | +0.059 [+0.037, +0.079] |
| NLL score | +0.075 [-0.011, +0.150] | -0.050 [-0.070, -0.027] |
| top-1 | 0 (argmax preserved) | 0 |
| noul band | -0.107 [-0.130, -0.085] | **-0.395** [-0.429, -0.360] |
| I (slice proxy) | +1.56 [-0.45, +3.99] | **+17.28** [+15.20, +19.40] |
| C (slice proxy) | +0.09 [-2.65, +5.58] | -5.95 [-8.19, -1.32] |
| capability-like | +0.82 [-1.02, +4.31] | **+5.66** [+4.08, +8.41] |
| CompositeA-like | +4.83 [-0.76, +11.82] | **+40.30** [+31.20, +43.20] |

On the 840-row dev-card remainder the signs and significance are the same:
- shimcal - tfit: NLL -0.225 [-0.310, -0.153]; CompositeA-like +4.25 [-2.62, +12.20].
- shimcal - raw: CompositeA-like +38.26 [+28.59, +43.94].

**Reading:**
- **shimcal vs raw.** shimcal wins clearly on the Composite-relevant proxies, I +17 and CompositeA-like +40, because
  it removes the noul band, which jevkit scores as wrong. It pays a small NLL cost (+0.04) and C -6.
- **shimcal vs tfit.** shimcal is much better on NLL (-0.28). It is tied within the CIs on I, C and both composite
  proxies, and tfit is the biased arm.
- **Recommendation for a FUTURE cell:** pre-register native_shimcal (or an equivalent no-band readout fixed before the
  slice) as the primary readout instead of the per-type slice-fitted T map. The T_noul fallback (0.26) both
  over-sharpens yes/no answers, which hurts LTT, and wrecks NLL.
- This is a recommendation from diagnostic evidence, not a claim.

## Inputs

| file | sha256 | source |
|---|---|---|
| calib_probs.jsonl | ced3145c0c5c8eea... | stack/qwen35/jev3-lm @95a731f6 receipts/2026-10-12-jev3-lm-s-zbmhc/jevkit/ (= that receipt's SHA256SUMS) |
| cell.json | 85c9195b871f3f5b... | same receipt (= SHA256SUMS = landing.json) |
| slice_floor.json | d205bcab77b9f3bb... | same receipt (= SHA256SUMS) |
| tfit.json | e8c6a8d9d02ba4c9... | same receipt (= SHA256SUMS) |
| identity.json | (SHA256SUMS here) | same receipt: the native decider is ckpt-step0000150.pt (sha da7276b0); the trainable export is trainable-step0000150 (sha 8e936008) |

All files were taken from git at 95a731f6; no artifact tar was downloaded. Slice ids are 844233f2 (asserted).

## Reproduce

```
python -I -X utf8 receipts/2026-10-12-ev-guar-jev3-zbmhc/guarantees_jev3_zbmhc.py --repo . \
  --out guarantees_jev3_zbmhc.json --calib inputs/job-zbmhc/calib_probs.jsonl --cell inputs/job-zbmhc/cell.json \
  --floor inputs/job-zbmhc/slice_floor.json --tfit inputs/job-zbmhc/tfit.json \
  --shim <95a731f6>/third_party/h2o_lightning_shim/h2o_lightning_shim.py
python -I -X utf8 receipts/2026-10-12-ev-guar-jev3/tables_jev3.py guarantees_jev3_zbmhc.json > tables_jev3_zbmhc.md
python -I -X utf8 receipts/2026-10-12-ev-guar-jev3-zbmhc/readout_slice_zbmhc.py --repo . \
  --calib inputs/job-zbmhc/calib_probs.jsonl --cell inputs/job-zbmhc/cell.json --tfit inputs/job-zbmhc/tfit.json \
  --out readout_slice_zbmhc.json
```

# RESCORE_D19_CAL: independent rescore of the D19 calibrated variant (evidence lane, 2026-10-11)

Subject: the lead's D19 calibrated secondary row (PREREG `B-PREREG-D19-TEMP` + addendum 1, state commit ce26e16;
lead ledger row for job-jjpah / art-h2b72, receipt e3a19e93). Rule: the calibrated row is SECONDARY, never instead of
the untempered primary; the claim rule (B-PREREG-CLAIM + addenda) is unchanged.

**Overall verdict: AGREES with every number the lead reported. The full-DEV Index of the calibrated candidate (which
the lead could not compute) is 49.44 [47.72, 50.77]. That is 0.05 below the untempered 49.49, and it does not beat
either Clef. The global dev-pool T (0.806 < 1, which SHARPENS) makes calibration WORSE on full DEV (board ECE
0.0705 to 0.1235) and on JevBench (Calibration 74.01 to 68.28). Only the language area improves. Do not use this
calibrated variant in any claim.**

## Verdict table

| # | Quantity | Lead | Evidence (independent) | Verdict |
|---|---|---|---|---|
| 1 | T, primary (NLL minimiser over all 804 pool rows, bounds 0.25-4) | 0.806211 | 0.8062114 (own bisection on dNLL/dβ); 0.806211 (`devcal_eval fit` @ e6bfc4b) | AGREES (|Δ| < 1e-6) |
| 2 | T on the 200 dev-card rows / on the 604 rest | 0.8267 / 0.7998 | 0.82656 / 0.79981 | AGREES |
| 3 | Add-1(e) identity gate: 200 dev-card rows vs card-step0002800 (art-p3gpi cell.json: NLL 0.33762, acc 0.890) | NLL 0.3380, acc 0.890, PASS | NLL 0.33799 (dNLL +0.00038), acc 0.890 (dacc 0), PASS | AGREES |
| 4 | dev-804 NLL before to after | 0.3111 to 0.2985 | 0.31110 to 0.29853 | AGREES |
| 5 | dev-804 ECE15 (equal-width, top label) before to after | 0.0462 to 0.0329 | 0.04623 to 0.03289; argmax flips 0 | AGREES |
| 6 | Fit rows exclude DEV / proxy600 / JevBench | stated | 804 ids (12 sources) and 634 source-item tokens: 0 hits in the dev-manifest (1362c23e…), the test-manifest, proxy600 (dev_proxy600.json), the JevBench public pin and the fingerprints | AGREES |
| 7 | Full-DEV tempered rows | d6e4b10c…, 49,444 decisions (46,414 choice + 3,030 noul), 0 flips | ours 7c40c115… from b6bab803…: same 49,444, 0 flips; vs lead max \|Δp\| 5.8e-7, max \|Δnoul\| 1.0e-7, 0 argmax disagreements | AGREES (the sha differs only by float serialisation) |
| 8 | jevkit tempered preds | 229049b6… from a07aea11…, 0 flips | ours 0a70b884…: vs lead max \|Δp\| 1.9e-7, max \|Δscore\| 4.5e-8, 0 argmax disagreements | AGREES |
| 9 | jevkit untempered rescore | Composite A 62.25 | I 52.64, C 74.01, S 78.60, K 52.58, Capability 63.32, Composite A 62.25 [44.08, 64.86] (jevkit @ agent-main 7c131f19) | AGREES (exact) |
| 10 | jevkit calibrated | I 52.93, C 68.28, Cap 60.61, Composite A 61.27 [44.27, 64.61] | I 52.93 [44.59, 61.26], C 68.28 [53.72, 74.92], S 78.60, K 52.58, Capability 60.61, Composite A 61.27 [44.27, 64.61]; hard-choice ECE 0.182 to 0.230, noul abstention 16.2% to 13.5% | AGREES (exact): NEGATIVE |
| 11 | Full-DEV Index, calibrated (B-PREREG-CLAIM, land_fulldev, B 2000, seed 0) | not computed | **49.44 [47.72, 50.77]**; vs Clef 27B **−12.45 [−14.34, −10.87]**; vs Clef-flash **−7.85 [−9.91, −6.24]**; excl51 49.65 [47.90, 51.06], −12.41 [−14.38, −10.83] / −7.89 [−9.95, −6.19]; beats_both False / False | NEW (job-zyp3h) |
| 12 | Full-DEV Index, untempered (identity check in the same job) | 49.49 [47.77, 50.80], −12.40 / −7.80 | 49.49 [47.77, 50.80], −12.40 [−14.29, −10.82] / −7.80 [−9.84, −6.18]; excl51 49.70, −12.36 / −7.85 | AGREES (exact) |

## 1. T fit and validity

- Inputs: art-h2b72 (tar sha fa7ca530…, matches the ledger) extracted into a fresh directory; `out/devcal/pool/units-step0002800.json`
  (804 items, ids sha 283b2b9f… = the identity record) and `devcard/units-step0002800.json` (200).
- Two minimisers: `python -I -m jobs.m2_first_cell.devcal_eval fit` from claude/b-d19-devcal @ e6bfc4b (golden section
  on β = 1/T) gives T = 0.806211, usable = true. Ours is independent code (`rescore_cal.py`, about 15 lines: bisection on
  the exact gradient dNLL/dβ = mean[E_q log p − log p_y], q = softmax(β log p)) and gives T = 0.8062114. Agreement < 1e-6 (tolerance 0.005).
- Gate (pre-registered add-1(e)): the training card is read from art-p3gpi's own cell.json (`d19_land/dl/x/final/out/cell.json`,
  dev_cards step 2800: NLL 0.337618, acc 0.890), not from the devcal job's copy. The rescored dev-card units give NLL 0.337993
  and acc 0.890, which passes.
- Observation (not a gate failure): on the same 200 ids, the dev-card pass and the pool pass differ by up to |Δp| 0.042
  (batch-composition numerics). The pool subset gives NLL 0.33793, also inside the gate.
- Also from the lead's own fit code (T_devcal_eval.json; not in the lead's ledger note): at T, the pool's debiased-L2 ECE
  gets slightly WORSE (0.0370 to 0.0378) and so does the Brier score (0.15613 to 0.15658). Mean confidence moves from 0.867
  to 0.909 against an accuracy of 0.896, so the tempered model is over-confident on average. Only NLL and the 15-bin binned ECE improve.
- The dev-card ids sha in the identity record (57dd272c…) uses a different convention (record ids) from our sorted
  item-id sha (77942e83…). The subset relation was checked directly: all 200 ids are in the pool.

## 2. Tempering applied independently, and jevkit

Our own code applies softmax(log p / T) per decision: choice probabilities, noul as {p, 1−p}, and the score type as the
expected k of the tempered distribution. choice and confidence are re-derived from the argmax. Diffs against the lead's
files are in the table above (rows 7 and 8). jevkit was run from agent-main 7c131f19 (frozen scorer) with
`--price <job-9j3uu final/out/jevbench/price.json>` (size-class estimate $0.05 / $0.15 per M, the 13.3B class).

## 3. Full-DEV Decision Index (job-zyp3h, CPU-only on the SF box)

The laptop had 0.50 GB free RAM, and land_fulldev needs about 4 GB. So the landing ran as job `ev_cal_land` (claude/ev-d19-cal-rescore
@ 99bc675; python:3.12-slim + the pinned decision-index kit 87d4650b + numpy; no model). Context 5b8c4ed6… (54.7 MB),
inputs sha-pinned in `jobs/ev_cal_land/data.sha256` and verified in-job: DEV rows d1202ab8…, calibrated 7c40c115…, untempered
b6bab803…, Clef 27B job-zfexa f5ad788b…, Clef-flash job-t5yg2 2cb2e961…. The job ran 144 s and was billed $0.15 (hold $2.52).
Artifact art-u7rbx, tar sha 521fce98….

| system (full DEV 12,104) | Index [95% CI] | vs Clef 27B | vs Clef-flash | board ECE | ECE 10-bin |
|---|---|---|---|---|---|
| D19 calibrated (T 0.806) | 49.44 [47.72, 50.77] | −12.45 [−14.34, −10.87] | −7.85 [−9.91, −6.24] | 0.1235 | 0.0453 |
| D19 untempered (primary) | 49.49 [47.77, 50.80] | −12.40 [−14.29, −10.82] | −7.80 [−9.84, −6.18] | 0.0705 | 0.0406 |
| Clef 27B | 61.89 [60.28, 63.28] | | | 0.0403 | 0.0138 |
| Clef-flash | 57.29 [55.88, 58.67] | | | 0.0314 | 0.0235 |

full_excl51 (12,053 rows): calibrated 49.65 [47.90, 51.06], −12.41 [−14.38, −10.83] / −7.89 [−9.95, −6.19]. Untempered
49.70, −12.36 / −7.85. `beats_both` is False in every view. The Add-2 lineage variant is not recomputed: it cannot flip a
CI whose upper bound is −6.2.

### Full-DEV calibration per area (harness.calibration fields; ECE 10-bin equal width / debiased L2, 15 equal-mass bins)

| area | n | acc (both) | untempered ECE10 / deb15 | calibrated ECE10 / deb15 | Clef 27B ECE10 | Clef-flash ECE10 |
|---|---|---|---|---|---|---|
| overall | 31,754 | 0.8676 | 0.0406 / 0.0563 | 0.0453 / 0.0897 | 0.0138 | 0.0235 |
| arts | 2,045 | 0.6015 | 0.0581 / 0.0599 | 0.0949 / 0.1041 | 0.0425 | 0.0136 |
| knowledge | 4,532 | 0.6679 | 0.1326 / 0.1560 | 0.1759 / 0.2029 | 0.0534 | 0.0257 |
| language | 18,702 | 0.9570 | 0.0375 / 0.0379 | **0.0189** / 0.0397 | 0.0373 | 0.0424 |
| retrieval | 2,301 | 0.7558 | 0.0414 / 0.0552 | 0.1153 / 0.1320 | 0.0466 | 0.0725 |
| tools | 4,174 | 0.8761 | 0.0420 / 0.0515 | 0.0380 / 0.0756 | 0.0203 | 0.0237 |

Top-1 accuracy is identical by construction (0.86761 in both files). The dev pool is about 90% accurate, like the language
area, so its T sharpens: that helps language and hurts every lower-accuracy area. This is the per-area mismatch that
EV-PREREG-AREA-TEMP anticipates. A per-area T fitted on train-side rows is the registered next step. JevBench needs the
separately registered JevBench-side T (addendum /a).

## Receipts

`receipts/2026-10-10-ev-d19-cal-rescore/` (SHA256SUMS covers every file): rescore_cal.py, idcheck.py, fit_ours.json,
T_devcal_eval.json, fit_devcal.log, idcheck.txt, apply_dev.json, apply_jev.json, jev_preds.T.ours.jsonl,
jev_report.untempered.json, jev_report.T.ours.json, job-zyp3h/ (land_calibrated.json, land_untempered.json,
area_ece.json, preflight.json, data-verify.txt, run.log, SHA256SUMS, MANIFEST.jsonl, submit.json, CONTEXT_MANIFEST.json,
job-zyp3h.get.json with the signed URL redacted). The 20 MB tempered full-DEV file (7c40c115…) is shipped inside the
job-zyp3h context and was sha-verified in-job.

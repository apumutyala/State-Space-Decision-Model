# Conformal coverage + LTT abstention on our own trained stack (jev3 held-out slice)

Evidence lane, 2026-10-12. Governing rule: **B-PREREG-GUAR-JEV3** (origin/state PREREG.md, pushed at state `745fc478`
BEFORE any slice probability was downloaded or read). Cost $0 (CPU, laptop, ~30 s). One shot: no other seed, alpha,
epsilon, delta, grid or lambda0 was run.

## Verdict

- **Coverage certificate (split conformal): HOLDS on TEST for our stack.** JEV3-SYM seed 0 (native) under the
  pre-registered headline variant (Mondrian per type) covers 0.888 of TEST decisions at alpha 0.10 and 0.935 at
  alpha 0.05 (n 418). Neither is flagged; the flag line is 0.871 / 0.929. The sets are wide, though: mean size 1.62,
  singleton rate 0.476.
- **Abstention certificate (LTT, epsilon 0.05, delta 0.10): FALSIFIED for our stack on the jev3 distribution.** No
  lambda is certified for native or native_shimcal under any of these: alpha 0.10 or 0.05, variant M, T or F,
  the Bonferroni primary, the FS-CRR (c = 1, c = 8) and fixed-sequence-binomial secondaries, the per-type LTT, or the
  clean dev-card-remainder sensitivity. The certified act fraction is **0** and the stack abstains on everything.
- **Why it fails (measured on the LTT half, alpha 0.10, variant T):** our most confident decisions still err at about
  epsilon, and there are too few of them to certify anything.
  - Best lambda 0.95: 71 acts with 3 errors (4.2%). Binomial p 0.52 against a Bonferroni level of 9.9e-4.
    Certifying would take about 135 acts at 0 errors (Bonferroni) or 45 (a single test).
  - **noul hedging:** 48.7% of the slice's yes/no decisions sit in the 0.2-0.8 band. Only 23% reach p_max >= 0.95,
    and those err 6.4% of the time (above epsilon). At lambda 0.95, noul gives 54 acts and 3 errors.
  - **choice:** confident and correct, but rare. Only 12.9% reach p_max >= 0.95, all of them correct (17 acts, 0 errors).
  - **score:** no singletons at all. n_FIT is 16 < 100, so per the prereg score gets the full label set, and no score
    decision reaches 0.95.
  - native_shimcal removes the noul band (0%) but does not fix the errors: the best is lambda 0.98 with 69 acts and
    3 errors, and the FS-CRR stops at 0.95 with 96 acts and 7 errors.
- **Context (HYBRID H2O-Lightning-4B v1.1 host + SSDM head/data, on the DIFFERENT jev2 slice 142a593a):** both H2O-host
  cells DO get a non-trivial certificate.
  - x7c42 native: lambda 0.90, TEST act fraction 0.608, acted error 0.024.
  - q668q native: lambda 0.85, TEST act fraction 0.805, acted error 0.036.
  - These certify on the jev2 distribution, for a hybrid host. They say nothing about the jev3 distribution or about
    our stack.

## Headline table (variant T = Mondrian per type; TEST; all slice rows)

| source | arm | alpha | TEST cov (n) | mean size | singleton | certified lambda | TEST act | TEST acted err |
|---|---|---|---|---|---|---|---|---|
| **JEV3-SYM s0 (6zs79), jev3** | **native** | 0.10 | **0.888** (418) | 1.62 | 0.476 | **none** | **0** | - |
| JEV3-SYM s0 (6zs79), jev3 | native | 0.05 | 0.935 (418) | 1.87 | 0.323 | none | 0 | - |
| JEV3-SYM s0 (6zs79), jev3 | native_shimcal | 0.10 | 0.895 (418) | 1.65 | 0.462 | none | 0 | - |
| JEV3-SYM s0 (6zs79), jev3 | native_shimcal | 0.05 | 0.933 (418) | 1.85 | 0.335 | none | 0 | - |
| H2O x7c42 (hybrid), jev2 | native | 0.10 | 0.919 (1180) | 1.17 | 0.834 | 0.90 | 0.608 | 0.024 |
| H2O x7c42 (hybrid), jev2 | native | 0.05 | 0.954 (1180) | 1.37 | 0.706 | 0.90 | 0.608 | 0.024 |
| H2O x7c42 (hybrid), jev2 | native_shimcal | 0.10 | 0.920 (1180) | 1.18 | 0.832 | 0.95 | 0.601 | 0.021 |
| H2O x7c42 (hybrid), jev2 | native_shimcal | 0.05 | 0.954 (1180) | 1.37 | 0.711 | 0.95 | 0.601 | 0.021 |
| H2O-L q668q (hybrid), jev2 | native | 0.10 | 0.918 (1180) | 1.04 | 0.948 | 0.85 | 0.805 | 0.036 |
| H2O-L q668q (hybrid), jev2 | native | 0.05 | 0.956 (1180) | 1.18 | 0.844 | 0.85 | 0.805 | 0.036 |
| H2O-L q668q (hybrid), jev2 | native_shimcal | 0.10 | 0.925 (1180) | 1.06 | 0.930 | 0.92 | 0.797 | 0.032 |
| H2O-L q668q (hybrid), jev2 | native_shimcal | 0.05 | 0.956 (1180) | 1.19 | 0.843 | 0.92 | 0.797 | 0.032 |

The LTT step acts only on singletons with p_max >= lambda. Its lambda therefore barely depends on alpha in these data,
and it is identical across alpha for every arm above.

## Per type (headline arm: 6zs79 native, variant T, TEST)

| type | n TEST | cov a0.10 | size | singleton | singleton acc | cov a0.05 | size | singleton | act |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.74 | 0.350 | 0.854 | 0.934 | 2.13 | 0.248 | 0 |
| noul | 269 | 0.870 | 1.44 | 0.561 | 0.768 | 0.933 | 1.62 | 0.375 | 0 |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 1.000 | 4.42 | 0.000 | 0 |

- **score (pooled-small, full set):** score has n_FIT 16, below 100, so per the prereg it gets the full label set.
  Its coverage of 1.0 is therefore trivial.
- **Marginal variant M:** score is undercovered at 0.25 and FLAGGED at alpha 0.10. The marginal guarantee does not
  protect a small type. That is why per-type Mondrian is the headline.
- **Per family (6 families, ~67 FIT rows each):** every family is below MIN_N 100. Variant F therefore collapses to
  the marginal fit, and **no per-family guarantee exists ("insufficient")**.
  - Family TEST coverage under T at alpha 0.10 runs from 0.800 (judge_hard, n 80, FLAGGED) to 0.952 (ambiguous).
  - Under M, long_policy is 0.757 (FLAGGED).
  - This is conditional undercoverage. The marginal guarantee allows it, so it is not a violation.
- The full per type / per family tables for every arm, subset, alpha and variant are in `tables_jev3.md`. Every number
  is also in `guarantees_jev3.json`.

## Sensitivity: dev-card remainder (clean: selection never saw these rows)

job-5x4vr picked step 200 using dev-card NLL on 360 slice rows (cell.json `dev_card_sample`, ids_sha256 8f313487;
sorted-join sha b2db7f9b). On the other **840 rows** the model was fixed independently of FIT, LTT and TEST, and the u
assignment is unchanged.

| source | arm | alpha | TEST cov (n) | size | singleton | lambda | act | acted err |
|---|---|---|---|---|---|---|---|---|
| 6zs79 | native | 0.10 | 0.887 (301) | 1.66 | 0.465 | none | 0 | - |
| 6zs79 | native | 0.05 | 0.950 (301) | 1.92 | 0.306 | none | 0 | - |
| 6zs79 | native_shimcal | 0.10 | 0.890 (301) | 1.69 | 0.449 | none | 0 | - |
| x7c42 (hybrid, jev2) | native | 0.10 | 0.915 (1076) | 1.15 | 0.849 | 0.91 | 0.577 | 0.019 |
| q668q (hybrid, jev2) | native | 0.10 | 0.920 (1076) | 1.03 | 0.947 | 0.85 | 0.813 | 0.037 |

Same verdict. Reusing the slice for selection is not what decides the outcome either way.

## Secondaries (never verdict-bearing)

For our stack, every secondary certifies nothing:
- **FS-CRR from 0.95, c = 1** (the parent's answer-free rule gives c = 1 here, since every row is one decision): stops
  at lambda 0.95, "not rejected". native: 71 acts / 3 errors, p 0.45, log capital 0.79. native_shimcal: 96 / 7, p 0.53.
- **FS-CRR, c = 8** (the FSCRR literal value): p 0.90.
- **Fixed-sequence binomial from 0.95:** no lambda certified.
- **Per-type LTT** (choice and noul live; Bonferroni delta/(2 x 101)): no lambda for either type.

For the hybrid context cells, FS-CRR c = 1 certifies close to the primary:
- x7c42 native: lambda 0.90, act 0.608.
- q668q native: lambda 0.84, act 0.815.

## Method (as pre-registered)

- **Data.** calib_probs.jsonl holds the per-option untempered probabilities of every slice decision, written in-job
  by `fit_native_temperature`. Gold is the record's gold index.
  - Asserted: decision ids are unique, there is one decision per record id, the probabilities sum to 1, and the
    record-id sha equals the pin.
  - Asserted: argmax top-1 equals the in-job slice_floor native_top1 exactly (0.678333 / 0.846111 / 0.890000).
- **Score.** jevkit LAC, s = 1 - p_gold. jevkit has no APS.
- **Split.** u = int(sha256("B-PREREG-GUAR-JEV3|v1|" + decision_id)[:12], 16) / 16^12. u < 1/3 goes to FIT,
  u < 2/3 to LTT, the rest to TEST. The split is by id only and is the same for every arm.
  - 6zs79: 376 / 406 / 418.
  - jev2 sources: 1,203 / 1,217 / 1,180.
- **Conformal.** jevkit `fit_conformal` on FIT, in three variants: M (marginal), T (per type) and F (per family).
  - A group with n_FIT < 100 goes into pooled-small.
  - If pooled-small itself has n_FIT < 100, it gets the full label set.
- **LTT.** jevkit `ltt_select` default: per-decision exact binomial, Bonferroni delta/101 over DEFAULT_LAMBDAS, rule
  "act iff |C| = 1 and p_max >= lambda". Rows are i.i.d. with one decision each, so the binomial is exact.
  - Secondaries use the opt-in `method="fixed_sequence", fs_start=0.95, cluster_test="crr"` modes.
- **TEST flag.** coverage < 1 - alpha - 2 sqrt(alpha (1 - alpha) / n).
  - 2 of the 72 all-decisions cells (3 sources x 2 arms x 2 subsets x 2 alpha x 3 variants) are flagged: 6zs79 native_shimcal,
    alpha 0.05, M and F, 0.926 vs 0.929.
  - The T variant is never flagged at the all-decisions level.
- **native_shimcal.** The vendored H2O shim `format_answer` (T 0.8, yes/no commit floor 0.801) applied offline to the
  same probabilities, matching screen.py `shim_calibrate`.
  - Reimplemented in the driver and checked against the vendored module (jev3-sym `third_party/h2o_lightning_shim/
    h2o_lightning_shim.py`, sha 355791c3) on every row of all three sources: max |d| = 0.0.

## Honest caveats (apply to every number)

1. **The slice is not fully unseen.** It was used to select the checkpoint (5x4vr step 200, by dev-card NLL on 360
   slice rows) and for the T fit.
   - T stayed 1.0, so native = native_tfit.
   - A T fitted on all slice rows is a symmetric function of the data, so it preserves exchangeability.
   - FIT, LTT and TEST are exchangeable: same slice, random split by id.
   - The remainder sensitivity removes the selection reuse, and the verdict is unchanged.
2. **This is a statement about jev3-distribution decisions only.** The jev3 slice differs from DEV and from public
   JevBench. Nothing here transfers to DEV, JevBench or the board under shift.
   - The H2O context cells are on the jev2 slice (another distribution) and use a hybrid host.
3. **No comparison with Clef.** There are no Clef predictions on jev3, and the operator directive forbids new benchmark
   runs of existing models.
4. **Gold is the in-job label** carried in calib_probs. The slice records are not local, so gold was not re-read
   independently. x7c42's rescore matched calib_probs gold to the local jev2 records 3,600/3,600.
5. **Single seed, single checkpoint.** About 400 rows per third, so the type and family cells are small.
6. **Not a board claim.**

## Inputs (downloaded by me via the job records' signed URLs; tars deleted after extraction)

| source | artifact tar sha256 (= job record) | final.tgz (= MANIFEST) | calib_probs.jsonl | cell.json | scored ckpt |
|---|---|---|---|---|---|
| job-6zs79 (art-xkhwy) | 146a6996c675d6cd... | 0b16d6ed | d9b04bf3 | 03ff0ba0 | job-5x4vr trainable-step0000200 **5cb7da42** (checkpoint.json role best; art-8xs32 933cc0ac) |
| job-x7c42 (art-cwk49) | 0c538a615ef838f7... | deb9ce22 | 2f53e661 | 76eda3ab (= ev-x7c42-rescore cell.in-job.json) | step 200, 3c04939e (latest.json best) |
| job-q668q (art-5si2g) | 912871774226f93d... | 0a9924c9 | c0c21b23 | d6e35025 | step 750, 5f130a5b (latest.json best) |

- Slice ids: jev3 is **844233f2**28b718ed... (1,200 rows); jev2 is 142a593a... (3,600 rows).
- jevkit/guarantees.py sha c8970343 is claude/ev-guarantee-proofs @a06343b6 plus the cherry-picked FS-CRR commit
  (claude/ev-guar-fscrr 2d25520e). It is **unmodified** by this work. tests/guarantees: 44 passed, 1 skipped.
- `inputs/job-6zs79/` holds the primary calib_probs.jsonl (committed), plus slice_floor, tfit, checkpoint.json and the
  MANIFEST.
- The jev2 calib_probs are not committed. Their shas are in `inputs/job-*/inner_sha256.txt`, and they can be
  re-extracted from the artifacts.

## Reproduce

```
python -I -X utf8 receipts/2026-10-12-ev-guar-jev3/guarantees_jev3.py --repo . --out guarantees_jev3.json \
  --shim <jev3-sym>/third_party/h2o_lightning_shim/h2o_lightning_shim.py \
  --source "jev3sym_6zs79=<calib_probs.jsonl>::<cell.json>::844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078::<slice_floor.json>" \
  --source "h2o_x7c42=...::142a593a1257d6729b5480fe67647b1af0c5678d2a929654f6325e128fce00be::..." \
  --source "h2o_q668q=...::142a593a1257d6729b5480fe67647b1af0c5678d2a929654f6325e128fce00be::..."
python -I -X utf8 receipts/2026-10-12-ev-guar-jev3/tables_jev3.py guarantees_jev3.json > tables_jev3.md
```

## What would change it

The failure has a measured cause: confident yes/no answers are wrong at about epsilon, and confident choice/score
answers are rare. The fix is on the model side, not in the statistics:
- **noul confidence:** a lead-lane change, for example the typed/LM cells, that drives confident yes/no error well
  below 5%.
- **More confident choice/score decisions.**
- **Evaluating any such cell:** re-run this exact pipeline on its slice calib_probs. It is pure CPU.

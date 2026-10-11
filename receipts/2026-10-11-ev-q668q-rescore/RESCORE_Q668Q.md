# Independent rescore: JEV2-H2O-L = job-q668q (A-PREREG-JEV2-H2O-L + the Holm line of A-PREREG-JEV2-H2O amendment 3)

Evidence lane, 2026-10-11. Lead landing: stack/qwen35/jev2-h2o @d13d941a (receipts/2026-10-11-jev2-h2o-L-q668q/), package @1e12ec12.
Label: "H2O-Lightning-4B v1.1 + SSDM jev2 data (L)". Caveat: open public items only; the board's sealed split may rank
differently (open-to-board I gap measured on H2O: -17.6). Never a board score. Template: claude/ev-x7c42-rescore @bf706854.

## Verdicts
| question | verdict |
|---|---|
| PRIMARY (L alone) | **NOT MET.** Δ Composite A (native_shimcal − in-job H2O v1.1 control) = **−0.969**, 95% CI **[−3.375, +1.124]**, 97.5% CI **[−3.735, +1.409]**, one-sided bootstrap p(Δ ≤ 0) = **0.810** (0.810 with +1). |
| HOLM over x7c42 + L | **NOT CLAIMED.** p: x7c42 0.565 (Δ −0.409), L 0.810 (Δ −0.969). The smaller p is x7c42's at 0.565, which is not < 0.025 (its 97.5% CI [−2.06, +1.58] includes 0), so the procedure stops and neither arm is tested further. **"H2O-Lightning-4B v1.1 + SSDM jev2 data beats H2O v1.1 on local Composite A" is NOT CLAIMED**, in any order: L would also fail at 0.05 on its own (95% CI includes 0, and the point is negative). Caveat: each arm is a single run. Run-to-run GPU nondeterminism is not captured; x7c42 showed training diverging from step 1 under the same seed (vs job-9fy9p), and this rescore finds the same untrained control drifting between jobs (below). |
| Number agreement with the job | **AGREES.** 95 fields of screen.json rescored from raw preds with the frozen jevkit CLI: max \|Δ\| = 1.4e-14, 0 flags (tolerance 0.05). The lead's README / ledger numbers agree (97.5% CI [−3.73, +1.41]; per-type ΔI −3.42 / −0.63 / −4.90; transfer correlation −0.32 vs x7c42 +0.15). |
| Step-0 floor | **floor beaten.** Slice top-1 89.22% vs 80.25% (3,600 decisions, 1 per id). Δ = **+8.97 pp**, 95% CI [+7.61, +10.19] (iid) / [+7.72, +10.22] (family-stratified). Exact McNemar p = 7.9e-45 (441 vs 118 discordant). The lead's "0.890 vs 0.800" is rounding (0.8922 / 0.8025); CI [+7.69, +10.22] is a different bootstrap, both exclude 0. |
| Selection | **AGREES. Step 750 selected and scored** (dev-card NLL argmin 0.4031; cell.json best = step 750; identity-child.json checkpoint ckpt-step0000750.pt sha d2c99e3d…; exported trainable-step0000750.safetensors 5f130a5b…). |
| Order of events | **Selection BEFORE public: AGREES** (cell.json 18:27:42Z < first public-231 pred native.preds 18:32:58Z). **slice_floor BEFORE public: DISAGREES, as in x7c42** (calib_probs / tfit 18:38:35Z, slice_floor 18:42:49Z, after native 18:32:58Z and stock_shim 18:38:52Z). Not a breach: the floor is a fixed pass/fail on an already frozen checkpoint vs the untrained host and cannot change the selection. |
| Host / integrity | **AGREES** on every pinned item. Control vs x7c42's control: **NOT bitwise identical** (same code path; see (2)). |

## (1) PRIMARY: recomputed from raw preds (B 1000, seed 16, stock_shim strata, Speed/Cost held at measured values)
| quantity | lead / in-job | rescore | verdict |
|---|---|---|---|
| Comp A native_shimcal / control | 75.690 / 76.659 | 75.690 / 76.659 | AGREES |
| Δ Comp A [95%] | −0.97 [−3.38, +1.12] | −0.9689 [−3.3753, +1.1236] | AGREES |
| Δ Comp A [97.5%, Holm] | [−3.73, +1.41] | [−3.7345, +1.4092] | AGREES |
| one-sided p | – | 0.810 | new |
| Δ I / Δ C / Δ Cap | −2.98 [−8.26, +1.89] / −0.59 / −1.79 | −2.981 [−8.263, +1.894] / −0.594 [−7.129, +5.663] / −1.788 [−6.890, +2.702] | AGREES |
| S / K | – | 90.64 vs 91.19 (−0.55, latency, held fixed) / 61.23 both | – |
| per-type ΔI (cc) | choice −3.42, noul −0.63, score −4.90 | −3.419 [−9.16, +2.28], −0.627 [−12.03, +10.15], −4.897 [−15.02, +2.01] | AGREES (points) |
| raw `native` Δ A | – | −3.388 [−6.258, −1.568]; 97.5% [−6.82, −1.28] | in screen.json, AGREES |
| `native_tfit` Δ A | – | −2.790 [−5.335, −0.904]; T 0.914 fitted on slice calib_probs only (not at bound, not band-constrained) | AGREES |

## (2) Host / integrity
- **Artifact.** Downloaded myself via the job record (`autor job get job-q668q`). art-5si2g is a 130,908,160-byte tar, sha256 `91287177…bcf38`, equal to the job record's artifact.sha256. All six phase tgz shas equal MANIFEST.jsonl. Arm preds: native 01dde99c…, native_shimcal 31f23ff6…, native_tfit da5cc5f7…, stock_shim fe91cb41…; slice_floor_rows 8bae1242…, calib_probs c0c21b23…, cell.json d6e35025…, screen.json b86430f2….
- **rev 193ad740 loaded: AGREES.** run.log `HOST h2oai/h2o-lightning-4b@193ad740…`; host-verify revision = resolved_sha = 193ad740925b…; snapshot-verify 10/10 OK (incl. model.safetensors); identity weights c65443b1…, missing_text_keys 0. Package files pinned in CONTEXT_MANIFEST at m1_sha 1e12ec12.
- **Licence copied: AGREES.** out/license/LICENSE bbedc3fd…f1e57a and README.md 614272b0…, both equal host-verify.
- **Exclusion lists: AGREES.** CONTEXT_MANIFEST pins mix_exclude_ids c0c175f5… and musique_provenance_unverified 5cebf60b…; the local draw-report gives `global_exclude n_ids 127, found 127` (23 + 104).
- **prob_indep dropped: AGREES.** draw-report `excluded_subs {prob_indep: 124}`; rescan: 0 prob_indep rows in train (17,959) and slice (3,600).
- **Records = in-job pins: AGREES.** Local train records uncompressed sha ee3f6d28… = cell.json; the slice records file b40cac9c… is **byte-identical to JEV2-H2O's (x7c42) slice** (ids and canonical records equal), as the prereg requires.
- **max_len 4,096, 0 truncation: AGREES.** Train: `no_truncation_tokenize rows 17959, max_row_tokens 3980, t_cap 4096` (fail-closed). Public 231: max input_tokens 3,952, 0 over 4,096.
- **Train draw ids:** sorted-newline sha **7763e55e…a363** (= draw-report train_ids_sha256); in-job drawn_ids_sha256 convention 5c84ca2b…. Slice ids 142a593a…00be (sorted newline; e09721b7… with trailing newline = the forced-calib file sha).
- **Disjointness slice vs train: AGREES** (0 ids, 0 base ids, 0 source items, 0 content = state + questions). **Flag (not a breach under the prereg's definition): 49 slice rows share an identical STATE with a train row** (x7c42's 2,664-row draw: 3), questions differ in all 49, the train row's gold equals the slice gold in 29. 24 are glaive_fc tool picks (same short user request, different catalog: glaive duplicates under different source ids that `--dedupe-source glaive_fc` did not merge), 12 prob_dice, 11 jev2 tool_* and 2 others. Floor sensitivity: without these 49 rows Δ = **+9.01 pp [+7.74, +10.36]** (step 0 0.8015 → 0.8916). On the 49 rows themselves: 0.878 → 0.939. Recommend a state-level dedupe for glaive_fc / prob_dice in the jev2 generator.
- **Slice or train vs public-231: 0 ids, 0 identical states, 0 identical state + question.**
- **Control vs x7c42's control (sha 3555841e): NOT bitwise.** Same code (git diff 48069603..1e12ec12 touches no code: params, pointer, ids, spec, test only) and the same identity (lm_label path, LoRA scaling 0, tied embed fp32 label rows, T 0.8, noul floor 0.801). Only **57/231 answers are bitwise equal; top-1 is identical on 231/231 (205 = 205)**; max \|dp\| choice 0.0189 (2 items > 0.01), noul 0.0075, score 0.0074. Against job-4g6m8 (e00d5ac7) and job-zbtbj (60a06842) this control agrees to **max \|dp\| 1.2e-6** (34/231 bitwise; I 77.5604 vs 77.5604, C 83.4192 vs 83.4192). So q668q's control reproduces the board-anchored H2O runs, and **x7c42's control is the drifted one** (I 77.59, C 82.85, A 76.57 vs 76.66 here). The drift is unstructured: the per-option logit ratio x7c42 / 4g6m8 has median slope 1.0000 (p10–p90 0.995–1.005), so it is not a temperature or scale leak; it is float-level run-to-run nondeterminism in the screen path. The same happens on the slice: the step-0 dev card (cell path) is bitwise equal to x7c42's (NLL 0.490837, 295/360), but the screen's step-0 slice column differs on 7/3,600 decisions (2,889 vs 2,886 correct). **Both primaries are paired in-job, so each is valid**; cross-job comparisons of arms (x7c42 76.16 vs L 75.69) carry ~0.1 A of control drift.

## (3) Selection (slice 142a593a, 360 hash-sampled dev rows, sample ids sha b7dd0844… = x7c42's)
| step | 0 | 250 | 500 | **750** | 1000 | 1250 | 1500 | 1750 |
|---|---|---|---|---|---|---|---|---|
| dev NLL | 0.4908 | 0.4579 | 0.4398 | **0.4031** | 0.4056 | 0.4045 | 0.4096 | 0.4173 |
| dev top-1 (/360) | 295 | 289 | 299 | **308** | 310 | 308 | 310 | 308 |

- Selected by NLL: step 750 (top-1 peaks at 1000/1500 with 310, not the registered criterion). No collapse events.
- **Training length: 1,845 of the planned 2,244 steps.** The step-10 rate check measured 3.93 s/step (vs 2.55 in 9fy9p) and shortened the plan to 1,832 (pre-registered rule, min 1,000); cosine horizon fixed at 1,231. 1,845 steps × 8 = 14,760 rows ≈ **0.82 epoch**; **the scored checkpoint (step 750) saw 6,000 rows ≈ 0.33 epoch.** "~1 epoch" describes the run, not the scored model.
- Config differs from x7c42 only in data / budget keys (train_set, n_train_records, max/min steps, max_minutes, ckpt/dev-card cadence 250, calib_slice / dev_card_records name, notes).
- Dev-sample cross-check: selected 0.8556 and step 0 0.8194 on the 360 dev ids inside slice_floor_rows = the dev cards exactly.

## (4) Step-0 floor (3,600 rows, 1 decision per id → no clustering needed)
- Selected-arm column recomputed from calib_probs.jsonl (argmax vs gold): 3,600/3,600 match; gold = local slice records 3,600/3,600. Per-family slice Δ equal the in-job slice_floor.per_family exactly (max \|Δ\| 0).
- Per type: choice +6.18 [+4.72, +7.70] (McN 1e-14), noul +4.97 [+2.89, +7.14] (2e-5), score **+20.80 [+17.59, +24.01]** (2e-34).

## (5) Secondaries (EV-PREREG-SECONDARY-AXES; native_shimcal − control; the failed primary stays failed)
- Per axis: ΔI −2.98 [−8.26, +1.89]; ΔC −0.59 [−7.13, +5.66]; ΔS −0.55 (fixed); ΔK 0 (fixed); ΔCap −1.79 [−6.89, +2.70].
- Top-1 202 vs 205: Δ −1.30 pp [−3.90, +1.30]; 4 arm-only vs 7 control-only; exact McNemar p 0.549. Per type: choice 121 vs 124 (2/5, p 0.45), noul 66 vs 66 (2/2), score 15 vs 15.
- Choice NLL 0.3568 vs 0.3068, Δ **+0.050** [−0.015, +0.119] (x7c42: +0.007).
- I < 50 in 0% of resamples for both arms; Δ A < −10 in 0%.
- Comp A quantiles (1/2.5/5/25/50/75/95/97.5/99): arm 70.72/71.30/71.96/73.86/75.04/76.23/77.72/78.07/78.39; control 72.37/72.85/73.61/75.16/76.10/76.96/78.06/78.32/78.69; Δ −3.76/−3.38/−3.00/−1.77/−1.02/−0.21/+0.78/+1.12/+1.47.
- Noul 0.2–0.8 band: control 0/74, native_shimcal 0/74, raw native **16/74**, native_tfit 13/74 (the reported band gate fails for the raw arm, as in x7c42). Slice band rate 0.184 at T 1, 0.172 at T 0.914.
- Raw arms: native A 73.27 (I 64.56, C 85.20), Δ A −3.39; native_tfit A 73.87, Δ A −2.79. The shim calibration (T 0.8 + noul floor) carries native_shimcal's I (+10.0 over raw native).

## (6) TRANSFER (diagnosis only; nothing fitted to public items)
Slice = selected − step 0 on the held-out slice (200 per family). Public = native_shimcal − control on the 231 items. x7c42 columns from claude/ev-x7c42-rescore @bf706854 rescore-extra.json.
| family | slice step-0 | L slice Δ pp [CI] (McN) | x7c42 slice Δ | public n / control top-1 | L public Δtop-1 (+/−) | L public ΔI | x7c42 public Δtop-1 / ΔI | flag |
|---|---|---|---|---|---|---|---|---|
| adversarial | 67.5 | **+30.5 [+24.0, +36.0]** (9e-19) | +14.5 | 6 / 100 | 0 | +0.9 | 0 / +0.6 | gain slice only |
| tradeoff | 55.0 | **+20.0 [+13.0, +27.5]** (5e-7) | +15.5 | 6 / 83.3 | **−33.3 (0/2)** | **−65.9** | 0 / 0 | public loss |
| severity | 74.5 | **+17.0 [+11.5, +23.0]** (6e-8) | +3.5 | – | – | – | – | not on public |
| trap | 84.0 | **+14.5 [+10.0, +19.5]** (3e-8) | +10.0 | 8 / 100 | 0 | +1.5 | 0 / +5.9 | gain slice only |
| probability | 70.0 | **+14.0 [+7.5, +21.0]** (4e-5) | +3.5 | 10 / 100 | **−20.0 (0/2)** | −21.4 | −10.0 / −10.7 | public loss |
| adequacy | 70.5 | **+13.5 [+8.0, +19.0]** (3e-6) | +8.5 | 12 / 91.7 | +8.3 (1/0) | +16.7 | 0 / 0 | gain on both (1 item) |
| temporal_numeric | 58.0 | **+12.5 [+5.5, +19.0]** (5e-4) | +3.0 (n.s.) | 15 / **33.3** | 0 (1/1) | −3.9 | 0 / +5.0 | gain slice only |
| policy | 74.0 | **+11.5 [+5.5, +17.5]** (2e-4) | +7.0 | 12 / 100 | 0 | 0 | 0 / 0 | gain slice only |
| multi_hop | 81.0 | **+9.0 [+5.0, +13.5]** (1e-4) | +3.5 | 18 / 83.3 | −5.6 (0/1) | −8.3 | +5.6 / −1.9 | public loss (1 item) |
| ambiguous | 91.5 | **+8.0 [+4.5, +12.0]** (1e-4) | +8.0 | 7 / 85.7 | +14.3 (1/0) | +21.1 | 0 / 0 | gain on both (1 item) |
| state_tracking | 84.0 | **+6.5 [+2.5, +11.0]** (0.007) | −0.5 | – | – | – | – | not on public |
| routing | 84.0 | +3.5 [−2.0, +9.0] | +2.0 | 12 / 100 | 0 | 0 | 0 / 0 | |
| long_policy | 86.0 | +3.0 [−2.5, +8.0] (n.s.) | +1.5 (n.s.) | 19 / **73.7** | 0 (0/0) | −7.9 | 0 / +2.2 | |
| judge_hard | 69.5 | −0.5 [−8.5, +7.5] (n.s.) | 0.0 | 17 / **70.6** | 0 (1/1) | 0 | −5.9 / −11.8 | |
| tool_selection / enum | 99.5 / 99.5 | +0.5 / 0 | 0 / 0 | 12 / 100, – | 0 | 0 | 0 | |
| fact / intent | 98.5 / 97.5 | −0.5 / −1.5 | −0.5 / +0.5 | 12 / 100, 24 / 100 | 0 | 0 | 0 | |
| extraction / ordinal / routing_hard | – | not in the slice | – | 24 / 100, 12 / 100, 5 / 100 | 0 | 0 / +5.6 / 0 | 0 / +4.5 / 0 | |

By type: slice choice +6.18, noul +4.97, score +20.80 pp (all CIs > 0). Public top-1: choice −2.16 (2/5), noul 0 (2/2), score 0 (0/0). Public cc: choice −3.42, noul −0.63, score −4.90 [−15.02, +2.01]. Unlike x7c42 (score +6.08 on public), **no type's gain transfers.** Slice-vs-public Δtop-1 correlation over the 15 shared families: **−0.32** (x7c42 +0.15).

**Control error locations (26/231, identical to x7c42's control by top-1):** temporal_numeric 10, judge_hard 5, long_policy 5, multi_hop 3, ambiguous 1, adequacy 1, tradeoff 1.

**Reading.**
1. **One (partial) epoch moved temporal_numeric on the slice and nothing else of the three.** temporal_numeric: +12.5 pp [+5.5, +19.0] (x7c42 +3.0, n.s.). long_policy: +3.0, still n.s. (x7c42 +1.5). judge_hard: −0.5, flat (x7c42 0.0).
2. **None of it reached public.** temporal_numeric public top-1 stays 5/15 (1 fixed, 1 broken; ΔI −3.9); judge_hard 12/17 (1/1); long_policy 14/19 (0/0, ΔI −7.9). The 26 control errors sit in the same places, and L fixes 4 of them (adequacy, ambiguous, temporal_numeric, judge_hard, 1 each) while breaking 7 previously correct items (tradeoff 2, probability 2, multi_hop 1, temporal_numeric 1, judge_hard 1).
3. **Bigger slice gains, bigger public losses.** The slice gains roughly doubled vs x7c42 (adversarial +30.5, tradeoff +20.0, probability +14.0, severity +17.0), but the families with the largest slice gains are the ones losing public items (tradeoff −2, probability −2): consistent with x7c42's shape-shift reading (public states 2–6x longer and a different type / K mix in these families; rescore-extra.json `transfer_by_family.shape_*`). The slice is in-distribution for training, not for public.

## Files (SHA256SUMS in this directory)
- Scripts: scripts/ev_rescore_jevkit.py (unchanged from @536fae54) and scripts/ev_rescore_q668q.py (new; derived from ev_rescore_x7c42.py: reports dir arg, 3-way control identity, selection block, x7c42 calib identity, x7c42 transfer columns + correlation).
- In this directory: check_integrity_q668q.py and check_state_overlap_floor.py, each with its .out.txt.
- Outputs: rescore.json, rescore.stdout.txt, rescore.stderr.txt (empty), rescore-extra.json, reports/*.rescore.jevkit.json, price.json (= the in-job declared price; byte-equal to x7c42's).
- In-job copies: screen.in-job.json, cell.in-job.json, slice_floor.in-job.json, tfit.in-job.json, host-verify.in-job.json, MANIFEST.in-job.jsonl, job-records.json (q668q, 4g6m8, zbtbj; signed URLs redacted).
- Artifact shas: q668q 91287177…bcf38; 4g6m8 8feba886…5181; zbtbj d0a04884…8acd. Comparators: x7c42 stock_shim preds 3555841e…, slice_floor_rows d1bf3590…, cell.json 76eda3ab… (stack @c5a2fb33).

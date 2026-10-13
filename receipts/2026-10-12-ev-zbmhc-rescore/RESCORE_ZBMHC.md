# Independent rescore: job-zbmhc (JEV3-LM-S, A-PREREG-JEV3-LM Amendment 2)

Evidence lane, 2026-10-12. Kit: `scripts/ev_cell_rescore.py` + `scripts/ev_score_flip.py` + `scripts/ev_score_perturb.py` at
claude/ev-score-inv @c47c0a46 (this branch is cut from it); canonical order from evidence's own
`gold_letter_canonical.py` (claude/ev-canonical-prior @1ccdc676), not the lead's `decide/canon_shim.py`.
Governing rules (origin/state PREREG.md): A-PREREG-JEV3-LM + Amendment 1 (93f0e4b2) + Amendment 2 (64da4631),
EV-PREREG-COMPOSITE-LOCAL, EV-PREREG-SECONDARY-AXES.

**PRIMARY verdict: NOT CLAIMED (CI straddles 0). The rescore AGREES with the lead.** Paired delta Composite A of
native_tfit minus the NON-canonical in-job stock+shim control = **-0.78 [-16.36, +3.47]** (66.97 vs 67.74), one-sided
p 0.736; Holm does not allow a claim. Per the prereg: "not claimed" + the point estimate; no follow-up tuning on public-231.
**KEEP gate 1 (G1 bitwise): PASS. AGREES.** **KEEP gate 2 (score op @c47c0a46): PASS, bitwise. AGREES** (re-run by me).
**Letter shortcut: no FLAG. AGREES.** **Single seed, replication pending.**
**One disagreement in interpretation (not in a number): the T_noul fallback did not cause the primary's loss; it
saved ~29 Composite points relative to T_noul = 1 (diagnostic, section 5).**

## Inputs (downloaded by me and checked)
| item | sha256 | check |
|---|---|---|
| job-zbmhc artifact art-mddiz.tar (131,788,800 B), via `autor job get` + the job's download URL | 6b69fc5cc543b44f... | = job record artifact.sha256 |
| phases setup/guard/heldout/cell/screen/final.tgz | ddd78f45 / 26bd2ba8 / a2ad5492 / 55590d1f / e12f0e8a / 10bed5dd | = MANIFEST.jsonl |
| lead receipts @95a731f6 receipts/2026-10-12-jev3-lm-s-zbmhc (51 files in its SHA256SUMS, excl. landing.json) | | all 51 byte-identical to the artifact's final.tgz out/ files; landing.json 30de29ec = artifact |
| screen.json | 703e037c | kit diff: 124 fields, max abs d 0.0 |
| scored checkpoint: ckpt-step0000150.pt | da7276b0b44ecbdd... | = cell.json scored_checkpoint = identity-child.json checkpoint = trainable-step0000150.json source_ckpt_sha256 |
| exported step-150 weights trainable-step0000150.safetensors (in tar) | 8e9360083ad6b6ea... | my hash of the tar member = .sha256 sidecar = latest.json best = cell.json best_export (sidecar json b49f81dd) |
| H2O target preds (job-4g6m8), my copy from claude/lead-h2o-eval @f5615683 | e00d5ac7 | = the in-artifact target_ref.preds.jsonl |
| price (Qwen3.5-4B deepinfra 0.03/0.15), from the 6zs79 rescore inputs | c3fd78b3 | = in-job price |
| flip baseline claude/ev-flip-rate run2/flip_ci_8op.json | c7f7c930 | qwen35_4b_shim 7.55% |
| slice records ssdm-data/m2/build/bschema-draw-jev3-slice/records.jsonl.gz | ids (sorted, newline-joined) 844233f2 | = pre-registered slice sha (1,200 records) |

Primary command (secondaries: the same with `--primary native_shimcal | canon_stock | native`, no g1/flip8):
```
python -I -X utf8 scripts/ev_cell_rescore.py --art <final.tgz extracted> --control stock_shim --primary native_tfit \
  --out primary_native_tfit --price inputs/price_qwen35_4b.json \
  --slice-ids-sha 844233f228b718edeffb198f622fc9c8a605eed3a346c128993f8aac80453078 \
  --target "h2o=inputs/h2o_preds.jsonl::inputs/price_qwen35_4b.json" --mtimes inputs/mtimes_zbmhc.json \
  --identity-child <art>/out/screen/identity-child.json --g1 native=<art>/.../g1/native --g1 stock_shim=<art>/.../g1/stock_shim \
  --g1-max-dp 0 --flip8 <art>/.../flip8 --flip-baseline-receipt "inputs/flip_ci_8op.json::qwen35_4b_shim" --expect-steps 200
```
The kit's `--score-inv` section needs base/perturbed preds next to base.jsonl; the job ships them per arm
(score_inv/native/, score_inv/stock_shim/), so the score op was run directly with `ev_score_flip.py` (section 3).
Kit flags (benign, as in 6zs79): `target_ref` is not in screen.json arms (it is the H2O file, sha e00d5ac7 = my copy).

## Lead's claims, number by number
| claim (lead) | rescore | verdict |
|---|---|---|
| PRIMARY native_tfit vs non-canonical stock+shim dCompA -0.78 [-16.36, +3.47], not claimed | -0.7758 [-16.3645, +3.4737]; 97.5% [-18.43, +5.00]; p 0.736; Holm not claimable | **AGREES** |
| Comp A 66.97 vs 67.74 | 66.967 vs 67.743 | **AGREES** |
| G1: 565 pairs, 0 flips, bitwise | reverse 213 + shuffle 213 + relabel 139 = 565; 0 flips, 0 invalid, max abs dp 0.0 | **AGREES** |
| score op @c47c0a46: 0 flips on reverse 18 / shuffle 54 / relabel 18 / permute 18 | my `ev_score_flip.py --max-dp 0` re-run: n 18/54/18/18, 0 invalid, 0 flips, max abs dp 0.0, max abs dE 0.0; exit 0 | **AGREES** |
| score-op control fails 2/8/3/4 | stock_shim: 2/8/3/4 flips, max abs dp 0.39/0.39/0.50/0.47; exit 3 | **AGREES** |
| A-share slice -4.1 pp, public -0.7 pp, no flag | slice choice pred-A 34.38% vs gold-A 38.46% = -4.09 pp; public 23.74% vs 24.46% = -0.72 pp | **AGREES** |
| 8-op flips 3.64% avg (4.14% pooled) vs 7.55% | avg 3.64% [2.65, 4.78]; pooled 66/1,593 = 4.14%; per op state 1.0, instr 4.3, option_para 6.1, reverse/shuffle/relabel 0, whitespace 9.1, markup 8.7 | **AGREES** |
| slice floor +4.5 pp [+1.9, +7.1] | +4.50 pp (0.7117 vs 0.6667, n 1,200; 151 vs 97 discordant, McNemar p 0.0007); iid [+2.16, +7.00], family-stratified [+1.92, +7.00] | **AGREES** on point / lower bound; upper +7.0 vs +7.1 (<= 0.1 pt, scheme unstated) |
| best step 150 (NLL 0.577), not the last of 200 | dev cards (360 slice rows): 0.7413 / 0.6424 / 0.6183 / **0.5773** / 0.5811 at 0/50/100/150/200; argmin = 150 = scored | **AGREES** |
| order: slice_first 23:04:00-23:07:12Z < first public pred 23:07:31Z | slice_first.json 23:04:00-23:07:12Z; first public pred native.preds.jsonl mtime 23:07:31Z (no public preds file earlier) | **AGREES** |
| native_shimcal +3.48 [-1.09, +12.56] (71.22) | +3.4757 [-1.0936, +12.5607]; 71.219; p 0.087 | **AGREES** (secondary) |
| canon_stock -22.29 | -22.29 [-33.49, -3.27] (45.45) | **AGREES** |
| raw native -29.86 | -29.86 [-39.93, -12.64] (37.88) | **AGREES** |
| H2O gap -9.81 | -9.81 [-29.80, -4.64]; control minus H2O -9.04 [-20.96, -3.85] (cross-job S/K) | **AGREES** |
| T_noul rule infeasible -> fallback T 0.26; slice noul NLL 0.470 -> 0.997 | infeasible: no T in [0.25, 4] meets band <= 0.05 (min band 0.1066 at T <= 0.2606); fallback rerun T 0.26064, NLL 0.9974 (T 1: 0.4701) | **AGREES** (numbers) |
| ... "sharpened wrong noul answers" (read: the fallback hurt) | slice: yes, wrong noul NLL 1.44 -> 4.30. Public: the fallback RAISED Composite A by +29.36 [+12.27, +38.76] over T_noul = 1 | **DISAGREES with the implied attribution** (section 5) |

## Checks
1. **Primary + secondaries** (ev_cell_rescore.py, control = the non-canonical stock_shim, target = H2O e00d5ac7): see the
   table. Secondary axes for the primary: dI -3.23 [-9.82, +3.90], dC +2.76 [-9.17, +9.87], dCapability -0.23
   [-8.61, +5.62]; top-1 176 vs 173 (+1.3 pp, McNemar p 0.71); per type cc: choice +2.56, noul +1.25, **score -13.50
   [-19.39, -5.47]** (score T 0.43; cause not measured); noul band public: native_tfit 9/74, stock_shim 0/74, native 35/74.
2. **G1 used real perturbations, bitwise: PASS.** Every op n > 0 (213/213/139); on disk 213/213, 213/213, 139/139
   rows in the regenerated in-memory order and changed vs base; both arms used the same perturbed.jsonl (1fab827a);
   G1 base preds = the native arm (231/231). Teeth: stock_shim FAILS on the same files (27/21/13 flips, max abs dp 0.84).
   G1 was run on the raw native readout; native_tfit is a per-(type, K) temperature of it, so invariance carries over
   (rebuild check in section 5: max abs dp 2.8e-17).
3. **Score op re-run independently: PASS, bitwise.** I regenerated the perturbation files with
   `ev_score_perturb.py --public` @c47c0a46: base.jsonl f63f7483 and perturbed.jsonl 6aefb6fa are byte-identical to the
   in-job files; then `ev_score_flip.py --public --max-dp 0` on the landed native preds (exit 0) and stock_shim preds
   (exit 3, the teeth). Outputs: score_inv_native/, score_inv_stock_shim/.
4. **A-share** (tools/ashare.py; canonical order from evidence's key, independent of canon_shim):
   - The in-job calib_probs option order equals evidence's canonical order on all 416 slice choice rows (0 mismatches);
     calib gold = records gold (0 mismatches); 0 argmax ties; public choice items with a tie set: 0.
   | set | n | gold-A | pred-A | pred-A minus gold-A | top-1 | flag (> 5 pp) |
   |---|---|---|---|---|---|---|
   | slice choice, all | 416 | 38.46% | 34.38% | -4.09 pp | 61.06% | no |
   | slice choice, without ambiguous | 216 | 27.31% | 25.93% | -1.39 pp | 68.06% | no |
   | slice choice, ambiguous only | 200 | 50.50% | 43.50% | -7.00 pp | 53.50% | no |
   | public choice | 139 | 24.46% | 23.74% | -0.72 pp | 77.70% | no |
   The model under-predicts A everywhere; no letter shortcut is visible. Caveat (prereg): the slice shares the
   A-heavy canonical gold, so slice gains cannot rule a shortcut out; the public share is the cleaner check.
5. **T_noul fallback** (tools/tnoul.py, my own re-implementation; tnoul/tnoul.json):
   - **Rule infeasible: CONFIRMED.** Stock band on the slice = 0/741 (stock_p_shim strictly inside (0.2, 0.8)); 247/741
     sit exactly on the shim's commit floor 0.801 / 0.199, i.e. the stock readout's band is 0 **by construction** of the
     shim floor, so the threshold is 0.05 for an unfloored readout. Trained arm band: T 4 98.9%, T 1 39.5%, T 0.5
     20.5%, T 0.3 12.4%, T 0.25 10.7% (79/741). Band is monotone in T; the 400-point grid on [0.25, 4] has 0 feasible
     points. Reaching 0.05 needs T ~0.13 (outside the registered [0.25, 4]; NLL there ~1.9).
   - **Pre-specified, not ad hoc: YES, but only in code.** PREREG.md's text has no infeasibility clause; the only rule
     text is: "T_noul is chosen to minimise slice NLL subject to slice yes/no band rate <= stock's + 5pp". The fallback
     is in the registered package: `fit_noul_t_banded` docstring "Infeasible everywhere -> the T with the lowest band
     rate (flagged)", added in 36b790f1 (10:20 -0700) with the infeasible-case test in 28e23da4 (10:44), both before the
     PREREG entry 14cb45ce (10:51), which names package @28e23da4; unchanged at a8bca917. Tie-break among equal-band T
     = lowest NLL, which gives 0.2606 rather than the bound 0.25 (my re-run reproduces 0.26064). Documentation gap:
     the prose should state the fallback; it was approved by reference to the sha only.
   - **How much of the primary's loss it caused: none; it helped (DIAGNOSTIC ONLY, not a replacement primary).** I
     rebuilt native_tfit from native.preds.jsonl + tfit.json (max abs dp vs the in-job file 2.8e-17), then the same arm
     with T_noul = 1 (choice 1.024, choice:3 0.955, score 0.429 unchanged). Public noul band 9/74 -> 35/74.
     native_tfit minus (T_noul = 1): **dCompA +29.36 [+12.27, +38.76]**, dI +14.54 [+7.31, +19.93], dC -5.47
     [-7.59, +2.46]; the T_noul = 1 arm scores 37.60 (stock_shim minus it +30.14). The slice NLL rise (0.470 -> 0.997)
     is real, but jevkit scores a band answer as wrong, so on Composite the fallback was a large net gain. The
     residual -0.78 vs stock is not due to T_noul; the paired per-type numbers point at **score items** (cc -13.50,
     score T 0.43; whether that T causes it is not measured) and routing (-25 pp top-1, 0/3). This is a public-231 diagnostic: it must not select any T.
6. **Lineage**: scored ckpt = ckpt-step0000150.pt da7276b0 = the slice-selected best (dev-card NLL argmin, step 150 of
   200), exported as trainable-step0000150.safetensors 8e936008 (the step-200 export 2f508bb8 is not the scored one).
   Amendment 2 schedule in cell.json: max_steps 200, warmup_steps 50, lr_schedule cosine, lr_min_frac 0.1,
   lr_horizon_steps 200 (anti_collapse horizon 200, horizon_fixed true, lr_scale 1.0), dev_card_every 50 (cards at
   0/50/100/150/200, 360 rows each), collapse_events [] (no LR halving), train steps 200. The per-step LR is not logged
   in train_log.jsonl (step, loss, peak_gb, t only), so the schedule is verified from config, not from a trace.
   Event order (landing.json mtimes, UTC): dev cards 22:47:30 / 22:51:08 / 22:54:41 / 22:58:12 / 23:01:44; cell.json
   23:02:51; slice_first 23:04:00; calib_probs 23:05:41; slice_floor 23:07:11; tfit + slice_first done 23:07:12;
   **first public pred native.preds 23:07:31**; g1 native 23:08:31; flip8 + target 23:10:58; stock_shim 23:11:14;
   canon_stock 23:12:26; native_tfit / native_shimcal 23:12:58. Selection and every slice fit precede any public pred.
   Selected step != last, so Amendment 2's "budget may have truncated learning" caveat is not triggered.
7. **Near-miss rule**: the primary is **not claimed** (point -0.78). native_shimcal (+3.48, p 0.087) is a pre-registered
   SECONDARY and cannot replace it; the T_noul = 1 arm above is a diagnostic. No T, readout or checkpoint may be
   chosen from these public-231 numbers; any successor cell selects and fits only on the slice under a new prereg line.

## Files
- `primary_native_tfit/` (kit report: CELL_RESCORE.md, cell_rescore.json, core.rescore.json, per-arm reports);
  `secondary_native_shimcal/`, `secondary_canon_stock/`, `secondary_native/`; `diag_tnoul1/` (kit run with the
  T_noul = 1 preds as a target).
- `score_inv_native/`, `score_inv_stock_shim/`: my ev_score_flip.py runs.
- `ashare/ashare_native.json`; `tnoul/tnoul.json` + `tnoul/native_tfit_noul1.preds.jsonl` (diagnostic preds).
- `job-zbmhc/`: screen.json, landing.json, cell.json, identity-child.json, identity.json, slice_first.json, tfit.json,
  slice_floor.json, MANIFEST.jsonl, job_record.json (presigned URL removed), tar_members.txt, ckpt/ sidecars +
  member shas, dev_cards/, reports/ (G1, flip8, in-job score_inv).
- `inputs/`, `tools/` (landing_to_mtimes.py, ashare.py, tnoul.py, gold_letter_canonical.py copied from @1ccdc676).
The artifact tar and its extraction were deleted after the receipts were committed; arm preds are in the lead's
receipts @95a731f6 (byte-identical to the artifact).

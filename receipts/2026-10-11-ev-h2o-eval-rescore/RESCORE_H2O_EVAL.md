# Independent rescore: EV-PREREG-H2O-HOST eval-only job-4g6m8 (H2O-Lightning-4B v1.1 @193ad740)

Evidence lane, 2026-10-11. The lead's claims are in claude/lead-h2o-eval @f5615683, receipts/2026-10-11-h2o-host-eval/.
**Verdict: AGREES on every number.** 235 numbers were compared with decision.json. The largest |diff| is 7.1e-15,
there are 0 flags, and no |Δ| is above 0.05. **No contender's CI excludes 0**, so the primary is NOT met for any arm.
The temperature fit follows EV-PREREG-AREA-TEMP/a. G1 PASSES for c_symfork and FAILS for the control.

Caveat: these are open public items only. The board's sealed split may rank differently (the open-to-board I gap measured on H2O is −17.6).

## How I checked
- **Artifact.** I downloaded it myself through the job record's signed URL. art-jrasm is a 1,433,600-byte tar with sha256
  `8feba886b7461c834161b7b6f9f84fa0840cf7ec4466b297f78f1ef78d185181`, which equals `job get job-4g6m8` artifact.sha256.
  Every arm's preds.jsonl sha starts with the lead's prefix. The artifact's decision.json is byte-identical to the
  committed one (`1daa1be8…`).
- **Script.** `scripts/ev_rescore_h2o_eval.py` (this branch). It re-derives every arm from `raw_public.jsonl` and refits
  every T from `slice_raw.jsonl` plus the jev2a gold. It recomputes the slice selection with its own code. It runs
  `jevkit score --public` (the frozen CLI) per arm with the declared H2O price class (deepinfra Qwen/Qwen3.5-4B,
  0.03/0.15 per 1M). It computes paired deltas vs the in-job control on the same draws (B 1000, seed 16) by two paths:
  its own loop and jevkit's `bootstrap()._pairs`. It also covers EV-PREREG-SECONDARY-AXES and regenerates the G1
  perturbations in memory. Outputs: `rescore.json`, `reports/*.report.json`, `price.json`.

## (1)-(3) Shas, revision, control
| check | result | verdict |
|---|---|---|
| tar sha vs job record | 8feba886…5181 = 8feba886…5181 | AGREES |
| preds sha prefixes (h2o e00d5ac7, a_T b2a37caf, a_Ttype 8077cad9, a_Tfam 4bdf446b, b_native 08a436bc, b_shimcal 9cfc498e, c_symfork c3d986c6) | 7/7 match | AGREES |
| revision | params, run_meta, HF requested, HF resolved, verify.json and the snapshot dir name are all 193ad740925b176a3b70a5a13a7cff2f2fadd01e. All 12 file pins match (weights c65443b1…). | AGREES |
| control vs job-zbtbj H2O preds (sha 60a06842, = SCORECARD.json) | 231/231 answers **bitwise identical**, top-1 per item identical, 205 = 205 | AGREES |
| control Comp A 76.78 | 76.78 (I 77.56, C 83.42, S 92.12, K 61.13). zbtbj's own run scores A 76.64: I, C and K are identical, and S is 91.31 vs 92.12 (p50 latency 53.2 ms in zbtbj vs 48.5 ms here). The +0.14 comes from latency alone, and every Δ below is in-job paired, so it does not affect any comparison. | AGREES |

## (3) Every arm, rescored (jevkit CLI) and paired vs the in-job control (B 1000, seed 16)
| arm | lead A | rescore A | I | C | top-1 | noul band | lead Δ A [CI] | rescore Δ A [CI] (jevkit native CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| h2o (control) | 76.78 | 76.78 | 77.56 | 83.42 | 205 | 0/74 | – | – | AGREES |
| a_T | 77.09 | 77.09 | 78.00 | 84.39 | 205 | 0 | +0.31 [−0.44, +1.19] | +0.313 [−0.44, +1.19] (same) | AGREES |
| a_Ttype | 76.85 | 76.85 | 77.65 | 83.62 | 205 | 0 | +0.07 [−0.74, +0.88] | +0.065 [−0.74, +0.88] | AGREES |
| a_Tfam | 77.29 | 77.29 | 78.03 | 85.31 | 205 | 0 | +0.51 [−0.21, +1.35] | +0.509 [−0.21, +1.35] | AGREES |
| b_native | 68.10 | 68.10 | 53.87 | 77.18 | 205 | 32/74 (43.2%) | −8.68 [−18.39, −5.94] | −8.679 [−18.39, −5.94] | AGREES |
| b_shimcal | 76.78 | 76.78 | 77.56 | 83.42 | 205 | 0 | 0.00 [0, 0] | 0.000 [0, 0]; answers equal the control's to 2.2e-16 (124/231 bitwise, the rest float round-off) | AGREES |
| c_symfork | 14.11 | 14.11 | 29.65 | 26.51 | 121 | 10/74 (13.5%) | −62.67 [−72.62, −41.37] | −62.671 [−72.62, −41.37] | AGREES |

Secondary axes (EV-PREREG-SECONDARY-AXES) agree to within 1e-14 with decision.json. That covers per-axis CIs, per-type I/C, Comp A
quantiles, the fraction of resamples with I < 50 (0 for all arms except b_native 0.123 and c_symfork 0.999), Δ top-1
plus McNemar, and Δ choice NLL. The a_* arms leave top-1 unchanged (0 discordant pairs). Every gain is Calibration from
post-processing. Δ choice NLL is a_T −0.014 [−0.026, −0.001], a_Ttype −0.013 [−0.023, −0.002] and a_Tfam −0.021 [−0.030, −0.011]: the NLL
improves, but neither Δ C nor Δ A has a CI that excludes 0. Re-deriving all five score-only arms and the control from
raw_public.jsonl through the pinned vendored shim (sha 355791c3 = the job pin) gives the same decision on 231/231 for
every arm, with max |dp| ≤ 5.6e-17. a_Tfam applies T_f to 74 public items, the ones in the five jev2a families. The other 157 use T_J.

## (4) T-fit provenance: COMPLIANT with EV-PREREG-AREA-TEMP/a
- **Fitted only on the held-out slice.** The fit (`h2o_eval.fit_temperatures`) reads only `slice_raw.jsonl`, the
  shim's raw label log-probs of the 1,000 slice records, and their jev2a gold. The public items enter only
  afterwards, in `derive`. I re-selected the slice with my own code from bschema-draw-jev2a records.jsonl.gz
  (uncompressed sha 393d0f0f… = the pin), taking per family8 the 200 smallest sha256(salt + "\n" + id). Its ids sha is
  **204e9bfe…2a9c**, which matches the prereg/params, and the slice_raw ids are the same set.
- **Ids sha recorded before the job.** `h2o_slice_ids.txt` plus ids_sha256 204e9bfe were committed in e46f9afb
  (2026-10-11T05:26Z) and are unchanged in c1936746 (06:17Z). The job ran preflight at 06:58Z and started at 07:27Z.
- **Never on public items.** The slice has 0 id overlap and 0 exact state-text overlap with the 231 public items.
- **≥ 200 decisions per family.** Every family has exactly 200 decisions (fact, judge_hard, multi_hop,
  temporal_numeric, tool_selection), 0 skipped for missing gold, and 0 failed rows. That is at the boundary but
  compliant. By type: choice 519, noul 451, score 30.
- **Refit.** My continuous optimum (golden section on log T) matches the lead's 1,401-point grid to ≤ 0.12% everywhere,
  which is within the grid spacing of 0.25%. T_J is 0.67625 (mine) vs 0.67629 (lead). Per family: fact 0.3945/0.3942,
  judge_hard 0.8141/0.8143, multi_hop 0.6989/0.6984, temporal_numeric 0.8248/0.8244, tool_selection 0.5053/0.5050. No T sits at a bound.
- **Caveats (not breaches).** (i) About 30% of the slice derives from public corpora: boolq 41, gsm 88, musique 80, glaive 100.
  H2O's own training data is unknown, so "never in the scored model's training" can be asserted for our use only.
  (ii) A-PREREG-JEV2-4B pools jev2a for training. If these T values are ever reused on a trained arm, the 1,000 slice
  ids must be excluded from that training.

## (5) G1: AGREES. c_symfork PASS, control FAIL. The perturbations are real.
- The perturbations were regenerated in memory with `jevkit.perturb.make_perturbations(load_public(), ops, (0,))`:
  565 rows (reverse 213, shuffle 213, relabel 139). They match the artifact's perturbed.jsonl on ids, ops, label maps
  and question content (0 mismatches). In memory, **option order changed on 213/213 reverse and 213/213 shuffle**, and
  the labels changed on 139/139 relabel.
- **The on-disk perturbed.jsonl is written with `sort_keys=True`** (jobs/sym_eval/perturb_eval.py:73), so its option order
  equals the in-memory order on 0/213 reverse and 0/213 shuffle. That is the PR #245/#246 defect, but here it affects
  only the receipt file. The requests used the in-memory objects (`run_items(ad, [p.item for p in ps])`). Proof: the
  control's answers moved (|dp| > 0) on 139/213 reverse, 139/213 shuffle and 139/139 relabel. **Fix proposed:** write that file
  without sort_keys, so nobody re-running from it gets no-op reorders.
- **Flips recomputed** with `jevkit.perturb.flip_report` from both arms' base and perturbed preds. They equal the lead's
  counts and max |dp| exactly:

  | arm | relabel (139) | reverse (213) | shuffle (213) | max abs dp | G1 |
  |---|---|---|---|---|---|
  | h2o control | 6 flips | 6 | 5 | 0.433 / 0.329 / 0.296 | FAIL (AGREES) |
  | c_symfork | 0 | 0 | 0 | 0.0 on all 565 (no answer moved at all) | PASS (AGREES) |

  The lead's row-identity check shows sym-fork rows bit-identical on 565/565. The G1 base preds equal the arm preds
  bitwise for both arms, so the forward is deterministic.
- **Nuance.** For the control, the 74 noul pairs under reverse and shuffle are no-ops: 0 noul answers moved, because the H2O
  shim renders yes/no with fixed true/false names. The control's effective reorder denominator is therefore 139 choice
  pairs per operator, and its flip rate on real reorders is 6/139 and 5/139. c_symfork's invariance is structural (identical rows), not
  an artefact of no-op inputs. It is invariant but much weaker: 121/231 top-1 and A 14.11. It is a G1 probe, not a contender.

## (6) b_native's noul collapse: CONFIRMED (noul I 76.57 → 8.15)
jevkit treats a yes/no answer with 0.2 < P(yes) < 0.8 as an abstention (`abstain`, not correct). Noul I is
chance-corrected, 100·(acc − 0.5)/0.5 (scoring.py `_cell_cc`). Read raw (T 1, no floor), 32/74 noul answers fall in the
band. The median max(p, 1−p) is 0.830 and the minimum is 0.506. 24 of those 32 were on the correct side, so jevkit-correct falls
from 66 to 42 of 74 and noul I falls to 8.15. The card top-1 is unchanged at 205, because P(yes) ≥ 0.5 still picks the same side. The
shim's commit floor (0.801) moves any max(p, 1−p) < 0.801 to 0.801 on its own side. In the control, 26 noul answers sit exactly at
0.801 and 0 are in the band. The shim's post-processing (T 0.8 plus the floor) is therefore worth +23.7 I and +8.7 Composite A on H2O.
The floor is what empties the band. This job did not isolate the floor and T separately.
b_shimcal (native → the shim's T and floor) reproduces the control exactly, as the algebra predicts.

## Shas
art-jrasm tar 8feba886…5181. MANIFEST.jsonl b19c2069…; phases final.tgz 10acdfdd…, run.tgz 3f984f63…, setup.tgz b7a12427…;
raw_public.jsonl 9ef2c65b…; slice_raw.jsonl 6b17eaf4…; slice.json 1f05bebe…; temperature_fit.json f6543ba5…;
g1 perturbed.jsonl 7102241c… (both arms); g1 h2o base/pert preds d36e0ef3… / 9d003e15…; g1 c_symfork base/pert preds
2387f0ad… / b34fc5ea…. Rescore outputs: SHA256SUMS in this directory.

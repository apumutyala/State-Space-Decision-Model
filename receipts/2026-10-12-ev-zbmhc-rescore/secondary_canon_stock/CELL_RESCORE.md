# Cell rescore: canon_stock vs stock_shim (FLAGGED)

Artifact `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\zbmhc\final` (screen.json sha 703e037cccd3). jevkit CLI per arm (--public, declared price); paired stratified bootstrap B=1000 seed 16 (strata of the baseline table), Speed/Cost held fixed; tol 0.05.

## (1) PRIMARY
| quantity | value |
|---|---|
| Comp A primary / control | 45.451 / 67.743 |
| delta Comp A [95%] | -22.2922 [-33.4870, -3.2746] |
| 97.5% / 99% | [-34.6483, -2.0502] / [-35.2672, -0.8297] |
| one-sided p (P(delta<=0)) | 0.9990 (+1: 0.9990) |
| delta I [95%] | -14.529 [-21.50, -6.08] |
| success at 95% | False |
| in-job | -0.7758 [-16.3645, +3.4737] success False |
| screen.json diff | 124 fields, max abs diff 21.516392949870905, FLAGGED |

Holm family canon_stock (alpha 0.05):

- canon_stock: p 0.9990, rank 1, level 0.0500, tested True, CI at level [-33.487, -3.275], claimable **False**

## (2) SECONDARY (EV-PREREG-SECONDARY-AXES; never turns a failed primary into a success)
- delta I: -14.53 [-21.50, -6.08]
- delta C: +8.38 [-4.17, +10.70]
- delta S: -0.69 [-0.69, -0.69] (held fixed)
- delta K: -0.13 [-0.13, -0.13] (held fixed)
- delta capability: -3.08 [-11.71, +1.22]
- delta composite_A: -22.29 [-33.49, -3.27]
- top-1 (jevkit correct) 160 vs 173: -5.63 pp [-9.96, -1.73], 8 / 21 discordant, McNemar p 0.0241
- choice NLL 0.6289 vs 0.6147 (+0.0143 [-0.0804, +0.1082]); I<50 in {'a': 0.952, 'b': 0.09}
- choice: cc +1.92 [-6.22, +9.35]; top-1 +0.72 pp [-4.33, +5.76] (8/7, McN 1)
- noul: cc -38.97 [-56.39, -22.18]; top-1 +0.00 pp [+0.00, +0.00] (0/0, McN 1)
- score: cc -6.53 [-20.07, +6.72]; top-1 +0.00 pp [+0.00, +0.00] (0/0, McN 1)
- noul 0.2-0.8 band: canon_stock 27/74, native 35/74, native_shimcal 0/74, native_tfit 9/74, stock_shim 0/74, target_ref 0/74

## (3) TARGET gap (paired on ids; S/K at each system's own value; cross-job: the target's S comes from its own job / hardware run and K from its own declared price; only I and C are paired on the same items)
| target | delta Comp A [95%] (primary - target) | control - target | S prim / S target | K prim / K target | caveat |
|---|---|---|---|---|---|
| h2o | -31.33 [-44.58, -9.35] | -9.04 [-20.96, -3.85] | 90.71 / 92.12 | 61.10 / 61.13 | cross-job |

## (4) INTEGRITY
- items 231, same ids all arms True
- canon_stock: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3932 (cap 4096)
- native: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3932 (cap 4096)
- native_shimcal: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3932 (cap 4096)
- native_tfit: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3932 (cap 4096)
- stock_shim: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- target_ref: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- train no-truncation: {'rows': 5836, 'max_row_tokens': 4066, 't_cap': 4096}; cell {'train_steps': 200, 'max_steps': 200, 'best': {'step': 150, 'nll': 0.5772574858044974, 'accuracy': 0.7277777777777777, 'path': 'ckpt-step0000150.pt'}, 'scored_checkpoint': {'path': 'ckpt-step0000150.pt', 'step': 150, 'sha256': 'da7276b0b44ecbdd415844bfe442bbeeaaa33d327c585bd75dc0407eba12b126'}, 'stock_arm_identity': {'arm': 'stock_shim', 'shim': 'h2oai/h2o-lightning-4b@v1.1', 'temperature': 0.8, 'lm_head': 'tied embed_tokens, fp32 label rows', 'lora': 'off (scaling 0)', 'n': 231, 'status': {'ok': 231}}}

## (5) SELECTION
- best {'step': 150, 'nll': 0.5772574858044974, 'accuracy': 0.7277777777777777, 'path': 'ckpt-step0000150.pt'}; argmin NLL step 150 (= best: True); trained steps 200 / max 200; scored {'path': 'ckpt-step0000150.pt', 'step': 150, 'sha256': 'da7276b0b44ecbdd415844bfe442bbeeaaa33d327c585bd75dc0407eba12b126'} (= selected: True)
| step | NLL | top-1 |
|---|---|---|
| 0 | 0.741262 | 0.6778 |
| 50 | 0.642359 | 0.6917 |
| 100 | 0.618340 | 0.7056 |
| 150 | 0.577257 | 0.7278 |
| 200 | 0.581146 | 0.7194 |
- step-0 floor: 0.7117 vs 0.6667 (1200 decisions), delta +4.50 pp [+2.16, +7.00] iid / [+1.92, +7.00] family-stratified, McNemar p 0.000732 (151 vs 97); slice ids sha 844233f228b7; in-job diff 1.3877787807814457e-17; pass True
- order (--mtimes receipts\2026-10-12-ev-zbmhc-rescore\inputs\mtimes_zbmhc.json): selection cell.json 2026-10-12T23:02:51Z vs first public pred 2026-10-12T23:07:31Z (out/screen/jevkit/native.preds.jsonl): selection before public **True**; floor files before public True

## (6) G1 (reverse / shuffle / relabel; from files)
- **native: G1 PASS** (bitwise True, max|dp| 0.0, bound 0.0); base preds = arm: {'canon_stock': 0, 'native': 231, 'native_shimcal': 0, 'native_tfit': 0, 'stock_shim': 0, 'target_ref': 0}
  - option_reverse: n 213, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - option_shuffle: n 213, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - key_relabel: n 139, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 139/139 rows in in-memory order, 139/139 changed vs base
  - in-job report: g1_pass True, counts agree True
- **stock_shim: G1 FAIL** (bitwise False, max|dp| 0.8350774053756308, bound 0.0); base preds = arm: {'canon_stock': 0, 'native': 0, 'native_shimcal': 12, 'native_tfit': 0, 'stock_shim': 231, 'target_ref': 9}
  - option_reverse: n 213, invalid 0, flips 27 ([+8.3, +17.3] %), max|dp| 0.8176519553248887; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - option_shuffle: n 213, invalid 0, flips 21 ([+6.0, +14.1] %), max|dp| 0.8350774053756308; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - key_relabel: n 139, invalid 0, flips 13 ([+4.8, +14.0] %), max|dp| 0.6675414484513097; on disk: 139/139 rows in in-memory order, 139/139 changed vs base
  - FAIL: option_reverse: 27 flips
  - FAIL: option_reverse: max|dp| 0.8176519553248887 > 0.0
  - FAIL: option_shuffle: 21 flips
  - FAIL: option_shuffle: max|dp| 0.8350774053756308 > 0.0
  - FAIL: key_relabel: 13 flips
  - FAIL: key_relabel: max|dp| 0.6675414484513097 > 0.0
  - in-job report: g1_pass False, counts agree True

## (6b) Score-item invariance (separate from G1; does not change the G1 verdict)
- not evaluated (no --score-inv)

## (7) FLIP battery
- not evaluated (no --flip8 and no jevkit/g1 dir)

## (8) TRANSFER (slice = selected - step 0; public = primary - control)
corr(slice delta, public delta correct) -0.618; corr(slice, public top-1) -0.707

| family | slice step-0 | slice delta pp [CI] (McN) | public n | control top-1 | public delta top-1 pp (+/-) | public delta I | flag |
|---|---|---|---|---|---|---|---|
| adequacy | 64.0 | +5.0 [-3.0, +13.0] (0.26) | 12 | 83.3 | +0.0 (0/0) | -66.7 |  |
| adversarial |  |  | 6 | 66.7 | +16.7 (1/0) | -20.8 | not in the slice |
| ambiguous | 54.0 | -0.5 [-7.5, +7.0] (1) | 7 | 42.9 | +14.3 (1/0) | +21.1 |  |
| extraction |  |  | 24 | 100.0 | -4.2 (0/1) | -3.6 | public loss |
| fact |  |  | 12 | 91.7 | +0.0 (0/0) | +0.0 | not in the slice |
| intent |  |  | 24 | 91.7 | +4.2 (1/0) | +6.9 | not in the slice |
| judge_hard | 68.5 | +9.5 [+2.5, +16.0] (0.011) | 17 | 58.8 | +0.0 (0/0) | -17.6 | GAIN SLICE ONLY |
| long_policy | 54.0 | +5.5 [+0.0, +11.0] (0.071) | 19 | 52.6 | -5.3 (1/2) | -4.9 | public loss |
| multi_hop |  |  | 18 | 55.6 | +0.0 (1/1) | -32.3 | not in the slice |
| ordinal |  |  | 12 | 100.0 | +0.0 (0/0) | -1.2 | not in the slice |
| policy | 81.5 | +5.0 [+1.0, +9.5] (0.041) | 12 | 91.7 | +0.0 (0/0) | -16.7 | GAIN SLICE ONLY |
| probability |  |  | 10 | 40.0 | +0.0 (0/0) | +0.0 | not in the slice |
| routing |  |  | 12 | 91.7 | -16.7 (0/2) | -20.0 | public loss |
| routing_hard |  |  | 5 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| temporal_numeric | 78.0 | +2.5 [-2.0, +7.5] (0.44) | 15 | 13.3 | +20.0 (3/0) | +5.4 |  |
| tool_selection |  |  | 12 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| tradeoff |  |  | 6 | 83.3 | -16.7 (0/1) | -15.9 | public loss |
| trap |  |  | 8 | 87.5 | +0.0 (0/0) | -32.4 | not in the slice |

## Flags
- core: target_ref: arm present in the preds but not in screen.json jevkit.arms
- core: screen.json has no vs_stock_shim.target_ref
- core: diff primary.delta_composite_A.point: rescore -22.2922 vs screen -0.7758 (|d| 21.5164 > 0.05)

## Warnings
- none

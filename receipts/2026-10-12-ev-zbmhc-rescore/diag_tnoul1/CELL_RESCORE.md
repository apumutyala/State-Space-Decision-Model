# Cell rescore: native_tfit vs stock_shim (FLAGGED)

Artifact `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\zbmhc\final` (screen.json sha 703e037cccd3). jevkit CLI per arm (--public, declared price); paired stratified bootstrap B=1000 seed 16 (strata of the baseline table), Speed/Cost held fixed; tol 0.05.

## (1) PRIMARY
| quantity | value |
|---|---|
| Comp A primary / control | 66.967 / 67.743 |
| delta Comp A [95%] | -0.7758 [-16.3645, +3.4737] |
| 97.5% / 99% | [-18.4299, +5.0005] / [-22.6337, +7.8999] |
| one-sided p (P(delta<=0)) | 0.7360 (+1: 0.7363) |
| delta I [95%] | -3.229 [-9.82, +3.90] |
| success at 95% | False |
| in-job | -0.7758 [-16.3645, +3.4737] success False |
| screen.json diff | 124 fields, max abs diff 0.0, FLAGGED |

Holm family native_tfit (alpha 0.05):

- native_tfit: p 0.7360, rank 1, level 0.0500, tested True, CI at level [-16.364, +3.474], claimable **False**

## (2) SECONDARY (EV-PREREG-SECONDARY-AXES; never turns a failed primary into a success)
- delta I: -3.23 [-9.82, +3.90]
- delta C: +2.76 [-9.17, +9.87]
- delta S: -1.13 [-1.13, -1.13] (held fixed)
- delta K: -0.13 [-0.13, -0.13] (held fixed)
- delta capability: -0.23 [-8.61, +5.62]
- delta composite_A: -0.78 [-16.36, +3.47]
- top-1 (jevkit correct) 176 vs 173: +1.30 pp [-3.03, +5.63], 16 / 13 discordant, McNemar p 0.711
- choice NLL 0.5427 vs 0.6147 (-0.0720 [-0.2016, +0.0452]); I<50 in {'a': 0.244, 'b': 0.09}
- choice: cc +2.56 [-6.29, +11.36]; top-1 +1.44 pp [-4.32, +7.19] (10/8, McN 0.815)
- noul: cc +1.25 [-16.17, +18.67]; top-1 +9.46 pp [+2.70, +17.57] (8/1, McN 0.0391)
- score: cc -13.50 [-19.39, -5.47]; top-1 +5.56 pp [+0.00, +16.67] (1/0, McN 1)
- noul 0.2-0.8 band: canon_stock 27/74, native 35/74, native_shimcal 0/74, native_tfit 9/74, stock_shim 0/74, target_ref 0/74

## (3) TARGET gap (paired on ids; S/K at each system's own value; cross-job: the target's S comes from its own job / hardware run and K from its own declared price; only I and C are paired on the same items)
| target | delta Comp A [95%] (primary - target) | control - target | S prim / S target | K prim / K target | caveat |
|---|---|---|---|---|---|
| tnoul1 | +29.36 [+12.27, +38.76] | +30.14 [+14.39, +42.02] | 90.26 / 90.26 | 61.10 / 61.10 | cross-job |

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
corr(slice delta, public delta correct) -0.276; corr(slice, public top-1) -0.034

| family | slice step-0 | slice delta pp [CI] (McN) | public n | control top-1 | public delta top-1 pp (+/-) | public delta I | flag |
|---|---|---|---|---|---|---|---|
| adequacy | 64.0 | +5.0 [-3.0, +13.0] (0.26) | 12 | 83.3 | +0.0 (0/0) | -33.3 |  |
| adversarial |  |  | 6 | 66.7 | +0.0 (0/0) | -4.3 | not in the slice |
| ambiguous | 54.0 | -0.5 [-7.5, +7.0] (1) | 7 | 42.9 | +14.3 (2/1) | +21.1 |  |
| extraction |  |  | 24 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| fact |  |  | 12 | 91.7 | +0.0 (0/0) | +0.0 | not in the slice |
| intent |  |  | 24 | 91.7 | +4.2 (1/0) | +6.9 | not in the slice |
| judge_hard | 68.5 | +9.5 [+2.5, +16.0] (0.011) | 17 | 58.8 | +23.5 (4/0) | +23.5 | gain on both |
| long_policy | 54.0 | +5.5 [+0.0, +11.0] (0.071) | 19 | 52.6 | +0.0 (3/3) | -7.8 |  |
| multi_hop |  |  | 18 | 55.6 | +0.0 (1/1) | +3.2 | not in the slice |
| ordinal |  |  | 12 | 100.0 | +0.0 (0/0) | -2.9 | not in the slice |
| policy | 81.5 | +5.0 [+1.0, +9.5] (0.041) | 12 | 91.7 | +0.0 (0/0) | +0.0 | GAIN SLICE ONLY |
| probability |  |  | 10 | 40.0 | +20.0 (2/0) | +27.4 | not in the slice |
| routing |  |  | 12 | 91.7 | -25.0 (0/3) | -30.0 | public loss |
| routing_hard |  |  | 5 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| temporal_numeric | 78.0 | +2.5 [-2.0, +7.5] (0.44) | 15 | 13.3 | +26.7 (5/1) | +7.1 |  |
| tool_selection |  |  | 12 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| tradeoff |  |  | 6 | 83.3 | +0.0 (0/0) | +0.0 | not in the slice |
| trap |  |  | 8 | 87.5 | +12.5 (1/0) | -10.5 | not in the slice |

## Flags
- core: target_ref: arm present in the preds but not in screen.json jevkit.arms
- core: screen.json has no vs_stock_shim.target_ref

## Warnings
- none

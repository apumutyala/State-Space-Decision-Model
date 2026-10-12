# Cell rescore: native vs stock_shim (FLAGGED)

Artifact `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\dl6zs79\final` (screen.json sha 10198ebcf26e). jevkit CLI per arm (--public, declared price); paired stratified bootstrap B=1000 seed 16 (strata of the baseline table), Speed/Cost held fixed; tol 0.05.

## (1) PRIMARY
| quantity | value |
|---|---|
| Comp A primary / control | 29.981 / 66.645 |
| delta Comp A [95%] | -36.6632 [-47.9664, -19.1925] |
| 97.5% / 99% | [-49.4179, -16.2830] / [-51.7253, -14.0228] |
| one-sided p (P(delta<=0)) | 1.0000 (+1: 1.0000) |
| delta I [95%] | -19.210 [-28.88, -8.72] |
| success at 95% | False |
| in-job | -36.6632 [-47.9664, -19.1925] success False |
| screen.json diff | 97 fields, max abs diff 0.0, FLAGGED |

Holm family native (alpha 0.05):

- native: p 1.0000, rank 1, level 0.0500, tested True, CI at level [-47.966, -19.192], claimable **False**

## (2) SECONDARY (EV-PREREG-SECONDARY-AXES; never turns a failed primary into a success)
- delta I: -19.21 [-28.88, -8.72]
- delta C: +6.18 [-11.86, +10.38]
- delta S: -6.21 [-6.21, -6.21] (held fixed)
- delta K: +0.06 [+0.06, +0.06] (held fixed)
- delta capability: -6.52 [-18.40, -1.21]
- delta composite_A: -36.66 [-47.97, -19.19]
- top-1 (jevkit correct) 131 vs 171: -17.32 pp [-22.94, -11.69], 10 / 50 discordant, McNemar p 1.62e-07
- choice NLL 0.8354 vs 0.6146 (+0.2208 [+0.0631, +0.3731]); I<50 in {'a': 1.0, 'b': 0.173}
- choice: cc -3.86 [-14.12, +5.67]; top-1 -2.88 pp [-10.07, +3.60] (10/14, McN 0.541)
- noul: cc -80.83 [-101.88, -55.63]; top-1 +5.41 pp [-4.05, +14.86] (9/5, McN 0.424)
- score: cc -16.18 [-35.31, +5.71]; top-1 -33.33 pp [-50.00, -16.67] (0/6, McN 0.0312)
- noul 0.2-0.8 band: native 48/74, native_shimcal 0/74, native_tfit 48/74, stock_shim 0/74, target_ref 0/74

## (3) TARGET gap (paired on ids; S/K at each system's own value; cross-job: the target's S comes from its own job / hardware run and K from its own declared price; only I and C are paired on the same items)
| target | delta Comp A [95%] (primary - target) | control - target | S prim / S target | K prim / K target | caveat |
|---|---|---|---|---|---|
| h2o | -46.80 [-59.36, -30.87] | -10.14 [-26.64, -4.85] | 85.00 / 92.12 | 61.28 / 61.13 | cross-job |

## (4) INTEGRITY
- items 231, same ids all arms True
- native: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3914 (cap 4096)
- native_shimcal: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3914 (cap 4096)
- native_tfit: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3914 (cap 4096)
- stock_shim: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- target_ref: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- train no-truncation: {'rows': 5824, 'max_row_tokens': 4064, 't_cap': 4096}; cell {'train_steps': 258, 'max_steps': 728, 'best': {'step': 200, 'nll': 0.6150277999701741, 'accuracy': 0.7111111111111111, 'path': 'ckpt-step0000200.pt'}, 'scored_checkpoint': {'path': 'trainable-step0000200.safetensors', 'step': 200, 'sha256': '5cb7da4246efce72c17d15cf5af9fc6b8137a7e76e371a970a67c89303835310'}, 'stock_arm_identity': {'arm': 'stock_shim', 'shim': 'h2oai/h2o-lightning-4b@v1.1', 'temperature': 0.8, 'lm_head': 'tied embed_tokens, fp32 label rows', 'lora': 'off (scaling 0)', 'n': 231, 'status': {'ok': 231}}}

## (5) SELECTION
- best {'step': 200, 'nll': 0.6150277999701741, 'accuracy': 0.7111111111111111, 'path': 'ckpt-step0000200.pt'}; argmin NLL step 200 (= best: True); trained steps 258 / max 728; scored {'path': 'trainable-step0000200.safetensors', 'step': 200, 'sha256': '5cb7da4246efce72c17d15cf5af9fc6b8137a7e76e371a970a67c89303835310'} (= selected: False)
| step | NLL | top-1 |
|---|---|---|
| 0 | 0.901269 | 0.4194 |
| 100 | 0.690952 | 0.6944 |
| 200 | 0.615028 | 0.7111 |
- step-0 floor: 0.6783 vs 0.6675 (1200 decisions), delta +1.08 pp [-1.83, +4.09] iid / [-2.00, +4.00] family-stratified, McNemar p 0.534 (193 vs 180); slice ids sha 844233f228b7; in-job diff 0.0; pass False
- order (--mtimes C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\inputs\mtimes_6zs79.json): selection cell.json 2026-10-12T08:53:58Z vs first public pred 2026-10-12T09:10:24Z (out/screen/jevkit/native.preds.jsonl): selection before public **True**; floor files before public True

## (6) G1 (reverse / shuffle / relabel; from files)
- **native: G1 PASS** (bitwise True, max|dp| 0.0, bound 0.0); base preds = arm: {'native': 231, 'native_shimcal': 0, 'native_tfit': 76, 'stock_shim': 0, 'target_ref': 0}
  - option_reverse: n 213, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - option_shuffle: n 213, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - key_relabel: n 139, invalid 0, flips 0 ([+0.0, +0.0] %), max|dp| 0.0; on disk: 139/139 rows in in-memory order, 139/139 changed vs base
  - in-job report: g1_pass True, counts agree True
- **stock_shim: G1 FAIL** (bitwise False, max|dp| 0.8424278008753008, bound 0.0); base preds = arm: {'native': 0, 'native_shimcal': 9, 'native_tfit': 0, 'stock_shim': 231, 'target_ref': 10}
  - option_reverse: n 213, invalid 0, flips 28 ([+8.5, +18.2] %), max|dp| 0.8209214392295661; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - option_shuffle: n 213, invalid 0, flips 21 ([+6.0, +14.1] %), max|dp| 0.8424278008753008; on disk: 213/213 rows in in-memory order, 213/213 changed vs base
  - key_relabel: n 139, invalid 0, flips 12 ([+4.3, +13.1] %), max|dp| 0.6722267413776731; on disk: 139/139 rows in in-memory order, 139/139 changed vs base
  - FAIL: option_reverse: 28 flips
  - FAIL: option_reverse: max|dp| 0.8209214392295661 > 0.0
  - FAIL: option_shuffle: 21 flips
  - FAIL: option_shuffle: max|dp| 0.8424278008753008 > 0.0
  - FAIL: key_relabel: 12 flips
  - FAIL: key_relabel: max|dp| 0.6722267413776731 > 0.0
  - in-job report: g1_pass False, counts agree True

## (7) FLIP battery
- ops (8/8): state_paraphrase, instruction_paraphrase, option_paraphrase, option_reverse, option_shuffle, key_relabel, whitespace_noise, markup_noise
- cell: average 4.14% [+2.94, +5.35]; state_paraphrase 1.9%, instruction_paraphrase 5.2%, option_paraphrase 8.7%, option_reverse 0.0%, option_shuffle 0.0%, key_relabel 0.0%, whitespace_noise 10.4%, markup_noise 6.9%
- reported baseline qwen35_4b_shim: 7.55% [+5.71, +9.67] (comparable to this cell's average: True)

## (8) TRANSFER (slice = selected - step 0; public = primary - control)
corr(slice delta, public delta correct) +0.577; corr(slice, public top-1) -0.502

| family | slice step-0 | slice delta pp [CI] (McN) | public n | control top-1 | public delta top-1 pp (+/-) | public delta I | flag |
|---|---|---|---|---|---|---|---|
| adequacy | 64.5 | -8.0 [-16.0, +0.0] (0.076) | 12 | 75.0 | +8.3 (1/0) | -50.0 |  |
| adversarial |  |  | 6 | 66.7 | +16.7 (1/0) | +35.9 | not in the slice |
| ambiguous | 54.0 | +3.5 [-5.5, +13.0] (0.56) | 7 | 42.9 | +0.0 (2/2) | +0.0 |  |
| extraction |  |  | 24 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| fact |  |  | 12 | 91.7 | +8.3 (1/0) | -83.3 | not in the slice |
| intent |  |  | 24 | 91.7 | +8.3 (2/0) | +13.9 | not in the slice |
| judge_hard | 69.0 | +3.5 [-5.5, +12.5] (0.53) | 17 | 58.8 | -11.8 (2/4) | -17.6 | public loss |
| long_policy | 54.0 | +3.0 [-2.5, +9.0] (0.43) | 19 | 52.6 | -5.3 (1/2) | -12.2 | public loss |
| multi_hop |  |  | 18 | 55.6 | -5.6 (1/2) | -29.1 | public loss |
| ordinal |  |  | 12 | 100.0 | -50.0 (0/6) | -47.7 | public loss |
| policy | 81.0 | +2.5 [-3.0, +7.5] (0.47) | 12 | 83.3 | +0.0 (0/0) | -33.3 |  |
| probability |  |  | 10 | 40.0 | +30.0 (3/0) | +21.4 | not in the slice |
| routing |  |  | 12 | 91.7 | -58.3 (0/7) | -70.0 | public loss |
| routing_hard |  |  | 5 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| temporal_numeric | 78.0 | +2.0 [-3.5, +7.5] (0.62) | 15 | 13.3 | +13.3 (3/1) | +0.0 |  |
| tool_selection |  |  | 12 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| tradeoff |  |  | 6 | 83.3 | +0.0 (1/1) | -15.9 | not in the slice |
| trap |  |  | 8 | 87.5 | +12.5 (1/0) | -29.4 | not in the slice |

## Flags
- core: target_ref: arm present in the preds but not in screen.json jevkit.arms
- core: screen.json has no vs_stock_shim.target_ref
- scored checkpoint trainable-step0000200.safetensors (step 200) != selected {'step': 200, 'nll': 0.6150277999701741, 'accuracy': 0.7111111111111111, 'path': 'ckpt-step0000200.pt'}
- step-0 floor NOT beaten: +1.08 pp [-1.8333333333333313, 4.08541666666667]

## Warnings
- none

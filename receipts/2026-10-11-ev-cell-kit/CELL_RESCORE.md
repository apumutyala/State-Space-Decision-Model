# Cell rescore: native_shimcal vs stock_shim (AGREES)

Artifact `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\cellkit\x7c42-art` (screen.json sha 7fcd2d63a34e). jevkit CLI per arm (--public, declared price); paired stratified bootstrap B=1000 seed 16 (strata of the baseline table), Speed/Cost held fixed; tol 0.05.

## (1) PRIMARY
| quantity | value |
|---|---|
| Comp A primary / control | 76.161 / 76.570 |
| delta Comp A [95%] | -0.4091 [-1.7538, +1.3947] |
| 97.5% / 99% | [-2.0596, +1.5790] / [-2.2897, +1.7762] |
| one-sided p (P(delta<=0)) | 0.5650 (+1: 0.5654) |
| delta I [95%] | +0.644 [-2.52, +3.21] |
| success at 95% | False |
| in-job | -0.4091 [-1.7538, +1.3947] success False |
| screen.json diff | 95 fields, max abs diff 0.0, AGREES |

Holm family native_shimcal, q668q (alpha 0.05):

- native_shimcal: p 0.5650, rank 1, level 0.0250, tested True, CI at level [-2.060, +1.579], claimable **False**
- q668q: p 0.8100, rank 2, level 0.0500, tested False, CI at level –, claimable **False**

## (2) SECONDARY (EV-PREREG-SECONDARY-AXES; never turns a failed primary into a success)
- delta I: +0.64 [-2.52, +3.21]
- delta C: -2.13 [-6.92, +5.09]
- delta S: -0.57 [-0.57, -0.57] (held fixed)
- delta K: +0.00 [+0.00, +0.00] (held fixed)
- delta capability: -0.74 [-3.60, +3.33]
- delta composite_A: -0.41 [-1.75, +1.39]
- top-1 (jevkit correct) 204 vs 205: -0.43 pp [-2.16, +1.30], 2 / 3 discordant, McNemar p 1
- choice NLL 0.3138 vs 0.3068 (+0.0070 [-0.0247, +0.0415]); I<50 in {'a': 0.0, 'b': 0.0}
- choice: cc -1.14 [-4.66, +2.28]; top-1 -0.72 pp [-2.88, +1.44] (1/2, McN 1)
- noul: cc -3.01 [-9.02, +0.00]; top-1 -1.35 pp [-4.05, +0.00] (0/1, McN 1)
- score: cc +6.08 [+0.15, +10.51]; top-1 +5.56 pp [+0.00, +16.67] (1/0, McN 1)
- noul 0.2-0.8 band: native 16/74, native_shimcal 0/74, native_tfit 14/74, stock_shim 0/74

## (3) TARGET gap (paired on ids; S/K at each system's own value; cross-job: the target's S comes from its own job / hardware run and K from its own declared price; only I and C are paired on the same items)
| target | delta Comp A [95%] (primary - target) | control - target | S prim / S target | K prim / K target | caveat |
|---|---|---|---|---|---|
| h2o_4g6m8 | -0.62 [-1.96, +1.25] | -0.21 [-0.40, -0.05] | 90.77 / 92.12 | 61.23 / 61.13 | cross-job |

## (4) INTEGRITY
- items 231, same ids all arms True
- native: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- native_shimcal: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- native_tfit: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- stock_shim: 231 records / 231 unique, dup 0, missing 0, option order off 0, status {'ok': 231}, max input tokens 3952 (cap 4096)
- train no-truncation: {'rows': 2664, 'max_row_tokens': 3930, 't_cap': 4096}; cell {'train_steps': 333, 'max_steps': 333, 'best': {'step': 200, 'nll': 0.4565996776152714, 'accuracy': 0.8194444444444444, 'path': 'ckpt-step0000200.pt'}, 'stock_arm_identity': {}}

## (5) SELECTION
- best {'step': 200, 'nll': 0.4565996776152714, 'accuracy': 0.8194444444444444, 'path': 'ckpt-step0000200.pt'}; argmin NLL step 200 (= best: True); trained steps 333 / max 333; scored None (= selected: None)
| step | NLL | top-1 |
|---|---|---|
| 0 | 0.490837 | 0.8194 |
| 100 | 0.485235 | 0.8111 |
| 200 | 0.456600 | 0.8194 |
| 300 | 0.463988 | 0.8222 |
- step-0 floor: 0.8461 vs 0.8017 (3600 decisions), delta +4.44 pp [+3.47, +5.31] iid / [+3.50, +5.31] family-stratified, McNemar p 3.09e-19 (244 vs 84); slice ids sha 142a593a1257; in-job diff 0.0; pass True
- order (--mtimes C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-CHLabs\8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a\scratchpad\cellkit\x7c42-art\landing.json): selection cell.json 2026-10-11T14:51:16Z vs first public pred 2026-10-11T14:55:38Z (out/screen/jevkit/native.preds.jsonl): selection before public **True**; floor files before public False

## (6) G1 (reverse / shuffle / relabel; from files)
- not evaluated (no jevkit/g1 dir and no --g1)

## (7) FLIP battery
- ops (8/8): state_paraphrase, instruction_paraphrase, option_paraphrase, option_reverse, option_shuffle, key_relabel, whitespace_noise, markup_noise
- cell: average 2.44% [+1.26, +3.75]; state_paraphrase 1.0%, instruction_paraphrase 0.9%, option_paraphrase 1.3%, option_reverse 2.8%, option_shuffle 2.3%, key_relabel 4.3%, whitespace_noise 3.0%, markup_noise 3.9%
  - paired minus qwen35_4b_shim: average -5.11 pp [-7.27, -3.04]
- qwen35_4b_shim: average 7.55% [+5.71, +9.67]; state_paraphrase 2.9%, instruction_paraphrase 3.5%, option_paraphrase 6.9%, option_reverse 12.2%, option_shuffle 9.4%, key_relabel 8.6%, whitespace_noise 8.2%, markup_noise 8.7%
- reported baseline qwen35_4b_shim: 7.55% [+5.71, +9.67] (comparable to this cell's average: True)

## (8) TRANSFER (slice = selected - step 0; public = primary - control)
corr(slice delta, public delta correct) +0.153; corr(slice, public top-1) +0.153

| family | slice step-0 | slice delta pp [CI] (McN) | public n | control top-1 | public delta top-1 pp (+/-) | public delta I | flag |
|---|---|---|---|---|---|---|---|
| adequacy | 70.0 | +8.5 [+3.5, +13.5] (0.0015) | 12 | 91.7 | +0.0 (0/0) | +0.0 | GAIN SLICE ONLY |
| adversarial | 67.5 | +14.5 [+8.5, +20.0] (2.4e-06) | 6 | 100.0 | +0.0 (0/0) | +0.6 | GAIN SLICE ONLY |
| ambiguous | 91.5 | +8.0 [+4.5, +12.0] (3.1e-05) | 7 | 85.7 | +0.0 (0/0) | +0.0 | GAIN SLICE ONLY |
| enum | 99.5 | +0.0 [+0.0, +0.0] (1) |  |  |  |  | not on public |
| extraction |  |  | 24 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| fact | 98.5 | -0.5 [-1.5, +0.0] (1) | 12 | 100.0 | +0.0 (0/0) | +0.0 |  |
| intent | 97.5 | +0.5 [+0.0, +1.5] (1) | 24 | 100.0 | +0.0 (0/0) | +0.0 |  |
| judge_hard | 69.5 | +0.0 [-4.0, +3.5] (1) | 17 | 70.6 | -5.9 (0/1) | -11.8 | public loss |
| long_policy | 86.0 | +1.5 [-2.0, +5.0] (0.55) | 19 | 73.7 | +0.0 (0/0) | +2.2 |  |
| multi_hop | 81.0 | +3.5 [+1.0, +6.5] (0.016) | 18 | 83.3 | +5.6 (1/0) | -1.9 | gain on both |
| ordinal |  |  | 12 | 100.0 | +0.0 (0/0) | +4.5 | not in the slice |
| policy | 74.0 | +7.0 [+2.0, +12.0] (0.0094) | 12 | 100.0 | +0.0 (0/0) | +0.0 | GAIN SLICE ONLY |
| probability | 69.5 | +3.5 [-2.0, +9.0] (0.3) | 10 | 100.0 | -10.0 (0/1) | -10.7 | public loss |
| routing | 84.0 | +2.0 [-2.0, +6.0] (0.48) | 12 | 100.0 | +0.0 (0/0) | +0.0 |  |
| routing_hard |  |  | 5 | 100.0 | +0.0 (0/0) | +0.0 | not in the slice |
| severity | 74.5 | +3.5 [-1.0, +8.0] (0.17) |  |  |  |  | not on public |
| state_tracking | 83.5 | -0.5 [-4.0, +3.0] (1) |  |  |  |  | not on public |
| temporal_numeric | 58.5 | +3.0 [-2.0, +8.0] (0.34) | 15 | 33.3 | +0.0 (1/1) | +5.0 |  |
| tool_selection | 99.5 | +0.0 [+0.0, +0.0] (1) | 12 | 100.0 | +0.0 (0/0) | +0.0 |  |
| tradeoff | 54.5 | +15.5 [+9.5, +22.0] (3.1e-06) | 6 | 83.3 | +0.0 (0/0) | +0.0 | GAIN SLICE ONLY |
| trap | 84.0 | +10.0 [+5.0, +15.0] (0.00018) | 8 | 100.0 | +0.0 (0/0) | +5.9 | GAIN SLICE ONLY |

## Flags
- none

## Warnings
- identity-child.json missing: the scored checkpoint cannot be tied to the selected step (pass --identity-child)
- no G1 dir (jevkit/g1 or --g1): G1 not evaluated

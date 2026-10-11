# Local JevBench scorecard (EV-PREREG-COMPOSITE-LOCAL)

**Caveat (every number): open public items only; the board's sealed split may rank differently (open->board I gap measured on H2O: -17.6).** Local Composite is never compared with the board's 72.5.

**Verdict:** no arm beats H2O-Lightning-4B v1.1 on Composite A under the rule. CI>0 on Composite A: none; on Capability: none. Best paired Δ Composite A: **torchcast-decision-12b (double-bos)** -3.6 [-6.6, -0.8]; best paired Δ Capability: **torchcast-decision-12b (double-bos)** -1.0 [-7.7, 5.2].

- Best of OUR candidates (our trained cells + our serving configs on stock hosts; excludes the external systems H2O, decider, torchcast, d1): Δ Composite A **qwen3.5-4b+h2o_shim** -9.4 [-23.1, -4.2]; Δ Capability **D19 (rwkv7-13.3b+joint)** -17.2 [-26.6, -10.7].
- Best of OUR TRAINED cells (D19, sizecliff 4B/2B/0.8B arms): Δ Composite A **D19 (rwkv7-13.3b+joint)** -14.4 [-31.7, -10.6]; Δ Capability **D19 (rwkv7-13.3b+joint)** -17.2 [-26.6, -10.7].

Baseline: `h2o-lightning-4b` (job-zbtbj, our run, pinned 193ad740; kit parity 205/231 = card). Protocol: jevkit CLI per arm (--public, own declared price); paired stratified bootstrap B=1000 seed 16 (strata of the baseline table) on the same 231 ids; S/K held at each arm's own axes (composite_A) or S held at the baseline's S (composite_A_commonS). 28 arms, 231 public items, same ids in every arm. Flags: 2; max |diff| vs each in-job report: 0.0.

| # | arm (job) | I | C | S | K | Capability [CI] | Composite A [CI] | ΔComp A vs H2O [CI] | ΔCap vs H2O [CI] | ΔComp A, S=H2O's [CI] | price class ($/M in) | tok/dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | h2o-lightning-4b (job-zbtbj) | 77.6 | 83.4 | 91.3 | 61.1 | 80.5 [72.6, 84.1] | 76.6 [72.8, 78.3] | baseline | baseline | baseline | qwen35_4b (0.03, base_model_reference) | 654 |
| 2 | torchcast-decision-12b (double-bos) (job-r3uc4) | 76.4 | 82.6 | 92.1 | 53.5 | 79.5 [71.4, 84.5] | 73.1 [69.4, 75.1] | -3.6 [-6.6, -0.8] (CI<0) | -1.0 [-7.7, 5.2] | -3.7 [-6.7, -0.9] (CI<0) | gemma12b_ref (0.05, base_model_reference) | 705 |
| 3 | torchcast-decision-12b (plain) (job-r3uc4) | 76.0 | 82.0 | 91.7 | 53.6 | 79.0 [71.3, 83.8] | 72.8 [69.3, 74.8] | -3.8 [-6.8, -0.9] (CI<0) | -1.5 [-8.2, 4.9] | -3.9 [-6.8, -1.0] (CI<0) | gemma12b_ref (0.05, base_model_reference) | 704 |
| 4 | decider-12b-v1 (job-r3uc4) | 72.5 | 80.9 | 92.0 | 54.8 | 76.7 [68.8, 81.2] | 72.4 [68.6, 74.4] | -4.3 [-7.8, -1.1] (CI<0) | -3.8 [-11.2, 3.0] | -4.4 [-7.9, -1.2] (CI<0) | gemma12b_ref (0.05, base_model_reference) | 641 |
| 5 | decider-12b v2 (job-r3uc4) | 70.8 | 77.9 | 90.1 | 54.8 | 74.3 [66.0, 80.3] | 71.0 [66.9, 73.7] | -5.6 [-9.3, -1.9] (CI<0) | -6.1 [-14.1, 1.9] | -5.4 [-9.1, -1.7] (CI<0) | gemma12b_ref (0.05, base_model_reference) | 641 |
| 6 | qwen3.5-4b+h2o_shim (job-zbtbj) | 56.3 | 69.3 | 91.6 | 61.1 | 62.8 [53.3, 70.1] | 67.3 [52.0, 71.5] | -9.4 [-23.1, -4.2] (CI<0) | -17.7 [-26.5, -8.2] (CI<0) | -9.4 [-23.1, -4.2] (CI<0) | qwen35_4b (0.03, base_model_reference) | 654 |
| 7 | gemma-4-e4b+v3w (job-zbtbj) | 58.9 | 57.4 | 91.0 | 64.0 | 58.2 [45.6, 67.9] | 65.6 [42.3, 71.3] | -11.1 [-33.5, -4.4] (CI<0) | -22.3 [-33.6, -10.1] (CI<0) | -11.0 [-33.5, -4.4] (CI<0) | ref_0p02 (0.02, base_model_reference) | 783 |
| 8 | D19 (rwkv7-13.3b+joint) (job-9j3uu) | 52.6 | 74.0 | 78.6 | 52.6 | 63.3 [52.4, 67.9] | 62.3 [44.1, 64.9] | -14.4 [-31.7, -10.6] (CI<0) | -17.2 [-26.6, -10.7] (CI<0) | -12.6 [-30.4, -8.8] (CI<0) | le13b (0.05, size_class_estimate) | 759 |
| 9 | sizecliff-4B native (job-27anq) | 51.0 | 59.6 | 85.2 | 61.2 | 55.3 [45.7, 61.1] | 62.1 [41.2, 65.7] | -14.6 [-34.2, -10.1] (CI<0) | -25.2 [-32.7, -17.4] (CI<0) | -13.8 [-33.7, -9.3] (CI<0) | sizecliff_4B (0.03, size_class_estimate) | 654 |
| 10 | sizecliff-4B native_shimcal (job-27anq) | 53.6 | 54.8 | 85.2 | 61.2 | 54.2 [44.4, 61.0] | 61.6 [44.3, 65.7] | -15.1 [-30.5, -10.2] (CI<0) | -26.3 [-33.7, -17.6] (CI<0) | -14.3 [-30.0, -9.3] (CI<0) | sizecliff_4B (0.03, size_class_estimate) | 654 |
| 11 | sizecliff-4B lean (job-27anq) | 49.7 | 55.2 | 85.4 | 65.9 | 52.4 [43.3, 59.2] | 60.6 [36.0, 65.9] | -16.1 [-39.9, -10.1] (CI<0) | -28.1 [-36.1, -19.1] (CI<0) | -15.4 [-39.5, -9.3] (CI<0) | sizecliff_4B (0.03, size_class_estimate) | 457 |
| 12 | sizecliff-4B v3w (job-27anq) | 48.8 | 71.0 | 85.2 | 58.9 | 59.9 [49.8, 64.0] | 60.3 [38.1, 66.0] | -16.3 [-37.7, -9.8] (CI<0) | -20.6 [-29.4, -14.8] (CI<0) | -15.5 [-37.3, -9.0] (CI<0) | sizecliff_4B (0.03, size_class_estimate) | 783 |
| 13 | qwen3.5-4b+v3w (job-zbtbj) | 46.1 | 73.3 | 91.7 | 58.8 | 59.7 [48.6, 65.1] | 53.7 [24.5, 67.4] | -23.0 [-50.0, -8.5] (CI<0) | -20.8 [-30.0, -13.3] (CI<0) | -23.0 [-50.0, -8.5] (CI<0) | qwen35_4b (0.03, base_model_reference) | 783 |
| 14 | rwkv7-g1k-13.3b+v3w (job-zbtbj) | 43.4 | 82.0 | 86.1 | 52.6 | 62.7 [54.7, 63.2] | 45.6 [33.9, 61.1] | -31.0 [-42.5, -14.9] (CI<0) | -17.8 [-26.6, -12.4] (CI<0) | -30.5 [-42.2, -14.3] (CI<0) | le13b (0.05, size_class_estimate) | 759 |
| 15 | rwkv7-g1k-13.3b+v3w (9j3uu) (job-9j3uu) | 43.4 | 82.8 | 83.4 | 52.6 | 63.1 [54.9, 63.5] | 45.6 [34.3, 59.9] | -31.1 [-42.4, -15.6] (CI<0) | -17.4 [-26.3, -11.9] (CI<0) | -30.3 [-41.9, -14.6] (CI<0) | le13b (0.05, size_class_estimate) | 760 |
| 16 | sizecliff-2B native_shimcal (job-haxmm) | 43.0 | 54.0 | 87.5 | 66.5 | 48.5 [37.4, 56.1] | 43.3 [21.7, 64.2] | -33.3 [-54.4, -11.5] (CI<0) | -32.0 [-43.0, -21.7] (CI<0) | -33.0 [-54.3, -11.0] (CI<0) | sizecliff_2B (0.02, size_class_estimate) | 654 |
| 17 | qwen3.5-4b+m0 (job-zbtbj) | 42.3 | 68.9 | 89.2 | 58.6 | 55.6 [46.1, 62.1] | 43.1 [21.9, 65.1] | -33.5 [-54.2, -10.9] (CI<0) | -24.9 [-33.3, -16.5] (CI<0) | -33.4 [-54.1, -10.7] (CI<0) | qwen35_4b (0.03, base_model_reference) | 793 |
| 18 | d1-3b (job-zbtbj) | 39.7 | 66.2 | 94.3 | 58.7 | 53.0 [43.9, 59.0] | 37.2 [18.7, 62.6] | -39.4 [-57.0, -13.2] (CI<0) | -27.5 [-35.7, -19.1] (CI<0) | -39.6 [-57.1, -13.5] (CI<0) | d1_list (0.04, size_class_estimate) | 592 |
| 19 | sizecliff-0.8B native_shimcal (job-reknq) | 39.1 | 51.9 | 87.2 | 75.5 | 45.5 [35.1, 51.9] | 35.2 [18.5, 57.4] | -41.5 [-57.5, -18.1] (CI<0) | -35.0 [-44.9, -26.0] (CI<0) | -41.2 [-57.4, -17.7] (CI<0) | sizecliff_0.8B (0.01, size_class_estimate) | 654 |
| 20 | nemotron-3-nano-4b+v3w (job-zbtbj) | 36.3 | 66.3 | 83.5 | 58.6 | 51.3 [40.6, 56.9] | 29.4 [12.7, 50.5] | -47.2 [-63.3, -24.5] (CI<0) | -29.2 [-38.9, -20.2] (CI<0) | -46.8 [-63.1, -23.7] (CI<0) | le4b (0.03, size_class_estimate) | 793 |
| 21 | sizecliff-0.8B native (job-reknq) | 34.3 | 61.8 | 87.2 | 75.5 | 48.1 [38.4, 51.7] | 26.8 [15.9, 38.4] | -49.8 [-60.0, -36.7] (CI<0) | -32.4 [-42.2, -25.3] (CI<0) | -49.6 [-59.9, -36.4] (CI<0) | sizecliff_0.8B (0.01, size_class_estimate) | 654 |
| 22 | sizecliff-2B native (job-haxmm) | 34.9 | 56.3 | 87.5 | 66.5 | 45.6 [37.7, 52.1] | 26.8 [16.0, 49.3] | -49.8 [-61.1, -26.9] (CI<0) | -34.9 [-42.5, -25.5] (CI<0) | -49.6 [-61.0, -26.5] (CI<0) | sizecliff_2B (0.02, size_class_estimate) | 654 |
| 23 | qwen3.5-2b+v3w (job-zbtbj) | 32.3 | 66.2 | 92.6 | 64.0 | 49.3 [38.7, 51.9] | 23.0 [13.0, 37.3] | -53.6 [-63.6, -38.3] (CI<0) | -31.2 [-41.8, -25.3] (CI<0) | -53.7 [-63.6, -38.4] (CI<0) | ref_0p02 (0.02, base_model_reference) | 783 |
| 24 | lfm2.5-2.6b+v3w (job-zbtbj) | 31.8 | 55.6 | 94.0 | 59.2 | 43.7 [34.0, 50.4] | 21.0 [10.3, 37.6] | -55.7 [-66.1, -38.9] (CI<0) | -36.8 [-46.2, -27.8] (CI<0) | -55.7 [-66.1, -39.1] (CI<0) | le4b (0.03, size_class_estimate) | 757 |
| 25 | sizecliff-2B v3w (job-haxmm) | 31.0 | 66.6 | 87.4 | 64.2 | 48.8 [40.0, 51.9] | 20.6 [11.2, 33.9] | -56.0 [-65.1, -41.0] (CI<0) | -31.7 [-40.3, -25.5] (CI<0) | -55.9 [-65.1, -40.8] (CI<0) | sizecliff_2B (0.02, size_class_estimate) | 783 |
| 26 | sizecliff-0.8B v3w (job-reknq) | 30.6 | 62.2 | 87.1 | 73.2 | 46.4 [36.5, 48.7] | 20.2 [11.6, 29.9] | -56.4 [-64.9, -45.6] (CI<0) | -34.1 [-43.8, -28.4] (CI<0) | -56.3 [-64.8, -45.3] (CI<0) | sizecliff_0.8B (0.01, size_class_estimate) | 783 |
| 27 | qwen3.5-2b+m0 (job-zbtbj) | 30.7 | 63.6 | 90.3 | 63.9 | 47.2 [37.6, 51.6] | 20.1 [11.2, 33.2] | -56.6 [-65.4, -42.1] (CI<0) | -33.3 [-43.0, -25.8] (CI<0) | -56.5 [-65.3, -42.0] (CI<0) | ref_0p02 (0.02, base_model_reference) | 793 |
| 28 | rwkv7-g1i-2.9b+v3w (job-zbtbj) | 29.5 | 67.2 | 88.6 | 59.2 | 48.4 [39.8, 52.3] | 18.2 [9.9, 31.4] | -58.5 [-66.8, -44.3] (CI<0) | -32.1 [-41.6, -23.5] (CI<0) | -58.4 [-66.7, -44.2] (CI<0) | le4b (0.03, size_class_estimate) | 759 |

Δ columns: paired candidate − H2O on the same 1,000 stratified resamples; **CI>0** = the 95% CI excludes 0 in our favour (the rule's success condition); (CI<0) = H2O significantly better. Capability = f(I, C) only, so it is independent of latency and price.

## Caveats per row
- **Speed comparability:** every arm's S comes from `pred latency_s` on the 72 open standard+judge items, in-process wall time on 1x NVIDIA H100 80GB HBM3 with the self-hosted x2 + 0.15 s adjustment, but measured in DIFFERENT jobs (zbtbj, 9j3uu, 27anq, haxmm, reknq, r3uc4: different processes, readouts and batch paths). Same hardware class, not the same measurement session, so S differences of a few points may be measurement path. The `ΔComp A, S=H2O's` column holds S at H2O's measured S for every arm (isolates I, C, K); where it and the own-S Δ disagree in sign of significance, the latency path decides the row.
- **Price class** is each system's own declaration (column): zbtbj host-scan price.json per candidate; sizecliff jobs use the A-PREREG-SIZECLIFF declared $/M input with output 0; 9j3uu and r3uc4 their own price.json. size_class_estimate rows are estimates, not list tariffs.
- **Kit parity:** torchcast-decision-12b failed kit parity (198 vs 203); its rows are reported, not comparable as that system's true score.
- **qwen3.5-4b+h2o_shim** is stock Qwen3.5-4B served with H2O's vendored shim readout (no training of ours); it is the ledger's 'stock 4B + shim, local Composite 67.3' arm.
- **RWKV7-13.3B stock** appears twice (host scan zbtbj and D19's in-job parent 9j3uu): I identical 43.4, C/S differ by run (82.0/86.1 vs 82.8/83.4), a direct measure of cross-job S noise (~2.7 points).
- **Open public items only**; the board's sealed split may rank differently (open→board I gap on H2O −17.6).

Latency sources (job, lane, n timed): job-27anq self-hosted (x2 + 0.15 s) n=72; job-9j3uu self-hosted (x2 + 0.15 s) n=72; job-haxmm self-hosted (x2 + 0.15 s) n=72; job-r3uc4 self-hosted (x2 + 0.15 s) n=72; job-reknq self-hosted (x2 + 0.15 s) n=72; job-zbtbj self-hosted (x2 + 0.15 s) n=72

Files: SCORECARD.json (all axes, CIs, pairs, sha256 of every preds/price/report), arms.json (manifest), reports/ (per-arm jevkit reports), prices/ (sizecliff price classes from each screen.json), scorecard.png, SHA256SUMS. Code: scripts/ev_scorecard.py, scripts/ev_scorecard_report.py.

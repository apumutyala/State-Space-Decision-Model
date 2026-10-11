# EV-PREREG-H2O-HOST eval-only arms: job-4g6m8 (2026-10-11), H2O-Lightning-4B v1.1 @193ad740

Branch claude/lead-h2o-eval @c1936746, mode `h2o_eval_v11`, one H100, art-jrasm (sha256 8feba886...5181), exit 0,
wall 9 min. Revision verified (snapshot commit = HF-resolved sha = 193ad740; all 12 file pins). LICENSE + README copied
(no NOTICE file exists at this revision). Held-out slice: jev2a, 1,000 ids, ids sha256 204e9bfe...2a9c (0 failed rows).

Open public items only; the board's sealed split may rank differently (open->board I gap measured on H2O: -17.6).

| arm | Composite A | I | C | S | K | top-1 | noul 0.2-0.8 band | Δ A vs control [95% CI] |
|---|---|---|---|---|---|---|---|---|
| h2o (control, own shim, T 0.8) | 76.78 | 77.56 | 83.42 | 92.12 | 61.13 | 205/231 | 0.0% | - |
| a_T (T_J 0.676, slice-fit) | 77.09 | 78.00 | 84.39 | 92.12 | 61.13 | 205 | 0.0% | +0.31 [-0.44, +1.19] |
| a_Ttype (0.693 / 0.623 / 0.777) | 76.85 | 77.65 | 83.62 | 92.12 | 61.13 | 205 | 0.0% | +0.07 [-0.74, +0.88] |
| a_Tfam (T_f per jev2a family) | 77.29 | 78.03 | 85.31 | 92.12 | 61.13 | 205 | 0.0% | +0.51 [-0.21, +1.35] |
| b_native (T 1, no floor) | 68.10 | 53.87 | 77.18 | 92.12 | 61.13 | 205 | 43.2% | -8.68 [-18.39, -5.94] |
| b_shimcal (== control algebraically) | 76.78 | 77.56 | 83.42 | 92.12 | 61.13 | 205 | 0.0% | 0.00 [0, 0] |
| c_symfork (G1 probe, NOT a contender) | 14.11 | 29.65 | 26.51 | 83.98 | 61.19 | 121 | 13.5% | -62.67 [-72.62, -41.37] |

Primary (EV-PREREG-COMPOSITE-LOCAL): no contender's CI excludes 0. All the gains come from Calibration, through the
post-processing; top-1 is unchanged. G1: c_symfork 0 flips, max |dp| 0, rows bit-identical on all 3 operators (PASS).
The control has 6 / 6 / 5 flips (relabel / reverse / shuffle) out of 139 / 213 / 213. Full numbers (per-type
breakdown, McNemar, NLL, quantiles): decision.json. Pred sha256s: SHA256SUMS.

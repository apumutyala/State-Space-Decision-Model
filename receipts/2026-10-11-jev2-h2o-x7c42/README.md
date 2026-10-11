# JEV2-H2O landing: job-x7c42 (art-cwk49, tar sha256 0c538a61...)

Mode `jev2_h2o_shim`, stack/qwen35/jev2-h2o @48069603, A-PREREG-JEV2-H2O + amendments 1-3. Billed $2.66 (2,530 s).

- PRIMARY (native_shimcal vs the in-job H2O v1.1 control, public-231, B 1000 seed 16): delta Composite A -0.41
  [-1.75, +1.39] (76.16 vs 76.57). NOT a success (CI spans 0). Per axis: delta I +0.64 [-2.52, +3.21], delta C -2.13
  [-6.92, +5.09], delta Capability -0.74 [-3.60, +3.33]; S 90.77 vs 91.34, K equal.
- Selection: step 200 of 333 by dev-card NLL on 360 rows of slice 142a593a (0.4566 vs step-0 0.4908); selection
  frozen in cell.json at 14:51:16Z, before any public-231 pred (native 14:55:38Z, stock 15:01:05Z, shimcal 15:05:08Z).
- Step-0 floor (whole slice, 3,600 decisions, 0 unmatched): top-1 0.850 vs 0.800, paired delta +4.44 pp
  [+3.44, +5.39] (B 1000 seed 16, computed at landing from slice_floor_rows.jsonl). PASS.
- Per type (public-231, delta I of native_shimcal - control): choice -1.14 [-5.68, +2.29], noul -3.01 [-9.02, 0.00],
  score +6.08 [+0.09, +10.74].
- Raw readout without the shim T / floor (`native`): delta Composite A -3.18 [-6.13, -1.37], noul band 16/74 (control 0)
  -> the noul band gate fails for the raw arm; native_shimcal 0/74.
- tfit T 0.93 (band-constrained no). G1 (shim mode, reported only): fails (labelled prompt not invariant).
- Per-family deltas: landing.json (by_family_public, slice_floor.per_family).

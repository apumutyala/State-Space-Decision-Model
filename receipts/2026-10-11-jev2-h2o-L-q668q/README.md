# JEV2-H2O-L landing: job-q668q (art-5si2g, tar sha256 91287177...)

Mode `jev2_h2o_shim_l`, stack/qwen35/jev2-h2o @1e12ec12, A-PREREG-JEV2-H2O-L (+ Holm, amendment 3). Billed $9.39 (8,940 s).

- PRIMARY (native_shimcal vs the in-job H2O v1.1 control, public-231, B 1000 seed 16): delta Composite A -0.97,
  95% CI [-3.38, +1.12], 97.5% CI [-3.73, +1.41] (74.58 vs 77.56 I). NOT a success; with x7c42 (-0.41 [-2.06, +1.58]
  at 97.5%) Holm claims neither arm.
- Per axis: delta I -2.98 [-8.26, +1.89], delta C -0.59, delta Capability -1.79; per type delta I: choice -3.42,
  noul -0.63, score -4.90.
- Training: the step-10 rate check measured 3.93 s/step (vs 2.55 in job-9fy9p on the 2,664-row draw), so the plan was
  shortened to 1,832 of 2,244 steps (ran 1,845); selected step 750 by dev-card NLL (0.4031 vs step-0 0.4908). Selection
  frozen in cell.json at 18:27:42Z, before any public-231 pred (native 18:32:58Z).
- Step-0 floor (3,600 slice decisions): top-1 0.890 vs 0.800, paired delta +8.97 pp [+7.69, +10.22]. PASS.
- Slice-vs-public transfer per family (landing.json transfer_by_family): correlation -0.32 (x7c42: +0.15). Slice gains
  do not transfer to public-231.
- noul band: native_shimcal 0/74, raw native 16/74. tfit T 0.91. G1 (reported) fails.

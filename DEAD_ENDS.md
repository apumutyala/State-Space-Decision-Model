# DEAD_ENDS: falsified ideas (don't retry without new evidence)

- **Q9-AF full 8-layer GDN conversion (job-2xe5w):** L3 and L27 fail hidden_cos 0.95. The 6-layer hybrid loses 26 Index pts and 12 MMLU pts. Closed under rule v1 (#140).
- **Order-debias calibration on stock RWKV7 (PR #151):** the "first-position bias" is label bias. Calibration lowers the Index by 2–3 pts.
- **Frozen-feature readout heads on Qwen3.5-9B (proxy loop R00–R17):** best 39.24 on SELECT, far from 50. KD heads add only +0.4.

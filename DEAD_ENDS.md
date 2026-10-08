# DEAD_ENDS: falsified ideas (don't retry without new evidence)

- **Q9-AF full 8-layer GDN conversion (job-2xe5w):** L3 and L27 fail hidden_cos 0.95. The 6-layer hybrid loses 26 Index pts and 12 MMLU pts. Closed under rule v1 (#140).
- **Order-debias calibration on stock RWKV7 (PR #151):** the "first-position bias" is label bias. Calibration lowers the Index by 2–3 pts.
- **Frozen-feature readout heads on Qwen3.5-9B (proxy loop R00–R17):** best 39.24 on SELECT, far from 50. KD heads add only +0.4.

- 2026-10-08 (A): Plain Clef-flash KD on injected rows. The teacher's top answer is the attacker target on 23% of `d1a` rows and 14% of `hijack` rows, so KD trains the student toward following injections. D2 (job-zhv3f) runs without the mask, so read any change in its steer score as possibly caused by the teacher. Future KD cells mask injected rows: branch stack/rwkv7/kd-injmask.

- **C historical language folds (2026-10-08 triage, no new run):** PR #67 fixed feature scorer transferred only 100/300; #69 byte-SSM paraphrase curriculum kept 0/100 comparisons; #72 comparison curriculum regressed 126/180→114/180; #76 fixed read replay lost new-family comparison; #81 typed-emission recipe fell to 519/1080 with only 29/360 comparisons. Preserve original receipts/checkpoints and do not rerun unchanged. #82/#83 improvements were mixed/inconclusive, not qualification. Full operator closure list is plans/codex/LANGUAGE_PR_TRIAGE_20261008.md; no PR or artifact was deleted.

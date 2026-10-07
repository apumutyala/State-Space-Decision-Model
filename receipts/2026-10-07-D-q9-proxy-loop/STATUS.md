# Status: q9 proxy loop — PAUSED before `select` (2026-10-07)

**State.**
- **stock-check:** passed under A3. The masked delta is +0.047 against a tolerance of 0.3.
- **SELECT receipts:** all 15 stock-arm recipes have one: R00–R08, R11, R12 and R14–R17 (`select/`, `select/select_table.md`).
- **Missing:** R09 needs the e5 feature cache; R10 and R13 need the conv1 cache. Neither cache exists yet.
- **Not run:** `select`, the HOLDOUT, `select_summary.json`, `holdout_attempt.json` and `verdict.json`.

**Why the loop is paused** (coordinator decision, 2026-10-07):
- The best SELECT score is R06 at 39.24. An e5 or conv1 recipe would need about +11 points to reach the GO threshold of 50.0.
- Any likely HOLDOUT result leads to the same action, an escalation to the user.
- So spending about $13.5 (expected cost) on the four missing cache parts is not actionable now.
- The coordinator is holding those four jobs and has not submitted them.

**Launch packages, ready to resume.** All four were built by `python -m jobs.common.launch q9_feature_cache --mode <part>` at agent-main a37a237c292b, which contains A3 (d7f1290). Each passed its gate: 148 tests passed, 0 failed, 2 allowed skips, no lint waivers.

| Part | Tarball | sha256 | Size | Worst case | Hard cap |
|---|---|---|---|---|---|
| e5_dev | `C:\Users\apumu\q9fc_e5_dev\q9_feature_cache-e5_dev-a37a237c292b.tar.zst` | `52a42f7e8af3ac9b30332132841560cb5c57cecd5065cac83575f4ebd1e4c60f` | 23,765,499 B | $15.35 | $30 |
| e5_train | `C:\Users\apumu\q9fc_e5_train\q9_feature_cache-e5_train-a37a237c292b.tar.zst` | `0ccd2785b1e794291b40c0e0f54311b84c2b30ec675805bd82f070a55a68208d` | 417,497,487 B | $12.87 | $30 |
| conv1_dev | `C:\Users\apumu\q9fc_conv1_dev\q9_feature_cache-conv1_dev-a37a237c292b.tar.zst` | `5b82347603b7dce6c5bf8e7b5dbd944bf55fb78feb8d861d031206c03cc1853f` | 273,639,676 B | $15.35 | $30 |
| conv1_train | `C:\Users\apumu\q9fc_conv1_train\q9_feature_cache-conv1_train-a37a237c292b.tar.zst` | `737d34756a5937df20ef4b4fec92bf0fb6cd4e9ac470df5e4c5e3456c9f3ef2f` | 667,350,372 B | $12.87 | $30 |

The `submit_job` arguments for each part are in its `submit.json`.

**How the conv1 packages get the converted layer.**
- `gdn_converted.pt` is extracted from art-2bybh. That artifact's sha256 is dd491ade…, and it matches the platform record.
- The file goes into the context at `data/gdn/gdn_converted.pt`, with sha256 8a50c36c….
- The pin is checked three times:
  - at build time (`build_context.sh:52`);
  - recorded in `CONTEXT_MANIFEST.json` `data_sha256`;
  - in the job before extraction (`cache_job.sh:121`, exit 10 on mismatch).

**Resume order** (unchanged):
1. Submit the four parts.
2. Download and verify each part, then unpack it into the same cache root as stock.
3. Run `recipes --only R09,R10,R13` into `C:\Users\apumu\q9cache\q9fcl_run`. Every completed receipt is reused.
4. Run `select`.
5. Run the single-shot HOLDOUT.

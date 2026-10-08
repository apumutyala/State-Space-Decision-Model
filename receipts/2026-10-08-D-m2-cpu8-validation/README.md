# M2 first-cell CPU-8 staged package validation

This receipt covers the staged launcher/test package only. It is not a model-training or model-evaluation result.

- Source: merged `agent-main@11fb140ce64802222d8abe796267e22e14fe4959`.
- Prepared workload context: `m2_first_cell-11fb140ce648.tar.zst`, 469,179,636 bytes, SHA-256 `be4c1d3893358067c82962c077e5d7eadfaf74a27c3a0dd2b66d34e94953e6b7`. External path: `C:\Users\apumu\research\CHLabs\ssdm-data\prepared-contexts\m2-first-cell-11fb140-be4c1d3893358067\`.
- CPU package: `agent-main-11fb140` source, M2 test runner and pinned inputs; 9 files / 498,078,038 bytes. Full package and individual pins are in `staging.json`.
- Hardware: SF Compute `cpu-8`; returned rate `$0.00804/min`, under the `$0.012/min` stop threshold.
- Node: `ssdm-codex-m2cpu8-ae1d800871324d77960377a94e1532c2` / `ba0ebac6-b214-430f-9217-dd9d47fdd4ed`; stopped by controller.
- Exact remote command id: `cmd-xgjgs`. Controller elapsed 2,532 seconds; workload RC 0; export RC 0; retries 0; final output imported and hashes verified.
- Test result from `submit.json` and recursive JUnit parse: 153 passed, 0 failed, 0 errors, 4 skipped. All four skips require a Git checkout and are enumerated in `submit.json`; `skipped_other_pins=[]`. JUnit's top-level nested-suite attributes are zero, so use its testcase records plus the launcher summary rather than its root counters.
- Actual CPU-8 node charge: `$0.2634` for 32.78 minutes, from post-run `usage.json`. Cumulative CPU-8 metered charge after the prior three attempts: `$0.4490` (`$0.1856 + $0.2634`). October SF MTD from that snapshot: `$57.982162`; available balance: `$467.013684`. No active nodes/jobs in the post-run inventory.
- No GPU, model weights, optimizer steps, H100 request, or model predictions were involved. The completed tests prove the staged package/launcher path; they do not prove training success, decision quality, safety, or superiority over Clef.
- Full outer archive: `launch-output.tar`, 562,452,480 bytes, SHA-256 `7154ced963f52ab35e05bd91b7c0cf0674dd16f5a9d7ec54ba4751288e9f8fae`, retained externally beside the working receipt at `C:\Users\apumu\research\CHLabs\ssdm-data\receipts\m2-cpu8-validation-20261008-run4\`.

The next eligible step is to refresh issue #169 with this merged-source context and exact proof, then independently review its exact H100 bundle and run fresh preflight/`validate_job`. This receipt alone does not authorize the H100 run.
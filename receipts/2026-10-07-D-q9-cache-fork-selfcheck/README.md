# q9 feature cache: self-check v2 receipt (2026-10-07)

Why: job-8vjve and job-b7d3n stopped at the v1 self-check (exit 2). This receipt contains:
- `job-8vjve_selfcheck_v1.json`: the job's v1 numbers and environment (H100, torch 2.13.0+cu126, transformers 5.17.0, fla 0.5.2).
- `probe_bf16_noise_qwen35_08b.json` (`scripts/q9_cache_bf16_noise_probe.py`): real Qwen3.5-0.8B, CPU, 4 SELECT DEV requests, 12 options.
  - fp32: cached fork == plain forward to 1.5e-3, which is the bf16 *storage* of the fork feature (LM 0-1e-5).
  - bf16: cached and plain are each ~0.02 from the fp32 truth; between themselves 0.018-0.029.
- `selfcheck_v2_bug_injection_qwen35_08b.json` (`scripts/q9_cache_selfcheck_bug_probe.py`): the v2 gate (`extract.selfcheck`, constants in `jobs/q9_feature_cache/cache_config.json`) on the same model and rows, for the exact path and four injected fork bugs:

| Path | v2 verdict | Logic: fork rel (fp32) | Logic: LM (fp32) | v1 metric (bf16 vs bf16) |
|---|---|---|---|---|
| exact | PASS | 2.4e-6 | 4.9e-6 | 0.029 |
| conv states zeroed | FAIL | 3.1e-2 | 9.8e-2 | 0.045 |
| recurrent states zeroed | FAIL | 3.4e-1 | 2.5e-1 | 0.345 |
| recurrent states x 0.5 | FAIL | 1.2e-1 | 1.5e-1 | 0.129 |
| last KV position dropped | FAIL | 6.9e-3 | 1.1e-2 | 0.032 |

Logic tolerance: fork 1e-3 and LM 1e-3 per continuation token. The cache check passes only if the production bf16 path's distance to the fp32 truth is ≤ 1.5x the plain bf16 forward's, plus 0.005 (fork) and 0.02 per token (LM). The measured fork ratio on the exact path is 0.85-1.08.

## v3 (after the PR #111 review)

`selfcheck_v3_bug_injection_qwen35_08b.json` re-runs `scripts/q9_cache_selfcheck_bug_probe.py` with the v3 gate on the same model and requests. v3 adds:
- every row of each request (up to 2);
- options {0, K//2, K-1}: 12 fork checks;
- q_end / q_pre / e vs the plain fp32 hidden states;
- the fla fp32 pass (record-only). On CPU it runs the same torch kernels as the truth, so it equals the logic number here.

| Path | v3 verdict | Logic: fork rel (fp32) | Logic: LM (fp32) | q_end / q_pre / e rel (max) | Cache fork ratio |
|---|---|---|---|---|---|
| exact | PASS | 2.4e-6 | 6.6e-6 | 1.7e-3 (bf16 storage floor; bound 5e-3) | 0.85-1.08 |
| conv states zeroed | FAIL | 3.3e-2 | 3.5e-1 | 6.6e-2 | 1.34-1.94 |
| recurrent states zeroed | FAIL | 3.4e-1 | 8.2e-1 | 2.6e-1 | 12.7-16.4 |
| recurrent states x 0.5 | FAIL | 1.3e-1 | 3.0e-1 | 1.4e-1 | 4.8-5.8 |
| last KV position dropped | FAIL | 6.9e-3 | 2.3e-2 | 7.7e-3 | 0.85-1.16 |

The reviewer's further injected bugs (q_end off-by-one, q_pre / option-span shift, later-row LM off-by-one, middle-option swap) all fail v3 on the tiny bf16 model (`tests/test_q9_feature_cache.py::test_selfcheck_v3_catches_the_reviewers_missed_bugs`). The PR #111 reviewer reproduced the v2 bug-injection results on Qwen3.5-2B (`rev111_bugprobe_2b.json`, coordinator scratchpad).

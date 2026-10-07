# q9 feature cache: re-price from job-kxc5e (2026-10-07)

Source: artifact art-daqb2 (sha256 `fab1a940…`, job-kxc5e, q9_feature_cache stock_dev @ bcbb1ea). Only the small rate and progress files were extracted.

| Measurement | Value | File |
|---|---|---|
| Rate-guard stop | "534 units (9.7% of the weight) in 8.0 min of scoring -> 87 min > cap 65.0" | `job-kxc5e_rate_stock.json` |
| Guard clock start (`.total.json` appeared) | 4.75 min into the arm = 07:03:55Z | `job-kxc5e_rate_stock.json` |
| First extracted unit | 07:08:55Z, i.e. **300 s** after the clock start: model load + v3 fp32 self-check | `job-kxc5e_progress_select.jsonl` |
| Extraction rate | 534 units, 191.3 planned s (weights excluding the first unit) in 169.1 s = **0.884× the priced time** (per 100-unit windows: 0.71, 0.70, 0.75, 0.96, 0.89, 1.09) | `job-kxc5e_progress_select.jsonl`, `job-kxc5e_progress_weights.jsonl` |
| True stock_dev projection | 4.75 + 5.0 + 1,989.8 planned s × 0.884 / 60 = **~39 min** vs the 65-min cap; the guard's 87 min extrapolated the one-time 300 s | — |

Conclusion:
- The box is faster than priced. The exit 15 was a clock-placement error (LESSONS L-29). The fix writes the total after the self-check and adds a measured 5.0-min allowance to the preflights.
- The causal_conv1d torch fallback the log warns about does not make the job slower than priced. Adding the CUDA kernel is not needed, and it would need its own parity check; it is not done.
- stock_dev stays one part: its new cap is 70 min against a measured ~39 min.

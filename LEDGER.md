# Spend and experiment ledger

Credits: SF Compute $525. Batch rate is $2.97/H100-h and interactive is $3.96/h; both were verified 2026-10-06. Expiry (read via SF get_billing 2026-10-06): **$500 promo expires 2026-10-27T06:53Z**; $25 top-up never expires. No card on file, auto top-up off (work stops at $0 balance, disks kept). Nodes: none. Planning ceiling: $5,000 (a planning figure, not an approved allowance beyond the credits).

| Date | Track | Experiment / hypothesis | Planned $ | Stop condition | Actual $ | Verdict | Receipt |
|---|---|---|---|---|---|---|---|
| 2026-10-06 (planned, not launched) | D | Rung-2 H100 hardware qualification: numerics, throughput, memory, artifact recovery (jobs/qual/JOB_SPEC.md) | 9.90 (build 20 min + run 180 min at $0.0495/min; validate_job run worst case $8.91; hard cap = workspace cap MTD+$20, required) | exit 1/2/3/4/5/6/7/8/9; max_duration 180 min; queue TTL 24 h | — | — | jobs/qual/ |

## Negative results and false positives

- 2026-10-06, Track L, engineering fold: stop extending the alternate L-only Oracle/Replay API draft. The shared D OracleBackend/ReplayBackend now exercises the actual parent wire through the L bridge, including probability-preserving replay and explicit misses. Maintaining a second incompatible API is superseded by that integration; this is not a model-performance falsification. The local draft remains in `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-backends/language_growth/backends.py`; no source was deleted. Cost $0. Replacement evidence: PR #13's shared replay integration and its versioned receipt.
- 2026-10-06, Track L: the standalone fixture-pack receipt at source `2fe814a` / receipt `8285863` is not reproducible from that source alone: validation used the unmerged 0.1 contract from PR #4. Claude reproduced the content with that dependency but correctly rejected the standalone receipt claim in [PR #5](https://github.com/apumutyala/State-Space-Decision-Model/pull/5). This is a failed reproducibility claim, not a learned-model result. Paid cost: $0. Keep the original evidence; supersede it with a fresh receipt only after the contract is merged, explicitly recording the contract SHA and canonical JSON hash rule. No spend gate may use the old receipt.

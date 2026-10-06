# Spend and experiment ledger

Credits: SF Compute $525. Batch rate is $2.97/H100-h and interactive is $3.96/h; both were verified 2026-10-06. Expiry: **check first**. Planning ceiling: $5,000.

| Date | Track | Experiment / hypothesis | Planned $ | Stop condition | Actual $ | Verdict | Receipt |
|---|---|---|---|---|---|---|---|

## Negative results and false positives

- 2026-10-06, Track L, engineering fold: stop extending the alternate L-only Oracle/Replay API draft. The shared D OracleBackend/ReplayBackend now exercises the actual parent wire through the L bridge, including probability-preserving replay and explicit misses. Maintaining a second incompatible API is superseded by that integration; this is not a model-performance falsification. The local draft remains in `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-backends/language_growth/backends.py`; no source was deleted. Cost $0. Replacement evidence: PR #13's shared replay integration and its versioned receipt.

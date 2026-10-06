# Spend and experiment ledger

Credits: SF Compute $525. Batch rate is $2.97/H100-h and interactive is $3.96/h; both were verified 2026-10-06. Expiry: **check first**. Planning ceiling: $5,000.

| Date | Track | Experiment / hypothesis | Planned $ | Stop condition | Actual $ | Verdict | Receipt |
|---|---|---|---|---|---|---|---|

## Negative results and false positives

- 2026-10-06, Track L: the standalone fixture-pack receipt at source `2fe814a` / receipt `8285863` is not reproducible from that source alone: validation used the unmerged 0.1 contract from PR #4. Claude reproduced the content with that dependency but correctly rejected the standalone receipt claim in [PR #5](https://github.com/apumutyala/State-Space-Decision-Model/pull/5). This is a failed reproducibility claim, not a learned-model result. Paid cost: $0. Keep the original evidence; supersede it with a fresh receipt only after the contract is merged, explicitly recording the contract SHA and canonical JSON hash rule. No spend gate may use the old receipt.

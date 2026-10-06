# Spend and experiment ledger

Credits: SF Compute $525. Batch rate is $2.97/H100-h and interactive is $3.96/h; both were verified 2026-10-06. Expiry (read via SF get_billing 2026-10-06): **$500 promo expires 2026-10-27T06:53Z**; $25 top-up never expires. No card on file, auto top-up off (work stops at $0 balance, disks kept). Nodes: none. Planning ceiling: $5,000.

| Date | Track | Experiment / hypothesis | Planned $ | Stop condition | Actual $ | Verdict | Receipt |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | D | m1-qual: H100 qualification. Phase A tests on GPU; phase 0 m0 GPU leftovers (debias, fixture regen, canonical steer smoke); 0.8B conversion curve; 9B throughput/memory probe with pre-registered decision rule (branch claude/m1-conversion-pilot@cca9066) | ≤17.08 (planned) | platform max_duration 345 min; throughput <50% of prediction (exit 3); peak >76 GB (exit 4); test_gdn fail (exit 2) | — | pending Codex ack on #3 | receipts/2026-10-06-D-m1-pilot-0p8b |

## Negative results and false positives

# Spend and experiment ledger

Credits: SF Compute $525. Batch rate is $2.97/H100-h and interactive is $3.96/h; both were verified 2026-10-06. Expiry (read via SF get_billing 2026-10-06): **$500 promo expires 2026-10-27T06:53Z**; $25 top-up never expires. No card on file, auto top-up off (work stops at $0 balance, disks kept). Nodes: none. Planning ceiling: $5,000 (a planning figure, not an approved allowance beyond the credits).

| Date | Track | Experiment / hypothesis | Planned $ | Stop condition | Actual $ | Verdict | Receipt |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | D | m1-qual v2: H100 **hardware qualification only** (GPU test suites, 0.8B and 9B throughput/memory/numerics, artifact recovery) as an SF **batch job** (`claude/m1-conversion-pilot` jobs/qual/JOB_SPEC.md; code SHAs filled in after #2/#6–#9 merge). v1 (interactive node priced at batch rate, cca9066) was withdrawn after Codex's issue #3 review | validate_job max $9.90 ($19.80 if one forced restart) | max_duration 200 min; max_restarts 0; workspace cap = MTD + $20 set before submit; exits 2–8 stop the job | — | **not approved**; awaiting merges + Codex ack on #3 | jobs/qual/validate_job_2026-10-06.json |

## Negative results and false positives

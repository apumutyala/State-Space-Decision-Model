# B-2: proxy600, a pre-declared Index screen (2026-10-08, agent B, CPU only)

**Takeaway.** `jobs/common/dev_proxy600.json` (644 rows; ids sha `7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6`)
ranks 15 scored systems with Spearman 0.94 (Kendall 0.87) against the 2,008-row subset. On system pairs 1-5 points
apart it gets the sign right 91% of the time, with a mean gap error of 1.1 points (90th percentile 2.5). Its paired
95% CI covers the full-subset gap for all 53 pairs. It is a screen, not a verdict: it resolves 28% of those pairs
(CI excludes 0), against 62% on the full subset.

## Design (fixed before validation; `jobs/common/dev_proxy.py`)
- Strata: the 38 Index benchmarks x 5 length bins. Units: whole source cases (`harness.cases`).
- Quota per benchmark: largest remainder, floored at max(4, n/100) rows and capped at the benchmark's rows. Three
  pre-declared allocations: `rows` (proportional to subset rows), `weight` (proportional to Index weight) and `mix`.
- Within a stratum, cases are taken in sha256(seed:catalog:case) order. The seed is `proxy-v1`; it was never searched.
- Choice: leave-one-system-out. For each held-out system, the design is picked by the mean pair-gap error over the
  other systems' pairs and all 101 seeds. `weight` was chosen 15/15 times at every size.

## Validation (`analysis/proxy_validate.py`, `validate.json`)
- Systems: stock v1 and v2, the 8 R3 readout variants, D1 child, the arith and geo ensembles, Clef 27B and Clef-flash
  (Index 24.9-62.1).
- Exactness: the case-mask Index equals a direct frozen-scorer decompose of the proxy rows, max |err| 0.0. A
  `dev_subset score --subset dev_proxy600.json` re-check on ens_arith matched exactly.

| weight design | rows | Spearman | gap err mean / p90 (pairs 1-5) | sign agree | LOSO held-out sign / gap err |
|---|---|---|---|---|---|
| 400 | ~443 | 0.92 | 1.43 / 3.18 | 0.89 | 0.92 / 1.21 |
| **600 (declared)** | **644** | **0.943 (0.953)** | **1.14 / 2.49** | **0.906** | **0.916 / 1.13** |
| 800 | ~834 | 0.97 | 0.90 / 2.00 | 0.96 | 0.93 / 0.79 |

The 400 and 800 rows are medians over 100 alternative seeds, since no proxy was built at those sizes. The 600 row
is the declared seed, with the seed median in parentheses. The p90 values are pooled over the alternative seeds.

The proxy's own Index has a 95% CI width of 8.2 (5.5 on the full subset). The paired CI width is 6.7 (4.4 on the
full subset). The mean |dIndex| is 1.96 for the declared seed and 1.39 at the seed median. This offset is common to
both arms and cancels in paired screening.

## Cost (measured D1 job-c7pq2 per-row wall times)
The Index stage drops from 71.2 to about 29.6 minutes, which saves 41.6 minutes per cell ($2.06 batch, $2.62 quoted).
The evals phase drops from 178 to about 136 minutes, because G12 (82 minutes) plus fit and card (18 minutes) are
unchanged.

## Reproduce
`python -X utf8 -I <runner adding repo+kit to sys.path> analysis.proxy_validate --systems systems.json --seeds 100 --B 1000 --out validate.json`
`python -X utf8 -m jobs.common.dev_proxy build --n 600 --design weight --seed proxy-v1`

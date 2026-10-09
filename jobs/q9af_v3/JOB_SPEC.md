# q9af_v3: Q9-AF v3, two chained H100 jobs (A: conversion, B: long context + on-policy + the gate)

**Takeaway.** The design and the binding gate are pre-registered in `README.md`. In short: v3 is promoted over v2 iff
the paired proxy600 Index (v3 − v2) under the generic v3w readout has a CI lower bound > 0. This file covers the
phases, caps and money.

| Job | max_duration | Worst case (lint: (60 build + max_duration) × $0.084) | Platform hold (max_duration × $0.063) | Expected (incl. ~15 min build) |
|---|---|---|---|---|
| A (`job_params_A.json`) | 581 min | **$53.84** | $36.60 | 541 min = **$45.44** at $0.084 ($34.08 at $0.063) |
| B (`job_params_B.json`) | 219 min | **$23.44** | $13.80 | 157 min = **$13.19** at $0.084 ($9.89 at $0.063) |
| A + B | 800 min | $77.28 | $50.40 | **$58.63** |

The billing basis is the H100 list price of $5.04/GPU-h, which is $0.084/min (`batch_usd_per_min`). The platform holds
max_duration × $0.063 (`costs.QUOTE_USD_PER_MIN`). Each job makes one attempt (`max_restarts` 0) and is under the
900-min platform cap. A failed attempt salvages its newest all-8 checkpoint as `gdn_salvage.pt` (never a Q9-AF
input).

## Phases (caps in minutes; every cap is checked against its projection by `jobs/q9af_v3/preflight.py` at setup)

| Job | Phase | Cap | Projection (basis) |
|---|---|---|---|
| A | setup / weights | 8 / 10 | 2.4 / 0.3 (job-5cmat; w3c2e measured 3.3 total) |
| A | D | 30 | 15 (v2 clean estimate: 4.9 measured + ~8.3 full-eval-v1 guard) |
| A | G | 85 | 75 capped + 0.7 overshoot + 1.3 × 5.1 uncapped = 82.3 |
| A | P | 425 | 400.45 capped (40 + 360 + one step per stage) + 1.3 × 18.1 uncapped = 423.9 |
| A | X | 8 | 4 (export + copies) |
| B | setup / weights | 8 / 10 | as A (`sf_fetch --stage q9`: weights only) |
| B | F | 15 | 4 (2 × ~2.1 GB streamed + sha256) |
| B | P | 110 | 85.7 capped (45 + 40 + one step each) + 1.3 × 15.2 uncapped = 105.4 |
| B | X | 8 | 3 |
| B | L_stock / L_v3 / L_v2 | 15 each | 7.06 each (v3w proxy600 fork volume 1.92M positions + 1,401 calls at 22,144 tok/s, 0.2406 s/call) |
| B | S | 8 | 1 (measured 5cmat) |
| both | reserve | 15 | |

Floors and peak (exits 3 / 4):
- stage 1 stops below 9,476 tok/s (measured 18,297);
- the A distillation stages stop below 2,600 tok/s (worst measured 5,210);
- the 32k stage stops below 1,500 tok/s (estimate 4,000);
- on-policy stops below 100 scored tok/s (estimate 461);
- every stage stops at a 76 GB peak (measured 39.7).

Data budgets, each checked at 1.15 × the predicted use:

| Budget | Have | Need |
|---|---|---|
| short-region train tokens | 263.2M | 226.7M |
| reserved 32k tokens | 16.8M | 12.4M |
| on-policy prompts | 12,000 | 7,066 |

## Runner and outputs
`jobs/q9af_v3/q9af_v3_job.sh A|B` runs the phases above. Each phase is exported and verified into `MANIFEST.jsonl` and
`phases/*.tgz`. `/output/Q9AF3_DONE` is written only by `jobs/common/finalize.py`, and only with a valid record: job
A's `STAGE_A_DONE`, or job B's PROMOTE / REPORT.

Outputs:
- **job A:** `/output/stageA/` (`gdn_converted.pt`, `export.json`, `lc.u32`, `onpolicy.npz`, `gen_heldout.npz`,
  `heldout_{web,chat,math}.u32`, `mix.json`, `split.json`, `results_A.json`, `gen.json`, `MEMBERS.sha256`) and
  `decision.json`;
- **job B:** `gdn_converted.pt` (+ `.sha256`, `export.json`) and `decision.json`, plus the index arms, compares and
  scores in `phases/S.tgz`.

The CPU dry run (`Q9AF3_DRYRUN=1`, `tests/test_q9af_v3.py`) runs this very runner, with only the fetch, GPU and kit
steps stubbed.

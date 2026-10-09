# q9af_v3: Q9-AF v3 (recall redesign). S (cheap sensitivity) → A (recall-gated conversion) → B (32k + on-policy + gate)

**Takeaway.** The design and every binding gate are pre-registered in `README.md`. In short:
- **S** decides whether any single layer converts with ≤ 0.05 MQAR / NIAH loss (GO iff ≥ 6 of 8).
- **A** stops converting the moment a stage breaches recall or proxy600 bounds.
- **B** promotes v3 over v2 only on a proxy600 CI lower bound > 0 with recall within 0.20 of stock.

This file covers the phases, caps and money.

| Job | max_duration | Worst case (lint: (60 build + max_duration) × $0.084) | Platform hold (max_duration × $0.063) | Expected (incl. ~15 min build) |
|---|---|---|---|---|
| S (`job_params_S.json`) | 203 min | **$22.09** | $12.79 | 176 min = **$14.78** at $0.084 |
| A (`job_params_A.json`) | 863 min | **$77.53** | $54.37 | 775 min = **$65.10** at $0.084 |
| B (`job_params_B.json`) | 219 min | **$23.44** | $13.80 | 157 min = **$13.19** at $0.084 |
| S + A + B | 1,285 min | $123.06 | $80.96 | **$93.07** |

The billing basis is the H100 list price of $5.04/GPU-h, which is $0.084/min (`batch_usd_per_min`). The platform holds
max_duration × $0.063 (`costs.QUOTE_USD_PER_MIN`). Each job makes one attempt (`max_restarts` 0) and is under the
900-min cap.

## Phases (caps in minutes; every cap is checked against its projection by `jobs/q9af_v3/preflight.py` at setup)

| Job | Phase | Cap | Projection (basis) |
|---|---|---|---|
| S | setup / weights / D | 8 / 10 / 15 | 2.4 / 0.3 / 5 (60M-token clean mix) |
| S | P | 160 | 122.1 capped (15 stage 1 + 10 × 10 single-layer KD + 2 × 3 control stage 1) + 1.3 × 23.6 uncapped (11 recall probes à 1.5, KL evals, load) = 152.7 |
| A | setup / weights / D | 8 / 10 / 30 | 2.4 / 0.3 / 15 |
| A | G | 85 | 75 capped + 0.7 + 1.3 × 5.1 = 82.3 |
| A | I_stock | 15 | 7.06 (v3w proxy600 volume: 1.92M positions + 1,401 calls at 22,144 tok/s, 0.2406 s/call) |
| A | P (4 stage invocations, one deadline) | 620 | 560.5 capped (40 + 520 + one step per stage) + 1.3 × 42.7 uncapped (4 loads, 9 recall + 8 KL probe evals, stage evals, final) = 616.0 |
| A | I_stages (4 arms) | 64 | 4 × 7.06, each ≤ 0.95 × 16 / 1.3 |
| A | S / X | 8 / 8 | 2 / 4 |
| B | setup / weights / F / P / X | 8 / 10 / 15 / 110 / 8 | 2.4 / 0.3 / 4 / 105.4 / 3 |
| B | L_stock / L_v3 / L_v2 / S | 15 / 15 / 15 / 8 | 7.06 each / 1 |
| all | reserve | 10 (S), 15 (A, B) | |

Floors and peak (exits 3 / 4):
- stage 1 stops below 9,476 tok/s (measured 18,297);
- KD stops below 2,600 tok/s (worst measured 5,210);
- 32k stops below 1,500 (estimate 4,000);
- on-policy stops below 100 (estimate 461);
- every stage stops at a 76 GB peak (measured 39.7).

Data budgets, each checked at 1.15 × the predicted use:

| Budget | Have | Need |
|---|---|---|
| S train tokens | 60M | 51.6M |
| A short-region tokens | 263.2M | 169.3M |
| B 32k blocks | 16.8M | 12.4M |
| B on-policy prompts | 12,000 | 7,066 |

## Runner and outputs
`jobs/q9af_v3/q9af_v3_job.sh S|A|B` exports and verifies every phase (`MANIFEST.jsonl`, `phases/*.tgz`).
`/output/Q9AF3_DONE` is written only by `jobs/common/finalize.py`, with a valid record:

| Job | Valid records |
|---|---|
| S | GO / PARTIAL / NO_GO |
| A | STAGE_A_DONE or STOPPED (`gates.json` per stage) |
| B | PROMOTE / REPORT |

Outputs:
- **S:** `decision.json` (the per-layer table and verdict), plus `results.json` in `phases/P.tgz`;
- **A:** `/output/stageA/*` (12 members + `MEMBERS.sha256`, the artifact B pins with `jobs.q9af_v3.artifact fill`),
  or `gdn_salvage.pt` when STOPPED;
- **B:** `gdn_converted.pt` + `export.json` (with `gdn_dk`) + `decision.json`.

The CPU dry runs (`Q9AF3_DRYRUN=1`, `tests/test_q9af_v3.py`) run this very runner, with only the fetch, GPU and kit
steps stubbed.

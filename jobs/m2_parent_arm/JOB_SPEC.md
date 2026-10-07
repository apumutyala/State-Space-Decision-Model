# m2_parent_arm: score the frozen parent once per gate (≤ $20). DRAFT, NOT SUBMITTED

**What it does.** It scores the m1 parent on the frozen development cohort (`train.cohort` v2, sha256
`be6efe2b…`) once per gate. Every m2 cell on that parent then reuses the result as its paired comparison arm. No
training happens in this job. The parent is served the way m0/m1 serve it: the stock decision readout
(`decide.backends.hf_readout`) through `decide.Decider`, with the identity calibration.

**Why it is a separate job.** The parent is frozen, so its units are the same for every cell. Without the cache,
every cell would score them again. Each arm costs about 26.8 M tokens, so the cache halves each cell's eval volume.

## Units (identical to the child's: `train.m2_gates.score_stage` scores both arms)

| Stage | Units | Used by |
|---|---|---|
| `g12` | cohort `dev` decision items + `steer_dev` (1,800 requests: clean + 4 D1-A = 9,000 units) | `verdict_g12` at every child checkpoint |
| `g13` | cohort `retention` decision items + `steer_gate` (1,000 requests: clean + 8 held-out D1-B = 9,000 units, `final=True`) + the FULL Index DEV slice minus the rows the parent's decider refuses before any forward, frozen by id hash at the start of this phase before any Index row is scored (`jobs.common.dev_full`, `--index-frozen-out`; pin its sha256 as `index_frozen_sha256.<parent>` in `jobs/m2_pilot/job_params.json`) | `verdict_g13` at gates (capability = full Index DEV **plus** the retention set, issue #36 review adopted 2026-10-07; the 2,008-row subset is a screen only; Index TEST stays sealed) |

Cache layout: `<out>/arm.json` plus `units-g12.json` and `units-g13.json`. Each stage entry records three things:
- **`units_identity`:** cohort sha256, unit spec (K ≤ 16, pre-registered token cap 2,048, template, layout), the
  sha256 of the decision item ids, the sha256 of every variant's `unit_sha`, and the Index DEV run ids.
- **`arm`:** the parent's family, repo@revision, the converted-checkpoint sha256 for Q9-AF, and the served revision id.
- **the file's sha256.**

`train.m2_gates.read_arm_cache` refuses a cache in any of these cases:
- the units differ from the child's;
- the file hash drifted;
- the parent identity differs, for example a Q9-AF cache whose converted sha256 differs from the child's parent.

## Phases (`job_params.json`; H100 batch $0.0495/min)

| Phase | Cap (min) | Content | Exit |
|---|---|---|---|
| setup | 15 | records / DEV rows sha256; cohort rebuild == committed; pinned snapshot (+ `snapshot.sha256` for RWKV7); full DEV rows | 5 / 10 |
| guard | 5 | `jobs.m2_pilot.guard_ready` (content index 7cbe6c89…, refuse otherwise) | 30 / 31 / 32 |
| preflight | 1 | `score_parent estimate --tok-s $TOK_S` from a **measured** prefill rate (rwkv7 stage-A probe; probe9b throughput for Q9-AF); projection over 95% of the 150 scoring minutes → stop before scoring | 15 |
| g12 | 60 | parent arm, stage g12 → cache | 9 / 10 / 7 |
| g13 | 90 | parent arm, stage g13 → cache | 9 / 10 / 7 |
| report | 2 | `validate.valid_cache`; the marker is written only at 0 | 9 / 1 |

Schedule 173 + reserve 7 = 180. **Worst case** (20 build + 2 × 180) × $0.0495 = **$18.81**, under the **$20 cap**.

## Cost estimate (`score_parent.estimate`; volumes measured on the committed cohort, the rate is NOT)

Token volume (RWKV7-G1k tokenizer; every question row of every role record, read once per question row as the
prefix-fork readout does; Index DEV = the FULL DEV slice (issue #36), PROJECTED as the 2,008-row subset's exact
prefix-fork positions (`jobs/common/readout_volume_rwkv7.json`, 6.39 M, which replaced the proxy-tokens × 2 guess of
4.74 M; `jobs/PERF_AUDIT.md`) × the full/subset proxy-token ratio 6.07 = 38.79 M; re-measure on the first run):

| Stage | Tokens |
|---|---|
| g12 | dev 3.89 M + steer 5.38 M = **9.27 M** |
| g13 | retention 5.72 M + steer 5.43 M + full Index DEV 38.79 M (projected) = **49.94 M** |

Measured on cohort v2 (`VOLUME_COHORT_SHA256`; a test fails if the cohort changes without re-measuring).

Projected cost by prefill rate (+ 35 min fixed: build, setup, finalize):

| Prefill tok/s (measured in stage A / probe9b) | g12 min | g13 min | total min | $ |
|---|---|---|---|---|
| 2,500 | 62 | 333 | 430 | 21.27 (over the 150-min scoring cap → exit 15, re-price) |
| 5,000 | 31 | 166 | 232 | 11.50 (over the 150-min scoring cap → exit 15; run g13 as its own job) |
| 10,000 | 15 | 83 | 134 | 6.62 |
| 20,000 | 8 | 42 | 84 | 4.17 |

The volumes are upper bounds: records whose questions are all filtered out by K/T still count.

The child's evals phase in `jobs/m2_pilot` is larger than the arm alone: arm + temperature fit + dev card =
66.6 M tokens (34.2 M before the full-DEV switch). At its 50-minute cap it needs ≥ ~23,400 tok/s (`--mode child`). Below that rate, run G12 inside the cell and G13 as its own gate job (`train.m2_gates --stage g13`, same caching rules),
and price it with the same `estimate`.

## Before submit

- A measured `TOK_S` for the chosen parent.
- For Q9-AF:
  - the full-8 converted checkpoint;
  - its sha256 pinned in `jobs/m2_pilot/m2_pilot_cell_q9af.json` (currently `null`, so the job exits 10);
  - preferably also `base_config_sha256`.
- The usual spend steps: a `spend` issue, a LEDGER row, `validate_job` against $20, and the credit-expiry check.
- Margins: `train.promotion.MARGINS` (Codex #36 contract, comments 6028537932 + 6028610849; status in `MARGINS`).
  This job produces no verdict; it only caches the parent's units.

## Cohort v3 retention: per-arm G13 cost (measured volume × measured rates)

Cohort v3 (`train/cohort_v3.py`, sha256 `0c0309b9…c32f`) expands the G13 `retention` role from 1,670 to 39,672
records (36,005 usable items under RWKV7, 35,992 under Qwen3.5; issue #36 sizing, PR #102 `margin_sim`). This job
still pins cohort v2 (`VOLUME_COHORT_SHA256`); moving it to v3 is a follow-up once PR #102 lands (see the v3 PR).

Volume is exact, not projected (`receipts/2026-10-07-D-cohort-v3/measure.py` → `measure.json`): it is the prefix-fork
plan of the decide HF readout (`jobs.common.readout_cost.row_volumes`, real template + tokenizer) over every retention
record with at least one usable item. `arm_decision_fields` calls `decide` only for those records.

| Backbone (rates) | Positions | Calls | Minutes per arm | $ per arm | Conservative (pack call per call) |
|---|---|---|---|---|---|
| RWKV7-G1k-13.3B (job-tkfaj: 3,757 tok/s, 0.117 s/call) | 21.18 M | 64,054 | **218.9** | **$10.83** | 734.5 min / $36.36 (0.60 s/call) |
| Qwen3.5-9B / Q9-AF (job-5cmat: 22,144 tok/s, 0.0595 s/call) | 20.51 M | 63,930 | **78.8** | **$3.90** | 267.9 min / $13.26 (0.2369 s/call) |

- Time is `positions / tok_s + calls × call_s` (the `jobs.common.readout_cost.project` model). Dollars use H100 batch at $0.0495/min.
- **Per gate:** the parent arm is paid once per parent (cached); each candidate child arm pays the same amount again.
- **v2 share:** v2's 1,670 records are 8.6 min / $0.43 (RWKV7) and 3.1 min / $0.15 (Q9) of the totals above.
- **Upper bound (the `VOLUME_TOKENS` convention):** prompt tokens of every question row, with no K/T filter. This reproduces v2's 5,724,231 exactly. For v3 it is 129.6 M (RWKV7: 575 min, $28.46) and 132.2 M (Q9: 99.5 min, $4.93). Records whose rows are all filtered out are never forwarded, so this bound over-counts.
- **On RWKV7, calls dominate:** 64k × 0.117 s ≈ 125 min, against ≈ 94 min of prefill. Packing tails across records would cut this. That is not done here.
- **Caps:** the 90-minute g13 phase above cannot hold RWKV7's 219 minutes. G13 with v3 needs its own job or phase, with caps re-sized from these numbers; Q9's 79 minutes fits only without the steer and Index terms.
- **Cost authority:** cost is pre-authorised (coordinator, cohort v3 task). The $20 hard cap of this draft still applies until its params are re-priced.

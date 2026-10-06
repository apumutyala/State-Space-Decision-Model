# m2_parent_arm: score the frozen parent once per gate (≤ $20). DRAFT, NOT SUBMITTED

**What it does.** It scores the m1 parent on the frozen development cohort (`train.cohort` v2, sha256
`be6efe2b…`) once per gate. Every m2 cell on that parent then reuses the result as its paired comparison arm. No
training happens in this job. The parent is served the way m0/m1 serve it: the stock decision readout
(`decide.backends.hf_readout`) through `decide.Decider`, with the identity calibration.

**Why it is a separate job.** The parent is frozen, so its units are the same for every cell. Without the cache,
every cell would score them again. Each arm costs about 25.7 M tokens, so the cache halves each cell's eval volume.

## Units (identical to the child's: `train.m2_gates.score_stage` scores both arms)

| Stage | Units | Used by |
|---|---|---|
| `g12` | cohort `dev` decision items + `steer_dev` (1,800 requests: clean + 4 D1-A = 9,000 units) | `verdict_g12` at every child checkpoint |
| `g13` | cohort `retention` decision items + `steer_gate` (1,000 requests: clean + 8 held-out D1-B = 9,000 units, `final=True`) + the 2,008-row Index DEV subset (`jobs/common/dev_subset_m1.json`) | `verdict_g13` at gates (capability = Index DEV subset **plus** the retention set, coordinator decision 2026-10-06; Index TEST stays sealed) |

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
| setup | 15 | records / DEV rows sha256; cohort rebuild == committed; pinned snapshot (+ `snapshot.sha256` for RWKV7); DEV subset rows | 5 / 10 |
| guard | 5 | `jobs.m2_pilot.guard_ready` (content index 7cbe6c89…, refuse otherwise) | 30 / 31 / 32 |
| preflight | 1 | `score_parent estimate --tok-s $TOK_S` from a **measured** prefill rate (rwkv7 stage-A probe; probe9b throughput for Q9-AF); projection over 95% of the 150 scoring minutes → stop before scoring | 15 |
| g12 | 60 | parent arm, stage g12 → cache | 9 / 10 / 7 |
| g13 | 90 | parent arm, stage g13 → cache | 9 / 10 / 7 |
| report | 2 | `validate.valid_cache`; the marker is written only at 0 | 9 / 1 |

Schedule 173 + reserve 7 = 180. **Worst case** (20 build + 2 × 180) × $0.0495 = **$18.81**, under the **$20 cap**.

## Cost estimate (`score_parent.estimate`; volumes measured on the committed cohort, the rate is NOT)

Token volume (RWKV7-G1k tokenizer; every question row of every role record; Index DEV = kit proxy tokens × 2):

| Stage | Tokens |
|---|---|
| g12 | dev 3.49 M + steer 5.38 M = **8.87 M** |
| g13 | retention 6.69 M + steer 5.43 M + Index DEV 4.74 M = **16.86 M** |

Projected cost by prefill rate (+ 35 min fixed: build, setup, finalize):

| Prefill tok/s (measured in stage A / probe9b) | g12 min | g13 min | total min | $ |
|---|---|---|---|---|
| 2,500 | 59 | 112 | 206 | 10.22 (over the 150-min scoring cap → exit 15, re-price) |
| 5,000 | 30 | 56 | 121 | 5.98 |
| 10,000 | 15 | 28 | 78 | 3.85 |
| 20,000 | 7 | 14 | 56 | 2.79 |

The volumes are upper bounds: records whose questions are all filtered out by K/T still count.

The child arm in `jobs/m2_pilot` has the same volume. At its 50-minute evals cap it needs ≥ ~9,000 tok/s. Below
that rate, run G12 inside the cell and G13 as its own gate job (`train.m2_gates --stage g13`, same caching rules),
and price it with the same `estimate`.

## Before submit

- A measured `TOK_S` for the chosen parent.
- For Q9-AF:
  - the full-8 converted checkpoint;
  - its sha256 pinned in `jobs/m2_pilot/m2_pilot_cell_q9af.json` (currently `null`, so the job exits 10);
  - preferably also `base_config_sha256`.
- The usual spend steps: a `spend` issue, a LEDGER row, `validate_job` against $20, and the credit-expiry check.
- Margins stay **PENDING USER APPROVAL**. This job produces no verdict; it only caches the parent's units.

# CHAMPION — the current best candidate on `agent-main`

`agent-main` holds only the clean best candidate plus shared infra. Every hypothesis variant lives on its own branch
`stack/<family>/<variant>` (e.g. `stack/q9af/full8`, `stack/q9af/head-r256-ce-brier`, `stack/rwkv7/cell-ckpt-rows-lean`);
infra/tooling stays on `claude/*` / `codex/*`. Every measured variant gets a row in `CANDIDATES.md` (negative results
included). Enforced by `tests/test_jobs_lint.py::test_champion_and_candidates_ledgers`.

## Current champion (2026-10-06)

**None trained yet.** No decision-model checkpoint has been trained, so there is no promoted champion and no
`champion/*` tag.

Best measured attention-relevant candidate (a pipeline/backbone measurement, **not** a champion and **not
attention-free**: 7 of 8 attention layers are still softmax attention):

| Field | Value |
|---|---|
| Family | Q9-AF (Qwen3.5-9B with attention layers converted to GDN) |
| Variant | one layer converted (layer 23 of the 8 attention layers), probe9b |
| Branch | `claude/q9-one-layer-probe` (pre-policy); next work on `stack/q9af/full8` |
| Merged sha | launched from `agent-main` after PR #73 (`24f164e`; gate rule pinned to `511e8dd`). Confirm against the CONTEXT_MANIFEST in art-2bybh |
| Config | `configs/probe9b_h100.json` (full), `jobs/probe9b/job_params.json` |
| Checkpoint artifact | art-2bybh, sha256 `dd491ade…` (full hash in issue #68) |
| DEV subset (2,008 rows) | 39.9 [36.5, 42.9] vs stock Qwen3.5-9B 38.3 [35.0, 41.4]; paired diff **+1.66 [+0.19, +3.18]** |
| vs Clef-flash | Clef-flash 55.8 [53.0, 58.5] on the same subset (job-sgihj): about −16 pts; paired CI not computed |
| vs previous champion | n/a (none) |
| Full DEV | not measured |
| ECE (board protocol) | 0.112 (stock 0.148) |
| Steer / hijack | not measured on this candidate |
| Latency | not measured on this candidate |
| Cost | job-5cmat $5.01 (101 min, exit 0) |
| Receipts | job-5cmat, issue #68, LEDGER.md row 2026-10-06 |
| Verdict | `proceed-all-8` under the pre-registered rule (511e8dd) |

## Promotion rule

A variant replaces the champion on `agent-main` only if **all** of these hold:

1. It beats the champion with a **paired CI above zero** on the DEV subset (same 2,008 rows, `jobs/common/dev_subset_m1.json`).
2. It passes the safety promotion gates **G12/G13** (`train.promotion`; margins per issue #36, provisional until Codex approves).
3. It is measured, receipted (`receipts/` + job id + artifact sha256) and **independently reviewed** (COORDINATION.md "Review sufficiency").

On promotion, tag the merge `champion/<YYYYMMDD>-<name>` and update this file. Tag every measured candidate
`cand/<YYYYMMDD>-<name>`. Until a champion exists, the first trained candidate that passes 2–3 and beats stock
Qwen3.5-9B on the paired DEV-subset CI becomes the champion.

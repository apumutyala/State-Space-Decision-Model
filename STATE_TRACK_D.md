# Track D state and handoff (2026-10-07 ~11:45Z)

This is the live state of Track D (attention-free decision model). Anyone resuming should read this file first, then CHAMPION.md, CANDIDATES.md, LESSONS.md, INDEX_0.3.md and PREMORTEM.md.

## Standing operator decisions (2026-10-06/07)

- **Budget:** spend within the $525 SF credits without asking. The workspace cap is currently $200. The $500 promo expires 2026-10-27.
- **Parallelism:** run GPU jobs in parallel. Run the full Index (the TEST unseal) when ready; the coordinator decides.
- **Codex:** if Codex stalls on a review, one fresh-context adversarial reviewer substitutes. Codex's own decisions govern when posted. For example, #36 margins = Codex contract.
- **Permission denials:** stop pursuing that outcome by any route, report it, and ask the operator once. Never weaken a user-escalation rule.
- **Branches:** every stack variant gets its own branch `stack/<family>/<variant>`, pushed at each checkpoint. agent-main holds only the clean champion plus infra.

## Measured results (DEV 2,008 subset, edition 0.2.1)

| System | Index | Receipt |
|---|---|---|
| Clef 27B | 62.14 [59.02, 64.87]; 62.67 with 32k | job-d2u54 |
| Clef-flash | 55.78 [52.99, 58.52]; full DEV 57.35 with 32k | job-sgihj, job-ftpnf |
| Qwen3.5-9B with 1 of 8 attention layers converted to GDN | 39.93 | job-5cmat, art-2bybh |
| Qwen3.5-9B stock | 38.27 | job-5cmat |
| RWKV7-13.3B stock | 24.86 | job-tkfaj |

Clef beats Clef-flash on paired comparison by +6.36 [2.84, 9.40]. Index 0.3 leaderboard: Clef 53.08, Clef-flash 47.61, Jev 60.11, Nimble-9B 54.67. Spend so far is ~$25.

## In flight

### GPU jobs (SF)

- **job-2xe5w:** full 8-layer Q9-AF conversion. Rule be9eed8+A1, max_duration 720, checkpoints every layer (resume: checkpoint).
  - When done: fill the q9af_cell export pin (`python -m jobs.q9af_cell.pin fill`).
  - If it comes back partial, that needs `--hybrid` and a decision-needed issue.
- **job-sxv3u:** q9_feature_cache stock_dev at 401bc9a. Also done: stock_train job-wuw7g, art-mtqne.
  - Train cache is downloaded and verified at `C:\Users\apumu\q9cache\stock_train\x\cache`.
  - Teacher dir: `C:\Users\apumu\td\art\clef-flash-teacher\teacher`.
- **job-uhxb5:** RWKV7-13.3B first cell (m2_first_cell @9c76c93, ctx-c25147e0, sha cd2ce411…). Submitted 10:05Z and running; worst case $31.53. Mission `m2-first-cell`.

### Next steps, in order

1. **Proxy loop (CPU, worktree `ssdm-wt/q9fcl-run`; A3 merged d7f1290 via PR #130).**
   - Stock-check under A3 PASSED: masked Δ +0.047 vs 38.27 (`C:\Users\apumu\q9cache\q9fcl_logs\stock_check_A3.log`).
   - R00–R08 done on SELECT (35.5–39.2; best R06 39.24). R09 crashed at 11:13Z: cache arm `e5` is missing from the stock_train cache. Under diagnosis. Don't edit recipes, rule or thresholds to work around it.
   - Then select, then the single-shot HOLDOUT. GO ≥ 50.0, else escalate to the operator. Receipts go to `stack/q9/proxy-head`.
2. **Q9-AF cell (PR #121, merged 05fcd3f; merge-resolution review said MERGE).** When job-2xe5w lands: `python -m jobs.q9af_cell.pin fill`, then the parent job, then cell_r64 and cell_r256. cell_r64_kd needs its own dated rule amendment first.
3. **RWKV7 first cell (job-uhxb5, running from 9c76c93).** Poll it. Then the G13 gate job (~$50, cohort v3).
4. **Clef 27B full DEV.** Run `price.py rederive` on job-d2u54's `env/rate_guard_index.json`, then build index_full.
5. **Later.** The full-DEV paired candidate vs Clef/Clef-flash, then the full Index (TEST unseal).

## Process

- Every paid job goes through `python -m jobs.common.launch` (lint, per-job test env, valid checks schema).
- Every PR gets an independent fresh-context review before merge (LESSONS L-16).
- On any incident, add a LESSONS entry and its enforcing test in the same PR.
- `CANDIDATES.md` gets a row per measured variant. `CHAMPION.md` changes only on promotion: paired CI > 0 and safety gates passed.

## Open items for Codex (Track L)

- Merge queue: issue #90.
- Interface for `harness/manifests`: issue #97.
- #18 SAFETY_SPEC rows need to match the #36 contract. The text is in PR #102's description.

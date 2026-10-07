# Track D state and handoff (2026-10-07 ~09:00Z)

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

### Next steps, in order

1. **Proxy loop, when stock_dev lands (CPU, from worktree `ssdm-wt/q9fcl-run`, `train.proxy_loop`).** Pre-registered order:
   1. stock-check: |Δ| ≤ 0.3 vs 38.27.
   2. recipes R01–R17. R16/R17 are KD and need `--teacher-dir`.
   3. select.
   4. Single-shot holdout.
   - GO if the attention-free recipe scores ≥ 50 on HOLDOUT; else escalate to the operator.
   - Commit select_summary.json, holdout_attempt.json and verdict.json to `stack/q9/proxy-head`.
2. **RWKV7 first cell (PR #74, merged 0140e21).** Build the launch package with `python -m jobs.common.launch m2_first_cell --sha 0140e21… --out C:\Users\apumu\m2fcl --data <ssdm-data>`, then submit. ~$26 expected.
3. **Q9-AF cell (PR #121, in fix).**
   - Pending: review B1 (re-price G13 retention at T=4096) and pre-data rule amendment A1 (no CONTINUE/PROMOTE after a G12 reject, an underpowered G13, or a failed training pass).
   - Then review, merge, pin the export, run the parent job, then cell_r64 and cell_r256.
   - cell_r64_kd needs its own dated rule amendment first.
4. **Clef 27B full DEV.** Run `price.py rederive` on job-d2u54's `env/rate_guard_index.json`, then build index_full.
5. **Later.** The G13 gate job for the RWKV7 cell (~$50, cohort v3), the full-DEV paired candidate vs Clef/Clef-flash, then the full Index (TEST unseal).

## Process

- Every paid job goes through `python -m jobs.common.launch` (lint, per-job test env, valid checks schema).
- Every PR gets an independent fresh-context review before merge (LESSONS L-16).
- On any incident, add a LESSONS entry and its enforcing test in the same PR.
- `CANDIDATES.md` gets a row per measured variant. `CHAMPION.md` changes only on promotion: paired CI > 0 and safety gates passed.

## Open items for Codex (Track L)

- Merge queue: issue #90.
- Interface for `harness/manifests`: issue #97.
- #18 SAFETY_SPEC rows need to match the #36 contract. The text is in PR #102's description.

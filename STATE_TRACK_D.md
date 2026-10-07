# Track D state and handoff (2026-10-07 ~12:40Z)

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

Clef beats Clef-flash on paired comparison by +6.36 [2.84, 9.40]. Index 0.3 leaderboard: Clef 53.08, Clef-flash 47.61, Jev 60.11, Nimble-9B 54.67. Spend so far is ~$57 (incl. full8 $26.34, RWKV7 cell $3.69).

## In flight

### GPU jobs (SF)

Nothing is in flight. Finished today:

- **job-2xe5w (full8, rule be9eed8+A1, code 53b4cf7): SUCCEEDED, verdict `partial-conversion`.**
  - Layers 3 and 27 were rejected (hidden_cos 0.949 and 0.926, against a 0.95 floor). The result is the hybrid Q9-H2: 6 layers converted, NOT attention-free.
  - Non-binding numbers: Index −26.0 [−28.9, −22.2] vs stock; MMLU −12.2 pts. $26.34. Artifact art-ipnxz.
  - Receipt: PR #139. Decision: issue #140.
  - `pin fill` without `--hybrid` refuses this export, so the q9af_cell cannot run on it.
- **job-uhxb5 (RWKV7 first cell, m2_first_cell @9c76c93): FAILED, exit 20.** The `cell` gate returned rc 1 about 3 minutes into the cell phase.
  - The parent_g12 and parent_index phases passed and exported.
  - $3.69. Artifact art-sq6mp.
  - Root cause: CUDA OOM in the pre-training smoke (recompute fork without act_ckpt). Fix + parent-arm seed reuse in PR #147 (review FIX-THEN-MERGE; fixes in progress).
- Earlier: q9_feature_cache stock_dev job-sxv3u and stock_train job-wuw7g (art-mtqne).
  - The train cache is at `C:\Users\apumu\q9cache\stock_train\x\cache`.
  - Teacher dir: `C:\Users\apumu\td\art\clef-flash-teacher\teacher`.

### Next steps, in order

1. **RWKV7 first cell: fix, review, resubmit.** Reuse the parent exports if the job supports it. Then run the G13 gate job (~$50, cohort v3). This is now the only attention-free line unless #140 decides otherwise.
2. **Decision #140** (operator): (a) close Q9-AF and keep RWKV7 as the sole line [recommended]; (b) pre-register a v2 conversion of L3/L27; (c) ship the Q9-H2 hybrid, which cannot satisfy GOAL. The Q9-AF cell (#121, merged 05fcd3f) is blocked on it.
3. **Proxy loop (worktree `ssdm-wt/q9fcl-run`, A3 = d7f1290): PAUSED before select.** Receipts are on `stack/q9/proxy-head` under `receipts/2026-10-07-D-q9-proxy-loop/` (stock_check.json, select_table.md, DEVIATIONS.md, STATUS.md).
   - Stock-check PASSED: masked Δ +0.047.
   - All 15 stock-arm recipes are done; the best SELECT is R06 at 39.24, about 11 below GO 50.
   - R09, R10 and R13 need the e5/conv1 cache arms, which were never built: four jobs, ~$13.5. Their packages are built but held as not actionable, since any HOLDOUT outcome means escalating.
   - To resume: submit the four cache jobs, then `--only R09,R10,R13`, then select, then the single HOLDOUT.
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

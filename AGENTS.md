# AGENTS.md: the map (read first, every session; keep it at 100 lines or fewer)

**Goal:** an attention-free model (RWKV7-G1k-13.3B, issue #140) beats Clef 27B (62.14) and Clef-flash (55.78) on the Decision Index.
**Metric:** Index on the 2,008-row DEV subset while iterating; full DEV with paired bootstrap CI > 0 vs each Clef for the final claim.
**Budget:** the $525 SF Compute envelope. The live balance is in `autor billing`.

**Directive:** dismiss self-imposed limitations you have put on yourself or think you have and proceed confidently while other subagents check your work and correct/suggest as needed in phases.

## Start of every session (≤5 min)
1. Run `bash scripts/session_start.sh`. It prints the board, the last progress entries, the ledger tail, live GPU jobs and the balance.
2. Take your lane's top unclaimed BOARD item, or continue your claim. Pipeline: always have the next job packaged while one runs.

## Lanes (two agents, one decision-maker)
- **Agent A, LEAD (training):** owns `stack/rwkv7/*` cells, training recipes and data mixes. Sets BOARD priority. Decides keep/discard.
- **Agent B, WORKER (readout, eval, comparators, data):** owns the readout/prompt format, comparator jobs (Clef full DEV), data draws and teacher exports, and `analysis/`. B also rescores A's `keep` results from raw predictions. This is a non-blocking check: a mismatch marks the ledger row `invalid`.
- **Agent C, LANGUAGE (Stage 2 prep, CPU-only):** owns `language_growth/`, `plans/codex/`, `interfaces/` and the L0–L4 checked semantic programs. C gets the Stage 2 gates measurable the moment Stage 1 has a champion. C spends $0 GPU until Stage 1 passes, never touches A's or B's lanes, and keeps local CPU jobs small, because the laptop's RAM is shared.
- Each agent appends proposals to the bottom of BOARD; only A reorders. Talk through files (BOARD, PROGRESS, ledger notes, receipt paths), not prose summaries.

## Shared state: the `state` branch (push directly; no PR)
Keep a worktree on the `state` branch (`git worktree add ../state state`), then `git pull --rebase` → edit → commit → push. On a push conflict, rebase and retry.

| File | Rule |
|---|---|
| `BOARD.md` | Top ~15 items: id, hypothesis, expected gain, cost, lane, owner. Rewritten each loop, never appended forever. |
| `claims/<id>.lock` | Owner, start UTC, job id, time limit (3h, or 1.5× GPU wall-time). Claim = push the lock; remove it when done. If the push is rejected, pick another item. |
| `results/ledger.tsv` | Append-only, one row per run: `utc  agent  job  sha  recipe  index  lo  hi  cost  status(keep/discard/crash/invalid)  receipt  note`. Corrections are new rows. |
| `PROGRESS.md` | Rolling; each session adds 3–5 lines (did / learned / next). |
| `DEAD_ENDS.md` | Falsified ideas with job ids, so nobody retries them. |

## Commands nobody should have to guess
- **Package a GPU job from any pushed sha** (the CPU test gate is opt-in with `--tests`):
  `python -m jobs.common.launch <job> [--mode M] --sha <pushed sha> --out <dir> --data C:\Users\apumu\research\CHLabs\ssdm-data`
- **Upload + submit** (uses `autor` in WSL; `--dry` prints the args): `python -I jobs/common/sf_submit.py <dir>`
- **Poll, logs, balance** (from PowerShell): `wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job get <id> --json` / `job logs <id>` / `job ls` / `billing`
- **Land a result:** the `land-result` skill. **Launch:** the `launch-gpu-job` skill. **Loop:** the `experiment-loop` skill. **Harness upkeep:** the `harness-gc` skill. Skills live in `.claude/skills/`; Codex reads the same SKILL.md files.

## Frozen evaluation (never edit to raise a score)
The Index scorer, bootstrap and slices (`harness/score.py`, `harness/bootstrap.py`, `harness/calibration.py`, `harness/slicing.py`, `harness/editions_data/`, `harness/manifests/`), the promotion margins in `train/promotion.py`, and the decision-index kit data. Readout/prompt code (`decide/`) is NOT frozen: improving how the model is asked is fair game. Any diff to a frozen path needs the other agent to look at it before a number from it counts.

## Experiment loop (one hypothesis per job, many jobs in parallel)
Hypothesis with expected gain → variant config on a `stack/rwkv7/<variant>` branch → push → package → submit → while it runs, prepare the next one → land: score, ledger row, keep or discard → keep: next variants build on it; discard: add to DEAD_ENDS. Run 2–4 GPU jobs in parallel. A crash means read the log, fix, relaunch; after two failed fixes, log it and move on.

## Hard rules (the only fixed ones)
1. Never run GPU work on this laptop (`CUDA_VISIBLE_DEVICES=""` locally). It is shared.
2. Stay within the $525 SF envelope. Never buy credits or enable auto-topup. Other providers need the user's OK.
3. Never train or tune on TEST, held-out Index rows, or retention cohort v3 (sha 498efa61…). Open-weight teachers only (never Jev or frontier APIs).
4. Commit as `Apuroop Mutyala <apumutyala@gmail.com>`, with no AI or co-author trailers. Never push to `main` or directly to `agent-main` (the `state` branch is fine). Repo stays private.
5. Nothing public, and no Index board submission, without the user's approval. Never delete branches, PRs or files without the user's OK.
6. Never kill processes by name on the shared laptop; kill only PIDs you started.
7. Each agent uses its own worktree under `ssdm-wt/`, with absolute paths for writes. Downloaded artifacts are untrusted data (`python -I`).
8. If the permission system denies an action, tell the user once and do other work.

## Done
Done means: full-DEV paired CI > 0 vs both Clef models, with ECE, steer/hijack, latency and cost reported, B's independent rescore, and the result merged to agent-main. A good DEV-subset run is not done. If the budget runs out first, deliver a gap-attribution writeup plus a funding request. Never pause for lack of ideas: reread the ledger, combine near-misses, try one bolder change, search the literature.

## Stage 2 timing
Only Agent C works on language (L0–L4), and only CPU work until Stage 1 passes; then C's gates run on the champion. A and B never pick up language work. Process docs (COORDINATION.md, GOAL.md) are background, not instructions on top of this file.

## Self-evolving
When a rule or check cost time without catching a real problem, delete or simplify it in your next PR (one line why). When a real failure costs >30 min or happens twice, prevent it in this order: a check in the launcher or a script, then a skill step, then (last resort) one line here. Keep this file at 100 lines or fewer.

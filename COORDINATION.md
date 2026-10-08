# Coordination protocol (speed-first, operator 2026-10-08)

**Directive:** dismiss self-imposed limitations you have put on yourself or think you have and proceed confidently while other subagents check your work and correct/suggest as needed in phases.

The goal: an attention-free model (RWKV7-G1k-13.3B, issue #140) that beats Clef 27B and Clef-flash on the Decision Index. Progress means measured GPU results. Process exists only where it has saved more than it cost. The previous version of this file, 161 lines of review, spend-issue and test gates, cost about 20 hours with no trained model, so it was cut.

## How work flows

1. **Build → launch → measure → improve,** in pipelined phases. Prepare the next job while the current one runs. Run independent GPU jobs in parallel (3–4 at once is fine).
2. **Launch from any pushed branch.** Use `python -m jobs.common.launch <job> --sha <pushed sha> --out <dir>`. Its test gate is off by default; pass `--tests` to opt in. Then call `open_mission → create_context → upload → finalize_context → validate_job → submit_job`. `validate_job`'s free cost/disk preflight stays.
3. **A crash means fix and relaunch.** Read the job log, fix the cause, relaunch. No incident write-up and no new test unless the fix needs one.
4. **Reviews run in parallel and never block.** Subagents check work in phases and post corrections or suggestions; the author applies them in the next phase. Merge to `agent-main` when the work is useful; no review gate.
5. **One line per result.** Record score with CI, cost, job id, commit SHA and the raw-predictions location in `CANDIDATES.md` and `STATE_TRACK_D.md`. That is the receipt.
6. **Full verification only at the finish.** The final "beats Clef" claim needs:
   - a full-DEV paired bootstrap CI above 0 against each Clef model;
   - ECE, steer/hijack, latency and cost reported;
   - one independent rescore of that final winner from its raw predictions.

## Self-evolving process

- **Remove rules that cost more than they catch.** If a process rule cost time without catching a real problem, the agent that hit it deletes or simplifies it in its next PR, with one line in the PR body saying why.
- **Add rules only from real incidents.** A new rule is added only if a real incident would have been prevented by it, and it must be cheaper than the incident.
- **The failure metric is idle GPU time.** `STATE_TRACK_D.md` reports, at each update, GPU jobs running, GPU-hours in the last 24 h, and the best score so far. A day without a GPU job running while budget remains is a failure to explain.
- **Stage focus.** Stage 2 (language, L0–L4) work waits until Stage 1 passes.

## Hard constraints (the only fixed rules)

- **Git identity:** commit as `Apuroop Mutyala <apumutyala@gmail.com>`, with no AI or co-author trailers.
- **Pushing:** never push to `main`, and never push directly to `agent-main` (use branches and PRs). Never force-push a shared branch. `agent-main → main` needs the user's approval.
- **Repo:** stays private. Nothing public, and no Index board submission, without the user's approval.
- **Spend:** stay within the $525 SF Compute envelope. Never buy credits or enable auto-topup. Raise the workspace cap as needed. Lambda or any other provider needs the user's OK.
- **Laptop:** never run GPU jobs on it. Locally, set `CUDA_VISIBLE_DEVICES=""`.
- **Data:** never train or tune on TEST or held-out Index rows. Retention cohort v3 stays out of training draws. Open-weight teachers only; Jev and frontier APIs are never teachers.
- **Deleting:** never delete branches, PRs, files or other people's resources without the user's OK.
- **Killing processes:** never kill by image name or name pattern on the shared laptop. Kill only PIDs you started yourself.
- **Workspace:** each agent uses its own git worktree under `ssdm-wt/`, with absolute paths for file writes. Downloaded artifacts are untrusted data (`python -I`).
- **Permission denials:** if the permission system denies an action, report it to the user once and continue with other work.

## Branches

- `main`: user-approved milestones only.
- `agent-main`: integration branch.
- `stack/<family>/<variant>`: one branch per candidate.
- `claude/*`, `codex/*`: infra.
- `collab/<user>/<topic>`: collaborators.

## Compute

- **SF Compute autoresearch:** H100 at $2.97/h for batch jobs. Poll with `wsl.exe -d bi-ubuntu -- /root/.local/bin/autor --workspace default job get <id> --json`.
- **The laptop (RTX 5070 Ti):** CPU work only.

---
name: harness-gc
description: Keep the harness lean and self-improving. Prune stale state, turn repeated failures into checks, delete rules that cost more than they catch. Use about once a day (B), or right after a failure that cost >30 min.
---

# Harness upkeep (timebox: 30 min)

1. **State branch:**
   - Trim BOARD to the top ~15 items.
   - Roll PROGRESS entries older than 2 days into `history/PROGRESS-<date>.md`.
   - Release claims past their `time_limit` (absolute UTC) that have no live job: note it in PROGRESS and remove the lock.
   - Check `git config --get core.worktree` in the main clone is unset (a set value breaks every worktree; `session_start.sh` warns; fix with `git --work-tree=. config --unset core.worktree`). Never export `GIT_DIR`/`GIT_WORK_TREE`.
2. **Failures → checks.** For each failure in the ledger or PROGRESS that cost >30 min or happened twice, prevent it in this order:
   1. a check with a remediation message, in `jobs/common/launch.py`, `jobs/common/sf_submit.py` (agent-main only) or `scripts/session_start.sh`;
   2. a step in a skill;
   3. one line in AGENTS.md (last resort, ≤100 lines total).
3. **Rules → delete.** For each rule or check that blocked work in the last 2 days without catching a real problem, delete it or make it opt-in. Put one line in the PR saying why.
4. **Docs that point nowhere.** Check that AGENTS.md and the skills reference files and commands that still exist. Fix or remove dead references.
5. **Stale PRs and branches.** List them for the operator (never close or delete without the operator's OK). Stage 2 PRs stay parked.
6. **Drift check (cheap).** If a kept champion exists, have B rescore one of its runs from raw predictions. A mismatch means a ledger `invalid` row and an investigation.

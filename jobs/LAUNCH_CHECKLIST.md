# Launch checklist (every paid job; both agents)

Each line is enforced by code. If a line has no check, it is not on this list. Failure history: `LESSONS.md`.

## Before the PR

Work on the right branch:
- Hypothesis variants go on `stack/<family>/<variant>`, pushed at every checkpoint.
- Infra goes on `claude/*` / `codex/*`.
- `agent-main` holds only the champion (`CHAMPION.md`) plus shared infra.
- Every measured variant gets a `CANDIDATES.md` row.

Each item below is enforced by `tests/test_jobs_lint.py`, which runs every check in `jobs/common/lint.py` over every `jobs/<job>/job_params*.json`:

| # | Rule | Check | Lesson |
|---|---|---|---|
| 1 | `build_timeout_minutes >= 60` | `build_timeout` | L-09 |
| 2 | phase caps + reserve **==** `max_duration_minutes` (no unaccounted minutes) | `schedule` | L-05 |
| 3 | every cap is measured, not guessed: `expected_minutes` from a receipt, cap >= 1.3–1.5x it (per-job test) | job's own test | L-04, L-05 |
| 4 | `costs.worst_case_usd` (build + 2 attempts) printed as `$X.XX` in `JOB_SPEC.md`, <= `hard_cap_usd` | `worst_case_stated` | L-08 |
| 5 | registered in `jobs/common/costs.py` `PARAMS` **and** `jobs/DECISION_TREE.md` | `registered` | L-08 |
| 6 | a test **executes the runner `.sh` in a subprocess** (dry-run mode on CPU); Python-level tests do not count | `shell_test` | L-10 |
| 7 | no literal backslash-n continuation in any `jobs/**/*.sh` (regex `[\\]n[ \t]+--`, asserted to fire) | `no_backslash_n` | L-10 |
| 8 | verdict jobs: pre-registered rule module (`decision_rule` in job_params), literal constants pinned by a test, committed **before** the run | `decision_rule` | L-07 |
| 9 | the runner calls a measured-rate preflight / `jobs.common.rate_guard` before its long phase | `rate_guard` | L-01, L-08 |
| 10 | `.gitattributes` pins `*.sh *.py *.json *.sha256 Dockerfile` to LF; `git -c core.autocrlf=true archive HEAD` has no CR | `test_gitattributes_pin_lf`, `test_git_archive_head_has_no_cr_even_with_autocrlf` | L-14 |
| 11 | no control characters in `jobs/ tests/ scripts/` sources (heredoc escape collapse) | `test_no_control_characters_in_sources` | L-11 |
| 12 | gate reference is run through the production path on CPU first; it must pass its own gate | job's own test | L-06 |
| 13 | memory: a test or receipt shows peak at the real shape and micro-batch (estimates have been off by 2x) | job's own test | L-02, L-12 |

**Waivers.** Waivers live in `jobs/common/lint.py::WAIVERS`.
- Each waiver is dated, gives a reason, and covers legacy params files only.
- A stale waiver fails the test, so the list only shrinks.
- New jobs cannot be waived.

## Launch (coordinator, after merge and an independent review)

```
python -m jobs.common.launch <job> [--mode M] --sha <merged sha> --out <dir> [--data $SSDM_DATA] [--allow-waived]
```

The command does the following. Each step refuses on failure.
1. Fetches origin and checks that the sha is an ancestor of `origin/agent-main`.
2. Runs `git -c core.autocrlf=false archive <sha>` and stages the result in `<dir>/stage/`.
3. Runs `jobs.common.lint` on the staged tree. Waived failures are allowed only with `--allow-waived`; they are recorded in `submit.json` and must be justified in the launch receipt.
4. Runs the job's CPU tests on the staged tree: every `tests/` file naming `jobs/<job>/` or `jobs.<job>`, plus `test_jobs_lint.py`.
5. Builds the context: the Dockerfile (with its CMD rewritten), `code/m1/`, and pinned data. Each data file's sha256 is checked against its pin file in the archive. `CONTEXT_MANIFEST.json` is written.
6. Writes a deterministic tar (sorted, mtime 2026-01-01, uid/gid 0, modes 644/755), compressed with Python `zstandard`. No `zstd` binary is needed.
7. Asserts there is no CR in any shipped `*.sh`, `*.sha256`, `*.json` or `Dockerfile`. `-text` vendored paths and `receipts/` are exempt.
8. Prints the sha256 and size.
9. Writes `<dir>/submit.json` with these calls in order:
   - `open_mission`
   - `create_context(sha256, size_bytes)`
   - upload (`curl -T`)
   - `finalize_context`
   - `validate_job`
   - `submit_job`, with chip, mission, `idempotency_key = <key>-<context sha256[:12]>`, build_timeout, command, env, max_duration, max_restarts, resume, queue_ttl, hf_cache, model pin and checks

The coordinator then works through `submit.json` in this order:
1. Add a planned `LEDGER.md` row.
2. Get the `spend` issue acknowledged.
3. Run `set_workspace_cap(MTD + cost.set_workspace_cap_increment_usd)`.
4. Paste the calls from `submit.json`.
5. Run `validate_job`. It must pass without `force`.
6. Run `submit_job`.

Fields for new jobs go in `job_params.json`. Legacy jobs use tables in `launch.py` transcribed from their JOB_SPECs (`tests/test_launch.py` cross-checks them). New jobs add:
- `"submit": {"command", "key", "checks", ["env", "clock_lock", "depends_on"]}`
- `"context": {"dockerfile", ["cmd_rewrite"], "data": [{"src", "dest", "pin"}]}`
- `"decision_rule": "<path.py>" | null`

`m2_pilot` and `m2_parent_arm` still use their own `build_context.sh`, because they need the parent-arm cache and a `guard_ready` dry run.

## After the job (same PR as the fix)

- Record the actual $ in `LEDGER.md`.
- Add a `CANDIDATES.md` row and a `cand/<YYYYMMDD>-<name>` tag if a variant was measured.
- **Any failure, timeout or surprise:** append a `LESSONS.md` entry **and** its enforcing check (a test, lint check or guard) in the same PR.

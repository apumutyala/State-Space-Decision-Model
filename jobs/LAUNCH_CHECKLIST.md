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
| 8 | every new job declares `decision_rule`: a pre-registered rule module whose literal constants are pinned by a test and committed **before** the run, or `null` + `no_verdict_reason` (JOB_SPEC wording is never trusted) | `decision_rule` | L-07 |
| 9 | the runner **executes** `python -m jobs.common.rate_guard` / `python -m jobs.<job>.preflight` (or `bash ...preflight...sh`) on a non-comment line before its long phase | `rate_guard` | L-01, L-08 |
| 10 | `.gitattributes` pins `*.sh *.py *.json *.sha256 Dockerfile` to LF; `git -c core.autocrlf=true archive HEAD` has no CR | `test_gitattributes_pin_lf`, `test_git_archive_head_has_no_cr_even_with_autocrlf` | L-14 |
| 11 | no control characters in `jobs/ tests/ scripts/` sources (heredoc escape collapse) | `test_no_control_characters_in_sources` | L-11 |
| 12 | gate reference is run through the production path on CPU first; it must pass its own gate | job's own test | L-06 |
| 13 | memory: a test or receipt shows peak at the real shape and micro-batch (estimates have been off by 2x) | job's own test | L-02, L-12 |

**Waivers.** Waivers live in `jobs/common/lint.py::WAIVERS`.
- Each waiver is dated, gives a reason, and covers legacy params files only.
- A stale waiver fails the test, so the list only shrinks.
- New jobs cannot be waived.

## Launch (coordinator, after an independent review)

```
python -m jobs.common.launch <job> --sha <sha> --review-url <PR URL> --out <dir> \
  [--mode M] [--candidate-branch stack/<family>/<variant>] [--active-reservation-job JOB] \
  [--data $SSDM_DATA] [--allow-waived]
```

Champion/integration runs use a SHA merged into `origin/agent-main`. Candidate runs may use the exact pushed tip of a
`stack/<family>/<variant>` branch before promotion; the launcher records that source branch and the independent-review
PR URL. The candidate branch tip must equal `--sha`. Do not merge candidate code into `agent-main` merely to measure it.

The command does the following. Each step refuses on failure.
1. Fetches origin and checks either that the SHA is an ancestor of `origin/agent-main`, or that it exactly equals the
   pushed tip of the supplied `stack/<family>/<variant>` candidate branch. A GitHub PR review URL is required and
   recorded in `submit.json` for both paths. The launcher validates its URL shape only; the coordinator must verify
   the independent review is for this exact head before proceeding.
2. Runs `git -c core.autocrlf=false archive <sha>` and stages the result in a fresh `<dir>/stage/`. Symlinks or other non-regular members, an unknown `--mode`, or a reused `--out` are refused.
3. Runs **the sha's own** `jobs/common/lint.py`, loaded from the staged tree (not the caller's checkout). Waived failures are allowed only with `--allow-waived`; they are recorded in `submit.json` and must be justified in the launch receipt.
4. Runs the job's CPU tests on the staged tree: every `tests/` file naming `jobs/<job>/` or `jobs.<job>`, plus `test_jobs_lint.py`.
5. Builds the context: the Dockerfile (with its CMD rewritten), `code/m1/`, and pinned data. Each data file's sha256 is checked against its pin file in the archive. `CONTEXT_MANIFEST.json` is written.
6. Writes a deterministic tar (sorted, mtime 2026-01-01, uid/gid 0, modes 644/755), compressed with Python `zstandard`. No `zstd` binary is needed.
7. Asserts there is no CR in any shipped `*.sh`, `*.sha256`, `*.json`, `*.py` or `Dockerfile`. `-text` vendored paths and `receipts/` are exempt.
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
2. Get the `spend` issue reviewed per COORDINATION 'Review sufficiency'. The other agent's acknowledgement is requested but not blocking.
3. Add every other active reservation with a repeated `--active-reservation-job <costs.PARAMS job key>`. Then run
   `set_workspace_cap(MTD + cost.set_workspace_cap_increment_usd)`. The increment includes the configured
   `workspace_cap_jobs` plus these active reservations; a chained job without its configured list is refused. For
   RWKV7 stage B, it covers both `rwkv7_readout` and `rwkv7_lora`, not only the child. The operator must reconcile
   this list against the live workspace before submitting.
4. Paste the calls from `submit.json`.
5. Run `validate_job`. It must pass without `force`.
6. Run `submit_job`.

Fields for new jobs go in `job_params.json`. Legacy jobs use tables in `launch.py` transcribed from their JOB_SPECs (`tests/test_launch.py` cross-checks them). New jobs add:
- `"submit": {"command", "env", "key", "checks", ["label", "clock_lock", "depends_on"]}`; platform fields (chip, chip_count, mission, build_timeout, max_duration, max_restarts, resume, queue_ttl, hf_cache, model) are required in job_params: launch.py has **no defaults**, and its output must equal the JOB_SPEC submit block field by field (`test_submit_args_equal_the_job_spec_field_by_field`)
- `"context": {"dockerfile", ["cmd_rewrite"], "data": [{"src", "dest", "pin"}]}`
- `"decision_rule": "<path.py>"`, or `null` + `"no_verdict_reason"`
- `"workspace_cap_jobs": ["<costs.PARAMS job key>", ...]` when multiple jobs can bill against the workspace concurrently

`m2_pilot` and `m2_parent_arm` still use their own `build_context.sh`, because they need the parent-arm cache and a `guard_ready` dry run.

## After the job (same PR as the fix)

- Record the actual $ in `LEDGER.md`.
- Add a `CANDIDATES.md` row and a `cand/<YYYYMMDD>-<name>` tag if a variant was measured.
- **Any failure, timeout or surprise:** append a `LESSONS.md` entry **and** its enforcing check (a test, lint check or guard) in the same PR.

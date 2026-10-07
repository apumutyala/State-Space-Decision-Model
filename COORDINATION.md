# Coordination protocol

Codex owns the two gated SSDM stages under the user’s 2026-10-07 takeover direction. Fresh-context adversarial subagents provide independent review; Git is the shared record, and no claim, plan or result counts until it is in a reviewed PR. The user approves anything that reaches `main`.

## Who owns what

| Track | Owner | Scope |
|---|---|---|
| **D: SSM-Decision (System 1)** | `codex` (user-directed Track D takeover; branches `codex/d-*` and `stack/rwkv7/*`) | Backbones (Q9-AF conversion, RWKV7, Mamba-3 reference), decision heads, typed outputs, calibration, injection/steer evaluation, Decision Index harness, the semantic-interface arms *inside* the core (records, KEEP/SET/RETRACT/UNKNOWN/CONFLICT) |
| **L: SSM-Language (grown from D)** | `codex` (branches `codex/l-*`; paths `plans/codex/`, `language_growth/`) | Semantic program language (Assert/Infer/Compare/Explain/Ask/Quote/Hypothesize/Plan), module contracts and checker, realizer/grammar, verified emission, L0–L4 growth, authority and information-flow specification |
| **Shared** | Codex, reviewed by a fresh-context adversarial subagent | `interfaces/DECISION_LANGUAGE_V0.md` (canonical D→L seam; existing changes recorded in `INTERFACE.md`), evaluation harness, intervention protocol, spend ledger, `techniques/` |

Codex owns both tracks. A fresh-context adversarial subagent reviews each PR and does not become a file owner or spend approver.

## Branches

```
main          user-approved milestones only (tagged m0, m1, ...)
agent-main    integration branch; reviewed PRs merge here
codex/d-<topic>  Track D takeover work branches
codex/l-<topic>  Track L work branches
stack/rwkv7/<variant>  measured RWKV7 decision-model variants
```

- Branch from `agent-main`. Rebase on it before opening a PR.
- Never push directly to `agent-main` or `main`. Never force-push a shared branch.
- Stage only your own paths (no `git add -A` over the other agent's uncommitted files).
- Small PRs (one experiment, one module, or one doc), merged often. A PR open more than 2 working sessions gets split or closed.
- **Branches and the champion.**
  - Every hypothesis variant gets its own branch, `stack/<family>/<variant>`, pushed at every checkpoint. WIP commits are fine there.
  - Infra goes on `claude/*` or `codex/*`.
  - `agent-main` holds only the best candidate (`CHAMPION.md`) plus shared infra.
  - Every measured variant gets a row in `CANDIDATES.md`, kept or folded.
  - Promotion requires a paired DEV-subset CI above zero vs the champion, plus G12/G13, a receipt and an independent review. Tag the promotion `champion/<YYYYMMDD>-<name>`; tag measured candidates `cand/<YYYYMMDD>-<name>`.

## Before any paid job / before any PR

- Read `LESSONS.md`, the failure classes we already paid for.
- Follow `jobs/LAUNCH_CHECKLIST.md`.
  - `tests/test_jobs_lint.py` must pass.
  - Launch only through `python -m jobs.common.launch <job> --sha <merged sha> --out <dir>`, then paste its `submit.json`.
- **Every incident appends a `LESSONS.md` entry and an enforcing check (a test, lint check or guard) in the same PR.** A lesson without a check is not done.
- **Never kill processes by image name or name pattern** (`taskkill /IM` or `/FI` window-title filters, `pkill`, `killall`) on the shared laptop. That kills other agents' and Codex's processes. Kill only PIDs you started yourself and recorded (`taskkill /PID <pid>`, `kill <pid>`). See LESSONS L-23, enforced by `tests/test_no_kill_by_name.py`.

## PR rules

Every PR uses the template in `.github/pull_request_template.md`.

- **Independent review is mandatory.** A fresh-context adversarial subagent that did not write the change reviews each PR into `agent-main`. The coordinator does not count their own review as independent.
- **Results PRs need an independent re-measurement.** Before merge, the reviewer re-runs the numbers from the committed receipt, or at least recomputes them from the raw predictions. A win that doesn't reproduce is logged as a false positive in `LEDGER.md`.
- **Interface changes** (anything touching `INTERFACE.md` or a type in it) need an `interface` issue, an independent fresh-context adversarial review, and a version bump. Changes reaching `main` still require the user's approval.
- Review verdicts: `approve`, `request-changes` (with specific blocking items), or `comment`. Disagreements that survive two rounds go to the user as a `decision-needed` issue.

## Messaging

| Need | Channel |
|---|---|
| Review, question about a specific change | PR review comments |
| Blocker, cross-track request, design question | GitHub issue with a label: `blocker`, `interface`, `decision-needed`, `result`, `spend` |
| Anything the user must decide | issue labelled `decision-needed`, assigned to the user, plus one line in the agent's chat |
| Handoff notes | `coordination/` (one file per handoff/response) |
| Running status | `STATUS.md` (one section per track, updated in each PR) |

Chat with the user is not a channel between agents. If it matters to the other agent, it goes in an issue or PR.

## When to merge to `agent-main`

Merge when either of these is true:
1. **Milestone:** a gate in `PLAN.md` passes or fails, with its receipt.
2. **Unblock:** the other track needs the change (new interface version, harness fix, checkpoint pointer).

Also merge before any paid run, so the exact code that ran is in shared history.

## When `agent-main` goes to `main`

At each milestone (m0 harness parity, m1 backbone decision, m2 decision core v1, m3 L0 language, ...), one agent opens `agent-main → main`. It carries a short takeaway-first summary with charts. The user approves it, and it gets a tag.

## Receipts (required for any number)

Every reported number needs all of these:
- the commit SHA;
- the exact command and config;
- data and split hashes;
- seeds;
- the hardware, and the rate actually billed;
- median and spread over N runs;
- the raw predictions file (or its hash plus a storage location for anything over 50 MB).

Receipts live in `receipts/<date>-<track>-<slug>/`.

## Compute and spend

- **Budget:** a single shared $525 SF Compute envelope: $75 joint setup/baseline/throughput gate · $250 System 1 · $100 language · $100 shared reserve. $5k planning ceiling. The first commitment is the joint $75 gate. Exceeding a line needs the user's explicit authorization.
- **Before launching a paid job:**
  - add a planned row to `LEDGER.md`, with the estimated $ and a stop condition;
  - set a runtime ceiling;
  - stop the node afterwards;
  - record the actual cost.
- **Credit expiry:** check it first. If the credits are promotional, they may expire about 3 weeks after grant.
- **Teachers:** open-weight only (Clef, Qwen, Gemma). Jev and frontier-API terms forbid distillation.
- **Hardware:** one experiment per GPU at a time. Agents coordinate node use through `spend` issues.

## Git identity

Commit as `Apuroop Mutyala <apumutyala@gmail.com>`, with no AI or co-author trailers. The repo stays private.

## Phases, pipelining and proof before spend (binding for both stages)

1. **Break work into phases with explicit exit proofs.** Each phase names its hypothesis, deliverable, exit proof (test, fixture or receipt) and the decision it unlocks. No phase starts paid compute until the previous phase's proof is merged on `agent-main`.
2. **Spend ladder.** Order the work as:
   - $0 local/CPU work: code, fixtures, smoke tests on tiny models;
   - real-hardware qualification: scope and maximum cost derived from the measured workload and the approved track allocation (the former fixed ≤$20 qualification ceiling is superseded by the user's 2026-10-07 direction);
   - the capped experiment;
   - scale-up.

   Each paid rung needs the previous rung's merged receipt, a `spend` issue with the exact estimate and stop condition, an independent fresh-context adversarial review, and the user's explicit OK in that issue.
3. **Parallelise within a track.** Codex runs independent bounded sub-tasks concurrently using fresh-context subagents or worktrees, one owner and branch per task (`codex/d-<task>`, `codex/l0-<task>`). Work that doesn't depend on hardware (code, data, analysis, review) never waits for GPU work.
4. **Pipeline across stages.** While one stage measures, the next is implemented and the previous one is reviewed. GPU time is serialised (one experiment per GPU); nothing else is.
5. **Adversarial review.** A fresh-context adversarial subagent reviews each PR before merge and independently re-derives reported numbers from receipts or raw predictions.
6. **Fold rules.** Fold a line when a measurement falsifies it, when a better backlog item outranks it, or after two consecutive checks with no progress. Log the fold in `LEDGER.md`.

## Compute providers

1. **Primary: SF Compute autoresearch.** $525 credits; H100 at $2.97/h batch and $3.96/h interactive.
2. **Backup: Lambda Cloud.** Use it when SF lacks capacity or a different shape is needed (e.g. A100-40GB at $1.99/h, or multi-GPU).
   - **Key:** read from `%USERPROFILE%\.lambda_key`. Never print it, and never put it in argv or a repo.
   - **Tooling:** reuse the kill-safe tooling in `chl-typed-control/infra/` (`lambda_ctl.py`, `deadline_runner.sh`, `watchdog.py`).
   - **Per run:** every run arms an on-instance deadline (`deadline_runner.sh`) *and* the local watchdog before launch.
   - **Prefixes:** each paid worker uses an explicitly assigned instance-name prefix. Codex only lists and terminates instances it launched; never terminate an unowned instance.
   - **Billing:** continues until *terminate*; an OS shutdown doesn't stop it.
   - **Spend:** counts against the $5k ceiling, not the SF credits. It's logged in `LEDGER.md` with provider = lambda, and goes through the same `spend` issue and spend ladder.
3. **Local:** the RTX 5070 Ti laptop (12 GB) handles CPU-only $0-rung smoke tests. Do not run GPU jobs on anyone's laptop; workers share the device.

## Independent review and spend sufficiency (user direction, 2026-10-07)

Codex owns Track D and Track L; fresh-context adversarial subagents are the independent reviewers. One nonauthor reviewer is required before merge to `agent-main`. Fix material findings first and rerun the relevant checks. Results require independent remeasurement. These rules never authorize a merge to `main`.

Every paid job requires its own `spend` issue, planned `LEDGER.md` row, hard runtime/cost stop, and the user's explicit OK in that issue after independent review. Stay within the approved envelope and track allocation; a reviewer's GO is not spend authorization.

## Lean review (user direction, 2026-10-06)

Reviews check only these:
- correctness;
- contamination and test-slice integrity;
- safety-claim soundness;
- spend safety.

They don't rebuild datasets or re-verify byte-for-byte reproducibility. A recorded hash plus the build's own guard is sufficient. Small data-quality fixes (well under 1 pt impact) never block training. Every PR, including docs and spec-only changes, receives one fresh-context nonauthor review.

## Collaborators (2026-10-06)

Human teammates with write access, and the coding agents they send, work on research directions of their own choosing without delaying the D-first critical path or gated language stage. Full onboarding: `HANDOFF.md` (humans) and `AGENT_HANDOFF.md` (agents).

- **Branches.** `collab/<github-user>/<topic>`, branched from `agent-main` and rebased on it before a PR. Never push to `main` or `agent-main`; never touch `claude/*` or `codex/*` branches or worktrees; never force-push a shared branch; stage only your own paths.
- **PRs.** Into `agent-main` only, when ready, using the PR template with a scoped claim and a receipt. Merge needs one independent fresh-context adversarial reviewer who did not write the code; findings are fixed first. Results PRs get their headline number re-derived by someone else.
- **Shared contracts.** `interfaces/DECISION_LANGUAGE_V0.md`, `INTERFACE.md`, `decide/schema.py` wire shapes, `safety_eval/PREREG.md`, `SAFETY_SPEC.md` rows and `harness/manifests/` change only through an `interface` issue, independent adversarial review and version bump. Anything reaching `main` needs the user's approval.
- **Integrity.** Never open or tune on the sealed TEST slice, Decision Index test items or the held-out D1-B templates. Open-weight teachers only. Every training set goes through the contamination guard with a committed log.
- **Spend.** Collaborators do not spend the shared SF credits (or Lambda, which counts against the $5k ceiling) without the user's explicit OK on a `spend` issue that carries the estimate, stop condition and a planned `LEDGER.md` row; the spend ladder applies ($0 CPU → real-hardware qualification sized within the approved track allocation → capped → scale-up). Do not impose the superseded fixed ≤$20 qualification ceiling. Keep a measured per-job maximum and stop condition, never exceed the shared $525 envelope or $250 System 1 line without the user's updated authorization, and never run GPU jobs on anyone's laptop. Record actual cost and stop nodes.
- **Identity.** Each contributor commits under their own name and email; nobody commits as another person. The repo stays private.

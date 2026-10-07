# Coordination protocol

Two agents build one model family. Git is the only shared memory: no claim, plan or result counts until it is in a reviewed PR. The user approves anything that reaches `main`.

## Who owns what

| Track | Owner | Scope |
|---|---|---|
| **D: SSM-Decision (System 1)** | `claude` (Claude Code; branches `claude/*`) | Backbones (Q9-AF conversion, RWKV7, Mamba-3 reference), decision heads, typed outputs, calibration, injection/steer evaluation, Decision Index harness, the semantic-interface arms *inside* the core (records, KEEP/SET/RETRACT/UNKNOWN/CONFLICT) |
| **L: SSM-Language (grown from D)** | `codex` (branches `codex/*`; paths `plans/codex/`, `language_growth/`) | Semantic program language (Assert/Infer/Compare/Explain/Ask/Quote/Hypothesize/Plan), module contracts and checker, realizer/grammar, verified emission, L0–L4 growth, authority and information-flow specification |
| **Shared** | both, cross-reviewed | `interfaces/DECISION_LANGUAGE_V0.md` (canonical D→L seam; Claude deltas in `INTERFACE.md`), evaluation harness, intervention protocol, spend ledger, `techniques/` |

Ownership means you write it and merge it after review. It does not mean the other agent can't open PRs against it; those just need the owner's approval.

## Branches

```
main          user-approved milestones only (tagged m0, m1, ...)
agent-main    integration branch; both agents merge here after cross-review
claude/<topic>    Track D work branches
codex/<topic>     Track L work branches
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

## PR rules

Every PR uses the template in `.github/pull_request_template.md`.

- **Cross-review is mandatory.** The other agent reviews every PR into `agent-main`. Nobody approves their own PR.
- **Results PRs need an independent re-measurement.** Before merge, the reviewer re-runs the numbers from the committed receipt, or at least recomputes them from the raw predictions. A win that doesn't reproduce is logged as a false positive in `LEDGER.md`.
- **Interface changes** (anything touching `INTERFACE.md` or a type in it) need both agents' approval plus a version bump.
- Review verdicts: `approve`, `request-changes` (with specific blocking items), or `comment`. Disagreements that survive two rounds go to the user as a `decision-needed` issue. Neither agent overrules the other.

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

- **Budget:** a single shared $525 SF Compute envelope (one allowance, not one per agent): $75 joint setup/baseline/throughput gate · $250 System 1 · $100 language · $100 shared reserve. $5k planning ceiling. The first commitment is the joint $75 gate. Exceeding a line needs the other agent's sign-off.
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

## Phases, pipelining and proof before spend (binding for both agents)

1. **Break work into phases with explicit exit proofs.** Each phase names its hypothesis, deliverable, exit proof (test, fixture or receipt) and the decision it unlocks. No phase starts paid compute until the previous phase's proof is merged on `agent-main`.
2. **Spend ladder.** Order the work as:
   - $0 local/CPU work: code, fixtures, smoke tests on tiny models;
   - ≤$20 qualification on real hardware: throughput, memory, numerics parity;
   - the capped experiment;
   - scale-up.

   Each rung needs the previous rung's receipt, plus a `spend` issue with the estimated $ and a stop condition, acknowledged by the other agent.
3. **Parallelise within a track.** Each agent runs independent sub-tasks concurrently: subagents or worktrees, one branch each (`claude/m0-<task>`, `codex/l0-<task>`). Work that doesn't depend on hardware (code, data, analysis, review) never waits for GPU work.
4. **Pipeline across stages.** While one stage measures, the next is implemented and the previous one is reviewed. GPU time is serialised (one experiment per GPU); nothing else is.
5. **Adversarial review inside each track** before the cross-agent PR. Any reported number is re-derived by a different agent or process.
6. **Fold rules.** Fold a line when a measurement falsifies it, when a better backlog item outranks it, or after two consecutive checks with no progress. Log the fold in `LEDGER.md`.

## Compute providers

1. **Primary: SF Compute autoresearch.** $525 credits; H100 at $2.97/h batch and $3.96/h interactive.
2. **Backup: Lambda Cloud.** Use it when SF lacks capacity or a different shape is needed (e.g. A100-40GB at $1.99/h, or multi-GPU).
   - **Key:** read from `%USERPROFILE%\.lambda_key`. Never print it, and never put it in argv or a repo.
   - **Tooling:** reuse the kill-safe tooling in `chl-typed-control/infra/` (`lambda_ctl.py`, `deadline_runner.sh`, `watchdog.py`).
   - **Per run:** every run arms an on-instance deadline (`deadline_runner.sh`) *and* the local watchdog before launch.
   - **Prefixes:** each agent uses its own instance-name prefix, `ssdm-claude-` or `ssdm-codex-`. An agent only lists and terminates instances under its own prefix, and never terminates the other agent's instances.
   - **Billing:** continues until *terminate*; an OS shutdown doesn't stop it.
   - **Spend:** counts against the $5k ceiling, not the SF credits. It's logged in `LEDGER.md` with provider = lambda, and goes through the same `spend` issue and spend ladder.
3. **Local:** the RTX 5070 Ti laptop (12 GB) handles all $0-rung smoke tests. Check `nvidia-smi` before starting, since workers share it.

## Stalled cross-review (user authorization, 2026-10-06)

When the other agent's review is stalled, Track D may merge its own PRs into `agent-main` with the user's explicit authorization. Three conditions must all hold:

1. An independent internal adversarial review approved the PR.
2. The other agent's known findings were fixed first, each with a regression test.
3. The integrator test run on the merged tree passes.

The merge comment records all three. The other agent reviews after the merge, and any finding gets a fix PR. A result claim the other agent couldn't reproduce stays labelled as reproduced by one track only.

This rule never applies to `main`. Spend follows 'Review sufficiency' below (one independent review — the other agent's or a fresh-context adversarial reviewer — plus the envelope and workspace cap); the other agent's acknowledgement on the spend issue is requested but not blocking.

## Review sufficiency (user decision, 2026-10-06)

One independent review is sufficient to merge or to launch spend. It can be either:
- the other agent's review; or
- a fresh-context adversarial reviewer subagent that didn't write the code.

The review's findings must be fixed first, and the spend must stay within the approved envelope and workspace cap. Every spend still gets a ledger row and a launch receipt. The other agent reviews afterwards, and any findings get fix PRs. This never applies to `main`.

## Lean review (user direction, 2026-10-06)

Reviews check only these:
- correctness;
- contamination and test-slice integrity;
- safety-claim soundness;
- spend safety.

They don't rebuild datasets or re-verify byte-for-byte reproducibility. A recorded hash plus the build's own guard is sufficient. Small data-quality fixes (well under 1 pt impact) never block training. Docs and spec-only changes need no review.

## Collaborators (2026-10-06)

Human teammates with write access, and the coding agents they send, work on research directions of their own choosing without delaying or disturbing the two owning tracks. Full onboarding: `HANDOFF.md` (humans) and `AGENT_HANDOFF.md` (agents).

- **Branches.** `collab/<github-user>/<topic>`, branched from `agent-main` and rebased on it before a PR. Never push to `main` or `agent-main`; never touch `claude/*` or `codex/*` branches or worktrees; never force-push a shared branch; stage only your own paths.
- **PRs.** Into `agent-main` only, when ready, using the PR template with a scoped claim and a receipt. Merge needs one independent review per "Review sufficiency" above (the other track's agent, or a fresh-context adversarial reviewer that did not write the code); findings fixed first; lean-review scope applies. Results PRs get their headline number re-derived by someone else.
- **Shared contracts.** `interfaces/DECISION_LANGUAGE_V0.md`, `INTERFACE.md`, `decide/schema.py` wire shapes, `safety_eval/PREREG.md`, `SAFETY_SPEC.md` rows and `harness/manifests/` change only through an `interface` issue approved by both owners, with a version bump.
- **Integrity.** Never open or tune on the sealed TEST slice, Decision Index test items or the held-out D1-B templates. Open-weight teachers only. Every training set goes through the contamination guard with a committed log.
- **Spend.** Collaborators do not spend the shared SF credits (or Lambda, which counts against the $5k ceiling) without the user's explicit OK on a `spend` issue that carries the estimate, stop condition and a planned `LEDGER.md` row; the spend ladder applies ($0 CPU → ≤ $20 qualification → capped → scale-up). Never run GPU jobs on anyone's laptop. Record actual cost and stop nodes.
- **Identity.** Each contributor commits under their own name and email; nobody commits as another person. The repo stays private.

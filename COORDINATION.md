# Coordination protocol

Two agents build one model family. Git is the only shared memory: no claim, plan or result counts until it is in a reviewed PR. The user approves anything that reaches `main`.

## Who owns what

| Track | Owner | Scope |
|---|---|---|
| **D: SSM-Decision (System 1)** | `claude` (Claude Code) | Backbones (Q9-AF conversion, RWKV7, Mamba-3 reference), decision heads, typed outputs, calibration, injection/steer evaluation, Decision Index harness, the semantic-interface arms *inside* the core (records, KEEP/SET/RETRACT/UNKNOWN/CONFLICT) |
| **L: SSM-Language (grown from D)** | `partner` | Semantic program language (Assert/Infer/Compare/Explain/Ask/Quote/Hypothesize/Plan), module contracts and checker, realizer/grammar, verified emission, L0–L4 growth, authority and information-flow specification |
| **Shared** | both, cross-reviewed | `INTERFACE.md` (the D→L seam), evaluation harness, intervention protocol, spend ledger, `techniques/` |

Ownership means you write it and merge it after review. It does not mean the other agent can't open PRs against it; those just need the owner's approval.

## Branches

```
main          user-approved milestones only (tagged m0, m1, ...)
agent-main    integration branch; both agents merge here after cross-review
claude/<topic>    Track D work branches
partner/<topic>   Track L work branches
```

- Branch from `agent-main`. Rebase on it before opening a PR.
- Never push directly to `agent-main` or `main`. Never force-push a shared branch.
- Small PRs (one experiment, one module, or one doc), merged often. A PR open more than 2 working sessions gets split or closed.

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

- **Budget:** $525 in SF Compute credits and a $5k ceiling. Track D gets $300 of the credits, Track L $100 and the shared pool $125 (harness, reserve, independent re-runs). Both tracks need sign-off from the other agent to go over.
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

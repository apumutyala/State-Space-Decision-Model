# Start here

Welcome. This folder is the onboarding pack for the State-Space-Decision-Model project. It's a snapshot published on `main`; the live work happens on the **`agent-main`** branch.

## Read in this order

1. **`HANDOFF.md`**: the story, the architecture, what's been measured, what failed, the open decisions, and the stage-by-stage plan with costs. Start here.
2. **`RESEARCH_DIRECTIONS.md`**: a menu of directions you could take, including probabilistic graphical models (MRFs/CRFs, HMMs, probabilistic circuits) and others. Each comes with a first experiment, a cost, the key papers, and honest caveats. You choose.
3. **`AGENT_HANDOFF.md`**: the rules and orientation to hand to your coding agent before it touches the repo: branches, sealed test data, spend, review.
4. **`COORDINATION.md`**: how the two owning agents (Claude: decision model; Codex: language growth) and collaborators work together.
5. **`README.md`**: a copy of the repo overview.

## Where things live

- **Code, data pipelines, safety specs, job specs and results:** branch `agent-main`. Paths cited in these docs (`GOAL.md`, `PLAN.md`, `SAFETY_SPEC.md`, `controlsafe/`, `jobs/`, `techniques/` …) refer to that branch: `git checkout agent-main`, or browse it on GitHub with the branch selector.
- **Your work:** a branch named `collab/<your-github-user>/<topic>`, created from `agent-main`. Open PRs into `agent-main` when ready. Never push to `main` or `agent-main`.
- **Data is not in git.** The Decision Index TEST slice is sealed; never open it or tune on it.
- **No GPU jobs on laptops.** Paid cloud jobs need the project owner's explicit OK.

## Snapshot date

Published 2026-10-06. For the latest status, see `STATUS.md`, `LEDGER.md` and the open PRs on `agent-main`.

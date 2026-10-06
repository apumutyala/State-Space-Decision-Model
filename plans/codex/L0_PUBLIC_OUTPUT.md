# Public-output boundary (L0f preparation)

- Hypothesis: a dedicated public entry point can expose only freshly checked language or one fixed failure status, while planner traces remain permissioned diagnostics.
- Deliverable: a small public wrapper around the typed planner, rechecking/releasing its program before returning text. Never serialize PlannerResult or provider exceptions to public callers.
- Exit proof: CPU cases for valid text, kernel/provider/identity exceptions mapped to the same failure shape, rejected/forged text not emitted, and changed authority denied. Tests are scoped to the callable boundary; there is no network authentication or universal timing-side-channel claim.
- Decision unlocked: satisfy the concrete public projection part of proposed G11 after independent review and merge. No learned capability, spend, or external action is authorized.

Code/tests under `language_growth/`, branch `codex/l0-public-output`, worktree `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-public-output`. Use the existing planner, kernel and renderer; no new service or generic framework.

Integration dependency: the kernel and renderer are merged on `agent-main`. The typed planner and bridge are supplied by PR #13 (`codex/l0-typed-planner`, source `e19c308`); this branch does not vendor them. The CPU proof must use those exact PR #13 files until #13 merges, then rerun from a clean merged checkout before claiming the L0f gate.

# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).

## Track L — SSM-Language (owner: codex)
- 2026-10-06: plan in plans/codex/LANGUAGE_GROWTH_PLAN.md; interface interfaces/DECISION_LANGUAGE_V0.md. Next: L0 types/interpreter/renderer + joint fixtures (CPU), oracle/replay backends.
- Free-running development runner now invokes the planner from initial task/evidence only, then checks task relevance separately from kernel validity. It retains failures and distinguishes repair conditions. CPU scripted tests cover wrong-but-valid output, equivalent comparisons, equality, and surface tampering. No learned or held-out result is claimed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

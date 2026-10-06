# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).

## Track L — SSM-Language (owner: codex)
- 2026-10-06: plan in plans/codex/LANGUAGE_GROWTH_PLAN.md; interface interfaces/DECISION_LANGUAGE_V0.md. Next: L0 types/interpreter/renderer + joint fixtures (CPU), oracle/replay backends.
- L0e runtime fixture replay checks the real kernel/renderer and saves per-attempt outputs. Authored task labels remain annotations; provider calls are zero. Internal review checked label isolation, private-state serialization and scripted repair accounting. Depends on #4, #5, #10 and #11; cross-review and learned gates remain pending.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

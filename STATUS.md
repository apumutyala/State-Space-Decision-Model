# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-c, PR claude/m0-steer-eval): safety_eval v0.1 + PREREG (steer headline) + Qwen3-0.6B pipeline smoke receipt (receipts/2026-10-06-D-m0-steer).
- 2026-10-06 (PR #19, claude/clef-baseline): Clef/Clef-flash decide() backend over vendored joint_schema_model.py, CPU mapping tests (Clef's own tokenizer), SF Compute job spec (not submitted; gated on the PREREG k=8 D1-B amendment + a spend issue, ~$18 expected / $47.52 ceiling).
- 2026-10-06 (m0-b): `decide/` adapter + replay/oracle/HF-readout backends + replay fixture v0 (37 cases, Qwen3.5-0.8B) delivered to Track L. Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M load locally (GPU). Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- 2026-10-06: plan in plans/codex/LANGUAGE_GROWTH_PLAN.md; interface interfaces/DECISION_LANGUAGE_V0.md. Next: L0 types/interpreter/renderer + joint fixtures (CPU), oracle/replay backends.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

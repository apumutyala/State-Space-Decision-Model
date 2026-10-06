# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b): `decide/` adapter + replay/oracle/HF-readout backends + replay fixture v0 (37 cases, Qwen3.5-0.8B) delivered to Track L. Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M load locally (GPU). Receipt: receipts/2026-10-06-D-m0-decide/.
- 2026-10-06 (m2-prep, $0): `heads/` (backbone-agnostic hidden-state interface; options-first ordered + symmetric layouts; integrator / final-state fork / bounded-S snapshot / Head B / Clef-attention control; noul/choice/score cumulative-link) and `train/` (frozen backbone + LoRA r=256 default, CE vs LS-CE+Brier(+RPS) arms, feature temperature, eval-card hook with harness/steer stubs, checkpoint/resume, H100 estimator). CPU tests on tiny random backbones only. First paid-run spec: train/M2_RUN_SPEC.md (not launched).

## Track L — SSM-Language (owner: codex)
- 2026-10-06: plan in plans/codex/LANGUAGE_GROWTH_PLAN.md; interface interfaces/DECISION_LANGUAGE_V0.md. Next: L0 types/interpreter/renderer + joint fixtures (CPU), oracle/replay backends.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

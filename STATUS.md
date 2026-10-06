# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-c, PR claude/m0-steer-eval): safety_eval v0.1 + PREREG (steer headline) + Qwen3-0.6B pipeline smoke receipt (receipts/2026-10-06-D-m0-steer).
- 2026-10-06 (safety, PR claude/g4-g5-guarantees): G4 erase-and-check certificate (`safety_eval/certify.py`, exact k=1/2 + randomized-ablation arm) and G5 LTT selective risk (`safety_eval/selective.py`, feasibility floor, cluster-thinned bounds, shift stress test) as wrappers around decide(); PREREG §6; CPU stand-in receipt receipts/2026-10-06-D-g4g5-standin (pipeline only).
- 2026-10-06 (m0-b): `decide/` adapter + replay/oracle/HF-readout backends + replay fixture v0 (37 cases, Qwen3.5-0.8B) delivered to Track L. Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M load locally (GPU). Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- 2026-10-06: plan in plans/codex/LANGUAGE_GROWTH_PLAN.md; interface interfaces/DECISION_LANGUAGE_V0.md. Next: L0 types/interpreter/renderer + joint fixtures (CPU), oracle/replay backends.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

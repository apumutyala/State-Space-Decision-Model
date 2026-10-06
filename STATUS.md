# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.
- 2026-10-06 (m0-b): `decide/` adapter + replay/oracle/HF-readout backends + replay fixture v0 (37 cases, Qwen3.5-0.8B) delivered to Track L. Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M load locally (GPU). Receipt: receipts/2026-10-06-D-m0-decide/.
- 2026-10-06 (m2-prep, $0): `heads/` (backbone-agnostic hidden-state interface; options-first ordered + symmetric layouts; integrator / final-state fork / bounded-S snapshot / Head B / Clef-attention control; noul/choice/score cumulative-link) and `train/` (frozen backbone + LoRA r=256 default, CE vs LS-CE+Brier(+RPS) arms, feature temperature, eval-card hook with harness/steer stubs, checkpoint/resume, H100 estimator). CPU tests on tiny random backbones only. First paid-run spec: train/M2_RUN_SPEC.md (not launched).

## Track L — SSM-Language (owner: codex)
- Interface, corrected fixture pack and repaired semantic kernel/profile0.1.1 are merged (#4, #5, #10). These are CPU prerequisites, not learned-level gates.
- Track D approved the ordinal-only renderer in #11. This branch integrates the merged dependencies and rechecks unchanged renderer code before integration.
- Planner and development evaluation remain in cross-review/integration; no paid run or learned L0-L4 gate has passed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

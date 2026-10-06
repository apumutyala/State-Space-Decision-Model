# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.
- 2026-10-06 (m0-c, PR claude/m0-steer-eval): safety_eval v0.1 + PREREG (steer headline) + Qwen3-0.6B pipeline smoke receipt (receipts/2026-10-06-D-m0-steer).

- 2026-10-06 m1-prep (PR claude/m1-conversion-pilot): attention->GDN pipeline built + kernels verified (fla works on Windows); 0.8B curve INCOMPLETE (local GPU runs stopped after crash); 9B est. $96-269 full / $26-36 one-layer; <=$20 H100 qualification spec in receipts/2026-10-06-D-m1-pilot-0p8b/README.md (not launched).
- 2026-10-06 (safety, PR claude/g4-g5-guarantees): G4 erase-and-check certificate (`safety_eval/certify.py`, exact k=1/2 + randomized-ablation arm) and G5 LTT selective risk (`safety_eval/selective.py`, feasibility floor, cluster-thinned bounds, shift stress test) as wrappers around decide(); PREREG §6; CPU stand-in receipt receipts/2026-10-06-D-g4g5-standin (pipeline only).
- 2026-10-06 (PR #19, claude/clef-baseline): Clef/Clef-flash decide() backend over vendored joint_schema_model.py, CPU mapping tests (Clef's own tokenizer), SF Compute job spec (not submitted; gated on the PREREG k=8 D1-B amendment + a spend issue, ~$18 expected / $47.52 ceiling).
- 2026-10-06 (m0-b): `decide/` adapter + replay/oracle/HF-readout backends + replay fixture v0 (37 cases, Qwen3.5-0.8B) delivered to Track L. Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M load locally (GPU). Receipt: receipts/2026-10-06-D-m0-decide/.
- 2026-10-06 (m2-prep, $0): `heads/` (backbone-agnostic hidden-state interface; options-first ordered + symmetric layouts; integrator / final-state fork / bounded-S snapshot / Head B / Clef-attention control; noul/choice/score cumulative-link) and `train/` (frozen backbone + LoRA r=256 default, CE vs LS-CE+Brier(+RPS) arms, feature temperature, eval-card hook with harness/steer stubs, checkpoint/resume, H100 estimator). CPU tests on tiny random backbones only. First paid-run spec: train/M2_RUN_SPEC.md (not launched).

## Track L — SSM-Language (owner: codex)
- Public-output wrapper is ready for cross-review: public audience required before planning, returned program rechecked/released, and ordinary failures projected to one fixed status without traces or reasons. CPU-only callable-boundary proof; no network-authentication or learned capability claim.
- Interface, corrected fixture pack and repaired semantic kernel/profile0.1.1 are merged (#4, #5, #10). These are CPU prerequisites, not learned-level gates.
- Track D approved the ordinal-only renderer in #11; it is now merged. Derived proof steps are kernel ordinals, not model-authored labels.
- The wrapper still depends on PR #13's typed planner and bridge. Development evaluation remains in cross-review/integration; no paid run or learned L0–L4 gate has passed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

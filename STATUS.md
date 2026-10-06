# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- Public-output wrapper is ready for cross-review: public audience required before planning, returned program rechecked/released, and ordinary failures projected to one fixed status without traces or reasons. CPU-only callable-boundary proof; no network-authentication or learned capability claim.
- Interface, corrected fixture pack and repaired semantic kernel/profile0.1.1 are merged (#4, #5, #10). These are CPU prerequisites, not learned-level gates.
- Track D approved the ordinal-only renderer in #11; it is now merged. Derived proof steps are kernel ordinals, not model-authored labels.
- The wrapper still depends on PR #13's typed planner and bridge. Development evaluation remains in cross-review/integration; no paid run or learned L0–L4 gate has passed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

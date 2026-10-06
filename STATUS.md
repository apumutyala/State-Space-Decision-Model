# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- Interface, corrected fixture pack and repaired semantic kernel/profile0.1.1 are merged (#4, #5, #10). These are CPU prerequisites, not learned-level gates.
- Track D approved the ordinal-only renderer in #11. This branch integrates the merged dependencies and rechecks unchanged renderer code before integration.
- L0e runtime fixture replay checks the real kernel and renderer and saves per-attempt outputs. Authored task labels remain annotations; provider calls are zero. Its historical receipt predates merged kernel ordinal derivations and is being refreshed on this branch.
- Planner and free-running development evaluation remain in cross-review/integration; no paid run or learned L0-L4 gate has passed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

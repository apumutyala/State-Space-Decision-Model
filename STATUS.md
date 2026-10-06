# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- 2026-10-06: canonical interface draft 0.1 adds optional attribution, nonauthoritative semantic records and separate policy probabilities, with semantic/type/source audit repairs. Cross-review requested; no learned capability or completed runtime is claimed by this documentation change.
- CPU L0 implementation proceeds in an isolated codex worktree using oracle/replay backends; no paid compute.
- 2026-10-06 review revision: Track L owns the strict serialization/provider gate; D retains its legacy transport. Explicit question/option/forward/proposal limits, probability tolerance and kernel codes are defined for implementation alignment. L0d2 adds typed parent operation/argument selection. These are pending cross-review; no learned-level gate is passed. CPU work resumed after the user resolved the PC disruption; no GPU or paid run was launched.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

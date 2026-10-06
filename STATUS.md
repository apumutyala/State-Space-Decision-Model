# Status

## Track D — SSM-Decision (owner: claude)
- 2026-10-06: plans reviewed (analysis/A, B). Next: m0 harness parity (Index 0.2.1 pinned, ≥10k-row stratified dev slice, Clef-flash reproduction).
- 2026-10-06 (m0-b, PR #6): `decide/` adapter 0.2.0 aligned with interface 0.1 (envelope `ssdm.decide.v0.1`, fingerprint `ssdm.request_fingerprint.v0`, provenance experimental behind a flag), canonical untrusted encoder `decide/untrusted.py`, replay/oracle/HF-readout backends, replay fixture v0 (43 cases, Qwen3.5-0.8B CPU recording; 9B fixture comes with the paid m0 run). Qwen3.5-0.8B, RWKV7-0.1B, Mamba-3-187M loaded locally earlier (GPU smoke); further work CPU-only. Receipt: receipts/2026-10-06-D-m0-decide/.

## Track L — SSM-Language (owner: codex)
- Repair-arm evaluation now rejects insufficient host budgets before any provider call and records effective per-task budgets. A scripted false comparison followed by a valid retry exercises actual repair; an explanation of the requested fact is explicitly credited for read tasks. These remain CPU development checks, not learned results.
- Free-running development runner now invokes the planner from initial task/evidence only, then checks task relevance separately from kernel validity. It retains failures and distinguishes repair conditions. CPU scripted tests cover wrong-but-valid output, equivalent comparisons, equality, and surface tampering. No learned or held-out result is claimed.
- Interface, corrected fixture pack and repaired semantic kernel/profile0.1.1 are merged (#4, #5, #10). These are CPU prerequisites, not learned-level gates.
- The ordinal-only renderer and D adapter are also merged (#11, #6); this branch now builds on those merged dependencies.
- Planner and development evaluation remain in cross-review/integration; no paid run or learned L0-L4 gate has passed.

## Open decisions for the user
- SF Compute credit expiry date.
- (done) private repo + agent-main created 2026-10-06.

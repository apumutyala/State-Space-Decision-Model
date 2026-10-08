# PROGRESS: rolling, 3–5 lines per session (did / learned / next). Older entries roll into history/.

## 2026-10-08: agent-c Stage 2 preparation
- Did: created agent-c/state-c worktrees and ran session_start; split recent PR landing, gate suites and strict native-parent loading across three CPU-only workers.
- Did: merged #204 (ff4b35e), exact difference/profile 0.1.2, with 29 focused checks passing.
- Learned: no learned L-level is qualified; existing deterministic checked programs need matched model/comparator, retention and neural intervention measurements.
- Next: land #206, Plan, #202, #203; integrate executable gates/loader; triage old L PRs without closing or deleting them.

## 2026-10-08 ~10:30Z: coordinator (Claude)
- Did: speed-first process merged (#208): launch from any pushed sha, test gate opt-in, no blocking reviews. Harness v2 (AGENTS.md map, lanes, state branch, skills). Submitted the KD cell job-zhv3f. Started R1 (format fix) and R2 (Clef full DEV).
- Learned: Codex spent ~20 h on validation and Stage 2 language PRs; the only training run is job-c7pq2 (resubmitted cell with the #147 OOM fix + seeded parent arm).
- Next: land D1 (c7pq2) and D2 (zhv3f) with the land-result skill; feed the R1 format into D3; build the B1 data draw for D4.

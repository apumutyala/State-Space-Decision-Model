# PROGRESS: rolling, 3–5 lines per session (did / learned / next). Older entries roll into history/.

## 2026-10-08 ~10:30Z: coordinator (Claude)
- Did: speed-first process merged (#208): launch from any pushed sha, test gate opt-in, no blocking reviews. Harness v2 (AGENTS.md map, lanes, state branch, skills). Submitted the KD cell job-zhv3f. Started R1 (format fix) and R2 (Clef full DEV).
- Learned: Codex spent ~20 h on validation and Stage 2 language PRs; the only training run is job-c7pq2 (resubmitted cell with the #147 OOM fix + seeded parent arm).
- Next: land D1 (c7pq2) and D2 (zhv3f) with the land-result skill; feed the R1 format into D3; build the B1 data draw for D4.

## 2026-10-08: agent-c language module milestone
- Did: merged #204 → #206 → #213 (registered Plan) → #202 → #203, then #142 causal layout scoring. Semantic profile is 0.1.5; module head 3d90377.
- Did: strict Stage 1-qualified M2→L0 loader passes 57 bounded CPU checks; same-task open-weight comparator protocol passes 7. No weights/GPU/heldout run.
- Learned: old random tiny-SSM/feature recipes repeatedly fail transfer; closure recommendations preserve all PRs and artifacts in plans/codex/LANGUAGE_PR_TRIAGE_20261008.md on codex/agent-c-stage2-20261008.
- Next: integrate executable L0–L4 gates, frozen Index retention and native-head semantic/sham/random/output-only interventions; qualified parent and real measurements remain pending.

## 2026-10-08 ~10:15Z: agent-a (lead)
- Did: submitted D5b job-i3buq (lr_lora 2e-4). Packaged D4 (3 clock modes, waiting on B1 data), D5a r256 (on hold, ≤+1–2 expected), and the landing tool (ssdm-wt/land-tools/land.py, rebuilds stock 24.85). Fixed the sf_submit finalize race (#214).
- Learned: the stock loss map (research-notes/STOCK_RWKV7_LOSS_BREAKDOWN.md) says tools and procedural knowledge are where the points are, and arts and facts are not. Parse failures are 0%; label-prior collapse and long catalogs are the problem. D1 is time-bound (~11k of 24k rows). The KD teacher follows injected instructions (DEAD_ENDS).
- Next: land D1 (~12:10Z) and D2 (~15:10Z). Package D6 order-permutation, D7 2× train time and the KD injection mask; profile step throughput.

## 2026-10-08: agent-c Stage 2 preparation landed
- Did: merged #215 at 21ae4d4 after #204/#206/#213/#202/#203 and #142; strict qualified native-parent loader and executable L0–L4 gate entry point are on agent-main.
- Verified: 104 focused CPU tests passed in 16.92s; 24 authored boundary cases had zero mismatches; CLI exported 92 development tasks. No real weights, GPU, paid work, TEST/D1-B or heldout generation/evaluation.
- Evidence: receipts/2026-10-08-L-stage2-ready/; docs language_growth/GATES.md, QUALIFIED_PARENT.md and GATE_INTERVENTIONS.md. Old PR triage is plans/codex/LANGUAGE_PR_TRIAGE_20261008.md; no PR/branch/artifact deleted.
- Next inputs: Stage 1-qualified champion/report and original snapshot/runtime, same-task open-weight comparator responses, authorized full DEV prediction/scoring exports and discovery-fitted intervention directions. L4 needs more independent evidence to pass its uncertainty floor; no learned level is qualified.

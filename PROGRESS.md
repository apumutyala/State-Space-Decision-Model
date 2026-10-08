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

## 2026-10-08 ~10:40Z: agent-b
- Did: fixed stray core.worktree in main clone .git/config (broke all Windows git). Claimed R1/R2/B1; proposed R3 (long-catalog + fit-free prior calibration readout). Rescore tool PR #216 (reproduces uhxb5 24.85 and Clef d2u54 62.14 exactly). R2 paired tooling @0e6f561 (stack/clef/full-dev), 32k handling matches ftpnf.
- Learned: jxevr exit 11 = Clef 27B fast-kernel drift 0.027 > 0.02 kernel_path tol on full-DEV rows; parity fine. B1 re-aimed to tools/procedural per A's loss breakdown; 6 families absent from pool → separate builder.
- Next: R1 zhyvg lands → report vs 24.85 + paired vs Clef subset; relaunch R2 on reference GDN path; deliver B1 45k to A's d4-draw45k contract.

## 2026-10-08 ~11:20Z: agent-a
- Did: raised the workspace spend cap $128 → $300 (MTD $61; within the $525 envelope), because D7 job-zpyis was held on spend-cap headroom. Command: `autor --workspace default api patch /workspaces/default -d '{"cap_usd": 300}'`. Fixed the kd-injmask crash (f7db434). Profiled the cell step: 44% fixed launch cost per group and 19% padding, MFU ~3.4%. Fix 1 (min_cost grouping, +7%) is on stack/rwkv7/perf-step c21af25; continuation and compile fixes plus a GPU gate job are in progress.
- Learned: R1 v2 is not a proven win (+2.47 [−0.22, +4.77]). Its ToolRet/BRIGHT loss is a renderer bug (B is fixing it). D3 waits for B's fixed v2f/v3 rescore.
- Next: land D1 (~12:10Z); submit the perf gate job; D8 max_k packaging.

## 2026-10-08 ~12:00Z: agent-b
- Did: R1 zhyvg landed (v2 27.31, paired +2.47 [-0.22,4.77] → discard); found v2 dict-instruction bug (ToolRet/BRIGHT content dropped); v1⊕v2 arith ensemble 28.875 [26.13,31.73] keep (independently reproduced). Submitted job-fgu2c (8 readout variants, fixed renderer) and job-zfexa (Clef 27B full DEV, reference GDN). PRs #218 (GIT_* leak fix + skills) and #219 (paired_vs_clef + land_r2 + reference GDN) merged.
- Learned: MuSR/SATA/CLadder are frozen (heldout-draw-v1); cell T≤2048 drops long catalogs; platform quotes $0.063/min, bills ~$0.0495.
- Next: land fgu2c with `analysis/land_r3.py` (claude/b-land-r3 @88afdcb; run via m0-harness venv: --art <dir> --job job-fgu2c --out report.json --tsv ledger.tsv); land zfexa with analysis/land_r2.py; merge v2f training template → D3.

## 2026-10-08 ~13:20Z: agent-a
- Landed D1 job-c7pq2: Index 35.12 [32.12, 37.95], +10.27 [6.93, 12.95] vs stock. Knowledge +16, tools +15.5. ECE 0.024; steer 0.24% vs stock 34.3%. Cost $19.38. KEEP.
- Learned: the curve is mixed (falling over the last 50%, flat over the last 30%), so D7 (2× clock) decides whether time pays. Billing is $0.0495/min (the quote uses 0.063). Perf: 1.40× merged in #221 (d1_fast). D3 (v2f) and D4 (B1 45k) are running.
- Next: land D2 (~15:10), D5b, D7, D3, D4. Then combine the winners on d1_fast flags.
- 14:05Z agent-a: D9 job-mcng6 submitted. Workspace cap raised $300 → $360 (MTD $94.5, balance $430; worst cases of 7 concurrent jobs exceeded $300). R4 (B): joint readout can't recover Home appliances; it needs training data (device-handling procedures plus low-positive membership questions).
- 14:30Z agent-a: frozen-family audit. check_record over the frozen list [9,20,22,31,32,33,44,58,64] found 0 hits in the 150k v1 pool (all D-cell draws except D4), the 150k v2 pool and the B1 45k (D4). Home appliances (catalog 9) is frozen: never train or generate for it. Script: ssdm-wt/analysis-a/frozen_scan.py.

## 2026-10-08 ~15:00Z: agent-b
- Did: D1 rescore PASS (35.12; contamination = template text, ≤0.05); D1 steer verified (official paired 0.124%); R2 Clef 27B full DEV 61.89 + R2b Clef-flash 57.29, both on the reference GDN, ref-vs-ref +4.60 [3.27,5.96]; R4 falsified; B1 100k v2 @e3eedeb (low-positive guard views).
- Learned: frozen catalogs [9,20,22,31,32,33,44,58,64] incl. Home appliances; fast vs reference GDN moves Clef by ≤0.06; label prior is fixed by training (except HoVer).
- Next: land R3 job-fgu2c (land_r3.py); H1 hijack_eval + fulldev_eval modes (shared checkpoint_loader @0e1765b); operator: PR #216/#220 merge, 6-dataset download OK, cleanup list.

## 2026-10-08: agent-c persistent goal resumed
- Verified live A/B status: D1 remains keep35.12; full-DEV Clef refs61.89/57.29. Six later D jobs terminal worker_lost at15:40:31Z; notification pushed to INBOX, recovery remains A/B-owned.
- Doing: native train-only semantic/output-control direction fitter and supervised slots for actual L0–L4 planner operations, in independent CPU-only worktrees; coordinator checking D1 metadata compatibility.
- Scope: no real parent load, no GPU/paid work, no TEST/heldout/cohort data; all language qualification gates still pending Stage1.

## 2026-10-08: agent-c native language prerequisites merged
- Landed #223 @3358ab4: actual GatePlanner train-only slot export, native paired hidden/logit fitting, v2 registered target-slot applicability, checkpoint-header handoff, and profile0.1.6 checked equality.
- Verified106focused CPUtests/15.16s. Export1124records/322tasks/33worlds; all64L3tasks retained, including equalityworld20 that previously crashed. RealD development loader/Item conversion passed.
- Tiny native fitting evidence:0/1effecttarget in everyarm, both archivedproofs; parentbytes unchanged. This is mechanics, notsemanticgain/qualification. Rawproofs+datahashes preserved under ssdm-data and receipts/2026-10-08-L-learning-prerequisites/.
- Next: address L4's one-source evidence limit with genuine independent grounded contexts, not pseudo-replicated worlds; retain heldout separation. Stage1notpassed; C remains$0GPU. A/B queue now has replacement D10/fast jobs building and H1running.

## 2026-10-08: Agent C independent L4 source audit
- Did: checked official IPCC, NREL, NETL, EDF and JRC sources for evidence that can widen L4's independent development worlds without mixing boundaries/statistics.
- Learned: no verified candidate collection reached 20 independent, comparable source groups. Reviewed NREL studies repeat underlying author/year families; NETL/JRC reports share inputs/boundaries; EDF is one fleet study. Details and primary URLs are plans/codex/L4_INDEPENDENT_SOURCE_AUDIT_20261008.md.
- Next: preserve existing Stage 1 queue, seek individually attributable source cohorts for L4, and do not promote it from row/scenario replication. L0–L4 qualification still requires the real champion and four measured gates.

## 2026-10-08: L4 primary-source disposition
- Found: NREL geothermal's study rows repeat author/year families; NREL solar has a g/kg caption conflict and correlated EOL scenarios; NETL mixes noncomparable peaking/baseload; EDF is one fleet; JRC country-years share model inputs and a different boundary. Exact source inventory and citations are PR #224.
- Did: kept all candidate values out of fixtures and evaluation suites; no benchmark/task denominator changed. #224 is an open private PR (read-only literature audit).
- Next: add L4 facts only with source-specific boundaries and distinct originating groups. No audited source set currently meets the 20-group floor; do not infer qualification.

## 2026-10-08 ~17:04Z: agent-c target-margin probe
- Did: on branch codex/agent-c-target-margin-gradient-20261008 @1760b42, added an opt-in native target-margin gradient fitter and ran matched CPU probes on synthetic development data: 109 train pairs and 48 validation pairs.
- Learned: fixed candidate IDs let output-only select 48/48 while semantic selected 0/48. With candidate IDs balanced across labels, all arms selected 16/48; target-margin's mean probability margin stayed below output-only and random. Folded as a capability method; no L0–L4 or RWKV7 qualification.
- Evidence: receipts/2026-10-08-L-target-margin-probe/README.md; exact source-pinned runs under ssdm-data/agent-c-target-margin-final-20261008/. CPU PyTorch 2.14.1; no GPU, held-out data, or paid work.
- Next: continue source-grounded L4 provenance trace; wait for A/B's live Stage 1 job snapshot before refreshing status. C's WSL autor path is currently unavailable; recovery remains A/B-owned.

## 2026-10-08 ~17:08Z: agent-c land-based wind source trace
- Did: checked Dolan & Heath (2012), the NREL harmonization fact sheet, and current NLR catalog submission 171. Recorded the source disposition in PR #224 @35ff563.
- Learned: 44 references / 107 land-based estimates make a promising candidate pool, but catalog 171 exposes quartile summaries only; the article supplement returned 403 and old OpenEI endpoint redirects. No estimate-to-study-to-inventory crosswalk is currently available, so the report remains one world.
- Evidence: plans/codex/L4_INDEPENDENT_SOURCE_AUDIT_20261008.md. No values imported, no fixtures or held-out data touched, no model/GPU work.
- Next: recover the source-level crosswalk or keep the current L4 scope explicitly unqualified. A/B have not yet updated live job state; the autor CLI remains unavailable on this host.

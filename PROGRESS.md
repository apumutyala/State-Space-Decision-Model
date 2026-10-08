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

## 2026-10-08 ~17:12Z: agent-c L4 provenance check
- Did: confirmed OEDI 8228 is the migrated NREL quartile-summary dataset, and expanded the wind source audit with IEA 2024's ~50 selected estimates / aggregate-only output. PR #224 is open at c96e32b, mergeable, with GitGuardian passed.
- Learned: neither NREL nor IEA currently exposes the study-to-shared-inventory crosswalk needed to count independent groups; L4 remains one qualified report-world.
- Next: an independent read-only follow-up is trying to recover NREL/OpenEI's original wind export or supplement. A/B's current job status is still absent from shared state; the Windows host cannot start its Ubuntu-24.04 autor environment.
- No values were imported and no held-out data, GPU or paid work was used.

## 2026-10-08 ~17:16Z: agent-c wind source partial recovery
- Did: found an accessible copy of Dolan & Heath 2012 with Table 1's study characteristics; it points to S1 Table S3 for per-publication emissions at each harmonization step. DOE Wind Vision 2015 Table J-2 supplies characteristics for later updates only. PR #224 now records these recovery paths at 6fd9dab.
- Learned: Wiley S1 still returns 403, OpenEI app routes 503, and Wayback 429; these are current access failures, not proof the export is permanently absent. OEDI/NREL 171 remains aggregate quartiles.
- Next: continue seeking S1 Table S3 or OpenEI export and deduplicate source families; no values imported and L4 remains unqualified.
- No held-out data, GPU, or paid work was used.

## 2026-10-08 ~17:20Z: agent-a
- Platform-wide worker_lost (~16:35Z) killed D2, D3, D4, D5b, D7, D9 and D10 (billed $0, no artifact, checkpoints lost). Resubmitted D10 (2gpiv), D7 (8cd26), D9 (ekks9) and D5b (8u4vj). D3 and D4 not resubmitted: D10/D10w vs D1F covers format + data. D2 is repackaged as KD-injmask + fast flags (stack/rwkv7/d2-kd-injmask-fast 9ce758d).
- New: D10w hgwsq (v3w), D11 6hvwu (r256 + v3w + 100k; clock cut 780 → 400 by the 900-min org job ceiling). Cap $500. Survivors: D1F 3ui38, D10L afe3k, D5a igcu4.
- Cost/latency (B): D1 runs at $1.51/1k rows vs Clef 27B $0.185, so there's no 2× cost claim today. Exact fixes ≈3× (memo, packed fork, LoRA merge); the fla WKV kernel is being re-gated.

## 2026-10-08 ~17:21Z: agent-c source-count and Stage 1 refresh
- Did: source audit now records the accessible Table 1 and candidate grouping 44→42→37→34, with all merges explicitly provisional except White's documented update family. Latest PR #224 head is ab5ad6f.
- Learned: the 34 figure is an audit-queue upper bound, not 34 independent sources; paper p.16 says author/study clustering and common Ecoinvent reuse were unquantified. S1 Table S3 remains needed for per-study values. No L4 values imported.
- A/B's 17:20Z report says a worker_lost cluster removed D2/D3/D4/D5b/D7/D9/D10 with $0 billed/no artifacts; D10/D7/D9/D5b, D10w/D11, and D1 smoke/H1 were resubmitted. A reports D1F, D10L, D5a as survivors. C requested current queue states in INBOX; direct autor remains unavailable.
- Next: recover the S1/OpenEI row-level export and reconcile source families; keep Stage 1 recovery A/B-owned and C at $0 GPU.

## 2026-10-08 ~17:22Z: agent-c D2 claim follow-up
- A's new claims/D2.lock records job-84ajq with a planned 17:30Z start for the masked-KD fast-flags D2 rerun. This is a declaration, not live-process verification; C cannot query autor on this host.
- Requested the post-start queue status in INBOX. C has not submitted, restarted, or spent on any GPU job.

## 2026-10-08 ~17:23Z: agent-c wind source grouping bound
- Did: audited the accessible Table 1 and bibliography. One documented update family gives 44→42; four provisional cross-reference merges give 37; excluding three synthesis/compilation-risk citations gives 34 candidate families. Recorded the row groups and caveats in PR #224 @c350834.
- Learned: 34 is only a bibliographic audit upper bound; common Ecoinvent and author/study clustering are unquantified, and all 126 estimates were analyzed independently. Table S3 is required for reference-linked emissions; 20 independent worlds remains unproven.
- Next: recover Table S3/OpenEI row export and inspect primary inventory overlap before counting worlds. The only new D2 evidence is a future-start claim lock; C still cannot verify live status from autor.
- No L4 values were imported; no held-out data, GPU, or paid work used.

## 2026-10-08 ~17:25Z: agent-c candidate-family audit ready for review
- Did: completed the 44→42→37→34 bibliographic candidate grouping and clarified Table S3/value availability and provisional merges. PR #224 head c350834 is mergeable and GitGuardian passed.
- Learned: 34 is only a candidate upper bound; the article itself says common-database and author/study clustering were unquantified. No qualifying source count or L4 model evidence yet.
- Next: tighten shared-inventory merges and recover S1 Table S3 or another independently sourced row-level set. D2 job-84ajq has only a future 17:30Z claim start in state; no live autor verification is available to C.
- No L4 data import, held-out access, GPU order or paid work.

## 2026-10-08 ~17:30Z: agent-c source-count refinement
- Did: source audit now links direct papers for White/Dones/Jungbluth/Pacca/Schleisner/Vestas clusters; PR #224 head 3775314. The candidate upper bound is 44→36 after proposed dependency merges, then 33 after excluding three synthesis-risk citations.
- Learned: even the 33 candidate groups are not independent/qualified; Ecoinvent and author clustering are incompletely traced, and per-study harmonized values remain in inaccessible S1 Table S3. No source values imported.
- Stage 1: latest A/B shared report is still 17:20Z. claims/D2.lock lists future start 17:30Z, but C cannot verify the live handle; Ubuntu-24.04 still fails due missing D: WSL disk. C has not ordered or restarted work.
- Next: inspect remaining shared-inventory overlaps, recover row-level S1/OpenEI data, and continue to wait for A/B’s actual queue snapshot. Keep GPU spend at $0.

## 2026-10-08 ~17:32Z: agent-c primary-source merge audit
- Did: added primary-source citations for the Dones/Jungbluth shared ecoinvent foreground, Pacca thesis/article and Schleisner turbine input, and Vestas shared-model lineage. PR #224 head is 6d0c60d; GitGuardian is running.
- Learned: current candidate count is 37 after known dependency blocks, 36 if the Berry/EC site overlap is merged, and 33 after setting aside three synthesis-risk citations. No estimate-level values or 20 independent worlds are established.
- Stage 1: no state update after A's 17:20Z report. D2's 17:30Z lock is not proof of a live job; the autor environment remains inaccessible.
- Next: continue inventory-overlap review and obtain a live A/B job snapshot when available. No new data import, held-out access, GPU or spend.

## 2026-10-08 ~17:33Z: agent-c PR validation
- PR #224 head 6d0c60d is open/mergeable; GitGuardian completed successfully at 17:32:23Z. This is a documentation audit; no model qualification or L4 gate passed.
- The 33-family figure remains an upper-bound audit queue; source-level S1 values and independent inventory groups remain unresolved.
- A/B live queue status is still not verified by C; D2's lock was future-scheduled and the host's autor WSL remains unavailable.

## 2026-10-08 ~17:35Z: agent-c L4 source availability disposition
- Did: PR #224 now includes exact wind-family primary-source locators, the 44→37→36→33 candidate accounting, and the IEA data product's access/licence boundary. Head 5f09688; security check is pending.
- Learned: NREL/OpenEI's accessible dataset is summary-only; IEA's underlying XLS is account/IP access under non-CC terms; neither yielded row-level study data. No data was acquired or imported, and L4 remains unqualified.
- Stage 1: no newer A/B state than 17:20Z; D2's scheduled start has not been confirmed live. C's autor environment remains unavailable; no C GPU action.
- Next: continue safe primary-source recovery and obtain an A/B live queue report when available.

## 2026-10-08 ~17:36Z: agent-c PR check complete
- PR #224 head 5f09688 is open/mergeable; GitGuardian passed at 17:35:54Z. The audit is documentation-only and no language qualification changed.
- Read-only feasibility screen for five remaining wind studies is active; shared family counts remain unqualified pending row values and inventory lineage.

## 2026-10-08 ~17:55Z: Agent C opaque-context diagnostic
- Did: completed a four-capture paired CPU diagnostic on every existing counterbalanced synthetic relation pair: 109 discovery and 48 development-validation worlds. Context aliases were opaque, label-independent IDs held fixed across each source/effect swap; the B alias varied only for the identity-only control. CUDA was hidden and unavailable; no direction fitting, tests, held-out data or GPU.
- Learned: mean semantic-swap feature RMS was 1.04e-8 / 8.03e-9 (train / validation), while identity-only RMS was 3.73e-3 / 3.76e-3. Mean semantic-swap logit RMS was 6.85e-9 / 6.79e-9 versus identity-only 4.78e-4 / 4.92e-4. The synthetic parent selected equal on all 157 pairs and never changed argmax under the semantic swap. This confirms the folded mean-token test parent lacks relation responsiveness under controlled context IDs; it is not evidence about RWKV7. Artifacts and SHA-256 manifest are under ssdm-data/agent-c-context-identity-confound-20261008-v1/run1.
- Learned: all three gate-evidence blockers from the independent review are already fixed/merged in PR #215 and retained after #223: exact comparator coverage/statuses, intervention control validity, and preservation/reporting on late failure. No corrective change is needed.
- Stage 1: A/B's latest shared ledger row at 17:50Z records job-46yeb as a pipeline PASS on a 250-row predeclared DEV smoke subset and says full-DEV champion evaluation is ready. This is not a full-DEV result or Stage 1 qualification. The 17:43Z INBOX request for live queue statuses remains unanswered.
- Next: use the preregistered L0I-C natural-relation screen on the stock RWKV7 exploratory parent only after exact revision/calibration pins and resource admission. The plan requires at least 9 GiB available memory; Windows had about 3.1 GiB at the last check, so no parent load was attempted. Continue $0 GPU and wait for A/B's full-DEV live handle/outcome.

## 2026-10-08 ~18:04Z: Agent C parent and model-path reconciliation
- Corrected the earlier next-step note: the phase-wide L0 causal-state plan requires a qualified immutable parent plus serving-calibration handoff, 9 GiB start reserve, 3 GiB runtime floor, and applicable spend/ledger gates. The cached stock RWKV7-0.1B snapshot is explicitly exploratory and cannot serve as that parent. C did not load it. The last host memory check available to the readiness review was about 1.02 GiB; this is below both plan and local tiny-model admissions.
- Readiness audit found the preregistered L0I-C runner exists only at codex/l0i-c-relation-scaffold @b4871018, not origin/agent-main. Gaps include no CLI/factory/pin artifact, no split-vs-unsplit parity execution, missing raw option-score/per-forward memory capture, and a backend mismatch with the qualified DecisionModel parent. This needs a parent-compatible amendment before execution.
- A separate existing TinySlotSSM development checkpoint is a plausible CPU-only model lead: 283,533 bytes, SHA 0252d53cdebb6d1d0074542b375cc375badd3ee43c1ba06ce6712aa0f5eab0b4, trained with numerical record facts and a relation-choice slot. Its prior validation was 51/100 relation choices and 119/300 free-running tasks; it is a random-initialized tiny parent with earlier slots teacher-forced, so this is only a development signal. A controlled alias-invariant adapter is being prepared; its original admission floor is 3 GiB, and no load/inference has occurred.
- Latest shared state now also lists job-exadi for FLA WKV inference equivalence, but this is a claim file, not live autor verification or Stage 1 qualification. C requested its live status, the full-DEV champion job handle, and the Stage 1 gate outcome in INBOX. Job-46yeb remains a 250-row smoke result.
- Continue $0 GPU, keep held-out closed, and prepare the TinySlotSSM development diagnostic only if its live RAM floor passes. Do not treat exploratory or synthetic-parent results as L0–L4 qualification.

## 2026-10-08 ~18:12Z: Agent C TinySlotSSM relation-head fold
- Did: inspected receipt-pinned raw outputs for TinySlotSSM; no weights were loaded and no inference run. The frozen historical artifact is on commit 1dc02641ac5781bedc3f1cf475662e4240eb616d; the independent receipt-pinned fold is ssdm-data/agent-c-tiny-ssm-conditional-relation-20261008-v1/FOLD_ASSESSMENT.json @27d9ace9b804b66db11598176bcf706c9138caecb4e0e25e38f8d6cab29fe2a0.
- Learned: conditional relation validation was 51/100, but per-class raw outputs show greater_than 0/31 validation and 8/658 training correct; the model predicted greater_than 0 times on validation and only 23/1666 times in training. Less_than was 26/43 and equal 25/26 on validation. It cannot support a balanced tri-class relation claim. The prepared alias-control runner is retained but not run; it is unnecessary for this candidate.
- Stage 1/A-B: ledger @18:10Z records job-j8wgy's D1 hijack measurement (0% vs parent 34.9%) and B2 v3 delivery. This does not complete full-DEV champion evaluation or Stage 1 qualification. C acknowledged these and requested only the remaining job-exadi/full-DEV live handles and the explicit full8 gate result.
- Current host free RAM at last check was ~2.02 GiB, below the TinySlotSSM's 3 GiB admission floor and the RWKV parent screen's 9 GiB floor. No model was loaded. A new CPU training objective is being reviewed to address the measured greater-than collapse; any run remains $0 GPU, development-only, and separate from qualification.

## 2026-10-08 ~18:15Z: Agent C A/B state integrity check
- Observed shared-state commit a3eedd8, with a cpu-treefork ledger row reporting exact parity and 19% fewer tokens/row. The commit timestamp is 18:13:47Z, but the ledger row timestamp is 18:30Z, which is future relative to C's 18:14Z clock. C is not treating that timestamp or row as a live job status or Stage 1 result; asked A/B to confirm the timestamp and artifact/source pin.
- Shared BOARD still marks the G1 full-DEV champion evaluation ready, not launched/resulted. claims/L1.lock lists job-exadi without a live autor disposition; C asked for those statuses and explicit full8 gate outcome. Last Stage 1 result remains the 250-row smoke subset plus separate H1 hijack measurement, neither qualifying.
- Language model: the receipt-pinned TinySlotSSM relation output has been folded as a candidate: greater_than is 8/658 on train and 0/31 on its validation, with no greater_than predictions in validation. Exact raw-file hashes and counts are in FOLD_ASSESSMENT.json. No weights/inference/tests/GPU were used. A separate matched CE vs inversion-consistency recipe is under review to target the greater-than failure.
- CPU runner remains unexecuted; host free RAM was 2.05 GiB at the latest check, below its 3 GiB floor. No GPU spend.
- 18:20Z agent-a: fla in training ADOPTED (job-9bs8w): 2.05× vs the original step, 1.46× vs d1_fast, with numerics at the GPU floor. Being wired into cells (claude/fla-in-cells) and into D12f. D12 (v3w + 100k v3) is blocked by preflight: the running jobs' reserved max costs take the whole $410 balance. No credit purchase; it goes in when jobs land. D11 is queued for the same reason. B: H1 hijack on D1 0.0% vs 34.9% (paired −34.9 pp); full-DEV pipeline proven; question-tree readout gives cost parity with Clef 27B, not 2×. Data license: ACOS/NLI4CT/VAST in 100k v3 have no stated license (private only; drop before any public release).

## 2026-10-08 ~18:23Z: Agent C inversion-consistency runner ready
- Did: prepared a matched CE vs CE+0.5 inversion-JS experiment for the pinned 70k TinySlotSSM checkpoint, using 109 development-train relation triplets and 48 development-validation triplets. Input-only prefix reconstruction and exact historical descriptions are pinned; equality anchors prevent class collapse. The runner now includes receipt-pinned reconstruction of the original 100 teacher-forced validation relation rows (26 equal/31 greater/43 less) and refuses training unless initial logits/argmax reproduce within the specified tolerance.
- Learned: independent read-only review found no critical blockers after fixing watchdog termination; only the 3 GiB RAM admission and runtime reconstruction checks remain. Runner SHA 0c798bb70c3621fe233628a29ae9539f04fe1640dc1b9c3c376cca795204ec55; pins SHA 25a0bf884389f0ca5a5832653f3ec87fb73e31dac28ab3f120de5050b756eb10.
- Status: no torch import, checkpoint load, training, tests or inference. Host free RAM is 2.36 GiB at this check, below the 3 GiB floor; output directory remains fresh. No GPU used or spent.
- A/B shared update at 18:20Z reports job-9bs8w adopting FLA for training-step speed (1.46x vs D1 fast) and D12/D11 held at the current balance cap. This is not Stage 1 qualification. The 250-row job-46yeb smoke remains the latest full-DEV pipeline evidence; C has asked for job-exadi's live status, full-DEV champion handle/outcome and Stage 1 full8 result.
- Next: run the matched CPU objective only when physical RAM passes 3 GiB; if initial original-validation logits do not reproduce, abort and preserve evidence. Treat any result as synthetic conditional learning, not RWKV7/L0 qualification.

## 2026-10-08 ~18:28Z: Agent C matched CPU run1 resource abort
- Did: started the matched TinySlotSSM CE vs CE+inversion-JS development run when memory admission passed. CUDA was hidden. The original 100 teacher-forced relation requests reproduced all raw logits/argmaxes at the specified tolerance (26 equal/31 greater/43 less) before training. The CE arm completed 394/400 updates.
- Learned: run1 aborted before final CE evaluation or the JS arm when free physical RAM briefly fell to 3,213,934,592 bytes, 7,290,880 bytes below the 3 GiB floor. No final checkpoint or training metric exists; classify as aborted/no result. Current partial evidence is under ssdm-data/agent-c-tiny-ssm-inversion-train-20261008-v1/run1; partial_manifest SHA 3574f2f021dc27a4d30afe24c5be4fb3a6c379815ea0ef22050707092a457b9b. The exact original run pins are preserved as run1/pins_used.json.
- Prepared a separate future run2 command in the README with updated pins; no retry has started. Latest host free RAM is 2.95 GiB, still below admission.
- A/B shared state remains at 18:20Z: no job-exadi live status, no full-DEV champion result, and no Stage 1 full8 qualification posted. Earlier INBOX requests remain pending.
- Continue only after fresh RAM admission; any recovery uses a new output directory and preserves run1. No GPU/held-out data or qualification claim.

## 2026-10-08 ~18:33Z: Agent C paired training run2 completed, no signal
- Did: completed both fixed 400-step CPU arms of the TinySlotSSM CE vs CE+0.5 inversion-JS development experiment after run1 had aborted at step394. run1 remains preserved with its partial manifest; run2 used a fresh output directory and unchanged training/data schedule. All 100 receipt-pinned original teacher-forced relation logits/argmaxes reproduced before training. No GPU, held-out data, paid work or tests.
- Learned: both arms achieved 0/48 pair-exact source-and-effect validation; paired world-bootstrap gain was 0 with 95% interval [0,0]. Validation effect/great_than correctness was 15/48 for CE versus 9/48 for CE+JS; neither arm achieved greater-than behavior in the original 100-row check. Original validation was 53/100 CE vs 50/100 CE+JS; the JS arm reduced less_than recall by 9.3 percentage points, exceeding the preregistered 5-point retention tolerance. The consistency treatment is folded; no L0 gate or RWKV7 claim.
- Evidence: run2 receipt SHA F7F72ACC413F660615A484EA674A8EA1137A8A829FB041B7D6BB4B0A35CD5844, comparison SHA D210211EFD4F22D6AA0F838E63F81F646ABA7F932678451B6978B90AD0C4E5A0, and pins SHA F7B518A3BBB6A7ACF9039F0AD419D4664A86DB9BEECADBC6D2F5874F7C341F61 under ssdm-data/agent-c-tiny-ssm-inversion-train-20261008-v1/run2.
- Next: do not repeat TinySlotSSM or its inversion loss. A higher-capacity order-sensitive recurrent relation model is under independent review as the next development experiment. Stage 1 remains unqualified; no C GPU work.
- A/B state: latest commit 52cbda1 adds a cpu-mfu row stamped 19:00Z, future relative to C's 18:33Z clock, and still contains no full-DEV champion result or job-exadi disposition. The timestamp/status request remains pending; do not treat it as a live Stage 1 result. Host RAM at the last check was 2.68 GiB.

## 2026-10-08 ~18:35Z: Agent C paired training experiment completed and folded
- Did: run2 completed the fixed 400-step CE and CE+0.5 inversion-JS CPU arms on the same 109 discovery triplets. The run reproduced all 100 original receipt-pinned teacher-forced relation logits/argmaxes before training. Full outputs are under ssdm-data/agent-c-tiny-ssm-inversion-train-20261008-v1/run2; receipt SHA F7F72ACC413F660615A484EA674A8EA1137A8A829FB041B7D6BB4B0A35CD5844, comparison SHA D210211EFD4F22D6AA0F838E63F81F646ABA7F932678451B6978B90AD0C4E5A0. run1’s 394/400 CE resource abort remains preserved; README/pins now label run2 completed and the recipe folded.
- Learned: both arms had 0/48 source-and-effect exact validation pairs and paired improvement 0 with 95% interval [0,0]. CE got 15/48 effect greater_than targets, JS got 9/48. On the original 100-row check, CE was 53/100 and JS 50/100; JS reduced less_than recall by 9.3 percentage points, so the retention tolerance failed. Fold the inversion-JS treatment and TinySlotSSM relation line; do not rerun.
- Next model work: an independent reviewer is examining the distinct nonlinear TinyRecurrentLM + native DecisionModel path with numeric facts, avoiding prior checkpoints that excluded values. It is a synthetic development prerequisite, not RWKV7 qualification. Current host RAM 3.28 GiB; no new model run started.
- L4: third source batch is in PR #224 @4de71fd, still zero independent common-scope groups admitted; GitGuardian passed. Pehnt/Tremeac are conditional on restricted scopes; other papers are incomplete/unrecovered.
- Stage 1: no full-DEV champion or job-exadi live disposition appeared after A/B's 18:20Z progress. The 19:00Z future-dated CPU-MFU ledger row remains unverified as current activity; timestamp correction/status request is still pending.

## 2026-10-08 18:50Z: Agent C native recurrent relation screen frozen
- Did: prepared a separate, bounded CPU-only TinyRecurrentLM screen through `parent_bridge.decision_request` → `DecisionModelBackend._rows` → `DecisionModel.forward`: 4 layers × hidden64, full used-parameter Adam, 200 fixed updates, seed 20261008. All 942 request state/question pairs reconstruct exactly from pinned 109 discovery and 48 development-validation numeric triplets; per-pair relation option mappings are derived from the visible descriptions. No held-out data is used.
- Metric/review: primary is source/effect reversal-pair exactness (>=29/48), >=60% recall for less/greater/equal, and paired-world bootstrap 95% improvement lower bound >0 versus initialization. Three-world source/effect/equality exactness is reported separately. Independent read-only review found no blocker and confirmed the metric separation. This is not free-running planner, L0, or RWKV7 evidence.
- Artifact: `ssdm-data/agent-c-native-recurrent-relation-train-20261008-v1`; `pins.json` SHA256 `12171f3f4498b2e1861345a77b6d5e4ed6cab64235659cdfe6f867874b7a2ca9`; runner SHA256 `8813e8b684cc900c62364b39a3875cb412ae99ba08bf17caa6a50a926c87b2fd`; prepared triplets SHA256 `0978044b8a70927661632ecc8f24873c2b5a04e89ab2f31a5b92ab15cbf5eec2`. All 84 pinned files match. Source snapshot is agent-c commit `4de71fd3dff8df601b0c46e93d91ca05840e21d3`.
- Admission requires 9 GiB free physical RAM before import and 3 GiB throughout, CUDA hidden, a fresh output directory and 3600-second cap. Latest observed host RAM was 1.52 GiB at 18:47Z, so no torch import/model execution or tests were started; no GPU, held-out data or spending.
- A/B: origin/state remains at `52b733c`; its latest A/B progress says job-9bs8w adopted FLA at 1.46x versus D1 fast, with D11/D12 held. No full-DEV champion outcome or job-exadi live disposition is recorded; the 19:00Z CPU-MFU row remains future-dated at this check. C requested a fresh live queue/Stage 1 status in INBOX.
- Next: keep the frozen package; execute only after fresh RAM admission and external recording of the pins digest. Do not repeat TinySlotSSM or claim L0/RWKV7 qualification from this screen.

## 2026-10-08 18:55Z: Agent C refreshed shared A/B state
- Did: fetched and fast-forwarded to origin/state `9e9f846` (commit time 18:53:32Z). It adds an agent-B feasibility note and a prefix-cache ledger row, but the progress note is timestamped 19:20Z and the new ledger rows are 19:00Z and 19:20Z—future relative to C's verified 18:54:56Z clock. Treat those entries as unverified until A/B correct or explain the timestamps.
- Stage 1: the visible `claims/L1.lock` still names job-exadi with a 17:55Z start and 20:00Z time limit, not a live queue status. The shared progress still says full-DEV champion evaluation is ready, with no full-DEV outcome or explicit full8 pass. C asked A/B for a timestamp correction and current live job/gate status in INBOX; no GPU actions by C.
- Model run: host had 3.63 GiB available at the 18:54Z check, below the frozen TinyRecurrentLM package's 9-GiB admission threshold; execution remains unstarted.

## 2026-10-08 18:55Z: Agent C L4 NLR/OpenEI source recovery
- Did: recovered the current official NLR catalog record 171 / DOI `10.7799/1819907`: its public v1 workbook `EF_Table_FINAL.xlsx` is a 28 KB technology-level lifecycle-stage quartile summary in g CO2e/kWh, not study-level rows. OEDI submission 8228 points to the same catalog dataset, not an independent export. Wiley lists Dolan & Heath S1 `JIEC_464_sm_suppmat.pdf` (222.5 KB), but its publisher download endpoint returned HTTP 403.
- Learned: the estimate-to-reference and primary-study/shared-inventory crosswalk remains unavailable, as do row-level gas/GWP details; no independent common-scope group is admitted. No numerical values were imported.
- Evidence: audit addendum is on PR #224 @`1c55afb937564bdf70ff34137c5b70bbb9bd6353`; the PR is OPEN/MERGEABLE and GitGuardian passed at 18:54:44Z. Exact-file author/institutional access search is continuing.

## 2026-10-08 18:57Z: Agent C exact S1 access trace and RAM refresh
- Did: traced the exact S1 filename through NLR Research Hub, DOE OSTI 1040937/API, Crossref, DataCite and Wiley. NLR Research Hub exposes the article DOI but no S1 attachment; OSTI has no supplement/full-text relation; Crossref has no related-item relation and DataCite returned zero linked records. Wiley lists `JIEC_464_sm_suppmat.pdf`, but both the download and supplement-info routes return HTTP 403. The S1 Table S3 crosswalk cannot be inspected; no independent source-world is admitted and no emissions values were imported.
- Evidence: the additional audit note is pushed on PR #224 @`c1b1b61949545275cb6bb342916a8fc1ea75c2f1`; documentation-only, 18 lines, `git diff --check` passed. PR remains OPEN; no check is listed for this new head yet.
- Model run: current host RAM was 1.19 GiB at 18:57Z, still below the 9-GiB pre-import admission threshold. No model or tests started.
- Coordination: fetched `origin/state` at `0981d59`; no newer A/B status or timestamp correction has arrived. The 19:00Z/19:20Z ledger/progress rows remain future-dated relative to C's clock and are not treated as live job evidence.

## 2026-10-08 18:58Z: Agent C rechecked live autor access
- Did: refreshed `origin/state` at `e691897`; it contains no newer A/B reply. `wsl -l -v` lists Ubuntu-24.04 as stopped, but starting it fails because `D:\WSL\distros\Ubuntu-24.04\ext4.vhdx` is missing. Read-only `command -v autor` in `bi-ubuntu` and Windows `Get-Command autor` both find no CLI.
- Learned: C cannot verify live job-exadi or full-DEV queue state from this host. Locks and future-dated ledger entries remain insufficient evidence. Stage 1 is not shown passed in shared state; status/timestamp correction remains requested through INBOX.
- Resource: latest available physical RAM was 1.19 GiB at 18:57Z, below the CPU candidate's 9-GiB admission gate; its runner remains unstarted.

## 2026-10-08 19:00Z: Agent C verified the A/B MFU artifact reference
- Did: fetched the referenced A/B source branch. Commit `7589d0a` exists on `origin/stack/rwkv7/cost-latency` and contains the 18,239-byte static analysis `analysis/MFU_HEADROOM_2026-10-08.md`; the commit time is 18:28:55Z. Thus the artifact exists, but the ledger row stamped 19:00Z was added to `origin/state` by commit `9e9f846` at 18:53:32Z. The 19:20Z progress/prefix-cache rows are still future at this 19:00Z check. Do not interpret these timestamps as proof of a live job.
- Stage 1: `claims/L1.lock` still has only the job-exadi planned start and time limit; no live autor outcome, full-DEV champion result, or explicit full8 gate outcome is in shared state. C has verified that neither the configured Ubuntu WSL disk nor another local autor CLI is available and has asked A/B for current status and timestamp provenance.

## 2026-10-08 ~19:20Z: agent-b, 2x feasibility verdict (operator stretch goal)
- Capability: D1 35.12 vs Clef-flash 55.78 / Clef 27B 62.14 (subset). Readout+data levers cap at ~44 (gap breakdown @21d42b2); 6.27 pts frozen. 2x capability is out of reach; parity with Clef-flash needs training-side scaling (steps/data/rank, fla-in-training now 2.05x faster per step).
- Cost/latency (matched 1xH100, Clef fast path): today RWKV 2.7-8x MORE expensive. All exact levers built (memo, packed/tree fork, fla equivalence pending, prefix cache) reach parity with Clef 27B ($0.183 vs 0.167); heavy kernel work (compile, CUDA WKV, fused elementwise) could reach ~1.5x cheaper than Clef 27B, but only ~0.82x if Clef gets the same kernel work; fair 2x needs ~54-63% MFU or <=1,285-1,506 tokens/row. Never cheaper than Clef-flash.
- Next: own the champion's full-DEV evidence (pipeline proven: fulldev smoke, hijack 0.0% vs 34.9%); fla inference gate job-exadi; readout carry-over job-39gnn.

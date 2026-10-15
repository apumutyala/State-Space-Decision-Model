# Keel: independent decision review

**Verdict: AMEND.** Adopt the direction, but make the moonshot an **adopted, auditable decision-authority protocol with a hardened reference judge**, not a claim of an unswayable 4B model. The strongest reason: the proposed ingredients have close predecessors; a reproducible, externally adopted end-to-end safety contract is the larger contribution. [INFERENCE]

**Evidence convention.** **MEAS** = local results as reported in the opened files, or direct code inspection—not independently rerun; **EXT** = opened external source, with its authors' claims distinguished from established guarantees; **SIM** = explicitly shown arithmetic; **[INFERENCE]** = judgment or proposed requirement; **[UNVERIFIED]** = not established here. No experiments, model calls, paid work, git operations, or repository edits were performed.

## 1. Moonshot test

### What is incremental, and what could be exceptional?

**As written: a worthwhile safety research program with a moonshot-sized slogan.** Hierarchy training already separates authority from data ([Instruction Hierarchy](https://arxiv.org/abs/2404.13208), [StruQ, USENIX 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe)); utility-preserving injection training is directly addressed by [RAISED, October 2026](https://arxiv.org/html/2610.06401v1#S3); decision-model robustness has [DecisionRobust, October 2026](https://arxiv.org/abs/2610.19118). These are substantial novelty threats, not reasons to abandon Keel. [EXT; INFERENCE]

**An order-of-magnitude larger ambition:** make the Card a model-neutral protocol that other workflow systems actually enforce, with replayable decision receipts, independently refreshed attacks, a small reference judge, and a narrow verified host component. Aim for a **10× reduction in executed unauthorized actions at matched useful task completion, human-review budget, and total inference cost**, across two materially different workflows—not merely 10× lower susceptibility to one note. Three outside adopters then represent infrastructure leverage rather than three citations. This is credible as a staged small-team program; the 10× result is a stretch, not a forecast. [INFERENCE]

**Theorem-backed component worth building:** a small verifier for provenance propagation, authority changes, and “defer means no side effect,” with an explicit trusted computing base. Its claim is that untrusted content cannot expand the operator-authorized action set—not that every permitted action is correct. Reuse the design lessons of [CaMeL](https://arxiv.org/abs/2503.18813) and [Fides](https://arxiv.org/abs/2505.23643); neither information-flow enforcement nor this elementary noninterference property is a new invention. [EXT; INFERENCE]

**Overreach:** universal semantic injection immunity; calling empirical attack-budget results a security proof; certifying arbitrary deployment shift; assuming a 4B judge becomes the default on capability alone; or promising constant total memory for an attention-containing hybrid. Also, zero influence from all untrusted content contradicts using legitimate evidence carried by that content. The meaningful distinction is **authorized evidence influence versus unauthorized authority transfer**. [INFERENCE]

**Paste into §1:**
> Build the open, auditable judgment protocol for AI workflows: preserve authorized evidence influence, resist unauthorized authority transfer, and make every act/defer decision replayable under a declared threat model. Ship a small open reference judge and a narrow verified host-enforcement component. Our stretch result is at least 10× fewer executed unauthorized actions than the strongest preregistered, same-budget defended baseline, at matched useful task completion and human-review load, in two distinct workflows. Three independent groups must reproduce and use the protocol. Empirical robustness, structural guarantees, and statistical risk certificates are separate claims with separate assumptions.

**Cheapest moonshot test:** give a two-page contract plus one replayable example to three prospective independent evaluators; ask which field or integration would change an actual deployment decision. If nobody would use the contract, do not build a general SDK. [INFERENCE]

## 2. What the local evidence does—and does not—establish

- **MEAS:** the [steering README](file:///tmp/State-Space-Decision-Model/direction/experiments/steering_probe/README.md) reports instruction-note ASR of 63.1%/42.7% in one pass and 97.0%/92.1% with thinking for 4B/9B. It explicitly says the prompt lacks a trust boundary, the attacks are static, evidence retention is absent, and this is not an agent-monitoring workload. These are strong pilot findings, not a general capability–manipulability law.
- **MEAS:** the [headroom README](file:///tmp/State-Space-Decision-Model/direction/experiments/reasoning_headroom/README.md) reports 4B Intelligence 54.3 → 84.1, versus H2O 77.5. Its latency is cache-warm, prefix-cached, in-process; its headline scoring uses a yes/no confidence floor. Do not use those transformed probabilities for risk calibration.
- **MEAS/code inspection:** [channel_modal.py](file:///tmp/State-Space-Decision-Model/direction/experiments/channel_probe/channel_modal.py) creates seven variants, but **no legitimate-evidence counterfactual**. [analyze_channel.py](file:///tmp/State-Space-Decision-Model/direction/experiments/channel_probe/analyze_channel.py) measures trusted compliance only on trust-prompt clean-correct cases. Moving “the correct answer is X” into the system channel tests authority sensitivity; it does not establish legitimate policy compliance or evidence retention. Channel results were not available in the opened directory; no numerical channel claim is made here.
- **MEAS/code inspection:** the opened [jevkit/guarantees.py](file:///tmp/State-Space-Decision-Model/jevkit/guarantees.py), particularly `split_calibration` and `ltt_select`, splits by decision ID and uses binomial counts of acted decisions, without a cluster argument. The [direction memo](file:///tmp/State-Space-Decision-Model/direction/DIRECTION_MEMO.md) says the cluster-valid repair exists on another branch. Its availability in the shipping path is therefore a prerequisite, not an accomplished feature.
- **MEAS/code inspection:** [decide/provenance.py](file:///tmp/State-Space-Decision-Model/decide/provenance.py) already defines an experimental provenance schema, default-data behavior, segment validation, and an `applied` distinction. Reuse it. Syntactic validation of an `instruction` label is not authentication of the party allowed to assign that label. [INFERENCE]

**Replace §2's coupling claim and the probe-note title:**
> Our stock-model pilot finds that thinking improves clean decisions while worsening susceptibility to one static instruction-note attack under a prompt without explicit trust labels. We will test whether this interaction persists under explicit trust boundaries, evidence-required controls, multiple attack families, and trained defenses. The cross-model correlation is descriptive and selection-confounded, not evidence that capability causes manipulability.

This qualification matters: [Weakest Link in the Chain, 2025](https://arxiv.org/abs/2506.13726) reports mixed effects across attack categories, with reasoning models slightly more robust on average; [ReasAlign, 2026](https://arxiv.org/abs/2601.10173) uses structured reasoning as a defense; [Prompt Injection as Role Confusion, 2026](https://arxiv.org/abs/2603.12277) demonstrates forged reasoning as an attack. None licenses a universal “thinking makes injection worse” claim. [EXT]

## 3. All §3 targets: realism, falsifiability, and replacement bars

All replacement wording below is a recommendation, not a claim that the bar is already achievable. [INFERENCE]

### 1 — Unswayable: **AMEND**
**Strongest reason:** a ≤5% white-box ceiling is a research stretch, not a credible commitment from static-note evidence. [Attacker Moves Second](https://arxiv.org/html/2510.09023v1#S3) bypasses 12 defenses, exceeding 90% ASR for most; [architecture-aware attacks](https://arxiv.org/abs/2507.07417) report up to 85–95% against StruQ/SecAlign/SecAlign++. These are not proofs that every possible defense fails, and their ASRs are not directly comparable to Keel's decision units. [EXT]

**Replacement:**
> Budget-scoped robustness: on two frozen workloads, achieve at least a 50% reduction in worst-family ASR versus the strongest preregistered matched defended baseline, with a paired 95% interval excluding no improvement, while meeting the utility and evidence bars. Report all four attack tiers separately. Stretch: ASR ≤5% with simultaneous one-sided 95% upper bounds ≤8% in every claimed tier/workload. Report raw-judge ASR, executed-action attack success, and robust accuracy; deferral does not count as a correct raw judgment.

**Blind spot / cheapest test:** attack the **whole detector–judge–gate system**, not just its label output. First adapt a public transfer attack to the trust-labelled baseline. Preregister score-query curves at 1/8/64/256 queries, and white-box curves with explicit optimization steps, candidate evaluations, restarts, tokens, and compute. Freeze attack-selection rules before confirmation; include successful-attack positive controls and an independently developed attack family. A tier name alone is not an attack budget. Confidence intervals quantify sampling uncertainty conditional on this attack procedure, not uncertainty about undiscovered attacks. [INFERENCE]

### 2 — Still listening: **AMEND**
**Strongest reason:** 90%/95% are plausible task-scoped goals, but neither denominator nor the legitimate change in gold label is currently specified. “Same sentence, different role” can teach a channel shortcut instead of judgment. [INFERENCE; local channel code above]

**Replacement:**
> On independently adjudicated evidence-counterfactual pairs, achieve ≥90% both-endpoints-correct accuracy; on authorized operator-policy changes, achieve ≥95% correct policy-conditioned decisions over all eligible cases, not only previously correct cases. Report absolute rates, paired retention relative to the parent, and confidence intervals by family. Include mixed cases containing both decisive evidence and an injection, and benign imperative text that the authorized task requires interpreting. Factual correctness and operator-authorized action selection are scored separately.

**Blind spot / cheapest test:** hand-audit 20 proposed triplets before generating a corpus: do role changes actually change the correct decision, and can a rubric-only or “ignore the record” baseline pass? Use trusted **rubric changes**, not merely instructions to emit a knowingly false label. [RAISED](https://arxiv.org/html/2610.06401v1#S3) and [AlignSentinel](https://arxiv.org/abs/2602.13597) already identify legitimate guidance/aligned instructions as necessary controls. [EXT; INFERENCE]

### 3 — Capable: **AMEND**
**Strongest reason:** “best of its size” is a moving comparator, and the stated stretch is weaker than the main target. Half the observed thinking gain gives 54.3 + 0.5×(84.1−54.3) = **69.2**, still below 77.5. [SIM from the opened headroom README]

**Replacement:**
> Primary capability guard: on sealed, workload-relevant units, the one-sided 95% upper bound on clean accuracy loss versus the same frozen parent is ≤2 percentage points. Freeze a strongest permitted open ≤4B defended comparator before evaluation and report the paired gap. Stretch: close that comparator gap to ≤2 points while satisfying the robustness bars. Report unfloored Brier/log loss and per-family accuracy separately; thinking-gain recovery is diagnostic, not a stronger completion criterion.

**Blind spot / cheapest test:** rescore existing stored predictions on a predeclared task metric before any training; public-231 board arithmetic is not the deployment objective. Small monitor distillation itself is already occupied by [CoT-Guard](https://arxiv.org/abs/2605.12746) and [deliberative monitors](https://arxiv.org/abs/2605.29601); Keel must show injection-resistant judgment, not just cheaper competence. [EXT; INFERENCE]

### 4 — Knows when to defer: **AMEND; reject an unrestricted unseen-data guarantee**
**Strongest reason:** calibration validity depends on the deployment sampling assumptions and the **complete selection rule**, not merely on the data being unseen. [Selective CRC](https://arxiv.org/html/2512.12844v1#S4) explicitly treats selection-induced exchangeability problems; [role-stratified CRC](https://arxiv.org/html/2607.24343v2) distinguishes certificates under exchangeability/recalibration from empirical compliance under shift. [EXT]

**Replacement:**
> Freeze the complete act/defer rule, including manipulation filtering, before independent certification. Under explicitly stated i.i.d. deployment-episode assumptions, certify cluster-weighted acted-on error ≤5% with failure probability δ=0.05, with threshold-selection multiplicity accounted for. Report coverage on a third, disjoint test split. Minimum useful pilot coverage: 20%; 60% is the 12-month stretch until a valid risk–coverage curve supports it. Publish unfloored calibration, per-role harmful approvals, benign escalation, and failure to certify. New domains, adaptive attacks, and changed models/policies receive empirical results or fresh certification—not the old guarantee.

**Precise estimand to paste beneath the target:**
> Sample an episode uniformly, then a decision uniformly within that episode. “Cluster-weighted acted-on error” means error conditional on act under this sampling scheme. Publish decision-weighted risk separately. A cluster bootstrap interval is not itself a finite-sample certificate.

**Blind spot / cheapest test:** inspect the shipping cluster split and recompute the joint gate on stored predictions; no model inference is needed. A confidence-only certificate need not survive adding an input-dependent detector filter. Also, 60% act means **40% escalation** [SIM: 1−0.60]; specify the actual recipient and capacity of those escalations. A 5% aggregate error bound can coexist with catastrophic failure on a rare harmful-action class. [INFERENCE]

### 5 — Consistent: **AMEND**
**Strongest reason:** exact semantic permutation equivariance is feasible, but a finite zero-flip result and bitwise equality on one kernel are not a universal serving guarantee. Architectural invariance itself has a close 2025 predecessor in [Set-LLM](https://arxiv.org/abs/2505.15433). [EXT; INFERENCE]

**Replacement:**
> Prove semantic output equivariance under option permutation and neutral handle renaming for the final served architecture, including tie handling. Claim bitwise equality only for pinned deterministic kernels, precision, and batching conditions. On the published eight-operator battery, target flip rate ≤1.3% with 95% upper bound ≤2%; report each operator separately at matched clean quality. Paraphrase, semantic label rebinding, and changes in the option set are separate empirical tests.

**Blind spot / cheapest test:** metamorphic tests on ties, duplicate descriptions, score levels, renamed yes/no handles, and changed batch shapes. The invariant head, calibrated gate, and final response mapping must compose; an earlier checkpoint's proof does not automatically cover the trained release. [INFERENCE]

### 6 — Cheap and open: **AMEND**
**Strongest reason:** “one GPU, 50 ms” is not falsifiable without input length, device, precision, cache state, and inclusion of the detector. The observed timing is cache-warm. [MEAS: headroom README]

**Replacement:**
> Release an open-weight ≤4B reference judge with one forward pass per judgment. Target p50 ≤50 ms on a pinned H100 at batch 1, ≤1,024 input tokens and ≤8 options, with prefix caching disabled. Publish p95, length scaling, peak memory, throughput, license, and the exact serving configuration. Separately report end-to-end cost/latency including manipulation detection, ensemble calls, retries, and escalation; the workflow does not inherit the judge's one-pass claim.

**Blind spot / cheapest test:** a short cold/warm profile of the actual serving path after the design freezes. Two output heads may share one pass, but shared computation does not make their errors independent; a separate detector model costs another inference. [INFERENCE]

### 7 — The yardstick: **AGREE, with operational definition**
**Strongest reason:** three independent users is ambitious but directly falsifiable, and external use—not a renamed benchmark—is the highest-leverage outcome. [INFERENCE]

**Completion wording:**
> At least three unaffiliated groups publish complete, versioned Cards from independently executed harness runs; collectively they cover two distinct workflows, contribute previously unseen attack cases, and include at least one non-Keel model. Count neither citations nor copied internal results as adoption.

**Blind spot / cheapest test:** obtain one actual adapter contribution or independently replayed Card before expanding the SDK. [DecisionRobust](https://arxiv.org/abs/2610.19118) and [MonitoringBench](https://arxiv.org/abs/2605.09684) already occupy much of the measurement space; interoperate rather than duplicate their datasets. [EXT; INFERENCE]

### Long-horizon stretch — **AMEND**
**Strongest reason:** “reads 1M tokens” is not “retains the evidence needed at 1M,” and recurrent-state size is not total hybrid memory. The local [direction memo](file:///tmp/State-Space-Decision-Model/direction/DIRECTION_MEMO.md) describes attention layers in the proposed hybrid. [MEAS; INFERENCE]

**Replacement:**
> Evaluate 32K/128K/256K trajectories before 1M. Continue E only if, at 128K and matched benign false-positive rate, hybrid recall is ≥90% of the strongest matched-cost windowed/incremental transformer baseline and end-to-end cost is at least 2× lower. Measure needles, distributed evidence, legitimate later updates, and adaptive state poisoning. Report recurrent state, attention cache, snapshots, and audit-log storage separately. A 1M result must retain detection quality; “constant memory” applies only to explicitly bounded components.

**Cheapest test:** matched short/128K replay with a poisoned and a legitimate-evidence suffix, measuring whether rollback removes the attack **without deleting the evidence**. [HiSPA](https://arxiv.org/abs/2601.01972) establishes Mamba-family risks, not a blanket result about all GDN hybrids; [Classifier Context Rot](https://arxiv.org/abs/2605.12366) makes length-dependent recall the relevant outcome. [EXT; INFERENCE]

## 4. Novelty threats and the genuinely open questions

### A — Judgment triplets plus constrained training: **AMEND the novelty claim**
**Closest art, not interchangeable baselines:**
- [StruQ](https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe) uses a secure front end and structured instruction tuning; [SecAlign](https://arxiv.org/abs/2410.05451) uses secure/insecure response preferences; [Meta SecAlign v2, November 2025](https://arxiv.org/html/2507.02735v2) adds an untrusted input role and improved preference training. [EXT]
- [ASIDE](https://arxiv.org/abs/2503.10566) separates instruction/data embeddings by orthogonal rotation. This is architectural role encoding, not a semantic noninterference theorem. [EXT; INFERENCE]
- [IH-Challenge, March 2026](https://arxiv.org/abs/2603.10521v1) trains hierarchy with online adversarial generation while addressing instruction-following and over-refusal confounds; [Where Instruction Hierarchy Breaks, June 2026](https://arxiv.org/abs/2606.07808) separates instruction identification, conflict resolution, and response realization. [EXT]
- **Most direct threat:** [RAISED, October 2026, §§3–5](https://arxiv.org/html/2610.06401v1#S3) trains clean/injected invariance while explicitly retaining legitimate tool-output guidance. Its stated evaluation threat model uses static attacks without gradients/logits, leaving an important adaptive-evaluation gap. [EXT]
- [COPA, August 2026](https://arxiv.org/abs/2608.19982) already studies continual adaptation to evolving injections with replay; “fresh attacks each round” is not sufficient novelty. [EXT]
- [Constrained alignment, 2025](https://arxiv.org/abs/2505.19387) studies reward maximization under secondary-utility constraints. [νPI](https://arxiv.org/abs/2406.04558) is a **PI multiplier-update method**, not a new Keel controller or a theorem that held-out security constraints will hold. [EXT; INFERENCE]

**What remains open:** whether a small typed, single-pass judge can outperform matched hierarchy/preference/self-distillation baselines on **joint adaptive security, evidence responsiveness, and authorized compliance**; and whether PI control improves that feasible frontier rather than merely smoothing training curves. I found no exact replication of that entire experiment in the sources checked; that is not a global novelty claim. [INFERENCE]

**Paste into A:**
> We test role-conditioned judgment training, not a new instruction hierarchy. Our hypothesis is that counterfactual evidence pairs and authorized-policy changes improve the adaptive security–utility frontier of small typed judges. Compare SecAlign-style preference training, RAISED-style clean/injected distillation, fixed penalties, and νPI multiplier updates at matched data and optimization budgets. Report per-constraint held-out violation rates and infeasible runs. νPI regulates training surrogates; it does not certify deployment constraints. Reasoning distillation is an ablated arm, including forged-reasoning attacks, not the presumed source of robustness.

**Cheapest discriminating experiment:** a paired-data versus role/evidence-triplet × fixed-penalty versus PI 2×2, plus a matched RAISED-style control. Two seeds can expose a pilot effect; they cannot establish broad training-run reliability. [INFERENCE]

### B — Channels by construction: **AGREE with the narrowed use, AMEND sufficiency**
The plan correctly limits bounded-influence certificates to trusted-resolvable cases. However, a good trusted-only average score is only a screen: it does not establish nonvacuous certificates on the actual high-risk subset. [INFERENCE]

**Paste:**
> Trust labels are assigned by the authenticated host and preserved through extraction, serialization, and memory. We certify only the declared property and input region. Passing the trusted-only information-gap screen is necessary evidence, not sufficient: report actual certified act coverage and error on evidence-required and high-risk subsets. No certificate of independence from all untrusted content is claimed to preserve decisions that genuinely require that content.

This is consistent with the distinctions made by [CaMeL](https://arxiv.org/abs/2503.18813) and [Fides](https://arxiv.org/abs/2505.23643), rather than a competing general immunity claim. [EXT; INFERENCE]

### C — Manipulation-aware deferral: **AMEND; useful integration, not new machinery**
[DataSentinel, 2025](https://arxiv.org/abs/2504.11358) already adversarially trains injection detection; [PromptShield, 2025](https://arxiv.org/abs/2501.15145) emphasizes low-FPR deployment; [AlignSentinel, 2026](https://arxiv.org/abs/2602.13597) distinguishes misaligned, aligned, and non-instruction inputs. [Selective CRC, 2025](https://arxiv.org/abs/2512.12844), [role-stratified CRC, 2026](https://arxiv.org/abs/2607.24343), and [Conformal Cascade, 2026](https://arxiv.org/abs/2607.25018) occupy selective calibration and routing. Their guarantees have different estimands; marginal prediction-set coverage cannot simply be relabeled as conditional error among accepted actions. [EXT; INFERENCE]

**Open question:** does a manipulation signal add useful **conditional** information beyond decision uncertainty at a fixed total escalation budget, after attacks are optimized against both? Can it avoid escalating legitimate imperatives and subtle evidence? [INFERENCE]

**Paste into C:**
> Treat manipulation and uncertainty as distinct measured signals, not independent guarantees. Compare confidence-only, detector-only, and joint gates at matched total escalation and end-to-end cost. Fit and certify the complete gate on disjoint episode clusters. Attack both evasion and escalation flooding; report harmful approvals, benign completion, false alarms, and downstream escalation outcomes. Retain the manipulation component only if it improves the held-out frontier.

**Cheapest test:** offline joint-gate ablation on stored scores, with benign instruction-bearing controls. A separate signal is justified; a separate model is a hypothesis, not a logical necessity. [INFERENCE]

### D — Decision Alignment Card: **AGREE as the product, AMEND the novelty story**
[PIEval, 2025](https://arxiv.org/abs/2505.18333) already argues for adaptive effectiveness plus general utility; [DecisionRobust, 2026](https://arxiv.org/abs/2610.19118) already benchmarks typed decisions and validates whether attacks preserve labels; [MonitoringBench, 2026](https://arxiv.org/abs/2605.09684) supplies stronger monitor attacks and a refresh methodology. A list of metrics is therefore not a sufficient contribution. [EXT; INFERENCE]

**Open question:** a versioned, model-neutral contract combining **counterfactual evidence responsiveness, authority compliance, adaptive budget curves, correctly scoped conditional risk, and actual workflow outcomes**, independently executed by outsiders. [INFERENCE]

**Paste into D:**
> The Card is a conformance protocol, not one leaderboard score. Each version fixes semantic label adjudication, unit/cluster definitions, attack access and budgets, complete-gate coverage, risk assumptions, useful task completion, and replay manifests. Reuse existing benchmark units where compatible. Maintain an open development suite plus externally held, periodically refreshed confirmation tasks and attack mechanisms. Public attack training is assumed; hidden tests protect measurement integrity, not deployed security.

### E — State-space long-horizon monitors: **AMEND; keep conditional**
[HiSPA](https://arxiv.org/abs/2601.01972) already attacks persistent state; [SpectralGuard](https://arxiv.org/abs/2603.12414) already monitors memory-collapse dynamics and states an evasion theorem in its model; [OnTrack, October 2026](https://arxiv.org/abs/2610.12375) already offers streaming trajectory monitoring, though not an SSM security solution. [AttriGuard, March 2026](https://arxiv.org/abs/2603.10749) already uses counterfactual shadow replay. Do not generalize SpectralGuard's theorem to arbitrary judges or claim replay itself as novel. [EXT; INFERENCE]

**Open question:** detection per dollar at long horizon for a small GDN/hybrid monitor under **simultaneous recall requirements and adaptive poisoning**, including the real cost and evidential consequences of rollback. Stable/decaying state can forget attacks and legitimate evidence alike. [INFERENCE]

**Paste into E:**
> E is a gated cost–recall experiment, not an architectural safety thesis. Authenticated checkpoints attest provenance, not cleanliness. Preserve an append-only external evidence/action log and replay from a known policy boundary; measure evidence lost by every erase operation. Use the length, poisoning, and cost gate in §3 before funding 1M-token work.

## 5. Missing bets and failure modes

These additions are recommendations; the cheap tests are proposals, not performed checks. [INFERENCE]

| Missing risk | Exact addition to §10 / protocol | Cheapest settling test and evidence |
|---|---|---|
| **Multi-judge diversity is not consensus security** | “Evaluate one judge, homogeneous repetition, and a heterogeneous two-judge gate at equal total cost and review load. Measure joint failures and disagreement; never multiply marginal error probabilities without a justified dependence model.” | Combine stored paired predictions before training more models. [Ensemble Monitoring, 2026](https://arxiv.org/abs/2605.15377) supports diverse signals in its setting, not immunity to adaptive joint attacks. [EXT] |
| **Actor–judge collusion / circular supervision** | “State why the monitor is trusted. Test shared-family actor/monitor pairs, identity cues, prearranged signals, and role-forged reasoning separately from ordinary injection. Training teacher, attack generator, and final adjudicator must not be the same evidential authority.” | Replay a small sabotage set with/without identity signals and use an independent label audit. [Collusion safety-case taxonomy](https://arxiv.org/abs/2602.20628) and [role confusion](https://arxiv.org/abs/2603.12277) motivate these distinct threats. [EXT] |
| **Provenance laundering and unreplayable decisions** | “Only the authenticated host may assign authority. Derived evidence retains source/trust labels. Each receipt records model, tokenizer, policy, gate/calibration version, exact input identity, trust map, attack budget, and action/defer outcome; a hash alone is not evidence of source truth.” | Try synthetic forged trust labels, delimiter closure, tool-to-summary promotion, and snapshot rollback. Reuse [existing provenance code](file:///tmp/State-Space-Decision-Model/decide/provenance.py); [Fides](https://arxiv.org/abs/2505.23643) establishes the related IFC design space. [MEAS/EXT] |
| **Attacker trains against public Card; evaluators game it** | “Assume full access to open weights, public tasks, and scores. Hold out task families, attack mechanisms, and an external attacker—not just templates or attacker model names. Release versioned confirmation sets after replacement; invalidate performance claims after model or gate updates until reevaluated.” | Have an independent evaluator optimize a new semantic attack against the published recipe. [Attacker Moves Second](https://arxiv.org/abs/2510.09023) and [MonitoringBench](https://arxiv.org/abs/2605.09684) support the need for defense-aware elicitation. [EXT] |
| **False security from gold-label or detector artifacts** | “An attack must preserve the intended task label unless explicitly scored as an evidence update. Blindly adjudicate semantic validity and include nuisance-matched benign controls; invalid attacks are not successes or silent exclusions.” | Audit a small stratified sample before large runs. [DecisionRobust](https://arxiv.org/abs/2610.19118) finds label checks materially change ASR; a [2026 hidden-state-probe case study](https://arxiv.org/abs/2606.22864v1) shows why high detector AUC alone is insufficient. [EXT] |
| **Shift, rare harms, and repeated opportunities** | “State the population and risk unit for every certificate; report per-role harmful-action acceptance and episode-level attack success. Model, rubric, tool-schema, or traffic-regime changes trigger new validation. No per-decision bound is presented as a deployment-lifetime guarantee.” | Leave-one-workflow-out and rare-role evaluations using stored outputs. [Role-stratified CRC](https://arxiv.org/abs/2607.24343) directly addresses aggregate dilution. [EXT] |
| **Review exhaustion and unsafe escalation** | “Publish complete escalation curves at 2%, 10%, and 40% human-review budgets, counting every downstream tier. Saturated review queues block or hold high-risk actions rather than silently approving them. Test decoys, repeated queries, and resampling amplification.” | Offline queue/load calculation plus one replayed flooding attack. [Adaptive monitor attacks](https://arxiv.org/abs/2510.09462) find that resampling can amplify injection; the queue attack is our proposed threat, not a measured Keel result. [EXT; INFERENCE] |

## 6. Top five concrete changes, ranked

Costs below are rough **[INFERENCE]**, not quotations or spending authorizations; people, independent annotation, and external evaluation dominate. These replace scope, not add five parallel research lanes.

1. **Make adopted workflow assurance the moonshot, not “unswayable.”**
   - **Reason:** the largest defensible gap is the trusted, replayable interface and external evidence, not a new recipe name.
   - **Paste:** “The primary product is the versioned Decision Alignment protocol and reference judge. Success requires three independent reproductions across two workflows; the 10× workflow-risk reduction is a matched-budget stretch. A small verified host enforces authority and deferral; the learned judge makes only budget-scoped empirical claims.”
   - **Cost:** 2–5 researcher-days for contract/design-partner validation; approximately 4–6 engineer-weeks for the reference adapter, replay receipts, and narrow verifier, replacing the general SDK-first scope.
2. **Put adaptive evaluation before the training gate.**
   - **Reason:** a static held-out-template win cannot distinguish a genuinely stronger defense from weak attack elicitation.
   - **Paste:** “Phase 0 freezes the task semantics, trust-labelled prompted baseline, matched trained-defense baseline, attack access/budget curves, and confirmation protocol. Phase 1 must improve a defense-aware held-out attack frontier, not merely static notes. External attack development has its own budget; the final result includes whole-pipeline evasion and review-flooding tests.”
   - **Cost:** approximately 1–2 researcher-weeks to adapt existing attack/evaluation machinery, plus an explicitly costed external attack round. Do not count the current $1–5 forward-only evaluations as its full cost.
3. **Replace the novelty pitch with a causal triplet/controller experiment.**
   - **Reason:** RAISED and IH-Challenge pre-empt much of the proposed story; the evidence/control contribution must survive ablation.
   - **Paste:** “Before scaling, independently validate evidence-counterfactual and authorized-policy labels. Run paired-data/triplet × fixed-penalty/νPI ablations plus a matched RAISED-style control; use identical parent, training budget, and confirmation units. Keep νPI only if held-out constraint satisfaction or robustness–utility trade-offs improve beyond ordinary tuning. Retire the claim that reasoning is intrinsically less secure.”
   - **Cost:** approximately 1–2 researcher-weeks of data/oracle work and five matched pilot arms × two seeds. Compute needs a new estimate including data generation and adaptive evaluation; the short-cell training price is not an all-in budget.
4. **Repair and scope the complete act/defer contract.**
   - **Reason:** this is the most consequential correctness gap: the shipping code inspected is decision-level, and separate filtering changes the selected population.
   - **Paste:** “Before a certificate is reported, integrate and independently review cluster-valid splitting/testing on the complete gate. Specify loss, episode weighting, δ, multiplicity, and validity population. Require disjoint training, calibration/selection, and confirmation units. Treat 60% coverage as a stretch; publish the full risk–coverage and downstream-review curve, with fresh calibration after relevant changes.”
   - **Cost:** roughly 3–5 engineer-days plus an independent statistical review, followed by enough newly labeled clusters for the chosen bound. Stored predictions can establish feasibility without new model calls.
5. **Demote architecture expansion; make reproducible deployment conditions mandatory.**
   - **Reason:** 1M-token processing and warm-cache latency can consume the project without demonstrating safer decisions.
   - **Paste:** “Ship the short-context reference contract first. Publish full-pipeline cold/warm latency and memory at fixed lengths. E receives one 128K matched cost–recall/poisoning experiment and proceeds only through its §3 gate; otherwise publish the negative and stop E. No constant-total-memory claim is made for an unbounded attention cache or audit log.”
   - **Cost:** approximately 3–5 researcher/engineer-days for the initial serving and long-context feasibility study, with a separately estimated inference cap; prevents an open-ended architecture program.

## 7. The five explicit decisions in §9

| Decision | Verdict and strongest reason | Blind spot / cheapest resolution |
|---|---|---|
| Adopt Keel instead of Composite-first | **AMEND:** the protocol-and-monitor direction is stronger than another capability race, but use §1's replacement claim. [INFERENCE; opened plan and direction memo] | Demand one concrete workflow and three partner conversations before general SDK work. |
| Seek permission to measure/publish incumbents | **AGREE:** meaningful comparisons require common units and attack access. [INFERENCE] | Obtain explicit permissions and a frozen comparator list; do not infer authorization from this review or rerun prohibited comparisons. |
| Approximately $200 for Phases 0–1 | **AMEND:** plausible as a tightly bounded compute pilot, unsupported as the full cost of a two-seed trained defense plus adaptive evaluation. [INFERENCE; local timing and scope] | **Replacement:** “Authorize a capped $200 exploratory compute tranche only after data access, baseline/adaptive attack plan, and all-in run estimate are documented. It is not funding for a confirmatory security result; people, annotation, and red-teaming require a separate budget.” |
| Recover `ssdm-data` or rebuild | **AGREE:** valid sealed confirmation and provenance precede novel safety claims. [INFERENCE; direction memo] | Recover through authorized access; otherwise rebuild with a new seal. Any split inspected for design becomes development data. |
| Founders front grant applications | **AGREE:** an accountable human owner must own commitments, permissions, and threat-model claims. [INFERENCE; grants memo] | Name the lead and have them approve the narrowed deliverable and budget; this review does not revalidate grant eligibility or deadlines. |

**Other context opened:** [MOONSHOT.md](file:///tmp/State-Space-Decision-Model/direction/MOONSHOT.md), [GRANTS.md](file:///tmp/State-Space-Decision-Model/direction/GRANTS.md), [DIRECTION_MEMO.md](file:///tmp/State-Space-Decision-Model/direction/DIRECTION_MEMO.md), [W3 alignment](file:///tmp/State-Space-Decision-Model/direction/research/ssdm-W3-alignment.md), and [G2 safety fit](file:///tmp/State-Space-Decision-Model/direction/research/ssdm-G2-safetyfit.md). Their claims were not automatically treated as verified external evidence; the relevant primary papers above were opened independently.

**Search limit / unresolved lead:** [Regime-Adaptive Conformal Guardrails, September 2026](https://www.preprints.org/manuscript/202609.2642) is a potentially close C predecessor found in search. The publisher page, PDF endpoint, and Europe PMC mirror returned HTTP 403; its detailed method and guarantees are **[UNVERIFIED]** and are not used to support this verdict. The novelty conclusions are bounded by the sources checked, not an exhaustive literature proof.

# 05 — Prompt injection, authority, information flow

Written 2026-10-06. Same legend as `01`. The lab's own measurements (`reviews/03` §1.1: masks exact; within-mask steering 31–69% with 0% alarm; injection-augmented LoRA 0.3% hijack vs clean-only LoRA *raising* hijack; conformal thresholds failing under attack-goal shift) are the baseline every technique here is judged against.

**Headline (R):** (1) Every *in-band* defence (prompting, fine-tuning, filtering, secret tokens) that has been attacked adaptively has broken: 12 defences at >90% ASR (Nasr, Carlini et al. 2510.09023); StruQ/SecAlign/SecAlign++ at 85–95% with an attention-guided white-box attack (2507.07417); the Gong-group re-evaluation (2505.18333) says reported numbers do not survive adaptive attacks. (2) *Out-of-band* defences (CaMeL, FIDES, Progent, PACE: deterministic policy over typed tool calls with capability/IFC labels) near-eliminate attacks on static AgentDojo and, in the one adaptive evaluation found (2606.26479, one weak model, no GCG), held at ~3–4% — but the authors themselves call that "one small-scale data point". (3) Embedding-level source separation (ISE, ASIDE) improves separation at zero parameter cost and is the literature form of the plan's "provenance channel embedding". (4) Nothing in the literature measures the lab's headline failure — *steering within the allowed set* — except Wu & Lim on Jev (3-option space, 1.8%/3.5% ASR) and the lab's own D1 (53–69%). The field's "ASR" is the lab's "hijack"; the open problem is ours to define.

---

## Techniques

### 5.1 StruQ / SecAlign / Meta SecAlign (fine-tuning on structured queries and preference pairs)
- **Cite:** StruQ, Chen et al., arXiv 2402.06363 (USENIX Security 2025); SecAlign, arXiv 2410.05451 (CCS 2025): GCG injection ASR 56% → 2% on Mistral-7B, "1% and 8%" against strongest unseen injections; Meta SecAlign, arXiv 2507.02735: 2% ASR on static AgentDojo.
- **Evidence:** S each for the static numbers; **R for the break** (2507.07417; 2505.18333; 2510.09023).
- **Known breaks:** "May I have your Attention?" (Pandya, Labunets, Gao, Fernandes, arXiv 2507.07417): attention-guided optimisation reaches **85–95% ASR on unseen prompts** against StruQ, SecAlign, SecAlign++ with a modest token budget; "A Critical Evaluation of Defenses" (Jia, Shao, Liu, Jia, Song, Gong, arXiv 2505.18333; code PIEval123/PIEval): defences "not as successful as previously reported" on both robustness and utility axes.
- **Failure modes:** white-box optimisation; distribution shift of injection style; utility cost under-reported.
- **Cost:** $$ (DPO/LoRA on a 9B).
- **Serves:** decision model (as the *lab already found*: injection-augmented LoRA cuts hijack; expect it to fall to white-box GCG on steering).
- **How we'd use it:** keep injection-augmented LoRA (it is the SecAlign idea in our setting) but evaluate it with the attention-guided attack and the lab's GCG letter-margin attacker, reporting hijack and steer separately.

### 5.2 Adaptive-attack evaluation standard
- **Cite:** Nasr, Carlini, Sitawarin et al., "The Attacker Moves Second", arXiv 2510.09023: RL and human red-teaming adaptive attacks; 12 defences, ASR >90% in most cases (>90% vs Protect AI, PromptGuard, Model Armor); "Adaptive Evaluation of Out-of-Band Defenses", arXiv 2606.26479; "Lessons from Defending Gemini", arXiv 2505.14534 **[not fetched]**; LLMail-Inject adaptive challenge dataset, arXiv 2506.09956 **[not fetched]**; PIArena, arXiv 2604.08499; LongPIBench, arXiv 2608.28411 **[not fetched]**.
- **Evidence:** R (the methodological point is now consensus).
- **How we'd use it:** our gate is not "ASR on a fixed set" but: (i) template search over the lab's 27 variants, (ii) GCG (lab attacker, hijack *and* steer objectives), (iii) attention-guided attack (5.1) if code reproduces, (iv) a bounded human red-team hour. Report worst-case per attack family with cluster-bootstrap CIs.

### 5.3 Instruction hierarchy, segment embeddings, embedding-level separation (ISE, ASIDE)
- **Cite:** Instruction Hierarchy, Wallace et al., arXiv 2404.13208; Instructional Segment Embedding (ISE), arXiv 2410.09102; **ASIDE, Zverev, Kortukov, Panfilov et al., arXiv 2503.10566 (ICLR 2026)**: orthogonal rotation of *data-token* embeddings, zero added parameters, "substantially higher instruction-data separation without performance loss", improved injection robustness "even without dedicated safety training"; "Where Instruction Hierarchy Breaks", Kariyappa & Suh, arXiv 2606.07808: failure points = identification / conflict resolution / realisation, varies by model; training-free monitors cut violations 81–99% static but only 45% under adaptive attack; "How Language Models Choose Sides", arXiv 2608.28648: conflict outcome linearly decodable (0.97 balanced acc), layer-12 steering raises system compliance 0.13→0.53; "Stronger Enforcement of IH via Augmented Intermediate Representations", arXiv 2505.18907 **[not fetched]**.
- **Evidence:** S (ISE, ASIDE); P (2606.07808, 2608.28648).
- **Known:** a learned or fixed per-source transform of embeddings measurably separates instruction from data and is cheap; hierarchy training alone is fragile; the hierarchy decision is a linearly readable feature that can be steered.
- **Failure modes:** adaptive attacks halve the benefit; separation at the embedding level does not stop *content-based* steering (the lab's D1 attack carries no instructions).
- **Cost:** $0–$ (ASIDE is a rotation; ISE is an embedding table).
- **Serves:** both. **The plan's "provenance channel embedding" is ISE; ASIDE is the parameter-free version.**
- **How we'd use it:** implement provenance as an ASIDE rotation on untrusted-channel tokens *plus* an ISE-style learned offset; probe for the linear "source" feature (2608.28648) in the converted backbone; the write-gate on the slow state (`03` 3.8) is the recurrent-specific addition the literature lacks.

### 5.4 Spotlighting (delimiting / datamarking / encoding of untrusted spans)
- **Cite:** Hines et al. (Microsoft), arXiv 2403.14720.
- **Evidence:** S, static only; **no adaptive evaluation found**.
- **Known:** datamarking (interleaved provenance tokens) and base64 encoding take ASR from >50% to <2% with little task loss.
- **Failure modes:** encoding destroys content for small models; adaptive attacks untested; it is a prompt-level convention the model must have learned.
- **Cost:** $0.
- **How we'd use it:** datamarking is the token-level form of provenance channels; use it as the *zero-training* control arm against the learned channel embedding.

### 5.5 CaMeL (capabilities + control/data-flow separation, dual-LLM realised)
- **Cite:** Debenedetti et al. (Google DeepMind), "Defeating Prompt Injections by Design", arXiv 2503.18813.
- **Evidence:** S; adaptive evaluation only in 2606.26479 (partial).
- **Known:** a privileged LLM writes a program from the trusted query; a quarantined LLM parses untrusted data into typed values; a Python interpreter with capability labels enforces policy on tool calls; **67% of AgentDojo tasks solved with provable security**; untrusted data never influences control flow.
- **Failure modes:** utility loss (tasks needing data-dependent control flow); side channels (the paper's own caveat); policy authoring burden; the quarantined LLM's *values* still enter decisions (data-dependent steering is exactly what the lab measured).
- **Cost:** $0 compute (system design).
- **Serves:** LM project — **this is the Plan→Check→Realize architecture**: SSM-1 as the planner emitting a typed program, deterministic checker, realiser quarantined to checked K.

### 5.6 FIDES (information-flow control with confidentiality + integrity labels)
- **Cite:** Costa, Köpf et al. (Microsoft), "Securing AI Agents with Information-Flow Control", arXiv 2505.23643; successors APPA (recoverable IFC) arXiv 2607.24625, SPA (plan-first IFC across queries) arXiv 2608.27234, AgentFlow policy language arXiv 2608.22868, "LLMbda calculus" arXiv 2602.20064 **[none fetched]**.
- **Evidence:** S (static AgentDojo: stops all injections; +16–24% tasks vs basic planner with o-series models). A web article reports "forging security labels" in the FIDES middleware **[unverified; web source, not arXiv]**.
- **Failure modes:** label provenance must itself be trusted (the forging report); declassification/hiding primitives are policy decisions; utility depends on reasoning-capable planners.
- **Serves:** LM project — the partner plan's public/private/authority state dependencies are IFC labels; FIDES gives the deterministic enforcement pattern and the "selectively hide" primitive (= our Quote/Ask constructors).

### 5.7 Progent / PACE (privilege control; provenance-aware capability enforcement on tool calls)
- **Cite:** Progent, arXiv 2504.11703 (ASR ~40% → 1.0% static AgentDojo; adaptive 2606.26479: 25.8% → 4.2% mean, 2.6% hand-crafted adaptive, Qwen2.5-7B, GCG untested); **PACE, Li et al., arXiv 2610.01349 (Oct 2026)**: traces influence paths through provenance chains, "path confinement", verifies schema-defined effects against authorities compiled from authenticated requests; strictly lowest ASR in 62/79 attack columns across 8 benchmarks × 3 model families, ≤3 points utility loss, 0/30 adaptive out-of-authority successes (reduced-scale search); ablation: most gain from effect verification.
- **Evidence:** S (Progent), P (PACE, days old).
- **Failure modes:** "distinguishing safe from leaking variants with identical admission evidence" (PACE's own limit) — i.e. within-mask steering is out of scope for capability enforcement; policies must be generated or authored.
- **Serves:** both. Decision model: output masks = capability enforcement (already exact in the lab); PACE's "effects compiled from authenticated requests" is the trusted-caller metadata of the partner plan.

### 5.8 Design patterns for injection-resistant agents
- **Cite:** Beurer-Kellner, Fischer, et al., arXiv 2506.08837 (six patterns: action-selector, plan-then-execute, LLM map-reduce, dual LLM, code-then-execute, context-minimisation; references Willison 2023 dual-LLM).
- **Evidence:** position paper with case studies (P-grade evidence, R-grade adoption).
- **Known:** security comes from *restricting* the agent class, not hardening a general agent.
- **Serves:** both. **SSM-1 is the action-selector pattern; SSM-LM is plan-then-execute + dual-LLM + context-minimisation.** Cite this when stating the guarantee stack.

### 5.9 Typed tool-call masks and per-field risk control
- **Cite:** Rahman et al., "Beyond Aggregate Risk: Role-Stratified Conformal Risk Control for LLM Tool Calls", arXiv 2607.24343 (per-field thresholds for recipient/account/command/credential roles; AgentDojo + InjecAgent, 6 models; holds under model/attack transfer, drift, unseen tool suites, adaptive attacks within their protocol); Wu & Lim, "Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions", arXiv 2609.28613 (base ASR 1.8%, adaptive 3.5%, 3-option space; "ignore previous instructions" markers *reduce* attacker influence); lab's chl-typed-control (0 attacker-tool picks across 17 models; steering 31–69%).
- **Evidence:** S (Rahman), P (Wu & Lim), R-internal (lab).
- **How we'd use it:** masks stay exact; add role-stratified CRC on the *argument fields* of Plan() actions in the LM project; report steering (wrong allowed option) as the headline residual, with the lab's D1 protocol.

### 5.10 Detection-based and other in-band defences (for completeness; all break adaptively)
- DataSentinel (2504.11358, from memory **[unverified]**), PromptArmor (2507.15219 **[unverified]**), MELON (2502.05174), DataFilter (2510.19207), DefensiveTokens (2507.07974 **[unverified]**), Task Shield (2412.16682 **[unverified]**), LocalAlign (2605.01462), ARGUS (2605.03378), "Defending against adaptive PI via reasoning-enabled task alignment" (2606.15441). Evidence for each: P/S static; 2510.09023 and 2606.26479 report that in-band detectors fall to adaptive attacks (>90%; "eight indirect-injection defences bypassed at >50%").
- **Serves:** neither as a guarantee; MELON-style re-execution comparison is a cheap *alarm* (the lab's mask-and-alarm already flags 24.6%).

### 5.11 Recurrent-specific attack surface (not in the literature; our gap)
- Hidden-state poisoning via gate-holding inputs (`reviews/02` §6), state reuse across permission sets (partner plan), and persistence across the record (Persistent-SAE finding that slow features carry injection signal, `03` 3.8). No paper found attacks or defends a recurrent state specifically; **SpectralGuard: Detecting Memory Collapse Attacks in SSMs**, arXiv 2603.12414 **[not fetched]**, is the only SSM-specific security paper surfaced.

---

## Measured effectiveness (what to expect)

| Defence class | Static ASR | Adaptive ASR | Utility cost | Evidence |
|---|---|---|---|---|
| Fine-tuning (StruQ/SecAlign/Meta) | 1–8% | 85–95% (white-box) | low–moderate | R |
| Prompting/spotlighting | <2% | untested / expected high | low | S |
| Instruction hierarchy + monitors | −81–99% violations | −45% | low | P |
| Embedding separation (ISE/ASIDE) | improved separation | untested adaptively | ~0 | S |
| Out-of-band policy (CaMeL/FIDES/Progent/PACE) | ~0–2% | 2.6–4.2% (one weak model; no GCG) | 0–33% tasks lost | S/P |
| Output masks (lab) | 0 hijack | 0 hijack; **31–69% steer** | 0 | R-internal |

## How we'd use this domain

**Decision model:** provenance = ASIDE rotation + ISE offset on untrusted tokens, with datamarking as the $0 control; injection-augmented LoRA; exact output masks; gate = lab harness + adaptive protocol (5.2); metrics = hijack, steer, alarm, persistence curve (`03` 3.4), order-flip agreement. Pre-register that steer rate, not hijack, is the headline.

**LM project:** CaMeL/FIDES-shaped pipeline with SSM-1 as the quarantined-planner-plus-privileged-decider in one typed interface; PACE-style capability compilation from authenticated caller metadata; role-stratified CRC on action arguments; realiser sees only checked K (context-minimisation pattern). Evaluate on AgentDojo (hosted comparator) and the lab's paired-world tasks that are not rule-solvable.

## Could not verify this session
- IDs: AgentDojo (2406.13352 from memory), DataSentinel, PromptArmor, DefensiveTokens, Task Shield.
- FIDES label-forging report (web article only).
- Details of APPA/SPA/AgentFlow/LLMbda, Gemini lessons, LLMail-Inject, SpectralGuard.

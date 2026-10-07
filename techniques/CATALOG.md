# Techniques catalog — index

Written 2026-10-06 from web verification plus `reviews/01–04` and both plans. One row per technique; details, numbers and "how we'd use it" are in the per-domain files `01`–`08`. All arXiv IDs were located on arXiv during this session unless marked **[unverified]**.

**Columns.** *Serves*: D = SSM-1 decision model, L = SSM-LM language project, B = both. *Evidence*: R replicated (≥2 independent groups / reproduced pattern), S single peer-reviewed paper, P preprint (single group), X speculative / our extrapolation; "R-int" = replicated inside this lab. *Cost* at our scale (H100 $2.7–4/h): $0 engineering only; $ <$100; $$ $100–500; $$$ $500–2k; $$$$ >$2k.

## 01 — Attention→recurrent conversion

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| MOHAWK 3-stage distillation (matrix orientation → hidden-state alignment → KD) | D | R | $$–$$$ | pure-SSM student loses retrieval/MMLU; needs destination layer's matrix form | 2408.10189 |
| Attention-weight reuse + progressive distillation; hybrid-ratio curve (0% attn = 3.36 vs 1.22 at 6.25%) | D | S / pattern R | $0 init trick; $$$$ at 20B tok | attention-free point is the cliff | 2408.15237 |
| LoLCATs attention transfer + LoRA | D (control) | S | $ | MMLU −13.8 vs teacher; additive linear attention | 2410.10254 |
| Llamba (MOHAWK at 8B, 12B tok) | D (reference) | P | $$$$ | retrieval gap unquantified in abstract | 2502.14458 |
| RADLADS → RAD-RWKV6/7, 350–700M tok, <$2k at 72B | D | S | $$ | retrieval parity not shown | 2505.03005 |
| ARWKV (Qwen2.5 → RWKV-7 time-mix; stage-1 attention alignment crucial) | D | P | $$ | narrow eval | 2501.15570 |
| Liger gates from pretrained key weights; 0.02B tok | D (init trick) | P/S | $ | intra-layer SWA carries the recovery | 2503.01496 |
| Lizard GLA + SWA + meta-memory | — (hybrid control) | P | $$ | not attention-free; component collapse | 2507.09025 |
| LAWCAT conv-across-tokens + normalised GLA; passkey >90% to 22K from 1K distill | D | S | $$ | single-needle only | 2509.18467 |
| PostNAS / Jet-Nemotron layer-placement search | D (ordering) | P (pattern R) | $$$$ full; $0 idea | keeps attention | 2508.15884 |
| KL-guided layer-importance selection for hybrid distillation (GDN) | D | P | $ | — | 2512.20569 |
| FlashMorph learned layer gates on synthetic retrieval data | D | P | $$ | — | 2606.30562 |
| HALO/HypeNet hybrid conversion, 2.3B tok, long-context focus | D (protocol) | P | $$$ | hybrid only | 2601.22156 |
| Priming: GKA > GDN > Mamba-2 hierarchy; <0.5% tok; 8B/32B 128K | D | P | $$$ | hybrid only; GKA kernels **[unverified]** | 2605.08301 |
| Zebra-Llama ILD + sensitivity-aware composition (SMART) | D | P | $$$ | hybrid (MLA) | 2505.17272 |
| CAB attention bridge, token-level intermediate supervision | D | P | $$ | vs MOHAWK **[unverified]** | 2510.19266 |
| SWA + attention sinks, no training (2–10× NIAH/BABILong vs retrofitted linear) | B (control arm) | P | $0 | attention | 2608.28444 |
| Component-collapse diagnosis; HedgeCATs; scheduled SWA dropout | D (fallback audit) | P | $0 | — | 2510.05901 |
| Gated delta-rule mixers keep retrieval; additive saturate; architecture > scale | D | P | $0 | 140M–1.7B only | 2504.14366 |
| Gather-and-Aggregate heads (one head: MMLU 66→25); 15% heads keep retrieval, 84% MMLU | D | P (×2 groups) | $ | — | 2504.18574; 2510.19861 |
| Accept-or-rollback per-layer conversion inside hybrid Qwen3.5-0.8B (fidelity + ΔNLL gate) | D (protocol) | P (weak eval) | $ | 200-item eval | 2609.21139 |
| Distilled hybrid reasoning models (M1: 6/28 attention) | L | P | $$$ | keeps ~20% attention | 2504.10449 |
| Qwen3.8-Next design ablations (why 3:1 keeps attention) | D (context) | P (snippet) | — | **[not fetched]** | 2608.30320 |
| SUPRA / Hedgehog (older linearisation ceilings) | D (context) | S | — | ICL and long-context shortfalls | 2405.06640; 2402.04347 |

## 02 — SSM recall without attention

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Just Read Twice (JRT-Prompt +11.0±1.3 over 16 recurrent LMs; JRT-RNN prefix-linear-attention 96–99% of transformer) | B | S / ordering R | $0 / $ | 2× prefill; harness rendering rule | 2407.05483 |
| Based; MQAR/Zoology as state-capacity diagnostic | D (probe) | R | $0 | Based uses a small window | 2402.18668; 2312.04927 |
| Delta-rule fixed-size memories (GDN, RWKV-7, negative eigenvalues, Mamba-3) | D | R | — | capacity ∝ state; fuzzy recall; interference | 2412.06464; 2503.14456; 2411.12537; 2603.15569; 2404.08819 |
| Test-time regression / TTT memories (MesaNet, Titans, Atlas, LaCT) | pilot core only | P; Titans/Atlas claims **[unreplicated]** | $$$$ | new kernels; from scratch | 2501.12352; 2506.05233; 2501.00663; 2505.23735; 2505.23884 |
| Log-linear attention (O(log L) states) | D (definition edge) | S | $$$$ | state grows | 2506.04761 |
| HOLA bounded exact sidecar cache (residual-based eviction; beats GDN on RULER to 32K) | B (trusted-channel arm) | P | $ | small attention cache | 2607.02303 |
| OPRM chunked inference, entropy-selected chunk (+14–51% LongBench, training-free) | D (= snapshot forks) | S | $0 | drops cross-chunk relations | 2505.07793 |
| ReMamba two-pass selective compression; DeciMamba Δ-norm decimation; LongMamba | D (diagonal arm) | S | $ | Mamba-1-specific | 2408.15496; 2406.14528; 2504.16053 |
| State collapse diagnosis + training-free mitigations (>1M tok) | D (probe) | S | $ | Mamba-2-specific | 2410.07145 |
| Interference wall + distance curriculum (0.02→1.00 recall with distractors) | D (probe + mixture) | P (toy) | $0 | transfer unknown | 2609.16183 |
| Birdie RL-mixed retrieval objectives; paraphrase augmentation; CoT distillation for recall | D (mixture) | S | $0 | — | 2411.01030; 2510.26912; 2505.03320 |
| RAG or one transformer layer closes the RNN representational gap | B | S | — | — | 2402.18510 |
| Memory-augmented recurrences (MARCH, Sparse Delta Memory, RAM-Net, hierarchical) | pilot core watch-list | P | $$$$ | from scratch | 2608.12435; 2607.07386; 2602.11958; 2608.02347 |

## 03 — SSM interpretability and control

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Implicit (hidden) attention matrices of Mamba / gated recurrences | B | R | $0 | smooth attention; aggregation choices | 2403.01590; 2405.16504 **[not fetched]** |
| MambaLRP conservation-corrected relevance | D | S | $0 (Mamba) / eng (GDN) | attribution ≠ cause | 2406.07592 |
| LaTIM token-to-token decomposition | D | S | $0 | Mamba-1/2 only | 2502.15612 |
| Causal tracing, ROME editing, knockout in Mamba | B | R | $ | non-unique localisation | 2404.03646; 2505.24244 |
| EAP/ACDC circuit discovery in Mamba (IOI) | D | S | $ | — | 2407.14008 |
| Activation-subspace bottleneck scalar steering (+8.27% avg; Stable-Mamba) | D (control probe) | P | $ | mechanism unexplained | 2602.22719 |
| Non-unique causal localisation (state sink; dual heads; Δ vs output dissociation) | B (method rule) | P | $0 | — | 2606.00930 |
| WriteSAE rank-1 write atoms on GDN/Mamba-2/RWKV-7; replacement test 92.4%; GDN logit formula R²=0.98; cache-level steering | B | P | $–$$ | small models; SAE ≠ causal | 2605.12770 |
| SAE concept comparison SSM vs transformer ("dark matter"); Persistent SAEs (feature timescales; injection signal persists) | B | P | $ | — | 2609.24440; 2607.17117 |
| Influence Score (controllability/observability-style per-token influence) | D (telemetry) | P | $0 | trajectory-conditioned | 2511.17970 |
| Gramians / HSVs / in-training balanced truncation (CompreSSM, HSVR) | D (diagonal arm only) | R math / S use | $0 | not for GDN; per-layer LTI | 2510.02823; 2510.22951 |
| Decay-spectrum analysis; PoST geometric spectral init; Mamba Modulation | B | P/S; lab decay-init R-int | $0 | eigenvalue view invalid for GDN; use retained-norm | 2604.07658; 2509.19633; 2506.11891 |
| "Mamba lens" tooling | — | **[unverified: not found]** | — | — | — |

## 04 — Causal interpretability by construction

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Causal abstraction (formal target; IIA metric) | B | R | $0 | vacuous without encoding assumptions | 2301.04709; 2602.16612 **[not fetched]** |
| Interchange Intervention Training from init (zero loss ⇒ causal abstraction) | B (pilot core; head) | S + applied follow-ups; no SSM/LLM-scale use found | $0 extra / $$ pilot | needs full causal model; leakage outside aligned slice | 2112.00826; 2212.09897 |
| DAS / Boundless DAS; pyvene; RAVEL; MIB | B (probe cards) | R | $ | linear-only; needs near-perfect task accuracy | 2303.02536; 2305.08809; 2403.07809; 2402.17700; 2504.13151 |
| Non-linear representation dilemma (random nets reach 100% with free maps) | B (rule) | S (NeurIPS 2025 spotlight) | $0 | — | 2507.08802; 2602.16698; 2511.04638 |
| Concept-DAS faithful bidirectional steering (distribution matching + DII) | L | S (ICLR 2026) | $ | not always > preference-opt baselines | 2602.05234 |
| Concept bottleneck models; leakage (soft concepts), hard CBM, CEM | B | R | $–$$ | sequence-level side channel | 2106.13314; 2105.04289; Havasi 2022; 2209.09056 |
| Minimal CBM (IB regulariser; "never a bottleneck"); MI leakage estimators | B (leakage test) | S (ICLR 2026) / P | $ | accuracy cost unquantified | 2506.04877; 2504.09459; 2504.14094; 2603.05629 |
| CB-LLM concept layer for classification + controlled generation | L (output check) | S | $$ | — | 2412.07992 |
| Monet monosemantic experts (262k/layer); geometric-routing MoE (+321%/−23% causal steering) | pilot core | S / P | $$$$ from scratch | partial specialisation | 2412.04139; 2604.14434 |
| Backpack sense-vector interface | L (lexical control) | S | $$ | non-contextual senses; scale | 2305.16765; 2605.28669 **[not fetched]** |
| Codebook features (VQ bottlenecks per layer) | pilot (discrete interface arm) | S; caveat P | $$ | codes can be polysemantic | 2310.17230; 2407.19532 |
| Steerling-8B "interpretability scales with capability" | B (existence proof) | P (unverified by third parties) | — | — | 2608.07594; survey 2604.16042 |

## 05 — Prompt injection, authority, information flow

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| StruQ / SecAlign / Meta SecAlign fine-tuning | D (= injection-augmented LoRA) | S static; **R broken** (85–95% adaptive) | $$ | white-box attacks; utility under-reported | 2402.06363; 2410.05451; 2507.02735; 2507.07417; 2505.18333 |
| Adaptive-attack evaluation standard (12 defences >90% ASR) | B (gate design) | R | $–$$ | — | 2510.09023; 2606.26479 |
| Instruction hierarchy; ISE segment embeddings; ASIDE data-token rotation (0 params) | B (= provenance channel) | S | $0–$ | content-based steering untouched; adaptive halves gains | 2404.13208; 2410.09102; 2503.10566; 2606.07808; 2608.28648 |
| Spotlighting (delimit / datamark / encode) | B ($0 control) | S static; no adaptive eval | $0 | small models; untested adaptively | 2403.14720 |
| CaMeL control/data-flow separation (67% AgentDojo with provable security) | L (= Plan→Check→Realize) | S | $0 | utility loss; side channels; values still steer | 2503.18813 |
| FIDES IFC labels + deterministic policy; successors APPA/SPA/AgentFlow | L | S | $0 | label provenance; forging report **[unverified web]** | 2505.23643; 2607.24625; 2608.27234; 2608.22868 |
| Progent privilege control (adaptive 4.2%/2.6%, one weak model) ; PACE provenance-aware capability enforcement (62/79 lowest ASR; ≤3 pt utility) | B | S / P | $0 | within-mask steering out of scope | 2504.11703; 2610.01349 |
| Six design patterns (action-selector = SSM-1; plan-then-execute + dual LLM = SSM-LM) | B | position / adoption R | $0 | — | 2506.08837 |
| Role-stratified per-field CRC on tool-call arguments; Jev decision hijacking (1.8%/3.5%) | B | S / P | $0 | sample size per role | 2607.24343; 2609.28613 |
| Detection-based in-band defences (MELON, DataFilter, LocalAlign, ARGUS …) | — (alarm only) | P/S static; broken adaptively | $0 | — | 2502.05174; 2510.19207; 2605.01462; 2605.03378 |
| Recurrent-state attack surface (gate-holding, persistence, state reuse) | B | X (lab-identified gap); SpectralGuard **[not fetched]** | $ | — | 2603.12414 |

## 06 — Calibration and selective prediction

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Temperature / Dirichlet scaling | B | R | $0 | one T cannot fix length dependence; shift | 1706.04599 **[ID unverified]**; 1910.12656 |
| Predicted temperature (Thermometer per task; ATS per token) | B (= per-question T with length feature) | S each | $0–$ | overfit calibration split | 2403.08819; 2409.19817 |
| Debiased ECE; ECE_sweep; SmoothECE | B (reporting) | R | $0 | noise floor (review §9) | 1909.10155; 2012.08668; 2309.12236 |
| Calibration under shift: weighted CP, robust CP, ACI | B (deployment) | R | $0 | density ratio / shift budget / feedback needed | 1906.02530; 1904.06019; 2008.04267; 2106.00170 |
| LTT; CRC; Selective CRC (selected-risk bound); BB-CRC for small sets | B (deployment) | R / S | $0 | exchangeability under attack; ~2,995 zero-error cases for 0.1% | 2110.01052; 2208.02814; 2512.12844; 2509.23007 |
| Selective classification: LR-test optimal rule; gap decomposition (calibration doesn't reorder); AUGRC metric | B (deployment) | S each; softmax-response strength R | $ | density-ratio estimate | 2505.15008; 2510.20242; 2407.01032; 1901.09192; 1907.00208 |
| PriDe option-ID prior debiasing (equivariance audit) | D (unit test) | R | $0 | — | 2309.03882 |
| Decision calibration (utility-level) | B | S | $0 | **[ID unverified]** | 2107.05719 |

## 07 — Verified generation from semantic programs

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Grammar-constrained decoding engines (Outlines, XGrammar, SynCode, DOMINO) | L | R | $0 | syntax only; distortion; can force malicious code | 2307.09702; 2411.15100; 2403.01632; 2403.06988; 2608.28229 |
| Format tax (instruction, not mask, causes most loss); generate-then-reformat; MCMC grammar-aligned sampling; CRANE | L (ablation) | R / S | $0 | — | 2604.03616; 2408.02442; 2506.05754; 2510.01902; 2502.09061 |
| Semantic prefix oracles (zero false prunes; +15 pts); Ctrl-G HMM×DFA; NeuroLogic | L (rung B) | P / S | $0 | code-domain evidence; HMM cost | 2609.35425; 2406.13892; 2010.12884 |
| Grammatical Framework linearisation from typed abstract syntax; ACE-in-GF; CNL survey | L (rung A, exact) | R | $0 (eng weeks) | stilted; coverage by hand | Ranta 2011; 1303.4293; 1507.01701 |
| Rich MRs: MRS/ERG realisation vs parsing asymmetry (76 BLEU vs 57–65 F1, ~1% exact); UMR→AMR loses tense/aspect/reference; scope only in MRS/DRS | L (AST design; round-trip rule) | S / R | $0 | — | 2609.30071; 1904.11564; 2502.11973 |
| Data-to-text faithfulness: PARENT, ToTTo, plan-then-generate, neural pipeline, NLI omission/hallucination checks | L (rung B gate; selection error) | R | $ | NLI model errors | 1906.01081; 2004.14373; 2108.13740; 2203.16279; 2011.10819 |
| Claim-level verification (FActScore/VeriScore); decomposition dilemmas | L (metric) | R / S | $ | decomposition is a bottleneck | 2305.14251; 2406.19276; 2411.02400; 2403.11903 |
| Cycle consistency (CycleGT; logic-consistency from parses) — training signal, not proof | L | R / S | $ | LLM parser unreliable | 2006.04702; 2108.00577 |
| Typed generation targets with guarded execution (VirtualSet: 20/20 hallucinated actions intercepted) | L (Plan constructor) | P | $0 | schema must lift to typed world | 2607.18821 |

## 08 — Developmental tracking

| Technique | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Checkpoint suites; 50-seed stability and outlier detection; checkpoint averaging | B | R | $ (3 seeds) | — | 2304.01373; 2503.09543; 2501.00656 |
| Probe accuracy + control-task selectivity across checkpoints; stitching for layer comparison | B | R | $0 | probes non-monotone | 2506.02132; 2309.07311; 2506.06609 |
| LLC / rLLC / susceptibilities / influence dynamics; RMSProp-SGLD benchmark; SIVE | side paper; logged only | S (Timaeus); **R-int negative (lab)** | $ ≤1B; unreliable at 9B-LoRA | SGLD bias invisible to R-hat; sign flips | 2308.12108; 2402.03698; 2402.02364; 2410.02984; 2504.18274; 2510.12071; 2507.21449; 2606.22389 |
| Loss-to-loss prediction; emergence mirage; per-seed capability forecasting with pre-registered gate | B (phase detection) | R / P | $0 | precursor must exist | 2411.12925; 2304.15004; 2207.08799; 2609.19000 |
| Data-mixture control (DoReMi, RegMix, mixing laws, ADO, Aioli, Skill-it) | B | R family | $0 | optimises loss; lab: loss ≠ decisions | 2305.10429; 2407.01492; 2403.16952; 2410.11820; 2411.05735; 2307.14430 |
| Checkpoint eval card (our assembly: Index slice, NLL, calibration, steer, recall, mechanism, leakage, logged LLC) | B | X (design) | $ / checkpoint | — | files 02–06, 08 §8.6 |

## 10 — Clef recipe and gap attribution

| Technique / finding | Serves | Evidence | Cost | Failure modes | Cite |
|---|---|---|---|---|---|
| Index 0.3 Full score (20% public / 50% private same-skills / 30% private new-domains); Clef 53.08, Clef-flash 47.61, Jev 60.11, Nimble-9B-v3 54.67 | D (targets) | P (board data) | $0 | PLAN tiers stale; `decision-needed` | Space `data/v03.json` 2026-10-06 |
| Clef recipe: frozen backbone + r=256 LoRA + 128M routing head + LS-CE + Brier + RLCD on synthetic schema permutations | D (anchor E2) | P (blog; data undisclosed) | $–$$ | public-format overfit (public→Full −8.5) | blog.cloudflare.com/clef-decision-models |
| Measured gap stock Q9 → Clef-flash: 17.5 Index pts on DEV subset, all post-training; option-key tax (letter/option_N/hash) | D | R-int (lab receipts) | $0 | 12–60 rows per benchmark | `receipts/2026-10-06-D-recipe-gap` |
| Readout fusion: LM answer-logprob × trained head (+1.27 [−0.38, 2.83], in-sample) | D (head input E4) | R-int (diagnostic) | $0 | per-benchmark weights forbidden; must be feature-gated | same receipt |
| Head + LoRA ≈ generation SFT at 8B; head-only ≈ 40% of LoRA gain | D | P | $ | — | 2607.03801; 2501.17994 |
| LS inverts confidence ranking (fix: logit normalisation) | D | S | $0 | — | 2403.14715 |
| Attention-free retention at 7–8B: MMLU 88–95%, GSM8K ~49%; no MMLU-Pro/GPQA/tool numbers exist | D | P | — | log-likelihood scoring understates gaps | 2505.03005; 2502.14458; 2501.15570; 2603.26556 |
| Anchor-then-ablate (vs component-first from stock) | D (method) | X | — | readout×training, training×calibration, conversion×training must be end-to-end | file 10 §10 |

## Cross-domain verdicts (one line each)
- Attention-free conversion will lose retrieval; the fallback (1 layer or 15% heads) is the expected result — pre-register it (01).
- Options-first ordering + chunk-and-aggregate forks + retrieval-objective mixture are the three cheap recall levers with evidence (02).
- Causal claims on SSMs need alternative-circuit controls and linear alignment maps; WriteSAE is the only GDN-native write-level tool (03, 04).
- Provenance channels = ISE/ASIDE; the guarantee stack = action-selector / plan-then-execute patterns; steer-rate is the metric the field does not measure (05).
- Predicted temperature + debiased ECE for the Index; SCRC + AUGRC + reordering scores for deployment (06).
- Rung A is a GF grammar; round trips go through the symbolic parser; the AST needs scope/tense/number before L1 (07).
- Card metrics over seeds, NLL-based phase detection, mixture control on decision metrics, LLC logged only (08).
- Clef-flash's 17.5-point lead over stock Qwen3.5-9B is post-training, not backbone; under Index 0.3 transferable levers (readout, joint decoding, calibration, generator data) beat public-format supervision; run anchor-then-ablate (10).

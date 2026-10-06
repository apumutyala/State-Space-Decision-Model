# 07 — Verified language generation from semantic programs

Written 2026-10-06. Same legend as `01`. Settled by the reviews: the "hard grounding theorem" is a syntactic lemma about atomic-claim membership in K (`reviews/02` §11); not covered are selection, omission, composition, scope/negation, aggregation, implicature, correctness of K; Ctrl-G's HMM is overkill when a DFA mask suffices (§12); the partner plan requires `Meaning(r) = AssertedMeaning(p)` over a defined formal language and says a fallible learned parser is not a proof.

**Headline:** (1) **Rung A has a 20-year-old exact solution:** Grammatical Framework linearisation from a typed abstract syntax is meaning-preserving by construction and parses back exactly for the controlled fragment (ACE-in-GF covers ~20 languages). (2) **Round-trip verification with an LLM parser is not a verifier:** Claude Opus realises MRS at 76.3 BLEU but parses text→MRS at 57–65 F1 with ~1% exact match vs 91 F1 for the symbolic ERG parser (2609.30071, Sep 2026) — use the symbolic parser for the round trip and restrict to its coverage. (3) **Meaning representations are not equally sufficient:** AMR drops tense/number/scope; UMR→AMR conversion provably loses tense/aspect/reference (2502.11973); MRS/DRS carry scope. The plan's AST must encode tense, number, quantifier scope, negation and time before any "faithful realisation" claim at L1+. (4) **Constrained decoding is syntactic and distorts the distribution;** most of the "format tax" is caused by the *format instruction*, not the decoder mask (2604.03616), and is recoverable by generate-then-reformat. (5) Claim-level verification helps at response level and can hurt at claim level (2411.02400); omission needs its own NLI check (2011.10819).

---


> **Deployment status (2026-10-06, accepted contract):** every deployed utterance must have whole-answer licensed semantics. Free connective text (§7.3 rung B), text outside the guaranteed region (§7.8), per-claim/NLI/round-trip checks and the "fraction of tokens outside the guarantee" metric are **excluded from the deployment plan**. They are kept here only as external research comparisons and ablation arms. Copy-slot correctness, NLI or round-trip scores do not prove relation, scope or omission faithfulness. Paper-specific claims marked [unverified] or [not fetched] must be verified before they drive any paid decision.

## Techniques

### 7.1 Grammar-constrained decoding engines (syntactic guarantee)
- **Cite:** Outlines, arXiv 2307.09702; XGrammar, arXiv 2411.15100 (byte-level pushdown automaton for CFGs); SynCode, arXiv 2403.01632; DOMINO, arXiv 2403.06988 (precomputation + speculative decoding, near-zero overhead; ~25% overhead on a 70-rule C grammar); llguidance (no paper **[unverified]**); distance-guided decoding for guaranteed CFG compliance, arXiv 2608.28229; GRAMMAR-LLM (ACL Findings 2025).
- **Evidence:** R (engines are production software).
- **Known:** every emitted string is in L(G); overhead is small for regex/JSON and grows with grammar size; token/byte alignment handled by the engines.
- **Failure modes:** only syntax; distribution distortion (7.2); grammar-constrained decoding can be used to *force* malicious code (2606.11817); large natural-language grammars are slow.
- **Cost:** $0.
- **Serves:** LM project (realiser mask; also the typed-program emission mask for the planner).

### 7.2 Distribution distortion and the format tax
- **Cite:** "Let Me Speak Freely?", Tam et al., arXiv 2408.02442 (format restrictions hurt reasoning, help classification); **"The Format Tax", Lee, D'Antoni, Berg-Kirkpatrick, arXiv 2604.03616**: the *instruction* to produce a format causes most of the loss before any decoder constraint; recent closed models show little tax; mitigations: freeform-then-reformat, extended thinking; CRANE (reasoning with constrained generation by alternating), arXiv 2502.09061; "Draft-Conditioned Constrained Decoding", arXiv 2603.03305; MCMC grammar-aligned sampling, arXiv 2506.05754 (Metropolis–Hastings to the true grammar-conditional; ASAp converges slowly); Constrained Adaptive Rejection Sampling, arXiv 2510.01902; AdapTrack, arXiv 2510.17376.
- **Evidence:** R for the distortion; S for each mitigation.
- **Known:** masking + renormalising at each token ≠ conditioning the sequence distribution on the grammar; the gap matters when the grammar excludes the model's reasoning steps.
- **How we'd use it:** the realiser never reasons under the mask — the *program* is already decided; realisation is a template-fill, so distortion is benign. For the planner's typed-program emission, measure the format tax by an unconstrained-then-parse arm (2604.03616 protocol) before blaming the grammar.

### 7.3 Semantic-aware constrained decoding
- **Cite:** Semantic Prefix Oracles (semantic grammar specifications executed during Earley descent; safe pruning + dead-end freedom; zero false prunes on 65 compiler-valid programs; +15.2 correctness points across 12 models), Kronlund-Drouault, arXiv 2609.35425; Projectional Decoding (design-by-contract validity), arXiv 2605.30054 **[not fetched]**; Ctrl-G (HMM × DFA, 100% constraint satisfaction, lookahead), arXiv 2406.13892 (NeurIPS 2024); NeuroLogic (predicate-logic lexical constraints), arXiv 2010.12884; Relation-constrained decoding (NeurIPS 2022) **[not fetched]**.
- **Evidence:** P (prefix oracles; code domain), S (Ctrl-G, NeuroLogic).
- **Known:** attribute-grammar-style semantic checks can be executed incrementally with a continuability guarantee; Ctrl-G gives lookahead when fluency under a pure mask is poor.
- **Ablation only (excluded from deployment):** rung B (free connective text): incremental semantic checks (every entity mention resolves to a checked record; every numeric literal equals a checked value; negation scope matches the program) as a prefix oracle over the realiser grammar; Ctrl-G only if the DFA mask's fluency is measurably bad.

### 7.4 Grammatical Framework and controlled natural languages (rung A, exact)
- **Cite:** Ranta, *Grammatical Framework* (CSLI 2011; grammaticalframework.org); ACE-in-GF (Attempto Controlled English 6.7 subset in GF, ~20 languages; GitHub Attempto/ACE-in-GF; arXiv 1303.4293 multilingual semantic wiki); Kuhn, "A Survey and Classification of Controlled Natural Languages", Computational Linguistics 40(1) 2014, arXiv 1507.01701; multilingual FrameNet-based CNL grammar, arXiv 1511.03924.
- **Evidence:** R (decades of deployment; formal semantics for ACE).
- **Known:** GF separates an abstract syntax (typed trees = our semantic programs) from concrete syntaxes (linearisation); linearisation is deterministic and total; parsing is exact within the grammar; variants let a grammar license multiple surface forms per tree.
- **Failure modes:** stilted text; coverage grows only by grammar engineering; ambiguity in parsing if the concrete syntax is not injective (then meaning ≠ surface, the review's caveat); no learned fluency.
- **Cost:** $0 compute; engineering weeks.
- **Serves:** LM project — **rung A is a GF grammar whose abstract syntax is the Assert/Infer/Compare/Explain/Ask/Quote/Hypothesize/Plan algebra with copy-slots bound to checked records.** The LM's role in rung A is to choose among licensed variants (a decision), not to emit tokens.
- **How we'd use it:** build the L0/L1 grammar in GF (or a GF-like typed linearizer in Python with the same discipline). (The earlier "fraction of tokens in the guaranteed region" metric is withdrawn; whole-answer semantics is required.)

### 7.5 Rich meaning representations and their expressivity (what the AST must carry)
- **Cite:** Hajdik, Buys, Goodman, Bender, "Neural Text Generation from Rich Semantic Representations" (DMRS→text; BLEU 66→77 with ERG silver data), arXiv 1904.11564 (NAACL 2019); **Dan, "Scoring Both Directions: LLMs realize the MRS they cannot reliably parse", arXiv 2609.30071**; "Generating Text from Uniform Meaning Representation", arXiv 2502.11973 (UMR→AMR loses tense, aspect, referential features; pronoun consistency failures); AMR applications survey (EMNLP 2024, "decoders produce fluent sentences that may not retain the meaning"); DMR 2025 workshop proceedings (scope: MRS/DRS explicit, UMR optional scope node); Neural semantic parsing with extremely rich symbolic MRs, arXiv 2404.12698 **[not fetched]**; Online back-parsing for AMR-to-text (EMNLP 2020) **[not fetched]**.
- **Evidence:** R for the expressivity hierarchy (AMR < UMR < MRS/DRS on scope/tense); S for the LLM asymmetry.
- **Known:** realisation from a rich MR is easy for LLMs; parsing into it is not; AMR-level programs cannot state "no invoice was paid after March" unambiguously.
- **How we'd use it:** (i) the semantic program language gets MRS-level constructors for quantifier scope, negation scope, tense/aspect, number, time intervals *before* L1 (partner plan's "add deliberately"); (ii) expressivity-gap tests: take DRS/MRS-annotated corpora (Parallel Meaning Bank for DRS, from memory **[unverified]**; Redwoods/ERG treebanks for MRS) and measure what fraction of annotated meanings the AST can express; (iii) round trip via the ERG parser (ACE) where the text is in ERG coverage, never via an LLM parser.

### 7.6 Data-to-text faithfulness (metrics, planning, pipelines)
- **Cite:** PARENT, Dhingra et al., arXiv 1906.01081 (ACL 2019; aligns n-grams to the table; better human correlation than BLEU/ROUGE); ToTTo, arXiv 2004.14373 (controlled table-to-text with highlighted cells); Plan-then-Generate, Su et al., arXiv 2108.13740 (EMNLP Findings 2021; content plan controls intra/inter-sentence structure); Macro planning (TACL 2021); Neural Pipeline for zero-shot D2T, Kasner & Dušek, arXiv 2203.16279 (ACL 2022; template facts → ordering → aggregation → compression, each a separately checkable stage); NLI-based semantic accuracy, Dušek & Kasner, arXiv 2011.10819 (INLG 2020: omission check = text as premise, each fact as hypothesis; hallucination check = facts as premise, text as hypothesis); gold-standard accuracy methodology, arXiv 2011.03992; "A Logic-Based Approach to Hallucinations in Data-to-Text" (*SEM 2026, aclanthology 2026.starsem-conference.3) **[not fetched]**; ASPIRO parse-error-induced reprompting, arXiv 2310.17877; "From Tables to Quantified Statements: executable verification", arXiv 2609.23966 **[not fetched]**; SPOR compositional generalisation in D2T, arXiv 2405.10650.
- **Evidence:** R (PARENT, NLI checks, pipelines are standard).
- **Known:** the neural-pipeline decomposition is the published precedent for Plan→Check→Realize in NLG; NLI omission/hallucination checks are the cheap empirical gate the review asked for; PARENT measures *selection* fidelity against the data, which the syntactic lemma does not.
- **Ablation only:** rung B gate = NLI omission + hallucination checks (2011.10819 protocol) with a locally served NLI model; PARENT-style precision/recall against K as the "selection error" metric; pipeline stages mirror Kasner & Dušek so each stage has a checker.

### 7.7 Claim-level verification and decomposition
- **Cite:** FActScore, arXiv 2305.14251; VeriScore, arXiv 2406.19276; VeriFastScore, arXiv 2505.16973; "Decomposition Dilemmas", Hu, Long, Wang, arXiv 2411.02400 (decomposition helps response-level inputs, often hurts claim-level; depends on verifier strength and method); "The Alignment Bottleneck in Decomposition-Based Claim Verification", arXiv 2602.10380 **[not fetched]**; "A Closer Look at Claim Decomposition", arXiv 2403.11903; "Towards Effective Extraction and Evaluation of Factual Claims", arXiv 2502.10855.
- **Evidence:** R (FActScore-style pipelines); S (dilemmas).
- **Known:** decompose-then-verify is standard but decomposition quality is a bottleneck; the realised text's claims must be *aligned back* to program nodes — which we have for free in rung A (each sentence is a linearised node) and must recover in rung B.
- **How we'd use it:** claim precision on a closed-KB QA set (plan's metric) computed by aligning realised sentences to program nodes, then verifying each node against K with the deterministic checker; the LLM-based decomposer is used only for the free-text rung and its agreement with the node alignment is itself reported.

### 7.8 Round-trip / cycle consistency (as a check, not a proof)
- **Cite:** CycleGT, arXiv 2006.04702 (G2T/T2G cycle training on WebNLG); "Logic-Consistency Text Generation from Semantic Parses" (SNOWBALL), arXiv 2108.00577; "Assessing Semantic Consistency in Data-to-Text Generation" (INLG 2025) **[not fetched]**; NL2LOGIC AST-guided translation (EACL Findings 2026) **[not fetched]**; 2609.30071 (LLM parsers are unreliable in the parse direction).
- **Evidence:** R for the training trick; the "not a verifier" point is now S.
- **How we'd use it:** cycle consistency as a *training signal* for the free realiser; verification only through the symbolic parser within coverage; outside coverage the sentence is not deployable; it may be scored by 7.6–7.7 only in an ablation arm.

### 7.9 Typed generation targets and guarded execution (precedents for the Plan constructor)
- **Cite:** VirtualSet (typed set expressions over entity-edge worlds; Generic Constraint Projection validates before execution; virtual-world simulation then external approval; BIRD 67.5% vs 63.5%, p=0.001; 20/20 hallucinated action bodies intercepted, 0 FP), arXiv 2607.18821; Text2DSL, arXiv 2606.22586 **[not fetched]**; "Weave of Formal Thought", arXiv 2606.25987 **[not fetched]**; "Correctness-Guaranteed Code Generation via Constrained Decoding", arXiv 2508.15866 **[not fetched]**.
- **Evidence:** P.
- **How we'd use it:** Plan(actions, preconditions) realises as a typed expression validated by type projection and simulated in a virtual world before any effect — VirtualSet's two-stage guard is the published form of the partner plan's "execution still checks current authority and state".

### 7.10 Lexically / relation-constrained decoding (older, still useful)
- NeuroLogic (2010.12884), NeuroLogic A*esque (2112.08726 from memory **[unverified]**), Relation-Constrained Decoding (NeurIPS 2022): enforce that named entities/relations appear. Subsumed by 7.3 for us.

---

## How we'd use this domain (LM project, by rung)

| Rung | Mechanism | Guarantee | Residual measured by |
|---|---|---|---|
| A (L0–L1) | GF-style linearisation of typed AST with copy-slots; LM chooses among licensed variants | every atomic claim's slot values copied from checked K; exact parse-back within grammar | selection error (PARENT-style vs K), omission (NLI), scope/tense tests from MRS-level constructors |
| B (ablation only, not deployed) | LM realiser under DFA mask + semantic prefix oracle; NLI omission/hallucination gate; claim alignment to nodes; ERG round-trip within coverage | none beyond syntax + per-claim checks | claim precision, omission rate, round-trip agreement, fraction of tokens outside the guaranteed region |
| Plan | typed expression, type projection, virtual-world simulation, authority check | effects ⊆ authorised | wrong-permitted-action rate (lab harness) |

Format-tax ablation (2604.03616) on the planner's program emission; distortion is irrelevant in rung A by construction.

## Could not verify this session
- llguidance documentation; Parallel Meaning Bank ID; NeuroLogic A*esque ID; details of 2605.30054, 2404.12698, 2602.10380, 2606.22586, 2606.25987, 2508.15866, *SEM 2026 paper, INLG 2025 paper, EACL 2026 NL2LOGIC.

# 04 — Causal interpretability by construction

Written 2026-10-06. Same legend as `01`. This domain serves both projects but is load-bearing for the partner plan's "controlled branch consumes an explicit evidence interface, with no direct latent-to-decision bypass" and for the claude plan's "concept-supervised block + leakage test + completeness test".

**Headline:** (1) Interchange-intervention training (IIT) is the only method found that *trains* a network to realise a given causal model with a zero-loss guarantee of causal abstraction; it has not been applied to SSMs or at >1B scale in anything found this session. (2) Distributed alignment search (DAS) and its steering descendant (Concept-DAS, ICLR 2026) are the standard way to *test* whether a variable is linearly localised, but 2507.08802 (NeurIPS 2025 spotlight) proves causal abstraction is vacuous without a linearity-type constraint on the alignment map — random models score 100% with unrestricted maps. (3) Concept bottlenecks leak by default; ICLR 2026 ("There Was Never a Bottleneck") formalises this and proposes an information-bottleneck fix; the leakage test the reviews demanded is now a named sub-literature with information-theoretic estimators. (4) Intrinsically interpretable LMs have scaled to 8B with a claimed *negative* capability tax (Steerling-8B, Guide Labs, Aug 2026, P) — the only such claim; treat as the existence proof to beat, not as settled.

---

## Techniques

### 4.1 Causal abstraction (the formal target)
- **Cite:** Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability", arXiv 2301.04709 (v4 May 2025); "Causal and Compositional Abstraction", arXiv 2602.16612 **[not fetched]**.
- **Evidence:** R (theory; adopted by RAVEL, MIB, pyvene).
- **Known:** a high-level causal model abstracts a low-level one iff interchange interventions commute; generalised from mechanism replacement to arbitrary transformations; formalises polysemantic neurons, linear representation hypothesis, graded faithfulness.
- **Failure modes:** vacuity without encoding assumptions (4.3).
- **How we'd use it:** state the partner plan's invariant (`Check(K,Π,p) ⇒ Authorized ∧ K ⊨ AssertedMeaning(p)`) as the high-level causal model and the explicit semantic records as the aligned variables; IIA (interchange-intervention accuracy) is the acceptance metric for "intervention agreement".

### 4.2 Interchange Intervention Training (IIT) — training the causal structure in from init
- **Cite:** Geiger, Wu, Lu, Rozner, Kreiss, Icard, Goodman, Potts, "Inducing Causal Structure for Interpretable Neural Networks", arXiv 2112.00826, ICML 2022. Code: frankaging/Interchange-Intervention-Training. Type-level IIT for character structure: arXiv 2212.09897. Causal distillation (Wu et al. 2022, from memory **[unverified ID]**).
- **Evidence:** S (ICML) + several applied follow-ups (2503.14442 glucose; 2505.00555); no LLM-scale or SSM application found.
- **Known:** align causal-model variables to network representation slices; train the network so that swapping the aligned slice from a source input produces the counterfactual output of the causal model; **if the loss is zero the causal model is provably a causal abstraction of the network**; beats multi-task and data-augmentation baselines on interpretability (realising the target model) at equal task accuracy.
- **Failure modes:** needs a complete high-level model and counterfactual labels (we have both: the lab's exact state machines and oracle programs); alignment is fixed a priori (DAS relaxes this); capacity split between task and causal objective; representation may still encode extra information outside the aligned slice (leakage, 4.5).
- **Cost:** $0 extra compute beyond paired forward passes (2–3× batch cost); fits the partner plan's 100–400M pilot ($$).
- **Serves:** both. Decision model: train the explicit record state (KEEP/SET/RETRACT/UNKNOWN/CONFLICT) as the aligned variables. LM: the typed program AST nodes as aligned variables in the planner.
- **How we'd use it:** pilot arm "IIT-from-init" vs "same architecture, supervised state only" vs "unrestricted"; metrics: task utility, IIA under record-swap interventions, leakage (4.5), and the partner plan's prospective prediction test. This is the concrete content of "intervention-based training from init".

### 4.3 DAS / Boundless DAS and the non-linear representation dilemma
- **Cite:** DAS, arXiv 2303.02536; Boundless DAS (Alpaca-7B), arXiv 2305.08809; pyvene, arXiv 2403.07809; RAVEL benchmark, arXiv 2402.17700 (ACL 2024); MIB benchmark, arXiv 2504.13151; **Sutter, Minder, Hofmann, Pimentel, "The Non-Linear Representation Dilemma", arXiv 2507.08802 (NeurIPS 2025 spotlight)**; "Addressing divergent representations from causal interventions", arXiv 2511.04638; "Causality is Key for Interpretability Claims to Generalise", arXiv 2602.16698.
- **Evidence:** R for DAS; S for the dilemma result.
- **Known:** DAS learns an orthogonal rotation and a subspace such that interchange interventions in the rotated basis reproduce the causal model; works on distributed representations. **With unrestricted (non-linear) alignment maps, any network can be aligned to any algorithm — randomly initialised LMs reach 100% on interpretability benchmarks.** Interventions can also push activations off-distribution (2511.04638). Pearl-hierarchy framing: observational probes cannot support counterfactual claims (2602.16698).
- **Failure modes:** DAS assumes near-perfect task accuracy; brute-force over subspace dimension; linear-only variants.
- **Cost:** $ (DAS on a 9B head input is minutes–hours).
- **How we'd use it:** (i) restrict all alignment maps to linear/orthogonal and report map complexity; (ii) run DAS on the *head input* of Q9-AF for "chosen option", "provenance of decisive evidence", "option count" as a probe-stability column; (iii) RAVEL-style disentanglement test for the semantic record attributes (entity, relation, value, time).

### 4.4 Concept-DAS steering (faithful bidirectional steering)
- **Cite:** Bao et al., "Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions", arXiv 2602.05234 (ICLR 2026).
- **Evidence:** S.
- **Known:** replaces probability maximisation with a distribution-matching objective to counterfactual distributions; steering factors learned from data; on AxBench it does not consistently beat preference-optimisation baselines but improves with scale; safety case studies (refusal override, CoT backdoor neutralisation) keep utility.
- **How we'd use it:** the LM project's "licensed paraphrase/style" control (steer discourse choices without touching asserted meaning) and as the control arm against provenance-gate interventions.

### 4.5 Concept bottleneck models and leakage
- **Cite:** CBM, Koh et al. 2020 (arXiv 2007.04612, from memory **[ID unverified]**); leakage: Mahinpei et al. arXiv 2106.13314, Margeloiu et al. arXiv 2105.04289, Havasi, Parbhoo, Doshi-Velez NeurIPS 2022 (hard CBM + side channel); CEM, arXiv 2209.09056 (NeurIPS 2022); intervention-aware CEM (NeurIPS 2023) and "Avoiding Leakage Poisoning" (ICML 2025) in mateoespinosa/cem **[not fetched]**; **Almudévar, Hernández-Lobato, Ortega, "There Was Never a Bottleneck in Concept Bottleneck Models", arXiv 2506.04877, ICLR 2026** (MCBM: variational information-bottleneck regulariser so each component keeps only its concept's information); "Measuring Leakage in Concept-Based Methods: An Information Theoretic Approach", arXiv 2504.09459; "Leakage and Interpretability in Concept-Based Models", arXiv 2504.14094; "In Defense of Information Leakage", arXiv 2606.10669 (contrarian); "Rethinking CBMs: pitfalls to solutions", arXiv 2603.05629; CB-LLM, arXiv 2412.07992 (ICLR 2025).
- **Evidence:** R for leakage; S for MCBM; P for the estimators.
- **Known:** soft concept scores encode non-concept information; hard (binarised) concepts reduce leakage at an accuracy cost; CEMs recover accuracy with high-dimensional concept embeddings (and leak more by construction); MCBM constrains each component to its concept via an IB objective; information-theoretic leakage estimators exist; CB-LLMs get competitive classification and controllable generation with a concept layer.
- **Failure modes:** in a *sequence* model the concept trajectory is a wide side channel (review §2); leakage estimators need MI estimation; MCBM's accuracy cost not quantified in the abstract.
- **Cost:** $ (a concept head on frozen features) to $$ (joint training in the pilot).
- **How we'd use it (the plan's mandatory leakage test, made concrete):** (a) masked-concept probe: train a probe from h_sem→label with concept semantics removed; leakage = its accuracy above the concept-only predictor; (b) MI estimate per 2504.09459; (c) intervention effectiveness: flip a concept and measure decision change vs predicted; (d) MCBM IB regulariser as the trained-in mitigation, with its utility tax reported; (e) completeness: ablate h_free and show no accuracy loss, else the glass box has a back door.

### 4.6 Monosemantic-by-design modules (Monet; geometric-routing MoE)
- **Cite:** Monet, arXiv 2412.04139 (ICLR 2025): sparse dictionary learning inside SMoE pretraining, 262,144 experts/layer, parameters ∝ √experts; knowledge/language/toxicity manipulation "without degrading general performance". Geometric routing (cosine routing in low-dim metric space; rank-1 monosemantic experts), arXiv 2604.14434: temporal-expert steering +321% content probability, geographic suppression −23%, 15% of experts cleanly specialised, zero inference overhead. "Mixture of Experts Made Intrinsically Interpretable" (OpenReview, from search) **[not fetched]**.
- **Evidence:** S (Monet), P (geometric routing).
- **Failure modes:** from-scratch pretraining; expert specialisation is partial (15%); routing control is content steering, not a guarantee.
- **Serves:** partner plan's from-scratch pilot core ("sparse inter-layer communication"); not applicable to the converted 9B.

### 4.7 Backpack language models (sense-vector interface)
- **Cite:** Hewitt, Thickstun, Manning, Liang, arXiv 2305.16765 (ACL 2023); follow-up "Sense Representations Are Inducible Interfaces", arXiv 2605.28669 **[not fetched]**.
- **Evidence:** S.
- **Known:** each word = non-negative, context-dependent combination of a few non-contextual sense vectors; 170M Backpack matches GPT-2 small loss; sense editing gives global, predictable control (topic steering, debiasing).
- **Failure modes:** sense vectors are non-contextual (weak for compositional meaning); scaling beyond ~170M–1B not shown; not recurrent.
- **Serves:** LM project — a realiser-vocabulary control interface: lexical choice among licensed paraphrases could be a sense-vector weighting, auditable per word.

### 4.8 Codebook features (discrete bottlenecks at every layer)
- **Cite:** Tamkin, Taufeeque, Goodman, arXiv 2310.17230. Code: taufeeque9/codebook-features. Caveat paper: "The Interpretability of Codebooks in Model-Based RL is Limited", arXiv 2407.19532.
- **Evidence:** S; caveat P.
- **Known:** finetune with VQ bottlenecks per layer; hidden state = sum of a few discrete codes; loss close to or better than original with proper settings; control by activating codes.
- **Failure modes:** codes can be polysemantic / uninterpretable (the RL caveat); VQ training instability; a discrete bottleneck on a *recurrent state* is untested.
- **Serves:** partner plan's "discrete typed interface vs richer sparse concept representation" comparison — codebook on the semantic state is the discrete arm.

### 4.9 Intrinsically interpretable LMs at scale
- **Cite:** Guide Labs, "Scaling Inherently Interpretable Language Models" (Steerling-8B), arXiv 2608.07594 (Aug 2026); survey "Towards Intrinsic Interpretability of LLMs", arXiv 2604.16042 (ACL 2026; taxonomy: functional transparency, concept alignment, representational decomposability, explicit modularisation, latent sparsity).
- **Evidence:** P (single industry group; no third-party eval found).
- **Known:** claims "interpretability scales with capability rather than against it"; 8B competitive with peers trained on 2–16× compute; attribution to tokens, concepts and training data; closed-loop concept steering. Diffusion LM with causal mask.
- **Failure modes:** unverified by others; architecture details unclear from abstract; not recurrent.
- **Serves:** both as the existence proof that a concept-aligned design need not pay a tax; our Stage D Pareto map is the matched-comparison version of this claim.

### 4.10 Lab prior art to reuse
- Exact typed program support theorem (`chl-structured-models/CONTRACT.md`), exact decision-graph inference, known-λ submodels, source-separated vs modular comparison (−12.24 pp with a capacity-starved split) — the partner plan's "match useful capacity" rule comes from this.

---

## How we'd use this domain

**Decision model (converted 9B):** DAS (linear) on head inputs for probe-stability cards; CBM-style concept head on h_sem with MCBM regulariser and the four-part leakage test; completeness test on h_free; causal-tracing edit-vs-record test (`03` 3.4). No IIT on the frozen backbone (it changes the backbone); IIT can be applied to the *head* with the oracle decision program as the high-level model.

**From-scratch pilot core (partner plan, 100–400M):** three matched arms — unrestricted / modular / IIT-from-init with explicit records as aligned variables; IIA + prospective-prediction + leakage + utility, 3 seeds. Optional fourth arm: codebook bottleneck on the record state (discrete interface) vs sparse concept embedding (CEM-style).

**LM project:** typed-program AST nodes as IIT variables in the planner; Concept-DAS for discourse-only steering; Backpack-style sense weighting for lexical choice; CB-LLM concept detectors as an output-side safety check.

**Rules we adopt from the literature:** linear alignment maps only, reported complexity (4.3); alternative-circuit and output-only controls (`03` 3.7); every intervention claim at the counterfactual level needs the intervention, not a probe (2602.16698).

## Could not verify this session
- arXiv IDs for the original CBM paper and causal distillation; details of 2602.16612, 2605.28669, 2512.10805, "MoE Made Intrinsically Interpretable", intervention-aware CEM / leakage-poisoning papers.
- Any third-party evaluation of Steerling-8B.

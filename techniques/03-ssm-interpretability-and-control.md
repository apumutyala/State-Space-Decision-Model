# 03 — SSM interpretability and control

Written 2026-10-06. Same legend as `01`. The lab's own prior results (`reviews/03` §1.4: Gramians are per-layer LTI objects; the GDN2 switching counterexample; the certified cap's adherence tax) are assumed and not repeated.

## What the plan needs from this domain

Stage 0.6 / Stage D need cheap, stable per-checkpoint instrumentation on a *converted, frozen-ish 9B* (GDN layers + LoRA + head), plus a "glass-box conversion log" for layer swaps. The partner plan needs causal interventions on explicit state ("edit a fact, erase a source, clamp a relation … and predict the resulting decisions before running them"). This file lists what exists for Mamba/GDN/RWKV and what each can and cannot support.

**Headline:** attribution tools for Mamba are mature (MambaLRP, LaTIM, hidden-attention maps, causal tracing, ROME-style editing all work); **GDN/RWKV-7-specific tooling is thin and 2026-fresh** (WriteSAE is the first cache-level SAE/steering result on GDN/Mamba-2/RWKV-7); causal *localisation* in SSMs is non-unique (2606.00930), so every circuit claim needs alternative-circuit controls; Gramian/HSV/decay-spectrum tools apply per diagonal layer only — i.e. to Falcon-Mamba, not to Q9-AF's GDN layers — and are telemetry, not certificates (as the reviews already concluded).

---

## Techniques

### 3.1 Implicit attention matrices of Mamba / gated linear recurrences
- **Cite:** Ali, Zimerman, Wolf, "The Hidden Attention of Mamba Models", arXiv 2403.01590 (ACL 2025); unified implicit-attention formulation for gated linear recurrences, arXiv 2405.16504 **[not fetched]**.
- **Evidence:** R (used by MambaLRP, LaTIM, 2602.22719, G&A analysis).
- **Known:** the selective scan materialises as a data-dependent lower-triangular matrix per channel; attention-style explanations transfer; the same holds for GDN via its chunkwise matrix form.
- **Failure modes:** per-channel matrices are many; aggregation choices change the picture; "smooth" SSM attention (2504.18574) means sharp token-level claims are often unsupported.
- **Cost:** $0.
- **How we'd use it:** (i) locate G&A-like sharp rows in Q9's attention layers *and* check whether converted GDN layers acquire them (the "glass-box conversion log" metric); (ii) provenance check: fraction of implicit-attention mass from untrusted-channel tokens into the final state.

### 3.2 MambaLRP (conservation-corrected relevance propagation)
- **Cite:** Jafari, Montavon, Müller, Rezaei, arXiv 2406.07592, NeurIPS 2024. Code: FarnoushRJ/MambaLRP.
- **Evidence:** S; LaTIM independently confirms it is the strongest layer-wise aggregation method.
- **Known:** identifies Mamba components that break relevance conservation, reworks propagation there; SOTA faithfulness across models/datasets; cheap (one backward pass).
- **Failure modes:** Mamba-1/2 implementation; GDN/RWKV-7 require re-deriving the conservation rules; attribution ≠ causation.
- **Cost:** $0 for Falcon-Mamba; engineering for GDN.
- **Serves:** decision model (per-token evidence traces for the *backbone*; the integrator head is exact by construction per the plan).

### 3.3 LaTIM (token-to-token interaction decomposition for Mamba-1/2)
- **Cite:** Pitorro & Treviso, arXiv 2502.15612, ACL 2025. Code: deep-spin/latim.
- **Evidence:** S.
- **Known:** decomposes outputs into per-source-token contributions (ALTI-style); reveals copy diagonals; MambaLRP beats it on layer-wise aggregate faithfulness, LaTIM is better for *pairwise* interaction maps.
- **Serves:** decision model (option↔record token interaction maps for the fork heads).

### 3.4 Causal tracing, ROME editing, knockout in Mamba
- **Cite:** Sen Sharma, Atkinson, Bau, "Locating and Editing Factual Associations in Mamba", arXiv 2404.03646; Endy et al., "Mamba Knockout for Unraveling Factual Information Flow", arXiv 2505.24244 (ACL 2025).
- **Evidence:** R (two groups; findings mirror transformer results).
- **Known:** mid-layer components at the subject's last token carry facts; later layers act at the last prompt token; ROME works best on W_o projections; knockout of token-to-token interactions traces flow; some patterns are architecture-general.
- **Failure modes:** localisation claims are non-unique (3.7); editing success does not imply the semantic-record interface is faithful.
- **Cost:** $.
- **How we'd use it:** (i) partner plan's "edit a fact and predict the decision" test = ROME on the backbone vs edit on the explicit record — the explicit record must win or the record is decorative; (ii) **injection persistence curve**: knock out the untrusted span at distance τ from the decision and plot flip-rate vs τ per layer (the plan's replacement for the half-life bound).

### 3.5 Circuit discovery in Mamba (EAP / ACDC)
- **Cite:** Ensign & Garriga-Alonso, "Investigating the Indirect Object Identification circuit in Mamba", arXiv 2407.14008 (FAR AI).
- **Evidence:** S.
- **Known:** positional edge-attribution patching and ACDC adapted to Mamba; a single layer (39) is the IOI bottleneck.
- **Serves:** decision model (template for circuit-level glass-box claims about the head/backbone interface).

### 3.6 Activation-subspace bottleneck steering (Stable-Mamba)
- **Cite:** "Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks", arXiv 2602.22719 (Feb 2026).
- **Evidence:** P.
- **Known:** project hidden states onto implicit-attention-weighted vectors; facts and state updates concentrate in few delta-sensitive subspaces; a **scalar multiplier at the bottleneck gives +8.27% average across 5 SSMs × 6 benchmarks** with no task tuning; Stable-Mamba (256 new params) +15.3 pp RULER QA, +35 pp LRA PathFinder.
- **Failure modes:** unreplicated; why a scalar gain helps is unexplained (could be an entropy/temperature effect).
- **Cost:** $.
- **How we'd use it:** cheapest "control" probe: if a scalar at a named subspace moves the Index retrieval area, that subspace is where provenance gating should act; also a candidate explanation for why conversions lose retrieval (bottleneck entropy).

### 3.7 Non-unique causal localisation (the state-sink case)
- **Cite:** Jiang & Zhang, "A Circuit, Not The Circuit", arXiv 2606.00930 (May/Aug 2026).
- **Evidence:** P, but the methodological point is general.
- **Known:** Mamba-2 "state sink" (Δ-gate firing on BOS/newline); a nearly disjoint set of heads gives equal-or-larger ablation effects than the probe-identified specialists; 25–35% of heads are "dual" (respond alike, different causal profile); perturbing Δ costs ≤0.16 nats while suppressing the same heads' output costs 1.5–7.3 nats.
- **How we'd use it:** every "we found the mechanism" claim in Stage D reports (a) an alternative-circuit search, (b) the intervention surface used, (c) probe-vs-ablation dissociation. This matches the partner plan's "random and output-only controls".

### 3.8 Sparse autoencoders on recurrent state and writes
- **Cite:** Young, "WriteSAE: Sparse Autoencoders for Recurrent State", arXiv 2605.12770 (May 2026); "Comparing Latent Concept Formation in SSMs and Transformers via SAEs", arXiv 2609.24440 (Mamba-130m vs Pythia-70m, 10M tokens, "dark matter" features); Persistent SAEs, arXiv 2607.17117 (feature-specific timescales; slow features preserve prompt-injection signal); Concept-Bottleneck SAEs, arXiv 2512.10805 **[not fetched]**.
- **Evidence:** P each.
- **Known (WriteSAE):** rank-1 matrix atoms shaped like the model's own write to the GDN/Mamba-2/RWKV-7 cache; *replacement test* (swap the model's write for the scaled atom) beats deletion on 92.4% of positions; for GDN a closed-form (forget gate × read query × output embedding) predicts logit change with **R² = 0.98**; writing a direction into 3 cache positions at 3× norm makes rank-100–1000 tokens appear in 100% of continuations (from 33%); transfers to Mamba-2-370M at 88.1%. "First cache-level steering intervention in an SSM/hybrid recurrent layer."
- **Failure modes:** small models; no downstream-task steering; SAE features are not guaranteed causal (3.7); dictionary choice.
- **Cost:** $–$$ (SAE training on 9B GDN writes over ~100M tokens is a few GPU-hours).
- **How we'd use it:** (i) provenance audit — which write-atoms fire on untrusted tokens and whether they persist (Persistent-SAE timescales); (ii) "concept-supervised block" vs SAE dictionary comparison for h_sem (partner plan); (iii) the GDN logit-change formula is a free *predict-before-intervene* test.

### 3.9 Controllability/observability-style influence scores
- **Cite:** Mabrok & Zafari, "Controllability Analysis of State Space-based Language Model", arXiv 2511.17970 (Influence Score via backward recurrence; mamba-130m/2.8b; recency bias; mid-to-late layer concentration); Sparse Mamba (controllability/observability-structured SSM), arXiv 2409.00563 **[not fetched]**; Context-Selective SSMs with feedback, arXiv 2510.14027 **[not fetched]**.
- **Evidence:** P.
- **Known:** a per-token score of how strongly a token affects later states/outputs, computed from discretised parameters (observability-like backward pass); scale-dependent.
- **Failure modes:** trajectory-conditioned (LPV with frozen schedule), not a certificate — exactly the review's A2 caveat.
- **How we'd use it:** telemetry column: influence score of untrusted-channel tokens vs trusted; expect injection-augmented LoRA to lower it.

### 3.10 Gramians, Hankel singular values, in-training truncation (diagonal arm only)
- **Cite:** CompreSSM, arXiv 2510.02823 (ICLR 2026; in-training balanced truncation on LRU); HSVR, arXiv 2510.22951 (NeurIPS 2025; regularises ΣHSV); lab code `mechanistic-ssm-control/core/gramian.py`. From `reviews/02` §5: formulas verified to 1e-15; H∞ bound verified; H2 cost of modal truncation is |c_i b_i|²/(1−|λ_i|²).
- **Evidence:** R for the math; S for the training use.
- **Scope:** diagonal LTI per layer, or selective-diagonal (Mamba-1 / Falcon-Mamba) with the Δ schedule frozen per trajectory. **Not applicable to GDN** (matrix state with data-dependent rank-1 updates) or RWKV-7.
- **How we'd use it:** Falcon-Mamba arm only: per-layer HSV spectra as a compressibility/telemetry card; CompreSSM-style truncation as a Stage D "cheaper layer" swap with measured Δscore.

### 3.11 Decay-spectrum analysis and spectral initialisation
- **Cite:** Mamba Modulation (length generalisation via spectrum of the transition matrix), arXiv 2509.19633; Cao, "Optimal Decay Spectra for Linear Recurrences" (PoST: geometric log-decay spacing is minimax-optimal; position-adaptive scaling; gains on Mamba-2/RWKV-7/GDN/GLA/RetNet at 180–440M), arXiv 2604.07658; Universality of linear recurrences, arXiv 2506.11891; Stuffed Mamba outlier channels, arXiv 2410.07145; lab's `RETENTION_ABLATION.md` (+17 pp from decay-bias init).
- **Evidence:** P (2604.07658), S (others); the lab's own decay-init result is R-internal.
- **Known:** random decay init collapses the spectral gap (O(N⁻²)); linear spacing is exponentially worse than geometric; learned spectra drift; per-token effective decay |λ|(x_t) histogram per layer is the natural "memory horizon" card.
- **How we'd use it:** card column = per-layer histogram of per-token log-decay (write/retained-norm telemetry already designed in `bounded-intelligence` SSD kernel); flag layers whose slowest mode cannot span the record length (review §6: 16K needs |λ| ≥ 0.99996); for the from-scratch pilot, PoST-style geometric init is a free improvement.

### 3.12 Steering recurrent LMs (beyond 3.6, 3.8)
- No Mamba-specific steering-vector paper beyond 3.6 and WriteSAE was found this session; generic residual-stream steering (ActAdd/CAA, from memory **[unverified IDs]**) applies unchanged to the residual stream of any architecture. Concept-DAS steering (2602.05234, `04`) is architecture-agnostic.

### 3.13 "Mamba lens" tooling
- **Could not verify** a tool named "Mamba lens" / "MambaLens". nnsight and pyvene (2403.07809) expose hooks generically; pyvene's model-type coverage for Mamba/GDN was not checked **[unverified]**. Plan for custom hooks on flash-linear-attention modules.

---

## How we'd use this domain (per-checkpoint instrumentation card, Stage 0.6)

| Column | Tool | Cost | Caveat |
|---|---|---|---|
| G&A sharpness per layer | implicit attention (3.1) | $0 | smooth-attention artefact |
| Per-token evidence trace (backbone) | MambaLRP / LaTIM (3.2–3.3); exact for integrator head | $0 | attribution ≠ cause |
| Fact localisation + edit-vs-record test | causal tracing / ROME (3.4) | $ | non-unique (3.7) |
| Injection persistence curve | knockout at distance τ (3.4) | $ | replaces half-life bound |
| Provenance write audit | WriteSAE + Persistent-SAE timescales (3.8) | $–$$ | P-grade tooling |
| Control probe | bottleneck scalar steering (3.6) | $ | unexplained mechanism |
| Memory horizon | decay histogram (3.11) | $0 | GDN: use retained-norm, not eigenvalues |
| Compressibility (diagonal arm) | HSV spectra (3.10) | $0 | per layer, LTI only |
| Influence of untrusted tokens | influence score (3.9) | $0 | trajectory-conditioned |

Report every causal claim with the 3.7 controls. Nothing here is a certificate; the reviews' verdicts stand.

## For the LM project
Same card on the shared backbone; add WriteSAE atoms on the planner's state during realisation to check the realiser reads only the checked program (partner plan's "no direct latent-to-decision bypass" = ablate the backbone→realiser path and show zero effect beyond the program).

## Could not verify this session
- 2405.16504, 2409.00563, 2510.14027, 2512.10805 details; pyvene/nnsight Mamba/GDN support; generic steering paper IDs.

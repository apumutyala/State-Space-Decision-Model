# 02 — Technical pressure test of Doc A (v1 plan) and Doc B (v2 critique)

Date: 2026-10-05. Input: `reviews/00-INPUT-v1-plan-and-v2-critique.md`.
Stance: adversarial technical review (control theory, SSMs, Bayesian/conformal statistics, singular learning theory, constrained decoding).
Method: every technical claim was either (a) checked numerically with small numpy experiments (scripts in `C:\Users\apumu\AppData\Local\Temp\claude\fable-tech\`: `gramian_hsv.py`, `memory_depth.py`, `bayes_integrator.py`, `rlct_mc.py`, `rlct_refined.py`, `ece_cbf.py`; raw outputs in Appendix A), or (b) checked against the literature (Appendix B lists what was verified and what was not).

Verdict legend: **SOUND** / **OVERSTATED** / **FLAWED** (fixable) / **WRONG** (drop or replace).

---

## 0. Ranked summary (most consequential first)

| # | Claim | Verdict | One-line correction |
|---|-------|---------|---------------------|
| 1 | Doc B: "calibration is a property of the architecture" (integrator bank = Bayes filter) | **WRONG** | The normalizer is handled by the architecture; the *increments* are not. With redundant evidence a stateless increment is provably length-miscalibrated (sim: calibration slope 0.21 at T=16 for the naive sum; the log-loss-optimal stateless increment is slope 1.65 at T=4 and 0.77 at T=64). Keep Brier + post-hoc per-question temperature. |
| 2 | Doc A: forked-continuation scoring is permutation-equivariant and adequate | **FLAWED** | Equivariant only if the shared prefix is option-free (Doc A's own "questions first" ordering breaks it, since Jev/Clef questions carry their option lists). Doc B's "lossy" critique is correct and is the measured SSM recall gap; it lands on ~38 % of the Index (Retrieval & Classification 20 %, Tools & Automation 18.3 %). Clef's head attends over *all* final hidden states and is not subject to it. |
| 3 | Doc A: "100 % of decision-relevant info provably passes through h_sem" | **WRONG (vacuous)** | True of the input to every classifier's last layer. Interpretability needs concept supervision + leakage tests, or SAE/causal probes; the "provable" part is only the cut. |
| 4 | Doc A: LPV half-life bound = injection-persistence certificate; Doc B: "memory vs safety" tension | **WRONG as a safety claim; Doc B's critique is correct and understated** | A 16 K-token memory needs per-step |λ| ≥ 0.99996 ⇒ bound ≥ 16 K tokens; depth multiplies persistence (16-layer cascade with per-layer half-life 69 peaks at τ≈1484). And half-life is the wrong property: an injection acts at τ≈0. Replace with provenance-gated writes + an adversarial-record gate. |
| 5 | Doc B: λ ≤ r + (n−r)/2 for SISO diagonal LTI; Doc A: "RRR-like λ(H=n_state, r=Hankel rank)" | **Doc B SOUND for real modes (MC-verified: 0.5/1.0/1.5/2.0); Doc A WRONG** | For complex-diagonal (LRU/S5, Re readout) each mode is a conjugate pair: used = 2, unused = 1 ⇒ λ = n + r (jumps of 1, not ½). The RRR formula does not apply: the Hankel/shift structure gives coordinate singularities {p_i = 0}, not BA-factorization singularities. |
| 6 | Doc B: truncation threshold σ < √(log n / 8n) | **FLAWED** | Conflates balanced and modal truncation (modal H2² cost is exactly \|c_i b_i\|²/(1−\|λ_i\|²), unrelated to HSVs); 4σ² is a ~10× loose bound; the RLCT saving is 1 (real) / 2 (complex) for a genuinely used mode and only ½ / 1 for a spurious one; "n" is used for both state size and sample size. Corrected rule in §10. |
| 7 | Doc A: closed-form Gramians, HSV bound 2Σσ; Doc B: "don't compose" | **SOUND per layer; Doc B SOUND** | Formulas verified to 1e-15; H∞ bound verified (observed ratio 0.5–0.7). What composes is contraction/ISS/Lipschitz, not Gramians; stack certificate given in §5. |
| 8 | Doc A: discrete-CBF closed-form projection on affine barrier in h_sem | **SOUND math, OVERSTATED guarantee** | Guarantees forward-invariance of a half-space in h_sem only. With readout w ⟂ a the projection changes the logit by exactly 0. For a schema-constrained decision model, output-space masks/caps are exact and free; CBF-in-state is a solution looking for a problem. |
| 9 | Doc A: conformal "act iff \|C(x)\|=1", CRC/LTT, Mondrian | **FLAWED** | P(error \| singleton) ≠ α; for binary online CP it is (ε − P(empty))/P(singleton) (arXiv 2506.21802). Abstentions score as wrong on the Index. Zero-shot across 38 benchmarks breaks exchangeability; use weighted/robust CP or online ACI in deployment only. |
| 10 | Doc A: PID-Lagrangian on ECE with setpoint 0.02 | **FLAWED** | 15-bin ECE of a *perfectly calibrated* model is 0.057 ± 0.015 at batch 256, 0.027 ± 0.008 at 1024; the setpoint is below the estimator's noise floor until ≥ 4 K samples. Constraint is trivially satisfiable by the marginal predictor. Post-hoc per-question temperature scaling does the job. |
| 11 | Doc A: "hard grounding theorem" | **SOUND as a syntactic lemma; OVERSTATED as "no hallucination"** | Covers only atomic-claim membership in K_plan given injective grammar semantics. Not covered: selection, omission, composition ("because", juxtaposition), scope/negation, implicature, aggregation, correctness of K. |
| 12 | Doc A: SNGP "almost free" from contracting SSM layers | **WRONG** | SNGP needs the *lower* Lipschitz bound (distance preservation). Contraction = forgetting = many-to-one along time: the opposite. Only per-position residual-stream bi-Lipschitz is available. |
| 13 | Doc B: first experiment = frozen attention-free backbone + integrator head + adapters; 10-point stop rule | **Right direction, under-specified** | It is literally Clef's recipe (frozen backbone + rank-256 LoRA + head) on an SSM. Needs a backbone × head factorial with Clef's cross-attention head as control, and an *attribution* rule (per-area, per-length) instead of a 10-point rule. LoRA on a 7B SSM costs ≈ $175/run; freezing is for attribution, not cost. |

---

## 1. Forked-continuation scoring

**Claim (Doc A).** Cache h* after `[schema; state; question]`; run each option description from h* in parallel; z_k = wᵀφ(h_sem*, k) + α·lex_k + b; permutation-equivariant, schema-valid, O(K). **Doc B:** lossy because the state is compressed before the options are read.

**Permutation equivariance — conditionally SOUND.** If the shared prefix contains no option text and every option is scored by the same function from the same h*, the map (o_1..o_K) ↦ (z_1..z_K) is elementwise over a set, hence exactly equivariant, and softmax preserves this. It breaks when (i) the prefix lists options — and in the Jev/Clef contract a "question" *is* its option list, so Doc A's "questions, state, questions again" ordering as written is **not** equivariant; (ii) the >255 two-stage variant feeds the top-m sequentially; (iii) option descriptions share tokens with the question text (minor). The deeper point: equivariance and option-awareness pull in opposite directions in a single causal pass. If options are kept out of the prefix (for equivariance) the state cannot know what to retain; if they are put in (for retention) order-dependence enters and must be trained away by permutation augmentation (which is what Clef does with "permutating field orders").

**"Lossy" — SOUND, and the severity is known.** h* is a fixed-size summary: per layer d_inner × N scalars (Falcon-Mamba-7B: 64 × 8192 × 16 ≈ 8.4 M scalars) versus a 64 K-token input (× d_model 4096 ≈ 2.7 × 10⁸ scalars): ≥ 30× compression before any option is seen. This is the measured recall gap of fixed-state models: Jelassi et al. 2024 ("Repeat After Me") — a 410 M Pythia beats a 2.8 B Mamba on phone-book lookup beyond ~70 entries; Waleffe et al. 2024 (NVIDIA, 8 B scale) — pure Mamba/Mamba-2 lag Transformers on 5-shot MMLU, phone-book and long-context reasoning while hybrids do not. On the Decision Index 0.2.1 the areas most exposed are Retrieval & Classification (20.0 %) and Tools & Automation (18.3 %, BFCL-style extraction of many fields) — about 38 % of the headline number.

**Versus Clef's head.** Clef: "a small transformer head that reads the backbone's final hidden states, routes evidence from the state to each question, and scores all options of all questions jointly" — i.e. random access to all L positions through cross-attention in the head (O(L·d) memory, O(Q·K·L) head compute, negligible next to the backbone). The fork has access to one state. The two are not comparable at equal backbone quality on retrieval-type questions; they are comparable on short-utterance classification (BANKING77, CLINC).

**Alternatives and costs** (L context tokens, Q questions, K options, ℓ tokens per option, S snapshots, n state scalars):

| Design | Sequential cost | Extra memory | Equivariant? | Fixes recall gap? |
|---|---|---|---|---|
| Fork from final state (Doc A) | L + ℓ (K parallel) | K·n | yes (if prefix option-free) | no |
| Options-first + integrator (Doc B) | Q·K·ℓ + L (all options prepended) | O(1) | no (needs permutation aug.) | partly — state knows what to look for (JRT effect, Arora et al. 2024) |
| Read twice: [state][questions][state] with integrator in pass 2 | 2L + QKℓ | input buffer | no | yes-ish (JRT-Prompt evidence) |
| Snapshot forks: fork each option from S chunk-boundary states, aggregate by logsumexp/max | L + ℓ (S·K parallel) | S·n | yes | yes for local evidence; attention-free |
| Option-conditioned re-scan of top-m | L + Kℓ + m·L | O(1) | yes | yes, expensive |
| Bidirectional SSM (Hydra/Vim-style) + per-token option scorer | 2L | L·d | yes | yes (this is Laya's regime, ModernBERT) |
| Clef-style cross-attention head over final hidden states | L + QK·L (head) | L·d | yes | yes |

**Recommendation.** Treat "no attention anywhere" as a cost, not a feature. Default arm: snapshot forks (S ≈ L/512) with the backbone attention-free; control arm: a 1–2 layer cross-attention head over the final hidden states (Clef's design). Measure both per Index area and as a function of record length. Before promising "constant-memory streaming" as a win axis, measure the record-length distribution of the Index — if the median record is ~1 K tokens the axis is irrelevant to the benchmark.

---

## 2. "Head reads only h_sem ⇒ 100 % of decision-relevant information provably passes through the semantic block"

**VERDICT: WRONG (vacuous).** Every network has a cut through which all decision-relevant information provably passes: the input to the last layer. Zeroing head weights on h_free makes h_sem that cut; it says nothing about what h_sem *means*. Doc B does not catch this.

**What would make h_sem genuinely interpretable.**
- *Concept supervision* (CBM; Koh et al. 2020): h_sem = predicted concepts with labels. Known failure: **leakage** — soft concept activations encode input information beyond the concept semantics (Mahinpei et al. 2021; Margeloiu et al. 2021), so the head uses the leaked signal and interventions become unreliable. Hard/binarised concepts reduce leakage at an accuracy cost; Havasi et al. 2022 add a side channel + autoregressive concept predictor to manage it. In a sequence model the leakage channel is wider: h_sem is a *trajectory*, so any bit pattern over time is a side channel.
- *Leakage test* (mandatory if the claim is kept): train a probe from h_sem to the label with concepts masked; if it beats the concept-only predictor, h_sem leaks.
- *Causal probes*: an intervention on a named direction in h_sem must change the decision as predicted; correlation-only probes are not interpretability.
- *Completeness*: ablating h_free must leave accuracy unchanged, otherwise h_free is not "free" and the glass box has a back door.
- *Cheaper, backbone-agnostic* (fits Doc B's frozen-backbone plan): MambaLRP attribution (NeurIPS 2024) and "activation subspace bottleneck" steering in Mamba (arXiv 2602.22719, scalar steering at identified bottleneck activations) — these work on pretrained SSMs without a concept bottleneck.

**Recommendation.** Rewrite the claim as "the head is restricted to a concept-supervised bottleneck; leakage ≤ X % by test T". Keep h_free. Report leakage numbers alongside accuracy.

---

## 3. Bayes-filter / integrator-bank equivalence (Doc B)

**The identity is correct.** log p(k | x_{1:t}) = log p(k | x_{1:t−1}) + log p(x_t | k, x_{1:t−1}) − log p(x_t | x_{1:t−1}). The last term is k-independent, so softmax_k over s_k(t) = log p(k) + Σ_τ log p(x_τ | k, x_{<τ}) is the exact posterior **with no independence assumption — provided the increment is conditioned on the whole past**. That is the whole content: the architecture handles the normalizer; it does not make the increments correct.

**Where it fails.** If the increment is a function of x_t only (or of a representation that has forgotten the past), s_k is naive Bayes and is exactly the posterior only under conditional independence given k. Redundant evidence (restated facts, repeated tokens, paraphrases) is double-counted ⇒ overconfidence growing with length.

**Numerical check** (`bayes_integrator.py`; binary k, 4-symbol vocabulary, each token copies the previous one with prob. 0.7, else is a fresh class-conditional draw):

| T | exact posterior ECE | naive sum of *true* per-token LLRs: ECE / cal-slope | log-loss-optimal *stateless* increment (fit on T∈{4,16,64}): ECE / slope | increment with 1 token of memory: ECE / slope |
|---|---|---|---|---|
| 4 | 0.006 | 0.148 / 0.37 | 0.087 / 1.65 | 0.006 / 1.00 |
| 16 | 0.005 | 0.138 / 0.21 | 0.013 / 0.94 | 0.005 / 0.99 |
| 64 | 0.002 | 0.043 / 0.17 | 0.014 / 0.77 | 0.002 / 0.99 |

Log-loss training shrinks the stateless increment (fitted ≈ 0.23 × true LLR) and gets calibration right on average, but the optimal scale depends on T: under-confident for short, over-confident for long inputs. One token of memory restores exactness here; real redundancy needs more.

**Does log loss guarantee calibration?** Only that the population minimiser is the best-in-class in KL. The minimiser is calibrated iff the class is closed under the needed recalibration maps. A stateless-increment integrator is closed under a *global* temperature, so it can be calibrated for one evidence-length distribution, not across lengths. Hence: **"calibration is a property of the architecture" is WRONG**; what is true is "the architecture makes calibration *possible* if the increments are past-conditioned and sharp".

**Relation to Mamba's eigenvalue-1 modes.** Mamba: A = −exp(A_log) < 0, λ = exp(ΔA) with Δ = softplus(·) > 0, so |λ| < 1 strictly. λ → 1 requires Δ → 0, which also kills the write (ZOH gain ≈ ΔB), or A → 0⁻, a boundary of the parametrisation. So an exact integrator is approached, never represented, and "retain and write in the same channel at the same step" is impossible. Halloran et al. (arXiv 2406.00209, Thm 1) prove the maximal Lyapunov exponent ≤ 0 (non-expansive) — this *permits* marginal (λ=1) modes; it is not a forgetting guarantee. For the head, set λ := 1 structurally (it is sum-pooling of option-conditioned per-token scores) and let the backbone do the past-conditioning.

**Options-before-state.** Required if the backbone must compute option-conditioned increments. The late-fusion alternative Δs_k(t) = ⟨φ(x_t, past), ψ_k⟩ allows options anywhere and is exactly equivariant, but Σ_t collapses it to ⟨Σ_t φ_t, ψ_k⟩ — the pooled vector is again a fixed-size summary (same bottleneck as the fork). Max/logsumexp over t avoids this but needs either options-first (running max per option: O(K) state) or stored φ_t (O(L): Clef). This trade-off — *know the options in advance, or store O(L) features* — is fundamental and should be stated in the plan as such.

**Recommendation.** Keep the integrator head but (i) make increments come from the selective backbone (never from a context-free token embedding), (ii) feed the per-question temperature a function of evidence count/length, (iii) keep Brier + label-smoothing in training and post-hoc per-question temperature, (iv) evaluate ECE *per length bin*.

---

## 4. Memory bound Σ|c_i b_i||λ_i|^τ and the "n scalars" claim

**Single diagonal LTI layer, scalar channel — SOUND and tight.** y_t = Σ_τ g_τ x_{t−τ}, g_τ = Σ_i c_i b_i λ_i^τ, so |∂y_t/∂x_{t−τ}| = |g_τ| ≤ Σ_i |c_i b_i||λ_i|^τ. Verified: max_τ |g_τ|/bound = 0.996.

**Deep nonlinear stacks — not of that form.** With per-layer modal radius ρ and 1-Lipschitz nonlinearities the influence of a token at distance τ is bounded by the cascade: Σ_{τ_1+…+τ_L=τ} Π_l g^{(l)}(τ_l) ≤ C(τ+L−1, L−1) ρ^τ × gains — polynomial × exponential. Numerically (`memory_depth.py`, ρ = 0.99, single-layer half-life 69 steps): a 4-layer cascade peaks at τ ≈ 296 and halves 252 steps later; 16 layers peak at τ ≈ 1484 and halve ~500 steps later. Depth multiplies memory and injection persistence. Residual paths (delay 0) exist too but carry no memory.

**Selective SSMs.** λ_i(x_t) is input-dependent; the bound becomes Σ|c_i b_i| Π_s |λ_i(x_s)| and the model can choose Π ≈ 1 over arbitrary spans (Δ → 0) — no a-priori bound without clamping Δ (item 6).

**"n-dim linear state holds at most n scalars".** True per channel; the actual state is Σ_layers d_inner × N (Mamba-1 7B ≈ 8.4 M; Mamba-2 at 64 heads × 64 × 128 ≈ 0.5 M per layer × layers). Against 64 K × d_model input scalars that is ~30× compression, not "n". The qualitative conclusion — store only what known questions need — stands; fix the quantitative phrasing.

---

## 5. Gramians, HSV error bound, compositionality

**Formulas — SOUND.** For x_{t+1} = Λx_t + Bu_t, y = Cx: P_ij = (BB*)_ij/(1 − λ_i λ̄_j), Q_ij = (C*C)_ij/(1 − λ̄_i λ_j) (`gramian_hsv.py`: max deviation from a fixed-point Lyapunov solve 1.8e-15 / 4.5e-16). HSVs σ_i = √eig(PQ). Balanced-truncation bound ‖G − G_r‖_∞ ≤ 2 Σ_{i>r} σ_i (distinct σ_i counted once): verified, observed error/bound = 0.51–0.73.

**Two corrections.**
1. Complex-diagonal with real readout y = Re(Ch) is a *real* system of order 2n with 2n HSVs (computed: they are not the n complex-system HSVs). Count accordingly for LRU/S5.
2. The white-noise MSE (H2²) cost of balanced truncation was 0.07–0.14 × (4σ²): the "4σ²" figure (from (2σ)² and H2 ≤ H∞) is a valid but ~10× loose upper bound. Modal truncation (dropping a diagonal mode) costs exactly |c_i b_i|²/(1−|λ_i|²) and is unrelated to the HSVs (example: smallest HSV 0.0025 vs smallest modal cost 0.076). Doc B's truncation rule mixes the two (§10).

**Compositionality — Doc B is right.** Gramians are an LTI input-output notion; there is no "stack Gramian" through GLU/MLP mixing. What *can* be certified for a deep stack:
- *Incremental ℓ2 gain / Lipschitz*: per diagonal-LTI layer γ = ‖G‖_∞ exactly; per static block via spectral norms or LipSDP; stack gain Π γ_l (valid, loose).
- *Contraction / ISS*: each LRU layer is contracting with rate max|λ_i|; a cascade of contracting layers and Lipschitz static maps is input-to-state stable with the polynomial × exponential forgetting bound of §4 — this is the honest stack-level "injection persistence" certificate. REN (Revay–Wang–Manchester) gives direct parametrisations of contracting, Lipschitz-bounded nonlinear recurrences (Doc A's rung 4). Zubić & Scaramuzza (arXiv 2505.11602) give passivity/ISS conditions for selective SSMs with discontinuous gating (exponential forgetting under state-strict dissipativity with the selection schedule frozen) — the right reference for the clamped-Δ LPV rung.
- *Finite-horizon verification* (IBP/CROWN-style) for specific prompts — not a design-time certificate.
- Per-layer Gramians remain the right tool for *compression* (CompreSSM, ICLR 2026, arXiv 2510.02823, uses exactly in-training balanced truncation on LRU; HSVR, NeurIPS 2025, arXiv 2510.22951, regularises Σσ_i) and as a diagnostic.

---

## 6. LPV per-mode half-life bound; "memory vs safety"

**Bound — SOUND for one layer's modal contribution.** With λ_i(t) = exp(Δ_t a_i), Δ_t ∈ [Δ_min, Δ_max]: |λ_i(t)| ≤ exp(−Δ_min|Re a_i|) ⇒ the mode's contribution halves within ln2/(Δ_min|Re a_i|) steps.

**Why it is useless as a safety certificate (Doc B's critique is correct and if anything understated).**
1. *Vacuous in the useful regime*: to carry anything across 1 K / 16 K / 64 K tokens the slowest mode needs Δ·|Re a| ≤ 6.8e-4 / 4.2e-5 / 1.1e-5 per step (|λ| ≥ 0.9993 / 0.99996 / 0.99999), so the "injection half-life" bound is ≥ 1 K / 16 K / 64 K tokens.
2. *Depth*: the stack bound is the cascade of §4 — the influence does not even peak until ≈ (L−1)ρ/(1−ρ) steps.
3. *Coupled gain*: in Mamba the write gain scales with Δ, so clamping Δ_min forces a minimum write rate for every token.
4. *Wrong property*: an injection acts at τ = 0…few. Bounding its decay bounds how long a burglar stays, not whether he gets in. For a schema-constrained decision model the only injection effect is "wrong option" — a robustness property, measured, not certified.

**Alternatives.**
- *Two mode populations* (fast: certified half-life ≤ H; slow: long memory) — useful as a *memory architecture* and for compressibility; not a safety story.
- *Provenance/channel separation*: a provenance bit per token (trusted schema/question vs untrusted record/tool output) gates the write matrix: B_slow(x_t) := 0 for untrusted tokens. This is a structural, inspectable property ("untrusted text cannot write long-term state"), but note that in a decision model the untrusted record *is* what must be remembered — so the only thing provenance gating can protect is question/schema state, which the output contract already fixes.
- *Input-gated reset* at trusted delimiters: certified "forget the untrusted span" for the fast population.

**Recommendation.** Drop the half-life narrative. Keep fast/slow populations for memory and compression. Replace the safety claim with an adversarial-record gate (injected instructions, contradictory restatements, option-name poisoning) with measured flip rates.

---

## 7. Discrete CBF closed-form projection on an affine barrier in h_sem

**Math — SOUND.** h(x) = aᵀx + b, DCBF condition h(x⁺) ≥ (1−γ)h(x). With x⁺ = x̂ + u the minimum-norm u is u* = max(0, (1−γ)h(x) − h(x̂))/‖a‖² · a (`ece_cbf.py`: constraint met with equality, min-norm verified on random feasible perturbations).

**What it guarantees.** Forward invariance of the half-space {h ≥ 0} for the projected state (and exponential return if violated) — for one affine constraint this is clipping along a.

**What it does not guarantee.**
- Anything about the decision unless the readout has a component along a: with w ⟂ a the logit shift is 0 (measured −2e-16) while the barrier is "satisfied". The barrier is a probe direction; probes are correlational, and the subsequent layers are nonlinear, so an affine barrier in h_sem is not an affine barrier at the output.
- Multiple constraints need a QP and can be infeasible; a learned nonlinear barrier (BarrierSteer, arXiv 2602.20102) reports attack-success-rate reductions, not guarantees.
- Model mismatch: the "dynamics" f are the network itself, known exactly, but the *meaning* of the half-space is empirical.

**Recommendation.** For the decision model, constrain the *output*: hard masks on disallowed options and caps on p(unsafe option) are exact, free, and need no probe. Reserve CBF filtering for the generative LLM project, where the control is the steering vector and the state is the residual stream, and benchmark against BarrierSteer's protocol.

---

## 8. Conformal prediction

- **"Act iff |C(x)| = 1" — FLAWED.** Marginal coverage ≥ 1−α does not bound the error rate on singletons. For binary online smoothed CP the singleton error rate is σ = (ε − P(empty))/P(singleton) (arXiv 2506.21802, Prop. 2; the paper notes Linusson 2016 / Bortolussi 2019 formulas are only approximately right in the offline inductive setting). Example: ε = 0.05, P(empty) = 0.01, P(singleton) = 0.5 ⇒ 8 % error among acted-on cases. For K-way sets the singleton subset is not calibrated to any level without extra machinery; use selective CRC/LTT with the selective risk as the controlled loss (Angelopoulos et al.; "Selective Conformal Risk Control", arXiv 2512.12844) and report error–reject curves.
- **Index scoring — abstention is strictly harmful.** Harness README: "Unanswered, unsupported, errored and abstained requests count as wrong"; raw = native score × answered/requests, then chance-corrected. The act/abstain gate is a deployment feature only.
- **Exchangeability.** Zero-shot across 38 counted benchmarks with fixed rendering means there is no in-distribution calibration data; split CP calibrated on synthetic schemas guarantees coverage on the synthetic distribution. Mondrian per schema family helps only where the family matches. Options: weighted CP under covariate shift with estimated density ratios (Tibshirani et al. 2019), robust CP under bounded f-divergence shift (Cauchois et al.), online adaptive CP (Gibbs & Candès 2021) in deployment.
- **Recommendation.** For the Index: calibrated probabilities only. For deployment: LTT on selective risk with Mondrian strata = schema family, and publish the error–reject curve.

---

## 9. PID-Lagrangian on ECE

- **Estimator noise** (`ece_cbf.py`, 15 equal-width bins, perfectly calibrated binary model, conf ~ U(0.5, 1)): batch 128 → ECE 0.079 ± 0.021; 256 → 0.057 ± 0.015; 1024 → 0.027 ± 0.008; 4096 → 0.014 ± 0.004; 20 000 → 0.006 ± 0.002. The setpoint 0.02 is below the noise floor until ≥ 4 K samples; a minibatch-driven multiplier will wind up. Binned ECE is also biased, worst for calibrated models (Roelofs et al. 2022); use equal-mass bins or the debiased estimator (Kumar, Liang & Ma 2019).
- **Degeneracy.** {ECE ≤ 0.02} contains the marginal predictor; the Lagrangian trades sharpness for calibration and at equilibrium is a constant re-weighting of the Brier/MMCE surrogate — i.e. a regulariser, not a constraint with a guarantee.
- **Versus post-hoc.** Temperature scaling (Guo et al. 2017) reaches ECE ≈ 0.01–0.03 with one held-out parameter and zero training interference; Clef reports a low ECE with LS-CE + Brier alone (Doc A's 0.023 for AutoJev-27B is unverified). A per-question temperature predicted from the schema embedding, fitted post-hoc, is the cheap actuator and also absorbs the length-dependence of §3.
- **Recommendation.** Drop the PID loop. Train with LS-CE + Brier (Clef's recipe), fit per-question temperature post-hoc on held-out synthetic schemas, report debiased ECE and reliability diagrams per evidence-length bin.

---

## 10. Singular learning theory

**Aoyagi–Watanabe 2005 (reduced-rank regression, input N, output M, model rank H, true rank r ≤ H).** Writing the standard case split (as reproduced in Drton & Plummer 2017 §2.2):
- if N + r ≤ M + H, M + r ≤ N + H, H + r ≤ M + N and M + H + N + r even: λ = [2(H+r)(M+N) − (M−N)² − (H+r)²]/8, m = 1;
- same inequalities, sum odd: λ = [2(H+r)(M+N) − (M−N)² − (H+r)² + 1]/8, m = 2;
- M + H < N + r: λ = (HM − Hr + Nr)/2, m = 1;
- N + H < M + r: λ = (HN − Hr + Mr)/2, m = 1;
- M + N < H + r: λ = MN/2, m = 1.
Sanity: N = M = H = r = 1 gives ½ (regular scalar regression); N = M = 2, H = 1, r = 0 gives λ = 1, m = 2, matching the direct computation for K = |a|²|b|². (I could not retrieve the paper text; the formula above is from memory, cross-checked by these two cases — re-verify against the paper before citing.)

**Doc A's H-SLT-1 "LLC tracks RRR-like λ(H = n_state, r = Hankel rank)" — WRONG mapping.** RRR singularities come from the factorisation BA with H(M+N) free parameters. A diagonal SISO LTI model has 2n real parameters with shift structure; the singular locus at an unused mode is {p_i = c_i b_i = 0} × {λ_i free} — a coordinate singularity. No reason for RRR's formula to apply, and it does not.

**Doc B's λ ≤ r + (n−r)/2 — SOUND for real modes, and it is an equality.** Monte-Carlo volume scaling V(ε) ∝ ε^λ (`rlct_mc.py`, `rlct_refined.py`; local box around a true parameter, ε down to 1e-5):

| model | predicted | measured local slope |
|---|---|---|
| n=1, r=0 (real) | 0.5 | 0.50 |
| n=1, r=1 | 1.0 | 1.00 |
| n=2, r=1 | 1.5 | 1.48–1.51 |
| n=2, r=2 | 2.0 | 2.02 |
| n=2, r=0 | 1.0 | 0.82 at ε=1e-5, rising (expected (m−1)/log(1/ε) lag, m ≥ 2) |
| n=1 complex pair, r=0 | 1.0 | 1.00 |
| n=1 complex pair, r=1 | 2.0 | 2.0 |

Conditions: distinct true poles, p*_i ≠ 0, |λ| < 1, white-noise (or spectrum-bounded-away-from-zero) input, Gaussian noise. Separate (c_i, b_i) vs product p_i: same λ (the scale symmetry is a flat direction), different multiplicity. Additional loci (λ_i = λ_j coincidence, λ_i → 0) do not lower λ; they raise m.

> **Erratum (2026-10-07).** The sentence above is wrong for the global RLCT. Where prior mass reaches pole-coincidence loci, the loss near merged poles behaves like u² + v² + q²a⁴. The ideal (u, v, q·a²) gives λ = 1 + 1/4 = **5/4** (m = 1) for n = 2, r = 1, not 3/2. This was derived independently twice and checked numerically: the fiber method gives a slope of 1.250, and Monte Carlo gives 1.25–1.29 at the split point versus 1.486 at the generic point. The inequality λ ≤ r + (n−r)/2 still holds, and r + (n−r)/2 (and n + r for complex modes) are correct only as the *local* learning coefficient at a generic true parameter. The Codex counterexample (analysis/CODEX_CRITICAL_COMPARISON.md) was right. Source: research/Learning "Unifying Theory — Draft v0" B12/C1 and its independent "Review v0".

**Correction for complex modes.** In LRU/S5 with Re(·) readout each mode is a conjugate pair with 4 real parameters: used mode costs 2, unused costs 1 ⇒ λ = 2r + (n − r) = n + r, jumps of 1 per activated mode. Doc B's "½ per mode" is correct only in units of *real* state dimensions (count 2 per complex mode). State the unit.

**LLC on SSM/RNN literature.** Three targeted searches (2025–2026) surfaced no LLC/refined-LLC/susceptibility study on RNN/S4/S5/LRU/Mamba; Doc A's claim 6 stands as far as search can tell (absence not provable). A single-layer LTI LLC experiment that checks the n + r (complex) / r + (n−r)/2 (real) staircase against SGLD estimates would be a genuine small contribution; nothing above survives nonlinearity/selectivity, where λ must be estimated.

**Truncation threshold σ < √(log n / 8n) — FLAWED; corrected rule.** Let N_s be the sample size (Doc B's "n" is used for both state size and sample size), σ_ε² the noise variance, ΔL the H2² (white-noise MSE) increase from dropping the mode. Bayes prefers the smaller model iff N_s·ΔL/(2σ_ε²) < Δλ·log N_s, where Δλ = 1 (real) / 2 (complex) if the mode is genuinely used, and ½ / 1 if it is spurious (truth has one fewer mode). For *modal* truncation ΔL = |c_i b_i|²/(1−|λ_i|²) exactly (real pole; conjugate-pair analogue for complex). For *balanced* truncation ΔL ≤ 4σ² is valid but ~10× loose (measured 0.07–0.14 × 4σ²). Doc B's "½ log n" is the spurious-mode saving, and "4σ²" is the balanced-truncation bound applied to a modal decision — two different operations. Use the modal cost with the correct Δλ.

---

## 11. Verified emission "hard grounding theorem"

**SOUND as a lemma; OVERSTATED as "no hallucination".** What is provable, independent of weights: every string emitted by the product automaton G ⊗ Trie(K_plan) is in L(G) and every copied span is in K_plan; if the grammar's semantics is injective from surface form to logical form, then every *atomic* claim's logical form is in K_plan ⊆ K. That is a statement about the realiser's language, exactly analogous to Jev's "schema matching is guaranteed, not empirical" — and worth exactly as much.

**Not covered.** (i) *Selection*: the SSM may choose the wrong true fact (Doc B notes this). (ii) *Omission*: true facts arranged to mislead. (iii) *Composition*: "A, because B", "A; B" (implied relation), "all/some", comparatives, temporal reference ("now", "recently"), units — new claims not in K. (iv) *Scope/negation/anaphora*: grammar semantics of natural language is not injective. (v) *Aggregation*: totals, counts, "most" require computation not in K. (vi) *Implicature/presupposition*. (vii) *K itself* may be wrong or stale. (viii) Copy pointers preserve strings, not meaning in a new context.

**Recommendation.** Either restrict G to a controlled natural language with formal semantics (ACE/Attempto-style), in which case the theorem is real and the prose is stilted, or keep natural G and state the guarantee as "extractive faithfulness of atomic spans" plus an empirical NLI/entailment gate for composition and implicature. Do not plot 0 % hallucination.

---

## 12. Ctrl-G, SNGP-with-SSM, JEPA + HSVR, probabilistic circuits

- **Ctrl-G — SOUND, and overkill here.** Verified: sampling n tokens under a DFA with m edges and an HMM with h states costs O(n·m·h²); 100 % constraint satisfaction on CommonGen; the TULU2-7B experiments use a 32 768-state HMM (≈ 2 B parameters). The HMM exists to give the LLM *lookahead* on the constraint. Doc A's realiser only needs a DFA mask (G ⊗ Trie) — O(1) per token, no HMM — unless fluency under the mask is poor. Start with the mask.
- **SNGP "almost free" — WRONG direction.** SNGP needs (a) a distance-aware output layer (GP/RFF) and (b) a *bi*-Lipschitz feature map. Contraction supplies the upper Lipschitz constant; SNGP's whole point is the lower one (distance preservation), and contraction along time is many-to-one by design (forgetting). Only the per-position residual stream x + f(x), Lip(f) < 1 is bi-Lipschitz, and that is with respect to the *previous layer's* stream, not the input sequence. Also, for the Index the relevant "OOD" is CLINC150+OOS — a classification problem with an explicit out-of-scope option, not an OOD-detection problem; Clef-flash's 66.8 vs Clef's 97.4 there is the most plausible single win axis and needs no SNGP.
- **JEPA + HSVR — Doc A's collapse worry is right, and SIGReg does not fix it.** HSVR pushes Σσ_i → 0, i.e. toward memoryless dynamics; the JEPA predictive loss is trivially satisfied by embeddings that ignore the past. SIGReg (LeJEPA, arXiv 2511.08544) enforces an isotropic Gaussian *embedding distribution* — a memoryless token-wise map can satisfy it while the temporal dynamics have collapsed. Guard with a target HSV spectrum (lower bound) or apply HSVR only in a compression phase after pretraining (CompreSSM's in-training truncation is driven by the task loss, not a regulariser). LLM-JEPA (arXiv 2509.14252, NeurIPS 2025) shows fine-tuning gains on Llama/Gemma/OLMo; nothing on SSMs — low priority.
- **Probabilistic circuits for CRF constraints — SOUND, low leverage.** Compiling hard cross-field constraints to a circuit (semantic loss, Xu et al. 2018; semantic probabilistic layers, Ahmed et al. 2022) gives exact marginals/MAP in time linear in circuit size; size can be exponential for general CNF, polynomial for cardinality/implication constraints. The Index needs a probability for every option of every question; a joint model needs per-option *marginals*, which circuits provide. But Jev-style question maps rarely carry cross-field constraints; do this only after an error profile shows cross-field inconsistency.

---

## 13. Doc B's revised plan

**Is "freeze pretrained attention-free backbone + integrator head + adapters" the right first experiment?** Right direction, but note what it is: Clef's own recipe — "freezing Qwen3.8-27B for Clef and Qwen3.5-9B for Clef-flash, we jointly optimized the routing head alongside rank-256 low-rank adapters" (Cloudflare blog) — applied to an SSM. "Freeze + adapters" is not freezing; it is LoRA. Fine, but then the experiment must be designed for *attribution*, because the backbones differ in pretraining data, tokens and year, not only in architecture.

**Sharper version (weeks 1–3, ≈ $1.5–2 K).** Backbone × head factorial, same synthetic data, same losses (LS-CE + Brier), same LoRA rank, 2 seeds:
- Backbones: Qwen3.5-9B (Clef-flash's own; the control), Falcon-Mamba-7B (pure Mamba-1, 5.8 T tokens), xLSTM-7B (2.3 T tokens, 8 K ctx — note the context limit), RWKV-7 (2.9 B Goose; the 14 B/32 B releases claimed in search results must be verified on the Hub before planning on them), plus one hybrid (Falcon-H1 / Nemotron-H / Zamba2) to price "how much does a little attention buy".
- Heads: (a) Clef-style cross-attention head over final hidden states (control), (b) options-first integrator bank, (c) snapshot forks (§1), (d) final-state fork (Doc A).
- Cost: LoRA on a 7 B SSM over ~1.5 B synthetic tokens ≈ 6·7e9·1.5e9 ≈ 6e19 FLOP ≈ 70 H100-hours at 25 % MFU ≈ $175/run. The budget covers the full factorial; freeze for attribution, not for cost.
- Readout: zero-shot Index subset scored *per area* and *per record-length bin*, debiased ECE per question, p50/p99 latency vs record length.

**The 10-point stop rule — not sound as stated.** (i) Confounded: a 2024-data 7 B SSM will trail a 2026 9 B transformer on Knowledge & Reasoning (25.8 %) from pretraining alone; a >10-point gap says nothing about architecture. (ii) Arbitrary threshold with no attribution. Replace with: *if the SSM trails only on Knowledge & Reasoning → backbone problem, continue (swap/distil backbone); if it trails on Retrieval/Tools with heads (b)–(d) but not with the Clef-style head (a) → head problem, fixed; if it trails with head (a) too and the gap grows with record length → the fixed-state bottleneck binds, go hybrid.* Sample sizes are not the issue (a per-area subset of a few thousand items gives SE < 1 point).

**Cheaper, stronger designs.**
1. *Distil Clef-flash's own backbone into an SSM.* MOHAWK-style stage-wise distillation (Bick et al. 2024, Phi-Mamba) of Qwen3.5-9B into Mamba-2/Mamba-3, then Clef's recipe. Keeps the 2026 knowledge the Index rewards. Rough cost at 10–20 B distillation tokens ≈ 1e21 FLOP ≈ 1 000 H100-hours — at the budget limit; the "Mamba-in-the-Llama" variant that keeps ~25–50 % of attention layers is cheaper and more likely to retain retrieval. This is the only route by which "outperform Clef-flash overall" is even plausible under $5 K.
2. *Multi-teacher distillation with ground-truth labels* (Doc B's own plan): Clef-27B logits (open weights, free compute), Jev via API ($0.042/MTok → ~$100 for 2 B input tokens), frontier models through a System One adapter, labels from the synthetic generator. Temperature-scale each teacher before mixing. Fine.
3. *Test-time compute.* Read-twice (JRT-Prompt; Arora et al. 2024) is a model-side input transform, not prompt tuning, but check the harness maintainers' reading of "fixed rendering" before relying on it. Head ensembles are cheap. Self-consistency does not apply to a single-pass scorer.
4. *Win axes that are actually open*: CLINC150+OOS (flash 66.8 vs Clef 97.4 vs Jev 89.3 — flash fails out-of-scope), calibration (report debiased ECE per question type), long-record latency/memory (only if the Index has long records — measure first). Do not promise median latency: at 1 K-token records a 7 B SSM prefill is compute-bound like a 9 B transformer's; the SSM advantage appears at ≥ 16 K.

**Largest risk (Doc B) — agreed, and the literature already says which way it leans:** fixed-state models lose on in-context recall at every scale tested (Jelassi 2024; Waleffe 2024); hybrids do not. Plan for the hybrid fallback from day one rather than treating it as failure.

---

## 14. Inconsistencies between Doc A and Doc B

1. **Ordering.** Doc A forks options *after* the state (`[schema; state; question] → options`); Doc B's Bayes-filter argument requires options *before* the state. Doc A's "questions, state, questions again" with option lists inside the questions contradicts Doc A's own equivariance claim.
2. **Head.** Doc A: head reads h_sem of the final state via forks. Doc B: integrator bank accumulating per-token increments. Different heads, different information bottlenecks; neither document says which one the plan uses.
3. **Calibration.** Doc A needs a PID loop on ECE; Doc B says calibration is architectural. Both cannot be load-bearing; neither is right (§3, §9).
4. **Memory.** Doc A promises 64 K-context parity with Clef; Doc B's memory bound says the context cannot be stored; the plan never reconciles this with the snapshot/re-read options.
5. **Safety.** Doc A sells the LPV half-life bound; Doc B calls it useless with slow modes; the revised plan keeps "certified diagonal layers" without saying what is certified.
6. **SLT.** Doc A: RRR-like formula; Doc B: mode-counting formula. Incompatible; Doc B's is right (for real modes).
7. **Glass box vs frozen backbone.** Doc B's week-1–6 plan uses a pretrained Mamba/RWKV/xLSTM backbone — no h_sem split, no diagonal LRU structure, no Gramians, no LPV rung. Essentially all of Doc A's control toolkit becomes inapplicable until the "replace layers one at a time" phase, which Doc B schedules for weeks 6–8 at ~$700.
8. **Distillation ceiling.** Doc B says a student trained on Clef rarely exceeds Clef; Doc A's loss includes KL(p‖p_ref) to Clef-flash and distils Clef-flash — the model it is meant to beat.
9. **SNGP vs forgetting.** Doc A's "contracting ⇒ bi-Lipschitz almost free" is contradicted by Doc B's own memory bound (contraction = forgetting = non-injective).
10. **"No tradeoff" vs Pareto map.** Doc B's glass-box conversion "records cost" and builds a Pareto map — an admission of a tradeoff the user's framing denies. Say so explicitly.
11. **Conformal abstention vs Index rules.** Doc A plans act/abstain; Doc B's zero-shot-Index framing makes abstention score zero; unaddressed.
12. **Index description.** Doc A says edition 0.2 "averages 40 benchmarks in five equal-weight areas" and "132,422 decisions"; the harness README for 0.2.1 (2026-09-27) says 40 benchmarks, 38 counted, with area weights 10 / 18.3 / 20 / 25.8 / 25.8 % (square-root-of-size weighting). Cloudflare's table is labelled 0.2.1. Re-verify edition, weighting and decision count before quoting either.

---

## 15. Recommended revised first experiment (concrete)

1. **Week 0 (≤ $100):** reproduce Clef-flash locally on the Index subset; dump record-length distribution per benchmark; dump per-area scores; measure flash's CLINC+OOS failure mode (reliability diagram on in-scope vs OOS).
2. **Weeks 1–3 (~$2 K):** the backbone × head factorial of §13 with the attribution readout. Deliverable: a table gap(area, head, backbone, length-bin).
3. **Decision gate (replaces the 10-point rule):** attribute the gap per §13; choose pure-SSM + snapshot/Clef-head, or hybrid, or distilled backbone.
4. **Weeks 4–6 (~$1.5 K):** multi-teacher + GT distillation on the chosen configuration; post-hoc per-question temperature; submit to the Index.
5. **Weeks 6–8 (~$0.7 K):** glass-box conversion with *honest* certificates: per-layer HSVs for compression (CompreSSM), stack-level ISS/contraction bound with the cascade formula, concept-bottleneck h_sem with a leakage number, adversarial-record flip rate. No half-life safety claim, no "100 % provable" claim, no PID, no CBF for the decision model.
6. **Only then** the SLT side-quest (single-layer LTI LLC staircase), and the verified-emission realiser with a controlled grammar.

---

## Appendix A — raw numerical outputs (abridged)

`gramian_hsv.py` (n = 6 complex-diagonal SISO): max|P_cf − P_iter| = 1.8e-15; max|Q_cf − Q_iter| = 4.5e-16; HSVs [5.98, 2.00, 0.70, 0.244, 0.0957, 0.00248]; balanced truncation r = 1..5: H∞ error / (2Σσ) = 0.510, 0.594, 0.544, 0.661, 0.726; H2² error vs 4σ²_{r+1}: 2.28 vs 15.95, 0.175 vs 1.97, 0.036 vs 0.239, 0.0079 vs 0.037; modal H2² costs per mode [20.6, 0.076, 1.00, 3.01, 0.59, 0.19] vs smallest HSV 0.0025; real 2n-state realisation has 12 HSVs [3.83 … 1e-5].

`memory_depth.py`: single-layer max|g_τ|/bound = 0.996. Cascade (L, peak τ, half-decay after peak): ρ = 0.9 → (1,0,7) (4,26,25) (16,134,49); ρ = 0.99 → (1,0,69) (4,296,252) (16,1484,502); ρ = 0.999 → (1,0,693) (4,2997,2524) (16,14985,5031). Required per-step Δ|Re a| for 1 K / 16 K / 64 K retention: 6.8e-4 / 4.2e-5 / 1.1e-5.

`bayes_integrator.py`: table in §3 (ρ_copy = 0.7; fitted stateless increment f = [−0.31, −0.09, 0.09, 0.31] vs true LLR [−1.39, −0.41, 0.41, 1.39]).

`rlct_mc.py` / `rlct_refined.py`: slopes in §10; n=2, r=0 slopes at ε = 1e-2 … 1e-5: 0.44, 0.51, 0.63, 0.72, 0.78, 0.82; n=2, r=1: 1.00, 1.16, 1.36, 1.48, 1.51, 1.49; complex unused: 0.73, 0.85, 0.94, 1.00, 1.00, 1.01.

`ece_cbf.py`: ECE floor table in §9; CBF projection: required h(x⁺) ≥ −1.549, achieved −1.549, |u| = 0.410; readout shift with w ⟂ a: −2.4e-16; with w ∥ a: 0.410.

## Appendix B — literature / fact checks performed

Verified (fetched or found in search snippets): Cloudflare Clef blog (frozen backbone + rank-256 LoRA + routing head; LS-CE + Brier + RLCD; table numbers 94.20/90.93/79.74, 97.43/66.77/89.27, 98.47/98.76/95.75, latency 209.3/38.8/524.1; 64 K context; "permutating field orders, prompts, and schema structures"; GPQA and Laya columns and ECE not on the fetched page — unverified). TypeSafe post (2026-09-15, Diogo Almeida; 255 cardinality; two-stage; "0 %" hallucination as a schema guarantee; 70–500 ms; $0.042/MTok; the "193.6×/444.6×" figures were not on the fetched page — unverified). Decision Index harness README (0.2.1, 40 benchmarks/38 counted, area weights, chance-correction formula, abstain = wrong, three house rules). Ctrl-G (O(n·m·h²), 100 % CommonGen, 32 768-state HMM). Halloran et al. 2406.00209 (Thm 1, MLE ≤ 0, Mamba and Mamba-2, A ≤ 0 via log-space, Δ ≥ 0). Zubić & Scaramuzza 2505.11602 (ISS/passivity for selective SSMs with discontinuous gating). CompreSSM (ICLR 2026, 2510.02823). HSVR (NeurIPS 2025, 2510.22951). Mamba-3 (2603.15569; MIMO state 64 ≈ Mamba-2 state 128; public models ≈ 1.5 B). JRT (2407.05483). Jelassi 2402.01032; Waleffe 2406.07887. CBM leakage (Mahinpei 2021, Margeloiu 2021, Havasi 2022). Roelofs 2022 / Kumar 2019 on ECE bias. Conformal singleton error (2506.21802). LLM-JEPA (2509.14252), LeJEPA/SIGReg (2511.08544). BarrierSteer (2602.20102; Table 3 numbers not checked). Falcon-Mamba-7B (7.27 B, 5.8 T tokens), xLSTM-7B (2.3 T tokens, 8 K ctx), RWKV-7 Goose 2.9 B (larger RWKV-7 sizes: verify on the Hub).

Not found / unverifiable: any LLC study on RNN/S4/S5/LRU/Mamba (three searches, 2025–2026); Decision Index 0.2 leaderboard numbers with ECE (Kev plan); BabyHGRN numbers; Mamba-3 "100 % Parity"; "132,422 decisions"; NKI scan kernels. Aoyagi–Watanabe formula reproduced from memory with two sanity checks (paper text not retrievable here).

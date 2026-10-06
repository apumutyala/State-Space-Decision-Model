# RESEARCH_DIRECTIONS — a menu for a new collaborator

Written 2026-10-06 against `origin/agent-main` @ `3b4169a`, `SAFETY_SPEC.md` v1.2 (branch `claude/safety-spec`, PR #18, open), `controlsafe/CONTROLSAFE.md`, `jobs/PERF_AUDIT.md` (branch `claude/readout-fork`, PR #50, open), `techniques/CATALOG.md` + notes 01–09, `interfaces/DECISION_LANGUAGE_V0.md` (0.1.1), `plans/codex/LANGUAGE_GROWTH_PLAN.md`, and the Clef-flash baseline (`job-sgihj`) as recorded in `README.md` and issue #3 (I did not open the receipt directory itself). Literature was re-checked on arXiv on 2026-10-06; the status of every citation is in the appendix. Anything I could not check is marked **[unverified]**.

## How to use this document

This is a menu, not a plan. The owner has said the choice is yours. Read the orientation (two minutes), skim the at-a-glance table, then read the two or three options that interest you in full. Each option states, in plain language, what it is, why it would move *this* project (which goal, guarantee or axis of the frontier), the first experiment you could run on a laptop CPU for $0, the rough GPU bill if it goes further, exactly where it plugs into the repo, the papers to read first (verified), the risks, my honest probability-of-payoff *judgement* (labelled as such, never a measured number), and the result that should make you stop. Pick one, open a `collab/<your-github-user>/<topic>` branch from `agent-main`, write the falsifiable hypothesis in your first commit message, and run the $0 step before anything paid. The hard rules in `AGENT_HANDOFF.md` §1 apply to every option here; the two that bite most often are "no shared-credit spend without the owner's explicit OK in a `spend` issue" and "never open the sealed TEST slice".

## Orientation

**What the project is.** We are building an attention-free *decision model*: a recurrent (state-space) network whose text path has no softmax over positions and no KV cache that grows with input length. It answers typed questions over a host-supplied schema (`noul` yes/no, `choice` over 2–255 options, `score` on an ordinal scale) with a probability for every option. The target is the Decision Index 0.2.1 (38 benchmarks, five areas: Knowledge 25.85 %, Language 25.85 %, Retrieval 20.02 %, Tools 18.28 %, Arts 10 %), where the competitors are Cloudflare's Clef (27B, board 61.98) and Clef-flash (9B, ≈ 57.07), TypeSafe's closed Jev (57.91), and the best open 9B non-Clef entrant Nimble-9B-v3 (56.88). The route is not a new model: both Clef backbones are already 75 % Gated DeltaNet; "attention-free Clef" means converting the remaining 8 of 32 attention layers in Qwen3.5-9B (Q9-AF), with RWKV7-G1k-13.3B as the pre-registered pure-recurrent fallback. A second track (Track L, Codex) grows a language model from that decision parent in which every text output is a checked, typed semantic program rendered by a deterministic realizer, never an unchecked text path.

**The three-axis frontier.** Nothing here optimises one number. Every result is reported on three axes together: **capability** (Index score per area, calibration), **interpretability and control** (steer and hijack rate under adaptive attack, certified radius, attribution tests that survive perturb-and-recompute, structural guarantees G1–G16), and **efficiency** (latency under the Index's 1,000 ms gate, state bytes, dollars). An option is interesting if it moves the whole frontier, or if it prices one axis against another honestly (the "cost-of-attention-freedom curve" is the project's first such slice). The repo's own wording is "reported together"; "Pareto frontier" is the coordinator's framing.

**Where things stand (2026-10-06).** All CPU-side machinery is built and merged (~670 tests): the Index harness reproduces the official kit on DEV (12,104 rows; TEST 12,161 rows, sealed); the `decide()` adapter, the safety-eval battery and PREREG, the conversion pipeline, five decision heads, the m2 trainer, the controlsafe mechanisms (G15 trust gate, G16 protected judge, G4 snapshots, persistence bounds, G14 monitor) and Track L's L0a–L0f. **No model capability number exists yet.** The one measured baseline is **Clef-flash: 56.8 [55.4, 58.2] on our 12,104-row DEV slice** (knowledge 51.7, language 53.8, **retrieval 48.1 (weakest)**, **tools 80.8 (strongest)**, arts 51.3), consistent with the ≈ 57.1 public figure. The 9B qualification re-run (`job-b69gs`), the readout performance fix (PR #50) and the promotion margins (issue #36) gate the first real measurements. Spend to date is $1.37 of a $525 SF Compute envelope whose $500 promo expires 2026-10-27. The minimum detectable Index difference on DEV is ≈ 0.65–2.4 points depending on error correlation, so "a point" is a real result and "half a point" usually is not.

**Reading order if you are new:** `README.md` → `GOAL.md` → `PLAN.md` → `HANDOFF.md` → `SAFETY_SPEC.md` §1–5 → `AGENT_HANDOFF.md` §1, §4, §7 → the `techniques/0x` note for your option.

## The menu at a glance

Axis key: **Cap** capability, **Ctl** interpretability/control, **Eff** efficiency. "Payoff" is my judgement of the chance the option produces a result worth a receipt (positive *or* a clean negative), not a probability of beating Clef.

| # | Option | Axis | Goal / guarantee it touches | $0 step | GPU if any | Payoff (judgement) |
|---|---|---|---|---|---|---|
| A1 | CRF / MRF joint head over the typed questions of one request | Cap, Ctl | Index multi-question rows; G1 exact marginals; G9-style factor accounting | DEV label-dependence census; CPU exactness tests | one m2 cell $5–17 | low–moderate |
| A2 | Factor graphs with host-declared hard constraints (D policy, L registry) | Ctl, Eff | G2(a)/(d) inside the head; Track L unsupported-proposed rate | fixture-pack census; constraint factor == post-mask | none for L0 | moderate (L), low (Index) |
| A3 | HMM as rung-0 glass-box planner/realizer; HMM-guided decoding (Ctrl-G/GeLaTo/LTLA) | Ctl | Track L L0h comparator; G11 kept via licensed productions | EM on fixture programs; compare to rule/oracle | $0–20 if distilled from the parent | moderate as comparator |
| A4 | Probabilistic circuits: compile constraints, exact tractable inference (SPL) | Ctl, Eff | Track L admissibility by construction; D `policy_restriction` | compile v0.1 registry to a circuit; size and prune rate | none | moderate (L) |
| A5 | SSM ↔ HMM / Kalman: GDN as a Kalman filter with identity covariance; HMM as discrete SSM | Cap, Ctl | Retrieval gap (48.1); G14 telemetry; conversion destination layer | GKA-style layer in pure torch; MQAR at matched state | ≤ $20 on the 0.8B | moderate |
| A6 | Bayesian-filter integrator head (mean + variance, exact accounting) | Cap(ECE), Ctl | G7 calibration, G5 selective risk, G9(a) | tiny-backbone AUGRC test | one m2 cell | low–moderate |
| A7 | Conformal and calibration on PGM outputs | Ctl | G5, G7 reporting; joint-assignment sets (deployment only) | on existing prediction files | none | moderate (infra) |
| B1 | fla `chunk_rwkv7` / GDN kernels | Eff | m2 eval cap needs ≥ 12k tok/s; RWKV7 stage B | pure-torch vs eager numeric gate | ≤ $20 H100 gate | high (works), moderate (leverage) |
| C1 | G16 capability-cost study | Ctl, Cap | G16 veto/radius/independence as a free guarantee or a priced one | CPU dry run with trained tiny heads | one m2 cell | moderate |
| C2 | Diagonal / contraction-constrained certified layer swaps | Ctl, Cap | CERTIFIED decay + Gramian telemetry vs Index drop (a Pareto slice) | tiny-model curve | ≤ $20 on the 0.8B | moderate (curve), low (cheap certificate) |
| C3 | Persistence / Lyapunov / ISS bounds for selective SSMs (RWKV-7 open problem) | Ctl | CONTROLSAFE persistence row EMPIRICAL → CERTIFIED (as a cost bound, not safety) | pure math + CPU counterexample search | none | moderate |
| C4 | Hidden-state poisoning defences (G14 is weak: AUROC 0.57 / 0.86) | Ctl | G14 alarm ≥ 0.9; collapse probe on parents and Clef-flash | new detectors on the 10-seed CPU protocol | $6–9 parents, $3 Clef-flash | moderate–high (detector), unknown (parent) |
| D1 | Mechanistic interpretability of recurrent state (WriteSAE, state lens, linear DAS) | Ctl | G9(b)(c); provenance write audit; `interp/` is unbuilt | SAE on tiny GDN writes; replacement test | $10–30 on 9B writes | moderate |
| E1 | Track L semantic-program learning assets (audit set, edit pairs, rung-A grammar) | Cap(L), Ctl | L0h–L2 gates; G10/G11 | author fixtures and audit cases | L0g ≤ $20 (Codex) | high (assets), uncertain (learned) |
| F1 | Arts & Human Taste data sourcing (5.8 % of mix vs 10 % weight) | Cap | Arts area (Clef-flash 51.3) | registry + dedup build | none | moderate–high (build), unknown (score) |
| F2 | Jev evaluation protocol (adapter; API call is the owner's decision) | Ctl | SAFETY_SPEC §7 Jev rows | adapter + fixtures | API $ **[unverified]** | certain if approved |
| F3 | Multimodal / vision SSM encoder | Cap (scope) | text-only claim re-scoped; G1/G2/G8 intact | check kit for image rows; design | later, $$ | low near-term |
| G1 | SLT / LLC side paper with the lab's negative SGLD results | — | none (LLC is logged, never gated) | exact quadrature reference on tiny nets | $ at ≤ 1B | moderate (paper), zero (goal) |

Also open, documented in `AGENT_HANDOFF.md` §7 and not expanded here: a cheaper uncovered-insertion certificate for G4; a black-box search attack family (PISmith / TAP-style) for G6, filed as a PREREG amendment through Track D; the full-parent G14 collapse probe (folded into C4 below).

---

## Part A — Probabilistic graphical models (the owner's suggested area)

### A0. Why PGMs fit here, and three constraints to respect

The decision model already *is* a structured predictor: one request carries several typed questions over one state, the host supplies the option sets, and `decide()` must return a full distribution per question that sums to 1 (G1, abs. error ≤ 1e-9). Track L's planner is a sequence of typed choices under a declared grammar and registry. Both are places where a graphical model gives you three things the neural head alone does not: exact normalised marginals over structured outputs, hard constraints satisfied by construction rather than by post-masking, and parameters you can print in a table. Clef's "joint schema head" is a small transformer over all options of all questions; it is implicitly joint and opaque. A PGM is explicitly joint and legible. That is the pitch.

Three constraints you must design around:

1. **Served distributions are never replaced by policy-conditioned ones (G2(d)).** A factor graph that folds `policy_restriction` in as a hard factor produces the *policy-conditioned* stage; the raw and served stages over the full supplied option set must still exist. `decide/schema.py` wire shapes are owner-only; your head produces per-question marginals and hands them to `decide.schema.wire_answer`.
2. **Type/syntax filtering only, never evidence pruning (interface 0.1.1).** The Track L bridge "performs bookkeeping and type/syntax filtering, not gold-answer selection", and "candidate construction cannot query the checker for all true alternatives and hide the false ones before model scoring". A compiled constraint (A2, A4) may encode types, units, arity, acyclicity and permissions. It may not encode whether a fact is true. If your circuit needs an evidence predicate to be sound, you need an `interface` issue first.
3. **The Index scores abstention as wrong.** Conformal sets, selective risk and anything that abstains (A7) are deployment features reported off-Index, exactly as SAFETY_SPEC says for G4/G5.

Shared reading for Part A: Sutton & McCallum, *An Introduction to Conditional Random Fields*, 1011.4088 (V); Murphy, Weiss, Jordan on loopy BP, 1301.6725 (V); Darwiche & Marquis, *A Knowledge Compilation Map*, 1106.1819 (V); Choi, Vergari, Van den Broeck, *Probabilistic Circuits: A Unifying Framework*, 2020, starai.cs.ucla.edu/papers/ProbCirc20.pdf (V(search), no arXiv ID); Ghojogh, Sepanj, Ghojogh, *On the Relation of State Space Models and Hidden Markov Models*, 2601.13357 (V, Jan 2026).

### A1. CRF / Markov random field over the typed questions of one request

**What it is.** Keep the existing per-question head as the unary potential (one score per option). Add pairwise factors between questions that share a request (a small compatibility table keyed by question-type pair, or a low-rank bilinear factor over option-text features). Run exact inference (belief propagation on a chain or tree; junction tree for small question counts) to obtain per-question marginals. Train by maximum conditional likelihood on the m2 records. The pairwise tables are the "explanation": they say, in numbers, how answering q3 "yes" moves q7.

**Why it might matter here.** Capability on multi-question rows (a ContractNLI row has 17 questions over one 4.5k-token contract; the Index has many such benchmarks) if gold labels within a row are dependent given the state. Interpretability: exact factor accounting extends G9(a)'s "sums exactly" from tokens to questions. Efficiency: marginals are cheap next to the backbone.

**First experiment ($0, CPU).** (i) Census: using `harness/manifests/` and the kit's DEV rebuild, count rows with ≥ 2 questions per benchmark and compute the mutual information between gold labels of question pairs within a row, by benchmark and question type. If dependence is near zero almost everywhere, stop here and log it. (ii) Exactness: implement the CRF head following the integrator/fork pattern in `heads/heads.py`, register in `heads/model.py`, and add a brute-force test (enumerate all joint assignments for ≤ 4 questions × ≤ 4 options) that marginals match to 1e-9, plus the equivariance and attribution-sum tests from `tests/test_heads.py`. (iii) Learning signal: on `heads.backbone.tiny_hf_backbone` with synthetic correlated questions (the m2 dry-run infrastructure in `tests/test_m2_e2e_dryrun.py`), check the CRF beats the independent head in NLL.

**GPU cost.** One m2 cell on the selected parent (≈ $5–17 per the m2 estimates), only after parent selection and issue #36.

**Plugs into.** `heads/heads.py`, `heads/model.py`, `heads/prefix_features.py` (one pass per request already exists, PR #50), `train/` arms; outputs through `decide.schema.wire_answer`. Branch: `collab/<you>/pgm-crf-head`. Flag `attention_free=True` only if no softmax over positions is used in the factors.

**Read first.** 1011.4088 (V); Belanger & McCallum, SPEN, 1511.06350 (V, ICML 2016) for the energy-network relaxation if tables get large; Kuleshov & Liang, *Calibrated Structured Prediction*, NIPS 2015 (V(search), proceedings.neurips.cc) for what calibration means for marginals.

**Risks and judgement.** Within-row dependence may be small or already captured by the shared state (the backbone reads all questions). Clef's learned joint head is a stronger version of the same idea; you would be trading capacity for legibility. Coupling questions is a new steering surface: injected text that moves one question's evidence can now move another's answer, so G6 steer must be measured on this head, not assumed. Payoff judgement: low–moderate on score (if it moves anything, expect ≤ 1 point on multi-question benchmarks), moderate on the interpretability axis.

**No-go.** Label MI within rows ≈ 0 for the benchmarks that carry > 90 % of multi-question weight; or, on the 2,008-row DEV subset, the CRF head's paired difference vs the independent head has CI lower bound < 0 and point estimate below the MDE.

### A2. Factor graphs with host-declared hard constraints (multi-field joint decisions)

**What it is.** Represent a request's decision variables (D: the questions; L: a program's slots — operation, arguments, projections) as a factor graph whose hard factors are *declared by the host*: the allowed set (`policy_restriction`), cross-field consistency rules the host or registry states ("if q1 is `false`, q2's domain is {n/a}"), the registry's type, unit, arity and acyclicity rules, and audience permissions. Exact inference by variable elimination; the output distribution has zero mass on inadmissible joint assignments by construction. The precedent is FactorDLM (2609.32900, V, Sep 2026): relational constraints over JSON records with cross-field references as a factor graph, conditioned exactly by variable elimination at 0.4–6.9 % overhead.

**Why it might matter here.** It moves G2(a) from a post-hoc mask to a property of the inference, keeps G2(d) by producing the policy-conditioned stage separately, and for Track L it should cut the *unsupported-proposed* rate (programs the planner proposes that the kernel rejects for type/syntax reasons) without touching the kernel (G10 unchanged: the kernel still checks everything). Efficiency: fewer repair calls against `max_repair_calls`.

**First experiment ($0, CPU).** (i) Track L census: run the free-running scorer (PR #17) over `fixtures/language_growth` and classify every rejection as type/syntax/arity/acyclicity (eliminable by a factor graph) vs evidence (not eliminable, by the interface rule). (ii) D: implement `policy_restriction` as a hard unary factor in a tiny factor-graph module and prove by test that the policy-conditioned stage equals the current post-mask exactly, with `excluded_served_mass` unchanged. (iii) L: build the factor graph for the v0.1 registry (≤ 8 nodes, ops in `DECISION_LANGUAGE_V0.md` §Initial semantic operations) and measure elimination width on fixture programs.

**GPU cost.** None at L0. If used in L0h, it rides inside Codex's ≤ $20 qualification.

**Plugs into.** `language_growth/planner.py` (candidate construction), `language_growth/kernel.py` read-only (you consume the registry, you do not change acceptance), `decide/adapter.py` only through the owners (interface issue if any wire change). Branch: `collab/<you>/pgm-factor-constraints`.

**Read first.** 2609.32900 (V); 1301.6725 (V) for when exact elimination is infeasible; `interfaces/DECISION_LANGUAGE_V0.md` ("type/syntax filtering, not gold-answer selection"); `plans/codex/L0_PHASES.md`.

**Risks and judgement.** Elimination width can blow up with many cross-slot constraints; the registry must expose its constraints in a machine-readable form (today they live in kernel code); the biggest risk is accidentally encoding an evidence predicate, which the interface forbids and which would turn a guarantee into a bypass. Payoff judgement: moderate for Track L (deterministic, cheap, measurable), low for the Index score.

**No-go.** Fewer than ~20 % of planner rejections on fixtures are type/syntax (then the factor graph buys little); or no sound factorisation exists without an evidence predicate.

### A3. HMMs: a rung-0 glass-box decoder, and HMM-guided constrained decoding

**What it is.** Two arms that share machinery. (i) *Deployable glass box.* Track L's rung A realizer chooses among licensed productions (`techniques/07` §7.4: the model's job is a decision among variants, not token emission). An HMM over production sequences — hidden states = discourse slots, emissions = licensed productions — is a generative prior whose every parameter is a row in a table, with exact forward/Viterbi inference; the parent's typed choice multiplies the prior. Likewise, a small HMM over semantic-program node sequences is the simplest *learned* planner baseline for L0h ("rung 0"), in the tradition of Chiu & Rush's scaled HMM LMs (2011.04640, V) and low-rank fast inference (2201.02715, V). (ii) *Ablation only.* HMM × DFA guidance of a free-text realizer, as in GeLaTo (2304.07438, V), Ctrl-G (2406.13892, V) and LTLA (2511.16054, V: a neural head predicts a prefix-dependent latent prior while a shared HMM answers continuation queries exactly). `techniques/07`'s deployment note excludes free connective text from the plan, so this arm is a research comparison, not a shipping path, unless the contract changes.

**Why it might matter here.** Track L needs a learned-level comparator and an inspectable one; an HMM gives exact marginals, exact MAP, trivial calibration, and a one-page explanation. It keeps G11 (whole-answer licensed semantics) because it only chooses among licensed productions. It also gives a cheap "what does a Markov prior already explain" baseline that stops us over-crediting the neural parent.

**First experiment ($0, CPU).** Fit an HMM by EM on the fixture-pack program sequences (plus synthetic catalog programs from the L0 generators); evaluate as a planner on the L0 fixtures against the oracle program and the rule baseline using the free-running scorer: first-attempt accuracy, unsupported-proposed rate, avoidable asks. Separately fit a production-sequence HMM for the realizer and check that every Viterbi path is a licensed production sequence (G11 test).

**GPU cost.** None for the CPU HMM. Distilling an HMM from the 9B parent's typed-choice distributions (GeLaTo-style) is ≈ $5–20 and needs a parent revision, so it waits for m1.

**Plugs into.** `language_growth/planner.py` (as an alternative planner behind the same bounded-call interface), `language_growth/realize.py` (production choice only; never new strings), `language_growth/eval.py`. Branch: `collab/<you>/pgm-hmm-rung0`.

**Read first.** 2304.07438, 2406.13892, 2511.16054, 2011.04640, 2201.02715 (all V); Scribner, Vergari, Belle, *Tractable Hierarchical Control of Autoregressive LMs*, 2607.20483 (V, Jun 2026: LR(k) grammar satisfaction in polynomial time with a distilled tractable model); `reviews/02` §12 (the earlier verdict that a DFA mask suffices where Ctrl-G's HMM is overkill).

**Risks and judgement.** An HMM's Markov state is small; planner choices depend on the evidence, so a bare HMM will under-fit and the fix (LTLA's neural latent prior) re-introduces a learned component. "Proof-valid answer to the wrong question" is a task failure the HMM cannot see. The published HMM-guidance numbers are for free-text GPT-2/TULU-7B, not typed programs. Payoff judgement: moderate as a comparator and glass-box baseline; low as the final planner.

**No-go.** The HMM planner does not beat the rule baseline on L0 first-attempt accuracy, or the realizer HMM degenerates to a lookup table (nothing learned, nothing explained).

### A4. Probabilistic circuits: constraint compilation and tractable exact inference

**What it is.** Probabilistic circuits (sum-product networks, SDD/d-DNNF compiled formulas) support exact marginals, MAP and "probability that a constraint holds" in time linear in circuit size. Knowledge compilation (1106.1819, V) turns a logical constraint into such a circuit once; Semantic Probabilistic Layers (2206.00426, V) put a circuit on top of a neural net so that the output distribution satisfies the constraint with exact normalisation; semantic loss (1711.11157, V) trains against it; PyJuice (2406.00766, V) makes circuits fast on GPUs. 2025–26 work uses circuits for constrained decoding: tensorized finite automata circuit-multiplied with HMMs as SMC proposals (Dang et al., 2606.01926, V, ICML 2026); LR(k) grammars in polynomial time (2607.20483, V); Neural Probabilistic Circuits for interpretable attribute-based prediction (2501.07021, V). Limits: PCs lag transformers as LMs because of output and context-routing bottlenecks (2605.12940, V, May 2026), so use them for *constraints*, not as the backbone.

**Why it might matter here.** Track L: compile the registry's admissibility (operation signatures, type and unit compatibility, DAG ≤ 8 nodes, dependency permissions) into one circuit; the planner's proposal distribution then has exactly zero mass on inadmissible programs, and "probability the remaining program can still be completed admissibly" (continuability) is an exact query. D: `policy_restriction` and the ordinal monotonicity of `score` questions are small circuits. Interpretability: a compiled constraint is auditable; efficiency: fewer repair calls.

**First experiment ($0, CPU).** Compile the v0.1 registry constraints to an SDD/d-DNNF with an off-the-shelf compiler (PySDD or similar); record circuit size; replay the fixture-pack proposals and measure how many grammar-valid-but-type-invalid proposals the circuit removes versus a per-slot DFA mask. If the DFA already removes ≥ 95 %, write that down (it confirms `reviews/02` §12) and fold.

**GPU cost.** None unless you train a circuit surrogate for lookahead (PyJuice, $).

**Plugs into.** Same seam as A2 (`language_growth/planner.py` candidate construction; registry read-only). Branch: `collab/<you>/pgm-circuits`.

**Read first.** 1106.1819, 2206.00426, 1711.11157, 2606.01926, 2607.20483, 2605.12940 (all V); ProbCirc20 (V(search)).

**Risks and judgement.** Circuit size can explode with cross-slot constraints; the interface rule (type/syntax only) bounds what you may compile; the earlier review already judged DFA masks sufficient for syntax, so the win must come from *semantic-typing* constraints a DFA cannot express. Payoff judgement: moderate for Track L, low for the Index.

**No-go.** Compiled circuit exceeds ~10^6 nodes for v0.1, or the DFA mask already eliminates ≥ 95 % of inadmissible proposals on fixtures.

### A5. SSMs as filters: GDN ↔ Kalman, HMM ↔ discrete-state SSM

**What it is.** A linear-Gaussian state-space model with a Kalman filter is exact Bayesian filtering; an HMM is a discrete-state SSM whose forward algorithm is the same filtering; S4/Mamba/GDN are deterministic discretised linear systems with input-dependent ("selective") transitions. Ghojogh et al. (2601.13357, V) lay the three side by side and say where they are equivalent and where they diverge (stochastic vs deterministic state, inference vs forward pass). The sharp new result for us is Gated KalmaNet (Peng, Chattopadhyay, Zancato, Nunez, Xia, Soatto, 2511.21016, V, Nov 2025 / May 2026): **DeltaNet, Gated DeltaNet and Kimi Delta Attention are Kalman-filter recursions under an identity error-covariance assumption**; tracking the full covariance (test-time ridge regression) gives > 10 % relative gains on long-context RAG/LongQA to 128k. The same group's Priming paper (2605.08301, V) reports the hierarchy GKA > GDN > Mamba-2 predicting long-context reasoning. Kalman Linear Attention (2602.10743, V) writes the Kalman filter in information form as an associative scan. Meta-trained selective SSMs are provably Bayes-optimal on linear-Gaussian in-context tasks (2602.17744, V) and Mamba learns the Bayes-optimal Laplacian smoother for Markov chains (2502.10178, V); selective SSMs generalise as filters of unknown systems (2604.23818, V). Transformers can represent a Kalman filter (Goel & Bartlett, 2312.06937, V) and HMM prediction has a transformer-like "dual filter" (2505.00818, V).

**Why it might matter here.** (a) Capability, on the weakest area: our conversion destination is GDN. If GDN is a Kalman filter that throws away the covariance, a covariance-tracking destination layer for the 8 converted layers is the one principled lever on retrieval (48.1) that is not already in `PLAN.md`'s recall levers. (b) Control: the filter view names quantities — innovation, Kalman gain, covariance — that are natural telemetry for G14 (memory collapse looks like a gain/covariance anomaly) and for the persistence row. (c) Glass box: an HMM fitted to quantised state trajectories is a discrete surrogate of the recurrence whose likelihood under an input is an interpretable monitor.

**First experiment ($0, CPU).** (i) In `convert/gdn.py`'s pure-torch recurrent path, implement a GKA-style covariance-tracking variant of one layer; prove by test that it reduces to GDN when the covariance is pinned to identity; run MQAR and S-NIAH from `probes/core.py` on tiny from-scratch models at *matched state bytes* (the covariance costs d_k² per head — count it). (ii) Fit an HMM to quantised recurrent states of tiny RWKV-7 and all-GDN models under the `safety_eval.controlsafe_arms --arms g14` protocol (10 joint seeds) and report its AUROC as a collapse alarm against the retained-norm detector's 0.57 [0.43, 0.72] (RWKV-7) and 0.86 [0.78, 0.93] (GDN).

**GPU cost.** A GKA destination layer in the 0.8B one-layer conversion with the existing accept/rollback gate: ≤ $20. GKA kernels are not in `fla` as far as the repo could find **[unverified]**, so a Triton kernel is its own task.

**Plugs into.** `convert/gdn.py`, `convert/model.py` (destination-layer modes), `convert/pipeline.py` gate, `probes/core.py`; for telemetry, `controlsafe/monitor.py` and `safety_eval/controlsafe_arms.py::ARMS`. Branch: `collab/<you>/gka-destination` or `collab/<you>/hmm-state-monitor`.

**Read first.** 2511.21016, 2605.08301, 2602.10743, 2601.13357, 2602.17744, 2502.10178, 2604.23818 (all V); 2412.06464 Gated DeltaNet (V(repo)); `techniques/01` §1.14 and `techniques/02` §2.3–2.4.

**Risks and judgement.** GKA and Priming are one industry group's preprints (P grade); covariance tracking raises state bytes and latency, which is the efficiency axis we are trying to win and may fail the 1,000 ms gate projection; the "Bayes-optimal" theorems are for linear-Gaussian toy tasks; 2601.13357 is a unification, not a theorem that transfers guarantees. Payoff judgement: moderate on capability (it is a real, named hypothesis about the retrieval gap), moderate on telemetry.

**No-go.** GKA does not beat GDN on MQAR/S-NIAH at matched state bytes on tiny models or on the 0.8B swap; or its per-token cost breaks the latency gate projection; or the HMM surrogate's AUROC is no better than the retained-norm detector over 10 seeds.

### A6. A Bayesian-filter reading of the integrator head

**What it is.** The additive integrator (λ = 1) accumulates per-token contributions whose sum *is* the logit (G9(a)). A filter version keeps, per option, a mean and a variance (or a Beta/Dirichlet posterior for `noul`/`choice`), updates both per token with an innovation-weighted gain, and emits the mean as the logit and the variance as a native confidence. Accounting stays exact on the mean; the per-token gain is an interpretable "how much this token moved the belief"; the variance can replace or feed the predicted temperature.

**Why it might matter here.** Calibration is a board axis (ECE top-3 target; Clef-flash's ECE is unpublished and we measure it) and selective risk (G5) needs a score that *reorders* errors, which monotone calibration cannot do (Rabanser & Papernot, 2510.20242, V(repo)). A posterior variance is a candidate reorderer that costs nothing at inference.

**First experiment ($0, CPU).** Add the head beside the integrator in `heads/heads.py` with the same shape, equivariance and attribution-sum tests; on tiny backbones with synthetic schemas (`train/cohorts/`), compare AUGRC (2407.01032, V(repo)) of variance vs max-probability vs margin as selection scores, and debiased ECE with the variance-driven temperature vs the fitted predicted temperature (`train/calibration.py`).

**GPU cost.** One m2 cell on the selected parent.

**Plugs into.** `heads/heads.py`, `heads/model.py`, `train/calibration.py`, `harness/calibration.py`, `harness/selective.py`. Branch: `collab/<you>/pgm-filter-head`.

**Read first.** 2511.21016 (V) for the KF-with-identity-covariance framing; 2510.20242, 2407.01032, 2505.15008 (V(repo), `techniques/06`); Kuleshov & Liang NIPS 2015 (V(search)).

**Risks and judgement.** A learned variance is a new target for calibration attacks (G7's arm, 2401.02718 V(repo)): text that moves confidence without moving the decision. It may simply reproduce temperature scaling. "Kalman" is nominal unless you state the noise model. Payoff judgement: low–moderate on ECE; the honest selling point is exact accounting with a built-in uncertainty.

**No-go.** The variance does not reorder errors (AUGRC no better than softmax response) on the tiny-backbone cohort, or ECE gains vanish under the calibration-attack arm.

### A7. Conformal prediction and calibration on PGM outputs

**What it is.** Calibrating marginal queries of a structured predictor (Kuleshov & Liang, NIPS 2015, V(search)); conformal prediction sets over *whole* structured outputs, e.g. sets of label sequences with finite-sample coverage (Singer, Sengupta, Pazdernik, 2601.16999, V, Jan 2026, NER); conformal risk control and Learn-then-Test for selective risk (2208.02814, 2110.01052, 2512.12844, V(repo), already the G5 machinery); circuits with tractable uncertainty for shift detection (2302.06544, V).

**Why it might matter here.** If any of A1–A3 lands, its marginals need G7 reporting and its act/abstain needs G5, and a joint-assignment conformal set ("the set of whole-request answers with 95 % coverage") is a reporting object no competitor publishes. Deployment-only: the Index scores abstention as wrong.

**First experiment ($0, CPU).** On existing prediction files (the Clef-flash baseline's raw predictions if `job-sgihj` exported them; README notes its final composite step crashed on a side file, so check issue #3 first; otherwise the keyword stand-in decider's predictions from `receipts/2026-10-06-D-g4g5-standin/`), compute per-question debiased/Smooth ECE vs row-level calibration, and joint conformal set sizes at 90 % coverage with cluster bootstrap by benchmark (`harness/bootstrap.py`, `harness/calibration.py`).

**GPU cost.** None.

**Plugs into.** `harness/calibration.py`, `harness/selective.py`, `safety_eval/metrics.py` (reporting only; PREREG rows are append-only by Track D). Branch: `collab/<you>/pgm-conformal`.

**Risks and judgement.** Exchangeability fails under attack-goal shift (the prior lab saw 11.7–20.2 % realised risk at nominal 5 %; calibrate on attacked data, 2511.18562 V(repo)); the board's own ECE sample scope is undisclosed, so board-comparable ECE stays provisional. Payoff judgement: moderate as infrastructure; not a score lever.

**No-go.** Joint conformal sets are vacuous (cover > 50 % of assignments) at 90 % coverage on DEV.

---

## Part B — Systems

### B1. Fast RWKV-7 / GDN kernels via `fla`

**What it is.** The vendored RWKV7-13.3B uses a pure-PyTorch WKV kernel: measured prefill 3.6–4.3k tok/s on H100 (≈ 11.5 % of bf16 peak, `jobs/PERF_AUDIT.md`). The code exposes a hook, `RWKV7_WKV_FUNCTIONS["fla"]`, for flash-linear-attention's `chunk_rwkv7`. The `fla` repository ships RWKV-7 kernels (since 2025-01) and Gated DeltaNet / GDN-2 kernels (github.com/fla-org/flash-linear-attention, V: page fetched 2026-10-06).

**Why it might matter here.** Efficiency axis and a hard blocker: m2 child evals need ≥ ~12k tok/s to fit their 50-min cap, and a 61 M-token factorial cell at 4.3k tok/s is ~12 h instead of the specified 5.75 h. If RWKV7 is the parent, nothing in m2 moves without this.

**First experiment ($0, CPU).** Pure-torch `chunked` vs `eager` paths are already tested; write the registration shim and a CPU test that the shim falls back cleanly when Triton is absent. Pre-register the bf16 tolerance against the exact `eager` path on fixed sequences.

**GPU cost.** ≤ $20 H100 numeric + throughput gate (same sequences, bf16, pre-registered tolerance; `jobs/common/fla_guard.py` exit codes). Needs a `spend` issue and the owner's OK.

**Plugs into.** `convert/gdn.py`, `third_party/rwkv7_hf` (unmodified vendored code; register through the hook only), `jobs/rwkv7/probe_rwkv7.py` (measures `small_call_s`, `pack_call_s`), `jobs/PERF_AUDIT.md` §Kernels. Branch: `collab/<you>/rwkv7-fla-kernel`.

**Read first.** RWKV-7 2503.14456 (V(repo)); `jobs/PERF_AUDIT.md`; PR #41 and `fla` issue #640 (Hopper/Triton incompatibility that stopped `job-jbw22`; no URL recorded in the repo).

**Risks and judgement.** Triton/torch version traps (issue #640 happened once already); RWKV7 may not be chosen as parent, which lowers leverage but does not zero it (stage B LoRA pilot still needs it). Payoff judgement: high probability it works; moderate leverage.

**No-go.** bf16 mismatch beyond the pre-registered tolerance, or < 2× speedup over the pure-PyTorch path.

---

## Part C — Guarantees

### C1. G16 capability cost

**What it is.** G16 (`controlsafe/g16.py`) is a protected judge: `logit_k = b_k + a_k · E_k`, with `a, b` computed from the trust-gated P stream only (bitwise invariant to untrusted text) and `E_k = c·tanh(W_e h_D[k])` a bounded evidence channel from the full stream. It gives a STRUCTURAL veto, a CERTIFIED radius `ρ₂ = min_j μ_ij / √(‖a_i‖² + ‖a_j‖²)` and, when `b_i − b_j > c(‖a_i‖₁ + ‖a_j‖₁)`, a CERTIFIED independence (G4 at zero recompute). It is built and CPU-tested on random weights only; the independence rate is 0 with random weights, which says nothing about trained heads. The open question: what does it cost in Index points at evidence rank r ∈ {8, 16, 32}?

**Why it might matter here.** It is the one guarantee in the repo that could turn out to be *free*, and the project's novelty claims (SAFETY_SPEC §4) rest on publishing the cost of certificates, not just the certificates.

**First experiment ($0, CPU).** Train tiny heads (G16 arm vs D-head arm) on the synthetic development cohort via `train/run_cell.py` with `TrainConfig.controlsafe = {"trust_gate": true, "bridge_rank": 8, "trust_policy": "provenance"}`; report `trust_gate.capability_cost` with paired CIs (`harness.bootstrap.compare`) and the independence rate over 10 seeds. If the independence rate stays ≈ 0 after training on tiny models, note it; it does not condemn the 9B but it lowers the prior.

**GPU cost.** One m2 cell (≈ $12–17 on RWKV7, $4.7–6.6 on Q9-AF) after parent selection.

**Plugs into.** `controlsafe/g16.py`, `controlsafe/trust_gate.py`, `safety_eval/controlsafe_arms.py::ARMS`, a `train/` arm. Branch: `collab/<you>/g16-capability-cost`.

**Read first.** `controlsafe/CONTROLSAFE.md` (review items: the `request.ssdm` declaration is caller-forgeable; the P-only judge is nearly a per-position prior under the `provenance` policy); PACT 2605.11039 (V: injection is dangerous only when it determines an authority-bearing argument) and the inseparability theorem 2606.27567 (V(repo)) for why the out-of-band half is the only place the guarantee can live.

**Risks and judgement.** Under the `provenance` policy P sees template text plus declared host instructions only, so the P-only judge may be a weak prior and the independence condition rarely holds; the certificate would then be real but rarely applicable. Payoff judgement: moderate; either answer (free or priced) is a result the spec needs.

**No-go.** Lower CI of Δcap < −1.0 Index point at every r on the 2,008-row DEV subset *and* independence rate < 1 % on real decisions.

### C2. Diagonal / contraction-constrained certified layer swaps for the Pareto map

**What it is.** Replace converted GDN layers with (a) diagonal Mamba-style layers, for which per-layer Gramians, Hankel singular values and in-training balanced truncation are well defined (CompreSSM 2510.02823, V; HSVR 2510.22951, V, NeurIPS 2025), or (b) contraction-constrained layers with ‖A‖∞ < 1, which admit exact interval-bound-propagation certificates for small safety heads (Thapliyal, 2610.02853, V, Oct 2026). Measure the Index drop against the certified decay bound and Gramian observability, and plot the curve.

**Why it might matter here.** It is a clean slice of the three-axis frontier: how much capability does a *certified* persistence bound cost? The prior lab's operator-norm cap worsened adherence (9 → 17 forbidden choices), so SAFETY_SPEC lists this as an arm with a measured tax, never a guarantee of the parent (§5.2). Note the correction recorded in `CONTROLSAFE.md`: an induced-norm bound ‖A_t‖ ≤ ρ < 1 *does* compose across switching (GDN's `‖A_t‖₂ = α_t` is already CERTIFIED); the earlier counterexample was about spectral radius.

**First experiment ($0, CPU).** On tiny models (`convert/model.py` destination modes), sweep ρ and diagonal vs rank-1 transitions; report ppl ratio and MQAR against the bound from `controlsafe/persistence.py` and Gramian effective rank from `controlsafe/gramian.py` (dense S×S; at 9B you would need `JᵀJ` or randomised trace estimates).

**GPU cost.** ≤ $20 for one swap on the 0.8B with the accept/rollback gate.

**Plugs into.** `convert/model.py`, `convert/gdn.py`, `controlsafe/persistence.py`, `controlsafe/gramian.py`, `probes/core.py`. Branch: `collab/<you>/certified-layer-swaps`.

**Read first.** 2610.02853, 2510.02823, 2510.22951 (V); Cao, *Optimal Decay Spectra for Linear Recurrences* (PoST), 2604.07658 (V) for why decay spectra matter; `reviews/02` §4–6 and `reviews/03` §1.4 for the lab's negative results.

**Risks and judgement.** Gramian tools are per-layer LTI objects and do not apply to GDN's matrix state; the certified variant is small-model evidence from one author; capacity loss is expected. Payoff judgement: the curve is publishable regardless; the chance that a certified layer costs < 1 Index point is low.

**No-go.** Any contraction-constrained swap at ρ < 0.999 loses > 3 MMLU-subset points or collapses MQAR on the 0.8B. Then it is a priced arm, documented, done.

### C3. Persistence / Lyapunov / ISS bounds for selective SSMs (the RWKV-7 open problem)

**What it is.** CONTROLSAFE proves `‖A_t‖₂ = α_t ≤ ρ_h < 1` for GDN (CERTIFIED for every input) but for RWKV-7, `A_t = diag(w_t) − (kk_t·a_t) kk_tᵀ`, a counterexample refutes any *common 2-norm* bound below 1 (σ_max 1.134 at spectral radius 0.99); only the diagonal decay is bounded. Open: a weighted norm or joint-spectral-radius bound that composes, or a proof that none exists for the trained parameter ranges. The 2025–26 control-theory literature now treats exactly this object: Zubić & Scaramuzza prove exponential forgetting under strict dissipativity and sufficient conditions for global input-to-state stability uniformly over selection schedules, and turn them into an LMI training regulariser (2505.11602, V, v3 Jul 2026); Halloran et al. prove Lyapunov stability of Mamba/Mamba-2 recurrences under mixed-precision and PEFT perturbations (2406.00209, V, TMLR); Cirone et al. give the signature-projection view of selective recurrences (2402.19047, V, NeurIPS); StableSSM's stable reparameterisation (2311.14495, V, ICML 2024); consensus dynamics with local exponential stability for time-varying weights (2609.17997, V, Sep 2026).

**Why it might matter here.** It would upgrade the persistence row for RWKV-7 from EMPIRICAL to CERTIFIED and give the G14 alarm a principled threshold. Honesty rule: per SAFETY_SPEC §5.2 no decay or contraction number is a *safety* claim; this is a cost/persistence certificate and telemetry.

**First experiment ($0, CPU).** Pure math plus `tests/controlsafe`: search for a diagonal weighting D such that `‖D A_t D⁻¹‖₂ < 1` over the parameter ranges the vendored RWKV-7 actually produces (sample `w_t, kk_t, a_t` from the tiny and, offline, from the 13.3B's recorded activations if a receipt provides them); or produce the counterexample family and bound the joint spectral radius numerically (Jungers, *The Joint Spectral Radius*, Springer 2009, book, no arXiv).

**GPU cost.** None.

**Plugs into.** `controlsafe/persistence.py`, `controlsafe/adapters.py`, `tests/controlsafe/`, `CONTROLSAFE.md` tier table (state the tier exactly). Branch: `collab/<you>/rwkv7-contraction`.

**Read first.** 2505.11602, 2406.00209, 2402.19047, 2311.14495, 2609.17997 (all V); `controlsafe/CONTROLSAFE.md` §Per-layer decay.

**Risks and judgement.** The clean theorem may not exist; a bound that holds only for a parameter sub-range is still useful if the trainer can enforce the range (the LMI regulariser route), but that is a training arm with a tax. Payoff judgement: moderate; a negative result with a proof is a receipt too.

**No-go.** A counterexample under every diagonal weighting within the observed parameter ranges, with the joint spectral radius numerically above 1. Record it and move on.

### C4. Hidden-state poisoning defences (G14 is weak)

**What it is.** HiSPA (Le Mercier, Develder, Demeester, 2601.01972, V, EMNLP 2026 Findings): short trigger phrases drive a Mamba-class recurrence into a contracting regime and erase the context; the 52B hybrid Jamba-1.7-Mini collapses on RoBench-25 while pure transformers do not. Detectors: SpectralGuard (spectral radius of the discretised transition; F1 0.961; 2603.12414, V(repo)+V(search)), CLASP (token-level classifier on block outputs; 99.3 % document F1; 2603.12206, V(repo)+V(search)); attack taxonomy in 2604.16424 (V(repo)). Our own G14 retained-norm detector: 10 joint seeds, RWKV-7 mean AUROC 0.57 [0.43, 0.72] (5/10 seeds at or below chance), GDN 0.86 [0.78, 0.93]; target ≥ 0.9 not met on either family; the seed-0 "0.96 / 0.91" was withdrawn. The G15 P stream is immune by construction (tested).

**Why it might matter here.** G14 is the recurrent-specific row the transformer competitors do not face, and Clef-flash (24 GDN + 8 attention) is a hybrid of exactly the kind that collapsed, so it is the most informative baseline row in SAFETY_SPEC §7. Also a scheduling fact: the collapse probe runs at m1 on every candidate parent because it decides between pure-recurrent and hybrid as much as recall does.

**First experiment ($0, CPU).** Add CLASP-style (token classifier on block outputs) and SpectralGuard-style (per-layer effective-radius) detectors to `controlsafe/monitor.py` and run `safety_eval.controlsafe_arms --arms g14` on the same 10-seed tiny-model protocol; report AUROC with bootstrap CIs over seeds, plus an *adaptive* trigger that also minimises the detector score (currently unevaluated). Build `safety_eval/collapse.py` (trigger families, long-record construction with decisive evidence before the trigger, paired accuracy) against the keyword stand-in decider.

**GPU cost.** Collapse probe on the parents ≈ $6–9 (≈ 200 long requests × 10 triggers × 3 parents at 4–8k tokens), Clef-flash ≈ $3; both need `spend` issues and the owner's OK.

**Plugs into.** `controlsafe/monitor.py`, new `safety_eval/collapse.py`, `heads/backbone.py` telemetry hook, `safety_eval/controlsafe_arms.py::ARMS`. New attack families after test scoring are labelled post-hoc and never replace a PREREG row. Branch: `collab/<you>/g14-collapse`.

**Read first.** 2601.01972 (V), 2603.12414, 2603.12206, 2604.16424 (V(repo)); 2607.17117 Persistent SAEs (V(repo): slow features preserve injection signal); `controlsafe/CONTROLSAFE.md` §G14.

**Risks and judgement.** The trigger objective and the detectors both measure decay, so a good AUROC can be a pipeline check rather than detection power; tiny-model numbers are pipeline qualification, never model evidence (PREREG §0); the published detectors were not evaluated against an adaptive attacker that minimises their score. Payoff judgement: moderate–high that a CLASP/SpectralGuard-style detector beats 0.57/0.86 on the CPU protocol; the real question (does our trained parent collapse?) is unknowable without the GPU probe.

**No-go.** An adaptive trigger that also minimises the detector score drives AUROC to chance on the GDN tiny models. Then detection is not the answer and the structural route (G15/G16) is.

---

## Part D — Interpretability

### D1. Mechanistic interpretability of recurrent state

**What it is.** The repo has no `interp/` module yet (SAFETY_SPEC §6 names `interp/perturb_recompute.py` as new work for G9(b)). The 2026 tooling for GDN/Mamba/RWKV-7 state: WriteSAE (Young, 2605.12770, V) learns rank-1 write atoms shaped like the model's own cache write; the replacement test beats deletion on 92.4 % of positions and a closed-form (forget gate × read query × output embedding) predicts GDN logit change with R² = 0.98; it is the first cache-level steering result on these layers. The state lens (Wang & Reid, *Residual Streams Read, Recurrent States Remember*, 2609.32102, V, Sep 2026): recurrent state is a complementary carrier of workspace content whose recovery, persistence and causal use need separate measurements. Method rules: causal localisation in Mamba-2 is non-unique (Jiang & Zhang, 2606.00930, V), so every mechanism claim carries alternative-circuit and output-only controls; alignment maps must be linear (2507.08802, V(repo)); identification assumptions are disclosed (2605.08012, V(repo)); fidelity is measured with anytime-valid confidence sequences (2607.08349, V(repo)).

**Why it might matter here.** G9(b): attribution counts as evidence only if erasing the top-attributed spans and recomputing the full recurrence reproduces the predicted Δlogit (Spearman ≥ 0.8, sign agreement ≥ 90 % on ≥ 500 decisions). Provenance audit: which write atoms fire on untrusted tokens and how long they persist is the recurrent-specific question the injection literature cannot ask of transformers. Both feed the interpretability axis directly.

**First experiment ($0, CPU).** Train a WriteSAE on the writes of a tiny all-GDN model (`heads.backbone.tiny_hf_backbone`) over synthetic records with injected untrusted spans; run the replacement test and the GDN closed-form predict-before-intervene check; build `interp/perturb_recompute.py` against the integrator head on the tiny model with the brute-force controls.

**GPU cost.** SAE on the 9B's GDN writes over ~100M tokens ≈ a few GPU-hours ($10–30); linear DAS on head inputs ≈ $3–6.

**Plugs into.** New `interp/` (owner has named the file), `heads/heads.py::erase_token_effect`, `heads/backbone.py` hooks (custom hooks on the fla modules; pyvene/nnsight coverage of GDN is **[unverified]**). Branch: `collab/<you>/interp-writesae`.

**Read first.** 2605.12770, 2609.32102, 2606.00930 (V); 2303.02536 DAS, 2507.08802, 2605.08012, 2607.08349 (V(repo)); `techniques/03` and `04`.

**Risks and judgement.** SAE features are not guaranteed causal; the published results are on ≤ 1B models; every "we found the mechanism" sentence needs the 2606.00930 controls. Payoff judgement: moderate; the project explicitly needs this module and has none.

**No-go.** The replacement test on our GDN layers does not beat deletion, or linear DAS IIA on head inputs is no better than the random-map control.

---

## Part E — Language track

### E1. Track L semantic-program learning assets (L0h → L2)

**What it is.** Track L (Codex) has merged the CPU prerequisites (typed kernel, renderer, planner, fixtures, evaluator, public-output wrapper) and has not started the learned levels: L0g (≤ $20 hardware qualification of the real parent adapter) and L0h (frozen parent as planner over ReadFact/Compare/Assert/Ask, ≤ 8 nodes, ≤ 4 decision calls). Three things a collaborator can build without touching the owner's seam: (i) the **independent audit set** the plan calls for (1,000–2,000 cases grouped by evidence world, with necessary-fact pairs, nuisance paraphrases, malicious directives, unit/currency/time and open-world negation traps; human-authored labels), (ii) the **causal edit-pair corpus** for the L2 ablation (how changing one fact should change the program and answer, and which conclusions must stay stable), (iii) a **GF-style rung-A grammar** whose abstract syntax carries scope, tense, number and time *before* L1 (the `techniques/07` finding: LLMs realise MRS at 76 BLEU but parse it at ~1 % exact match, so round trips go through a symbolic parser, 2609.30071 V(repo)).

**Why it might matter here.** The goal's four per-level gates (useful-answer capability, no decision regression, relation-level soundness, causal semantic-intervention fidelity) are only as good as the audit set; CaMeL-shaped systems (2503.18813, V(repo)) are the family and their utility losses are known; a grammar with scope is what separates G11 from a template filler.

**First experiment ($0, CPU).** Author 50 audit cases in the fixture format with the pairing discipline above; run them through the oracle, replay and rule-baseline backends and the free-running scorer; report which cases the oracle program cannot even express (then the semantic language must grow before training).

**GPU cost.** None for assets. L0g (≤ $20) and L0h are Codex's spend lines.

**Plugs into.** `fixtures/language_growth/`, `language_growth/eval.py`, new grammar module under `language_growth/`; anything touching `interfaces/DECISION_LANGUAGE_V0.md` needs an `interface` issue and both owners. Branch: `collab/<you>/trackl-audit-set`.

**Read first.** `plans/codex/LANGUAGE_GROWTH_PLAN.md`, `interfaces/DECISION_LANGUAGE_V0.md`, `techniques/07`; 2503.18813, 2609.30071 (V(repo)); Ranta, *Grammatical Framework*, CSLI 2011 (book).

**Risks and judgement.** Parent not selected; a proof-valid answer to the wrong question is a task failure the kernel cannot see; audit labels must be independent of the model under test. Payoff judgement: high for the assets (they are needed whatever the model does), uncertain for learned results.

**No-go.** For L1 specifically: the parent fails *direct* answers on the catalog task; then it is a parent problem and no planner work fixes it.

---

## Part F — Data, evaluation, scope

### F1. Arts & Human Taste data sourcing

**What it is.** Arts is a fixed 10 % of the Index but 5.8 % of the 150k m2 training set; Clef-flash scores 51.3 there. The Index's Arts benchmarks use humour-edit, caption-ranking, music, colour and deliberation formats; the kit's internal names (BPoMP, Humicroedit, POP909, cfcolor, Habermas, New Yorker) are listed in `HANDOFF.md` **[unverified: I did not re-open the kit]**. Legitimate *train-split or synthetic* sources in those formats exist: Humicroedit (Hossain, Krumm, Gamon, 1906.00274, V; ~15k edited headlines with crowd ratings), the New Yorker Caption Contest benchmark (Hessel et al., 2209.06293, V), POP909 (Wang et al., 2008.07142, V, ISMIR 2020), the Habermas Machine deliberation statements (Tessler et al., *Science* 386(6719), 2024, doi:10.1126/science.adq2852, V(search); no arXiv ID).

**Why it might matter here.** Capability on an under-sourced area; cheap; no GPU.

**First experiment ($0, CPU).** Add sources to `data/sources.py::REGISTRY` (pinned URL/revision, files, licence, Index-overlap note), a converter in `data/convert.py`, a row in `data/SOURCES.md`; run `data/dedup.py` against all 155,390 Index rows and commit the log (prior builds dropped 14.5 % and 11.6 % for item-tier overlap); rebuild and check byte-identical rebuilds; sha256 lock in `data/manifests/sources.lock.json`.

**GPU cost.** None.

**Plugs into.** `data/` only. Train splits only. Branch: `collab/<you>/arts-data`.

**Risks and judgement.** Licences; item-tier overlap with the Index's own Arts rows (the guard will tell you); "taste" labels are noisy; whether the extra data moves the Arts score is unknown until an m2 cell runs. Payoff judgement: moderate–high that the build lands at ≈ 10 % of the mix; unknown on score.

**No-go.** Dedup drops > 50 % of a source's candidates, or a licence forbids training use.

### F2. Jev evaluation protocol

**What it is.** A harness-side `decide()` backend for TypeSafe's Jev API so the safety battery (D1-B steer, hijack best-of-5, GCG-lite, calibration under attack, G1/G2(b)/G8 probes) runs with the same units, seeds and cluster-bootstrap draws as on Clef and our candidate. Writing the adapter is $0; *calling* the API is the owner's decision: TypeSafe's MCA §2.3(b) bars using Jev outputs to build or "facilitate" a competing product, and the Acceptable Use Policy is unread (SAFETY_SPEC §7.5). Jev is never a teacher, label source or reference distribution.

**Why it might matter here.** Every Jev row today is filled from published numbers only (board 57.91 / ECE 0.074; Wu & Lim's decision hijacking 2609.28613, V(repo): base ASR 1.8 %, adaptive 3.5 % on 3 options; *Hidden Risks of Jev* 2610.04985, V(repo)). The protocol would make the "same report against Jev" in `GOAL.md` possible.

**First experiment ($0, CPU).** Adapter against a recorded-response fixture (no live calls), conformance tests, and the `decision-needed` issue with the AUP fetched and attached.

**GPU cost.** None; API cost **[unverified]**.

**Plugs into.** `safety_eval/` decider interface, `baselines/`. Branch: `collab/<you>/jev-protocol`.

**Risks and judgement.** Approval may not come; outputs must never be stored where they could become training data. Payoff judgement: certain to produce the comparison row if approved; approval uncertain.

**No-go.** The owner declines, or the AUP forbids evaluation.

### F3. Multimodal / vision SSM encoder

**What it is.** Clef has a vision path and a routing head that use attention; our claims are text-only until a vision gate passes (`PLAN.md`). The 2024–26 literature has Mamba-based VLMs (VL-Mamba 2403.13600, V; Cobra 2403.14520, V), distillation of a quadratic decoder-only VLM into a linear one (mmMamba, 2502.13145, V: 20.6× speedup, 75.8 % memory reduction at 103K tokens), and a controlled study finding SSM vision backbones competitive with or stronger than ViTs under matched initialisation on VQA and grounding (Kuo & Cascante-Bonilla, 2603.19209, V, Mar 2026).

**Why it might matter here.** Scope, not score: it decides whether "attention-free" can ever be claimed beyond the text path, and whether any image-bearing Index rows exist that we currently forfeit.

**First experiment ($0, CPU).** Check the kit (`harness/INDEX_NOTES.md`, suite rebuild) for image-bearing rows and their weight; write the backbone spec showing how a vision encoder would feed the recurrent state while G1, G2 and G8 stay structural.

**GPU cost.** Later and real ($$).

**Plugs into.** `heads/backbone.py::BACKBONES` (new `BackboneSpec`), `PLAN.md` claim scope (owners). Branch: `collab/<you>/vision-ssm-design`.

**Read first.** 2603.19209, 2502.13145, 2403.14520, 2403.13600 (all V).

**Risks and judgement.** Re-scoping the attention-free claim is an owner decision; the Index may score no images. Payoff judgement: low near-term leverage for the goal; strategically interesting.

**No-go.** The kit has no image-scored rows. Park.

---

## Part G — Side paper

### G1. SLT / LLC side paper with the lab's negative SGLD results

**What it is.** Singular learning theory's local learning coefficient (LLC) and susceptibilities are proposed as developmental-stage detectors. This lab's prior results (`reviews/03` §1.3, `techniques/08`): SGLD susceptibility estimates biased 1.3–3.3× with the bias invisible to loss-trace R-hat/ESS; an RMSProp-SGLD run passed the loss rule while not mixing (R-hat 9.4); per-example susceptibilities not estimable at 1.7B-LoRA; prospective reweighting moved loss but not decisions. Standing policy: LLC is logged, never gated. Timaeus' own benchmark (2507.21449, V(repo)) still picks RMSProp-SGLD as most faithful at O(100M) params, and influence dynamics (2510.12071, V(repo), ICLR 2026) would call a sampler's sign flip a developmental transition rather than an estimator failure. That disagreement is the paper.

**Why it might matter here.** Nothing on the three axes; it is a publishable negative result and a service to the field. Be clear-eyed that it does not move the Index goal.

**First experiment ($0, CPU).** Tiny networks with an exact quadrature reference for the LLC; batch-aligned SGLD with the lab's Lyapunov bias check and SIVE's same-state noise correction (2606.22389, V(repo)); reproduce the bias and show R-hat/ESS do not see it.

**GPU cost.** $ at ≤ 1B for the checkpoint-card logging arm; unreliable at 9B-LoRA per the catalog.

**Plugs into.** `train/evalcard.py` (logged column), `techniques/08`. Branch: `collab/<you>/slt-side-paper`.

**Read first.** 2308.12108, 2402.03698, 2507.21449, 2510.12071, 2606.22389 (V(repo), `techniques/08` §8.3).

**Risks and judgement.** If the bias disappears under Timaeus' recommended settings on our own reproduction, the "negative result" becomes a settings artefact (still reportable, less interesting). Payoff judgement: moderate as a paper, zero on the goal.

**No-go.** None needed; it is bounded by the budget you give it.

---

## General caveats (read before choosing)

- **Epistemic humility about the project itself.** No candidate model has an Index number. The parent is not selected. Only Clef-flash has been baselined on our harness; Clef-27B and the safety battery on either (jobs B, C) have not run, so no "better than Clef" sentence is possible yet for anyone. Every row of `SAFETY_SPEC.md` is a target until its named receipt lands; the G14 monitor is weak and unvalidated; the RWKV-7 contraction question is open; the board's ECE sample scope is undisclosed. Treat the status tables as of 2026-10-06 and re-read `LEDGER.md` and issue #3 before relying on them.
- **Vendor- and single-group-reported numbers are not ours.** Clef-flash ≈ 57.07 is a secondary source (our measured 56.8 agrees); Cloudflare's own comparisons were run on TypeSafe's suite and publish no safety numbers; the "5.9 % / 76 %" Clef figure in circulation is a third-party blog test. Gated KalmaNet and Priming are one industry group's preprints; Steerling-8B's "negative capability tax" has no third-party evaluation; several PGM and SSM-stability papers cited above are weeks to months old (P grade). Read the evidence legend in `techniques/CATALOG.md` (R/S/P/X) and grade what you cite.
- **Small-scale results are pipeline qualification, not evidence.** PREREG §0 and SAFETY_SPEC §5 item 12: smoke runs, tiny random backbones and stock ≤ 1B models qualify code paths; they never support a claim about the 9B/13B parents. Every $0 experiment in this menu is of that kind. Say so in your receipt.
- **The TEST slice is sealed.** `harness/cli.py` refuses to load it without `SSDM_UNSEAL_TEST=1` and a committed `SSDM_PREREG`; every unseal is logged. Development uses the DEV slice, the `safety_eval` dev split and the independent development cohort; selection on the Index DEV slice is itself discouraged (G12: it is a scheduled readout, not a selection signal). Never on Index test items, never on the held-out D1-B templates.
- **No shared-credit spend without the owner's OK.** Collaborators do not spend the $525 SF credits without an explicit go-ahead in a `spend` issue that states the estimate and stop condition, with a planned `LEDGER.md` row, a runtime ceiling and a workspace cap. Climb the ladder: $0 CPU → ≤ $20 qualification → capped experiment → scale-up, each after the previous receipt merges. Never run GPU work on a laptop.
- **Claims are scoped to what was measured.** "Attention-free" is disclosed with lineage ("Qwen3.5-9B with attention layers distilled to delta-rule") and text-only scope; no decay, Gramian or contraction number is a safety claim; "certified" never means "correct"; learned defences are empirical. Negative results and false positives go in `LEDGER.md` with their receipts, and a win is only a win after someone else re-derives it from the committed receipt.
- **Interfaces are owner-only.** `interfaces/`, `INTERFACE.md`, `decide/schema.py` wire shapes, `safety_eval/PREREG.md`, `SAFETY_SPEC.md` rows, `harness/manifests/` and `train/promotion.py` margins change only through an `interface` issue, both owners' approval and a version bump. Several PGM options (A2, A4) sit right next to that seam; say so in an issue before you need it.

---

## Appendix — citation status

**V** = I fetched `arxiv.org/abs/<id>` on 2026-10-06 and matched title/authors. **V(repo)** = fetched and matched by the repo's own notes (`techniques/09`, `CATALOG.md`) on 2026-10-06; I did not re-fetch. **V(search)** = title and identifier matched in search results or a publisher page, not an arXiv abstract page. **[unverified]** = not checked.

| ID / source | Short title | Status |
|---|---|---|
| 1011.4088 | Sutton & McCallum, Intro to CRFs | V |
| 1301.6725 | Murphy, Weiss, Jordan, Loopy BP | V |
| 1106.1819 | Darwiche & Marquis, Knowledge Compilation Map | V |
| 1511.06350 | Belanger & McCallum, SPEN (ICML 2016) | V |
| 1711.11157 | Xu et al., Semantic Loss (ICML 2018) | V |
| 2206.00426 | Ahmed et al., Semantic Probabilistic Layers | V |
| 2011.04640 | Chiu & Rush, Scaling HMM LMs (EMNLP 2020) | V |
| 2201.02715 | Chiu, Deng, Rush, Low-Rank Constraints (NeurIPS 2021) | V |
| 2304.07438 | Zhang et al., GeLaTo | V |
| 2406.13892 | Zhang et al., Ctrl-G | V |
| 2511.16054 | Yidou-Weng et al., LTLA | V |
| 2606.01926 | Dang et al., Tractable Proposals (ICML 2026) | V |
| 2607.20483 | Scribner, Vergari, Belle, Tractable Hierarchical Control | V |
| 2609.32900 | Su & Zhang, FactorDLM | V |
| 2406.00766 | Liu, Ahmed, Van den Broeck, PyJuice | V |
| 2302.06544 | Ventola et al., PCs That Know What They Don't Know | V |
| 2501.07021 | Chen et al., Neural Probabilistic Circuits | V |
| 2605.12940 | Zhao et al., Expressivity Boundary of PCs | V |
| ProbCirc20 (starai.cs.ucla.edu) | Choi, Vergari, Van den Broeck, PCs: A Unifying Framework | V(search) |
| Kuleshov & Liang, NIPS 2015 | Calibrated Structured Prediction | V(search) |
| 2601.16999 | Singer et al., Conformal NER sets | V |
| 2601.13357 | Ghojogh et al., SSMs and HMMs | V |
| 2602.17744 | Zhang & Xing, Bayesian Optimality of ICL with Selective SSMs | V |
| 2604.23818 | Tang et al., Selective SSMs for Filtering Unknown Systems | V |
| 2511.21016 | Peng et al., Gated KalmaNet | V |
| 2605.08301 | Chattopadhyay et al., Priming | V |
| 2602.10743 | Shaj et al., Kalman Linear Attention | V |
| 2502.10178 | Bondaschi et al., Mamba In-Context Learns Markov Chains | V |
| 2312.06937 | Goel & Bartlett, Transformer ↔ Kalman Filter | V |
| 2505.00818 | Chang & Mehta, Dual Filter for HMMs | V |
| 2402.19047 | Cirone et al., Foundations of Deep Selective SSMs | V |
| 2311.14495 | Wang & Li, StableSSM | V |
| 2505.11602 | Zubić & Scaramuzza, Stability of Selective SSMs | V |
| 2406.00209 | Halloran et al., Mamba Lyapunov-Stable Learners (TMLR) | V |
| 2609.17997 | Silvestre et al., Consensus Dynamics in Selective SSMs | V |
| 2510.02823 | Chahine et al., CompreSSM | V |
| 2510.22951 | Schwerdtner et al., HSVR (NeurIPS 2025) | V |
| 2604.07658 | Cao, Optimal Decay Spectra (PoST) | V |
| 2610.02853 | Thapliyal, Contraction-Constrained SSMs | V |
| 2605.12770 | Young, WriteSAE | V |
| 2609.32102 | Wang & Reid, Residual Streams Read, Recurrent States Remember | V |
| 2606.00930 | Jiang & Zhang, A Circuit, Not The Circuit | V |
| 2601.01972 | Le Mercier et al., HiSPA | V |
| 2603.12414 | Bonetto, SpectralGuard | V(repo), V(search) |
| 2603.12206 | Le Mercier, CLASP | V(repo), V(search) |
| 2604.16424 | Parmar, Safety/Security/Cognitive Risks in SSMs | V(repo) |
| 2605.11039 | Fan et al., PACT | V |
| 2606.27567 | Pant et al., Inseparability theorem | V(repo) |
| 2609.28613 | Wu & Lim, Decision Hijacking on Jev | V(repo) |
| 2610.04985 | Wang et al., Hidden Risks of Jev | V(repo) |
| 2503.18813 | Debenedetti et al., CaMeL | V(repo) |
| 2609.30071 | Dan, LLMs realise the MRS they cannot parse | V(repo) |
| 2303.02536 / 2507.08802 / 2605.08012 / 2607.08349 | DAS; non-linear representation dilemma; identification assumptions; anytime-valid fidelity | V(repo) |
| 2607.17117 | Persistent SAEs | V(repo) |
| 2412.06464 / 2503.14456 | Gated DeltaNet; RWKV-7 | V(repo) |
| 2208.02814 / 2110.01052 / 2512.12844 / 2511.18562 / 2401.02718 / 2510.20242 / 2407.01032 / 2505.15008 | CRC; LTT; Selective CRC; calibrate under attack; calibration attacks; selective-classifier gap; AUGRC; LR-optimal selection | V(repo) |
| 2308.12108 / 2402.03698 / 2507.21449 / 2510.12071 / 2606.22389 | LLC; LLC at scale; local-sampler benchmark; influence dynamics; SIVE | V(repo) |
| 2603.19209 / 2502.13145 / 2403.14520 / 2403.13600 | SSMs as vision encoders; mmMamba; Cobra; VL-Mamba | V |
| 1906.00274 / 2209.06293 / 2008.07142 | Humicroedit; New Yorker Caption Contest; POP909 | V |
| doi:10.1126/science.adq2852 | Tessler et al., Habermas Machine (Science 2024) | V(search) |
| github.com/fla-org/flash-linear-attention | RWKV-7 and GDN/GDN-2 kernels present | V (page) |
| Jungers 2009; Ranta 2011 | Joint Spectral Radius; Grammatical Framework | books, no arXiv |
| GKA kernels in `fla`; pyvene/nnsight GDN support; Index Arts benchmark names; Jev API cost; `fla` issue #640 URL | — | **[unverified]** |

# Inputs under review (pasted by user, 2026-10-05)

User goal: build an SSM-architecture "System 1" decision model that outperforms Cloudflare's Clef family (Clef, Clef-flash) and TypeSafe's Jev, with alignment + interpretability built in from the start, "no tradeoff in control/interpretability for intelligence". Then a second project: a controlled, aligned LLM built off it. Budget: up to $5k compute/data.
Source pages: https://blog.cloudflare.com/clef-decision-models/ and https://typesafe.ai/blog/introducing-system-one-models-and-jev

---------------------------------------------------------------------
## DOCUMENT A — v1 plan: "Decision-First, Glass-Box Language Models on Attention-Free SSM Backbones"
---------------------------------------------------------------------

Thesis: Build an attention-free diagonal-SSM decision model copying the Jev/Clef output contract (typed noul/choice/score, softmax over bounded option set, calibrated probabilities). Grow language as a verified-emission layer: SSM picks typed decisions over knowledge set K, deterministic checker confirms, grammar-constrained realizer is the only thing that emits text. "No hallucination" = hard structural property relative to K and grammar G; decision correctness = statistical (conformal).

### Key factual claims (to verify)
1. TypeSafe launch post (Sep 15 2026, Diogo Almeida): Jev "gives up string generation"; outputs "defined in advance"; "The model never makes type errors"; plots 0% hallucination; "Our number is not empirical. Schema matching is guaranteed." "193.6x faster, 444.6x cheaper" described by TypeSafe as "on the higher end of real world gains".
2. Clef model card: backbone Qwen3.8-27B (Clef) or Qwen3.5-9B (Clef-flash) with vision encoder + "joint schema head: a small transformer head that reads the backbone's final hidden states, routes evidence from the state to each question, and scores all options of all questions jointly"; "one logit per allowed option for each question". Training: label-smoothed CE + Brier + RLCD secondary objective (partial ordinal credit, whole-record precision reward, reference penalty). Apache-2.0 weights. Synthetic data "permutating field orders, prompts, and schema structures".
3. Laya = "ModernBERT-large (395M, bidirectional, fully fine-tuned) + decision head: 2 transformer layers, option-marker scorer, act/escalate head". Kev, AutoJev/pplx-decider are Qwen/Gemma/ModernBERT/LFM fine-tunes. LFM2 is "a hybrid of convolution and attention blocks" with GQA. No pure-SSM decision model among Decision Index entrants.
4. Jared Palmer's Kev plan: Decision Index 0.2 scores "Jev 51.67 (ECE 0.065), AutoJev-27B 50.94 (ECE 0.023), Kev-9B 35.41 (ECE 0.16), Kev-4B 31.31 (ECE 0.20)".
5. CompreSSM (ICLR 2026) in-training balanced truncation on LTI SSMs (LRU); for selective models proposes "averaging the dynamics over the input space before applying the reduction scheme". Hankel-singular-value regularization (Schwerdtner, Berman, Peherstorfer, NeurIPS 2025) "up to 10x more compressible". Halloran, Gulati & Roysdon arXiv 2406.00209 "Mamba SSMs are Lyapunov-stable learners": "maximal Lyapunov exponent for an arbitrary MambaBlock is first proven to be non-positive", covers Mamba and Mamba-2. Zubić & Scaramuzza arXiv 2505.11602 "parametric LMI conditions and kernel constraints that limit gating mechanisms".
6. No published LLC/refined-LLC/susceptibility study on RNN/S4/S5/LRU/Mamba. Exact RLCTs: reduced-rank regression (Aoyagi & Watanabe 2005), deep linear nets; HMM only upper bounds (Yamazaki & Watanabe 2005); LDS none.
7. Jev API: state + questions map; types noul / choice / score; Jev "supports a cardinality up to 255", bigger sets use "2 stage-system of scoring independently then making an explicit choice". Clef `systemone` helper takes Jev POST /v1/systemone body. Clef local run: snapshot_download("Cloudflare/clef-flash") → load_release_model → encode_record (max_length 16384) → collate_records; tested torch 2.11 / transformers 5.10.2 on one H200; joint_schema_model.py. Context 64K (Jev 32K).
8. Table (Cloudflare internal run, Decision Index 0.2.1): BANKING77 macro-F1 Clef 94.2 / flash 90.9 / Jev 79.7 / Laya 14.3; CLINC150+OOS 97.4 / 66.8 / 89.3 / 3.2; BFCL case-exact 98.5 / 98.8 / 95.8 / 38.1; GPQA Diamond 48.0 / 51.0 / 78.3 / 27.6; median latency ms 209.3 / 38.8 / 524.1 / 5.8.
9. Decision Index: frozen 132,422-decision suite; edition 0.2 "averages 40 benchmarks in the same five equal-weight areas"; chance-corrected "(score − chance)/(1 − chance)"; house rules ban prompt tuning, truncation, option filtering; abstentions score zero. Harness github.com/apolinario/decision-index.
10. Mamba-3: upgrades "inspired by classical control theory"; "100% accuracy on Parity"; Gu et al arXiv 2603.15569 "Mamba-3 (MIMO) with state size 64 matches perplexity of Mamba-2 with state size 128".
11. BabyHGRN (Haller et al., CoNLL-BabyLM 2024): BLiMP 64.44 Mamba-350M, 62.64 LLaMA-360M, HGRN2 best; xLSTM-340M 64.66 (arXiv) vs 51.20 (ACL PDF).
12. BarrierSteer (Tran et al. arXiv 2602.20102, Table 3, Qwen2-1.5B): 6.08 ms/token LSE variant vs 190.67 ms SaP.
13. Ctrl-G 100% constraint satisfaction on CommonGen, cost O(n|D|H^2). Conformal factuality (Mohri & Hashimoto ICML 2024). LLM-JEPA (Huang, LeCun, Balestriero 2025). SIGReg (LeJEPA). Smékal et al 2024 frequency-domain learning dynamics of linear SSMs. arXiv 2602.12499 training dynamics of Mamba selection. devinterp library features; Timaeus renamed Resolution; stagewise-RL paper Jan 2026; Wang & Murfet patterning. Timaeus: LLC estimates "insensitive to modes below a data-dependent threshold".
14. NKI Library scan kernels (Linear Scan, Selective Scan, SSD), nki.isa.tensor_tensor_scan; awslabs/state-space-models-neuron trains Mamba-2 on Trainium.

### Architecture (DF-SSM)
- Diagonal complex LRU/S5 layers, λ = exp(−exp(ν)+iθ), |λ|<1; GLU mixing; state split h=[h_sem; h_free], head reads only h_sem (zeroed weights) → claimed "100% of decision-relevant info provably passes through semantic block".
- Input order: questions, state, questions again.
- Forked continuations: cache h* after [schema; state; question], run each option description from h* in parallel; z = wᵀφ(h_sem*,k) + α·lex + b. Claims permutation-equivariance, schema validity, O(K) cost; two-stage above 255.
- Typed heads: noul sigmoid; choice softmax with per-question T; score = cumulative-link with monotone thresholds.
- CRF/factor graph across fields with hard −∞ constraint factors; probabilistic circuit for larger F.
- Confidence: aleatoric 1−H/logK; SNGP (claims contracting Lipschitz SSM layers give bi-Lipschitz features "almost free"); conformal set size.
- Loss: CE(label smoothing) + β Brier + γ RPS − η E[R] + μ KL(p‖p_ref); distill temperature-scaled Clef-flash.
- PID-Lagrangian on smoothed ECE (setpoint 0.02) weighting a differentiable surrogate (Brier reliability / MMCE); per-question temperature as fast actuator.
- Split conformal; act iff |C(x)|=1; CRC / Learn-Then-Test for selective risk; Mondrian per schema family.
- Verified emission: plan over K → deterministic checker → grammar G × Trie(K_plan) product-automaton masked decoding with copy pointers → round-trip parse → emit/abstain. "Hard grounding theorem": every atomic claim's logical form ∈ K, independent of weights.
- Ladder rungs 0–5 (HMM → LTI → selective → matrix-state → contracting REN → unconstrained MLP in h_free), climb only on gate failure.
- Control tools: closed-form diagonal Gramians P_ij=(BB*)_ij/(1−λ_i λ̄_j), HSVs; ‖G−G_r‖∞ ≤ 2Σσ_i; selective rungs as LPV with clamped Δ → per-mode half-life bound t½ ≤ ln2/(Δ_min |Re a_i|) "how long can a prompt injection persist"; polytopic LMI Gramians; bilinear-system generalized Lyapunov; semantic-block reachability/observability lower bounds; leakage bound; discrete CBF runtime filter with closed-form projection on affine barrier; LQR steering; corrigibility-as-reachability.
- SLT: H-SLT-1 — LLC of diagonal LTI SSMs tracks RRR-like λ(H=n_state, r=Hankel rank), jumps when an HSV crosses noise floor; patterning-MPC with Kalman stage observer over LLC/susceptibilities.
- JEPA on SSM with EMA target, SIGReg; energy as OOD signal; HSVR may cause collapse.
- Phases 0–6 (≈10M→300M params), gates (BANKING77 ≥85, CLINC+OOS ≥75, ECE ≤0.03, OOS AUROC ≥0.9, ...) — self-invented targets. 10 claimed novel contributions. First experiment: 25M LRU LM, 8 layers d=384, pretrain ~2B tokens in 2–4 GPU-days, then decision fine-tune on BANKING77/CLINC.

---------------------------------------------------------------------
## DOCUMENT B — v2 self-critique + revised plan (also by the user's prior assistant)
---------------------------------------------------------------------

What is wrong or unaccounted for:
- Benchmark comparison unfair: Decision Index is zero-shot across 40 benchmarks; v1 fine-tuned on BANKING77/CLINC150 so gates measured a different task.
- Scale: Clef on 27B backbone; Index includes knowledge-heavy tasks; 25M–300M from scratch won't beat it overall.
- Distillation ceiling: student trained mainly on Clef rarely exceeds Clef.
- "Closed-form Gramians" oversold: per linear layer, don't compose through nonlinear mixing; whole stack not certified.
- Memory vs safety tension: long contexts need slow modes; slow decay makes injection half-life bound useless.
- Forked option scoring lossy: state compressed before options read.
- Grounded ≠ correct: verified emission can still select wrong true fact.
- Thin evidence: 2026 preprints unverified; vendor numbers; invented gates.
- Too many novel parts (ten contributions, one engineer).

Equivalences:
- Decision head is a Bayes filter: log p(k|x1..t) = log p(k|x1..t−1) + log p(x_t|k,past) − normalizer. Linear recurrence with eigenvalue 1 and input-dependent increment = selective SSM with one integrator per option. Softmax of state = exact posterior if increments are true log-likelihood ratios; log loss/Brier push them there → "calibration is a property of the architecture". Consequence: question+options must come before state; input-dependent gating mandatory.
- Memory bound: token at distance τ moves a logit by at most Σ|c_i b_i||λ_i|^τ; n-dim linear state holds at most n scalars; 64K context can't be stored, only accumulated toward known questions.
- Hankel rank & LLC: white-noise input → population loss = squared distance of impulse responses g_k = Σ c_i b_i λ_i^k. SISO with n modes, true order r: each used mode costs 1, unused mode costs 1/2 (c_i b_i = 0 singularity) → λ ≤ r + (n−r)/2; jumps of ~1/2 per mode activation. Upper-bound sketch.
- Truncation threshold: free energy nL + λ log n; dropping mode with Hankel value σ costs ~4σ²n loss, saves ½ log n → truncate when σ < √(log n / 8n). Heuristic.

Revised plan:
- "Outperform all" on overall Index needs comparable pretraining scale; unlikely below a few B params; beating Clef-flash on chosen axes plausible.
- Weeks 1–6: integrator-bank head with question-first ordering on largest open pretrained attention-free backbone (Mamba, RWKV, xLSTM multi-B releases — verify sizes). Freeze backbone, adapters, Clef's recipe, real zero-shot Index.
- Win structural axes: Clef-flash 38.8 ms median latency, 66.8 CLINC+OOS, calibration error, constant-memory streaming.
- Close accuracy gap: GT labels + multiple teachers (Clef + frontier models via System One adapter), scale backbone.
- Glass-box conversion: replace layers one at a time with certified variants, record cost → Pareto map.
- Budget $5k ≈ 2,000 H100-hours at $2–3/hr. Baselines $400, data $1,200, head/adapter training $2,000 (6–8 runs on 1–3B attention-free backbone), control/interp $700, reserve $700.
- Weeks 1–2 (~$800): same integrator head on frozen attention-free backbone vs same-size transformer; if attention-free trails by >~10 pts on zero-shot subset, stop.
- Weeks 3–5 multi-teacher adapters with calibration target; weeks 6–8 replace top layers with certified diagonal layers; submit to Index.
- Largest risk: nobody has shown pretrained attention-free model supports zero-shot schema decisions well.
- Open question posed: lead with beating Clef, or guarantees first? (User answer: "no tradeoff... push all frontiers"; spend up to $5k to make it controllable+interpretable while still outperforming.)

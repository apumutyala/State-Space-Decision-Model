# W2 — Capability levers for small decision models, and why SSDM's data did not transfer

Author: W2Capability (sonnet-research) · Date: 2026-10-14 · Governing context: `local://ssdm-context.md`, `local://direction/brief.md`.
Claim labels: **MEAS** = SSDM repo receipt I read; **EXT** = external source I opened (URL given); **SIM** = my CPU arithmetic on published data (scratch: `/tmp/ssdm-scratch/W2Capability/`); **INF** = my extrapolation; **[UNVERIFIED]** = not confirmed in an opened source. Evidence grades for levers: **A** replicated / peer-reviewed; **B** several independent observational reports or one clean controlled study; **C** one source or confounded; **D** hypothesis only.

## Conclusions

1. **Intelligence (I) is the only axis worth buying; Calibration (C) is nearly free.** At H2O's operating point (I 60, C 90, S 92.6, K 60.3) +1 I = +0.37 Composite but +1 C = +0.16 (SIM, formula from the board JSON). C already sits at 88–94 for every serious row, and one post-hoc temperature gets a 4B label-token model there (JevK5 v0.3 89.9, decider-4b v2 89.6, Kev-27B 93.7 with one T). LS-CE+Brier/RLCD-style machinery is worth at most +0.6 Composite.
2. **Public-231 accuracy cannot rank the leaders, so local ΔComposite is a weak guide.** H2O (205/231 = 88.7%, board I 60.0) and JevBench-tuned JevK5 v0.3 (≈203/231 = 87.9%, board I 38.5) are within 2 items on public but 21.5 I apart on the board (EXT, SIM). The board is 50% sealed plus a public–sealed gap penalty. The next cell needs an independent lockbox, not only public-231.
3. **What separates winners from losers is data provenance, not volume.** Successes (H2O, JevK5 v0.3, Kev-27B, Perplexity, Nimble) used LLM/teacher-authored or heavily verified diverse scenario data, 2.7k–146k rows. Failures (reflex's four LoRA mixes and distillation, LLM-as-Jev's 95k public-label mixes, SSDM jev1/jev2/jev3) used programmatic generators, converted public datasets, or small System-1 distillation sets. n = 2 successes at 4B vs ≥ 6 failures, confounded; grade C+.
4. **Single most likely transfer-failure cause (INF, grade C+):** SSDM's rows are template-generated with rule-engine golds, so they teach generator-specific cues (slice +4.4/+9.0 pp, public flat, transfer corr +0.15/−0.32) while the remaining public headroom (23 of H2O's 26 errors) is in LLM-authored adversarial multi-step items. **Confirming test:** a same-host, same-size A/B of generator rows vs teacher-authored thinking-verified rows (≈ $5–8), plus a $0 adversarial-validation check first (§4).
5. **`temporal_numeric` is a capacity wall for every single-pass system ≤ 27B** (public accuracy 0.33–0.47 for stock 4B, 27B no-think, thinking-768 4B, JevK5, and H2O itself at 5/15). It is 10 of H2O's 26 errors. Only long thinking (≥ 2–4k tokens) solves it (EXT). The fix is reasoning internalisation from thinking-teacher traces, not more generator rows.
6. **Readout hygiene is the largest cheap swing, and it is an honesty problem.** JevBench counts a yes/no answer with P(yes) in (0.2, 0.8) as wrong. H2O ships a 0.801 floor; SSDM measured the floor + T worth +23.7 I on H2O's raw readout (MEAS). Any honestly calibrated trainee lands in the band on uncertain items (JEV3-SYM 48/74). Adopt only as a disclosed benchmark configuration with a raw "routing mode", as H2O does. Cross-reference W3.
7. **Base capability dominates I; the cost axis then caps which bases can win.** Recipe-constant scaling on the board: Quyet 4B 41.7 → 31B 73.4; Kev 4B 19.5 → 27B 64.2. K uses the *base-model reference price*: Qwen3.5-9B (K 46) caps Composite at ≈ 59 even at I = 75; Qwen3.8-27B (K ≈ 30) caps it at ≈ 21 and is Capability-ineligible (cost cap $0.0646/1k). No hybrid-SSM base above 4B can beat 72.5 Composite (SIM, §5.2).
8. **Post-training an open leader:** (a) thin LoRA + T on Qwen3.5-9B, expected I 55–62 and Composite ≤ 59; on 27B hybrids I 70–76 but cost-gated; (b) continuing Quyet: +1 I [−3, +5], $40–100/cell, contradicts the SSM mission; (c) continuing H2O: 0 [−2, +2], $3–12/cell, flat twice already; quantization: +0.1 Composite, no K change. Details §5.3.
9. **Ranked by I-gain per dollar:** readout hygiene ($0) > contrastive minimal-pair set (≈ 0.25 I/$) > teacher-authored thinking-verified scenario recipe (≈ 0.08 I/$, the only path to H2O parity) > rationale/implicit-CoT distillation for numeric items > everything else ≈ 0. Table in §5.
10. **Budget reality:** $11.67 left. The §7 ladder ($0 diagnostics, a ≈ $2–3 teacher-relabel audit, one ≈ $5–8 A/B, a small lockbox) totals ≈ $9–15, so run the audit and the A/B first. The recipe run (≈ $120–250 all-in for ~100k rows) needs a funding decision. Open-weight teachers are sufficient: Kev-27B was built with GLM-5.3, DeepSeek-V4-Pro, Mistral Large 3, gpt-oss-120b, MiMo-V2.6-Pro and reached board I 64.2 (EXT).

---

## 1. What the board actually rewards (formulas, numbers)

| Fact | Value | Source |
|---|---|---|
| Intelligence = 0.5·I_open + 0.5·I_sealed with a public/sealed gap penalty; tier weights easy .10 / standard .20 / judge .30 / hard .40; types choice .50 / noul .25 / score .25 (v1.5 text; v1.6.1 weights **[UNVERIFIED]**, `noul_method_v16.py` is not downloadable) | — | EXT [METHOD-v1.5](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5.md), [methodology page](https://benchmarkheaven.com/jev-models/methodology) |
| Noul: P(yes) ≤ 0.20 → No, ≥ 0.80 → Yes, between → abstention, counted wrong | CC = 100·(acc−0.5)/0.5 | EXT METHOD-v1.5 §3.1 |
| Score: scored on expected position (nMAE), not argmax | — | EXT METHOD-v1.5 §3.1 |
| Authors of public hard items: Claude Opus 5 and GPT-5.6 Sol, cross-reviewed, "reasoning over the state", target: a Flash-Lite-class model misses 20–40% | 111 hard public | EXT [HARD-TIER.md](https://github.com/fstandhartinger/jevbench/blob/main/datasets/HARD-TIER.md) |
| Public-231: 48 easy (≈ 12 tokens), 72 standard (14–40 tokens), 111 hard; 139 choice / 74 noul / 18 score; 36 of 111 hard states > 2,000 tokens | — | SIM from the three public files |
| Public noul gold: 35/74 yes (47%); hard-tier choice gold is the *first* label in 9/67 items (13%) vs ≈ 25% expected | position prior in the public items | SIM |
| Board v1.6.1 axes: H2O I 60.0 / C 90.0 / S 92.6 / K 60.3 → Composite 72.5 (reproduces to 72.52) | — | EXT [board JSON](https://benchmarkheaven.com/api/decision-results/jevbench/v1.6.1/json), SIM |
| Composite slope at H2O's point: ∂/∂I = 0.37, ∂/∂C = 0.16. To beat 72.5 at C = 90, S = 92.6, K = 60.3 you need I ≥ 60; at C = 85 you need I ≥ 63 | — | SIM |
| Capability = (I+C)/2; Quyet-1.0-Large 81.7 needs I = 73.4 at C = 90 (+13.4 I over H2O). Composite at I = 73.4 only reaches 76.8 | — | SIM |
| Local public-231 vs board: H2O local I 77.6 vs board 60.0 (−17.6); stock Qwen3.5-4B + H2O shim local I 56.3 / C 69.3 vs H2O 77.6 / 83.4 (**the "+21 I"**) | — | MEAS (ledger `job-zbtbj`, EVIDENCE_REPORT_V2 §0, §6) |
| Cost K = 100 − 30·log10(USD per 1k decisions / 0.001), USD from the **base-model reference price × measured tokens**; quantization is not priced | see §5.2 | EXT `price_basis` fields in the board JSON |
| Speed S = 80 − 10·log10(p50_adj) − 10·log10(p95_adj), p_adj = 2·raw + 0.15 s | raw 0.03 s → 91.8; 0.5 s → 77; 2 s → 66; 5 s → 58 | SIM |

**Consequence for "thinking at inference".** A 5 s raw p50 only costs S ≈ 58, but a thinking model pays input+output tokens (OpenJev-thinking: $0.24/1k, K = 28.5 → (28.5/50)² = 0.32 penalty) and breaches both Capability caps. Single-pass is the right frame; thinking belongs on the *teacher* side.

**Public accuracy does not rank the leaders (SIM from EXT cards and board JSON):**

| System (4B, Qwen3.5-4B base) | Public-231 accuracy | Board I / C |
|---|---|---|
| H2O-Lightning-4B v1.1 | 205/231 = 88.7% ([card](https://huggingface.co/h2oai/h2o-lightning-4b)) | 60.0 / 90.0 |
| JevK5 v0.3 | 48+68+87 = 203/231 = 87.9% ([card](https://huggingface.co/alibiserikbay/JevK5)) | 38.5 / 89.9 |
| JevK5 v0.2 | ≈ 199/231 = 86.1% (card) | 36.3 / 89.4 |
| SemIf (frozen) | 80.5% ([LLM-as-Jev Table 1](https://arxiv.org/abs/2610.02076)) | 27.1 / 83.6 |

Two models 2 items apart on public are 21.5 I apart on the board. Candidate explanations (INF): sealed-set generalisation, the noul band, and the gap penalty. JevK5's own card reports its v0.2 answered 33.1% of 308 sealed decisions correctly against a chance level of ≈ 29% (EXT, JevK5 card; chance from METHOD-v1.5 §0).

---

## 2. Recipe evidence: what the leaders disclose

| System | Base / adaptation | Board I / C (v1.6.1) | Disclosed training | Not disclosed | Source |
|---|---|---|---|---|---|
| **H2O-Lightning-4B v1.1** | Qwen3.5-4B, LoRA r64 / α128 (adapter shipped, merged weights default); per-type temperatures choice 0.75 / yes-no 0.8 / score 0.65; **yes/no floor 0.801** | 60.0 / 90.0 | 3 stages (broad → focused on long documents, numbers, dates, probabilities → final mix). Mix: documents/policies/rules 37%, numbers/dates/probability 22%, games & simulations (state tracking, planning) 12%, everyday language/knowledge 10%, support/ops 10%, safety 7%, coding/tool 2%. Types: choice 54 / yes-no 29 / score 16. "Labels from human annotation or agreement among several independent strong models." Dedup vs every eval set; held-out lockbox; "gap hunting"; no JevBench data; all-text English | Row count, epochs, which teachers, LR | [H2O card](https://huggingface.co/h2oai/h2o-lightning-4b) |
| **Clef-flash** | Qwen3.5-9B frozen + LoRA r256 + ~128M joint routing head; LS-CE + Brier; RLCD secondary | 42.2 / 85.5 (Decision Index Full 47.61 vs public 56.15) | "internal synthetic datasets permutating field orders, prompts and schema structures"; RLCD = partial credit for adjacent ordinal choices, reward for exact record outputs, reference penalty | Data, rows, steps, RLCD algorithm | [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/) |
| **MD1** | Qwen3.5-9B post-trained, 32k context | not in the v1.6.1 JSON; H2O's card quotes API composite 69.1 (#6) **[UNVERIFIED]** | "publicly available datasets subject to Microsoft's Open Data process, plus synthetic data"; supports explicit abstention options | Teachers, rows, losses; its 83.5% / 36-benchmark figures are Microsoft's own | [MD1 post](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), [Foundry card](https://ai.azure.com/catalog/models/Microsoft-Decision-1) |
| **Nimble-9B-v3** | Qwen3.5-9B LoRA (r64/α128 per SSDM note; primary card silent) | 56.5 / 77.0 | v1 recipe public: 2,676 contrastive records (1,338 base/counterfactual pairs; change one fact so the label flips), labels "synthetic and model-checked, not human-reviewed"; CE over candidate logits; r16, lr 5e-5, 1 epoch = 335 updates, bs 8, max_len 2,048; no teacher probabilities or rationales. v4 (9B/27B) reports Decision Index public 63.70 / 68.46 (own run, not on board) | v3 and v4 data | [Nimble repo](https://github.com/bespokelabsai/nimble), [training doc](https://github.com/bespokelabsai/nimble/blob/main/docs/NIMBLE_TRAINING.md), [dataset doc](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md), [console](https://console.bespokelabs.ai/) |
| **Perplexity Decider v1.1** | Qwen3.8-27B full FT, **non-causal attention** in full-attention layers, separate readout head | 75.7 / 89.9 | v1: human-labelled public datasets converted with Nimble converters; v1.1: Index 56.4 → 61.56, "most of the gains … lifting the causal mask and training on more data, including tasksource" | Mix sizes, ablations | [card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| **Kev-27B v2** | Qwen3.8-27B full FT, one epoch | 64.2 / 93.7 | 145,840 records (337,130 questions); 8×H200, 16.2 h = 129 GPU-h; lr 2e-6 backbone / 1e-4 head; CE over options, soft targets where available; shuffled options with "none of the above" and distractors inserted; minimal pairs; 0.85/0.15 weight average with v1; one T = 1.32. **Generated labels come from code or open-weight models: GLM-5.3, DeepSeek-V4-Pro, Inkling, Mistral Large 3, gpt-oss-120b, MiMo-V2.6-Pro** | Per-component effect | [card](https://huggingface.co/jaredpalmer/kev-27b) |
| **Quyet-1.0-Large** | Gemma-4-31B-it + merged rank-16 LoRA, letter readout, refit temperatures, 6k-token states, ≤ 10 options | 73.4 / 90.0 | none beyond recipe shape | Data | [card](https://huggingface.co/chinhnc/Quyet-1.0-Large) |
| **JevK5 v0.3** | Qwen3.5-4B, LoRA r16 attention, SemIf letter readout, one T | 38.5 / 89.9 | 17,408 questions written and answered twice by thinking models, kept only if both answers match intent (3,270 from Qwen3.6-27B, 14,138 from GPT-6 Luna, a **closed** model, not usable for SSDM); + 30,052 train-split public replay rows (26 datasets); 1 epoch, lr 3e-5 | — | [card](https://huggingface.co/alibiserikbay/JevK5) |
| **reflex** (frozen Qwen3.5-4B + readout engineering) | no training | official v1.2 #5 (71.7, older board) | Four LoRA mixes and a 27B distillation all lost general judgement | — | [repo](https://github.com/kshetrajna12/reflex) |

**Within-recipe base scaling (EXT, board JSON):** Quyet Medium 4B 41.7 → Large 31B 73.4; Kev 4B 19.5 → 8B 29.3 → 27B 64.2; Nimble v1 9B 43.1 → v3 56.5. A frozen Gemma-4-12B-it + letters (decider-12b v1) already reaches I 60.6 with no training. H2O's 4B 60.0 is therefore an *outlier above the recipe-constant ladder*: its data is the unusual ingredient. (Third-party rows are shown as evidence only, per the 2026-10-11 operator directive.)

**What H2O's mix suggests (INF):** its 12% "games & simulations (state tracking…)" and 7% safety/guardrails line up with families JevBench documents for the newer pool (METHOD-v1.5: "State tracking gets ≥ 12% of every type", tool-call guardrail noul, lead-qualification score). The sealed 1,200 items follow the *documented* pool, not only the 231 public items. A recipe that mirrors only the 18 public families (JevK5 did) is exposed.

---

## 3. Literature levers and their evidence

| Lever | What the evidence says | Grade for ≤ 9B decision models | Links |
|---|---|---|---|
| **Label-token (verbalizer) readout vs learned heads** | Joint suffix/letter scoring beats isolated scoring by ≈ 5 pts on identical backbones; frozen Qwen3.5-4B with a bracketed-number joint readout gets 81.4% public (188/231) with no training. Same-backbone comparison: Nimble (letter codes, r16–64 LoRA) Full 54.67 vs Clef-flash (head + r256) 47.61 (SSDM note `techniques/10-clef-recipe-and-gap-attribution.md`, secondary). Board 4B rows: label-token LoRA (H2O 60.0, Quyet-M 41.7, lev 41.1) vs head variants (Aplomb 45.8, janus 37.6). SSDM's own $0 pool of stock and head readouts: +1.27 in-sample | B (observational, no controlled head-vs-letter ablation) | [LLM-as-Jev](https://arxiv.org/abs/2610.02076), board JSON |
| **Prefill-only vs generate-then-answer** | Frozen Qwen3.5-4B with a hard thinking budget: 81.4% → ≈ 85–92% at 512–2,048 tokens → **94.8% at 4,096** (GPT-5.6 Sol 94.4%); judge_hard 0.65 → 0.94, but long_policy/multi_hop/temporal_numeric do not move at 768 tokens. Inference-time thinking is excluded by K and the Capability caps (§1) | A for the effect, **excluded as a serving mode** | LLM-as-Jev §4.5; [reflex teachers doc](https://github.com/kshetrajna12/reflex/blob/main/docs/results/teachers-27b-and-4b-think.md) |
| **Reasoning-then-answer distilled into prefill** | Rationale supervision as an auxiliary task lets small models beat much larger few-shot LLMs with less data ([Distilling step-by-step](https://arxiv.org/abs/2305.02301), abstract); stepwise CoT removal internalises arithmetic (GPT-2 Small 9×9 multiplication 99% vs failing beyond 4×4; Mistral-7B > 50% GSM8K with no steps) ([Implicit CoT](https://arxiv.org/abs/2405.14838)). No published test on decision models | C (analogy; nothing on JevBench-like items) | abstracts |
| **Pause tokens / latent reasoning** | Gains need *pre*training with pauses; fine-tune-only gives little ([Goyal et al.](https://arxiv.org/abs/2310.02226)); Coconut needs a curriculum and is shown on logical search ([Hao et al.](https://arxiv.org/abs/2412.06769)). Pause tokens add input tokens, so K must count them | D | abstracts |
| **Judge / reward-model training** | Prometheus 2 (custom rubrics, direct + pairwise), GenRM (verifier as next-token prediction with synthetic rationales, CoT + voting), Skywork-Reward-V2 (human-AI curated pairs; quality > scale), J1 (RL to think before judging), JudgeLM (swap augmentation). Gains come from rationales or curated data; SSDM's HelpSteer3/Aegis/PRM800K rubric rows gave transfer corr −0.50 | C for judge_hard/adequacy; SSDM attempt negative (MEAS) | [Prometheus 2](https://arxiv.org/abs/2405.01535), [GenRM](https://arxiv.org/abs/2408.15240), [Skywork-V2](https://arxiv.org/abs/2507.01352), [J1](https://arxiv.org/abs/2505.10320), [JudgeLM](https://arxiv.org/abs/2310.17631) |
| **Policy-following classifiers** | gpt-oss-safeguard reasons over a developer-supplied policy at inference; OpenAI notes classifiers trained on tens of thousands of high-quality labels can still beat it and it is compute-heavy. Qwen3Guard (0.6/4/8B, Apache-2.0) and Granite Guardian show a ≤ 8B model can carry policy-conditioned judgments when trained on policy-conditioned data | C; reasoning models are teachers, not serving models | [gpt-oss-safeguard](https://openai.com/index/introducing-gpt-oss-safeguard), [Qwen3Guard](https://arxiv.org/abs/2510.14276), [Granite Guardian](https://arxiv.org/abs/2412.07724) |
| **RL for calibrated confidence** | RLCR adds a Brier term to a correctness reward and improves calibration without accuracy loss (ordinary RLVR hurts it), but it trains *generated* reasoning plus a verbalised confidence ([Damani et al.](https://arxiv.org/abs/2507.16806)). RLCD (TypeSafe/Cloudflare): named, no algorithm or ablation published ([primer](https://docs.typesafe.ai/introduction/machine-learning-primer), Cloudflare blog). For a prefill model the expected reward of a distribution is differentiable, so RL adds nothing over proper-scoring losses except for non-decomposable joint-record rewards (INF) | D for ≤ 9B prefill models; B (preprint, theory + experiments) for RLCR as a *teacher* recipe | links in row |
| **Proper scoring losses / calibration** | Label smoothing improves calibration but **hurts distillation from a smoothed teacher** ([Müller et al.](https://arxiv.org/abs/1906.02629)): SSDM trains with LS 0.05 and distills from smoothed heads. Temperature scaling suffices for many classifiers ([Guo et al.](https://arxiv.org/abs/1706.04599)). Board: single-T label-token models reach C 88–94 | A (general), B (board) | links in row |
| **Distillation** | Hinton soft targets; [Beyer et al.](https://arxiv.org/abs/2106.05237) ("patient and consistent"); [GKD on-policy](https://arxiv.org/abs/2306.13649). Decision-model evidence is negative when the teacher is another single-pass model: SSDM D2 (Clef-flash KD, no gain) and reflex's 27B-teacher student (hard 0.658 → 0.613; temporal 0.40 → 0.27, multi_hop 0.67 → 0.56) | B (negative for System-1 teachers) | [reflex distill](https://github.com/kshetrajna12/reflex/blob/main/docs/results/lora-distill-qwen3.5-4b.md) |
| **Multi-task instruction tuning** | FLAN ([Wei](https://arxiv.org/abs/2109.01652)), T0 ([Sanh](https://arxiv.org/abs/2110.08207)), Super-NaturalInstructions ([Wang](https://arxiv.org/abs/2204.07705)), Flan Collection ([Longpre](https://arxiv.org/abs/2301.13688)), tasksource ([Sileo](https://arxiv.org/abs/2301.05948)): strong for models without instruction tuning. For Qwen3.5-4B and JevBench hard items it is neutral-to-negative: LLM-as-Jev 4B Reason-30k 77.1% vs 81.4% frozen, Intent-1ep 74.9%, reflex mixes hard 0.640 → 0.541–0.604. It *does* help Index-style public benchmarks (Perplexity credits tasksource) | B: use ≤ 20–30% as replay | [LLM-as-Jev Table 7](https://arxiv.org/abs/2610.02076), reflex [frozen-vs-trained](https://github.com/kshetrajna12/reflex/blob/main/docs/results/frozen-vs-trained.md) |
| **Anchored fine-tuning** | LoRA + KL anchors to the frozen model (λ = 1): 84.0% public vs 81.4% frozen and 80.1% for full FT; unanchored runs emit trailing text on 69/231 queries; LoRA "learns less, forgets less" ([Biderman](https://arxiv.org/abs/2405.09673)) | B | LLM-as-Jev §4.4 |
| **Synthetic data and label conventions** | Self-Instruct ([Wang](https://arxiv.org/abs/2212.10560)), [best practices](https://arxiv.org/abs/2404.07503): factuality/fidelity/diversity gate quality. Decision-model specific: JevK5 (double-answer agreement), Nimble (contrastive flips), Kev (open-weight generation + code labels), H2O (multi-model agreement) | B | cards in §2 |
| **Test-time augmentation / ensembling** | Two distinct option orders (+ yes/no both ways): frozen 4B hard 0.658 → 0.685, calibration axis 76.8 → 78.9; 27B hard 0.703 → 0.766. Wording ensembles hurt; four orders no better; PriDe-type debiasing exists ([Zheng](https://arxiv.org/abs/2309.03882)). SSDM's exact reorder invariance (sym-fork) is this lever by construction. Contextual calibration and re-reading were negative on stock (MEAS DEAD_ENDS R3) | B | [reflex order-averaging](https://github.com/kshetrajna12/reflex/blob/main/docs/results/order-averaging.md) |
| **Latent multi-hop / numeric limits** | Latent two-hop reasoning is partial and scales unevenly ([Yang](https://arxiv.org/abs/2402.16837), [Balesni](https://arxiv.org/abs/2411.16353)); transformers answering immediately cannot solve some simple composition problems and need steps ([Merrill & Sabharwal](https://arxiv.org/abs/2310.07923)). Empirically: temporal_numeric 0.33–0.47 for all single-pass 4B/27B systems | A (theory), B (decision-model data) | links in row |

---

## 4. Diagnosing the transfer failure

### 4.1 Facts that any explanation must fit (MEAS unless noted)

- H2O + 2,664 jev2 rows (x7c42): slice +4.44 pp [+3.47, +5.31], public ΔComp A −0.41 [−1.75, +1.39]. With 17,959 rows (q668q): slice +8.97 pp, public −0.97 [−3.38, +1.12]. Transfer correlation by family +0.15 → −0.32. Neither claimable under Holm.
- Slice gains landed in families already saturated on public (H2O at 83–100%). **23 of 26 public errors** sit in temporal_numeric (10), judge_hard (5), long_policy (5), multi_hop (3), where the slice showed no significant gain (+3.0 n.s., 0.0, +1.5 n.s., +3.5; temporal +12.5 on the slice in q668q, no public movement).
- Our own stacks: the readout drives the negatives. JEV3-SYM puts 48/74 yes/no answers in the (0.2, 0.8) band (stock/H2O 0/74); ΔComp A −36.66 native vs −1.07 with shimcal. JEV1 hedged 10/74. Ranking losses are real elsewhere (score top-1 −33 pp, routing −58 pp).
- jev1 draw: 3-template monoculture, 11 instruction strings, 66.2% noul-false prior, 14,540/30,046 noul rows with empty criteria (public: all 74 have criteria; yes-rate 47%), 0/96,000 prose states, long_policy p50 934 tokens (public 2,657; my SIM p50 3,016). Jev2 shape gate: length_p90 below public for jev2b, judge_hard, multi_hop, temporal_numeric.
- Public hard items were authored to have a "tempting wrong label", 2+ interacting conditions and relevant-looking near-miss filler; template-swapped nouns were banned for the authors (HARD-TIER, EXT). Rule-engine golds are exact but template-bound.
- Outside evidence with the same sign: reflex mixes 1–4 and a 3.4k-state 27B distillation lost hard-tier accuracy; LLM-as-Jev trained on 9k–95k human-labelled public rows and lost 4–6 pp. The two positive 4B results (JevK5 v0.3 hard 0.613 → 0.784 with 17k verified teacher-authored questions; H2O) used LLM-authored scenario data.

### 4.2 Hypotheses with $0–5 diagnostics

| # | Hypothesis | Support | Plausibility (INF) | Cheapest diagnostic and decision rule |
|---|---|---|---|---|
| **H1** | **Provenance/diversity: template-generated, rule-engine-gold rows teach generator cues, not the board's item-authoring conventions** | In-slice gains vs flat public; monoculture audit; outside failures and successes split by provenance | **High (≈ 45%)** | **T0a** ($0 CPU): adversarial validation, a gradient-boosted/logistic classifier on surface features (length, K, label-position, instruction n-grams, template id, JSON vs prose, criteria present) separating SSDM TRAIN rows from public-231; AUC > 0.9 confirms trivial separability. **T0b** ($0): no-state baseline, predicting gold from form alone (state blanked); accuracy ≫ chance on SSDM rows but ≈ chance on public = shortcut leakage. **T1** (≈ $5–8): same host, size, families and label priors; 2k generator rows vs 2k teacher-authored thinking-verified rows, 1 epoch each. If H1 is true the teacher arm beats the generator arm on public hard errors (McNemar) and the generator arm wins the slice. |
| H2 | Shape/length/format mismatch | C1 audit (above); jev2 shape gate failures | Medium (≈ 15%, overlaps H1) | **T0a** with only shape features; then re-weight/cap shapes to public statistics in T1 and compare per length bucket. Length alone was ruled out by multi_hop (equally long, uncovered, +11.8 pp). |
| H3 | Items need reasoning beyond a single pass | temporal_numeric 0.33–0.47 for every system, 4B thinking at 4k tokens = 94.8%, implicit-CoT literature | **High for temporal_numeric (≈ 20% of the total gap)** | **T2** ($1–3): run a hosted thinking teacher on the 15 public temporal_numeric items and the 10 H2O errors; if ≥ 80% correct, items are solvable with reasoning, so the target is internalisation. Do not train on the items; this is a diagnostic only. |
| H4 | Teacher/gold quality and conventions (rule-engine vs "careful expert") | BOARD proposal (a); jev3 human-rubric attempt (no transfer) | Low–medium (≈ 10%) | **T3** ($0.3–1.5): relabel 200 stratified SSDM rows and the public 111 hard items with 2 open thinking teachers; compare agreement-with-gold by family. SSDM rows agreeing with teachers ≪ public (e.g. < 85% vs > 90%) = convention/noise problem. Free variant: agreement of cached Clef-flash KD labels with rule-engine golds by family. |
| H5 | Headroom: only 26 public errors, 23 in families the data does not teach | MEAS | High (it caps what *any* data can show on public) | **T4** ($0): per-family ceiling: even fixing all 26 errors is ≈ +22 local I. An exact McNemar test needs ≥ 8 net fixed items (9 fixed / 1 broken, SIM) = 31% of H2O's remaining errors. Pre-register that bar. |
| H6 | Readout/calibration artefacts (noul hedging) | MEAS: −36.66 → −1.07 | High for *own-stack negatives*, not for H2O-host flatness | **T5** ($0): re-score saved probabilities through a fixed decisive mapping chosen on a non-public slice (already done: shimcal; needs registration before use). |
| H7 | Under-training | 0.093 epoch on JEV1 | Low (disfavoured by q668q: 7× data doubled slice gain, public flat) | **T6** (≈ $3): rows ladder 1k/4k/16k of teacher-authored rows, same recipe. Only informative after T1 shows a signal. |
| H8 | Wrong selection signal (checkpoint chosen by NLL on a held-out slice of the *same generators*) | PREREG: step selected by dev-card NLL | Medium | **T7** ($0–5): build an independent lockbox (≈ 600 items authored by one open model family from the public HARD-TIER spec, cross-reviewed by another, never trained on); re-score existing checkpoints (stock, x7c42, q668q, JEV1). If the lockbox tracks public flat while the slice rises, selection was the problem. |
| H9 | Measurement noise: n = 231, ±8 Composite; GPU nondeterminism | MEAS | Always present | Use paired McNemar with the pre-registered ≥ 8 net items; 2 seeds for any cell claimed. |
| H10 | Public ≠ sealed: even a public win may not move the board | H2O vs JevK5 (§1) | High as a *meta*-risk | Gate on the lockbox, then on a board submission (needs operator approval). |

### 4.3 The single most likely cause and its test

**Cause (INF, grade C+): H1.** The rows are produced by programmatic generators whose golds come from rule engines. The 4B host learns each template's cues (hence slice +4.4/+9.0 pp and rising with data) but those cues do not exist in LLM-authored adversarial items, so public accuracy is flat to negative; and the one place the public items have headroom (judge_hard, long_policy, multi_hop, temporal_numeric) is exactly where the generators show no learnable signal even in-distribution (judge_hard 0.0). The readout (H6) explains why our *own* trained stacks go below stock, not why H2O-host runs are flat.

**Confirming test (≈ $5–8 plus $0 pre-checks).**
1. T0a/T0b on CPU first. If AUC < 0.7 and the form-only baseline is at chance, H1 is weakened and H3/H5 move up.
2. T1: host = stock Qwen3.5-4B + H2O's published shim (best untrained arm, ΔComp −9.4 vs H2O). Arm G: 2,000 existing generator rows. Arm T: 2,000 teacher-authored rows covering the same families, matched label priors, matched lengths and K, each authored by one open thinking model from the public HARD-TIER spec and kept only if a second open model, answering blind, agrees (JevK5's both-answers-match filter; Kev's open-weight pipeline). LoRA r16, 1 epoch, KL anchor to the base. At SSDM's measured throughput of 0.136 GPU-h per 1k rows (MEAS: q668q $9.39 for 14.8k rows seen) each arm is ≈ 0.3 GPU-h; authoring + verifying 2k rows with DeepSeek-V4.1-Flash ≈ $2–4 (§6). Evaluate on public-231 **item level** and on the T7 lockbox.
3. Decision rule (pre-register): H1 is confirmed if Arm T beats Arm G by ≥ 8 net public items **or** Arm T is ahead on judge_hard + long_policy + multi_hop while Arm G is ahead on the generator slice. If both arms are flat on public and T improves only the lockbox, treat the public-231 as saturated and move to the lockbox as the primary metric.

---

## 5. Ranked lever list

### 5.1 Method

Gain is board-scale Intelligence relative to **B0** = stock Qwen3.5-4B + H2O-style shim (local I 56.3, board I ≈ 39 if the −17.6 local→board offset holds, INF). H2O parity is +21 over B0 locally (MEAS). Cost is all-in dollars: labelling (§6) + training + eval. Training cost basis: SSDM measured **0.136 H100-h per 1k rows** on the 4B (q668q, LoRA r16, max_len 4,096, $0.64 all-in per 1k rows seen); Kev-27B's 129 H200-h per 145,840 records implies ≈ 0.88 GPU-h per 1k rows at 27B, consistent with parameter scaling (EXT). Ranges are 80% intervals; all ΔI are INF unless an EXT/MEAS number is stated. Ranking key = central ΔI per dollar; **viability** says whether the lever can reach the stated targets.

### 5.2 Why the base matters more than the recipe: cost cliffs (SIM from board `price_basis`)

| Base (reference price per M input / output) | Board USD/1k, K | Max Composite at I = 75 (C 90, S 88) | Capability-ranked? |
|---|---|---|---|
| Qwen3.5-4B (DeepInfra reference) | 0.021, 60.3 | 76.3 | yes |
| Gemma-4-31B-it ($0.09 / $0.34, OpenRouter list) | 0.0445, 50.5 | 71.9 | yes (Quyet is open #1) |
| Qwen3.5-9B ($0.10 / $0.15) | 0.0626, 46.1 | **59.1** | yes, 3% under the $0.0646 cap |
| Qwen3.5-27B (reference not on the board; OpenRouter $0.26 in) | ≈ 0.17, ≈ 33 [INF] | ≈ 32 | no (> cap) |
| Qwen3.8-27B ($0.42 / $3.0 "frozen BASE_REFERENCES") | 0.21–0.53, 18–30 | ≈ 21 | **no** (Perplexity, torchcast, Kev, Clef all `capability_eligible = False`) |

To beat 72.5 Composite you need K ≥ 52 at I = 75 and K ≥ 57 at I = 65. At 9B the ceiling is 63.8 even at I = 100. **No hybrid-SSM base above 4B can beat H2O's Composite** under the current reference-price rule, and no Qwen3.8 size below 27B exists in the Qwen org listing (HF API, 2026-10-14: Qwen3.8-27B, Qwen3.8-Flash-Next 180B MoE, Qwen3.8-2.4T-A95B). Whether the board will re-price bases is a W1/W4 question.

### 5.3 The "post-train / quantize an open leader" lever (user question)

| Option | Expected ΔI (80%) | Cost | Main risk | SSM fit | Verdict |
|---|---|---|---|---|---|
| **(a1) Thin LoRA r16 + per-type T refit on Qwen3.5-9B** (Quyet recipe, hybrid GDN base, Apache-2.0) | Board I 55–62 vs 4B recipe-constant ≈ 41 (Nimble-v3 56.5, Quyet-M 41.7); +17 [+10, +22] vs B0 | ≈ 0.31 GPU-h/1k rows → 20k rows ≈ 6 H100-h ≈ $25–40 incl. eval | **Composite ceiling ≈ 59** (K = 46); Capability needs I ≥ 73.4 and no 9B has exceeded 56.5 on the board (< 3% chance, INF) | Good (hybrid base, LoRA on in_proj) | Wrong target; useful only as an I-probe |
| **(a2) Same on Qwen3.8-27B or Qwen3.5-27B** | Board I 70–76 (torchcast: merged LoRA + letters on Qwen3.8-27B = 71.8; Perplexity 75.7 full FT) | ≈ 0.6–0.9 GPU-h/1k rows → 10k rows ≈ $60–100 (one 80 GB GPU cannot hold 27B bf16 LoRA at 4k tokens without 8-bit/4-bit base, INF) | Raw Capability 81.6–82.8 but **not eligible** (cost cap) and Composite ≈ 21 | Good | Capability headline unreachable; strong teacher/reference only |
| **(b) Continued training of Quyet-1.0-Large** (Gemma-4-31B-it + merged r16 LoRA, Apache-2.0 + NOTICE) | +1 [−3, +5]. No controlled evidence that third-party continued training beats an author's own; H2O + jev2 was flat twice | 31B LoRA ≈ 0.6–0.9 GPU-h/1k rows → 10k rows ≈ $40–100 per cell plus ≈ $5 eval; two GPUs or a quantized base likely needed | Gap to the 81.7 target would be below the 231-item noise; it sits at K = 50.5, so any token growth drops it under the K = 50 cliff; **not SSM, contradicts the operator's custom-stack directive**; Composite 71.4 < H2O 72.5 anyway; leaders iterate weekly (Nimble v1→v4 in 25 days; H2O already ships v1.2.x), so the target moves | None | Low value; do not fund |
| **(c) Continued training of H2O-Lightning-4B v1.1** (Qwen3.5-4B hybrid GDN, r64 adapter shipped, Apache-2.0) | **0 [−2, +2]** (MEAS: −0.41, −0.97; CIs span 0; slice gains in saturated families) | $2.7–9.4 per cell (MEAS) | Nondeterministic training; H2O already ships v1.2.x (the comparator moves); a derivative of H2O is not SSDM's own model; upside only if data targets the 26 errors, where our generators taught nothing | Hybrid, yes | Do not repeat with generator rows; only as the host for T1-style teacher-authored data |
| **(d) Quantize (FP8/NVFP4) any of the above** | ΔI ≈ 0 [−1, 0]: H2O's FP8 changed 21/687 answers, 19 near-ties, McNemar p 0.63, temperatures not refit | $0–2 | none on K | Neutral | Do for real serving cost, not for score |

Quantization on the board: **K is unchanged** (SIM/EXT): `price_basis` for deck-31B (TorchAO FP8) and Quyet (bf16) both price the *same* base-model reference ($0.09 / $0.34 per M, Gemma-4-31B-it); decisio-31B (FP8 on load) the same. **Speed:** H2O's measured FP8 gain at 1 request in flight is 61 → 53 ms; passing a −13% latency through S = 80 − 10·log10(p50_adj) − 10·log10(p95_adj) with the board's 2×+0.15 s adjustment gives **ΔS = +0.3, ΔComposite ≈ +0.1** (SIM; [H2O card](https://huggingface.co/h2oai/h2o-lightning-4b), serving section).

### 5.4 Ranked levers (I-gain per dollar, with viability)

| Rank | Lever | Mechanism | Grade | ΔI central [80%] vs B0 | GPU-h at 4B | All-in $ | I/$ central [range] | Risks | Open-teacher compatible? | Viability for the Composite target |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Readout hygiene**: decisive-noul mapping chosen on a *non-public* slice, per-type T, two option orders and yes/no both ways, lettered yes/no + Evidence/Criterion framing, joint candidate scoring | Removes abstention-as-wrong, softness and position/"yes" prior | A/B (MEAS: floor+T +23.7 I on H2O raw; shim +10 I over v3w on Qwen; EXT: order-avg +2.7 pp hard, lettered yes/no +5.6 pp on toxic-chat) | 0 vs B0 (already included); **+10–25 vs an un-shimmed own stack** | 0 | $0–2 | ∞ | Honesty cost of a floor (§Conclusion 6); not a capability gain | yes (no teacher) | **Required**, not sufficient |
| 2 | **Contrastive minimal-pair set (Nimble style)**: flip one decisive fact so the label flips, "none-of-the-above" and shuffled options, teacher-verified; 3–5k rows | Forces evidence use over priors/templates; reported public→Full drop −2.5 vs Clef-flash −8.5 | C+ | +2 [0, +6] | 0.4–0.7 | $5–12 | 0.25 [0–1] | Unmeasured on JevBench I; small sets overfit | yes (open thinking teacher authors + verifies; Kev did this with open models) | Part of the recipe |
| 3 | **Anchoring** (KL to base, LoRA, ≤ 1 epoch, shuffle) as a modifier on any cell | Prevents drift and the "intent/short-context traps" | B | +1.5 [0, +3] | +30–50% of the cell | +$1–3 per cell | high but small base | Extra reference forward | yes | Part of the recipe |
| 4 | **Teacher-authored, thinking-verified, family-spec'd scenario data, 20–100k rows, 3 stages** (H2O/JevK5/Kev recipe: documents/policies, numbers/dates/probability, judging, multi-hop, state tracking, guardrails, support; answers agreed by ≥ 2 open thinking teachers or one with two independent answers; option keys rebuilt from text; lengths 300–4,000 tokens; ≤ 20–30% public replay) | Supplies the item-authoring distribution and conventions the board uses | B (JevK5 hard 0.613 → 0.784; H2O) | **+12 [+4, +20]**; the only lever with a credible path to the +21 needed for H2O parity | 5–14 (40–100k rows) | **$90–250** (§6 labelling $10–150 + $25–90 training + $10 evals) | 0.08 [0.02–0.2] | Public→sealed gap; funding; temporal_numeric likely unmoved | yes (Kev's open-teacher precedent) | **Only path to parity at 4B**; needs a funding decision |
| 5 | **Rationale distillation / implicit-CoT curriculum** for numeric and multi-step items: auxiliary rationale target trained jointly, then stepwise rationale removal; teacher = open thinking model | Internalises computation the single pass cannot do from labels alone | C | +2 [0, +6] (10 of 26 H2O errors are temporal_numeric) | +50–100% on the cell | $40–120 | 0.03 [0–0.1] | No published decision-model test; may only move templates (H1 again) | yes | Targets the largest remaining error mass |
| 6 | Judge/rubric specialist data (Prometheus-style rubric + reference answer, GenRM-style verification with rationales, subtle-error hard negatives) | judge_hard/adequacy | C | +1 [−1, +4] | 0.5–1 | $20–60 | 0.02 | SSDM attempt −0.50; JevK5 judge 0.76 → 0.65 | yes | Marginal |
| 7 | Public multi-task replay (tasksource/FLAN-style) | Index breadth, anti-forgetting | B | −1 [−4, +2] on JevBench I; positive on Index | 1–3 | $10–30 | ≈ 0 or negative | Negative on JevBench hard (LLM-as-Jev, reflex) | yes | Cap at ≤ 20–30% |
| 8 | KD from System-1 teacher logits | Matches teacher | B (negative) | 0 [−3, +1] | 1–3 | $30–80 | 0 | Teacher no better than student | yes | Skip |
| 9 | RLCD / RLCR-style RL; pause/latent tokens | Reward = Brier + partial ordinal credit; extra prefill compute | D | 0 [−1, +2] | ≥ 2–10 | $20–100+ | ≈ 0 | No public RLCD ablation; prefill-only models need no RL for decomposable rewards; pause gains need pretraining and extra tokens must be declared and priced | yes | Skip until rank 4 lands |
| — | (a)–(d) post-training or quantizing an open leader | §5.3 | — | see §5.3 | see §5.3 | see §5.3 | (b),(c) ≈ 0.01 or 0 | operator directive | yes | Not viable for the targets |

Notes:
- **I/$ for host changes** (a1/a2) is not comparable with data levers: they move the base, not the recipe. Per dollar, a1 is ≈ 0.4 I/$ and a2 ≈ 0.4 I/$, but §5.2 shows neither reaches a target.
- **Compounding is sub-additive**: Rank 4 includes ranks 2, 3 and 5 as ingredients; summing them would double count (INF).
- **The expected board I of the full recipe** is B0 + 12 ≈ 51 [43–59] (INF); H2O's 60 needs the upper half of that range. State this plainly to the chief: at 4B it is a coin-flip-or-worse at ≈ $150–250, with temporal_numeric as the stated blind spot.

---

## 6. Open-weight teachers, Oct 2026

Licenses are read from the model repos (HF API/LICENSE files, 2026-10-14). I found **no clause in any of these licenses prohibiting use of outputs to train other models**. The Kimi K3, GLM-5.3 and Qwen3.8-Max licenses add "Model as a Service" revenue conditions (> $20M / $10B / $50M) that do not bind SSDM today. Hosted-provider terms: a regex scan of DeepInfra, Together, Fireworks and OpenRouter terms pages found no output-training prohibition (OpenRouter bars "developing a competing service" by reselling API access); a full legal read is **[UNVERIFIED]**.

Prices are OpenRouter listings fetched 2026-10-14 ([models API](https://openrouter.ai/api/v1/models) + per-model endpoints); they vary by provider and day. Strength ordering is BenchLeader's daily index ([page](https://www.benchleader.com/best/open-weights-llm), 2026-10-14, secondary). Labelling cost = 100k rows × 900 input tokens + output tokens; (A) answer-only 10 output tokens, (B) short thinking 1,200, (C) long thinking 3,000. Cells are `cheapest provider / median-price`. Reasoning models may emit more than B/C unless effort is capped (INF).

| Teacher (license) | BenchLeader index | Input / output $/M (provider min–median) | 100k rows A | B | C | Notes |
|---|---|---|---|---|---|---|
| MiMo-V2.6-Pro (MIT) | 65.3 (#1) | 0.43 / 0.87 | $40 | $143 | $300 | 4 providers; Kev used it |
| Kimi K3 (Kimi K3 License: MIT-style + MaaS/attribution conditions) | 64.9 | 0.49–2.55 / 9.5–15 | $59–245 | $1,306–2,030 | $3,016–4,730 | Output price makes thinking labels expensive |
| GLM-5.3 (MIT-style, > $10B MaaS clause) | 63.7 | 0.06–1.19 / 0.63–4.4 | $10–112 | $93–635 | $205–1,427 | Kev used it |
| Qwen3.8-2.4T-A95B (Qwen3.8-Max license) | 63.2 | 2.0 / 6.0 | $186 | $900 | $1,980 | Too pricey for labelling |
| DeepSeek V4 Pro 0813 (MIT) | 62.7 | 0.13–0.97 / 0.44–3.1 | $14–91 | $65–456 | $144–1,008 | Kev used V4-Pro |
| GLM-5.3-Flash (MIT) | 62.2 | 0.04–0.14 / 0.15–0.5 | $4–13 | **$21–73** | **$47–163** | Cheap thinking |
| DeepSeek V4.1 Flash (MIT) | 61.6 | 0.016–0.19 / 0.18–0.89 | $2–18 | **$30–124** | **$62–285** | Cheapest strong thinker |
| Qwen3.8-27B (Apache-2.0, 48 GDN + 16 attn) | n/a | 0.02–0.20 / 0.5–2.3 | $2–20 | $62–297 | $152–716 | Perplexity/Kev backbone; reflex hard 0.703–0.766 non-thinking |
| gpt-oss-120b (Apache-2.0) | n/a | 0.03–0.10 / 0.15–0.55 | $3–10 | $21–75 | $48–174 | Kev used it |
| Gemma-4-31B-it (Apache-2.0) | n/a | 0.09–0.14 / 0.34–0.40 | $8–13 | $49–61 | $110–133 | The only 31B-class dense row that clears K = 50 |
| Others: Mistral Small 4 119B (Apache-2.0), MiniMax-M3 (minimax-community), Qwen3.5-397B-A17B (Apache-2.0), Nemotron-3-Ultra 550B (NVIDIA Open Model License, terms not read), Llama 4 Maverick (Llama license, "Llama" naming rule) | n/a | 0.15–0.6 / 0.6–3.6 | $14–56 | $86–476 | $194–1,114 | No strength or price advantage; Kev used Mistral Large 3 (OpenRouter listing not found, [UNVERIFIED]) |

**Reading the table.** For ~100k decision rows, a thinking-mode label from DeepSeek V4.1 Flash or GLM-5.3-Flash costs **$20–125**, and a **vote of two independent answers** doubles it. An answer-only (no-think) logit label from a 27B+ model costs **$2–20**, but §3 and SSDM's D2 show System-1 labels from a model comparable to the student add little. **Self-hosting is the expensive option for prefill-only labelling:** the repo's own plan (`data/TEACHERS.md`) prices Qwen3.8-27B × 2 permutations over 200k rows at $65–100. Hosted listings are 5–40× cheaper for the same prefill, so rent self-hosted GPUs only for throttle-free thinking at volume or if provider terms matter.

**Which teacher matches the board's label convention (INF):** the items were authored and cross-reviewed by frontier closed models (Opus 5, Sol). A thinking 4B reaches 94.8% on the public items at 4k tokens (EXT), so strong open thinkers should agree with those golds ≥ 90–95%; this agreement is exactly what T3 measures on a 311-row sample for ≈ $1.

**Closed-model teachers are not allowed** for SSDM; note JevK5's 14,138 GPT-6 Luna rows would be unusable in an SSDM clone. Kev's pipeline is the usable precedent.

---

## 7. Sequencing under $11.67 and cross-references

| Step | What | Cost | Unlocks |
|---|---|---|---|
| 0 | Register the T-rules and the ≥ 8-net-items bar; pre-register one decisive-noul mapping on a non-public slice (decide the honesty stance with W3) | $0 | Valid claims |
| 1 | T0a/T0b/T0c on CPU | $0 | Confirms or weakens H1 |
| 2 | T3 teacher relabel audit (311 rows × 2 open thinkers) and T2 numeric solvability probe | ≈ $1.5–3 | Labelling spec, H3, H4 |
| 3 | Build a small independent lockbox (T7; 300 items first, open author + open reviewer) | ≈ $2–4 | Selection signal; fixes H8 |
| 4 | T1 cross-provenance A/B on stock 4B + shim | ≈ $5–8 | Decision on funding rank 4 |
| 5 | If T1 is positive: request ≈ $120–250 for the 3-stage teacher-authored recipe; otherwise stop capability spend and pivot the claim set to guarantees | decision | — |

**Cross-references.** W3: the 0.801 floor and "decisive vs calibrated" conflict. W4: Composite cannot be won above 4B under the reference-price rule (§5.2), and non-causal prefix attention (Perplexity) is an architecture lever for hybrids; the SSM analogue is a repeat-the-prompt (JRT-style) pass; no controlled ablation exists. W1: whether the board re-prices bases; sealed families (state tracking, guardrail, lead qualification) in METHOD-v1.5.

---

## 8. Sources opened and not opened

Opened (every URL cited inline, plus): Nimble training/dataset docs and v3 index card; Intern-Decision-4B card; reflex weight-classes page; Microsoft Learn MD1 docs; board manifest; the three public item files; HF license files for Kimi K3, GLM-5.3, Qwen3.8-Max and license metadata for 30 models; the OpenRouter endpoints API; BenchLeader. arXiv papers in §3 were read at abstract level only; no numbers are quoted beyond the abstracts. LLM-as-Jev was read in full (PDF).

**Not opened or failed:** H2O's blog post (JavaScript-only page, no content); MD1 Tech Community post (HTTP 403); TypeSafe FAQ answers on training data (collapsed, not in HTML); `noul_method_v16.py` (404, so v1.6.1 noul/type weights are unverified); gpt-oss-safeguard technical report PDF; Nimble v4 training details (none published in what I found); Mistral Large 3 pricing; Qwen3.5-27B JevBench reference price; per-provider ToS beyond a regex scan; the SSDM-side training rows (outside the repo; T0a/T0b need them).

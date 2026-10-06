# 01 — Attention→recurrent conversion and SSM distillation

Written 2026-10-06 from web verification (arXiv abstracts, search snippets) plus `reviews/02`, `reviews/03`, `plans/claude/PLAN.md`. Every paper below was located on arXiv during this session unless marked **[unverified]**; numbers come from abstracts/snippets unless marked "(from memory)". Evidence legend: **R** replicated by ≥2 independent groups or reproduced in follow-ups; **S** single peer-reviewed paper; **P** preprint, single group; **X** speculative / our extrapolation. Cost legend: $0 engineering only; $ <$100; $$ $100–500; $$$ $500–2k; $$$$ >$2k. Cost basis: H100 at $2.7–4/h; 6·N·D FLOP at ~35% MFU ⇒ 1B tokens through a 9B model ≈ 40–50 H100-h ≈ $120–200 (teacher forward adds ~0.5×).

## What the plan needs from this domain

Stage B (PLAN.md) converts the 8 attention layers of Qwen3.5-9B (24 GDN + 8 attention) into Gated-DeltaNet layers "MOHAWK-style: layer-wise output matching, then end-to-end KL, ~1–2B tokens" with gate "within 2 pts of stock readout on the dev slice and on a long-context retrieval probe". Stage 0.4 swaps 1→8 layers one at a time at ~200M tokens. The question this domain has to answer is: *what is the cheapest recipe, and what does the literature say we will lose?*

**Headline answer (R):** every conversion study that reports it finds that the *fully* attention-free student loses in-context retrieval and few-shot MMLU relative to the teacher, that delta-rule destinations (GDN / RWKV-7 / GKA) lose the least, and that keeping 1–3 attention layers (or ~15% of heads) recovers retrieval almost entirely. No paper found reports an attention-free conversion matching its teacher on multi-needle retrieval beyond the distillation length. Plan accordingly: the hybrid fallback is the expected outcome, not a failure branch.

---

## Techniques

### 1.1 MOHAWK (Matrix Orientation → Hidden-state Alignment → Weight transfer + KD)
- **Cite:** Bick, Li, Xing, Kolter, Gu, "Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models", arXiv 2408.10189, NeurIPS 2024. Code: goombalab/phi-mamba.
- **Evidence:** R (reused by Llamba 2502.14458 same group; Zebra-Llama's ILD, CAB 2510.19266 and Priming 2605.08301 are independent re-implementations of the stage structure).
- **Known:** Phi-Mamba (1.5B, Mamba-2 student of Phi-1.5) with 3B tokens; ">33× less token budget … 5% higher average accuracy than the next best" open subquadratic model at the time. Stage 1 matches the *mixing matrices* (attention matrix vs the SSM's materialised matrix form), stage 2 matches block outputs, stage 3 distils end-to-end.
- **Failure modes:** the student is pure Mamba-2; retrieval/MMLU gaps vs teacher are reported but the abstract does not quantify them (Llamba does, below). Stage 1 needs the recurrent layer's *dual (matrix) form*; for GDN this exists (Yang et al. 2412.06464's chunkwise form), so the recipe applies to our target.
- **Cost at our scale:** stages 1–2 are per-layer and need only teacher forward + gradient through one block: 8 layers × ~100M tokens each ≈ $ – $$. Stage 3 at 1–2B tokens ≈ $$–$$$. Matches PLAN.md's $700.
- **Serves:** decision model (Stage B); LM inherits the backbone.
- **How we'd use it:** exactly as the plan says, with two changes from §1.20–1.21 below: convert layers in *importance order* (least important first) and gate each swap with an accept-or-rollback test.

### 1.2 Mamba-in-the-Llama (attention-weight reuse + progressive distillation + hybrid ratio sweep)
- **Cite:** Wang, Paliotta, May, Rush, Dao, arXiv 2408.15237, NeurIPS 2024.
- **Evidence:** S for the recipe; **R for the ratio curve** (same monotone pattern in Jet-Nemotron, "Some Attention", Priming).
- **Known:** 20B tokens, Llama-3-8B-Instruct teacher. Perplexity-style degradation vs teacher: **1.03 at 50% attention, 1.09 at 25%, 1.22 at 6.25%, 3.36 at 0% attention.** The 25% hybrid matches the teacher on chat benchmarks (29.61 LC win-rate AlpacaEval 2). Reusing Q/K/V/O weights to initialise Mamba projections is the key cost saver.
- **Failure modes:** the jump from 6.25% → 0% attention is ~3× the loss of all previous steps combined. This is the single most important number for Stage B.
- **Cost:** initialisation trick is free; their 20B tokens is $$$$ for us, but their ablations show most gain comes early.
- **Serves:** decision model (sets the prior that attention-free will cost; defines the fallback).
- **How we'd use it:** initialise each new GDN layer's q/k/v/o from the attention layer's projections (same trick), and pre-register the 6.25% point (keeping **1 of 8** attention layers in Q9) as the first fallback.

### 1.3 LoLCATs (attention transfer + LoRA)
- **Cite:** Zhang et al., arXiv 2410.10254, ICLR 2025.
- **Evidence:** S; the MMLU loss is independently reproduced as a pattern by Lizard (which uses it as the baseline it beats by 9.4–24.5 MMLU points).
- **Known:** two steps: train linear attentions to match softmax outputs (MSE), then LoRA. Very cheap (abstract: "orders of magnitude less memory and compute"; the "0.2% params / 40M tokens" figures are from memory **[unverified]**). **5-shot MMLU lags Llama-3-8B teacher by 13.8 points.**
- **Failure modes:** feature-map linear attention (Hedgehog-style) has no delta rule and no gating; retrieval and few-shot knowledge drop. Not the right destination layer for us.
- **Cost:** $.
- **Serves:** decision model only as a negative control (what a cheap, additive-state conversion costs).

### 1.4 Llamba (MOHAWK at Llama-3 scale)
- **Cite:** Bick et al., arXiv 2502.14458 (Feb 2025).
- **Evidence:** P (same group as MOHAWK).
- **Known:** Llamba-1B/3B/8B distilled from Llama-3.x into Mamba-2 with 12B tokens (<0.1% of teacher pretraining). Benchmark gaps vs teacher exist but exact MMLU figures were not retrieved this session **[unverified]**.
- **Failure modes:** pure-Mamba student; same retrieval ceiling as 1.1.
- **Cost:** 12B tokens is $$$$ at 9B.
- **Serves:** decision model (reference point for what MOHAWK buys at 8B).

### 1.5 RADLADS (rapid conversion to RWKV-6/7 variants)
- **Cite:** Goldstein, Alcaide, Lu, Cheah, arXiv 2505.03005, COLM 2025. Code: recursal/RADLADS-paper.
- **Evidence:** S.
- **Known:** three-stage conversion with **350–700M tokens (<0.005% of teacher tokens)**; Qwen2.5 7B/32B/72B converted to RAD-RWKV6 and RAD-RWKV7 (fully attention-free); "<$2,000 USD" for 72B; quality "close to the original transformer" on standard benchmarks. Long-context/retrieval numbers were not in the abstract **[unverified]**.
- **Failure modes:** standard-benchmark parity does not imply retrieval parity (see 1.17–1.19). RWKV-7 state is a delta-rule matrix state — same class as GDN, so the result transfers to our target layer type.
- **Cost:** ~$$ for a 9B conversion by their token counts. The cheapest attention-free recipe with a public codebase.
- **Serves:** decision model. Directly relevant to the RWKV7-13.3B arm *and* as an alternative recipe for Q9-AF (convert the 8 attention layers to RWKV-7 time-mix instead of GDN; both are delta-rule).
- **How we'd use it:** run RADLADS' stage schedule as the "cheap arm" of Stage B against MOHAWK's; whichever passes the retrieval probe at lower cost wins.

### 1.6 ARWKV (Qwen2.5 → RWKV-7 time mixing)
- **Cite:** Lin, Li, Yue, Liu, arXiv 2501.15570.
- **Evidence:** P.
- **Known:** "attention alignment in stage 1 is crucial"; 7B conversion fits one A100-80G; 32B on 4×8 A100.
- **Failure modes:** evaluation breadth limited; retrieval not reported.
- **Serves:** decision model (corroborates that stage-1 attention alignment matters).

### 1.7 Liger (gate construction from pretrained key weights; intra-layer hybrid)
- **Cite:** Lan, Sun, Hu, Du, Cheng, arXiv 2503.01496. Code: OpenSparseLLMs/Linearization.
- **Evidence:** P/S.
- **Known:** reuses key-projection weights to build gates (no new params); 0.02B tokens; recovers 90% (Mistral) / 93% (Llama-3) with 0.085% LoRA. "Liger Attention" mixes a sliding-window softmax branch into the layer.
- **Failure modes:** the 93% figure relies on the intra-layer SWA branch (see 1.18: such hybrids tend to route everything through SWA). The gate-from-keys trick is reusable on its own.
- **Serves:** decision model (initialisation trick for GDN gates).

### 1.8 Lizard (GLA + SWA + meta memory)
- **Cite:** Nguyen, Zhang, et al., arXiv 2507.09025.
- **Evidence:** P.
- **Known:** "near-lossless" teacher recovery, +9.4–24.5 5-shot MMLU over previous linearisation methods; good associative recall.
- **Failure modes:** not attention-free (sliding window). Component-collapse risk per 1.18.
- **Serves:** neither project directly; it is the hybrid control.

### 1.9 LAWCAT (conv-across-tokens + normalised GLA; long-context extrapolation)
- **Cite:** arXiv 2509.18467, EMNLP 2025 Findings. Code: zeyuliu1037/LAWCAT.
- **Evidence:** S.
- **Known:** Mistral-7B student distilled on **1K-length** sequences reaches **>90% passkey retrieval to 22K** (teacher limit 8K). Attention-free.
- **Failure modes:** single-needle passkey only; no multi-key or distractor results in abstract.
- **Cost:** $$ (short-sequence distillation is cheap).
- **Serves:** decision model. The short causal Conv1D is the same component that 2609.16183 finds dominates recall (+0.5) in fixed-state recurrences; keep it in the converted layer.

### 1.10 Jet-Nemotron / PostNAS (once-for-all supernet + layer-placement search)
- **Cite:** Gu et al. (NVIDIA/MIT), arXiv 2508.15884.
- **Evidence:** P (industry, single group), but the placement finding is R across 1.11, 1.12, 1.20.
- **Known:** augments a pretrained full-attention model with linear-attention paths, random path sampling with feature distillation, then beam search for attention placement under a budget. "Not all attention layers contribute equally"; "retaining a few full-attention layers … is essential for … retrieval".
- **Failure modes:** search cost; they keep attention layers (not attention-free).
- **Cost:** supernet training is $$$$ at 9B; the *idea* (search placement rather than uniform) is free.
- **Serves:** decision model (which of the 8 layers to convert first / last).

### 1.11 KL-guided layer selection for hybrid distillation
- **Cite:** Li, Yang, Tan, Mishra, Panda, Zhou, Kim, arXiv 2512.20569 (Dec 2025).
- **Evidence:** P.
- **Known:** layer-importance scores from a small amount of generic-text training; then attention-weight transfer → hidden-state alignment → KL matching → finetune; GDN as the linear layer; beats uniform interleaving and other heuristics.
- **Cost:** importance scoring is $ (small training); the pipeline is the same cost as 1.1.
- **Serves:** decision model.
- **How we'd use it:** compute the KL-importance of each of Q9's 8 attention layers *before* Stage 0.4; convert in ascending order; the ranking also predicts which layer the fallback keeps.

### 1.12 FlashMorph (learned layerwise gates on synthetic retrieval data)
- **Cite:** Lan, Zheng, Ren, Xia, Wang, Xiao, Qiu, Cheng, "Morphing into Hybrid Attention Models", arXiv 2606.30562 (Jun 2026).
- **Evidence:** P.
- **Known:** each attention layer gets a converted linear branch; weights frozen; layer gates optimised on synthetic retrieval data with a linearisation regulariser; discretised under a budget; then distil. Treats layer importance as interdependent.
- **Serves:** decision model (same purpose as 1.11, with a retrieval-specific objective; cheaper than PostNAS).

### 1.13 HALO / HypeNet (hybrid conversion with long-context focus)
- **Cite:** Chen et al., "Hybrid Linear Attention Done Right", arXiv 2601.22156 (Jan 2026).
- **Evidence:** P.
- **Known:** Qwen3 → hybrid with **2.3B tokens (<0.01%)**; claims prior methods need >10B tokens *and* give poor long-context performance; introduces HyPE position encoding for length generalisation.
- **Failure modes:** hybrid (keeps attention); no attention-free arm.
- **Serves:** decision model (the token budget and long-context evaluation protocol are what we should copy).

### 1.14 Priming (hybrid SSMs from pretrained transformers; GKA > GDN > Mamba-2)
- **Cite:** Chattopadhyay, Nunez, Kaul, Bowman, Becker, Zancato, Thomas, Xia, Soatto, arXiv 2605.08301 (May 2026).
- **Evidence:** P (industry group).
- **Known:** <0.5% of teacher tokens; 8B and 32B hybrids at 128K context; **expressiveness hierarchy GKA > GDN > Mamba-2 predicts long-context reasoning**; hybrid GKA-32B is +3.8 avg reasoning points over Qwen3-32B and within 1% of a transformer post-trained on the same data.
- **Failure modes:** hybrids only; "Gated KalmaNet" kernels are not in flash-linear-attention as far as found **[unverified]**.
- **Serves:** decision model (choose GDN over Mamba-2 as destination; corroborates Qwen's own choice).

### 1.15 Zebra-Llama (MLA + Mamba-2 hybrids, intermediate-layer distillation, sensitivity-aware composition)
- **Cite:** AMD, arXiv 2505.17272; HF models amd/Zebra-Llama-*.
- **Evidence:** P.
- **Known:** 7–11B tokens, 8B teacher; Zebra-Llama-8B beats Minitron-8B by 7% few-shot with 8× fewer tokens. "SMART" picks which layers become Mamba by sensitivity.
- **Serves:** decision model (another sensitivity-ordered conversion; confirms 1.10–1.12).

### 1.16 CAB — attention bridge for any-transformer-to-Mamba distillation
- **Cite:** Wang et al., arXiv 2510.19266 (v2 Sep 2026).
- **Evidence:** P.
- **Known:** token-level intermediate supervision through a lightweight bridge + layer-wise alignment; gains under limited supervision; no inference overhead. Comparison to MOHAWK not retrieved **[unverified]**.
- **Serves:** decision model (candidate for the data-poor regime of Stage 0.4).

### 1.17 Sliding-window attention beats retrofitted linear attention (negative result)
- **Cite:** Jolicoeur-Martineau, Sukthanker, Cameron, Gervais, arXiv 2608.28444 (Aug 2026).
- **Evidence:** P, but consistent with SUPRA (2405.06640, "persistent in-context learning and long-context modeling shortfalls"), LoLCATs, Mamba-in-the-Llama.
- **Known:** SWA + attention sinks, *no training*, matches or exceeds most retrofitted linear-attention models; **2–10× higher on NIAH and BABILong**.
- **Implication for us:** any attention-free claim must be benchmarked against the zero-cost SWA+sinks baseline on the same backbone, and the Index retrieval area will be where it loses.
- **Serves:** both (defines the honest control arm).

### 1.18 Component collapse in hybrid conversions
- **Cite:** Benfeghoul et al., "Paying Attention to Hybrid Attention: Untangling the Issues with Conversion Methods", arXiv 2510.05901.
- **Evidence:** P.
- **Known:** converted hybrids "inadvertently bypass the linear component, relying almost entirely on SWA"; common-sense benchmarks hide it. Fixes: inference-time hybridisation, HedgeCATs (attention-weight transfer + targeted LoRA), scheduled sliding-window dropout.
- **Implication:** if we keep any attention in a fallback hybrid, measure what the recurrent layers actually carry (ablate the attention branch) before claiming a "mostly recurrent" model.

### 1.19 What matters in linearising (architecture > scale)
- **Cite:** Haller, Golde, Akbik, arXiv 2504.14366 (v2 Jan 2026).
- **Evidence:** P (7 token mixers, 140M–1.7B).
- **Known:** performance hierarchies are stable across scale; gaps appear early and persist past ~10B tokens; **gated delta-rule mixers keep long-context retrieval precision, additive mixers suffer "irreversible state saturation"**; MMLU declines for non-gated forms. "State resolution capacity is the bottleneck, not training compute."
- **Serves:** decision model (do not expect more tokens to fix a bad destination layer; GDN is the right one).

### 1.20 Gather-and-Aggregate heads (which attention actually matters)
- **Cite:** Bick, Xing, Gu, "Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism", arXiv 2504.18574.
- **Evidence:** P from the MOHAWK group; corroborated by 2510.19861 (below).
- **Known:** retrieval = a Gather head + an Aggregate head, concentrated in a handful of heads; **disabling one in a pruned Llama-3.1-8B drops MMLU 66% → 25% (chance)**; SSMs produce smooth rather than sharp attention and fail at the Aggregate step; **replacing a single G&A head in a pretrained SSM with an attention variant boosts retrieval and benchmark scores**.
- **Companion:** Michalak & Abreu, "Some Attention is All You Need for Retrieval", arXiv 2510.19861: on RecurrentGemma-2B/9B and Jamba-Mini, retrieval depends exclusively on attention; **15% of heads keep near-perfect retrieval with 84% of MMLU**; removing attention → 0% retrieval with no SSM compensation.
- **Implication (the plan's "Distillation is the hinge"):** the expensive layers in Q9 are the ones hosting G&A heads. Locate them (implicit-attention sharpness on retrieval prompts, see `03`) before converting; expect the retrieval probe to fail when they go.
- **Cost:** $ (head-ablation sweep on the stock model).
- **Serves:** decision model.

### 1.21 Accept-or-rollback layer conversion inside an already-hybrid Qwen3.5
- **Cite:** Mohsenzadegan, Tavakkoli, Kyamakya, "TinyCeNN-LM", arXiv 2609.21139 (Sep 2026).
- **Evidence:** P, weak evaluation (200-item sanity set; 28–32% downstream accuracy numbers are near chance).
- **Known:** selective replacement of attention layers in SmolLM2-135M and **Qwen3.5-0.8B** (the same family and hybrid pattern as our target) with a dual gate: representation-fidelity *and* ΔNLL thresholds; layers 3, 7, 11 accepted in Qwen3.5-0.8B (ΔNLL +0.021), others rejected on fidelity even when NLL was fine.
- **Implication:** this is Stage 0.4's protocol in miniature; copy the dual gate (fidelity + NLL + our dev-slice/retrieval gate), not their layer type.
- **Serves:** decision model.

### 1.22 Reasoning-capable distilled hybrids (for the LM project)
- **Cite:** M1, arXiv 2504.10449 (Llama-3.2-3B → Mamba hybrid with 6/28 attention layers; distil → SFT → RL; MATH500 84.0, AIME24 28.5). Apriel-H1 2511.02651, "Thinking Slow, Fast" 2502.20339 (same idea at larger scale) **[not fetched]**.
- **Evidence:** P.
- **Serves:** LM project: shows a distilled hybrid backbone can carry generative reasoning after SFT/RL, and that it still keeps ~20% attention.

### 1.23 Older linearisation baselines (context only)
- Hedgehog (2402.04347, ICLR 2024): softmax-mimicking feature maps; SUPRA (2405.06640): uptraining at 5% cost with "persistent ICL and long-context shortfalls"; DiJiang **[not verified this session]**. These establish the additive-linear-attention ceiling that 1.19 explains.

### 1.24 Qwen's own ablations (why the 3:1 hybrid keeps attention)
- **Cite:** "On the Design of Qwen3.8-Next Architecture", arXiv 2608.30320 (snippet-level only, **[not fetched]**); Raschka's architecture notes; Qwen3-Next blog.
- **Known (snippet):** GDN-hybrid beats full-attention on 8/9 benchmarks and SWA-hybrid on 7/9; the stated reason for keeping 1-in-4 attention is "periodic token-level retrieval that is difficult for any finite-state recurrent memory to reproduce exactly".
- **Implication:** the model we are converting was *designed* with the retrieval function in the attention layers; its GDN layers were never trained to do it.

---

## What is known about retrieval / knowledge retention after conversion (summary table)

| Student type | Example | Retrieval / MMLU outcome | Evidence |
|---|---|---|---|
| Feature-map linear attention (additive) | LoLCATs, SUPRA, Hedgehog | MMLU −13.8 (LoLCATs); ICL and long-context shortfalls (SUPRA) | R |
| Mamba-2 (diagonal selective) | Phi-Mamba, Llamba | good average accuracy at 1–3B tokens; retrieval gap unquantified in abstracts | S |
| Delta-rule (GDN / RWKV-7 / GKA) | RADLADS, ARWKV, Priming, 2504.14366 | least degradation; keeps long-context retrieval precision; hierarchy GKA>GDN>Mamba-2 | P (×4 groups) |
| Attention-free vs 6.25% attention | Mamba-in-the-Llama | 3.36 vs 1.22 degradation | S, pattern R |
| Hybrid with ≥15% heads | 2510.19861 | near-perfect retrieval, 84% MMLU | P |
| SWA + sinks, no training | 2608.28444 | 2–10× better NIAH/BABILong than retrofitted linear | P |

**Nothing found:** an attention-free conversion that matches its teacher on *multi*-key retrieval or on BABILong. LAWCAT's 22K single-passkey is the strongest attention-free retrieval claim located.

---

## How we'd run Stage B (concrete, ordered)

1. **Before converting ($):** on stock Qwen3.5-9B run (a) KL-importance per attention layer (1.11), (b) head-ablation on a retrieval + MMLU slice to locate G&A heads (1.20), (c) SWA+sinks control on the same slice (1.17). Pre-register: "convert in ascending KL-importance; the layer(s) holding G&A heads are converted last".
2. **Per-layer swap (Stage 0.4, $$):** initialise GDN q/k/v/o from attention projections (1.2) and gates from key weights (1.7); keep the short causal conv (1.9); MOHAWK stage 1 (matrix orientation against GDN's chunkwise matrix form) and stage 2 (block output MSE) on ~100–200M tokens; dual accept/rollback gate (1.21): representation fidelity, ΔNLL, dev-slice per area, S-NIAH + MQAR at 1K/4K/16K.
3. **End-to-end (Stage B, $$–$$$):** KL to teacher on 1–2B tokens with a retrieval-heavy mixture (JRT/Birdie-style objectives, see `02`), length curriculum to the Index's record-length distribution.
4. **Fallback, pre-registered:** if the retrieval probe fails after the last swap, keep the single highest-importance attention layer (the 6.25% point of 1.2) or 15% of its heads (1.20) and report the model as "7/8 converted". Check component collapse (1.18) by ablating the retained attention.
5. **Cheap parallel arm:** RADLADS schedule to RWKV-7 time-mix (1.5) on the same 8 layers; pick by retrieval probe per dollar.
6. **Report** both arms against the SWA+sinks control, per Index area and per record-length bin, as the review demanded.

## Could not verify this session
- Exact MMLU/retrieval deltas for Llamba and RADLADS vs teachers (abstracts only).
- LoLCATs' "0.2% params / 40M tokens" figures.
- Whether Gated KalmaNet (Priming) has public kernels.
- Qwen3.8-Next design paper details beyond snippets.
- DiJiang, Apriel-H1, "Thinking Slow, Fast" details (not fetched).

# 02 — Fixing SSM recall without attention

Written 2026-10-06. Same legend as `01` (R/S/P/X evidence; $0/$/$$/$$$/$$$$ cost). Papers were located on arXiv this session unless marked **[unverified]**.

## What the plan needs from this domain

The pressure test (`reviews/02` §1, §3) established that a fixed-size state compresses the record ≥30× before any option is read, that this is the measured recall gap of fixed-state models (Jelassi 2402.01032; Waleffe 2406.07887), and that ~38% of the Index (Retrieval 20%, Tools 18.3%) sits on it. The plan's answers are (i) options/questions first, (ii) **snapshot forks** from S chunk-boundary states combined by logsumexp, (iii) an integrator head with structural λ=1. This file collects what the literature says about each lever and what else exists.

**Headline:** three independent lines converge: (a) *know the query before the data* (JRT: +11 points across 16 recurrent LMs), (b) *delta-rule states* are the best fixed-size memories (GDN/RWKV-7/Mamba-3/MesaNet), and (c) *chunk-and-select at inference* (OPRM: +14–51% LongBench on four recurrent 7–9B models, training-free) beats feeding the whole context into one state. The snapshot-fork design is the multi-chunk generalisation of (c); OPRM's finding that a *single best chunk* already beats full-context is both encouraging and a warning that cross-chunk relations are what fixed states lose.

---

## Techniques

### 2.1 Just Read Twice (JRT-Prompt, JRT-RNN)
- **Cite:** Arora et al., arXiv 2407.05483 (NeurIPS 2024).
- **Evidence:** S for the method; the ordering effect is R (it is the explanation of the Based/Zoology results and is reproduced by every "query-first" study).
- **Known:** data order determines what a fixed state must keep. JRT-Prompt repeats the context so the model sees all orders: **+11.0±1.3 points averaged over 16 recurrent LMs × 6 ICL tasks**, 11.9× FA2 prefill throughput. JRT-RNN uses non-causal *prefix-linear-attention* for the prompt: 99% of transformer quality at 360M/30B tokens, 96% at 1.3B/50B.
- **Failure modes:** doubling the prefill; non-causal prefix needs the architecture to support it; the harness's "fixed rendering" rule must be checked before using read-twice on the Index (review §13.3).
- **Cost:** $0 (input transform) / $ (prefix-linear-attention arm needs training).
- **Serves:** decision model (the ordering rule); LM (the planner reads K the same way).
- **How we'd use it:** options+questions → record is JRT-Prompt in disguise; test "options → record → options" too. For the decision model, a *non-causal* read of the record is legitimate (no generation over it), so a prefix-linear-attention or bidirectional (Birdie/Hydra-style) pass over the record is an attention-free arm worth one run.

### 2.2 Based and the recall–throughput frontier; MQAR as the diagnostic
- **Cite:** Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff", arXiv 2402.18668 (ICML 2024); Zoology / MQAR, arXiv 2312.04927.
- **Evidence:** R (MQAR is the standard diagnostic across GDN, RWKV-7, Mamba-3, log-linear attention papers).
- **Known:** recall accuracy is a function of **state size in bytes**; Based (linear attention + short sliding window) beats Mamba by 10.36 points on recall-intensive tasks at matched state; 24× FA2 generation throughput.
- **Failure modes:** Based's sliding window is attention (small, fixed); the pure linear-attention component alone does not reach the frontier.
- **Cost:** $0 to adopt MQAR as our probe.
- **How we'd use it:** MQAR at (num pairs, sequence length) grid = our standard "state capacity" card per checkpoint (see `08`).

### 2.3 Delta-rule fixed-size memories (the destination family)
- **Cite:** Gated DeltaNet, Yang et al., arXiv 2412.06464 (ICLR 2025); RWKV-7 "Goose", arXiv 2503.14456; DeltaNet with negative eigenvalues, arXiv 2411.12537; Mamba-3, arXiv 2603.15569; "Illusion of State", arXiv 2404.08819.
- **Evidence:** R (multiple groups; Qwen3-Next/3.5 production use).
- **Known:** GDN beats DeltaNet and Mamba-2 on S-NIAH (RULER); RWKV-7 is perfect on in-context and noisy recall at MQAR scale and recalls 72.93% at 256 KV pairs with a 8192-dim WKV state; negative eigenvalues unlock parity/state tracking; Mamba-3 adds complex-valued/MIMO state. The delta rule = online least-squares (test-time regression, 2.4), which is why it erases and overwrites keys rather than accumulating.
- **Failure modes:** capacity is still ~state size; "fuzzy" recall (paraphrased keys) is weaker than exact recall; interference from distractors (2.9).
- **Serves:** decision model (confirms GDN as the converted layer type and as the head's recurrence in Head B of the partner plan).

### 2.4 Test-time regression / test-time training memories (higher-capacity fixed-size memory)
- **Cite:** Wang, Shi, Fox, "Test-time regression", arXiv 2501.12352; MesaNet, arXiv 2506.05233; Titans, arXiv 2501.00663; Atlas, arXiv 2505.23735; LaCT, arXiv 2505.23884; TTT (2407.04620, from memory **[unverified ID]**).
- **Evidence:** P for each; Titans/Atlas are Google-only and their BABILong/NIAH claims (">2M context"; "+80% at 10M on BABILong") have no independent replication found this session **[unverified]**.
- **Known:** memory as a learner updated by gradient steps at test time; MesaNet solves the *cumulative* regularised least-squares (not instantaneous) in a chunkwise-parallel form; LaCT uses very large chunks (2K–1M) with window attention inside the chunk.
- **Failure modes:** new kernels; training from scratch in all papers; LaCT's in-chunk attention is attention. None has been used as a conversion destination.
- **Cost:** $$$$ for us except as a drop-in layer if kernels exist in flash-linear-attention (MesaNet/Titans availability **[unverified]**).
- **Serves:** watch-list for the from-scratch pilot core (partner plan); not for the $5k conversion.

### 2.5 Log-linear attention (logarithmically growing state)
- **Cite:** Guo et al., arXiv 2506.04761, ICLR 2026.
- **Evidence:** S.
- **Known:** replaces the fixed state with O(log L) hidden states (Fenwick-tree structure); Mamba-2 and GDN variants; matmul-rich training form.
- **Failure modes:** violates "fixed-size state" literally (state grows O(log L)); no conversion recipe.
- **Serves:** decision model only as a definitional edge case; if the definition admits O(log L) states, this is the natural attention-free recall upgrade.

### 2.6 Exact bounded sidecar memory (HOLA)
- **Cite:** Cui, "A Hippocampus for Linear Attention", arXiv 2607.02303.
- **Evidence:** P.
- **Known:** adds a *bounded* exact KV cache next to the recurrent state; eviction by the committed residual β·‖e‖; WikiText ppl 27.32→22.92 (below Transformer++ 26.88); beats GDN on RULER NIAH to 32K (16× training length).
- **Failure modes:** it is a small attention cache (bounded, not growing) — outside the strict definition; cost of the cache lookups.
- **Serves:** both, as the "cheapest violation": a bounded exact memory for *schema/options* only (trusted channel) would keep the plan's provenance story while fixing recall for what the head needs; worth a pre-registered arm.

### 2.7 Overflow prevention / chunked inference for recurrent LLMs (OPRM)
- **Cite:** Ben-Kish, Zimerman, Mirza, Wolf, Glass, Karlinsky, Giryes, arXiv 2505.07793 (v2 Sep 2025). Code: OPRM.
- **Evidence:** S (4 models, 2 benchmarks).
- **Known:** training-free; split the context into chunks, prefill each independently (speculative prefill), **select the chunk with lowest output entropy**, decode from it. LongBench: Falcon3-Mamba +14%, Falcon-Mamba +28%, RecurrentGemma +50%, RWKV6-Finch +51%; SOTA on LongBench v2; **single-chunk beats full-context even on tasks that presumably need cross-chunk relations**; recurrent models "underutilise long contexts even when trained for them".
- **Failure modes:** by construction drops cross-chunk relations; entropy selection is a heuristic; aggregation over chunks not studied.
- **Cost:** $0 (inference-only); S× prefill compute (parallel).
- **Serves:** decision model — **this is the snapshot-fork head at inference, minus the aggregation.**
- **How we'd use it:** implement snapshot forks as OPRM with three aggregators — max (OPRM), logsumexp (plan), learned mixture — and measure on the Index retrieval area per record length. OPRM's result predicts that max over chunks already recovers most of the gap for single-fact questions; logsumexp is needed for multi-fact (count/aggregate) questions, which the Tools area has.

### 2.8 Selective re-forwarding and decimation (ReMamba, DeciMamba, LongMamba)
- **Cite:** ReMamba, arXiv 2408.15496; DeciMamba, arXiv 2406.14528 (ICLR 2025); LongMamba, arXiv 2504.16053.
- **Evidence:** S each.
- **Known:** ReMamba: first pass selects top-k hidden states, second pass re-injects them via the selective gate: +3.2 LongBench, +1.6 L-Eval, "almost on par with same-size transformers". DeciMamba: use ‖Δ_t‖ as token importance, drop unimportant tokens before S6 to extend the effective receptive field (training-free length extrapolation). LongMamba: training-free receptive-field enlargement.
- **Failure modes:** Mamba-1-specific mechanisms (Δ gate); two passes double prefill; ReMamba needs light training.
- **Serves:** decision model (Falcon-Mamba diagonal arm specifically). DeciMamba's Δ-norm importance doubles as an interpretability signal (`03`).

### 2.9 State collapse and state capacity (what breaks past training length)
- **Cite:** Chen et al., "Stuffed Mamba", arXiv 2410.07145 (COLM 2025).
- **Evidence:** S.
- **Known:** Mamba-2 performance collapses beyond training length because a few outlier channels explode and the model "never learned to forget" when the state is over-sized for the training context; three training-free mitigations allow >1M tokens; continual training on longer sequences also works.
- **Failure modes:** mitigations are Mamba-2-specific; GDN/RWKV-7 analogues not studied here.
- **Cost:** $ (diagnostic on our backbones).
- **How we'd use it:** before promising anything on 16K–64K records, run each backbone at 2× and 4× its training length on NIAH and watch for the collapse signature (outlier channels in the state). Falcon-Mamba-7B trained at 8K; the Index record-length distribution (review §15 step 1) decides whether this matters.

### 2.10 Interference wall and curriculum (distractors, not length, kill recall)
- **Cite:** Boesch & Wee, "Anatomy of Associative Recall in Fixed-State Recurrences", arXiv 2609.16183 (Sep 2026).
- **Evidence:** P, toy scale.
- **Known:** with components matched, the **short causal convolution** dominates (+0.5 recall), rank-1 delta vs diagonal adds +0.19/+0.32 at 16/32 pairs; a model that solves 32-pair recall "falls to chance retrieving 4 pairs from a distractor haystack" — an *interference* wall from sparse supervision; a distance-based curriculum takes it from 0.021 to 1.000 on the same architecture (lock-in 7/10 vs 1/10 seeds).
- **Failure modes:** toy; unknown transfer to 9B.
- **How we'd use it:** the Index "state" field is a haystack with distractors; add a distractor-haystack probe (few needles, many irrelevant records) to the retrieval gate, and a distance curriculum to the Stage B/C mixture.

### 2.11 Training objectives that teach recall without changing the architecture (Birdie; paraphrase augmentation; CoT distillation)
- **Cite:** Birdie, Blouir, Smith, Anastasopoulos, Shehu, arXiv 2411.01030 (EMNLP 2024); Lee et al., arXiv 2510.26912 (paraphrase-augmented continual training improves hybrid recall; sequential hybrids better short, parallel better long); "Recall with Reasoning" CoT distillation for Mamba, arXiv 2505.03320 (EMNLP 2025).
- **Evidence:** S each.
- **Known:** Birdie: RL-chosen mixture of selective-copying, infilling, QA objectives + bidirectional processing markedly improves retrieval of SSMs with no architectural change. 2510.26912: paraphrase augmentation beats structural modifications targeting recall.
- **Cost:** $0 extra (changes the mixture of Stage B/C tokens).
- **How we'd use it:** Stage B's 1–2B distillation tokens should be ~20–30% retrieval objectives (copy, MQAR-style QA over records, paraphrased-key recall) rather than generic text; this is the cheapest lever in this file.

### 2.12 RNNs are not Transformers (yet): what closes the representational gap
- **Cite:** Wen, Dang, Lyu, arXiv 2402.18510 (ICLR 2025).
- **Evidence:** S (theory + experiments).
- **Known:** CoT improves RNN expressiveness but not enough; the bottleneck is in-context retrieval; **RAG or a single transformer layer** closes the gap to all polynomial-time problems.
- **Serves:** both. For the LM project, "retrieval as a tool call decided by SSM-1" is the RAG route and keeps the backbone attention-free.

### 2.13 Memory-augmented recurrence (from-scratch architectures; watch list)
- MARCH content-routed state anchors (2608.12435), Sparse Delta Memory (2607.07386), RAM-Net selectively addressable memory (2602.11958), Mamba with Hierarchical Memory (2608.02347), FG2-GDN doubly fine-grained control (2604.19021), Adaptive memory decay for log-linear attention (2605.06946). All P, all from scratch, none with a conversion recipe. **Not** for the $5k conversion; candidates for the partner plan's 100–400M pilot core.

### 2.14 Gated Slot Attention / bounded-slot memories
- **Cite:** Zhang et al., arXiv 2409.07146 (NeurIPS 2024) **[not fetched; from search listing]**.
- **Serves:** context only (ABC-style bounded slots are the ancestor of HOLA-style sidecars).

### 2.15 Query-first ordering evidence outside recurrent models
- A snippet-level result that "question first" helps gpt-4o-mini at 64K (arXiv 2502.12462) is **not** evidence about recurrent models; the recurrent-model evidence is JRT (2.1). Do not cite the former for our ordering claim.

---

## How we'd use this domain (decision model, ordered by cost)

1. **Ordering (free):** schema + options + questions before the record (JRT). Add "questions again after the record" as an arm since questions carry option lists (review §1 equivariance caveat): measure both and report order-agreement.
2. **Retrieval-objective mixture (free):** Birdie/paraphrase/distance-curriculum objectives in Stage B/C tokens (2.10, 2.11).
3. **Snapshot forks = OPRM with aggregation ($0 compute change):** chunk the record at S≈L/512 boundaries, fork each option from each chunk state, aggregate by max / logsumexp / learned; compare to final-state fork and to the Clef cross-attention control.
4. **Probes (card):** MQAR grid, S-NIAH, distractor haystack, state-collapse check at 2×/4× training length.
5. **Bounded exact sidecar for the trusted channel ($):** HOLA-style cache holding only schema/option keys; pre-register as the one allowed "small exact memory" arm and state it as such.
6. **Non-causal record read ($$):** prefix-linear-attention / bidirectional pass over the record (JRT-RNN, Birdie) for the decision model only.

## For the LM project
- Retrieval is a *decision* (which records) made by SSM-1; the realiser never needs the whole K in state (2.12 RAG route). The planner's recall is what matters, and it benefits from the same ordering and chunking.

## Could not verify this session
- TTT arXiv ID; MesaNet/Titans kernel availability; Titans/Atlas long-context claims (no independent replication located).
- Gated Slot Attention details.

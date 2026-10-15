# Conclusions
1. **[INFERENCE] Primary backbone: Qwen3.5-4B inside SSDM’s own hybrid decision stack**, retaining its 24 Gated DeltaNet (GDN) and eight attention layers; spend on transferable decision learning, not another architecture migration.
2. **[INFERENCE] Attention-free research backbone: RWKV7-G1k-7.2B-20260930**, with 2.9B as its mechanism proxy; use native recurrent weights rather than repeat Qwen’s failed eight-layer conversion.
3. **[SIM] At H2O’s exact operating point, one Intelligence point is worth 0.365 Composite, one Cost point 0.362, and one Speed point only 0.153**; halving raw latency adds just 0.258 Composite.
4. **[SIM] Eliminating all self-hosted model latency adds at most 0.574 Composite at that point**, because the evaluator’s 150 ms adjustment dominates; latency engineering cannot close SSDM’s capability deficit.
5. **[SIM] Halving priced input tokens adds 9.031 Cost points and approximately 2.956 Composite**, provided answer quality survives; cheaper GPU rental alone does not change the benchmark’s token-price floor.
6. **[SIM] A 9B model priced at $0.10/M input and consuming 600 tokens/decision cannot reach H2O’s 72.52 Composite even with I=100**, holding H2O’s Calibration and Speed fixed: its gated maximum is 65.28.
7. **[EXT] SSMs’ concrete advantage is bounded recurrent memory and reusable state boundaries, not universally faster prefill**; published Mamba-3 measurements actually favor the Transformer for prefill at tested short/medium lengths.
8. **[MEAS] SSDM’s $28.07, 196M-token attention-free conversion passed KL/fidelity gates but lost 21.77 Decision Index points**; likelihood preservation is not a sufficient capability criterion.
9. **[INFERENCE] A conditional $200 program can fund several bounded 4B adaptation/evaluation cells, but not a credible promise of lossless SSM conversion**; buy completed experiments and retention evidence, not nominal GPU-hours.
10. **[EXT/INFERENCE] Modal and Lambda advertise research grants; NVIDIA’s academic program currently says applications are closed**; count no unawarded credit, run nothing paid now, and preserve the remaining approximately $11.67.

## Evidence conventions and scope

**[EXT] Research date: 2026-10-14.** Every linked source below was opened in this workstream. EXT means external primary-source claim, including a publisher’s measurements, not an independent reproduction. MEAS means an existing internal measurement read from project evidence, not a new experiment. SIM denotes explicitly calculated scenarios; INFERENCE denotes a recommendation or extrapolation. These labels apply to the paragraph/table introduced by them. Unknown means not established in the opened sources; it does not mean unsupported in every possible implementation. No model weights, training jobs, paid APIs, account registrations, or submissions were used. Light arithmetic and a bounded 273 KB public pricing-JSON parse were the only substantive local computations.

**[MEAS] Binding project evidence:** [shared context](local://ssdm-context.md), [dead ends](/tmp/ssdm-state/DEAD_ENDS.md), [jevkit README](/tmp/State-Space-Decision-Model/jevkit/README.md), [score constants](/tmp/State-Space-Decision-Model/jevkit/spec.py), and [meter](/tmp/State-Space-Decision-Model/jevkit/meter.py). The operator permits a mostly-SSM hybrid main line and requires a strictly attention-free research line. The prohibition on further engineering/benchmarking third-party systems remains: the survey is desk research, not a proposal to run a model tournament.

## 1. Backbone survey: the October 2026 feasible frontier

### 1.1 Architecture and quality

**[EXT] Reading rule:** parameters below are advertised language-model sizes unless total/active is explicitly separated; multimodal encoders and embeddings can increase residency. Qwen’s small models are **dense**, notwithstanding family-wide marketing language about MoE. Generation/reasoning benchmark scores are not one-pass decision scores; evaluation settings differ materially. GPQA and GPQA Diamond are identified separately. No common classification suite or Decision Index result was found for most stock backbones; do not convert MMLU-Pro into a predicted Index.

**[EXT] Qwen release ledger:** the [official repository](https://github.com/QwenLM/Qwen3.6) now resolves to Qwen3.8 and records small Qwen3.5 releases on March 2, Qwen3.6 35B-A3B/27B in April, and Qwen3.8 2.4T-A95B/27B in August 2026. **No ≤9B Qwen3.6 or Qwen3.8 release is listed.** Active 3B does not make the 35B model a ≤9B-memory model.

| Backbone [EXT] | Parameters; sequence-layer mix | Native context; license | Reported quality and decision evidence |
|---|---|---|---|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B; 18 GDN + 6 attention, 24 total | 262,144; Apache-2.0 | Nonthinking MMLU-Pro 29.7, IFEval 52.1; thinking 42.3, GPQA 11.9, IFEval 44.0. Intended for prototyping/task-specific adaptation. |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B; 18 GDN + 6 attention | 262,144; Apache-2.0 | Nonthinking MMLU-Pro 55.3, IFEval 61.2; thinking MMLU-Pro 66.5, GPQA 51.6, IFEval 78.6. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B; 24 GDN + 8 attention, 32 total | 262,144, extensible 1.01M; Apache-2.0 | Card reports MMLU-Pro 79.1, GPQA Diamond 76.2, IFEval 89.8 under its published generation protocol. H2O is a post-trained derivative, not a stock result. |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B; 24 GDN + 8 attention | Same as 4B; Apache-2.0 | MMLU-Pro 82.5, GPQA Diamond 81.7, IFEval 91.5. Internal stock decision performance remains substantially below trained competitors. |
| [LFM2 family](https://huggingface.co/LiquidAI/LFM2-2.6B) | 354M/742M/1.17B: 10 gated short-convolution + 6 GQA; 2.57B: 22 convolution + 8 GQA | 32,768; LFM Open License v1.0 | 2.6B: MMLU 64.42, GPQA 26.57, IFEval 79.56. Vendor recommends narrow adaptation and cautions against knowledge-intensive tasks. |
| [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) | 8.3B total/1.5B active; 18 convolution + 6 GQA | 32,768; LFM license | MMLU-Pro 37.42, GPQA 29.29, IFEval 77.58 in its nonthinking comparison. |
| [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M), [350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | 230M: 8 convolution + 6 GQA; 350M: 10 + 6 | 32,768; LFM license | Respectively MMLU-Pro 20.25/20.01, GPQA Diamond 25.41/30.64, IFEval 71.71/76.96. Good instruction formatting is not general decision intelligence. |
| [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1.17B; 10 convolution + 6 GQA | 32,768; LFM license | MMLU-Pro 44.35, GPQA 38.89, IFEval 86.23; IFEval averages strict/loose prompt/instruction accuracies, not directly comparable to strict-only results. |
| [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | 2.69B; 22 convolution + 8 GQA | 131,072; LFM license | BFCLv4 56.88, IFBench 59.17, ToolSandbox 77.83; reasoning-only post-training. No opened MMLU-Pro/GPQA/decision score. |
| [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | 8.3B total/1.5B active; 18 convolution + 6 GQA | 128,000; LFM license | IFEval 91.84, IFBench 56.47; reasoning-only, so these scores do not establish single-prefill quality. Released May 28, 2026 ([blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)). |
| [Nemotron-H-8B Base](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K), [Reasoning](https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K) | 8B; base config: 24 Mamba-2 + 24 MLP + 4 attention blocks | Base 8,192; reasoning 128K; **NVIDIA internal scientific R&D license** | Base MMLU-Pro 44.01, MMLU 72.77. Older H weights are research-only, unlike newer Nano commercial releases. |
| [Nemotron Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | 9B; Mamba-2/MLP + 4 attention | 128K; NVIDIA Open Model License | Reasoning GPQA 64.0, IFEval instruction-strict 90.3; RULER128K 78.9 reasoning-off; Aug 18, 2025. |
| [Nemotron 3 Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | 3.97B; config pattern 21 Mamba-2 + 17 MLP + 4 attention | 262,144; NVIDIA Nemotron Open Model License | Mar 16, 2026; reasoning-off IFEval prompt 82.8/instruction 88; reasoning-on GPQA 53.2. Compressed from Nano-9B-v2; attractive four-attention design, but no SSDM transfer evidence. |
| [Nemotron 3 Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), **size exception** | 30B total/3.5B active; 23 Mamba-2/MoE units + 6 attention | Up to 1M; NVIDIA Nemotron license | MMLU-Pro 78.3, GPQA 73.0 no tools; not a small-residency replacement for 4B. |
| [Granite 4.0-H](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) | H-Micro 3B dense; H-Tiny 7B/1B active; both 36 Mamba-2 + 4 attention. H-Small 32B/9B active is an exception | 128K; Apache-2.0 | Micro/Tiny MMLU-Pro 43.48/44.94, GPQA 32.15/32.59, strict-instruction IFEval 86.94/84.78. Text classification explicitly supported. |
| Granite Nano [H-350M config](https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/config.json), [H-1B card](https://huggingface.co/ibm-granite/granite-4.0-h-1b) | 350M: 28 Mamba + 4 attention; 1B: 36 + 4 | 32K/128K; Apache-2.0 family | H-350M MMLU-Pro 13.14/IFEval61.27 in Liquid’s comparison, **not IBM replication**; own matched decision results unknown. |
| [Falcon-H1](https://falcon-lm.github.io/blog/falcon-h1/) | 0.5B/1.5B/1.5B-Deep/3B/7B: respectively 36/24/66/32/44 **parallel Mamba-2+attention** blocks, not alternating-layer ratios | Configs: 16K/128K/128K/128K/256K; Falcon-LLM License, not plain Apache | [7B card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct): MMLU-Pro 51.75, GPQA Diamond 56.9, IFEval85.35. Small variants’ matched quality unknown here. |
| [RWKV-7 G1 family](https://huggingface.co/BlinkDL/rwkv7-g1) | 0.1/0.4/1.5/2.9/7.2B; 12/24/24/32/32 recurrent blocks; 13.3B/61 blocks is exception; **zero attention** | Latest opened [G1k7.2B](https://huggingface.co/RWKV/RWKV7-G1k-7.2B-20260930) and [2.9B](https://huggingface.co/RWKV/RWKV7-G1k-2.9B-20260930) trained at25,600; Apache-2.0 | Current G1k matched MMLU-Pro/GPQA/IFEval unknown. Internal 13.3B D19 Index49.49 is not a7.2B result. Base models include instruction/reasoning data but are not safety-aligned assistants. |
| [Mamba-3 checkpoints](https://huggingface.co/api/models?author=state-spaces&search=mamba3&limit=30) | Official SISO187M/443M/893M/1.49B; MIMO187M/444M/894M/1.50B; no attention | [1.5B card](https://huggingface.co/state-spaces/mamba3-mimo-1.5b):24 blocks,2,048 context,100B FineWeb-Edu tokens; Apache-2.0 | [March2026 paper](https://arxiv.org/html/2603.15569v1):1.5B MIMO +1.8pp average downstream accuracy over matched GDN; not MMLU-Pro/Index parity. Public checkpoints exist; old repo README’s checkpoint list omits them. |
| [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear), **size exception** | 48B total/3B active; 3 KDA:1 global MLA | 1M; MIT repository | Reported base comparison MMLU-Pro51.0 at4K, RULER84.3 at128K; not a≤9B checkpoint. |
| [Zamba2](https://huggingface.co/Zyphra/Zamba2-2.7B-instruct) | 1.2/2.7/7B;38/54/81 positions with6/9/13 hybrid positions respectively; shared attention weights do not eliminate position-specific caches | Opened configs4,096; Apache-2.0 | 2.7B IFEval48.02 under its older protocol; matched MMLU-Pro/GPQA/decision metrics unknown. |
| [Jamba2-Mini](https://huggingface.co/ai21labs/AI21-Jamba2-Mini), **size exception** | 52B total / 12B active; [32-layer config](https://huggingface.co/ai21labs/AI21-Jamba2-Mini/resolve/main/config.json): 28 Mamba + 4 attention, MoE every second layer | 256K; Apache-2.0 | Card emphasizes enterprise instruction/grounding; numeric matched MMLU-Pro/GPQA unavailable in opened text. No ≤9B Jamba checkpoint established. |
| [Gemma4 E2B/E4B](https://ai.google.dev/gemma/docs/core/model_card_4), attention comparator only | Effective2.3/4.5B, **5.1/8B with embeddings**;35/42 layers of local512-token/global attention, no SSM | 128K; Apache-2.0 | MMLU-Pro60.0/69.4, GPQA Diamond43.4/58.6. Effective parameters must not be mistaken for weight memory or automatic benchmark price class. |

**[EXT] Configuration provenance for exact mixes/contexts:** Falcon [0.5B](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct/resolve/main/config.json), [1.5B](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct/resolve/main/config.json), [1.5B-Deep](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct/resolve/main/config.json), [3B](https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct/resolve/main/config.json), [7B](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/resolve/main/config.json); Zamba2 [1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B-instruct/resolve/main/config.json), [2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B-instruct/resolve/main/config.json), [7B](https://huggingface.co/Zyphra/Zamba2-7B-instruct/resolve/main/config.json); Nemotron [H8B](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/resolve/main/config.json), [Nano4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/config.json). These configs take precedence over family-wide context marketing.

**[EXT/INFERENCE] Licensing matters:** Liquid’s [actual license](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/raw/main/LICENSE) restricts commercial use beyond a $10M annual-revenue threshold; it is not unrestricted Apache. Falcon uses its own terms. Retain Apache Qwen/RWKV for the main open-stack thesis. Liquid short convolutions are dynamics-based but **not long-memory SSM recurrence**; their GQA layers still carry retrieval.

### 1.2 Runtime support and usable latency evidence

**[EXT] Support is architecture-level evidence, not certification of SSDM’s custom readout, forking, quantization, or LoRA path.** Sources: [vLLM supported architectures](https://docs.vllm.ai/en/latest/models/supported_models/), [SGLang implementation directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models), [llama.cpp implementation directory](https://github.com/ggml-org/llama.cpp/tree/master/src/models), and [FLA](https://github.com/fla-org/flash-linear-attention), inspected October14.

| Family [EXT] | vLLM / SGLang / llama.cpp evidence | Kernels and latency interpretation |
|---|---|---|
| Qwen3.5 | All three documented/native | FLA GDN, newer FlashQLA backend available. H2O derivative raw p50 **29.3ms**, p95 **56.4ms**, measured loopback; adjusted208.6/262.8ms. This is not guaranteed stock4B performance. |
| LFM2/2.5 dense and MoE | All three; cards explicitly document support | 1.2B vendor239tok/s AMD decode; 2.6B113tok/s Ryzen AI Max+395; 8B-A1B146tok/s same Ryzen and18.5K aggregate output tok/s on H100, SGLang0.5.12,1,024input/≤256output, sustained concurrency. None is serial decision p50. |
| Nemotron-H/Nano | vLLM/SGLang native files; llama.cpp nemotron-h and MoE files | Mamba-2 kernels; Nano4B card requires vLLM≥0.15.1 and recommends FP32 SSM cache. Comparable serial decision latency unknown. |
| Granite4-H | All three native architecture files | Mamba-2; no matched serial latency established. |
| Falcon-H1 | All three native; vLLM≥0.9.0 in card | Parallel Mamba/attention. Vendor up-to4×prefill/8×decode comparison concerns34B long context, not7B decision latency. |
| RWKV7 | llama.cpp native; no RWKV row/file found in inspected vLLM/SGLang indices | FLA RWKV7, bundled chunked/TileLang and Albatross. [Official family card](https://huggingface.co/BlinkDL/rwkv7-g1) reports7.2B FP16 RTX5090 b1 prefill11,289tok/s, decode145+tok/s, b960decode10,250+tok/s; shapes/workload differ from decisions. |
| Mamba-3 | Native mamba_ssm and FLA; no explicit Mamba3 architecture found in the three serving indices | Triton/TileLang/CuTe implementations; do not infer Mamba2 serving compatibility. See prefill counterexample below. |
| Kimi Linear | vLLM/SGLang/llama.cpp native | FLA KDA. Up-to6×decode is at1Mcontext, not a short-prefill result. |
| Zamba2 | vLLM native; no explicit SGLang/llama.cpp file found | mamba_ssm+causal-conv1d; card warns slow fallback inflates latency/memory. |
| Jamba2 | vLLM≥0.12 documented; llama.cpp Jamba file; SGLang explicit support not established | Card recommends FP32 SSM state and prefix caching; serial timing unknown. |
| Gemma4 | All three native | Optimized attention comparator; no matched serial timing for E2B/E4B established here. |

**[EXT] Latency cautions:** [DeepInfra’s April 3, 2026 0.8B report](https://deepinfra.com/blog/qwen-3-5-0-8b-via-deepinfra-api-benchmarks) gives 0.37s TTFT at 10K input and 403.5 output tokens/s, but internally inconsistent output-time arithmetic; treat it as vendor evidence, not an SSDM target. [Mamba-3’s March 17 blog](https://www.together.ai/blog/mamba-3) reports H100, batch 128, three repetitions: at 2,048 tokens, Transformer prefill 1.08s versus Mamba3-SISO 2.02s; at 16,384, 12.17s versus 16.22s. Its large prefill+decode advantage includes **as many generated tokens as input tokens**. SSDM generates none.

## 2. What recurrence actually buys SSDM

**[INFERENCE] Work model.** For fixed width and state dimension, a pure recurrent mixer processes an N-token state in O(N), retains O(1) history memory, and continues a q-token query in O(q). Full-attention mixing adds O(N²) prefill and O(qN) continuation, alongside linear projection/MLP work common to both. A hybrid with eight attention layers still has a quadratic term; 75% GDN does not make its entire forward linear. FlashAttention also avoids materializing the quadratic attention matrix, so attention’s working memory is not automatically quadratic.

**[SIM] Repeated-state scenario:** N=2,048 state tokens, 20 question/option suffixes of 64 tokens. Independent prefills process 42,240 tokens; one prefix plus suffixes processes 3,328, a **12.69×** reduction in tokenwise work. This is principally **prefix reuse**, available to attention engines too. For SSMs the suffix reads bounded state rather than all historical KVs; the distinction grows with N and branch count. A joint option head can avoid suffix work, but must retain quality.

**[SIM] Qwen4B memory example:** eight attention layers, four KV heads, head dimension 256 and BF16 give `8 × 2 × 4 × 256 × 2 = 32KiB/token`: 64MiB at 2K and 512MiB at 16K. Its 24 GDN layers with 32 value heads and 128×128 state require approximately **48MiB FP32 or 24MiB BF16**, plus convolution state/bookkeeping. Retained history is approximately 112MiB at 2K with FP32 GDN, not a tiny vector. A matched 32-attention-layer comparator has 256MiB KV at 2K. Pure RWKV7-7.2B’s `32 × 64 × 64 × 64` FP32 matrix states total 32MiB before shift states. A thousand live forks still consume tens of GiB.

**[INFERENCE] Correct forking comparison:** modern paged attention shares immutable prefix KV blocks by reference/copy-on-write; it need not physically duplicate the full prefix per branch. Conversely a recurrent state is mutable: each simultaneously advancing branch needs isolated writable state or a kernel that reads one immutable snapshot into private workspaces. Do not compare an optimized SSM with a deliberately naive KV-copy implementation. The hybrid needs **both** recurrent snapshots and shared KV blocks, positions and convolution history. The question-independent prefix must be identical; changing instructions before the state invalidates reuse.

**[INFERENCE] Primary serving choice: SSDM’s direct PyTorch/Transformers worker**, with pinned FLA GDN and attention kernels, not a new general-purpose generation server. Keep the model resident, tokenize once, preserve trusted metadata outside untrusted text, and use one prefix pass plus isolated label-equivariant option branches. Reuse proven sym-fork rather than permutation averaging. Project selected answer/option features instead of all 248K vocabulary logits at every token. Use length/option-count buckets and bounded microbatches for throughput; measure benchmark serial p50/p95 separately. Cache keys include exact tokens, model/adapter revision, dtype, policy and tenant. vLLM/SGLang are integration options only if the custom head and snapshot contract fit without another semantic path; their generation support alone is insufficient.

**[EXT] Hybrid caching is real engineering:** September 2026 [vLLM RFC55697](https://github.com/vllm-project/vllm/issues/55697) proposes semantic prefix boundaries and reports a related 1-to-9 classification workload improving 482→248ms TTFT on 35B/L40S. It remains an **open RFC**, not a promised 4B gain. [PR59175](https://github.com/vllm-project/vllm/pull/59175) documents checkpoint bugs dropping cache hits 89→49% in another workload. Pin the implementation and establish isolation/parity before adopting it.

**[MEAS/EXT] Recall loss matters:** SSDM’s [clean conversion](/tmp/ssdm-state/DEAD_ENDS.md) records 196M tokens, $28.07, KL 0.097, hidden cosine 0.951 and perplexity ×1.08–1.09, yet Index **35.82→14.05**, paired −21.77 [−25.87,−16.37], and retrieval **36.9→8.1**. [Gather-and-Aggregate, 2025](https://arxiv.org/abs/2504.18574) independently locates bottlenecks in a few heads; removing one in a pruned Llama drops MMLU 66→25. Constant memory does not guarantee associative recall, key binding, arithmetic or authority tracking.

**[INFERENCE] Guarantees belong in explicit contracts:** preserve exact option reorder/relabel invariance and hard output support by construction; conformal marginal coverage requires exchangeability and disjoint calibration. The recurrent transition’s stability is a different property from residual-stack stability, factual correctness, or injection resistance. The existing RWKV BF16 certificate failed and LTT certified no actionable set under cluster-valid tests ([context](local://ssdm-context.md)); neither becomes true by choosing a new backbone. A protected authority channel can be exact; semantic obedience still needs adversarial evidence.

## 3. JevBench arithmetic: optimize the actual objective

### 3.1 Sensitivity at H2O

**[EXT] Exact frozen [v1.6.1 result](https://benchmarkheaven.com/api/decision-results/jevbench/v1.6.1/json), measured October 4:**
`I=60.038399, C=89.953994, S=92.610845, K=60.298824; Composite=72.523227`.
Calibration 91.598176 belongs to **Sage**, not H2O. H2O’s cost is a carried estimate, $0.021055837/1K decisions, not a GPU invoice or newly measured token total.

**[SIM] With gates inactive,** `H=4/(1/I+1/C+1/S+1/K)` and `∂H/∂x=H²/(4x²)`.

| Scenario [SIM], other axes fixed | Composite | Change |
|---|---:|---:|
| H2O operating point | 72.523 | — |
| Intelligence +5 | 74.247 | +1.724 |
| Raw p50/p95 both halved; Speed 94.317 | 72.781 | +0.258 |
| Zero raw latency; Speed ceiling 96.478 | 73.097 | +0.574 |
| Cost axis +10 | 75.764 | +3.241 |
| Half priced tokens/cost; Cost +9.031 | 75.480 | +2.956 |
| I=40, C=89.954, maximum self-hosted S=96.478 and K=100 | 45.324 | Cannot rescue weak intelligence |

**[SIM] Local derivatives are I: 0.3648, C: 0.1625, S: 0.1533, K: 0.3616.** The complete headline formula is `Composite = H × min(1,I/50)² × min(1,S/50)² × min(1,K/50)²`. Raw latency is adjusted as `a = 2t + 0.15 seconds`; `S = 80 − 10log10(a50) − 10log10(a95)`. Equal raw p50/p95 of 10/30/100/500 ms give S = 95.39/93.56/89.12/78.79. Halving adjusted latency would improve Speed by 6.02, but halving already-fast raw latency does not. Capability is `(I+C)/2`: at C=90, beating 81.7 needs I>73.4. Decision Index likewise requires skill transfer, not a cheaper host.

### 3.2 Price classes, tokens and latency targets

**[EXT] There is no universal parameter-to-price function in jevkit.** [meter.py](/tmp/State-Space-Decision-Model/jevkit/meter.py) requires an explicit tariff/base reference/size estimate. The [pricing addendum](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-PRICING.md) applies a base-market floor; credits, free hosting, quantization and lower FLOPs do not defeat it. The [interpretation](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-PRICING-INTERPRETATION-1.md) permits standard manufacturer launch tariffs from day 1, although local meter validation still implements the simpler 30-day condition.

**[EXT] Frozen September 25 references:** [published rows](https://benchmarkheaven.com/api/decision-results/jevbench/v1.6.1/json) give 0.8B: $0.01/M input; 2B: $0.02; 9B: $0.10. The [DeepInfra disclosure correction](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-PRICING-DEEPINFRA-DISCLOSURE-CORRECTION-1.md) establishes 4B at **$0.03**, not an automatic $0.04. Older carried rows sometimes use $0.04; do not silently recompute H2O’s carried price. October [v1.6.18 notes](https://benchmarkheaven.com/api/jevbench/v1.6.18) retain the $0.03 frozen reference. These are benchmark references, not guaranteed bookable APIs today.

**[SIM] With no generation, mean input T and input price P dollars/M:**
`USD/1K decisions = T×P/1000`; `K = clip(100 − 30log10(T×P), 0, 100)`.

| Base reference [EXT]; calculations [SIM] | P/M input | 300 tokens: $/1K; K | 600 tokens: $/1K; K | 1,000 tokens: $/1K; K | 2,000 tokens: $/1K; K |
|---|---:|---|---|---|---|
| 0.8B | $0.01 | .003; 85.69 | .006; 76.66 | .010; 70.00 | .020; 60.97 |
| 2B | $0.02 | .006; 76.66 | .012; 67.62 | .020; 60.97 | .040; 51.94 |
| 4B | $0.03 | .009; 71.37 | .018; 62.34 | .030; 55.69 | .060; 46.66 |
| 9B | $0.10 | .030; 55.69 | .060; 46.66 | .100; 40.00 | .200; 30.97 |

**[SIM] K≥60 permits ≤2,154/1,077/718/215 tokens respectively**; avoiding the K<50 gate permits ≤4,642/2,321/1,547/464. At 9B, 600 tokens, C=89.954 and S=92.611, even I=100 gives only **65.28 Composite** after the squared Cost gate. At 300 tokens, 9B needs I≈65.43 to match H2O, before any latency loss. Thus 9B is a capability/Index escalation, not the default Composite model. Compression must preserve information; truncation or hiding branch tokens is not a legitimate gain.

**[SIM, not latency forecasts] Engineering targets on one warm GPU**, selected to illustrate operating ranges, not inferred from parameter count:

| Size | Raw p50/p95 target | Adjusted p50/p95 | Resulting Speed |
|---|---|---|---:|
| 0.8B | 8/15 ms | 166/180 ms | 95.25 |
| 2B | 12/25 ms | 174/200 ms | 94.58 |
| 4B | 25/55 ms | 200/260 ms | 92.84 |
| 9B | 50/100 ms | 250/350 ms | 90.58 |

**[INFERENCE] Actual SSDM latency remains unknown** until its exact readout/kernels are measured. Count the full logical input, schema, options and actual branches/passes; prefix caching alone does not reduce benchmark price. Serial benchmark requests do not award multi-question batching savings. Public231 is an older draw, not the board’s 300 open + 1,200 sealed; existing paired CI half-width approximately 8 Composite makes tiny speed wins scientifically unresolvable there ([jevkit](/tmp/State-Space-Decision-Model/jevkit/README.md)).

## 4. Conversion/distillation: what $200 can and cannot buy

| Method [EXT] | Verified primary result | SSDM interpretation [INFERENCE] |
|---|---|---|
| [MOHAWK, Aug 2024](https://arxiv.org/abs/2408.10189) | Matrix alignment → block alignment → end-to-end KD; 3B tokens pure Phi-Mamba, 5B hybrid | Useful staged objective; not a $200 nine-billion full reproduction. |
| [LoLCATs, Oct 2024/ICLR 2025](https://arxiv.org/abs/2410.10254) | Attention-output MSE then LoRA; 0.2% of prior methods’ parameters, 0.4% of their tokens | Cheap correction is possible, not guaranteed recall. Best hybrid results are not pure-SSM parity. |
| [Mamba-in-the-Llama, Aug 2024](https://arxiv.org/abs/2408.15237) | Weight reuse; quarter-attention student matches chat benchmarks; AlpacaEval 29.61, MTBench 7.35 | Keeping 25% attention has evidence—what Qwen3.5 already supplies. |
| [Llamba, Feb 2025](https://arxiv.org/abs/2502.14458) | 1B/3B/8B recurrent models, <0.1% of teacher training data; RADLADS comparison reports 8–12B tokens | Large pretraining savings still mean billions of tokens. |
| [RADLADS, July 2025 v3](https://arxiv.org/html/2505.03005v3) | 350–700M tokens; Qwen2.5 → RAD-RWKV6/7; 7B MMLU 68.2 versus teacher 71.7; publisher’s 72B conversion <$2,000 | Most plausible ≤$200 research recipe, but a different teacher/mixer; recovery does not establish decision transfer. |
| [LAWCAT, Nov 2025](https://arxiv.org/abs/2509.18467) | Causal convolution + normalized GLA; Mistral7B distilled at 1K retains >90% passkey retrieval to 22K | Positive recall evidence, not proof of multi-key distractor-resistant retrieval. |
| [GenDistill, Sep 2026 revision](https://arxiv.org/abs/2603.26556) | Likelihood gap 0.2pp can hide generation gap 20.8pp; Hybrid-KDA retains 86–90% teacher knowledge accuracy | Test SSDM’s actual typed-decision contract, not only KL/perplexity or irrelevant long CoT. |
| [TinyCeNN-LM, Sep 2026](https://arxiv.org/abs/2609.21139) | Qwen0.8B layers 3/7/11 accepted under fidelity + NLL; 200-item accuracy 28.5–32.0 | Copy accept/rollback discipline, not universal-conversion claims. |

**[EXT] Negative controls:** [August 2026 SWA study](https://arxiv.org/abs/2608.28444) reports 2–10× better NIAH/BABILong than many retrofits without training. [Component-imbalance study](https://arxiv.org/abs/2510.05901) finds converted hybrids can bypass their linear branch. Neither invalidates native hybrids; both invalidate casual attribution of quality to recurrence.

**[SIM] Conversion bill:** approximate student forward/backward as 6PD and teacher forward as 2PD: total 8PD FLOPs. At **150–350 realized TFLOP/s**—not measured SSDM throughput—and SF batch $3.78/h, 350–700M tokens cost approximately **$34–157 at 4B or $76–353 at 9B**, before alignment, optimizer memory, checkpoints, evaluation and failures. A 3B-token 9B MOHAWK-scale run is $648–1,512; 12B tokens $2,592–6,048. Freezing weights does not erase teacher execution or activation transport.

**[INFERENCE] Verdict:** ≤$200 can fund one capped partial-conversion experiment, not a credible promise of attention-free teacher parity. Allocate **zero main-line conversion dollars**. Separately funded research needs a new mechanism, layerwise accept/rollback, clean train-only recall probes with distractors/multiple keys, length extrapolation and typed-decision retention. Do not reopen the identical 196M-token recipe.

## 5. Compute economics and credits

### 5.1 Price snapshot

**[EXT] USD/GPU-hour; observed 2026-10-14 unless specified.** Dashes mean no established public quote in opened pages, not free. Hardware variants, host CPU/RAM, storage, egress, taxes and minimum GPU count differ. Spot/preemptible is not synonymous with cheap marketplace on-demand.

| Provider/source [EXT] | H100 | H200 | B200 | A100 80GB | L40S | RTX 4090 / 5090 | Spot or qualification |
|---|---:|---:|---:|---:|---:|---:|---|
| [SF internal rates](/tmp/State-Space-Decision-Model/jobs/common/rates.py), effective Oct 8, verified Oct 9 | 5.04 list; 3.78 batch | — | — | — | — | — | Friday batch 3.3768; not spot. Public pricing redirected to login; no account accessed. |
| [Modal](https://modal.com/pricing) | 3.9492 | 4.5396 | 6.2496 | 2.4984 | 1.9512 | — | GPU seconds × 3,600; CPU/RAM extra. Non-preemptible execution 3× base; region premium 1.15–1.75×. |
| [Runpod](https://www.runpod.io/pricing), updated Oct 8 | PCIe 2.89; SXM 3.99 | 5.29 | 7.99 | 1.79 | 1.09 | .89 / 1.19 | Rendered Pods list; dedicated on-demand. Exact spot quote not established. |
| [Vast public feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json), 22:30:15Z | SXM min 1.734; median 2.934 | 4.001 / 5.000 | 6.000 / 7.314 | SXM .454 / .801 | .467 / .641 | .140 / .534; .374 / .747 | Cells min/median unless noted; live rentable marketplace, not labelled spot. 4090 p10 .401; 5090 p10 .520. |
| [Lambda](https://lambda.ai/pricing) | SXM 3.99 | — | 6.69 | 2.79 | — | — | Per-GPU price, not proof of single-GPU allocation. No public spot quote established. |
| [Prime marketplace](https://app.primeintellect.ai/dashboard/on-demand-gpus), public JS render | Unavailable in 1-GPU view | 6.69 | Unavailable in 1-GPU view | 1.82 | 1.12 | .92 / 1.22 | Dynamic inventory; no GPU spot quote exposed. Search snippets are not bookable offers. |
| [Together](https://www.together.ai/pricing), GPU Clusters section | 3.99 | 5.99 | 8.19 | — | — | — | Published preemptible 1.99 / 2.99 / 4.09 respectively; check minimum allocation. Dedicated-endpoint prices differ. |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-gpus) | Removed Dec 2025 | — | — | 2.50 | 1.80 | — | Demo hosting, not default training procurement; GPU grants differ from cash credits. |

**[INFERENCE] Procurement rule:** compare `hourly_price / completed_useful_tokens_per_hour`, including failed/resumed jobs and setup. An 80GB A100 at approximately $0.80/h beats SF H100 at $3.78/h if it sustains >21% of H100 throughput; a $0.534/h 4090 needs >14.1%, assuming it fits. H200/B200 are poor default purchases for 4B unless measured memory/batching gains offset the premium. A $1.99 preemptible H100 with 20% replay waste costs $2.49 per useful hour before storage/setup. Checkpoint optimizer, data cursor, RNG and best weights, not only the last model.

### 5.2 SSDM work units

**[SIM] “1,000 steps” is not a compute unit:** effective batch 8, packed length 2,048–4,096 and 1,000–3,000 optimizer steps mean 16.384–98.304M tokens. Rank-64 LoRA still backpropagates activations through the frozen backbone; it is not head-only training. Use `tokens = steps × effective_batch × sequence_length`; padding and multiple option passes increase work.

| Work unit [SIM] | Assumed useful throughput | Time and GPU-only cost |
|---|---|---|
| 4B r64, 1K steps, 2K sequence, batch 8 | H100 1–4K tokens/s | 1.14–4.55 h; $4.30–17.20 at SF batch |
| 4B r64, 3K steps, 4K sequence, batch 8 | Same | 6.83–27.31 h; $25.80–103.22 |
| Same endpoints, 4090 | 250–1,000 tokens/s, not measured | 4.55–18.20 h and 27.31–109.23 h; $2.43–9.72 and $14.58–58.31 at Vast median |
| 231-item evaluation, mean 2K input, one pass | Batched 4K tokens/s | 462K tokens; 115.5 s; $0.12 |
| 2,000-item evaluation, same | Same | 4M tokens; 1,000 s; $1.05 |
| Combined 2,231 items | Same | 18.6 min; $1.17 before loading, serialization and forks |
| Frozen-feature cache, 100K records, mean 2K input | 4K tokens/s forward | 200M tokens; 13.9 h; $52.50 H100; 8K tokens/s halves this |

**[MEAS] A project-specific timing anchor is available:** the October 12 [ledger](/tmp/ssdm-state/results/ledger.tsv), lines 180–181, records JEV3-LM at **5.07 seconds/step**, 497/729 steps before timeout, **11.6 minutes pre-training overhead**, and $3.81 spent with no weights exported. Line 186 records the successful 200-step LM-S cell at **$2.63 / 2,272 billed seconds**. **[SIM]** At the former step rate, 729 steps need 61.60 training minutes; adding 11.6 overhead minutes costs approximately **$4.61 at $3.78/h before evaluation**. This is not a measured full-epoch completion and does not specify the same effective batch as the table. **[INFERENCE]** A $20–25 per-seed hold can therefore be reasonable for that particular epoch plus screening/slack; do not equate its 729 steps with an arbitrary 1–3K-step workload.

**[INFERENCE] These are planning envelopes, not receipts.** Reserve approximately 30% overhead and budget serial latency/perturbations separately. Eight perturbations multiply evaluation work; a state with 20 forks is not one 2K pass unless prefix reuse exists. Keep errors/refusals in cost accounting. Profile 100–200 steps on the exact stack before committing the long cell; include that pilot in the cap.

**[SIM/INFERENCE] Cache only what the head uses:** 100K records × 8 option vectors × 2,560 dimensions × 2 bytes ≈ **4.096GB**. All 2,048 token vectors require approximately **1,048.6GB**, and all 32 layers approximately **33.6TB**. Stream shards and train head minibatches from disk. Caching makes CPU head searches cheap, but **any LoRA/backbone update invalidates features**. Existing frozen-Qwen9B head search peaked at 39.24 SELECT ([dead ends](/tmp/ssdm-state/DEAD_ENDS.md)); caching is a mechanism filter, not a presumed route past Clef.

**[INFERENCE] Conditional $200 allocation:** $20 exact-stack pilot/kernel numerics; $60 for two 4B adaptation cells; $30 held-out/perturbation evaluation; $40 second-seed replication; $10 attention-free mechanism pilot; $40 reserve. Cap completion cost before launch: do not start a 27-hour upper-envelope job under a $30 cell cap. Approved spending remains zero. This is an option for operator approval/new credit, not the current $11.67 balance.

### 5.3 Programs actually visible now

| Program [EXT; inspected October 14] | Current availability / conditions | SSDM implication [INFERENCE] |
|---|---|---|
| [Modal academics](https://modal.com/academics) | Up to $10K advertised; interest form visible; award timing not promised. Starter [pricing](https://modal.com/pricing) includes $30/month. | Leading candidate if eligible; not a guaranteed $10K. |
| [NVIDIA academic grants](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/) | Top notice says **not accepting new applications**; legacy CFPs advertise up to 30K H100 hours and faculty eligibility. | Closed now despite apply links below; do not budget it. |
| [Google TRC](https://sites.research.google/trc/faq) | Rolling interest, temporary free TPU quota; other VM/GCS services not free; research sharing required. | CUDA/FLA kernel port is a real cost, not a free H100 substitute. |
| [Google Cloud research](https://edu.google.com/intl/ALL_us/programs/credits/research/) | Faculty/PhD/postdoc and eligible nonprofit institutions in approved countries; application route visible. | Amount/award date unknown; operator eligibility unestablished. |
| [AWS research credits](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/), updated Oct 8 | Rolling; academic faculty/staff/graduate students; 90–120-day review; student max $5K; faculty/staff uncapped; EC2 on-demand/spot. | Not this week’s runway. |
| [Azure academic research](https://www.microsoft.com/en-us/azure-academic-research/) | Research-credit contact route visible; current amount/deadline unknown. | Inquiry option, not funded capacity. |
| [Hugging Face community GPU](https://huggingface.co/docs/hub/spaces-gpus) | Space hardware grants in documented settings; demo-focused. | Public demo needs operator publication approval; not an unrestricted training grant. |
| [Lambda research](https://lambda.ai/research) | Up to $5K cloud credits; application link visible. | Strong alternative to Modal; selection/time unknown. |
| [Prime marketplace](https://app.primeintellect.ai/dashboard/on-demand-gpus) | No open general research-credit program established from official search and this opened marketplace. | Unknown, not a revived historical promotion. |

## 6. Decision and cheap-proxy ladder

**[INFERENCE] Primary: SSDM-Q4 hybrid**, initialized from Qwen3.5-4B with all 24 GDN/eight attention layers retained. Keep SSDM’s symmetry-preserving typed readout, trust boundary and explicit calibration contract. Adapt with small LoRA only when clean data improves held-out transfer. Reasons: operator-compatible, permissively licensed, already integrated, no distillation bill, credible 4B decision-derivative evidence, and a roughly 3.3× lower frozen input price than 9B. Risks: stock quality is not H2O quality; prior SSDM training hurt; attention still limits history memory and global-stability claims. The win must come from learning and representation, not renaming the stack.

**[INFERENCE] Research: SSDM-R7 on RWKV7-G1k-7.2B**, using 2.9B for mechanism work. Reasons: native zero attention, Apache, bounded history, September 30 weights and existing project assets. Risks: no verified G1k decision gain, no guaranteed small-model transfer, failed numerical certificate and prior fused-kernel gate failure. The 13.3B D19 is a historical quality anchor, not the economic default. Mamba3-MIMO1.5B is interesting for classical dynamics, but its 2K context, 100B-token base pretraining and integration burden weaken its immediate competition case. Do not add a third funded lane.

| Rung [INFERENCE] | What can run | What it establishes—and cannot |
|---|---|---|
| 12-core CPU, 14GB RAM | Streaming data/overlap audits, arithmetic, symmetry/calibration logic, cached heads; existing 0.8/2B quantized inference; 4B Q4 and 7.2B/9B Q4 short-context inference if the full envelope fits | Mechanism correctness/gross recall failures, not H100 latency or 4B capability. Nominal BF16 weights: 1.6/4/8/18GB for 0.8/2/4/9B before overhead; 9B BF16 cannot fit. Available memory during research was only approximately 1.3GiB: host labels are not free-memory measurements. |
| 24GB consumer GPU | 4B BF16 inference and checkpointed LoRA at small microbatch; QLoRA if needed; 7.2B RWKV inference; quantized 9B inference/QLoRA with care | Exact-model proxies are possible, but r64 activations, forks and 4K sequences can still OOM. Measure kernel numerics/throughput. RTX5090 has 32GB, not 24GB. |
| One H100 80GB or A100 80GB | 4B r64 batching/long contexts, paired decision evaluation, FP32 recurrent-state parity; memory-heavy 9B teacher/student experiments | Final same-stack timing and large activations—not routine CPU analysis. H100 is not mathematically necessary when A100/48GB capacity fits; optimize completed-cell cost. |

**[INFERENCE] Transfer validation:** freeze clean TRAIN/SELECT splits by document/source case and unseen generator/domain; public231 and Index evaluation items stay untouched for selection. Retrospectively rank existing matched 4B cells using the proposed proxy. Historical transfer correlations +0.15, −0.32 and −0.50 already reject blind trust in synthetic accuracy ([context](local://ssdm-context.md)). For new techniques compare identical examples/objectives across 0.8/2/4B, then require positive paired 4B validation effects without important recall/safety regressions. An explicit proposed gate is ≥80% sign agreement over at least ten predeclared comparisons, reporting ranking uncertainty. Failure makes the proxy a mechanism falsifier only, not a capability selector.

**[INFERENCE] Numerical transfer is separate:** compare reference CPU, consumer GPU and H100 with identical tokens, revision, dtype, branch boundaries and checkpoint. Preserve existing argmax/probability/Index gates rather than loosening them: FLA inference achieved 98.74% agreement and 1.89× speedup but failed its predeclared gates ([dead ends](/tmp/ssdm-state/DEAD_ENDS.md)). Require branch isolation and option equivariance before timing. Quantization must earn retention; it does not inherit BF16 calibration or certificates.

**[INFERENCE] Bottom line:** retain the small hybrid where benchmark price and recall coexist. Make SSDM’s own readout, trust control, explicit guarantees and transferable post-training the custom stack. Keep one bounded native attention-free research line. Do not sacrifice retrieval for architectural purity or rent expensive GPUs to optimize an objective already near its assumed latency floor.

# 01 — External-claims fact-check of the v1 plan (Doc A) and v2 critique (Doc B)

Date of check: 2026-10-05. Checker: adversarial fact-check pass; every verdict below is against a primary source where one exists (blog posts, HF model cards and raw files, the Decision Index JSON data files, arXiv HTML/PDF, vendor pricing pages, RunPod's live catalog API). Secondary summaries were used only where no primary was reachable and are labelled as such.

Verdict legend: **VERIFIED** (exact quote + URL) · **PARTLY** (what differs) · **UNVERIFIABLE** (what was tried) · **FALSE** (correct value + URL).

---

## 0. Headline findings, most decision-relevant first

1. **Clef's "attention-free vs. transformer" framing is wrong at the root.** Both Clef backbones are *hybrids that are 75 % linear-attention*: Qwen3.5-9B config `layer_types` = 24 × `linear_attention` (Gated DeltaNet) + 8 × `full_attention` (`full_attention_interval: 4`); Qwen3.8-27B = 48 linear + 16 full over 64 layers. The model cards describe the layout as "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))" and "16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))". The configs even carry `"mamba_ssm_dtype": "float32"`. Every top-15 entrant on the official Decision Index board except the Gemma-4 ones runs on these same hybrids. Kev's README documents the practical consequence: "For hybrid bases, each question runs as its own row" because the per-question attention mask cannot be applied inside linear-attention layers. A pure-SSM decision model is therefore *not* a leap into the unknown; it is removing the remaining 25 % attention layers from what already wins. Sources: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json , https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json , https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md

2. **The Decision Index facts in both documents are stale or mixed across editions.** From the primary data files (`data/index.json`, `data/index-v2.json`, `data/index-v0.1.json`, `data/methodology.json` in the HF Space `multimodalart/jev-decision-index`):
   - 0.1: 132,422 requests, 37 benchmarks in suite, 19-benchmark panel. (This is where the plan's "132,422-decision suite" comes from.)
   - 0.2: 121,057 requests, 40-benchmark panel, "five equal-weight areas".
   - **0.2.1 (live, generated 2026-09-28): 120,340 requests (119,898 scoreable), 43 benchmarks in suite, 38-benchmark panel, area weights knowledge 25.8 %, language 25.8 %, retrieval 20.0 %, tools 18.3 %, arts 10 % (√-size weighting), gold-★ benchmarks weigh 1.2.** Hardware "1 x NVIDIA RTX PRO 6000". Jev version jev-1.13.0.
   - "Zero-shot" is **not** a rule. The hard rules are: "No truncation", "No label filtering", "No prompt tuning", "No labels in the adapter", "No correctness-based retry", "Native abstentions stand", "Unanswered = wrong". Author-disclosed training on *test* items is counted as wrong (Eikos-27B, JevK5, open-jev), and one entrant (reflex 4B) was removed for leakage. Training on public *train/dev* splits (BANKING77, CLINC, MMLU-Pro train, etc.) is what nearly every entrant does. Doc B's "v1 fine-tuned on BANKING77/CLINC150 so gates measured a different task" is therefore only half right: fine-tuning on those train splits is allowed and normal; the point is that the index is 38 benchmarks and the other 36 need to work too.

3. **Current official board (0.2.1, 70 entrants + Jev; Clef not yet on it).** Jev 57.91 (ECE 0.074, 524 ms HTTP). Top open entries: Surogate Rune 26B-A4B v3 57.44 (Gemma-4-26B-A4B MoE, full FT, ECE 0.120, 120.5 ms); **Decider chat · Gemma-4-31B 57.33 — a *stock* Gemma-4-31B-it with an option-count temperature readout (ECE 0.047, 108.5 ms)**; pplx-decider-v1-27b 56.40 (engine `autojev-27b`, Qwen3.8-27B full FT, ECE 0.018); **simple-jev · stock Qwen3.8-27B 55.74**; Jebadiah 27B LoRA 54.67 (ECE 0.014, best calibration on the board). Clef is on the board only as an open PR (#54, lucataco, independent HF-Job run on RTX PRO 6000: **61.98**, median 115.3 ms, p95 613.1 ms; per area K&R 51.19 / LU 64.28 / R&C 62.95 / T&A 80.63 / Arts 47.88); Cloudflare's own internal run is 61.21. Another open PR (#58, Torchcast Decision 27B, Qwen3.8-27B LoRA) claims **65.00**. Implication: **a strong stock backbone with a good readout already ties Jev; the fine-tuned heads add only a few points. Backbone knowledge dominates the index.**

4. **Small models score near chance on the index — this is now measured, not argued.** Sub-1B official rows: best is JPT-0.8B 19.22 / Decision 1.0 Eos-0.8B 18.41 / Kev-0.8B 14.60; encoders: Lavoir ModernBERT-large 8.69, Laya 6.04, GLiNER2.5-Decide (486M) 11.21, Julia 1 (mmBERT-small 141M) 5.54. Pending PRs: Bekko System One v0 encoders 17M → 7.37, 68M → 12.83, 400M → 21.87; Dinah-0 150M encoder 27.63; Von 1.3 395M encoder 13.74; Tasksource-JEV-Nano 10.32. The plan's Phase-0–6 ladder (10M–300M params from scratch) would land in the 2–25 range against Jev's 57.9.

5. **The latency axis the plan wants to "win" is not where Clef-flash sits.** On the official single-process RTX PRO 6000 protocol: Decider 2B 8.1 ms (28.97), Decider 4B 12.6 ms (40.70), JevK5-4B 22.0 ms (38.81), Hopper (G) 1.2 4B 23.3 ms (40.77), Laya/Julia 5.8 ms (≈6). Clef-flash's 38.8 ms is Cloudflare's own measurement (hardware unstated); Clef's independent board run was 115.3 ms vs Cloudflare's 209.3 ms. Jev's 524 ms is an HTTPS round-trip that the board explicitly marks "Not comparable to the on-card single-process figures."

6. **Calibration target.** Board ECE (32 benchmarks, 1-in-6 sample, n = 72,594, confidence = probability on the chosen option, 10 bins): Jev 0.074 (acc 0.739, conf 0.812 — overconfident). Best on board: Jebadiah 27B 0.014, Xor 0.0149, pplx-decider 0.0178, Decider chat Qwen3.6-27B 0.0209, Decider 35B-A3B 0.0226. "Beat Clef on calibration" needs ECE ≲ 0.015; Clef's own ECE is not published anywhere (not on the card, not in the blog, not in PR #54).

7. **Jev's architecture is undisclosed** (no paper, no parameter count, no weights; "a new model architecture, parallel sampler for maximum efficiency, and ... RLCD"). Treat any claim about what Jev *is* as unknown. Jev is text-only at launch (Clef, pplx-decider, Rune take images; Clef takes video).

8. **Dates:** the TypeSafe post launched 2026-09-15 (multiple secondaries; the page itself now stamps "Sep 28, 2026, 7:32 PM UTC", i.e. it was edited). Cloudflare's post is 2026-10-01 (Michelle Chen). Decision Index 0.2 was published 2026-09-25 and 0.2.1 on 2026-09-27/28. The Kev PLAN.md numbers (0.2) were superseded three days after they were written.

9. **Budget arithmetic.** RunPod live catalog (API, 2026-10-05): H100 SXM $2.69 community / $3.49 secure / $4.79 serverless; H100 PCIe $1.99 / $2.89; H200 SXM $3.59 / $4.59; B200 $5.98 / $6.79; **RTX PRO 6000 (the index's reference card) $1.69 / $2.09**. Lambda on-demand: H100 SXM $3.99 (8×) to $4.29 (1×), H100 PCIe $3.29, B200 $6.69–6.99, GH200 $2.29. "$2–3/H100-hour" holds only for community-cloud SXM or PCIe. $5k ≈ 1,430–1,860 secure H100-SXM hours, or ≈ 2,400 secure RTX PRO 6000 hours (which also reproduces the board's latency protocol exactly).

10. **Attention-free backbones that exist now (pure, no attention layers):** RWKV7-G1k 13.3B / 7.2B / 2.9B / 1.5B (Apache-2.0, released 2026-09-30, official `RWKV/` org — the largest pure attention-free open LM today); Falcon-Mamba-7B and Falcon3-Mamba-7B (Mamba-1, TII licenses, 32K ctx for Falcon3); Codestral-Mamba-7B (Mamba-2, Apache-2.0); xLSTM-7B (NXAI Community License: free under €100M revenue, attribution required); Mamba-2 up to 2.7B; **Mamba-3 official checkpoints exist but only to 1.5B** (`state-spaces/mamba3-{siso,mimo}-{187m,443m,893m,1.5b}`, Apache-2.0, 100B FineWeb-Edu tokens, ctx 2048). Hybrids to flag: Qwen3.5/3.6/3.8 (3:1 GDN:attention), Gemma-4 (25 sliding-window + 5 global attention), RecurrentGemma/Griffin (local attention, window 2048), Zamba2 (shared attention blocks), Nemotron-H (~8 % attention layers), Falcon-H1 (parallel attention + Mamba-2 heads), LFM2 (10 conv + 6 GQA blocks at 1.2B). No SSM/RWKV/xLSTM entrant exists on the board or in the 70 open PRs; the only "recurrent-ish" entrants are two LFM2.5 RLCD fine-tunes scoring 6.76 and 1.38.

---

## 1. Document A — numbered claims

### Claim 1 — TypeSafe launch post
| Item | Verdict | Evidence |
|---|---|---|
| Date "Sep 15 2026" | **PARTLY** | Launch was 2026-09-15 per secondaries ("On Sept 15, 2026 TypeSafe AI ... released Jev", https://postcutoff.com/e/2026-09-15-typesafe-jev-system-one-model/ ). The page itself now shows "Sep 28, 2026, 7:32 PM UTC" — it has been edited since launch. OrcaRouter measured the API from Sep 24; Decision Index 0.2 (Sep 25) already scored Jev. |
| Author Diogo Almeida | **VERIFIED** | Byline on page. |
| "gives up string generation" | **VERIFIED** | "While Jev gives up string generation, it's optimized for structured outputs and *can't* hallucinate." |
| "defined in advance" | **VERIFIED** | "Possible outputs and structure are defined in advance" |
| "The model never makes type errors." | **VERIFIED** | verbatim. |
| plots 0 % hallucination | **VERIFIED** | Chart "Hallucination and Type-safety", Jev marked 0 %. |
| "Our number is not empirical. Schema matching is guaranteed." | **VERIFIED** | Full sentence: "Our number is not empirical. Schema matching is guaranteed, thus we can confidently add 0% into the plots." |
| "193.6x faster, 444.6x cheaper" + "higher end of real world gains" | **VERIFIED** | "193.6x faster, 444.6x cheaper on our home page" ... "these are on the higher end of real world gains." Also: "End-to-end response time is 70ms-500ms for TypeSafe" vs "3 to 329 seconds for frontier models". Pricing: "$0.042/MTok" input, output "FREE (too cheap to meter)". |
Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev

What the plan did not take from the post: architecture is explicitly not disclosed (FAQ "Is Jev just a smaller LLM?" and "Where does our training data come from?" are headers with no answers in the page text); "parallel sampler for maximum efficiency"; cardinality: "Jev supports a cardinality up to 255" and "For the higher cardinality choices, we do a 2 stage-system of scoring independently then making an explicit choice"; the Kahneman/Jevons naming. No benchmark names, no Decision Index, no ECE in the post.

### Claim 2 — Clef model card
| Item | Verdict | Evidence |
|---|---|---|
| Backbones Qwen3.8-27B / Qwen3.5-9B + vision encoder | **VERIFIED** | Card: "Qwen/Qwen3.8-27B with its vision encoder, stored as standard sharded safetensors"; flash: "Qwen/Qwen3.5-9B with vision encoder". |
| Joint schema head quote | **VERIFIED** | Card: "joint schema head: a small transformer head that reads the backbone's final hidden states, routes evidence from the state to each question, and scores all options of all questions jointly"; "one logit per allowed option for each question". |
| Training = label-smoothed CE + Brier + RLCD (ordinal partial credit, whole-record precision, reference penalty) | **PARTLY** (attribution) | These are in the **blog**, not the card. Raw README has no "label", "Brier", "RLCD", "synthetic" strings. Blog: "label-smoothed cross-entropy ... Brier loss to refine probability calibration ... RLCD ... granting partial credit to adjacent ordinal choices, rewarding fully precise record outputs, and applying a reference penalty to prevent distribution shift." |
| Apache-2.0 | **VERIFIED** | "Released under the Apache-2.0 license, following the base model Qwen/Qwen3.8-27B". |
| Synthetic data "permutating field orders, prompts, and schema structures" | **VERIFIED** (blog) | "This training leverages our own internal synthetic datasets permutating field orders, prompts, and schema structures." Training-data **scale is not disclosed anywhere**. |

Missed, and build-relevant (from blog + `joint_schema_model.py` + `joint_head_config.json`):
- "By freezing Qwen3.8-27B for Clef and Qwen3.5-9B for Clef-flash, we jointly optimized the routing head alongside rank-256 low-rank adapters." (so: frozen backbone + rank-256 LoRA + head)
- "During inference, Clef uses Qwen for a prefill-only pass, then scores the valid schema choices in parallel."
- Backbones are 3:1 Gated-DeltaNet:attention hybrids (see headline 1).
- Head config (27B): `{"hidden_size":5120,"width":1024,"routing_layers":2,"layers":4,"heads":16,"feedforward":4096}`; flash identical with `hidden_size:4096`. `joint_head.safetensors` = 256 MB (≈128M params in bf16).
- Sequence order is **state first, schema second**: `prefix → media_ids → state → schema → suffix`; schema rendered as `FIELD {i} / ID / TYPE / INSTRUCTION / ALLOWED OPTIONS: OPTION {j}: id + description`, with `question_span` and `option_spans` recorded. The head's `EvidenceRoutingLayer` (multi-head attention from options to the whole sequence) and 4 `TransformerDecoderLayer`s over all fields make ordering irrelevant for Clef — but not for a causal/recurrent readout, which is exactly Doc B's point.
- Per-option logit = `prior` (prior_scale · ⟨lexical_anchor, normalize(question_vec + global_vec)⟩) + `sigmoid(residual_gate)` · joint (residual scorer over [field ⊙ option, |field − option|] + cosine); learnable `prior_logit_scale`, `joint_logit_scale` clamped ≤ log(100). Options are embedded both lexically and contextually from their in-context spans — i.e. the backbone *reads the options in context*; this is not the plan's forked-continuation design.
- Types: `noul` → two options with default true/false criteria; `choice` → sorted criteria keys; `score` → enumerated ordinal criteria; `type_embedding(3)`.
- Media: Qwen `<|vision_start|><|image_pad|><|vision_end|>` / video placeholders; Workers AI page: up to 4 images, 1–64 questions per request, context 65,536.
- Blog: "Across the 43 eval benchmarks that we ran, our Clef models beat the decision models on latency"; "Clef is currently the leader when evaluated against the Jev Decision Index" — true against Cloudflare's own run and PR #54, not yet on the official board.
Sources: https://huggingface.co/Cloudflare/clef , https://huggingface.co/Cloudflare/clef/raw/main/joint_schema_model.py , https://huggingface.co/Cloudflare/clef/raw/main/joint_head_config.json , https://huggingface.co/Cloudflare/clef-flash/raw/main/joint_head_config.json , https://blog.cloudflare.com/clef-decision-models/ , https://developers.cloudflare.com/workers-ai/models/clef-flash/

### Claim 3 — Laya / Kev / AutoJev / pplx-decider / LFM2
| Item | Verdict | Evidence |
|---|---|---|
| Laya architecture string | **VERIFIED** | Card: "ModernBERT-large (395M, bidirectional, fully fine-tuned) + a decision head trained from scratch: 2 transformer layers, an option-marker scorer, and an act/escalate head. 421M total." English checkpoint ctx 512 (192 for options); multilingual 1,024. https://huggingface.co/convaiinnovations/laya |
| Kev = Qwen fine-tunes | **VERIFIED** | Kev-0.8B/4B/9B = Qwen3.5 bases + rank-16 LoRA + pointer head; Kev-27B = full FT of Qwen3.8-27B. Pointer head "scores each option's `</opt>` hidden state against the question's `<decide>` hidden state." https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md |
| AutoJev / pplx-decider | **VERIFIED, and they are one board entry** | `denis-pplx/autojev-27b`: "multimodal decision model based on Qwen3.8-27B", "full-weight SFT", "73,000 unique training examples", one H200, Apache-2.0 weights. Space commit "0.2.1: AutoJev-27B is now pplx-decider-v1-27b"; board row `pplx-decider-v1-27b`, engine `autojev-27b`, base `Qwen/Qwen3.8-27B`, 56.40, ECE 0.0178. Perplexity's card: fine-tuned from Qwen3.8-27B, text-classification head, images supported, Apache-2.0, 85.71 % on its own 11-benchmark panel vs Jev 84.51 %. |
| Gemma / ModernBERT / LFM fine-tunes exist | **VERIFIED** | Gemma-4-26B-A4B (Rune, JEV-Gemma4), Gemma-4-12B (Winnow, Jev-Omni), diffusiongemma; ModernBERT-large (Laya, Lavoir); LFM2.5-2.6B-RLCD (6.76) and LFM2.5-350M-RLCD (1.38) on board. |
| LFM2 = hybrid conv + attention with GQA | **VERIFIED** | LFM2-1.2B card: "Hybrid model with multiplicative gates and short convolutions: 10 double-gated short-range convolution blocks and 6 grouped query attention (GQA) blocks." |
| No pure-SSM entrant | **VERIFIED (negative)** | Zero matches for Mamba/RWKV/xLSTM/"state space"/SSM in `index.json` + `methodology.json`; none in PR pages 1–3 (70 PRs). Caveat: see headline 1 — the winning backbones are 75 % Gated DeltaNet. |

### Claim 4 — Kev PLAN.md scores
**VERIFIED verbatim, but superseded.** PLAN.md (dated 2026-09-30): "**Decision Index 0.2** (`multimodalart/jev-decision-index`, 40 benchmarks, chance-corrected): Jev 51.67 (ECE 0.065), AutoJev-27B 50.94 (ECE 0.023), Kev-9B 35.41 (ECE 0.16), Kev-4B 31.31 (ECE 0.20)." It adds: "The Kev entries are old checkpoints (the 4B entry predates round 10); Kev-27B is not on it." Current 0.2.1 board: Jev 57.91 / 0.074; pplx-decider (=AutoJev) 56.40 / 0.0178; Kev 9B 38.48 / 0.1378; Kev 4B 34.64 / 0.1756; Kev 0.8B 14.60 / 0.0743. Kev 1.0 blog (2026-10-01, author-run): Kev-27B 52.3 (ECE 0.019 pooled), Kev-9B 41.0, Kev-4B 38.0, Kev-0.8B 23.3. Sources: https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md , https://jaredpalmer.com/blog/introducing-kev

### Claim 5 — control-theory papers
| Item | Verdict | Evidence |
|---|---|---|
| CompreSSM, ICLR 2026, LRU, selective extension | **VERIFIED** | arXiv 2510.02823 (Chahine, Nazari, Rus, Rusch), "Published as a conference paper at ICLR 2026". Exact: "In the selective case, we propose averaging the dynamics over the input space before applying the reduction scheme." Also: "Future work will focus on extending CompreSSM to linear attention models, e.g., Gated Linear Attention, Mamba2, and Gated DeltaNet." https://arxiv.org/html/2510.02823 |
| HSVR, Schwerdtner/Berman/Peherstorfer, NeurIPS 2025, "up to 10x" | **VERIFIED** | arXiv 2510.22951: "the regularized state space layers are up to 10× more compressible than standard state space layers while maintaining high accuracy." Accepted NeurIPS 2025. |
| Halloran/Gulati/Roysdon 2406.00209 quote; covers Mamba-2 | **VERIFIED** | "the maximal Lyapunov exponent for an arbitrary MambaBlock is first proven to be non-positive"; "We thus synonymously use the term MambaBlock to refer to the SSM layer of both Mamba and Mamba-2 models." v3 Aug 2025, TMLR. https://arxiv.org/html/2406.00209 |
| Zubić & Scaramuzza 2505.11602 "parametric LMI conditions and kernel constraints that limit gating" | **PARTLY** (paraphrase) | Title: "Regularity and Stability Properties of Selective SSMs with Discontinuous Gating" (v3 Jul 2026). Abstract: "a parametric LMI together with universal kernel constraints" for quadratic storage; "sufficient conditions for global ISS uniformly over admissible selection schedules"; "a sampled block LMI for the Mamba selective-scan core" used as a regularizer that "reduces sampled Mamba-core LMI violations by roughly 92%". The constraints are on the kernel/storage, not phrased as "limiting gating mechanisms". |

### Claim 6 — SLT literature
| Item | Verdict | Evidence |
|---|---|---|
| No published LLC study on RNN/S4/S5/LRU/Mamba | **UNVERIFIABLE (negative); consistent with searches** | Searches for "learning coefficient"/LLC × Mamba/SSM/linear recurrent found nothing; Timaeus/Resolution empirical LLC work is on transformers and toy models. |
| RRR exact RLCTs (Aoyagi & Watanabe 2005) | **VERIFIED** | Neural Networks 18:924–933, learning coefficients and multiplicities for reduced rank regression. |
| HMM upper bounds only (Yamazaki & Watanabe 2005) | **VERIFIED** | "Algebraic geometry and stochastic complexity of hidden Markov models", Neurocomputing 69:62–84, upper bound. |
| LDS none; deep linear exact | **UNVERIFIABLE** (not separately checked beyond the above). |

### Claim 7 — Jev API / Clef local run
| Item | Verdict | Evidence |
|---|---|---|
| noul / choice / score; cardinality 255; two-stage | **VERIFIED** | See Claim 1 quotes. Secondaries on TypeSafe docs: Score has 2–10 levels. |
| `systemone` helper takes Jev POST body | **VERIFIED** | README: "systemone takes a Jev/SystemOne POST /v1/systemone request body and returns the same response"; signature `systemone(model, processor, request, max_length=16384)` expecting `"model","state","questions"` + optional `"images"/"videos"`. |
| snapshot_download → load_release_model → encode_record(max_length 16384) → collate_records; torch 2.11 / transformers 5.10.2; one H200; joint_schema_model.py | **VERIFIED** | README: "encode_record accepts max_length (default 16,384 tokens) and max_state_tokens to bound the input"; "Tested with torch 2.11 and transformers 5.10.2 on a single H200". File list: `joint_schema_model.py` 23.3 kB, `joint_head.safetensors` 256 MB, 12 shards ≈ 55 GB. |
| Context 64K (Jev 32K) | **PARTLY** | Clef: blog "our model has a 64k context window (compared to Jev's 32k)"; Workers AI page: 65,536. Jev per TypeSafe docs (via OrcaRouter/aimlapi/secondaries): ~64k total state + all questions, 32k for state + longest question. So "Jev 32K" is the per-question bound, not the total. |

### Claim 8 — benchmark table
**VERIFIED** against the Clef card ("Per-benchmark results from our internal run of the Decision Index 0.2.1 suite."): BANKING77 94.2 / 90.9 / 79.7 / 14.3; CLINC150+OOS 97.4 / 66.8 / 89.3 / 3.2; BFCL 98.5 / 98.8 / 95.8 / 38.1; GPQA Diamond 48.0 / 51.0 / 78.3 / 27.6; median latency 209.3 / 38.8 / 524.1 / 5.8. The plan **dropped two columns** (DiffusionGemma Jev: 96.5 / 74.3 / 83.5 / 44.9 / 84.4 ms; Kev 9B: 94.5 / 84.8 / 79.0 / 38.8 / 51.4 ms) and rows GSM8K 80.8 / 67.3 / 79.9, MMLU-Pro 65.9 / 65.3 / 82.7, p95 238.6 / 122.4 / 536.0 / 222.5. Blog adds ToolRet nDCG@10 69.19 / 66.43 / 65.28, API-Bank 91.93 / 93.11 / 88.19, Home appliances 82.95 / 97.73 / 52.27, When2Call 72.37 / 65.58 / 80.97, BRIGHT 45.91 / 39.26 / 47.52, Amazon ESCI 57.48 / 57.39 / 55.21, PhishNChips 79.60 / 75.05 / 62.55, and workflow evals (invoice 64.7 / 57.1 / 61.8; customer service 76.3 / 77.0 / 76.0; security 62.9 / 61.7 / 61.7; agent trace 68.5 / 69.8 / 71.6). Caveat: the Jev latency is an HTTPS round-trip; the index note says it is "Not comparable to the on-card single-process figures".

### Claim 9 — Decision Index
| Item | Verdict | Evidence |
|---|---|---|
| "frozen 132,422-decision suite" | **PARTLY** | 132,422 = edition **0.1** (`index-v0.1.json`: requests 132422, scoreable 131980, 37 benchmarks, panel 19). 0.2: 121,057; 0.2.1: 120,340 (scoreable 119,898). |
| "edition 0.2 averages 40 benchmarks in the same five equal-weight areas" | **VERIFIED (but not the live edition)** | Space README verbatim: "Decision Index 0.2 averages 40 benchmarks in the same five equal-weight areas (the edition panel `PANEL_V02` ...)". Live 0.2.1: "averages 38 benchmarks in five areas. Arts & Human Taste weighs 10%; the other four share 90% in proportion to the square root of their benchmark count (knowledge 25.8%, language 25.8%, retrieval 20.0%, tools 18.3%). Inside an area, gold ★ benchmarks weigh 1.2 and the rest 1.0". Gold: MMLU-Pro, BBH, GPQA Diamond, HLE, ANLI, WinoGrande, HellaSwag, BANKING77, CLINC150, BRIGHT, BFCL, API-Bank, ForecastBench. SGD and RouterBench are out of the index. |
| chance-correction formula | **VERIFIED** | "(score - chance) / (1 - chance) clipped to 0-1". |
| house rules | **VERIFIED** | "No prompt tuning, truncation, option pruning, retraining, or correctness-based retry." plus "No labels in the adapter", "Native abstentions stand". |
| abstentions score zero | **VERIFIED** | "Unanswered = wrong: Unsupported, refused and errored requests score zero against the full denominator. Nobody climbs by skipping the hard cases." |
| harness github.com/apolinario/decision-index | **VERIFIED** | README; MIT code; submissions via PR to `submissions/README.md`; "runs over 1,000 ms median latency per request are excluded from the board"; reproducible as one HF Job on RTX PRO 6000 (~7 GB download, 17 GB working). |
Latency protocol (methodology.json): 750 timed rows + 10 warm-up, "one process, one request at a time, no concurrency, on exactly one RTX PRO 6000 (96 GB)", "fastest inference path the model or its code supports (CUDA graphs, fused/fla kernels, flash attention, compiled paths, vLLM/SGLang, prefix/KV reuse, batching questions inside a row)". ECE protocol: "32 benchmarks with a per-field right or wrong answer (ToolRet, BRIGHT, RouterBench and ForecastBench excluded), each weighted equally; a 1-in-6 sample of requests; confidence is the probability placed on the chosen option. Rows an entrant trained on count as wrong here too."

### Claim 10 — Mamba-3
| Item | Verdict | Evidence |
|---|---|---|
| "inspired by classical control theory" | **VERIFIED** | Together blog (2026-03-17): "all three of these changes are inspired by the more 'classical' control theory and state space model literature." |
| "100 % accuracy on Parity" | **VERIFIED** | Paper table: Mamba-3 100.00, Mamba-2 0.90, GDN[-1,1] 100.00; "Mamba-3 solves Parity and Modular Arithmetic (without brackets)". |
| "Gu et al." | **PARTLY** | Authors: Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu (Gu is last author). arXiv 2603.15569, 2026-03-16. |
| state-size quote | **VERIFIED** | "Mamba-3 (MIMO) with state size 64 matches the perplexity of Mamba-2 with state size 128, effectively achieving the same language modeling performance with half the latency." |
Checkpoints: official `state-spaces/mamba3-siso-{187m,443m,893m,1.5b}` and `mamba3-mimo-{187m,444m,894m,1.5b}` on HF; mimo-1.5b: "1.50 B parameters", "pretrained on 100B tokens from FineWeb-Edu", apache-2.0, ctx 2,048, "It contains no attention layers". (Third-party blogs claiming Mamba-3 7B/34B/70B are false — no such checkpoints exist.)

### Claim 11 — BabyHGRN
**VERIFIED.** ACL PDF Table 2 (10M-word track, 5 epochs): Transformer (LLaMA) 360M 62.64; LSTM 300M 62.27; **xLSTM 340M 51.20**; Mamba 350M 64.44; HGRN2 360M 67.05. arXiv HTML (2412.15978) gives xLSTM 64.66 in the same table. The ACL version footnotes: "We note considerable training instabilities during pre-training of the xLSTM model". Note the table is the 10M track; the 100M-track BabyHGRN BLiMP is 77.5 vs BabyLlama 73.1. Sources: https://aclanthology.org/2024.conll-babylm.7.pdf , https://arxiv.org/html/2412.15978

### Claim 12 — BarrierSteer
**PARTLY (version drift).** arXiv 2602.20102 (Tran, Verma, Wong, Low, Rus, Xiao). **v1 (2026-02-23) Table 3**: SaP 190.67 ± 3.17; Top-2 25.38; QP 29.40; **LSE 6.08 ± 0.07** (~31×). **v2 (2026-05-21) Table 3**: SaP (100 steps) 190.67 ± 3.17; SaP (1 step) 139.97; Top-2 29.45; QP 33.77; **LSE 2.40 ± 0.06** (~79×). The plan quotes v1.

### Claim 13 — grab-bag
| Item | Verdict | Evidence |
|---|---|---|
| Ctrl-G 100 % CommonGen, O(n·|D|·H²) | **VERIFIED** | Zhang et al., NeurIPS 2024 (arXiv 2406.13892): 100 % constraint satisfaction on CommonGen; complexity "O(nmh²)" with m DFA edges, h HMM states. |
| Conformal factuality (Mohri & Hashimoto ICML 2024) | **VERIFIED** | PMLR v235, mohri24a. |
| LLM-JEPA (Huang, LeCun, Balestriero 2025) | **VERIFIED** | arXiv 2509.14252, Sep 2025. |
| SIGReg / LeJEPA | **VERIFIED** (exists) | LeJEPA paper referenced widely; not independently quoted. |
| Smékal et al. 2024 frequency-domain SSM learning dynamics | **VERIFIED** | arXiv 2407.07279, "Towards a theory of learning dynamics in deep state space models", ICML 2024 NGSM workshop. |
| arXiv 2602.12499 Mamba selection training dynamics | **VERIFIED** | "A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models" (2026-02-13), single-layer selective SSM + 2-layer MLP. |
| Timaeus renamed Resolution | **VERIFIED** | "Timaeus has merged into Resolution" (Resolution Alignment; timaeus.co/resolution.org). |
| stagewise-RL paper Jan 2026 | **VERIFIED** | arXiv 2601.07524, 2026-01-12, Elliott, Urdshals, Quarel, Farrugia-Roberts, Murfet. |
| Wang & Murfet patterning | **VERIFIED** | "Patterning: The Dual of Interpretability", timaeus.co, 2026-01-20. |
| LLC "insensitive to modes below a data-dependent threshold" | **VERIFIED** | Chen et al., "Modes of Sequence Models and Learning Coefficients", arXiv 2504.18048. |
| devinterp library features | **UNVERIFIABLE** as stated (no specific feature claimed). |

### Claim 14 — Neuron / NKI
**VERIFIED.** NKI Library (Neuron SDK 2.32.0) ships `nkilib.experimental.scan.selective_scan` ("Selective scan (SSM) as in Mamba models", fused discretization + recurrence + output projection, fp32 accumulation, Trn2/Trn3); Linear Scan and SSD (Mamba-2 chunked, TensorE matmul + VectorE scans, optional LNC sharding) kernels are listed in the NKI Library docs; `nki.isa.tensor_tensor_scan` is a VectorE instruction. `awslabs/state-space-models-neuron`: "implements the necessary code to train a Mamba-2 model on AWS Trainium instances using the `neuronx-distributed` library" (Mamba-2 7B, Mamba2Hybrid 8B; custom NKI kernels `mamba2_kernel.py`, `conv1d_grouped.py`, `mamba2_kernel_inference.py`). Caveat: tested with torch-neuronx 1.13.1 / neuronx-cc 2.16 — several SDK generations old; 4 commits.

---

## 2. Document B — factual assertions

| Assertion | Verdict | Notes |
|---|---|---|
| "Decision Index is zero-shot across 40 benchmarks" | **PARTLY** | 38 benchmarks (0.2.1). "Zero-shot" is not a rule; the rules forbid prompt tuning/truncation/option pruning/retraining and training on the suite. Train-split fine-tuning is normal (Kev: "10,000 examples from ten public datasets"; ezjev/Torchcast: "train/dev splits of public datasets"). |
| "Clef on 27B backbone" | **VERIFIED** (hybrid; see headline 1). |
| "Index includes knowledge-heavy tasks" | **VERIFIED** | Knowledge & Reasoning is 25.8 % and includes GPQA Diamond, MMLU-Pro, HLE, BBH, GSM8K, MuSR, CRUXEval, ChessBench, CLadder, SATA-Bench. Jev's GPQA 78.3 vs Clef 48.0 is the biggest single gap on the card. |
| "Mamba, RWKV, xLSTM have multi-billion releases" | **VERIFIED** | RWKV7-G1k 13.3B/7.2B/2.9B/1.5B (Apache-2.0, 2026-09-30); xLSTM-7B (NXAI Community License); Falcon-Mamba-7B / Falcon3-Mamba-7B; Codestral-Mamba-7B (Apache-2.0); Mamba-2 2.7B; Mamba-3 ≤ 1.5B. |
| "$2–3 per H100-hour" | **PARTLY** | Only community-cloud SXM ($2.69) or PCIe ($1.99–2.89). Secure SXM $3.49 (RunPod) – $4.29 (Lambda 1×). RTX PRO 6000 secure $2.09. |
| "Clef-flash 38.8 ms median latency" | **VERIFIED as Cloudflare's number** | Not the board's frontier (Decider 2B 8.1 ms, Decider 4B 12.6 ms, JevK5-4B 22.0 ms). |
| "66.8 CLINC+OOS" | **VERIFIED**. |
| "Distillation ceiling" | **weak counter-evidence** | autotrust/JEV-27B (KL-distilled from Jev 1.13, LoRA r=16 + 24-slot head, ≈9.2 B200-hours): six-benchmark mean 84.07 % vs Jev 83.85 %; ECE 0.0009 self-reported; not on the Index. |
| "nobody has shown pretrained attention-free model supports zero-shot schema decisions" | **VERIFIED (as of 2026-10-05)** | No SSM/RWKV/xLSTM entrant on the board or in open PRs. Nearest prior work: SSM rerankers and recurrent embedders (section 3.5). |
| "25M–300M from scratch won't beat it overall" | **VERIFIED empirically** | See headline 4. |

---

## 3. Current state of the field

### 3.1 Official Decision Index 0.2.1 board (primary: `data/index.json`, generated 2026-09-28T00:39Z; score = `scores.balanced_skill`; ECE from `calibration.ece`; latency = single-process RTX PRO 6000 median / p95 ms)

| # | Score | Entrant | Kind | Params | Base | ECE | med | p95 |
|---|---|---|---|---|---|---|---|---|
| – | **57.91** | **Jev 1.13.0** | hosted API | ? | ? | 0.074 | 524.1 | 536.0 (HTTPS) |
| 1 | 57.44 | Surogate Rune 26B-A4B v3 | full FT | 25.8B | google/gemma-4-26B-A4B-it | 0.1196 | 120.5 | 682.8 |
| 2 | 57.33 | Decider chat · Gemma-4-31B | inference technique (stock) | 32.7B | google/gemma-4-31B | 0.0471 | 108.5 | 1114.7 |
| 3 | 56.40 | pplx-decider-v1-27b (= AutoJev-27B) | full FT | 27.8B | Qwen/Qwen3.8-27B | 0.0178 | 101.4 | 1052.5 |
| 4 | 55.74 | simple-jev · Qwen3.8-27B (stock) | inference technique | 27.8B | Qwen/Qwen3.8-27B | 0.113 | 373.0 | 2796.8 |
| 5 | 54.67 | frontier-infra Jebadiah 27B | LoRA | 27.8B | Qwen/Qwen3.8-27B | 0.014 | 110.4 | 591.1 |
| 6 | 53.13 | Eikos-27B-FP8 | LoRA | 27.8B | Qwen/Qwen3.8-27B | 0.0574 | 129.7 | 509.8 |
| 7 | 52.16 | reflex Qwen3.8-27B-FP8 | inference technique | 27.8B | Qwen/Qwen3.8-27B | 0.0242 | 108.3 | 1239.7 |
| 8 | 51.35 | Decider chat · Qwen3.6-27B (stock) | inference technique | 27.8B | Qwen/Qwen3.6-27B | 0.0209 | 83.6 | 965.6 |
| 9 | 50.02 | Winnow-12B | LoRA | 12.0B | google/gemma-4-12B | 0.1679 | 72.5 | 356.2 |
| 10 | 49.47 | JoshuaSP diffgemma | diffusion readout | 25.8B | google/diffusiongemma-26B-A4B-it | 0.2155 | 260.4 | 537.3 |
| 11 | 49.37 | Jevfire Qwen3.8-27B | inference technique | 27.8B | Qwen3.8-27B | 0.0521 | 89.6 | 447.5 |
| 12 | 47.11 | Decider 35B-A3B | full FT | 36.0B | Qwen3.5-35B-A3B-Base | 0.0226 | 101.4 | 210.9 |
| 13 | 46.89 | JPT-9B | LoRA | 9.65B | Qwen3.5-9B-Base | 0.0773 | 148.6 | 461.9 |
| 14 | 43.49 | Decision 1.0 Lux 9B | head/adapter | 9.65B | Qwen3.5-9B-Base | 0.076 | 51.0 | 347.5 |
| 15 | 43.04 | JPT-4B | LoRA | 4.66B | Qwen3.5-4B-Base | 0.0793 | 139.1 | 405.7 |
| 16 | 42.60 | Jet v6.2 | LoRA | 4.66B | Qwen3.5-4B-Base | 0.1411 | 46.8 | 371.1 |
| 17 | 41.48 | Xor Qwen3.6-35B-A3B | full FT | 36.0B | Qwen3.6-35B-A3B | 0.0149 | 139.9 | 419.9 |
| 18 | 40.77 | Hopper (G) 1.2 | LoRA | 4.66B | Qwen3.5-4B-Base | 0.0927 | 23.3 | 238.2 |
| 19 | 40.70 | Decider 4B | full FT | 4.66B | Qwen3.5-4B-Base | 0.0837 | **12.6** | 188.5 |
| 20 | 40.53 | Jev-Omni | LoRA+head | 12.0B | gemma-4-12B | 0.161 | 54.9 | 627.9 |
| 21 | 40.28 | djev diffgemma | diffusion readout | 25.8B | diffusiongemma-26B-A4B-it | 0.2124 | 84.4 | 211.2 |
| 22 | 39.89 | Winnow-E4B | LoRA | 8.0B | gemma-4-E4B | 0.058 | 45.0 | 188.3 |
| 23 | 39.57 | Bespoke Nimble 9B v2 | LoRA | 9.65B | Qwen3.5-9B-Base | 0.024 | 77.3 | 621.4 |
| 24 | 38.81 | JevK5 | LoRA | 4.66B | Qwen3.5-4B-Base | 0.0268 | 22.0 | 226.3 |
| 25 | 38.54 | Interfaze lev | LoRA+head | 4.66B | Qwen3.5-4B-Base | 0.0638 | 70.8 | 687.7 |
| 26 | 38.48 | Kev 9B | LoRA+head | 9.65B | Qwen3.5-9B-Base | 0.1378 | 51.4 | 187.9 |
| 27 | 37.81 | Intern-Decision 4B | full FT | 4.66B | Qwen3.5-4B-Base | 0.0278 | 44.2 | 115.4 |
| 28 | 37.25 | razorback16 diffgemma | diffusion readout | 25.8B | diffusiongemma | 0.232 | 37.7 | 117.8 |
| 29 | 36.75 | NeoHorse-Jev-4B | head/adapter | 4.66B | Qwen3.5-4B-Base | 0.1039 | 49.0 | 273.9 |
| 30 | 36.43 | Solomon v1.1 | head/adapter | 27.8B | Qwen3.8-27B | 0.0807 | 223.9 | 1176.8 |
| 31 | 34.64 | Kev 4B | LoRA+head | 4.66B | Qwen3.5-4B-Base | 0.1756 | 52.1 | 141.1 |
| 32 | 34.36 | Decision 1.0 Nox | head/adapter | 4.66B | Qwen3.5-4B-Base | 0.1399 | 53.2 | 242.1 |
| 33 | 32.35 | Jobe | inference technique | 4.66B | Qwen3.5-4B-Base | 0.1232 | 53.5 | 2910.3 |
| 34 | 32.24 | mmastrac diffgemma | diffusion readout | 25.8B | diffusiongemma | 0.2019 | 126.7 | 274.6 |
| 35 | 29.91 | open-jev (pngwn) | LoRA | 4.66B | Qwen3.5-4B-Base | 0.0589 | 112.3 | 724.1 |
| 36 | 29.24 | Together Tev1 4B | full FT | 4.66B | Qwen3.5-4B-Base | 0.104 | 35.8 | 166.0 |
| 37 | 28.97 | Decider 2B | full FT | 2.27B | Qwen3.5-2B-Base | 0.0771 | **8.1** | 134.9 |
| 38 | 28.42 | openvons Qwen3-4B | inference technique | 4.0B | Qwen3-4B-Instruct-2507 | 0.3712 | 21.2 | 113.3 |
| 39 | 28.14 | FLock this-that 1.2 | full FT | 1.88B | Mapika/decider-2b | 0.1983 | 44.3 | 57.2 |
| 40 | 26.89 | Metask-Jev-4B | LoRA | 4.66B | Qwen3.5-4B-Base | 0.1035 | 57.3 | 369.4 |
| 41 | 25.94 | SemIf | inference technique | 4.66B | Qwen3.5-4B-Base | 0.0989 | 113.1 | 214.6 |
| 42 | 25.32 | Decision 1.0 Sol | head/adapter | 2.27B | Qwen3.5-2B-Base | 0.115 | 38.1 | 101.0 |
| 43 | 20.98 | mini-jev Qwen3-4B | inference technique | 4.0B | Qwen3-4B-Instruct-2507 | 0.3457 | 66.9 | 178.0 |
| 44 | 20.10 | Bosun v3.1 1.7B | LoRA+head | 1.72B | Qwen3-1.7B-Base | 0.1302 | 40.3 | 173.5 |
| 45 | 19.38 | Intern-Decision 2B | full FT | 2.27B | Qwen3.5-2B-Base | 0.0608 | 33.5 | 58.0 |
| 46 | 19.22 | JPT-0.8B | LoRA | 0.87B | Qwen3.5-0.8B-Base | 0.0688 | 117.5 | 341.2 |
| 47 | 18.41 | Decision 1.0 Eos 0.8B | head/adapter | 0.87B | Qwen3.5-0.8B-Base | 0.0833 | 39.7 | 84.3 |
| 48 | 14.60 | Kev 0.8B | LoRA+head | 0.87B | Qwen3.5-0.8B-Base | 0.0743 | 41.6 | 95.9 |
| 49 | 14.32 | Bosun v3.1 0.6B | LoRA+head | 0.60B | Qwen3-0.6B-Base | 0.1419 | 38.7 | 107.3 |
| 50 | 12.85 | Together Tev1 0.8B | full FT | 0.87B | Qwen3.5-0.8B-Base | 0.1257 | 28.5 | 133.9 |
| 51 | 11.94 | Intern-Decision 0.8B | full FT | 0.87B | Qwen3.5-0.8B-Base | 0.0255 | 33.7 | 39.2 |
| 52 | 11.69 | MoJev 0.8B | head/adapter | 0.87B | Qwen3.5-0.8B-Base | 0.1246 | 46.4 | 55.1 |
| 53 | 11.21 | GLiNER2.5-Decide | full FT, encoder | 486M | fastino/gliner2-large-v1 | 0.0884 | 23.3 | 228.6 |
| 54 | 8.69 | MoganAI Lavoir | full FT, encoder | 396M | answerdotai/ModernBERT-large | 0.1492 | 20.0 | 49.5 |
| 55 | 8.04 | jeff GLiFormer-large | encoder | 576M | knowledgator/gliformer-large-v1 | 0.0969 | 21.8 | 67.6 |
| 56 | 7.40 | CLM-v0.1-8B | full FT | 8.19B | Qwen3-8B-Base | 0.3227 | 46.8 | 130.9 |
| 57 | 6.76 | GLiNER 2.5 base | encoder | 194M | – | 0.367 | 14.1 | 62.1 |
| 58 | 6.76 | LFM2.5-2.6B-RLCD | full FT (conv+GQA hybrid) | 2.70B | LiquidAI/LFM2.5-2.6B-Base | 0.2545 | 39.3 | 95.4 |
| 59 | 6.52 | Decision 1.0 Kai | encoder | 308M | jhu-clsp/mmBERT-base | 0.1851 | 30.4 | 106.8 |
| 60 | 6.04 | Laya | full FT, encoder | 421M | ModernBERT-large | 0.1402 | **5.8** | 222.5 |
| 61 | 5.54 | Supersonic Labs Julia 1 | encoder | 141M | jhu-clsp/mmBERT-small | 0.4197 | **5.8** | 15.3 |
| 62 | 5.07 | system-one-gemma | LoRA+head | 268M | google/gemma-3-270m | 0.2387 | 31.9 | 748.0 |
| 63 | 4.54 | Decision 1.0 Lex | encoder | 308M | mmBERT-base | 0.1688 | 29.8 | 106.0 |
| 64 | 4.26 | GLiNER 2.5 multilingual | encoder | 287M | – | 0.2538 | 14.0 | 92.4 |
| 65 | 3.82 | GLiNER 2.5 small | encoder | 74M | – | 0.1608 | 13.6 | 35.2 |
| 66 | 3.78 | Qwen-2.5-1B-RLCD | inference technique | 1.54B | Qwen2.5-1.5B | 0.2052 | 43.8 | 75.2 |
| 67 | 2.97 | Lumma-Fev-0.6B | LoRA | 649M | FrontiersMind/Lumma-0.6B-Base | 0.2357 | 21.3 | 219.7 |
| 68 | 1.87 | Verdict | encoder | 151M | knowledgator/gliclass-modern-base-v2.0 | 0.154 | 11.4 | 60.1 |
| 69 | 1.78 | Lumma-Fev-0.1B | full FT | 153M | FrontiersMind/Nandi-Mini-150M | 0.1574 | 20.9 | 112.5 |
| 70 | 1.38 | LFM2.5-350M-RLCD | full FT | 354M | LiquidAI LFM2.5-350M | 0.5681 | 26.7 | 174.3 |

Jev calibration detail (board): acc 0.739, conf 0.812, ECE 0.074, Brier 0.356, 53.9 % of decisions in the ≥0.9-confidence bin at 0.98 conf / acc shown truncated. OrcaRouter's independent OOD audit: ECE 0.107 on synthetic material vs 0.024–0.032 on public benchmarks; API p50 151 ms, p95 247 ms, 0.49 % error rate over 76.2M tokens.

### 3.2 Pending submissions (open PRs on apolinario/decision-index as of 2026-10-06)
Clef 61.98 (#54, independent HF-Job run; Cloudflare internal 61.21); Torchcast Decision 27B **65.00** (#58; LoRA on StartLux-Decision-27B, Qwen3.8-27B; "Gated DeltaNet layers" via flash-linear-attention; 33 ms/yes-no on H100 HTTP); Darwin-27B-JEV 61.17 (#15); Matilda Jev 59.59 (#42); JEV-Gemma4-26B-A4B 58.05 (#33); Blink v0.3 26B-A4B NVFP4 57.48 (#47); Bespoke-Nimble-9B-v3 56.88 (#36); Xor 26B-A4B 56.08 (#40); **Wald-Q4B v1.1 54.59 (#30, 4B)**; JEV-27B 53.30 / JEV-9B 43.14 (#24); AJev lora5 Gemma-4-12B 52.22 (#53); **ezjev-4b-s2 51.15 (#55, Qwen3.5-4B + merged LoRA, 35.5 ms on RTX PRO 6000)**; Hopper 12B 47.52 (#63); Scion-1 9B 45.93; jiwo-4b 44.97 / jiwo-0.8b 28.00; Lev1-3.7B 42.28; kas-4b 40.1; RSI-Jev v5.0-VL 3B 38.38; RYOTIDE 36.63; Jev-Style-2B 34.83; jevstral-8b 34.29; EXAONE-4.0-1.2B-JEV 30.29; DecisionTune 1.0 29.57; **Dinah-0 150M encoder 27.63 (#29)**; **DM-JEPA 1.1 23.16 (#64, "Non-autoregressive Joint Embedding Predictive Architecture (JEPA), System 1 decision engine", 34.3 ms, max_state 16,384)**; OneJev-0.8B 22.03; Bekko System One v0 encoders 17M 7.37 / 68M 12.83 / 400M 21.87 (#56, "fine-tuned from Ettin rerankers"); Sieve-2B 21.74; Von 1.3 395M encoder 13.74; Tasksource-JEV-Nano 10.32. None uses a Mamba/RWKV/xLSTM backbone.

### 3.3 Vendor self-reports not on the official board
GLiDE (Fastino, closed API, "makes a fast initial assessment, then allocates additional reasoning when the choice is uncertain"): 64.81, "leads in all five evaluation areas", released 2026-09-30. Liquid d1 (closed API, backbone undisclosed, text-only free tier; $0.04/M): 58.9. Drex 1.5 (nace.ai, closed, "under 10B", 128K ctx, 0.65 s median at 8–32K): 58.28. Kev-27B (author-run): 52.3, ECE 0.019. autotrust/JEV-27B (open, distilled from Jev): not submitted. GLiNER2.5-Decide (340M encoder, Fastino): 60.1 on Fastino's own panel but 11.21 on the official board.

### 3.4 Attention-free pretrained LMs available now
| Model | Params | Pure? | License | Notes |
|---|---|---|---|---|
| RWKV/RWKV7-G1k-13.3B-20260930 | 13.3B (L61 D4096) | yes | Apache-2.0 | largest pure attention-free open LM; also 7.2B, 2.9B, 1.5B same date |
| NX-AI/xLSTM-7b | 7B, 2.3T tokens | yes (mLSTM) | NXAI Community License (Llama-3-style; commercial OK below €100M revenue; "Built with technology from NXAI" attribution) | |
| tiiuae/falcon-mamba-7b | 7B, ~5.5T tokens, Mamba-1 | yes | TII Falcon-Mamba License 2.0 (Apache-based + AUP + attribution) | ctx 8K in training |
| tiiuae/Falcon3-Mamba-7B-Base | 7B (+1.5T tokens), Mamba-1, 64 blocks, d=4096, state 16 | yes | TII Falcon-LLM License 2.0 | ctx 32K |
| mistralai/Mamba-Codestral-7B-v0.1 | 7B, Mamba-2 | yes | Apache-2.0 | code model; 256K retrieval tested |
| state-spaces/mamba2-2.7b / mamba-2.8b | 2.7–2.8B | yes | Apache-2.0 | |
| state-spaces/mamba3-{siso,mimo}-1.5b | 1.5B, 100B tokens | yes | Apache-2.0 | ctx 2048, state 128 |
| **Hybrids (flag)**: Qwen3.5/3.6/3.8 (3 GDN : 1 attention), Gemma-4 (25 SWA + 5 global), RecurrentGemma-2B/9B (Griffin, local attention window 2048), Zamba2 (shared attention blocks), Nemotron-H (~8 % attention), Falcon-H1 (parallel attention+Mamba-2 heads), LFM2 (conv + GQA), Jamba, MiniMax, Kimi Linear, Qwen3-Next. |

### 3.5 SSM / recurrent models for classification, reranking, decisions
- "State Space Models are Strong Text Rerankers" (Yan, Xu, Gupta, Srikumar; RepL4NLP 2025, arXiv 2412.14354): Mamba-1/2 reach reranking quality comparable to same-size transformers (Mamba-2-1.3B 53.6 nDCG@10 over 13 datasets) but are less efficient than flash-attention transformers in training/inference.
- "Linear-Time and Constant-Memory Text Embeddings Based on Recurrent Language Models" (Grantner, Sallinger, Flechl; arXiv 2604.18199, 2026-04-20): Mamba2, RWKV and xLSTM embedders with vertically chunked inference; "competitive performance ... substantially smaller memory footprint".
- Mamba Retriever (arXiv 2408.08066); ZeroMamba (vision zero-shot); Decision Mamba family (offline RL, unrelated to typed decisions).
- Halloran et al. (2406.00209) also report Mamba fine-tuning/ICL stability under mixed precision and PEFT.
- No work found on recurrent/SSM backbones for Jev-style typed-schema decisions; no SSM entrant on any decision benchmark.

### 3.6 Prices (2026-10-05)
Compute: RunPod API — H100 SXM 2.69 / 3.49 / 4.79 (community / secure / serverless $/h); H100 NVL 2.59 / 3.19; H100 PCIe 1.99 / 2.89; H200 SXM 3.59 / 4.59 / 5.93; H200 NVL – / 3.79; B200 5.98 / 6.79 / 8.64; B300 6.94 / 7.89; RTX PRO 6000 Server 1.69 / 2.09 / 3.49; RTX PRO 6000 WK 1.69 / 2.19; L40S 0.79 / 1.09; A100 SXM 80GB 1.39 / 1.59; RTX 5090 0.69 / 0.99; MI300X – / 2.39. Lambda on-demand: H100 SXM 3.99 (8×) / 4.09 (4×) / 4.19 (2×) / 4.29 (1×); H100 PCIe 3.29; B200 6.69–6.99; GH200 2.29; A100 SXM-80 2.79 (8×); no H200 on-demand listed.
Inference APIs (input $/M tokens, output free): Jev 0.042; Clef 0.24 (21,818 neurons/M; neurons $0.011/1k; 10k free/day); Clef-flash 0.09 (8,182 neurons/M); pplx-decider 0.04; Liquid d1 0.04. At 10M tickets/month tokencost.app computes $142 d1 / $150 Jev / $854 Clef.

---

## 4. What the two posts/cards say that the plan missed or misread (build-relevant)

1. **Clef is a head on a frozen hybrid backbone, trained with rank-256 LoRA + head jointly** — not a full post-train. A 128M-param attention head reading 5120-d states is the whole "decision" machinery. The plan's DF-SSM reinvents this head as forked continuations + CRF; Clef reads options in context and routes with attention.
2. **State comes before schema** in Clef's prompt; Kev puts "state and questions ... into one token sequence" with a mask so each question sees the state but not other questions, and caches the state KV. Doc B's "question+options must come before state" is the right conclusion for a causal recurrent readout, and Kev's "hybrid bases run one row per question" shows the cost: no cross-question masking in GDN layers → one forward per question, or question-first prompts.
3. **Multimodality**: Clef takes images and video (Qwen vision tower), 1–64 questions per request, 65,536 ctx; Jev is text-only. Image rows exist in the suite (Rune reports 83.4 % on 1,196 image items).
4. **Latency protocol**: one process, one request at a time, RTX PRO 6000, fastest path allowed (CUDA graphs, fla kernels, prefix cache, in-row question batching); >1,000 ms median excluded. Jev's 524 ms is HTTP; Clef's independent 115 ms vs Cloudflare's 209 ms shows hardware sensitivity. Design the serving path for this protocol from day one.
5. **ECE protocol**: 32 benchmarks, 1-in-6 sample, confidence = p(chosen), 10 bins, trained-on rows count wrong. Jev is overconfident (conf 0.81 vs acc 0.74). The best open calibration is 0.014 (Jebadiah) with a plain LoRA.
6. **Contamination policy**: author-disclosed test overlap is scored as wrong; a leaked adapter was delisted. Train-split use is fine. Keep an auditable dedup log against the frozen suite (`corpus_sha256 b2b56d6f…`).
7. **Stock-model baselines**: Decider-chat on stock Gemma-4-31B (57.33) and simple-jev on stock Qwen3.8-27B (55.74) nearly match Jev. The index measures backbone knowledge first and readout second. Any attention-free entrant will be bounded by its backbone's MMLU-Pro/GPQA-class knowledge; RWKV7-13.3B / Falcon-Mamba-7B / xLSTM-7B are far below Qwen3.8-27B there. Doc B's "backbone first, head second" prioritisation is correct; the 12-point gap between Clef and the best 9B (JPT-9B 46.89) is mostly backbone.
8. **Clef's weakest areas** (PR #54): Knowledge & Reasoning 51.19 and Arts 47.88 vs Tools 80.63. Jev beats Clef by 30 points on GPQA Diamond and 17 on MMLU-Pro. "Beat Clef" on Tools/Retrieval/Language is a backbone-plus-head problem; on Knowledge it is a backbone problem.
9. **Training-data scale** of Clef is not disclosed; Kev-27B used ~146k examples / 337k questions; AutoJev 73k examples on one H200 (286 updates); autotrust JEV-27B ≈640k distilled rows for ≈9.2 B200-hours. These are $1–3k-scale runs — inside the $5k budget for a head/LoRA on a frozen backbone, not for pretraining.
10. **Blog-post numbers are Cloudflare's internal run**; Clef's ECE is published nowhere. The "Clef is currently the leader" sentence is true against its own run and PR #54, but as of the Space's last data build (2026-09-28) the official board top is Jev 57.91.

---

## 5. Sources (primary unless noted)
- https://blog.cloudflare.com/clef-decision-models/ (2026-10-01, Michelle Chen)
- https://typesafe.ai/blog/introducing-system-one-models-and-jev (page stamp 2026-09-28; launch 2026-09-15 per https://postcutoff.com/e/2026-09-15-typesafe-jev-system-one-model/ )
- https://huggingface.co/Cloudflare/clef (+ /raw/main/README.md, /raw/main/joint_schema_model.py, /raw/main/joint_head_config.json, /tree/main); https://huggingface.co/Cloudflare/clef-flash
- https://developers.cloudflare.com/workers-ai/models/clef-flash/ ; https://developers.cloudflare.com/workers-ai/platform/pricing/
- https://huggingface.co/spaces/multimodalart/jev-decision-index (data/index.json, index-v2.json, index-v0.1.json, methodology.json, README.md, commits)
- https://github.com/apolinario/decision-index (README; PRs #54, #55, #56, #58, #64; PR lists pages 1–3)
- https://benchlm.ai/benchmarks/clefdecisionindex021 (secondary aggregation incl. Clef 61.21 / Clef-flash 57.07)
- https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md ; …/PLAN.md ; https://jaredpalmer.com/blog/introducing-kev
- https://huggingface.co/convaiinnovations/laya ; https://huggingface.co/denis-pplx/autojev-27b ; https://huggingface.co/perplexity-ai/pplx-decider-v1-27b ; https://huggingface.co/autotrust/JEV-27B ; https://huggingface.co/surogate/rune-26b-a4b-GGUF ; https://github.com/Mapika/decider
- https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model ; https://www.nace.ai/drex ; https://docs.liquid.ai/lfm/models/decision-models
- https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know (secondary; Jev architecture non-disclosure, OOD calibration audit)
- https://huggingface.co/Qwen/Qwen3.5-9B (+ raw config.json); https://huggingface.co/Qwen/Qwen3.8-27B (+ raw config.json); https://huggingface.co/google/gemma-4-26B-A4B ; https://huggingface.co/LiquidAI/LFM2-1.2B
- arXiv: 2406.00209 (html), 2505.11602, 2603.15569 (html), 2602.20102 (html v1 and v2), 2602.12499, 2510.02823 (html), 2510.22951, 2412.15978 (html) + https://aclanthology.org/2024.conll-babylm.7.pdf , 2407.07279, 2406.13892, 2509.14252, 2601.07524, 2504.18048, 2604.18199, 2412.14354; https://proceedings.mlr.press/v235/mohri24a.html ; https://www.together.ai/blog/mamba-3 ; https://github.com/state-spaces/mamba/issues/860
- https://huggingface.co/api/models?search=mamba3 ; https://huggingface.co/state-spaces/mamba3-mimo-1.5b ; https://huggingface.co/api/models?author=RWKV&search=G1k ; https://huggingface.co/RWKV/RWKV7-G1k-13.3B-20260930 ; https://huggingface.co/NX-AI/xLSTM-7b (+ raw LICENSE); https://huggingface.co/tiiuae/falcon-mamba-7b ; https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Base ; https://docs.mistral.ai/models/model-cards/codestral-mamba-7b-0-1
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/selective-scan.html ; https://github.com/awslabs/state-space-models-neuron
- RunPod live catalog via the Runpod MCP `list-gpu-types` (2026-10-05); https://lambda.ai/service/gpu-cloud
- https://timaeus.co/research/2026-01-20-patterning/ ; https://timaeus.co/research/2026-01-12-stagewise-reinforcement-learning ; https://resolution.org/research/

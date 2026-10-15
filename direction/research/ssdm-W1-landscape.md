# Conclusions
[MEAS] The live open-weight headline is **Capability**, not Composite: Quyet leads at **81.7**; H2O leads the separate Composite view at **72.5**. [J-live, accessed 2026-10-14]
[MEAS] The hosted API headline remains Composite: Sage leads at **74.0**; Liquid hosted d1 is **73.0**. [J-api, accessed 2026-10-14]
[MEAS] Decision Index **0.3.2** now has **166** text entries: d3 is ordinal first at **66.26**, but tied with Linne Oak under the board's tie rule. [DI, 2026-10-14]
[MEAS] Since the end of **2026-10-09**, Jev's Index ordinal rank fell **3→12** and Perplexity **1→6**, with their scores unchanged. [DI-old, DI]
[MEAS] Recent benchmark feeds add **58** Index configurations and **44** JevBench-cohort rows, not that many new companies or proven newly released checkpoints. [DI-old, DI, J-new]
[EXT] The category already has open Apache-licensed fine-tunes and frozen-base FP8 readouts; adopting Quyet is a real fast-follower option, not a novel architecture claim. [Quyet, Decisio, Deck; accessed 2026-10-14]
[EXT] Microsoft reports **5,250** safety requests across **11** benchmarks, but does not publish the named suite, per-test outcomes or calibrated risk guarantee. [MD1, 2026-10-09]
[MEAS] Bounded outputs do not stop injection: DecisionRobust reports Jev target attack success **79.3% (291/367)** on its curated, selection-biased clean-correct pairs. [Robust, v1.2.0; accessed 2026-10-14]
[INFERENCE] SSDM's defensible white space is a useful, auditable **risk/coverage contract under explicit assumptions**, not typed JSON, generic guardrails, or SSM branding. [§6–7]
[INFERENCE] Do not resume a broad leaderboard arms race: incumbents already combine hybrid recurrence, quantization, held-out calibration and production integration. [§5–7]

# Competitive and benchmark landscape — as of 2026-10-14

## 1. Evidence rules and scope

[MEAS] This report used opened public primary pages, published JSON and pinned history; no model was run, no inference endpoint was called, no account was created, and no weights were downloaded. Small local CPU parsing produced the tables. **MEAS** means a published evaluator measurement, not a measurement made here; **EXT** means a vendor/author statement; **INFERENCE** means interpretation or a derived strategic judgment. There are no **SIM** performance claims.

[MEAS] The companion [catalog](local://ssdm-W1-catalog.json), dated **2026-10-14**, contains **121** named systems/configurations: the requested competitors, every newly observed configuration in the compared Index snapshots and recent JevBench feeds, plus the newly requested Decisio leader. It records measurement dates, source URLs, null reasons and deployment contexts. A version, wrapper or calibrator can be a different evaluated system even when the base weights match. The recent-entry register is exhaustive **within these opened feeds**, not a claim that every unannounced model worldwide has been discovered.

[MEAS] Keep three clocks separate: product release, model-card/public-weight availability, and benchmark measurement. A benchmark's older run does not prove weights were then downloadable. Most exact checkpoint release dates are unknown and remain null. The Index snapshot is generated **2026-10-14T09:16:12Z**; its pinned comparison snapshot is generated **2026-10-09T23:53:22Z**. Vendor pages accessed later in the day can contain newer product statements without newer independent scores. [DI, DI-old]

### Which JevBench is canonical?

[EXT] The **Florian Standhartinger repository and Benchmark Heaven** are the operational canonical cost/calibration/speed leaderboard in this market slice: the repository names Benchmark Heaven as its canonical display, and Microsoft, H2O and Strands link/use it. That is evidence of reference usage, not measured market share. Decision Index is a separate, widely used capability/generalization board. [J-repo; MD1; H2O; Strands; accessed 2026-10-14]

| Claim class | Name/site | What it actually measures; date opened |
|---|---|---|
| EXT | [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), [Benchmark Heaven](https://benchmarkheaven.com/jev-models) | Typed decisions with Intelligence, Calibration, Speed, Cost; canonical for the surveyed vendors; 2026-10-14 |
| MEAS | [Decision Index space](https://huggingface.co/spaces/multimodalart/jev-decision-index) | Public plus private same-skill/new-domain capability; different scale; 2026-10-14 |
| EXT | [jevbench.dev](https://jevbench.dev/) | Verified-game benchmark, including Minecraft/StarCraft; not the I/C/S/K board; 2026-10-14 |
| EXT | [jevbench.xyz](https://jevbench.xyz/decision-models) | Command/intent-classification comparison and trained classical baseline; not the sealed typed-decision leaderboard; updated 2026-10-05, opened 2026-10-14 |
| EXT | [stern9/jev-bench](https://github.com/stern9/jev-bench) | Separate Jev-versus-Claude project; no opened evidence that it supplies the market leaderboard; 2026-10-14 |
| EXT | [QUB technical report](https://jevbench.github.io/report/JevBench-technical-report.pdf) | Another JevBench: metamorphic probability-coherence relations, not Benchmark Heaven; October 2026, opened 2026-10-14 |

[MEAS] GitHub `results/` contains older published result files; the opened repository listing did not provide a current frozen scorer bundle matching the newest boards. The board's clean frozen JSON and fuller raw artifact are available separately and have **different hash namespaces**. The `latest` API still carries a frozen **v1.6.1** result cohort and is not the complete merged live roster. Do not mistake a changed presentation headline or a JSON hash for a new model test. [J-repo, J-data, J-raw, J-latest; accessed 2026-10-14]

## 2. Competitor comparisons

### Published capability and four-axis measurements

[MEAS] The next table's I/C/S/K/Composite/Capability columns are the **frozen v1.6.1 artifact**, measured on the dates shown, not a synthetic mixture of later draws. Index cells are **current 0.3.2 full/public**, dated **2026-10-14**. These scales are not interchangeable; rounding is for readability. A dash means **no exact system/version match in that artifact**, not zero, bad performance or no product. Later hosted/fresh-draw scores appear immediately below. [J-data, J-raw, DI]

| System / configuration | I | C | S | K | Composite | Capability | Index full / public | Measurement date |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Jev 1.13.0 | 63.62 | 90.63 | 91.50 | 54.73 | 71.49 | 77.13 | 60.11 / 57.96 | 2026-10-05 |
| Clef | 59.08 | 91.58 | 83.40 | 28.10 | 16.75 | 75.33 | 53.08 / 61.71 | 2026-10-04 |
| Clef-flash | 42.18 | 85.46 | 88.40 | 46.80 | 36.64 | 63.82 | 47.61 / 56.15 | 2026-10-04 |
| H2O-Lightning-4B v1.1 | 60.04 | 89.95 | 92.61 | 60.30 | 72.52 | 75.00 | — / — | 2026-10-04 |
| Sage 1.3.0 | 65.59 | 91.60 | 93.62 | 58.23 | 74.05 | 78.60 | — / — | 2026-10-05 |
| Quyet-1.0-Large | 73.40 | 89.98 | 86.87 | 50.55 | 71.39 | 81.69 | 58.71 / 60.79 | 2026-10-04 |
| Perplexity Decider v1.1 27B | 75.72 | 89.87 | 86.79 | 29.89 | 20.62 | 82.79 | 62.75 / 62.25 | 2026-10-06 |
| Bespoke Nimble-9B-v3 | 56.54 | 77.05 | 89.20 | 46.09 | 53.48 | 66.79 | 54.67 / 57.19 | 2026-10-06 |
| decider-12b v1 | 60.63 | 83.76 | 90.48 | 57.76 | 70.42 | 72.20 | — / — | 2026-10-04 |
| decider-12b v2 | 63.01 | 81.99 | 90.48 | 57.76 | 70.88 | 72.50 | — / — | 2026-10-04 |
| torchcast-decision-12b | 60.54 | 82.93 | 91.75 | 56.36 | 69.91 | 71.74 | — / — | 2026-10-04 |
| Kev-9B v2 | — | — | — | — | — | — | 43.27 / 41.54 | not in frozen artifact |
| Laya English | 1.79 | 63.22 | 73.26 | 84.89 | 0.01 | 32.51 | 4.43 / 6.19 | 2026-10-02 |
| djev | — | — | — | — | — | — | 41.78 / 40.89 | not in frozen artifact |
| DiffusionGemma Jev (diffusion-jev-sglang) | 54.86 | 61.31 | 86.79 | 49.56 | 59.33 | 58.09 | — / — | 2026-10-06 |
| Strands-Decider 2B | — | — | — | — | — | — | 21.34 / 25.41 | not in frozen artifact |
| Surogate Rune 26B-A4B v3 | 56.06 | 90.88 | 86.95 | 49.60 | 65.05 | 73.47 | 57.43 / 58.28 | 2026-10-06 |
| deck-31B | 73.03 | 82.16 | 86.70 | 49.69 | 68.68 | 77.59 | 59.03 / 58.19 | 2026-10-05 |
| Microsoft-Decision-1 | — | — | — | — | — | — | — / — | not in frozen artifact |
| GPT-6 Luna Decisions | — | — | — | — | — | — | — / — | not in frozen artifact |
| Liquid AI d1 (hosted) | — | — | — | — | — | — | — / — | not in frozen artifact |
| Liquid AI d1-3B | — | — | — | — | — | — | 38.90 / 48.99 | not in frozen artifact |
| Liquid AI d1-omni-600M | — | — | — | — | — | — | 9.09 / 17.90 | not in frozen artifact |
| decisio v0.8.0 / Gemma-4-31B | 70.42 | 88.72 | 91.19 | 51.67 | 71.69 | 79.57 | — / — | 2026-10-06 |

[MEAS] Capability is shown even for cost-ineligible configurations; **Perplexity's 82.79 and Clef's 75.33 do not make them capped open-board leaders**. Conversely, Quyet's Composite **71.39** is below H2O's **72.52**, yet its Capability **81.69** is above H2O's **75.00**. A single unqualified “number one” claim is ambiguous. [J-data, dates in table]

| Claim class / later measurement | I / C / S / K | Composite / Capability | Timing context and status |
|---|---:|---:|---|
| MEAS Microsoft-Decision-1, 2026-10-10 | 57.17 / 84.40 / 81.55 / 61.45 | 69.11 / 70.78 | API p50 459 ms, serial speed subset; full native set; new gap reference, not frozen row [J-addenda] |
| MEAS Liquid hosted d1, 2026-10-06 | 62.1 / 86.8 / 89.2 / 62.8 | 73.0 / 74.4 | Full set, no equating; API median about 270 ms on live table [Liquid-profile, J-api] |
| MEAS GPT-6 Luna Decisions, 2026-10-06 | 56.9 / 90.1 / 89.3 / 48.6 | 62.5 / 73.5 | Fresh A5 plus public, equated; API median about 300 ms [OpenAI-profile] |
| MEAS d3, 2026-10-13 | See full precision catalog / fresh artifact | 5.57 / 82.97 | Different draw; cost $0.546/1k, outside default cost cap; not capped capability leader [J-new] |
| MEAS SubQ Snap preview, 2026-10-13 | Native probabilities measured; official K unavailable | null / 81.47 | API p50 466 ms; preview price not qualified for official cost/rank [J-new] |

### Deployment, licenses and probability semantics

[EXT] “Hybrid-SSM” below follows the requested taxonomy, but names the actual mix: Qwen's **Gated DeltaNet recurrent linear attention plus full attention**, not a claim that it is a classical Mamba SSM. The Qwen base cards disclose the mix; decision heads can add attention. Liquid's gated **short-convolution plus attention** is not automatically a formal SSM. Size names are nominal; exact served totals, vision components and active counts often differ or remain unknown. [Qwen4, Qwen27, Liquid-open; accessed 2026-10-14]

| Claim class | System/group | Base and architecture | Weights/license; recipe | Actual hosted input $/M; latency caveat |
|---|---|---|---|---|
| EXT | Jev; Sage | Undisclosed architecture/size | Closed; training recipes undisclosed | Jev 0.042; Sage Developer 0.05 plus paid thinking; measured API timing [J-data, Sage-price] |
| EXT | Clef / Clef-flash | Qwen3.8-27B / Qwen3.5-9B; GDN+attention and joint head | Apache-2.0; partial training disclosure | Current Workers AI 0.24 / 0.038, **not** frozen self-host cost [CF-price, CF-flash-price] |
| EXT | Microsoft-Decision-1 | Post-trained Qwen3.5-9B, hybrid | Closed; partial recipe, no released checkpoint | 0.042; vendor regional p50 85 ms differs from independent API 459 ms [MD1, J-addenda] |
| EXT | H2O-Lightning / Kev-9B | Qwen3.5-4B / 9B-Base; hybrid | Apache-2.0; LoRA and native readout disclosed | Self-host tariff null; short local timing is not HTTP service guarantee [H2O, Kev] |
| EXT | Perplexity v1.1 | Qwen3.8-27B hybrid; full-attention layers made noncausal | Apache-2.0; public readout/partial training | Verified input tariff null here; board base-reference estimate is not price [Pplx] |
| EXT | Nimble-9B-v3; Torchcast-12B | Qwen hybrid / Gemma attention | **CC BY-NC 4.0** decision weights/adapters; commercial restriction | Self-host tariff null; do not infer unrestricted commercial rights [Nimble, Torchcast] |
| EXT | Quyet; decider v1/v2; deck | Gemma attention; LoRA / stock-then-LoRA / stock-FP8 | Apache cards/code; Quyet corpus incomplete, Mapika fine-tune described, deck no training | Self-host tariff null; native truncation/readout contracts differ [Quyet, Mapika, Deck] |
| EXT | Rune | Gemma attention MoE | Apache metadata, gated access; complete card recipe unknown | Self-host tariff null; nominal active count is not exact memory [Rune] |
| EXT | Laya | ModernBERT attention encoder and transformer head | Apache-2.0; encoder decision recipe | CPU latency and short context; no attention-free claim [Laya] |
| EXT | Diffusion Jev / djev | DiffusionGemma attention encoder/decoder, one-step decision read | Open base; runtimes differ; **do not share their scores** | Maisa's historical quantized timing is not current bf16 measurement [Diffusion, djev] |
| EXT | Strands-Decider | Qwen3.5-2B-Base hybrid, pointer head | Apache-2.0 data/code/checkpoint recipe | Vendor local GPU timing; no current frozen JevBench score [Strands-card] |
| EXT | Liquid d1 / open d1 variants | Hosted backbone undisclosed; open LFM convolution+attention variants | Hosted proprietary; open **LFM Open License**, not Apache | Hosted 0.04; open single-question local timings workload-specific [Liquid-hosted, Liquid-open] |
| EXT | OpenAI Decisions | GPT-6 Luna native endpoint; size/architecture undisclosed | Closed | 0.10; typed refusal supported; no released training recipe [OpenAI-guide] |
| EXT | Clef-omni; SubQ Snap | Qwen3-Omni attention MoE / sparse attention+head | Apache-2.0 / closed preview | 0.15 / preview 0.05; launched 2026-10-09 / 2026-10-12 [CF-Oct9, SubQ] |

[EXT] Confidence fields are **not plug-compatible risk estimates**: Jev normalizes the maximum choice probability against uniform chance; Sage SystemOne reports the maximum probability directly and always answers, while its `/decide` surface can return null. Sage warns about option-order sensitivity. H2O's default yes/no floor **0.801** removes its uncertainty region; routing mode disables the floor. Mapika documents that Noul temperature **0.05** worsens ECE/log-loss despite unchanged argmax accuracy. A shared threshold does not mean shared semantics or calibration. [Jev-confidence; Sage-SystemOne; H2O; Mapika; accessed 2026-10-14]

## 3. What changed since 2026-10-09?

### Decision Index: added leaders versus changed measurements

[MEAS] The pinned end-of-day comparison is **0.3.1→0.3.2**. The board added **58** previously absent engines/configurations; some older versions also disappeared. Current leading full scores are **d3 66.26, Linne Oak 65.61, Decider-31B 64.62, Sieve-31B 63.81, Bobcat-1.3 62.83, Perplexity 62.75**. The first pair share tie group; Bobcat/Perplexity/Darwin share another. Ordinal position is not proof of statistically meaningful separation. [DI-old, DI, dated above]

| System | Oct 9 ordinal rank → Oct 14 | Full score Oct 9 → Oct 14 | Explanation |
|---|---:|---:|---|
| Perplexity Decider v1.1 (27B) | 1 → 6 | 62.75 → 62.75 | New higher-scoring rows; score unchanged |
| Jev | 3 → 12 | 60.11 → 60.11 | New higher-scoring rows; score unchanged |
| Quyet-1.0-Large | 7 → 19 | 58.71 → 58.71 | New higher-scoring rows; score unchanged |
| deck31b (Gemma 4 31B) | 5 → 17 | 59.03 → 59.03 | New higher-scoring rows; score unchanged |
| Bespoke Nimble 9B v3 | 18 → 41 | 54.67 → 54.67 | New higher-scoring rows; score unchanged |
| Cloudflare clef | 19 → 43 | 53.08 → 53.08 | New higher-scoring rows; score unchanged |
| Cloudflare clef-flash | 32 → 61 | 47.61 → 47.61 | New higher-scoring rows; score unchanged |
| Kev 9B v2 | 42 → 78 | 43.27 → 43.27 | New higher-scoring rows; score unchanged |
| djev diffgemma | 46 → 81 | 41.78 → 41.78 | New higher-scoring rows; score unchanged |
| Strands Decider 2B | 83 → 127 | 21.75 → 21.34 | 0.3.2 private same-skill coverage correction; not evidence of retraining |
| LiquidAI d1-3B | 52 → 91 | 40.08 → 38.90 | 0.3.2 private same-skill coverage correction; not evidence of retraining |
| Cloudflare clef-omni (30B-A3B) | 26 → 58 | 49.32 → 48.28 | 0.3.2 private same-skill coverage correction; not evidence of retraining |
| Laya | 107 → 156 | 4.43 → 4.43 | New higher-scoring rows; score unchanged |

[MEAS] The maintenance correction restored omitted **GPQA, MMLU-Pro and BBH** private same-skill coverage for **37** models, rather than revealing a training upgrade. Other named models' changed ranks with identical scores are displacement by new entrants. Full/public divergence is substantial: Clef **53.08/61.71**, Clef-flash **47.61/56.15**, Liquid open d1-3B **38.90/48.99**. Selecting by public-only launch figures can buy the wrong generalization profile. [DI, methodology/history; correction 2026-10-12, snapshot 2026-10-14]

### JevBench: presentation revisions are not all benchmark revisions

[MEAS] The live **open** page ranks primarily by **capped Capability**; its Composite view is secondary. The API page still leads with **Composite**. **v1.7.0** split open/API presentation without recomputing scores; **v1.7.8/.10** added the Liquid/OpenAI measurements; **v1.7.11/.13/.14** added measured open rows. After the cutoff, **v1.7.31** inserted MD1 and **v1.7.42** merged fresh-draw rows; these changed the roster/ranks. **v1.7.43–.46** changed charts/latency display only; the last two are dated **2026-10-11**, with no changed scores or ranks. Language/use-case supplement updates did not alter the headline scoring set or original price. [J-api revision history, accessed 2026-10-14]

[MEAS] In the current default open Capability view, **Quyet stays first**, Jeff enters second, MeTask's trained service enters third, Decisio moves **2→4**, deck **3→7**, and H2O **5→9**, while H2O retains open Composite first. The Jeff draw existed around the cutoff but entered the unified view on **2026-10-10**; “first measured” and “first on the merged board” differ. Current API Composite includes MD1 at **6** and attest at **7**; stale individual model profiles show older ranks. [J-data, J-live, J-api revision history; accessed 2026-10-14]

[MEAS] Fresh rows retain draw tags and **zero equating offsets pending anchor runs**. Those are published live positions, not established equal-difficulty rankings. The current open page reports **162** ranked systems; API page reports **30** ranked offerings; the frozen clean feed has **140** rows across statuses/groups. Do not call these contradictory sample sizes or use `latest` alone as the live roster. [J-live, J-api, J-data; accessed 2026-10-14]

### All recent JevBench-feed entries

[MEAS] The following is the full observed new-cohort register, including unranked wrappers and preliminary configurations. The catalog preserves axes, dates and status; none is silently backfilled into frozen v1.6.1. [J-new; dates in table]

| Feed/draw and measurement dates | Newly listed systems/configurations (names preserve the published row; not proof of product release date) |
|---|---|
| [MEAS] Fastlane, Oct 9–10 ([artifact](https://benchmarkheaven.com/api/jevbench/v1.6.4)) | decisor-4b; Jeff-1.0-Large; Metask rain 4B; metask-jev-rain-12B; ryotide_qwen9; Wald 4B |
| [MEAS] First regular draw, Oct 10 ([artifact](https://benchmarkheaven.com/api/jevbench/v1.6.9)) | Liquid AI d1-3B; Liquid AI d1-omni-600M; Blink v0.3 26B-A4B; ClassOne Gemma 4 E4B (Gemma-4-E4B backbone with ClassOne decision heads, schema-constrained single forward pass, self-hosted); ClassOne Qwen 3.5 2B (Qwen3.5-2B backbone with ClassOne decision heads, schema-constrained single forward pass, self-hosted); ClassOne Qwen 3.5 4B (Qwen3.5-4B backbone with ClassOne decision heads, schema-constrained single forward pass, self-hosted); Clef-omni; Gutsy 0.8B v0.3; jiwo 0.8b (Qwen3.5-0.8B backbone with a trained decision head, one forward pass, native probabilities, self-hosted); jiwo 4b (Qwen3.5-4B backbone with a trained decision head, one forward pass, native probabilities, self-hosted); modernbert-jev v2; RSI-Jev v6.1-VL 27B; Standard One 8B SH (Standard Thinking); Vega 0.8B; Vega 4B; Wald 4B v2.1 |
| [MEAS] Second regular draw, Oct 12–14 ([artifact](https://benchmarkheaven.com/api/jevbench/v1.6.18)) | Amos 1 large (Scira); Amos 1 medium (Scira); attest-31B-v1.0; Julia-1 (Supersonic Labs); Krino-2B; metask-jev-rain-12B-trained; metask-jev-rain-31B-trained; metask-jev-rain-31B-trained service v3; Microsoft-Decision-1 (Azure Foundry) — A2 re-run; Bespoke Nimble-v4 (Bespoke Labs); Bespoke Nimble-v4-max (Bespoke Labs); Phocinae-Largha-150M-v1; pjev-2b (Kailune AI, frozen MiniCPM5-2B + merged LoRA r64, letter-logit readout, T=1 stub); PurpleMIST-Mini-1.0 (Olaverse); Reflex Instinct 0.6B (Atlas AI); Respan Span-02; Sage 1.4 (Levanto Labs); Standard One 3B SH (Standard Thinking); SubQ Snap (preview, SubQ); Torchcast Decision 4B (v1.1); Uniphore-4B-Decider; vLLM-SR d3 |

[EXT] **SubQ Snap**, released **2026-10-12**, is strategically notable even though unranked: sparse attention plus a decision head, advertised **384K** text/image context and **$0.05/M** input in capacity-limited preview. Its old Index **0.2.1** claim **61.7** is not a current private-inclusive **0.3** score. The vendor's Decision RULER tests extend to **1M** tokens, above the advertised serving cap; do not imply that capacity is generally available. Linear-scaling long-context marketing is therefore already occupied. [SubQ, 2026-10-12]

## 4. Benchmark methodology and failure modes

### JevBench v1.6.1

[MEAS] The headline set has **300 public plus 1,200 sealed items**. Physical counts are **750 choice, 375 Noul, 375 score**, but the score gives the three types **equal weight**; tiers are easy **0.10**, standard **0.20**, judge **0.30**, hard **0.40**. Public and sealed competence each receive **0.50** weight, so their physical counts do not determine headline weights. Subject families cover written policies/law, software, mathematics, commerce, operations, language and safety; use-case labels span routing, moderation, guardrails, finance, insurance and others. Supplement radars are separate, not extra headline items. [J-method, J-raw, J-api; frozen publication 2026-10-06, opened 2026-10-14]

[MEAS] Choice competence corrects random-option chance. Noul answers in the open uncertainty band between **0.2 and 0.8** count as abstentions/wrong, rewarding decisive answers; score competence uses an ordinal-error baseline. The official O1S amendment switched score's baseline to a constant midpoint and clips negative competence by split/type before aggregation. It was selected after earlier scores were known, so version-locking matters. Calibration is a separate native-probability/outcome axis; operational failures count wrong in Intelligence but are **excluded from Calibration**. A high C score can coexist with refusals or unsupported long inputs. [J-repo, J-method, J-api; amendment noted 2026-10-03; accessed 2026-10-14]

[MEAS] **Capability=(I+C)/2**, ranked within **$0.0645946551724/1,000 decisions** and adjusted median **1.23295664 s**, the fixed twice-Jev historical reference envelope. Composite is the four-axis harmonic mean times squared low-axis gates:

`4/(1/I+1/C+1/S+1/K) × min(1,I/50)^2 × min(1,S/50)^2 × min(1,K/50)^2`.

[MEAS] The Speed transformation uses both quantiles: `S=80−10log10(p50_seconds)−10log10(p95_seconds)`. Cost is `K=100−30log10(USD_per_1000/0.001)`. Self-hosted quantiles receive **2×raw+150 ms**, while hosted timing includes actual network transit from the evaluator. The recorded speed subset has **124** public standard/judge requests; it is not production throughput, every scored request, or hardware-normalized latency. [J-raw, J-api; frozen/live methods opened 2026-10-14]

[MEAS] Cost is not GPU expenditure. Self-hosted rows use the exact **base-model reference token price**, or a declared fallback size class, times the applicable token count. API rows use their own qualified list price. Common token-priced cost excludes **23** items beyond Jev's accepted input range, although refusals still hurt I. Some open costs are carried from prior editions; H2O carries the previous same-base cost. Tariff cuts must be public/bookable under the published pricing rule; preview/free marketing is not automatically accepted. Quantization is not a lower official base-price class. [J-pricing, J-api, J-data; pricing addendum 2026-09-25, accessed 2026-10-14]

[EXT] Submission has a free ordinary queue; paid fastlane is **$49** for APIs/smaller models or **$99** for larger ones with a **48-hour** service target. We purchased nothing. Operators review runnable code/native schema and can measure hosted endpoints on request. Sealed draws rotate; provider exposure is logged before dispatch, a provider never gets scored again on an item it has seen, and exposure to two providers retires an item globally. APIs normally rerun at most every **three** refresh releases unless a verified new version arrives. Fresh smaller API draws are anchor-equated; new merged open draws can still await anchor offsets. [J-submit, J-api; accessed 2026-10-14]

[INFERENCE] Main gaming risks are public-set selection, temperature/decisiveness optimization, optimistic missing-response C, unequal runtime/HW, post-observation scoring changes, base-price rather than measured serving cost, and pending cross-draw equating. A sealed set reduces direct fitting but is not a prompt-injection threat model or a production risk guarantee. These are reasons to qualify claims, not discard all measurements. [J-method; H2O; Mapika; Bobcat; Strands]

### Decision Index 0.3

[MEAS] The public suite combines **37** benchmarks across knowledge, language understanding, retrieval/classification, tools/automation and arts/human taste; area weights are **0.2585/0.2585/0.2002/0.1828/0.1000**, with gold-standard benchmark weighting inside each area. Scores chance-correct and clip to the unit range; wrong/unsupported requests score zero. Public MCQ capability does not directly reward calibrated ordinal probabilities or certified abstention. [DI-method, DI-repo; accessed 2026-10-14]

[MEAS] Full score is **0.20 public + 0.50 private same-skill/fresh-data + 0.30 private new-domain**, after mean/standard-deviation equating based on **21 stock anchors**. Raw private means and scales are different; do not average the displayed raw private numbers. The board's **0.9-point** tie band is a heuristic fresh-sample uncertainty band, not a paired hypothesis test. Private content is withheld; the evaluator can assess domain transfer without releasing attack/data contamination guarantees. [DI-method, DI; 2026-10-14]

[MEAS] Public kit counts are **140,620 requests, 140,178 scoreable after 442 exclusions**; part of the methodology file still contains an older completion count. Reconstructed GSM8K distractors reduce leakage, ForecastBench was retired, WinoGrande was recategorized and leaked/broken RouterBench/SGD handling was corrected. An old vendor **0.2.1** number is not a current full **0.3.2** claim. [DI-method, DI-repo, d3-card; accessed 2026-10-14]

[EXT] Submission requires the full public run uploaded as an HF dataset and a PR with engine/checkpoint/hardware/native-inference details; maintainers rescore and run private tests. Eligibility uses single-GPU native timing, with median/mean/p80 each at most **1,000 ms** on an RTX PRO 6000; truncation/pruning and benchmark-specific wrapper tuning are prohibited. Transport failures can be retried up to **two** times, not incorrect answers. Native abstention stays visible but is wrong for capability scoring. Those rules do not mean a consumer CPU meets that latency or an API supplies an SLA. [DI-repo, submission instructions; accessed 2026-10-14]

### Microsoft, TypeSafe and robustness suites

| Claim class / suite | Public evidence | What cannot be concluded |
|---|---|---|
| EXT MD1 capability suite, 2026-10-09 [MD1] | 36 benchmarks, 147,137 questions; chart accuracy83.5/calibration92.2. Mixed public/private, vendor-controlled; Strands comparison covers only23 sets. | No complete opened task list, executable harness, per-item outputs or common-coverage head-to-head; not Decision Index or JevBench scores. |
| EXT MD1 safety, 2026-10-09 [MD1] | 5,250 harmful/jailbreak/injection requests over11 benchmarks; qualitative pass claim. Eight perturbation families report1.3% answer flips. | Named suite, attack budgets, per-test error/FPR and independent reproduction absent; observed flips are not mathematical invariance. |
| EXT TypeSafe workflows [TS-evals, TS-launch], opened2026-10-14 | Four fixed code graphs: invoice, service, incidents, agent trace; consensus of GPT-6 Astra and Claude Fable5.1 high-thinking reference probabilities; equal workflow weights, provider-default tested reasoning. | Agreement with powerful model consensus is not human-ground-truth correctness or safety; harness assumed correct. Homepage193.6×/444.6× gains are vendor workflow-conditioned, not universal. |
| EXT Cloudflare workflow replication [CF-card], 2026-10-01 | Same cohort/revision; exact-actions invoice Clef64.7/Jev61.8, service Flash77.0/Jev76.0; agent-trace primary-action Jev71.6/Clef68.5. | Metric differs by workflow; small vendor-selected examples do not determine public/private leaderboard ranking. |
| EXT Sage AgentHarm [Sage-safety], opened2026-10-14 | 352 prompts:176 harmful/176 benign;77% harmful prevention,95.5% benign allowed; threshold fitted to same benign subset. | In-sample FPR calibration, agent/judge dependence and model version uncertainty; not certified future-risk coverage. |
| MEAS DecisionRobust [Robust], v1.2.0 opened2026-10-14 | 1,058 candidates/441 originals, five text/image challenge sets; GT, clean/attacked inputs and per-model denominators public. | Selection-biased models marked; repeated originals counted per candidate; Target ASR is not older Flip ASR or population unsafe-action rate. |
| EXT JevAdvBench [Adv], 2026-09-25 | 812 questions/66 scenarios/9,744 one-edit variants with identical-rerun control; state injection and irrelevant-opinion attacks. | Clean-answer stability is not always ground-truth accuracy; an abstention gate does not remove every confident attack. |
| EXT TypedBench [Typed], 2026-10-08 | Policy generators, calibration floors, risk-cost and asymmetric decisions. | Useful targeted evidence; not a guarantee that confidence means correctness across domains. |
| EXT Hidden Risks [Hidden], 2026-10-04 | Privacy, membership/exfiltration, backdoor and dual-use risks in typed models. | Typed schema is not semantic alignment, privacy or harmlessness. |
| EXT Automated decision gates [Gates], 2026-09-29 | Matched decision/generative/classifier analysis; abstention, OOD gates, label-swap failure; trained classical baselines competitive. | Zero-shot leaderboard superiority does not imply superiority over task-supervised classifiers. |
| EXT QUB coherence JevBench [Coherence], October2026 | 50 metamorphic relations in five dimensions; probability/invariance/menu coherence. | Uniform outputs can score high coherence while being useless: demand capability and coverage alongside invariance. |

[MEAS] For one concrete injection comparison, DecisionRobust reports Target ASR **Jev79.3%(291/367, selection-biased), OpenAI28.2%(90/319), H2O37.3%(116/311), Perplexity57.1%(200/350), open d1-3B13.1%(35/268)**. Different clean-correct denominators and native surfaces make this a challenge-set comparison, not a universal robustness ordering. MD1 has no published row in the opened card. [Robust, version1.2.0, opened2026-10-14]

## 5. Fast-follower option: post-train an open leader

[EXT] **Yes: publicly licensed, already-specialized open bases exist.** Quyet's opened card states Apache-2.0, merged LoRA and normal Gemma architecture, including standard Transformers/vLLM loading; its native prompt and temperatures live in the package/configuration. Redistributors must retain NOTICE. Main separately verified ungated availability from **2026-10-10**; the current card establishes availability as accessed **2026-10-14**, not the historical original release date. No Quyet-derived SSDM training or quantization result has been measured in this workstream. [Quyet]

| Claim class / candidate | Published recipe: what actually produced it | Quantization/calibration facts and disclosure limit |
|---|---|---|
| EXT Quyet-1.0-Large [Quyet], accessed2026-10-14 | Gemma4-31B-it, rank16 merged LoRA, letter readout; prompt-v2 removes boilerplate and uses compact JSON. | Temperatures refit for changed prompt; complete training corpus/loss/compute not disclosed. Native state6000/prompt8000, ten options, truncation; no verified Quyet quantized-vs-bf16 paired score here. |
| EXT Decisio31B [Decisio], accessed2026-10-14 | **Frozen stock Gemma**, no post-training; first-answer logits, reusable state prefill, externally fit temperatures. | FP8 on load already deployed; frozen v0.8.0 independent Capability79.57/Composite71.69; current package is not identical to the benchmark version. |
| EXT deck-31B [Deck], accessed2026-10-14 | **Frozen stock Gemma**, no fine-tune; Cygnet-style option readout. | TorchAO FP8 weights/activations; temperature3.4 not fitted on JevBench; bounded native context. Already demonstrates quantization/readout without a new learned decision model. |
| EXT H2O-Lightning [H2O], accessed2026-10-14 | Qwen3.5-4B rank64 LoRA, merged weights; decision-oriented consensus/data handling and native one-output readout. | Separate calibration/shim; floor0.801 default versus unfloored routing mode; do not use default probabilities as an untouched conformal certificate. |
| EXT Bobcat1.3 [Bobcat], updated2026-10-14 | Three rank16 LoRA arms, one epoch, teacher KL+gold+ordinal RPS+base KL, uniform weight average; new human-labeled skill mixture, decontamination. | Released BF16 and FP8 serving; product versus per-type temperatures differ; public Index half used for selection. Full62.83 is independently published; injection still succeeds on document-screening subgroup. |
| EXT Bobcat Flash1.2 [Bobcat-Flash], accessed2026-10-14 | Two rank64 LoRA arms merged with0.6/0.4 weights; code-gold synthetic/state/long-context decisions plus teacher distributions. | FP8 deployed; properly distinguishes product log-loss temperature from JevBench-style Capability temperatures; release criterion changed after measurement. |
| EXT Xor currentv1.1 [Xor], accessed2026-10-14 | Gemma4-26B-A4B merged BF16 checkpoint xn-lora-xl-5050; native two-order candidate scoring. | Full blend provenance **not established**; explicitly do not carry v1.0's recipe into v1.1. Temperatures2.0/1.7/1.0 disclosed; no benchmark-clean-training claim. |

[MEAS] Official Cost uses base-reference prices, not bit width: Decisio's FP8 row is priced at **Gemma4-31B $0.09/M input**, and René31B-FP8 still has **K46.0**. Quantized serving can have a different measured S axis, but receives no automatic lower reference price. Native prefix reuse and shorter prompts change measured input/latency patterns; Quyet's many-question implementation rereads state, whereas Decisio reuses it. These are documented serving differences, not a forecast of SSDM savings. [J-data, J-api, Decisio; measurements2026-10-06/current sources accessed2026-10-14]

[EXT] Post-training is not guaranteed to improve generalization: Mapika v2's public hard-tier result is **79/111**, versus v1 **81/111**; Kev's card discloses failure of large in-distribution gains to transfer; Bobcat1.3's gains over1.2 mostly overlap uncertainty intervals. Quantized near-ties can change branch decisions and calibrated probabilities. This research therefore supplies no factual basis for promising that another LoRA or lower precision preserves Quyet's lead or adds a safety guarantee. [Mapika, Kev, Bobcat; accessed2026-10-14]

## 6. Market evidence and uncovered needs

[EXT] The concrete buyer-facing workloads are repetitive **agent control** (tool/call/stop/verify), moderation, model-tier routing, support/document triage, fraud/policy checks and pre-tool guardrails. Cloudflare describes its own threat-intelligence/Trust-and-Safety operations; Strands exposes tool-intervention hooks; Microsoft places MD1 inside Foundry workflows. Liquid demonstrates vision/audio decisions and device execution. These are product/documentation uses, not audited adoption volumes or willingness-to-pay estimates. [CF-launch, Strands, MD1, Liquid-open; publication dates in sources]

| Claim class / stated need | Primary observation and date | Consequence for a differentiated entrant |
|---|---|---|
| EXT Private/on-device execution | Jev launch HN asks for private offline/home-control use; Strands HN discusses local email filtering, GGUF/custom-head conversion and NPU friction. [HN-Jev, 2026-09-15–16; HN-Strands, opened2026-10-14] | Packaging, licensing, CPU/edge reliability and predictable memory matter, not only GPU microbenchmarks. |
| EXT Semantic reliability, not just schema | Launch HN questions valid-but-wrong outputs and automated household actions; Kev/Strands cards warn about consequential decisions. [HN-Jev; Kev; Strands-card] | Keep authorization/policy authority in deterministic code; a typed wrong action is still wrong. |
| EXT Calibration/abstention under deployment shift | Model cards repeatedly require workload-specific validation; Sage surfaces differ; Mapika shows calibration cost of confidence sharpening. [Sage-SystemOne; Mapika; Azure-doc] | Raw probability provenance and useful risk/coverage are stronger needs than a confidence field. |
| EXT Integration/portability | Mapika's deployment issue discusses ARM/CPU precision and NPU graph difficulties; Strands thread asks about standard conversion. [Mapika-issue, 2026-09-28–10-02; HN-Strands] | A nonstandard custom head or strict recurrence can impose real operational cost. |
| EXT Permission-aware context/auditability | Enterprise context vendor GoSearch explicitly separates decisions from permissions, current source evidence and execution. [GoSearch, 2026-10-14] | Evidence binding, policy versioning and decision replay are workflow needs, not something schema alone supplies. |
| EXT Security/injection | Foundry safety suite, Sage AgentHarm, Strands pre-tool gates, Bobcat injection tests and independent attacks all address this concern. [MD1; Sage-safety; Strands; Bobcat; Robust] | Safety is **not vacant**; generic moderation/guardrails are already marketed and measured. |

[INFERENCE] HN/issue comments are anecdotal practitioner signals, not a representative buyer survey. Explicit auditability demand is strongest in enterprise context/permission documentation and governance use cases, weaker as a quantified launch-thread purchasing criterion. No opened source establishes the market size or premium for mathematically certified decisions. That remains unknown.

[INFERENCE] The uncovered **combination**, not an untouched feature, is: commercial-friendly open deployment; preserved, non-floored probabilities; exact schema-equivalence behavior; an explicit injection/authority threat model; **nontrivial validated certified risk at useful coverage**; and audit/replay with stated drift invalidation. No surveyed vendor demonstrates that end-to-end contract. MD1 has empirical invariance/safety, Sage nullable escalation, H2O held-out routing curves, Kev calibration warnings, Strands intervention hooks, Bobcat detailed audits and SubQ long context. Claiming none offers safety or calibration would be false.

## 7. Ranked implications for SSDM

[INFERENCE] These are strategic priorities, not measured SSDM performance projections. SSDM's governing context says all capability targets failed, hybrid binding is the main line, strict attention-free is research, and current lower-tail certification establishes no useful threshold. Historical SSDM Index0.2.1 figures cannot be subtracted from today's0.3.2 scores. [SSDM-context, supplied2026-10-14]

1. **Sell a bounded decision-risk contract, not a universal smarter model.** Choose a consequential but reviewable routing/triage policy with known labels and explicit authority boundaries. Publish certified risk **and coverage**, audit assumptions and expiration; if coverage is vacuous, there is no market advantage. MD1/Sage/Bobcat make generic safety claims insufficient. [MD1; Sage-safety; Bobcat; Robust]
2. **Use an open leader as the capability control and potential product base.** Quyet/Decisio/deck/H2O supply legally and technically real follower routes; distinguish additional training from native readout, caching or quantization. Be explicit that adopting Gemma attention changes the product architecture, not evidence that SSDM's attention-free hypothesis succeeded. [§5]
3. **Preserve probability semantics before adding guarantees.** Never certify on top of a confidence floor, hidden temperature change or incompatible confidence definition. Separate benchmark-compatible settings from operational calibrated-risk settings; otherwise “aligned” would describe a display convention rather than the downstream decision. [H2O; Mapika; Sage-SystemOne]
4. **Exploit exact equivariance selectively.** Exact option relabel/reorder invariance can win audit-sensitive workloads, but pair it with correct decisions, useful coverage and quantified runtime. MD1's empirical low flips and the coherence benchmark's uninformative uniform baseline make a bare invariance badge too weak. [MD1; Coherence]
5. **Keep hybrid recurrence as an engineering option, not a moat claim.** Qwen decision leaders already use GDN plus attention; Liquid mixes convolution/attention; SubQ pitches linear context scaling. A strict attention-free entrant must prove its own domain-transfer and semantics without borrowing hybrid scores. [Qwen4; Qwen27; Liquid-open; SubQ]
6. **Compete on actual serving economics and portability.** Report a fixed workload's device, precision, p50/p95, load, context, multiquestion reuse, memory and validated branch stability. Official K does not reward compression directly; vendor short-query GPU timing does not establish edge/API latency. [J-pricing; J-api; Decisio; Bobcat]
7. **Make unseen-domain and adversarial evidence part of the claim.** Private Index transfer, injection targets, paraphrase/relabel tests and probability-preservation audits are stronger than public fit. Keep failure/refusal denominators visible. Do not claim unsupported privacy, arbitrary injection immunity or high-stakes autonomous authorization. [DI; Robust; Hidden; Kev]
8. **Separate frozen benchmarking from product evaluation.** Maintain exact checkpoint/adapter/readout/calibrator hashes and dated benchmark edition; show live-rank displacement separately from improvements. Do not chase a fresh-draw point lead with pending offsets, or compare SSDM's older index directly to current leaders. [J-data; J-api; DI-old; DI]

## Appendix A. Every new Index configuration observed after the cutoff

[MEAS] All following entries are absent from the **2026-10-09 23:53** snapshot and present in the **2026-10-14 09:16** snapshot. Scores/ranks are from the latter. This is a first-observed interval, not a fabricated release date; full deployment/license details remain null where not established. [DI-old, DI]

| Index addition absent on Oct 9 (source name) | Full / public | Ordinal rank Oct 14 |
|---|---:|---:|
| vLLM-SR d3 (27B) | 66.26 / 65.45 | 1 |
| linne-oak 1.0 (31B) | 65.61 / 63.94 | 2 |
| Decider 31B (NVFP4) | 64.62 / 64.27 | 3 |
| Sieve 31B | 63.81 / 62.52 | 4 |
| Bobcat 1.3 (31B) | 62.83 / 59.31 | 5 |
| Darwin-27B-ZTC | 62.46 / 62.63 | 7 |
| MORJ 1.0.1 (Gemma 4 31B) | 60.96 / 60.24 | 8 |
| AnyJev L0 · Gemma-4-31B | 60.22 / 59.79 | 9 |
| Sieve 27B | 60.17 / 60.04 | 11 |
| OneJev1.5-27B | 59.90 / 63.64 | 14 |
| Shisa DE-2 (repeat) | 59.77 / 57.57 | 15 |
| Matilda Jev v1.5 | 59.23 / 63.47 | 16 |
| RSI-Jev v6.1 VL 27B (effort auto) | 58.66 / 65.64 | 20 |
| decisio 0.6.0 (Gemma 4 31B) | 58.55 / 58.09 | 21 |
| Qwen Decision 27B noncausal LoRA | 58.52 / 57.70 | 22 |
| garas (Gemma 4 31B) | 58.34 / 56.59 | 23 |
| Wald-26B-A4B | 58.29 / 62.30 | 24 |
| linne-fern 1.0 (27B) | 57.81 / 58.92 | 25 |
| SynACK Decide 26B-A4B v7 | 57.46 / 59.68 | 28 |
| Xor 26B-A4B v1.1 | 57.07 / 57.62 | 30 |
| AJev lora1 (Gemma 4 26B-A4B) | 56.43 / 57.67 | 31 |
| frontier-infra Jebadiah 27B v2.1 | 55.94 / 57.03 | 33 |
| Nace.AI Drex v2.0 (35B-A3B) | 55.94 / 61.25 | 34 |
| vLLM-SR d3-flash (9B) | 55.79 / 57.79 | 36 |
| reflex Qwen3.8-27B-FP8 (pmscr) | 54.72 / 57.34 | 40 |
| OneJev1.5-9B | 53.29 / 59.03 | 42 |
| vLLM-SR d3-mini (4B) | 52.61 / 54.90 | 44 |
| Sieve 9B v2 | 50.89 / 53.35 | 49 |
| AJev lora7 (Gemma 4 12B) | 50.68 / 55.17 | 50 |
| Nace.AI Drex v1.5 | 49.72 / 58.08 | 51 |
| Standard One 8B SH (v4) | 49.56 / 57.40 | 52 |
| OneJev1.5-4B | 49.17 / 54.89 | 54 |
| CoCo-Decision-4B 2.3.0 | 48.82 / 55.38 | 55 |
| PurpleMIST-Flash-1.0 | 46.83 / 46.74 | 66 |
| RSI-Jev v6.1 VL 4B | 46.33 / 50.98 | 67 |
| Sieve 4B | 46.11 / 50.06 | 68 |
| frontier-infra Jebadiah 9B v2.1 | 44.91 / 47.20 | 72 |
| jiwo 4B v2 | 44.83 / 46.06 | 73 |
| Aplomb 1 | 43.77 / 43.49 | 76 |
| FlyMy.AI Decision 4B v1.6 | 43.65 / 42.70 | 77 |
| Saracura PT-BR 4B | 40.17 / 37.96 | 87 |
| Standard One 3B SH (v2) | 39.85 / 42.90 | 89 |
| vLLM-SR d3-nano (2B) | 38.86 / 42.22 | 92 |
| vLLM-SR d3-lite (0.8B) | 32.33 / 36.93 | 107 |
| Sieve 2B v2 | 30.84 / 37.26 | 112 |
| Cam1 | 30.78 / 30.72 | 113 |
| OneJev1.5-0.8B | 29.01 / 42.71 | 114 |
| vera-core | 28.07 / 30.11 | 117 |
| jiwo 0.8B v2 | 24.87 / 29.58 | 121 |
| Sieve 0.8B | 23.06 / 32.26 | 124 |
| Crivo-150M | 22.21 / 36.39 | 126 |
| Saina Helm 2.0 | 20.82 / 30.47 | 128 |
| vLLM-SR d3-edge (0.6B) | 20.24 / 28.82 | 130 |
| tasksource Decider nano v4 (150M) | 17.63 / 30.90 | 133 |
| DecisionTune 1.0 | 16.72 / 28.85 | 135 |
| Crivo-32M | 11.87 / 25.06 | 143 |
| vera-spark | 11.46 / 19.59 | 146 |
| Nirnaya 0.4B | 0.00 / 1.60 | 165 |

## Sources and dates

[MEAS] All links below were opened in this workstream on **2026-10-14**; dates preceding that are the source's publication/measurement, not the access date. Exact row-level provenance also appears in the catalog. Microsoft TechCommunity and a later GitHub API listing request returned403: **not opened on those attempts**, no new numbers attributed to them.

- **J-repo:** https://github.com/fstandhartinger/jevbench ; results description https://github.com/fstandhartinger/jevbench/blob/main/results/README.md ; current repository/publication structure accessed2026-10-14.
- **J-data:** https://benchmarkheaven.com/api/decision-results/jevbench/v1.6.1/json ; frozenv1.6.1, originally2026-10-06 with addenda; row dates in tables/catalog.
- **J-raw / J-latest:** https://benchmarkheaven.com/api/jevbench/v1.6.1 ; https://benchmarkheaven.com/api/jevbench/latest ; opened2026-10-14.
- **J-live / J-api / J-method:** https://benchmarkheaven.com/jev-models ; https://benchmarkheaven.com/jev-models/api ; https://benchmarkheaven.com/jev-models/methodology ; current board and revision/method notes, accessed2026-10-14.
- **J-pricing / J-submit:** https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.5-ADDENDUM-PRICING.md (2026-09-25); https://benchmarkheaven.com/submit (accessed2026-10-14).
- **J-addenda:** https://benchmarkheaven.com/api/jevbench/api-addenda ; MD1 full native measurement2026-10-10.
- **J-new:** https://benchmarkheaven.com/api/jevbench/v1.6.4 ; https://benchmarkheaven.com/api/jevbench/v1.6.9 ; https://benchmarkheaven.com/api/jevbench/v1.6.18 ; measurement windows2026-10-09–14.
- **DI / DI-old:** https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/v03.json (generated2026-10-14T09:16:12Z); https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/3a0b3bb1986dfea422746abe7df106d86b8ee4ec/data/v03.json (generated2026-10-09T23:53:22Z).
- **DI-method / DI-repo:** https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json ; https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json ; https://github.com/apolinario/decision-index ; https://huggingface.co/spaces/multimodalart/jev-decision-index/commits/main ; accessed2026-10-14.
- **MD1 / Azure-doc:** https://commandline.microsoft.com/microsoft-decision-1-model-foundry/ (displayed2026-10-09; HTML metadata2026-10-10; chart widgets decoded locally); https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision (updated2026-10-09).
- **TS-launch / TS-evals / Jev-confidence:** https://typesafe.ai/blog/introducing-system-one-models-and-jev (currently rendered metadata2026-10-14; not reliable original launch date); https://evals.typesafe.ai/ ; https://docs.typesafe.ai/confidence ; https://docs.typesafe.ai/model-jaggedness/jev-1.13 (reviewed2026-10-02).
- **CF-launch / CF-card:** https://blog.cloudflare.com/clef-decision-models (2026-10-01); https://huggingface.co/Cloudflare/clef ; https://huggingface.co/Cloudflare/clef-flash ; https://huggingface.co/Cloudflare/clef-omni ; accessed2026-10-14.
- **CF-price / CF-flash-price / CF-Oct9:** https://developers.cloudflare.com/workers-ai/models/clef ; https://developers.cloudflare.com/workers-ai/models/clef-flash ; https://developers.cloudflare.com/changelog/post/2026-10-09-clef-omni-workers-ai (2026-10-09 tariff/context/service changes).
- **Qwen4 / Qwen27:** https://huggingface.co/Qwen/Qwen3.5-4B ; https://huggingface.co/Qwen/Qwen3.8-27B ; accessed2026-10-14.
- **H2O / Kev / Laya:** https://huggingface.co/h2oai/h2o-lightning-4b ; https://huggingface.co/jaredpalmer/kev-9b (v2 date2026-09-30); https://huggingface.co/convaiinnovations/laya ; accessed2026-10-14.
- **Quyet / Mapika / Mapika-issue / Deck / Decisio:** https://huggingface.co/chinhnc/Quyet-1.0-Large ; https://huggingface.co/Mapika/decider-12b (v2 date2026-09-29); https://github.com/Mapika/decider/issues/18 (2026-09-28–10-02); https://github.com/krishna-gogineni-765/deck31b ; https://github.com/aminry/decisio ; accessed2026-10-14.
- **Pplx / Nimble / Torchcast / Rune:** https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b ; https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3 ; https://huggingface.co/torchcast-ai/torchcast-decision-12b ; https://huggingface.co/surogate/rune-26b-a4b-GGUF (gated body not available; metadata opened); accessed2026-10-14.
- **Bobcat / Bobcat-Flash / Xor:** https://huggingface.co/sanghwa-na/bobcat-1.3 (updated2026-10-14, declared release2026-10-11); https://huggingface.co/sanghwa-na/bobcat-flash-1.2 ; https://huggingface.co/juspay/Xor-26B-A4B ; accessed2026-10-14.
- **Diffusion / djev:** https://huggingface.co/google/diffusiongemma-26B-A4B-it ; https://github.com/Davipar/djev-dev ; current base/runtime accessed2026-10-14.
- **Strands / Strands-card:** https://strandsagents.com/blog/introducing-strands-decider/ (2026-10-01); https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19 ; accessed2026-10-14.
- **Sage-price / Sage-safety / Sage-SystemOne:** https://levanto.ai/pricing ; https://levanto.ai/agentic-security ; https://docs.levanto.ai/systemone ; accessed2026-10-14.
- **Liquid-hosted / Liquid-profile / Liquid-open:** https://www.liquid.ai/blog/d1-decision-model (2026-10-05, modified2026-10-09); https://benchmarkheaven.com/jev-models/liquid-d1 (measurement2026-10-06); https://www.liquid.ai/blog/open-d1 (redirects to /blog/d1-open, 2026-10-07); https://huggingface.co/LiquidAI/d1-3B ; https://huggingface.co/LiquidAI/d1-omni-600M ; https://huggingface.co/LiquidAI/LFM2.5-VL-3B ; https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M.
- **OpenAI-guide / OpenAI-profile:** https://developers.openai.com/api/docs/guides/decisions ; https://benchmarkheaven.com/jev-models/openai-decisions (measurement2026-10-06); accessed2026-10-14.
- **d3-card / SubQ:** https://huggingface.co/vllm-sr/d3 (vendor serving measured2026-10-13); https://subq.ai/introducing-subq-snap (2026-10-12).
- **Robust / Adv / Typed / Hidden / Gates / Coherence:** https://huggingface.co/datasets/yflantmy/DecisionRobust (version1.2.0, opened2026-10-14); https://arxiv.org/abs/2609.31142 (2026-09-25); https://arxiv.org/abs/2610.11392 (2026-10-08); https://arxiv.org/abs/2610.04985 (2026-10-04); https://arxiv.org/abs/2610.00346 (2026-09-29); https://jevbench.github.io/report/JevBench-technical-report.pdf (October2026).
- **HN-Jev / HN-Strands / GoSearch:** https://news.ycombinator.com/item?id=49717558 (2026-09-15 launch, comments2026-09-15–16); https://news.ycombinator.com/item?id=49802161 (2026-09-22); https://news.ycombinator.com/item?id=49987076 (opened2026-10-14); https://www.gosearch.ai/blog/ai-decision-models-in-2026/ (2026-10-14; cited for vendor's own permission/context needs, not its third-party tariff table).
- **SSDM-context:** local://ssdm-context.md and local://direction/brief.md (governing internal facts, supplied2026-10-14).

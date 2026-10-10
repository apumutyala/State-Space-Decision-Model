# 10 — The Clef recipe, the gap from stock Qwen3.5-9B, and what closes each slice

> **Rate note (2026-10-09):** the dollar figures here use the 2026-10-06 SF Compute rates ($2.97/h = $0.0495/min batch, $3.96/h interactive) and are left as written. From 2026-10-08 the H100 list price is $5.04/h and batch jobs bill $3.78/h ($0.063/min; Fridays $0.05628/min), so batch figures scale by x 1.27. Source and history: `jobs/common/rates.py`.

Written 2026-10-06. Same legend as `01`: evidence R (replicated), S (single peer-reviewed), P (preprint / single group), X (our extrapolation); cost $0 / $ (<$100) / $$ …. Sources are tagged **[P]** primary (the paper, model card, official blog, repo, board data file) or **[S]** secondary (aggregator, news, third-party summary). Every URL was fetched or found in search on 2026-10-06 unless marked **[unverified]**. Where something is not published, this file says **unknown**.

This file extends `01` (conversion), `02` (recall), `06` (calibration) and `reviews/01` (Clef/Jev fact-check). It does not repeat them. New here:
- the Index 0.3 change;
- our own per-benchmark gap measurement;
- a $0 readout-fusion experiment;
- an attention-free mapping of Clef's head;
- Clef as a teacher;
- a ranked experiment list;
- the component-first vs end-to-end answer.

Receipt for every number we measured: `receipts/2026-10-06-D-recipe-gap/`.

---

## 0. Headline (read this first)

1. **The goal post moved today.** Decision Index **0.3** went live on 2026-10-06. The ranking is now a *Full* score: 20% public suite, 50% private same-skill tests, 30% private new-domain tasks (Space `data/v03.json`, `generated_utc 2026-10-06T16:32:56Z`, sha256 `9f8a0a03…` [P]; kit PR #74 [P]).
   - On Full, **Clef is 53.08 (rank 19)** and **Clef-flash is 47.61 (rank =31)**.
   - **Jev is 60.11 (=2)**, Perplexity Decider v1.1 27B is 62.75 (#1), and the best 9B, Bespoke-Nimble-9B-v3, is 54.67 (#18), already above Clef.
   - Clef-flash loses 8.5 points from its public score (56.15) to Full. Its same-skills score is 44.01 and its new-domains score is 42.50.
   - GOAL.md says "0.2.1 (or the edition current at submission)". So **the T1/T2/T3 numbers in PLAN.md are stale.** Beating Clef on 0.3 Full is *easier* than on 0.2.1, but beating Jev is *harder* (60.11).
   - Re-targeting is a `decision-needed` for the user. This file does not change PLAN.md.
2. **The whole 17.5-point DEV gap between stock Qwen3.5-9B (38.27) and Clef-flash (55.78) comes from post-training on the same frozen backbone** [P: blog "freezing … Qwen3.5-9B for Clef-flash"]. It is not backbone knowledge.
   - Gross per-benchmark gains for Clef-flash sum to +22.0 Index points. Its losses sum to −4.5 (six stock-wins slices −4.3, BRIGHT −0.2).
   - Five benchmarks (Home appliances, WinoGrande, CRUXEval, API-Bank, CLadder) carry 59% of the net gap.
3. **Stock beats Clef-flash on six benchmarks** (RAGTruth, GPQA, ANLI, PhishNChips, HoVer, ACOS; −4.3 points of Index). This is where the LM's own answer-token readout knows something Clef's head discards. Clef's head never reads the LM's vocabulary logits.
   - Our **$0 test:** a log-linear pool of the two readouts gives **+1.27 [−0.38, +2.83]** over Clef-flash. That number is in-sample: w was picked from 3 values, and the CI includes 0.
   - Language rises by 3.6 and retrieval by 5.0. Tools falls by 5.4.
   - So a head that *also* consumes the LM answer-logprob is the first cheap design lever.
4. **Under 0.3, levers that generalise matter most.** These are readout, joint decoding, calibration, and diverse generator data. Format supervision on public train splits matters less.
   - Evidence that public overfit is penalised: Clef-flash's public→Full drop is −8.5, while Nimble-9B-v3 (LoRA r=64, not r=256) drops only −2.5.
   - Every m2 cell must report a **held-out-family proxy** next to the public DEV score (§9, E1).
5. **Answer to "components or bigger pieces?"** Use **anchor-then-ablate**.
   - First rebuild one end-to-end *anchor*: Clef-shaped head + LoRA + LS-CE+Brier on stock Q9, trained on our 150k mix. It should land near Clef-flash.
   - Then run one-factor-at-a-time component swaps *from the anchor*, not from the stock model. Gains measured from the stock model do not add up, because the readout, training and calibration interact (§10).
   - The conversion to attention-free (Q9-AF) is then one more swap, measured after task training, not with the stock readout.

---

## 1. Clef and Clef-flash: what is known

| Item | Clef-flash | Clef | Source |
|---|---|---|---|
| Backbone | Qwen3.5-9B, **frozen**; 24 GDN + 8 full-attention layers (3:1); vision encoder | Qwen3.8-27B, frozen; 48 GDN + 16 attention | blog [P] https://blog.cloudflare.com/clef-decision-models/ (2026-10-01, Michelle Chen); HF configs [P]; `reviews/01` |
| Adaptation | "jointly optimized the routing head alongside **rank-256** low-rank adapters". Released as merged full weights plus `joint_head.safetensors` (no separate LoRA; HF discussion #2 asks for one) | same | blog [P]; HF cards [P] https://huggingface.co/Cloudflare/clef-flash, https://huggingface.co/Cloudflare/clef |
| Head | `joint_head_config.json`: hidden 4096, width 1024, `routing_layers 2`, `layers 4`, heads 16, ffn 4096. About 128M parameters (256 MB bf16) | hidden 5120, same otherwise | HF repo [P]; vendored `third_party/cloudflare_clef/joint_schema_model.py` |
| Head mechanics (from the code) | (a) Option queries = context-span mean + lexical (mean of **output-embedding** rows of the option tokens) + question vector. (b) 2 `EvidenceRoutingLayer`s: options attend over the whole sequence. (c) A within-question softmax over options gives each field an option summary. (d) Fields = question + option summary + global + type embedding. (e) 4 `TransformerDecoderLayer`s: fields self-attend and cross-attend the sequence. (f) Logit = `prior_scale·cos(lexical, q+global)` + `σ(gate)·(joint_scale·cos(field, option) + MLP[f, o, f⊙o, \|f−o\|])` | same | vendored code (read 2026-10-06) |
| What the head does **not** use | The LM's next-token / vocabulary logits. There is no answer-token readout. The output embedding is used only as a lexical prior | same | vendored code |
| Prompt | system "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options." Then `STATE:`, schema, `<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`. **State first, schema second** | same | HF code [P] |
| Inference | "prefill-only pass, then scores the valid schema choices in parallel". All fields of a record are decided in one pass | same | blog [P] |
| Losses | "label-smoothed cross-entropy for valid schema outputs paired with a Brier loss to refine probability calibration" | same | blog [P] |
| RLCD | "Reinforcement Learning for Calibrated Decisions (RLCD) … secondary optimization target, granting partial credit to adjacent ordinal choices, rewarding fully precise record outputs, and applying a reference penalty to prevent distribution shift." The name is TypeSafe's (Jev launch post, 2026-09-15 [P]). Neither company has published an algorithm. The board's tracker says "described only at blog-post level; no public implementation" [P] | same | blog [P]; https://typesafe.ai/blog/introducing-system-one-models-and-jev [P] |
| Training data | "our own internal synthetic datasets permutating field orders, prompts, and schema structures". **Datasets, row counts, steps, compute and use of public train splits are unknown.** HF discussion #8 "Question about how Clef was trained" (2026-10-06) is unanswered | same | blog [P]; HF discussions [P] |
| Lineage | The blog says Cloudflare first "adapted the DiffusionGemma model to output deterministic probabilities by exposing the logprobs", building on Matt Mastracci's vLLM work (vLLM PR #57250; https://github.com/mmastrac/djev [P]). djev reads one denoise step of DiffusionGemma's answer slots; options must be single tokens; no Index number in the repo. Board entry "djev diffgemma": 0.3 Full 41.78. **No separate Cloudflare post on the DiffusionGemma experiment was found.** The blog links an X post (https://x.com/michellechen/status/2101091012559151480, not fetched) | — | blog [P]; djev README [P]; v03.json [P] |
| Context | 65,536 tokens; 1–64 questions per request; ≤4 images. `encode_record(max_length=16384)` silently truncates by default | same | https://developers.cloudflare.com/workers-ai/models/clef/ [P]; PR #54 [P] |
| License | Apache-2.0 | Apache-2.0 | HF cards [P] |

**Published scores.**

| Edition / source | Clef-flash | Clef |
|---|---|---|
| 0.2.1, independent run (PR #54, lucataco, **still open**; uncapped encode, refuses >65,536 tokens, 150,317/150,317 ok, RTX PRO 6000) | not submitted | 61.98; K 51.19 / L 64.28 / R 62.95 / T 80.63 / A 47.88; 115.3 ms median |
| 0.2.1, Cloudflare internal | — | 61.21 (`reviews/01`) |
| 0.2.1, benchlm aggregator | 57.07 **[S, unverified against any primary]** | — |
| **0.3, maintainers' lab run** (`v03.json`) | public 56.15, same-skills 44.01, new-domains 42.50, **Full 47.61**; ECE 0.0249; 53.4 ms | public 61.71, same 53.11, new 41.88, **Full 53.08**; ECE 0.0395; 102.1 ms |
| **Ours: DEV 12,104 rows** (job-sgihj, decide path) | **56.79 [55.40, 58.22]** | not run |
| **Ours: 2,008-row DEV subset** | **55.78 [52.99, 58.52]** | — |

Blog per-benchmark table (Clef / Clef-flash / Jev) [P]:
- BFCL 98.47/98.76/95.75; ToolRet 69.19/66.43/65.28; API-Bank 91.93/93.11/88.19; Home appliances 82.95/97.73/52.27; When2Call 72.37/65.58/80.97;
- BANKING77 94.20/90.93/79.74; CLINC150+OOS 97.43/**66.77**/89.27; BRIGHT 45.91/39.26/47.52; ESCI 57.48/57.39/55.21; PhishNChips 79.60/75.05/62.55;
- median latency 209.3/38.8/524.1 ms.

These are raw metrics, not our chance-corrected skills, so they do not compare directly with §4.

**What we can and cannot infer.**
- Clef-flash is "Qwen3.5-9B + rank-256 LoRA + 128M attention head + LS-CE + Brier (+ RLCD) on synthetic, schema-permuted data". That is the whole recipe. Nothing else is disclosed.
- Its near-perfect DEV scores on WinoGrande (93.1 full DEV), CLadder (97.0) and CRUXEval (79.5) are consistent with heavy format supervision on those families (X). They are also consistent with Clef-flash's large public→private drop on 0.3. Whether the suite's own items leaked is **unknown**. The maintainers' held-in/held-out checks would answer it; Clef's PR is still open.
- **0.3 also rebuilt GSM8K** because "the gold answer was always the center option" (kit issue #32 [P, via the agent report]). A trained head can learn such a positional artifact. That is a candidate explanation for part of Clef-flash's GSM8K 56 vs stock 23 on our 0.2.1 DEV (X, unmeasured).

## 2. Jev and the top entrants: what got them there

**Jev** (TypeSafe; launch post 2026-09-15, Diogo Almeida [P]):
- "new model architecture, parallel sampler … and … RLCD"; "Generates all outputs in a single query"; 70–500 ms.
- Docs (https://docs.typesafe.ai/models [P]): `jev-1.13.0`; 64k per request, 32k for state + longest question; text only; "not trained on customer requests".
- Architecture, data and reward are **unknown**. 0.3: public 57.96, same-skills 58.04, **new-domains 54.98**, Full 60.11.
- Jev transfers best to new domains. That, not the public score, is what 0.3 rewards.

Entrants (methods are author-reported in PRs and cards [P], unaudited; scores from `v03.json` [P] unless marked):

| Entrant (base) | Adaptation / readout | Data | Calib. | 0.3 Full (public → new-domains) | Lesson for us |
|---|---|---|---|---|---|
| Perplexity Decider v1.1 (Qwen3.8-27B) | full FT | undisclosed; WinoGrande / BBH / RAGTruth rows used for checkpoint selection (board disclosure) | ? | **62.75** (62.25 → 55.57) | full FT on 27B; best transfer |
| GLiDE no-think (≈28B, Fastino) | full FT; "fast initial assessment, then additional reasoning when uncertain" (thinking variant not on board) | undisclosed | ? | 60.21 (59.06 → 52.87) | — |
| Torchcast Decision 27B (Qwen3.8-27B) | merged LoRA; letter logits at the last position; wide-choice rounds for >26 options | "public datasets, train and dev splits only, plus benchmark-format decision data built from them"; "Decision Index results guided checkpoint selection" | per-type T | 59.91 (**65.10** → 50.83) | public-format supervision buys public score that does not transfer |
| Kev 27B (jaredpalmer/kev) | full FT, 145,840 records, 8×H200 ≈ 16 h | public datasets + generated policy/rule structures; "No Jev outputs" | T on held-out datasets | 58.78 (56.69 → **55.49**) | diverse *generated* rule data transfers |
| Decider chat (stock Gemma-4-31B) | no training; `Answer: (` letter logits; option-count temperature T(n) = 10.124 − 1.633 ln n | T fit on own train rows | T(n) | 57.62 | a pure readout + T(n) on a strong stock model ≈ 57 |
| **Bespoke-Nimble-9B-v3 (Qwen3.5-9B)** | **LoRA r=64, α=128 on all linear incl. DeltaNet `in_proj_*`**; single-token candidate codes A..Z, AA..; shared-prefix pass | v3 undisclosed. v1: 2,676 synthetic "flip one fact" contrastive pairs, CE over allowed candidate logits | T | **54.67** (57.19 → 49.05) | **best 9B; same backbone as Clef-flash, beats it by 7 on Full** |
| Clef-flash (Qwen3.5-9B) | r=256 + attention head | synthetic permutations | LS+Brier | 47.61 (56.15 → 42.50) | head + r=256 overfits the public formats (X) |
| vLLM-SR Lux 9B / JevK5 9B / JEV-9B / Sieve 9B | various | — | — | 48.21 / 48.35 / 45.21 / 44.59 | — |
| JPT-9B (Qwen3.5-9B) | LoRA r=16, letter readout | 49,221 questions, two option-shuffled copies, **multi-class Brier loss** | one T | 44.45 | — |
| Kev 9B v2 | r=16 LoRA + pointer head | 10k public + generated | T=2.19 | 43.27 | — |
| Scion-1 9B (Fireworks) | r=64 LoRA; **prompt sent twice** (JRT-style); 588 single-token labels | BANKING77 / CLINC train | T | 0.2.1 45.93 (PR #37) | a JRT-style repeat in a hybrid entrant |
| Wald-Q4B v1.1 (4B) | LoRA SFT; **writes ≤512 thought tokens** | synthetic WaldGen + ACOS/NLI4CT/RAGTruth/VAST; Haiku-generated, Sonnet-verified | — | 0.2.1 54.59; not on 0.3 (821 ms median) | CoT under the 1 s cap is possible at 4B |

Patterns, for a 9B attention-free entrant:
- **(i)** On 0.3 every 9B entrant's Full is below its public score. The private parts decide rank.
- **(ii)** The best 9B (Nimble) uses a *simpler* head than Clef (single-token codes read from LM logits) and a *lower* LoRA rank. Its data is contrastive and synthetic.
- **(iii)** The best transfer at 27B comes from diverse generated or undisclosed data (Kev, Perplexity), not from benchmark-format data (Torchcast).
- **(iv)** Readout + option-count temperature on a strong stock model is worth about 57 by itself (Decider chat).

## 3. Index mechanics that bear on the recipe

Unchanged from `harness/INDEX_NOTES.md`: chance correction, unanswered = wrong, no truncation, no option filtering, gold ×1.2, case-exact benchmarks. New or sharpened by the agent check of the kit README / `docs/engines.md` / methodology on 2026-10-06 [P]:
- **"No prompt tuning. The reference engines use one fixed rendering for every benchmark."** `docs/engines.md`: "do not adapt the prompt per benchmark". Per-benchmark temperatures are refused ("never per benchmark").
  - So a **per-benchmark router** between readouts is out.
  - A **feature-based** gate is in: one model, one rendering, with weights that depend on option count, length or question type. So is option-count T(n).
  - Batch calibration on unlabeled test rows (2309.17249) is not mentioned. Treat it as a `decision-needed` before use.
- **0.3 latency gate:** median, mean *and* 80th-percentile ≤ 1,000 ms on one RTX PRO 6000. Under 0.2.1 it was the median only.
- **0.3 submission:** maintainers must be able to run it ("weights on the Hub … the inference code … and its settings"). The private sets are never published.
- **Retrieval / external knowledge / tools / internet at inference, ensembles, few-shot in the adapter, kNN over training data: no explicit rule** in README, engines.md, suite.md or the methodology. Few-shot that varies by benchmark breaks "one fixed rendering".
  - A retrieval corpus would have to ship with the submission and fit the latency gate.
  - **Verdict: retrieval is not forbidden by any text we found, but it is unprecedented on the board and risky. Ask the maintainers first (`decision-needed`).**
- **Chain-of-thought is allowed** under the latency cap (Wald, Darwin). A separate Reasoning board is "coming in 0.3.x".
- 0.3 public changes: GSM8K rebuilt, **ForecastBench retired**, WinoGrande moved to Knowledge & Reasoning. Our per-benchmark table (§4) is on the 0.2.1 public DEV slice. ForecastBench's 0.85-point slice no longer counts on 0.3.

## 4. Measured gap: stock Qwen3.5-9B → Clef-flash, per benchmark

**Setup.**
- Arms:
  - Clef-flash (job-sgihj, `decide` path);
  - stock Qwen3.5-9B with our m0 readout (job-5cmat stock arm: JRT options-first prompt, chat template with empty think block, summed log-likelihood of `" " + option_id + terminator`, option-count temperature);
  - Q9 with one attention layer converted to GDN (job-5cmat conv arm);
  - RWKV7-13.3B stock state-fork readout (job-tkfaj).
- Rows: the same 2,008-row DEV subset (`dev_subset_m1.json`). Skills are the kit's chance-corrected per-benchmark scores (0–100).
- "Index pts" = area weight × gold-normalised benchmark weight × skill gap / 100. These sum exactly to the 17.51-point Index gap.
- Per-benchmark ± is the cluster-bootstrap SE. Many slices have 12–60 rows, so single-benchmark gaps below ~15 skill points are noise.

**Area level (subset):**

| Area (0.2.1 weight) | Clef-flash | Q9 stock | Q9 1-GDN | RWKV7-13.3B | Gap → Index pts |
|---|---|---|---|---|---|
| Knowledge (25.85) | 47.2 | 25.0 | 28.0 | 8.7 | **5.74** |
| Language (25.85) | 56.2 | 47.4 | 51.4 | 41.3 | 2.29 |
| Retrieval (20.02) | 45.9 | 40.2 | 41.6 | 23.8 | 1.14 |
| Tools (18.28) | 82.4 | 49.5 | 49.6 | 31.7 | **6.01** |
| Arts (10.0) | 48.0 | 24.6 | 20.1 | 13.6 | 2.34 |
| **Index** | **55.78** | **38.27** | **39.93** | **24.86** | **17.51** |

**Per benchmark**, sorted by Index-point gap. CF-sub is Clef-flash on the subset; CF-DEV is on all 12,104 DEV rows (tighter). The id class is the option-key format: letter `A..`; `option_N`; hash `k_…`; semantic words. q-acc is question-level raw accuracy, stock → Clef (from `paired_by_idclass.txt`). Lever codes are defined below the table.

| # | Benchmark (id) | Area | Wt | CF-sub | CF-DEV | Q9 | Q9-1GDN | RWKV7 | Gap skill | **Index pts** | id class; q-acc S→C | Lever |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Home appliances (9) | T | 3.39 | 91.7 ±7.7 | 94.9 | 8.3 ±8.0 | 8.3 | 0.0 | +83.3 | **+2.82** | hash; 82→100 per question, case-exact over ~18 fields | J, G |
| 2 | WinoGrande★ (28) | L→K in 0.3 | 2.93 | 100.0 | 93.1 | 30.2 ±14.2 | 39.5 | 44.2 | +69.8 | **+2.04** | letter; 65→100 | F |
| 3 | CRUXEval (43) | K | 2.39 | 84.9 ±7.2 | 79.5 | 1.5 ±6.8 | 5.3 | 0.0 | +83.3 | **+1.99** | option_N; 38→90 (chance 37) | G, R |
| 4 | API-Bank★ (3) | T | 4.06 | 97.3 ±2.6 | 92.0 | 51.7 ±8.3 | 54.4 | 14.2 | +45.6 | **+1.85** | option_N (53 tools, 25k-char catalogs); 53→97 | F, recall |
| 5 | CLadder (44) | K | 2.39 | 95.3 ±4.6 | 97.0 | 30.2 ±14.6 | 44.2 | 20.9 | +65.1 | **+1.56** | letter; 65→98 | G, F |
| 6 | BANKING77★ (4) | R | 3.64 | 85.3 ±4.4 | 88.1 | 49.8 ±5.0 | 58.2 | 33.3 | +35.5 | +1.29 | option_N (77); 67→89 | F |
| 7 | ForecastBench★ (48) | A | 1.67 | 78.6 ±4.7 | 63.3 | 27.6 ±11.7 | 24.4 | 22.2 | +51.0 | +0.85 | Brier-scored; **retired in 0.3** | C |
| 8 | MMLU-Pro★ (57) | K | 2.87 | 63.4 ±8.0 | 61.4 | 34.6 ±8.4 | 32.0 | 24.1 | +28.8 | +0.83 | letter (10); 42→67 | F, K, R |
| 9 | MuSR (32) | K | 2.39 | 74.1 ±8.7 | 83.6 | 40.8 ±11.7 | 48.2 | 3.9 | +33.3 | +0.80 | letter; 63→84 | G, R |
| 10 | GSM8K (30) | K | 2.39 | 56.1 ±10.2 | 67.3 | 23.2 ±10.3 | 31.3 | 9.6 | +32.8 | +0.79 | option_N (10); 36→64; **center-option artifact, rebuilt in 0.3** | R, (artifact) |
| 11 | NLI4CT (42) | L | 2.44 | 62.4 ±14.2 | 63.0 | 32.5 ±15.7 | 29.2 | 30.5 | +30.0 | +0.73 | semantic; 72→82 | F |
| 12 | Amazon ESCI (37) | R | 3.03 | 28.7 ±7.7 | 42.9 | 4.7 ±4.1 | 1.9 | 3.6 | +24.0 | +0.73 | letter; 57→63 (conservative macro-F1) | F, C |
| 13 | When2Call (62) | T | 3.39 | 58.3 ±4.4 | 54.7 | 39.4 ±4.6 | 35.6 | 18.2 | +18.9 | +0.64 | letter; 55→69 | F |
| 14 | ContractNLI (11) | L | 2.44 | 75.1 ±3.5 | 78.8 | 50.6 ±5.9 | 54.0 | 14.5 | +24.5 | +0.60 | semantic; 75→87 | F |
| 15 | SATA-Bench (33) | K | 2.39 | 41.1 ±7.6 | 39.3 | 19.9 ±6.4 | 19.9 | 8.1 | +21.2 | +0.51 | semantic, case-exact multi-select; 78→85 | J, G |
| 16 | FinEntity (39) | L | 2.44 | 100.0 | 97.6 | 79.5 ±6.9 | 89.3 | 64.6 | +20.5 | +0.50 | semantic; 85→100 | F |
| 17 | ToolRet (2) | T | 3.39 | 61.8 ±4.7 | 58.4 | 50.1 ±5.3 | 51.1 | 44.0 | +11.7 | +0.40 | nDCG; **3/51 stock rows over our 16,384 cap** | H, recall |
| 18 | BPoMP (20) | A | 1.39 | 94.3 | 94.1 | 67.1 | 70.0 | 25.7 | +27.1 | +0.38 | option_N; 82→97 | F |
| 19 | VAST (41) | L | 2.44 | 27.3 | 26.1 | 12.3 | 14.9 | **48.2** | +14.9 | +0.36 | letter; 49→53 | F |
| 20 | HellaSwag★ (29) | L | 2.93 | 96.9 | 97.4 | 84.5 | 78.3 | 87.6 | +12.4 | +0.36 | letter; 88→98 | F |
| 21 | Habermas (50) | A | 1.39 | 42.5 | 53.6 | 17.2 | 14.9 | 19.5 | +25.3 | +0.35 | option_N | F |
| 22 | New Yorker (64) | A | 1.39 | 58.3 | 61.0 | 35.9 | 13.5 | 19.9 | +22.4 | +0.31 | letter | F |
| 23 | BFCL★ (1) | T | 4.06 | 96.9 | 98.4 | 89.5 | 89.5 | 76.9 | +7.4 | +0.30 | semantic, case-exact; 96→99 | J |
| 24 | Humicroedit (21) | A | 1.39 | 42.4 | 45.6 | 22.0 | 15.2 | 5.1 | +20.3 | +0.28 | semantic | F |
| 25 | ChessBench (31) | K | 2.39 | 14.4 | 13.4 | 6.8 | 9.3 | 0.0 | +7.6 | +0.18 | semantic | K |
| 26 | HLE★ (45) | K | 2.87 | 6.2 | **0.0** | 0.0 | 0.0 | 0.0 | +6.2 | +0.18 | letter | K |
| 27 | cfcolor (23) | A | 1.39 | 12.7 | 38.5 | 0.7 | 0.7 | 0.7 | +12.0 | +0.17 | letter | ? |
| 28 | iSarcasmEval (40) | L | 2.44 | 44.9 ±29 | 18.1 | 40.0 | 57.1 | 74.3 | +4.9 | +0.12 | semantic | P |
| 29 | CLINC150★ (5) | R | 3.64 | 68.8 | 61.4 | 66.8 | 66.2 | 44.9 | +2.0 | +0.07 | option_N (151); **stock wins per question 82 vs 75** | P |
| 30 | BBH★ (58) | K | 2.87 | 35.9 | 51.3 | 35.9 | 35.9 | 19.1 | 0 | 0 | semantic | K |
| 31 | POP909 (22) | A | 1.39 | 1.3 | 0.7 | 1.3 | 1.3 | 0.0 | 0 | 0 | 129 options; **58/59 stock rows backend error** | H |
| 32 | BRIGHT★ (36) | R | 3.64 | 21.9 | 16.3 | 26.1 | 25.8 | 5.8 | −4.3 | −0.16 | **4/16 stock rows over 16,384 cap** | P, H |
| 33 | ACOS (38) | L | 2.44 | 19.8 | 19.3 | 29.1 | 34.3 | 16.5 | −9.3 | −0.23 | per-pair yes/no; stock 99 vs 95 | P |
| 34 | HoVer (61) | R | 3.03 | 13.2 | 20.2 | 21.1 | 26.3 | 0.0 | −7.9 | −0.24 | semantic | P |
| 35 | PhishNChips (56) | R | 3.03 | 50.0 | 55.2 | 68.4 | 65.8 | 52.6 | −18.4 | −0.56 | verdict; stock 89 vs 75 | P |
| 36 | ANLI★ (12) | L | 2.93 | 25.0 | 32.2 | 54.9 | 58.5 | 22.0 | −29.9 | −0.88 | letter; **stock 70 vs 51** | P |
| 37 | GPQA Diamond★ (25) | K | 2.87 | 14.3 ±14 | 35.6 | 52.4 ±17 | 52.4 | 0.0 | −38.1 | −1.09 | letter; stock 64 vs 36 (n = 14) | P |
| 38 | RAGTruth (59) | L | 2.44 | 0.0 | **0.0** | 54.5 | 54.5 | 4.7 | −54.5 | −1.33 | noul, F1 vs always-yes chance 0.5177 | P, C |

Lever codes:
- **F**: format supervision (LoRA + head on public train splits or look-alikes).
- **G**: generator-built verifiable synthetic data (no usable public train split).
- **J**: joint / multi-field decoding (case-exact records).
- **R**: reasoning without emitted CoT.
- **P**: keep the LM answer-logprob prior (stock already wins).
- **C**: probability quality (Brier / F1-thresholded).
- **H**: harness (errors, caps).
- **K**: backbone knowledge (no cheap lever).

**Readings (each is a hypothesis with the evidence named):**
- **Option-key format is a stock-readout tax, not a knowledge gap.**
  - Question-level skill, stock → Clef: letter keys 36.2 → 54.6; `option_N` 51.4 → 76.9; hash keys 67.5 → 99.2. On semantic keys the two are equal (80.9 vs 79.9).
  - Our stock readout scores the likelihood of the *key string*. When keys are opaque (`option_7`, `k_5cf8035f`), the LM must bind description → key in one step. A trained head reads the option spans directly.
  - Literature: NVIDIA's matched 8B study finds SSMs "struggle to route the knowledge of each answer into a single answer token", while they match transformers on cloze MMLU (Waleffe 2406.07887 [P]). This is the attention-free version of the same tax.
  - Implication: **the readout, not the backbone, is the first-order problem for an attention-free entrant.**
- **Case-exact multi-field slices amplify small per-question gaps.** Home appliances is 82% per question for stock but 8 skill case-exact. Clef is 100% per question. Joint field decoding plus near-perfect per-field accuracy is the lever (J). SATA and BFCL behave the same way.
- **CRUXEval, CLadder, MuSR, GSM8K** need reasoning that stock single-pass scoring does not show: stock CRUXEval is at chance. Clef reaches 80–97 without emitted CoT.
  - Possible drivers (X; unknown which): supervision on look-alike generated data; distractor-style cues learnable by a head; GSM8K's center-option artifact. The last is now removed in 0.3.
  - These four are 5.1 Index points on the subset. Under 0.3 the private same-skills tests will tell whether Clef's gains here are real skill. Its same-skills 44.0 vs public 56.2 suggests much of it is not.
- **Where stock wins** (RAGTruth, GPQA, ANLI, PhishNChips, HoVer, ACOS; −4.3 points) is where Clef's head lost the LM's own answer signal. Its head has no vocabulary-logit input.
  - Probing literature agrees that heads can lose or gain 0–25 points against the output readout depending on the task (2509.23782 [P]; 2410.02343 [P]).
  - RAGTruth = 0 on both subset and full DEV means Clef's hallucination F1 is at or below the always-yes chance line.
- **Harness artifacts** sit in both arms and roughly cancel in the gap. They do not cancel in absolute score:
  - POP909: 58/59 stock rows error (backend error, 129 options). Clef is ~1 too.
  - BRIGHT / ToolRet: 7 stock rows exceed our 16,384-token cap. The 0.3 rules need uncapped encodes up to the model's limit, as in PR #54.
  - These are free points for the entrant. Fix before m3.
- **The 1-GDN conversion did not hurt** (+1.66 [+0.19, +3.18], job-5cmat). Language and knowledge moved up and arts moved down. This was measured with the *stock readout*, which §10 argues is the wrong place to measure conversion loss.

## 5. Readout fusion: the $0 component test (receipt `2026-10-06-D-recipe-gap`)

**Hypothesis.** Clef-flash's head and the stock LM answer-logprob carry complementary evidence. If so, a pool beats either alone.

**Method.**
- Per question, pool p ∝ p_Clef^w · p_stock^(1−w). For noul questions, take the logit pool.
- Scored with the kit through `jobs.common.dev_subset`. The paired cluster bootstrap compares against Clef-flash.

| w | Subset Index | K / L / R / T / A | Δ vs Clef-flash [95% CI] |
|---|---|---|---|
| 1.0 (Clef-flash) | 55.78 | 47.2 / 56.2 / 45.9 / 82.4 / 48.0 | — |
| 0.7 | 57.04 | 47.8 / 59.8 / 50.9 / 77.0 / 49.6 | **+1.27 [−0.38, +2.83]** |
| 0.5 | 56.65 | 48.0 / 64.4 / 49.8 / 70.5 / 47.3 | +0.87 [−1.99, +3.34] |

**Verdict.** Complementarity is real by area, and the sign is opposite by area.
- At w = 0.5, language gains +8.2 and tools loses −11.8.
- The biggest per-benchmark gains at w = 0.7: iSarcasm, RAGTruth, CLINC, PhishNChips, BBH, cfcolor.
- The biggest losses: API-Bank, Home appliances, CRUXEval.

A *fixed* pool is near-neutral overall (in-sample; CI spans 0). Index rules forbid a per-benchmark weight. The correct form is a **learned, feature-gated pool inside the head**, with the LM answer-logprob as an input feature: one model, one rendering. That is E4 in §9.

Caveats:
- Both arms use attention.
- w was chosen from 3 values on the same rows.

## 6. State of the art that bears on the gap (new items only; see `01`, `02`, `06` for the rest)

### 6.1 Quality kept by attention-free students at 7–14B (adds numbers to `01`)

| Student (method) | Teacher | Attn kept | Tokens | MMLU s/t | Other | Source |
|---|---|---|---|---|---|---|
| QRWKV7-7B-I-RoPE (RADLADS) | Qwen2.5-7B-I | 0% | 0.35–0.7B | **68.2/71.7 (95%, 0-shot)** | WG 71.8; HS 79.3 | 2505.03005 [P] |
| QRWKV6-72B-I (RADLADS) | Qwen2.5-72B-I | 0% | ~0.5B (stage 2) | 77.5/83.4 (93%) | ARC-C 63.8/63.2 | same |
| Llamba-8B (MOHAWK, pure Mamba-2) | Llama-3.1-8B-I | 0% | 12B | **60.0/68.4 (88%, 5-shot)** | ARC-C 54.6/55.1 | 2502.14458 [P] |
| ARWKV-7B | Qwen2.5-7B-I | 0% | 0.83B | 62.4/71.7 | **GSM8K 40.0/82.3 (49%)**; IFEval 52/74 | 2501.15570 [P] |
| Mamba2-Llama3 (MambaInLlama) | Llama-3-8B-I | 0 / 12.5 / 25 / 50% | 20B | 44.2 / 50.9 / 53.9 / 56.7 (5-shot) | GSM8K at 12.5%: 35.0 | 2408.15237 [P] |
| Taylor-Calibrate, Qwen3-8B → GDN hybrid | Qwen3-8B | hybrid (fraction not confirmed) | 0.7B | 12-task avg 70.0/70.7 | RULER 76.7/94.0; 4.9–9.2× fewer tokens to matched recovery | **2606.16429 [P]** (new; nearest to our Qwen→GDN conversion) |
| GenDistill, Qwen3-0.6B pure KDA | Qwen3-0.6B | 0% | ~1B | 35.9/46.1 (generation scoring) | GSM8K 17.6/57.5; NIAH-64K 11% vs 100% | 2603.26556 [P] |

- **No paper reports MMLU-Pro, GPQA, BFCL, API-Bank, BANKING77 or CLINC for a 0%-attention distilled student** (searched 2026-10-06). The knowledge-area retention we will get is unknown. MMLU suggests 88–95%. GSM8K-style arithmetic suggests ~50% unless task-trained.
- **Scoring-protocol trap.** Log-likelihood MC scoring understates teacher–student gaps. Freezing attention showed 5.2 pp under generation scoring vs 0.5 pp under log-likelihood (GenDistill 2603.26556 [P]). Our probe used log-likelihood. **Treat +1.66 as optimistic.**
- Native attention-free models (vendor or self-reported):
  - RWKV7-g0b-13.3B: MMLU 76.5, MMLU-Pro 50.2 (protocol unspecified, likely generation; https://wiki.rwkv.com/basic/RWKV-Evals.html [P vendor]). Our stock readout of the G1k-13.3B gets knowledge skill 8.7. **That gap is the readout tax again, at its largest.**
  - xLSTM-7B: MMLU-Pro raw 24.2 (2503.13427 [P]).
  - Falcon-Mamba-7B: raw 22.9 (same table).
  - No pure attention-free 7B+ model has a BFCL or API-Bank number.
- Mechanisms to add to `01`:
  - Mamba does induction through its short convolutions, not the SSM state (2505.15105 [P]).
  - In hybrids the KV cache alone keeps 64–98% of retrieval, while recurrence alone collapses to 0 (2609.04434 [P]).
  - Jet-Nemotron: about 2 layers are critical for MMLU and 2–3 for retrieval, and they are *different* layers (2508.15884 [P]). This supports per-layer accept/rollback.

### 6.2 Readout and head (adds to `06`)

| Finding | Effect | Source |
|---|---|---|
| Letter (MCF) vs cloze (CF) readout | Llama3-8B ARC-C 79.3 vs ~60.2 (+19); MCF emerges at ~400B pretraining tokens | OLMES 2406.08446 [P] |
| Classification head + LoRA vs letter-generation SFT, full train splits, r=16 | **Qwen3-8B WinoGrande 61.0 → 89.3 (gen) / 89.5 (cls); HellaSwag 80.1 → 95.6**; head ≈ generation at 8B | 2607.03801 [P] |
| MLP probe on all-layer hidden states vs QLoRA | +3.44 vs +8.36 avg on 6 MC sets (Llama3-70B): a head on frozen features gets ~40% of the LoRA gain | InnerThoughts 2501.17994 [P] |
| QK-score from "select-and-copy" heads | Llama-3.1-8B MMLU +3.4, HellaSwag +24.8 | 2410.02343 [P] |
| Linear knowledge probe vs output | TruthfulQA +21 (Qwen2.5-7B); MMLU +0.2 | 2509.23782 [P] |
| Batch calibration (subtract mean per-option score) | +6 to +8 on ICL | 2309.17249 [P]; rule status unclear (§3) |
| Answer symbol bound in one middle layer by sparse attention heads | mechanistic; explains why attention-free letter readout is hard | 2407.15018 [P] (corrects the ID in the brief) |

### 6.3 Losses and calibration (adds to `06`)
- **Label smoothing inverts the selective-prediction ranking.** It lowers the max logit more on correct predictions than on wrong ones. Fix: post-hoc logit normalisation (2403.14715 [P], ICLR 2025). Clef trains with LS. This matters for Brier / ECE and the deployment risk-coverage curve, not for argmax.
- LS preserves calibration in SFT, but much less for large-vocabulary LLMs (2508.00264 [P]). A restricted option head avoids the vocabulary regime.
- SFT calibrates better than RL. Qwen-4B on CSQA: SFT ECE 8.8 vs GRPO 17.0, for +2.7 accuracy (2601.13284 [P]). RLCD's calibration claim is therefore the opposite of the default RL behaviour, and it is unpublished.
- No 7–9B numbers were found for Brier as a training loss on an LLM (unknown). JPT-9B trains on multi-class Brier (0.3 Full 44.45 [P, author]).

### 6.4 Knowledge-area levers without emitted CoT
- **Stepwise internalisation (ICoT-SI):** Mistral-7B GSM8K no-CoT 38 → 51 (+13) with no intermediate tokens; explicit CoT 68 (2405.14838 [P]).
- **System-2 → System-1 distillation fails on GSM8K** (52.8 with CoT → 7.1 distilled; 2407.06023 [P]). It succeeds on bias/perception tasks.
- **Execution-trace data:** CRUXEval-O +24 at 6.7B, but *with* monologue CoT (SemCoder 2406.01006 [P]). The no-CoT effect is unknown.
- **Answer-only causal SFT:** CLadder +14.4 (Llama-3.1-8B; CDCR-SFT 2508.12495 [S]).
- **MMLU-Pro is structurally CoT-dependent:** +19 from CoT for GPT-4o (2406.01574 [S]). A no-CoT 9B has a ceiling there.
- **Retrieval:** MassiveDS, Llama-2-7B, MMLU +3.5 (2407.12854 [P]). There is no 9B MMLU-Pro / GPQA number. Rule status: §3.
- **Pause / recurrent depth** need pretraining-time changes (2310.02226; 2502.05171 [P]). Not retrofittable cheaply.
- **CoT under the 1 s cap** is used by a 4B entrant (Wald, 821 ms median). Under 0.3 its p80 / mean would likely fail the gate (its p95 was 3.8 s). For an attention-free model, emitted CoT costs O(1) memory per token. That is the one place where recurrence is *cheaper* than the hybrid. Keep it in mind for the 0.3.x Reasoning board, not for T1.

## 7. An attention-free equivalent of Clef's two-stage routing head

| Clef component (code) | Function | Attention-free equivalent | Already in `heads/`? |
|---|---|---|---|
| Option queries = context-span mean + lexical (output-embedding mean) + question | each option knows its own text and question | same; uses no attention | **Yes**, partially: `features.build_head_input` option embeddings |
| `EvidenceRoutingLayer` ×2: options attend over the full sequence | *option-specific evidence routing* | (a) **options-first layout + per-option fork**: the state is read after the options, and each option's continuation forks the state (`fork`, `snapshot`). (b) **Head B**, a per-option selective recurrence r[t,k] = a·r[t−1,k] + b·v_t, with gates from (v_t, e_k, q). (c) **integrator**: unnormalised linear attention with the option as the query | **Yes**: `fork`, `snapshot`, `head_b`, `integrator`; `layout` options-first and symmetric |
| Within-question softmax over options → field option-summary | soft choice context per field | softmax over *options* is allowed (it is not over positions); keep it | Partly: `_sym_summary` (mean); add the softmax-weighted summary |
| Decoder ×4: fields self-attend and cross-attend the sequence | *joint cross-field decisions* (case-exact records) | **Field mixer**: 2 rounds of DeepSets-style symmetric update f_i ← f_i + MLP([f_i, mean_j f_j, max_j f_j]), permutation-equivariant over fields, no positions. Payload access stays per field through each field's fork state. Needs all fields in one prefill: [schema with all fields → state] once, then K forks per question by cache copy, instead of one row per question | **No.** Today it is one row per question (Kev-style). The cache fork exists (`backbone.fork_from_prefix`). **Gap to build** |
| Lexical prior `prior_scale·cos(lex_k, normalize(q + g))` | option-word prior, schema-bound | identical (embedding dot product) | Partly: the integrator bias b(e_k, q). Add Clef's exact form |
| *Not in Clef:* LM answer-token log-prob | the LM's own vote (where stock wins, §4–5) | add log p(key_k \| prompt) and the option-count feature as head inputs, with a learned gate | **No.** Our m0 readout computes it, but `heads/` does not consume it. **Gap to build** (E4) |
| Type embedding; noul / choice / score | schema-bound scoring | `typed.py` cumulative link for score | **Yes** |

Design notes:
- Clef's ordering (state first) only works because its head attends back. Ours must stay options-first + re-read (JRT; `02` §2.1).
- Scion-1 9B's "prompt sent twice" shows the repeat works in a hybrid entrant too [P].
- Nimble's single-token candidate codes (A..Z, AA..) suggest a cheap alternative to span pooling: render opaque keys as short codes in the *trusted* scaffolding and keep the original keys in the output map. This is one fixed rendering for every benchmark, so it is allowed (X). It removes the hash/`option_N` tax measured in §4.

## 8. Clef / Clef-27B as an open-weight teacher

- **License:** Apache-2.0 [P]. GOAL.md allows open-weight teachers, and `data/TEACHERS.md` already lists both.
- **Index rules:** distilling teacher probabilities on *our* records is allowed. Running the teacher on DEV/TEST rows to make training labels is not ("do not train on it").
- **What it would buy.** Soft per-option probabilities, ordinal smoothness, and multi-field consistency from a model whose head reads the same backbone as Q9-AF (Clef-flash).
  - Soft-label KD gave +10.4 MMLU for a DeBERTa student (2412.09807 [P/S]). LLM→9B is unknown.
- **The catch:** on 0.3 Clef-flash is a *weak* teacher where it matters: same-skills 44.0, new-domains 42.5, both below Nimble (52.3 / 49.1).
  - Distilling it imports its public-format overfit (X).
  - Clef-27B is better on same-skills (53.1) but not on new domains (41.9).
  - The best transfer teachers are Perplexity Decider v1.1 (new-domains 55.6) and Kev-27B (55.5). **Their licences were not checked: unknown.** Jev is excluded by its terms.
- **Contamination:** Clef's training data is undisclosed (HF discussion #8 unanswered). If it trained on suite items, its soft labels on *our* guard-deduped records carry no test labels. The residual risk is near-duplicates our five item-tier matchers miss (11.6% of candidates were already dropped; `receipts/2026-10-06-D-m2-data-150k`).
  - Record Clef's knowledge cutoff as **unknown**. ForecastBench, the leak-sensitive benchmark, is retired in 0.3.
- **Cost:** job-sgihj ran 12,104 DEV rows for $1.40 including load. A 30k-record teacher pass is ≈ $3–4 on one H100. 150k is ≈ $15–20. Clef-27B is ≈ 2.5–3× that, from the latency ratio (102 vs 53 ms board median).
- **Recommendation.** Use the teacher as a **mixed target** (gold + α·teacher) only after E2 shows our anchor is below Clef on public *and* the held-out-family proxy. Prefer Clef-27B over Clef-flash. Measure with E1's proxy, not only public DEV.

## 9. Ranked component experiments

Ranking: expected Index points per dollar, with a sign check against 0.3 (does it generalise?).
- Costs use the H100 batch rate $2.97/h and our measured rates: stock prefill ≈ 22.6k tok/s; 2,008-row subset scoring ≈ 12 min/arm; Clef-flash DEV $1.40.
- Training-cell $ are **estimates** until `cell_step_profile` (PR #86) reports. `train/M2_RUN_SPEC.md` estimates $12–17 per full cell. The ≤$5 versions below use a 30k-record subsample of the 150k mix.

| # | Experiment | Hypothesis | Cheapest measurement | Expected effect (source) | Composes into the stack as |
|---|---|---|---|---|---|
| **E0** | Readout fusion (**done**) | stock answer-logprob and Clef head are complementary | $0 CPU, existing per-row outputs | measured +1.27 [−0.38, 2.83] in-sample; opposite signs by area (§5) | motivates E4 |
| **E1** | Held-out-family proxy for 0.3 private sets | public DEV over-rewards format supervision; 0.3 private parts do not | $0: tag each DEV benchmark as covered / uncovered by our training families (150k mix: MMLU-aux, GSM8K, ANLI, HellaSwag, WinoGrande, SST-5, NQ, BANKING77, CLINC, BoolQ, Glaive, ToolACE, When2Call, Humicroedit); every cell reports both scores plus the gap | Clef-flash public→Full −8.5 vs Nimble −2.5 (`v03.json` [P]) | **gate for every later experiment**; a win on covered families only does not count |
| **E2** | **Anchor**: Clef-shaped control head + LoRA r=256 + LS-CE+Brier on stock Q9, 30k records | our pipeline reproduces most of the 17.5-pt gap on the same backbone | ≤$5 (subsample, 1 epoch) + $0.6 subset eval; full 150k cell later (~$12–30, estimate) | target ≥ 50 on the subset (Clef-flash 55.8); InnerThoughts/2607.03801 suggest LoRA+head carries most of it | **the reference point** all components ablate from; also the T-anchor for the conversion |
| **E3** | Head-only on frozen stock Q9 (LoRA off): integrator / head_b / snapshot / clef_control | how much of the gap the head alone closes; whether attention-free heads match the attention head given identical features | ≤$5: forward-only over 30k records (~45M tok ≈ 35 min) + training on GPU; same subset eval | head-only ≈ 35–40% of the LoRA gain (2501.17994 [P]); the attention-free vs attention head difference is **unknown**, which is the point | picks the 2 heads that go to LoRA cells; isolates the head from training |
| **E4** | LM answer-logprob + lexical prior + option-count feature as gated head inputs | protects the six stock-wins slices (−4.3 pts) without losing tools | $0 extra once E2/E3 run (same cell, one flag); first a $0 CPU re-fit of a feature-gated pool on existing outputs with option-count and question-type features, cross-fitted by source case | +1 to +3 Index (E0 bound; probe literature 0–25 per task, 2410.02343 / 2509.23782) | part of every head |
| **E5** | Opaque-key → short trusted codes (Nimble-style), one rendering for all | removes the hash / `option_N` key-binding tax for the stock and LoRA readouts | ≤$1: re-run the stock subset arm with the new rendering (12 min) | the letter / `option_N` / hash skill deficit is 18 / 25 / 32 pts at question level (§4); the realised Index gain is unknown | rendering layer; applies to every readout |
| **E6** | Generator data for uncovered high-gap families (executed-Python output MC; CLadder-style causal-graph queries; device-state multi-field records; select-all multi-select) | the 5 top-gap slices with no public train data respond to verifiable synthetic data, and it transfers (Kev-27B new-domains 55.5 used generated rule structures) | $0 CPU build through the existing dedup guard + one ≤$5 cell vs E2 | CRUX +24 with CoT (2406.01006), CLadder +14.4 answer-only (2508.12495 [S]); Home + CRUX + CLadder alone = 6.4 subset Index pts (MuSR, SATA +1.3 more) | data mix; must improve the E1 proxy, not just public |
| **E7** | Field mixer (§7) + one-prefill-K-forks layout | case-exact records need joint decoding | ≤$5 cell vs E2 on multi-field benchmarks (Home appliances, SATA, BFCL, PhishNChips, ACOS) | Home: per-question 82 → need ≥99 for case-exact; Clef reaches 100 with joint layers (§4) | head module; also cuts latency (one prefill per record) |
| **E8** | Loss arms: CE vs LS-CE+Brier (+RPS) + post-hoc logit normalisation | LS helps accuracy but hurts the confidence ranking | in E2/E3 cells (already wired in `train/`) | ECE / Brier / AUGRC; LS ranking inversion (2403.14715) | calibration stack; must be decided end-to-end (§10) |
| **E9** | LoRA rank r=64 vs r=256 | r=256 overfits public formats; r=64 transfers better | 2 ≤$5 cells, compared on E1 proxy | Nimble r=64 Full 54.67 vs Clef-flash r=256 47.61 (`v03.json`; confounded by data and head) | adapter size; also cheaper training |
| **E10** | Schema / field-order / option permutation + "flip one fact" contrastive augmentation | invariance and evidence sensitivity generalise to new domains | $0 data + same cell as E6 | Clef (permutations) and Nimble v1 (2,676 flip-one-fact pairs) both use it; Index shuffle test flags ≥5-pt drops | data mix |
| **E11** | Clef-27B soft-label teacher on our records (§8) | soft targets transfer calibration and ordinal structure | ≈ $3–4 teacher pass on 30k (Clef-flash) or ~$10 (Clef-27B) + one cell | +10 for a small student (2412.09807); 9B unknown; risk of importing public overfit | target mixing; only if E1 shows it helps the proxy |
| **E12** | Full 8-layer GDN conversion (planned, ~$150) measured **after** E2-style training | conversion loss is smaller after task LoRA than under the stock readout, or larger on GSM8K-like reasoning | E2 recipe re-run on the Q9-AF backbone; same subset, same proxy | MMLU retention 88–95% at 0% attention; GSM8K ~49% (§6.1); probe +1.66 under log-likelihood scoring (optimistic per 2603.26556) | the attention-free swap; final stack = Q9-AF + best E3–E10 settings |
| **E13** | ICoT-SI internalisation for GSM8K / MuSR / CRUX-style | single-pass reasoning recovers part of the CoT gain | $$ (multi-stage curriculum); after T1 | +13 GSM8K at 7B (2405.14838) | knowledge area; post-T1 |
| E14 | Retrieval augmentation for knowledge | — | `decision-needed` (rules silent, §3) | +3.5 MMLU at 7B (2407.12854) | parked |
| H | Harness: POP909 backend errors; uncapped encode to 65,536 | free points | $0 code fix + reuse the next eval arm | POP909 / BRIGHT / ToolRet rows currently score 0 | m3 hygiene |

Order to run, given hardware serialisation:
1. E1 and E5-prep ($0, now).
2. E3 and E2 back-to-back on one GPU session, sharing the feature pass. E4 and E8 ride inside them as flags.
3. E6, E7, E9 and E10 as single-factor cells from the anchor.
4. E12 with the winning settings.
5. E11 only if the proxy says the student is below the teacher on transferable skills.

## 10. Component-first or bigger pieces?

**Verdict: neither pure form. Use anchor-then-ablate, with three couplings that must be measured end-to-end.**

**Why not pure component-first from the stock model.**
- The stock model sits at 38.3. Every lever in §9 acts *through* the readout. A component measured on the stock readout (e.g. conversion +1.66, option-key rendering, fusion weight) is measured in a regime that disappears once a head is trained.
- 2607.03801 shows head vs generation readouts converge after LoRA at 8B. So readout differences measured before training overstate what survives.
- Gains from the stock model do not add up.

**Why not build the whole stack at once.**
- One end-to-end cell confounds head, rank, loss, data and conversion. It cannot attribute a miss, and GOAL.md's budget-wall exit requires attribution.

**Anchor-then-ablate.**
- The anchor (E2) is a near-replica of the only disclosed recipe that reaches ~56 on this backbone.
- Each ablation is one factor away from it, on the same subset, and paired. INDEX_NOTES §9: a ~0.65–1.0 point MDE for shared-backbone arms. So each costs one ≤$5 cell and is attributable.
- If the anchor itself misses Clef-flash by a lot, that is the first finding. It points at data, since Clef's data is the undisclosed part.

**Where we must go end-to-end** (components interact; measure only jointly):
1. **Readout × training.** Which head wins can flip once LoRA adapts features to the head. E3's frozen-feature ranking only short-lists. The top 2 heads must be re-run with LoRA (E2-style) before choosing.
2. **Training × calibration.** LS-CE+Brier, the post-hoc temperature T(n, length) and the logit normalisation after LS are not separable. ECE / Brier / AUGRC must be measured on the final trained model with its own post-hoc fit on a calibration split. Never take ECE from a component.
3. **Conversion × task training.** The attention-free swap must be scored after the same task training (E12 = E2 recipe on Q9-AF), not with the stock readout. Log-likelihood scoring understates student–teacher gaps (2603.26556), and task LoRA may recover part of the conversion loss.
4. **Data × generalisation (0.3).** Any data change is end-to-end by nature. Judge it on the E1 held-out-family proxy, because 0.3 weights private parts at 80%.
5. **Layout × head × latency.** One-prefill-K-forks (E7) changes both accuracy and the p80 latency gate. Measure them together.

**Where component-first is right and cheap.**
- Rendering (E5), harness fixes (H), and the gate design for E4 on existing outputs: all $0 or ≤$1, and independent of training.
- Head architecture short-listing on frozen features (E3).
- Teacher-label generation (E11): an offline artifact.

## 11. Could not verify / unknown

- Clef / Clef-flash training data, scale, steps, compute, public-train-split use, RLCD algorithm, knowledge cutoff: **unknown** (not published; HF discussion #8 open).
- Clef-flash on the 0.2.1 board: no primary number. 57.07 is benchlm only [S].
- Jev architecture, data, reward: **unknown**.
- Licences of Perplexity Decider v1.1 and Kev-27B weights: not checked.
- Nimble-9B-v3 training data: undisclosed (v1 documented).
- No published MMLU-Pro / GPQA / BFCL / tool-use numbers for any 0%-attention distilled 7–14B student.
- Whether Index rules permit batch calibration on unlabeled test rows or an external retrieval corpus: not addressed in any text found. Ask.
- A separate Cloudflare post on the DiffusionGemma experiment: not found. The X post linked from the blog was not fetched.
- Our per-benchmark CIs are wide (12–60 rows per benchmark on the subset). Rank the slices by full-DEV Clef numbers where they disagree (e.g. GPQA 14.3 subset vs 35.6 full DEV).

## Sources (fetched or found 2026-10-06)

Primary:
- Cloudflare blog: https://blog.cloudflare.com/clef-decision-models/ (2026-10-01).
- HF cards and code: https://huggingface.co/Cloudflare/clef, https://huggingface.co/Cloudflare/clef-flash; docs https://developers.cloudflare.com/workers-ai/models/clef/.
- djev: https://github.com/mmastrac/djev; vLLM PR https://github.com/vllm-project/vllm/pull/57250 (not fetched).
- TypeSafe: https://typesafe.ai/blog/introducing-system-one-models-and-jev (2026-09-15); https://docs.typesafe.ai/models.
- Board data: https://huggingface.co/spaces/multimodalart/jev-decision-index (`data/v03.json`, `data/index-v0.2.1.json`).
- Kit: https://github.com/apolinario/decision-index (README, docs/engines.md, PRs #5, #15, #30, #36, #37, #42, #54, #55, #58, #74).
- Entrant cards: github.com/bespokelabsai/nimble; huggingface.co/kirp/jpt-9b; github.com/jaredpalmer/kev; github.com/featherless-ai/simple-jev; invergent.ai/blog/rune.
- arXiv: 2505.03005, 2502.14458, 2501.15570, 2408.15237, 2606.16429, 2603.26556, 2406.07887, 2505.15105, 2609.04434, 2508.15884, 2503.13427, 2406.08446, 2607.03801, 2501.17994, 2410.02343, 2509.23782, 2309.17249, 2407.15018, 2403.14715, 2508.00264, 2601.13284, 2405.14838, 2407.06023, 2406.01006, 2407.12854, 2412.09807, 2310.02226, 2502.05171.
- Vendor: https://wiki.rwkv.com/basic/RWKV-Evals.html.

Secondary:
- https://benchlm.ai/benchmarks/clefdecisionindex021 (Clef-flash 57.07);
- 2508.12495 and 2406.01574 (search-summary level);
- news.radio-t.com, digitalapplied.com, orcarouter.ai (context only; no numbers used).

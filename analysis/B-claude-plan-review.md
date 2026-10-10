# B — Adversarial review of `plans/claude/PLAN.md`

> **Rate note (2026-10-09):** the dollar figures here use the 2026-10-06 SF Compute rates ($2.97/h = $0.0495/min batch, $3.96/h interactive) and are left as written. From 2026-10-08 the H100 list price is $5.04/h and batch jobs bill $3.78/h ($0.063/min; Fridays $0.05628/min), so batch figures scale by x 1.27. Source and history: `jobs/common/rates.py`.

Date: 2026-10-06. Reviewer stance: adversarial but fair; the plan had no independent review before this. Everything marked **verified** was checked today against a primary source (HF raw configs/cards, transformers source, arXiv, vendor legal/pricing pages, the Decision Index README). Likelihoods in §6 are my subjective estimates and are labelled as such. I ran no experiments; the arithmetic in §8 is mine and can be re-checked line by line.

Inputs read in full: `plans/claude/PLAN.md`, `reviews/00–04`, `plans/partner/PLAN.md` (only to locate where the claude plan is weaker; the partner plan has its own reviewer).

---

## 0. Verdict in five lines

1. The reframing — "attention-free Clef" = remove 8 of 32 layers from Qwen3.5-9B — is **correct and verified** (`layer_types`: 24 × `linear_attention`, 8 × `full_attention` at indices 3,7,…,31; HF code builds `Qwen3_5GatedDeltaNet` vs `Qwen3_5Attention` from that list). It is the best idea in either plan and the costs for it are realistic.
2. The **"must-hit" target (>57.9, beat Jev and Clef-flash) is mislabelled**. Clef-flash on the *same* 9B backbone, with attention intact and Cloudflare's undisclosed data, lands ≈57. The plan asks to match that head/data recipe at $1,300 while absorbing an attention-removal loss. I put it at 15–25 %.
3. **One hard blocker**: the Jev API as a soft-label teacher violates TypeSafe's Master Customer Agreement §2.3(b) (effective 2026-09-23): "use the Services or any Output to perform model distillation, train a model to imitate the output of the Services, or develop … a similar or competing product". Remove it.
4. **Three arithmetic errors**: Stage 0 is priced at $2–2.5/H100-h but SF Compute autoresearch is $3.96/h interactive and $2.97/h batch (verified today), so the reserve shrinks from 90 h to 20–65 h; the "ECE < 0.02" Stage C gate is below the estimator's noise floor on a 1-in-6 sample of a 5k-row slice (the same mistake the plan itself killed in v1); snapshot forks at chunk 512 store 2–4× more bytes per token than the KV cache they replace, which violates the plan's own definition of attention-free.
5. The plan is honest about tradeoffs and guarantees, but it has **no data plan, no authority/information-flow model, no prospective-intervention protocol, and the backbone stays a black box**. Project 2 is a constrained-decoding RAG planner, not a "controlled LM grown out of SSM-1".

---

## 1. Verification table (claims in the plan → what I found)

| # | Plan claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Qwen3.5-9B = 24 GDN + 8 full-attention layers, `full_attention_interval 4` | **Verified** | `Qwen/Qwen3.5-9B/raw/main/config.json`: 32 layers, full_attention at 3,7,11,15,19,23,27,31; hidden 4096; attention 16 heads × 256, 4 KV heads; GDN 16 key heads × 128, 32 value heads × 128, conv kernel 4; `mamba_ssm_dtype: float32`; vision tower 27 layers nested. Apache-2.0; `Qwen3.5-9B-Base` exists. |
| 2 | "Write the distillation script against `layer_types`" is a small change | **Verified, with a caveat** | `modeling_qwen3_5.py`: `if block_type == 'linear_attention': self.linear_attn = Qwen3_5GatedDeltaNet(...) elif 'full_attention': self.self_attn = Qwen3_5Attention(...)`; GDN uses `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` from the `fla` kernels hub with torch fallbacks; a cache holds `conv_states` and `recurrent_states` per layer. Caveat: attention head_dim 256 vs GDN head dims 128 and a different head count — the new GDN layers cannot be initialised by copying Q/K/V as Mamba-in-the-Llama/LoLCATs do; initialisation needs its own design (fresh init + the MOHAWK stage-2 alignment the plan already names). |
| 3 | Layer-swap distillation precedents preserve knowledge/retrieval | **Partly — knowledge mostly, long-context retrieval no** | RADLADS (COLM 2025, Qwen2.5-7B-Instruct → pure RWKV-7, 450–900M tokens, ~58 MI300X-h): MMLU 68.2 vs teacher 71.7; **RULER 39.7 / 26.1 / 16.5 at 4K / 8K / 16K vs teacher 69.7 at 16K**. Mamba-in-the-Llama (NeurIPS 2024, Llama-3-8B): MMLU 59.3 (50 % attn) / 52.7 (25 %) / 49.2 (12.5 %) / **43.2 (0 % attention)**. LoLCATs (ICLR 2025): Llama-3-8B MMLU 52.8 after linearisation. Jet-Nemotron/PostNAS (2508.15884): "for MMLU only two layers exhibit critical importance; for retrieval two to three" — i.e., a few attention layers carry recall, and in Qwen3.5 those are precisely the 8 the plan removes. Every 2026 SOTA conversion (HALO 2601.22156, KL-guided layer selection ICLR 2026, Opal 2609.31947 "replacing 75 % of attention layers") *ends* at the 3:1 ratio Qwen3.5 already has; none claims retrieval parity at 0 %. |
| 4 | $125 / 200M tokens pilot; $700 full stage (1–2B tokens) | **Realistic** (see §8A) | Pilot compute ≈ 11 H100-h at 35 % MFU vs 50 h budgeted; full stage ≈ 80 H100-h ≈ $220–290 vs $700. RADLADS' 7B conversion used ~58 GPU-h for 700M tokens. One H100-80GB suffices (18 GB weights; ~0.5B trainable GDN params). |
| 5 | RWKV7-G1k 13.3B exists, Apache-2.0, LoRA-tunable on one H100 | **Verified** | HF card: 13,270,298,624 params, 61 layers, 4096/16384, 64 heads × 64, training context 25,600, apache-2.0, HF Transformers format, "Transformers 5.15+", "TRL 1.10+ SFTTrainer", "PEFT LoRA", TileLang inference impl. 26.6 GB bf16 → LoRA fits on 80 GB. Caveat: training throughput of the bundled PyTorch WKV7 path is unverified; a community `-FLA` conversion exists. Quality (RWKV wiki, model labels g0b-13.3b etc., protocol unspecified): MMLU 0.765, MMLU-Pro 0.502 (0.612 CoT), GSM8K 0.923 — real knowledge, well below Qwen3.5-9B on MMLU-Pro-class items. No RULER/NIAH for G1k. |
| 6 | Index rules allow train-split fine-tuning; teacher labels OK | **Verified for the Index; blocked for Jev by TypeSafe** | README: "Do not train on it and do not republish it"; no rule on distillation; no truncation, no option filtering, no prompt tuning, unanswered = wrong, >1,000 ms median excluded; PR must include results dataset, engine/commit, hardware, `"complete": true`. One submission (PR #58 Torchcast) discloses "teacher soft labels from Qwen3.8-Flash-Next". **TypeSafe MCA §2.3(b)** bans distillation/imitation/competing-product development from Jev Output; §4.2 assigns Output ownership to the customer but the restriction stands. Frontier-API teachers (Anthropic/OpenAI/Google) carry equivalent clauses; only open-weight teachers (Clef-27B Apache-2.0, Qwen3.8-27B, Gemma-4) are clean. |
| 7 | Clef-flash ≈ 57.07 (secondary), Jev 57.91, Clef 61.98 PR | **Consistent with review 01** | Not re-fetched. Add: Bespoke-Nimble-9B-v3 (PR #36) = **56.88 with a LoRA on Qwen3.5-9B**, recipe undisclosed; i.e. the 9B backbone with attention is a Jev-class backbone *given the right data*. |
| 8 | "Integrator bank with structural λ=1" and "snapshot forks" | **Not novel as mechanisms; novel only as a decision-model combination** | λ=1 integrator = unnormalised linear attention with the option as the query (Katharopoulos 2020; AFT 2021), or sum-pooled late interaction (ColBERT MaxSim, Kev's pointer head); the exact additive-attribution property is that of self-explaining/additive models (SENN 2018, NAM 2020) and of ABC/Gated Slot Attention for bounded slots. Snapshot forks = SSM state checkpointing (Marconi, MLSys 2025) + softmax over a bounded set of memory slots (ABC, ICLR 2022; GSA, NeurIPS 2024). Say "known components, new application"; do not claim novelty. |
| 9 | Provenance channels (learned channel embedding, trusted gate) | **Prior art exists; evidence is partial** | ISE (ICLR 2025) = exactly "learned role/segment embeddings": +15.8 / +18.7 pts robust accuracy on StruQ / Instruction-Hierarchy benches. ASIDE (ICLR 2026) rotates data embeddings and reports ISE's linear offsets separate poorly in deep layers. StruQ, SecAlign, Spotlighting, CaMeL, Instruction Hierarchy all precede. Lab's own data: spotlighting 15.9 / 64.2 % hijack, SecOPD not better than untrained Gemma, universal GCG steer suffix 31.5 % with 0 alarms. **2606.27567** (Jun 2026) proves "perfect prompt-injection prevention is mathematically impossible" in shared-embedding architectures and demands "architectural separation of instruction and data channels". The plan's gate is in-pipeline; it should be sold as a measured reduction, never as separation. |
| 10 | SF Compute: "assume ~$2–2.5/h ⇒ 210–260 H100-h"; h100-1 shape 14 vCPU / 250 GB RAM / 250 GB disk / 12 h holds | **Rate wrong; shape unverified** | autoresearch.sfcompute.com (today): h100-1 "$0.066/min ($3.96/hr)", 15-min idle window; batch "$0.0495/min per GPU" = $2.97/h (matches partner). Snapshots $0.10/GiB-month with 250 GiB free; sandboxes $0.04572/GiB-h active. No vCPU/RAM/hold-duration figures on the page. $525 ⇒ 133 h interactive / 177 h batch, not 210–260. |
| 11 | Sub-1B entrants score 6–28; no SSM entrant; ECE 0.014 best | **Consistent with review 01** | Not re-fetched. |

---

## 2. Strengths (keep these)

- **The central reframing is right and cheap.** Turning "beat Clef" into "remove 8 layers and re-distil" is the only route under $5k with a non-trivial chance; the HF code makes the swap a config edit plus new-layer init; one H100-80GB suffices; costs have 2.5–4× slack.
- **Backbone × head factorial with Clef's cross-attention head as a priced control**, plus the attribution rule (knowledge gap → backbone; retrieval/tools gap → head) instead of v2's 10-point stop rule. This is what review 02 asked for.
- **Honest guarantee language.** The "what the reviews killed" table is accurate; masks are stated as exact and nothing more; the grounding lemma is scoped; abstention is kept off-Index; the half-life certificate is gone; "no tradeoff" is "pursued, not assumed" with a Pareto map (Stage D) rather than asserted.
- **Steering, not schema violation, as the safety metric**, with injection-augmented training and the lab's harness as the gate. Correct lesson from chl-typed-control.
- **Stage 0 on credits before the $5k**, each module with a kill question; independent re-run before submission (Stage E); spend controls reused.
- **The RWKV7-13.3B arm is a genuine hedge**: it exists, is Apache-2.0, has HF/PEFT support, and its MMLU is real.

---

## 3. Weaknesses and gaps

### 3.1 Target tiering is wrong (highest-leverage fix)
"Must-hit: first attention-free entrant to beat Jev (>57.9) and Clef-flash" fuses a near-certain achievement (first attention-free entrant: no competitor exists) with an improbable one. Evidence: stock Qwen3.5-9B-class readouts land 43–47 (Lux 43.5, JPT-9B 46.9); the best 9B LoRA is 56.9 (Nimble v3, recipe undisclosed); Clef-flash ≈57 with its 128M attention head and undisclosed synthetic data. The plan must (a) reproduce a Jev-class head/data recipe at $1,300 with no data plan, and (b) absorb the attention-removal loss. The literature on (b) says knowledge survives (−3 to −5 MMLU with RADLADS-quality recipes; −16 to −23 with weaker ones) but long-context retrieval collapses (RULER 16.5 vs 69.7 at 16K). Retrieval + tools are 38 % of the Index. Re-tier: Tier 1 (likely) first attention-free entrant above the best non-Clef 9B-class entry with best-in-class ECE; Tier 2 (coin flip) ≥ Clef-flash; Tier 3 (unlikely) > Jev; stretch > Clef.

### 3.2 No data plan
Stage C says "Clef's training recipe on train splits + multi-teacher soft labels + injection augmentation" with no row counts, no generator, no held-out-generator rule (A16), no dedup log against the frozen suite, no budget line. Cloudflare's data scale is undisclosed; Kev-27B used ~146k examples, AutoJev 73k, autotrust ~640k distilled rows. The partner plan has a 10k pilot → 100–300k records and a $900 line. This is the second-largest determinant of the outcome after the backbone and it is unspecified.

### 3.3 Jev-API teacher is contractually barred (see §1 row 6)
Also affects "frontier models through a System One adapter" (review 02 §13 idea) — their ToS carry the same clause. Open-weight teachers only.

### 3.4 Snapshot forks violate the plan's own definition and its memory story
Definition: "no softmax over positions, no KV cache that grows with length, fixed-size recurrent state." Snapshot forks keep S = L/512 full backbone states and take logsumexp (a softmax) over them: a position-indexed, length-growing memory with softmax at chunk granularity. Bytes (§8C): fp32 GDN state ≈ 71 MB per snapshot → 139 KB/token at chunk 512, versus 32 KB/token for the 8 attention layers' KV cache that was removed. Either cap S (fixed-size state, honest) or make snapshots a *head-level* construct (per-option partial sums over chunks, O(S·K) scalars), or raise the chunk to ≥ 4K and say so.

### 3.5 "Exact per-token attribution" is accounting-exact, not causally exact
With increments computed by a LoRA-tuned, state-carrying backbone, the increment at position t encodes evidence from t′ < t. The attribution says *where* evidence was booked, not *which token* caused it — the same leakage problem the plan correctly raises for concept bottlenecks. Needs an erase-token counterfactual test (Δlogit vs booked increment) before "exact" appears in a paper.

### 3.6 Interpretability of the backbone is absent beyond telemetry
"Concept-supervised semantic block with leakage and completeness tests" has no location in the architecture (between final states and head? inside the LoRA?), no concept labels in any data plan, and no budget. "Per-layer Gramians/HSVs" are not defined for GDN's input-dependent matrix state; only the decay spectra (gate statistics) are. The partner plan's prospective-intervention protocol (predict the effect of a record edit *before* running; random and output-only controls; unexplained-remainder reporting) is cheap and absent here.

### 3.7 No authority / information-flow model
Provenance is inferred from the prompt layout; nothing says labels come from trusted caller metadata, that untrusted spans are tokenised with `split_special_tokens` (the lab's own `<|im_start|>` exploit), or that cached recurrent state is never reused across permission sets. The partner plan has all three.

### 3.8 Statistical power of gates (§8D)
- "Within 2 pts of stock on the dev slice": paired SE ≈ 1.0 Index-point at 5k rows (OK, marginal); per-area ≈ 2.3 pts at ~1k rows (useless for the attribution rule the plan relies on).
- "ECE < 0.02" on a 1-in-6 sample of 5k rows (~830 items): a perfectly calibrated model reads 0.027 ± 0.008 at n≈1k (review 02 §9). The gate cannot be passed by a correct model. Same error as v1's PID setpoint.
- "Steer rate below stock baseline": no N, no cluster-bootstrap spec; D1-scale sets (224 cases × 8 templates) resolve only >15 pp differences.
- The long-context retrieval probe has no size, lengths, or threshold.

### 3.9 Stage D (glass-box conversion, $300) is underspecified and underfunded
Each layer swap is a mini re-distillation (Stage B at small scale) plus a full eval card; $300 buys 2–3 swaps at pilot scale. "Certified diagonal/contracting variants" have a measured adherence tax in the lab's own data (operator-norm cap: forbidden choices 9 → 17). Fine as an honest Pareto log; not fine as "the interpretability deliverable".

### 3.10 Q27-AF stretch is infeasible as budgeted (§8B)
27B bf16 = 54 GB; 16 new GDN layers trainable ≈ 1.6B params → Adam states ≈ 19 GB + grads + activations > 80 GB → H200 or 2×H100. Compute ≈ 260 H200-h ≈ $930 for the conversion alone, before re-running Stage C's head factorial on the new backbone (≈ $1,300) and eval. Realistic ≥ $2,200, drawn from a $1,200 reserve the plan also needs for reruns (A17: 2× reserve). Mark it "out of scope at $5k unless external compute appears".

### 3.11 Project 2 is thinner than its title
P2 (weeks 6–11, $1,000) is planner → checker → DFA-masked realizer with copy slots. The LM head is Q9-AF's Qwen LM head, whose generation quality after attention removal is never gated (no perplexity/instruction-following check anywhere). "Rung B free connective text + verifier" is unspecified. There is no language-growth staircase, no semantic interface beyond the DFA, no authority model. Honest about its guarantee scope, but it does not deliver "a controlled, aligned LM grown out of SSM-1"; it delivers a constrained RAG realizer on a frozen converted backbone.

### 3.12 Vision rows
Qwen3.5-9B's vision tower is a 27-layer attention ViT. Index image rows (~1,196 items, ≈1 % of rows) must be declared `Unsupported` (count as wrong) or served through attention. Either is fine; the plan should say which, since "attention-free" is the headline.

### 3.13 "Still Clef's backbone?" and the SSM-architecture claim
Scientifically a strength: Q9-AF vs Clef-flash is a clean ablation of (8 attention layers) × (head design) at matched pretraining. Positioning-wise: 24 of 32 mixers are untouched Qwen GDN layers; the result is "Qwen3.5-9B minus attention + our head". GDN qualifies under the plan's definition (delta-rule matrix state; Qwen's own config calls it `mamba_ssm`), so "attention-free" is defensible; "an SSM we built" is not. Disclose the lineage in the submission and the write-up.

### 3.14 Minor
- Stage A duplicates Stage 0.1/0.2; week numbering restarts after Stage 0 without saying what Stage 0 results feed in.
- "Rank backbones by area; retrieval/tools gap → head problem" conflates the stock letter-logit readout's limits with backbone recall.
- B and B′ "in parallel" cannot run on one h100-1 node under the one-experiment-per-resource rule.
- No hybrid stop rule: if the Clef-head control wins on retrieval with a length-growing gap, the plan has no stated action (review 02: "go hybrid"); under the user's goal the honest action is to log "attention-free falsified at this scale" and ship the hybrid as a separately labelled entry.

---

## 4. Wrong or unverified claims (verbatim → correction)

| Plan text | Status | Correction |
|---|---|---|
| "assume ~$2–2.5/h ⇒ ~210–260 H100-h" (Stage 0) | **Wrong** | $3.96/h interactive, $2.97/h batch (autoresearch.sfcompute.com, 2026-10-06) ⇒ 133–177 H100-h. Modules (112 h) cost $333–444; reserve 20–65 h, not ~90. |
| "h100-1 (80 GB, 14 vCPU, ~250 GB RAM) … 250 GB persistent disk, 12 h holds" | **Unverified** | Not on the pricing page I could read; 250 GiB of *snapshots* are free. Confirm at node creation. |
| "multi-teacher soft labels (Clef-27B local, Jev API ≈$100)" | **Wrong (contract)** | TypeSafe MCA §2.3(b) bars distillation/imitation/competing-model development from Jev Output. |
| "Must-hit: … beat Jev (>57.9) and Clef-flash" | **Mislabelled** | See §3.1. |
| "Pre-registered must-hit: … ECE < 0.02" on the dev slice | **Below noise floor** | ECE floor ≈ 0.027 at n≈1k; ≈ 0.014 at n≈4k. Gate on ≥ 4k rows with a debiased estimator, or gate on NLL/Brier. |
| "Snapshot forks … This fixes the compressed-before-read loss" + definition "no KV cache that grows with length" | **Internally inconsistent** | See §3.4. |
| "per-token, per-option evidence traces (exact for the integrator)" | **Overstated** | Accounting-exact, not causal; see §3.5. |
| "per-layer Gramians/HSVs" for the converted model | **Not defined for GDN** | Report gate/decay statistics; Gramians only for diagonal-LTI swaps in Stage D. |
| "Stretch: Q27-AF … $1,000 (from reserve + P2 if justified)" | **Underbudgeted ≥ 2×, needs >80 GB** | See §3.10. |
| "Distillation is the hinge … the knowledge in that 13B model is real" | **True but incomplete** | RWKV7-13.3B MMLU-Pro ≈ 0.50 (no CoT) vs Qwen-class ≈ 0.7+; knowledge gap on K&R (25.8 %) is large; and no G1k long-context numbers exist. |
| "Integrator bank … Snapshot forks" presented as the plan's own designs | **Prior art** | §1 row 8. |
| "Provenance channels … learned channel embedding" | **Prior art + impossibility result** | ISE/ASIDE; 2606.27567. |

---

## 5. Where the partner plan is stronger (only)

- Acceptance criteria with paired CIs and "interval crossing zero is inconclusive".
- Explicit data plan (10k pilot → 100–300k records; $900 line; dedup; held-out generators).
- Authority/info-flow: trusted caller metadata, public/private/authority state with declared dependencies, no cache reuse across permission sets, state versioning with model/policy/tokenizer.
- Interpretation as an experiment: prospective intervention predictions with random/output-only controls; "unexplained remainder" reported.
- A language-growth staircase (L0–L4) with per-stage gates and a typed semantic program language, rather than a DFA mask.
- Multimodal gate stated explicitly.
- Head B's cost accounting (O(KTd), O(Kd) state) is written down; the claude plan's snapshot-fork cost is not.

Where the claude plan is stronger (for balance): it uses a real 9B backbone instead of a from-scratch 100–400M core that the board shows lands at 2–28; its costs are itemised; it has a Stage 0 with kill questions.

---

## 6. Risk ranking (subjective likelihoods)

| Rank | Risk | P | Consequence |
|---|---|---|---|
| 1 | Q9-AF loses long-context retrieval (RULER-class collapse) and fails the Stage B retrieval probe | 60–70 % | Fallback to RWKV7 arm, which is weaker on K&R; must-hit dead |
| 2 | No attention-free config reaches > 57.9 within $5k | 75–85 % | "Must-hit" missed; plan has no Tier-1 success statement |
| 3 | Jev-API distillation executed as written | 100 % that the clause exists; enforcement unknown | Contract breach; tainted submission |
| 4 | Stage 0 overcommitted at real SF rates | certain (arithmetic) | Reserve 20–65 h; 0.3 or part of 0.4 must move to the $5k phase |
| 5 | ECE gate unpassable at dev-slice n | certain unless n ≥ 4k | False "fail" or silent gate relaxation |
| 6 | Snapshot forks cost more memory than the removed KV cache at chunk 512 | certain (arithmetic) | Definition violated; latency/memory story collapses |
| 7 | Stage C underdelivers for lack of a data plan | 60 % | Head factorial measures head differences on weak data; attribution confounded |
| 8 | Provenance gating leaves adaptive steering ≥ 30 % | 70 % | Safety claim must be "reduced", never "separated" |
| 9 | Q27-AF exceeds $1,000 or does not fit 80 GB | 80 % if attempted | Reserve consumed; reruns unfunded |
| 10 | Q9-AF generation quality degrades unnoticed and P2 inherits it | 50 % | P2 realizer fluency poor; no gate catches it |
| 11 | Backbone remains a black box; "interpretability built in" reduces to the head's additive trace + telemetry | certain as written | "No tradeoff" claim rests on the head only |
| 12 | Vision rows unhandled | certain, ~1 % of rows | Minor score loss; disclosure needed |

---

## 7. Concrete fixes (prioritised)

1. **Re-tier the targets** (§3.1) and pre-register the attribution table (area × head × backbone × length bin) as the primary deliverable; the Index score is one row of it.
2. **Delete the Jev teacher**; use Clef-27B (Apache-2.0), stock Qwen3.8-27B and Gemma-4-31B readouts as teachers, temperature-scaled per teacher; keep Jev strictly as a scored comparator.
3. **Reprice Stage 0** at $2.97 batch (detached jobs) / $3.96 interactive; drop 0.4 to 4 swaps × 100M tokens or move 0.3 to the $5k phase; keep ≥ 40 h reserve.
4. **Fix the gates**: dev slice ≥ 10k rows (≥ 2k per area); paired cluster bootstrap by benchmark; "within 2 pts" as a one-sided paired test; ECE on ≥ 4k rows with the debiased estimator and per-length bins, or gate on NLL/Brier; retrieval probe = NIAH/RULER-style at 1K/4K/16K, ≥ 500 items each, gate ≤ 5 pts at 4K, report 16K; steer gate with N and cluster CIs stated.
5. **Fix snapshot forks**: cap S (e.g., 8) so the state is fixed-size, or implement them as per-option chunk partial sums in the head (O(S·K) scalars), and publish bytes/token next to the KV cache they replace.
6. **Add a data plan** ($400–600): schema/record generator with held-out generators, row counts per area, dedup log against the frozen corpus hash, injection-augmentation share, and a label-review sample.
7. **Define the concept block or drop it**: if kept, it sits between final states and the head with labels derivable from schemas (field types, provenance, evidence spans); add erase-token counterfactual tests for the integrator's attribution; add the partner's prospective-intervention protocol (predict → run → score, with random and output-only controls).
8. **Add authority/info-flow minimums**: provenance from trusted metadata; `split_special_tokens` on untrusted spans; no state/cache reuse across permission sets; state versioned with model/policy/tokenizer.
9. **Add an LM-quality gate for Q9-AF** (held-out perplexity, a small instruction-following set) before P2 reuses its LM head.
10. **Add a hybrid stop rule** and the honest write-up it triggers.
11. **Q27-AF**: out of scope at $5k; state the H200/2×H100 requirement and ≥ $2.2k cost; never fund it from the rerun reserve.
12. **Disclose lineage** in the submission ("Qwen3.5-9B with its 8 attention layers distilled into Gated-DeltaNet layers; 24 GDN layers inherited; vision rows Unsupported").
13. **Minor**: merge Stage A into Stage 0.1/0.2; serialise B/B′ on one node; label Gramian/HSV telemetry as Stage-D-only.

---

## 8. Arithmetic appendix

**A. Q9-AF compute.** Text params ≈ 8.5e9 (9B incl. ~0.5–1B vision). Pilot 2e8 tokens: student 6·N·D = 1.0e19 FLOP + teacher forward 2·N·D = 3.4e18 → 1.4e19; at 350 TFLOP/s sustained (≈35 % of 989) → 3.9e4 s ≈ 11 H100-h vs 50 h budgeted (slack for 8 evals + overhead). Full stage 1.5e9 tokens → 1.0e20 FLOP → ≈ 81 H100-h ≈ $218 (community $2.69) / $283 (secure $3.49) / $241 (SF batch $2.97) vs $700 budgeted. Frozen non-GDN layers reduce backward further. Trainable new GDN params ≈ 8 × 67M ≈ 0.54B (in_proj_qkv 4096×8192, in_proj_z 4096×4096, out_proj 4096×4096, conv, a/b) → Adam ≈ 6.5 GB; fits one 80 GB card with checkpointing.

**B. Q27-AF.** 54 GB weights bf16; 16 GDN layers ≈ 1.6B trainable → ≈ 19 GB Adam + 3 GB grads + activations > 80 GB → H200 ($3.59/h) or 2×H100. 1.5e9 tokens: 2.4e20 + 8.1e19 = 3.2e20 FLOP → ≈ 257 GPU-h ≈ $925 before Stage-C head retraining (≈ $1,300) and eval (~$100). Total ≥ $2,200.

**C. Snapshot-fork memory.** GDN recurrent state per layer = 32 heads × 128 × 128 = 524,288 scalars + conv state ≈ 33k; × 32 layers (post-conversion) ≈ 17.8M scalars = 71 MB fp32 (`mamba_ssm_dtype: float32`) / 36 MB bf16 per snapshot. S = L/512 → 139 KB/token fp32 (70 KB bf16). Removed KV cache: 8 layers × 2 × 4 KV heads × 256 × 2 B = 32 KB/token. At 64K: 9.1 GB (fp32 snapshots) vs 2.1 GB (KV). Chunk 4096 brings snapshots to 17 KB/token.

**D. Gate power.** Paired difference on N rows with disagreement rate p_d: SE_raw ≈ √(p_d/N); Index scale divides by (1−chance) ≈ 0.6. N=5k, p_d=0.2 → 0.0063/0.6 ≈ 1.05 pts; N≈1k per area → ≈ 2.3 pts. ECE floor for a perfectly calibrated model (15 equal-width bins, review 02): n=256 0.057±0.015; 1024 0.027±0.008; 4096 0.014±0.004. A 1-in-6 sample of 5k rows ≈ 830 → floor ≈ 0.03 > 0.02 gate.

**E. Stage 0 money.** Modules 4+8+6+50+40+4 = 112 H100-h. At $2.97: $333 → reserve $192 ≈ 65 h. At $3.96: $444 → reserve $81 ≈ 20 h. Plan assumed ~90 h reserve.

**F. Budget shape.** P1 $2,800 + P2 $1,000 + reserve $1,200 = $5,000. Stage F "from reserve + P2" would leave < $200 for reruns and the independent re-run, violating the lab's own A17 (2× reserve) rule.

---

## 9. Sources checked today

- https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json ; https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md (Apache-2.0; `Qwen3.5-9B-Base`; "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))")
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://huggingface.co/RWKV/RWKV7-G1k-13.3B-20260930/raw/main/README.md ; https://wiki.rwkv.com/basic/RWKV-Evals.html ; https://wiki.rwkv.com/RWKV-Fine-Tuning/LoRA-Fine-Tuning.html
- RADLADS: https://arxiv.org/html/2505.03005 (Tables 2, 3, 5, 11) ; Mamba-in-the-Llama: https://arxiv.org/html/2408.15237 ; LoLCATs: https://arxiv.org/abs/2410.10254 ; Jet-Nemotron/PostNAS: https://arxiv.org/html/2508.15884v1 ; HALO: https://arxiv.org/abs/2601.22156 ; KL-guided layer selection (ICLR 2026): https://arxiv.org/pdf/2512.20569 ; Opal: https://arxiv.org/abs/2609.31947
- ISE: https://arxiv.org/pdf/2410.09102 ; ASIDE (ICLR 2026): https://arxiv.org/pdf/2503.10566 ; Inseparability: https://arxiv.org/abs/2606.27567
- Marconi: https://arxiv.org/pdf/2411.19379 ; ABC: https://arxiv.org/pdf/2110.02488 ; Gated Slot Attention: https://arxiv.org/pdf/2409.07146
- TypeSafe MCA: https://typesafe.ai/legal/mca (effective 2026-09-23, §2.3(b), §4.2)
- Decision Index README: https://raw.githubusercontent.com/apolinario/decision-index/main/README.md ; PR #36 (Nimble-9B-v3 56.88): https://github.com/apolinario/decision-index/pull/36 ; PR #58 (teacher soft labels disclosed)
- SF Compute autoresearch pricing: https://autoresearch.sfcompute.com/
- Not re-verified (taken from reviews/01–04): board scores, Clef-flash ≈ 57.07, prior-lab steering numbers, RunPod prices.

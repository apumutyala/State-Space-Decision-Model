# SSDM direction memo v0 — opinion, evidence, pathways (chief of staff, 2026-10-14)

Inputs: repo `agent-main` + `state` branch; five research workstreams (W1 landscape, W2 capability, W3 alignment, W4 architecture/compute, W5 post-mortem — full reports alongside this memo); one new experiment run today (reasoning headroom, §3). Tags: **MEAS** measured (ours or a cited receipt), **EXT** external source, **SIM** arithmetic, **INF** judgment.

## 0. Bottom line

1. **Yes, SSDM is a Jev/Clef alternative**: an open "System One" decision model (state + typed questions → calibrated probability per option, one forward pass). Its stated edge was state-space architecture plus control-theoretic guarantees. After 8 days and ~$773 of $785 credit, **every capability target is missed** (Decision Index: −7.8 vs Clef-flash; JevBench local Composite: best of ours −9.4 vs H2O; our trained 4B cells score *below* the untrained base). Budget left: $11.67. (MEAS, W5)
2. **The architecture is not the moat and never was.** The leaders are thin post-trains of strong open bases; Qwen3.5 (H2O, Clef-flash, MD1's base) is *already* a 3:1 Gated-DeltaNet/attention hybrid SSM. 73% of documented spend went to an attention-free 13.3B line that the later Composite objective could not use. (EXT W1/W4, MEAS W5)
3. **New result today: the intelligence we lack is already inside the 4B hybrid base — the one-pass format throws it away.** On the 231 public JevBench items, stock Qwen3.5-4B scores I 54.3 in one pass and **I 84.1 / C 87.2 when allowed to think first** (+29.9 I, 95% CI [+21.1, +38.0]); H2O-Lightning-4B (the open Composite leader) scores 77.5 / 82.8 in one pass. Thinking is not deployable as-is (Speed and Cost collapse: Composite 18), so **the capability program is: distill deliberation into the single pass, and deliberate only when uncertain.** (MEAS, §3)
4. **"Far more aligned" is the one axis where incumbents are both weak and silent.** Independent attacks redirect 61–73% of correct decisions across four decision systems, Jev included; DecisionRobust target-ASR (selection-biased set): Jev 79%, Perplexity 57%, H2O 37%, OpenAI 28%; Clef and MD1 publish no steering numbers; JevBench's yes/no rule *rewards* forcing confidence (H2O's 0.801 floor plus temperature is worth +23.7 I — we reproduced it today: 53.8 → 77.5). SSDM owns real assets here: exact reorder/relabel invariance by construction (0/565 flips on GPU), a trusted/untrusted stream construction, cluster-valid certification machinery, and a measurement discipline better than any decision-model paper W3 found. (EXT W3, MEAS)
5. **Recommended north star:** *a 4B hybrid-SSM decision model that knows when to think and cannot be talked into a decision* — one-pass speed by default, distilled from its own deliberation; bounded deliberation or deferral when uncertain; exact consistency; measured steering resistance with evidence retention. Composite/Capability become constraints we report, not the only goal.
6. **Answer to "just take Quyet, post-train and/or quantize?"** Legally and technically yes (Apache-2.0, ungated since Oct 10; Quyet itself is Gemma-4-31B + a rank-16 LoRA). As our main line, no: it is a dense transformer (breaks the SSM identity and the operator's Oct 11 directive); a 31B base pays 31B prices on JevBench Cost whatever the precision, so even with FP8 serving and recalibration it needs +1–2 board I just to tie H2O's Composite (+5 at today's speed); and continued training of leaders has been flat for us twice. Quantization helps our serving bill, not the score. Details §4. The better version of your idea is §3's: post-train an open *4B hybrid* base on its *own* deliberation.
7. **Faster/cheaper is an operating-model fix, not a hardware fix**: one lane, one hypothesis per cell, ~$1–5 evaluation runs and ~$3–8 short training cells on per-second GPUs (today's 4-model probe: 9 minutes wall, ≈$3), confirmatory-only preregistration, a sealed lockbox, and a freeze on Track L / Q9-AF / RWKV-13.3B / Clef-chasing.

## 1. What we have (keep) and what failed (stop)

| Keep (reusable, proven) | Stop / freeze |
|---|---|
| `jevkit`: reproduces the JevBench v1.6.1 board arithmetic exactly (re-verified here, sha256 `5cd8c133…`) and 3 open-weight rows; 89/89 tests pass locally | Track L (language growth): no qualified parent, hundreds of commits, no qualified level |
| Exact reorder/relabel invariance (sym-fork / answer-slot readout; GPU 0/565 flips) | Q9-AF attention→GDN conversion: KL fine, recall collapsed (Index 35.8 → 14.1) |
| Leak gates (doc-overlap, attack, fingerprint) — caught real leaks four times | RWKV7-13.3B line ($493 documented): Cost-gated for JevBench, 7.8 behind Clef-flash on the Index |
| Cluster-valid LTT/CRR certification code (on `claude/ev-guarantee-proofs`, not yet on agent-main) | Clef/Decision-Index chase as a primary target; H2O-host data cells (flat twice) |
| Sealed `jev-heldout` lockbox: 2,199 items, ~3× tighter CIs than public-231 | Rule-engine/template training data as the capability lever (W2: generator cues, no transfer) |
| Trusted-stream (G15) construction, real-BPE tested | Per-experiment prose preregistration for exploratory runs; bespoke ≤899-min job runner |

Where the money went (W5, documented $675 of $773 consumed): RWKV7 cells/evals $492.90; Q9-AF $70.55; JevBench Qwen cells $68.48; baselines $25.88. Scientific negatives $330; infrastructure failures/invalid measurements $93. The scarce resource was informative experimental cycles, not GPU price.

## 2. The market as of today (W1, W3, W4)

- **Two headlines.** The live open-weights JevBench board now headlines **Capability** = mean(I, C) within caps ($0.0646/1k, adjusted p50 1.23 s): Quyet-1.0-Large 81.7 leads. The Composite view: H2O-Lightning-4B 72.5 (open), Sage 74.0 (API). Decision Index 0.3.2: 166 entries, d3 66.26 first; ranks move with new entries, not improvements.
- **What wins:** strong open base + thin LoRA + letter/label readout + temperature refit + teacher-authored, verified scenario data (H2O, JevK5, Kev-27B, Perplexity, Nimble). Frozen-base readouts with FP8 (decisio, deck) reach Capability 77–80 with no training at all.
- **Composite arithmetic** (SIM, at H2O's point): one I point = 0.365 Composite, one Cost point = 0.362, one Speed point = 0.153, one C point = 0.16. Zero latency adds at most +0.57 (the board adds ×2 + 0.15 s to self-hosted latency). Halving priced input tokens adds +2.96. Cost uses the **base model's reference token price**: 4B $0.03/M in, $0.15/M out; 9B $0.10/M; Gemma-4-31B $0.09/M. A 9B at 600 tokens cannot reach 72.5 even at I = 100.
- **Public-231 does not rank leaders:** H2O (205/231) has board I 60.0; JevK5 v0.3 (≈203/231) has board I 38.5. Local numbers guide; only a lockbox or the board confirms.
- **Safety is not vacant but unmeasured where it matters:** MD1 reports 5,250 safety requests across 11 unnamed benchmarks with no per-benchmark numbers; Clef's card has no safety section; steering is the live failure (decision outputs are typed, so the attack is *steering within allowed options*, not jailbreak text).

## 3. New evidence: reasoning headroom on a 4B hybrid (MEAS, today)

Protocol: H2O's published shim prompt and label-token readout (prompt content identical across arms); arm *one-pass* = thinking off, one forward pass; arm *think* = same turn with thinking on (temperature 0.6, budget 3,072 tokens, budget-forced close), then the same "Answer:" label read; shim post-processing (T 0.8, yes/no floor 0.801) for both; scored by the frozen jevkit CLI on the 231 pinned public items (EVAL-ONLY: nothing trained, tuned or selected on them). vLLM 0.30.0, one H100 per model on Modal, single sampling seed. Validation: our H2O one-pass arm reproduces H2O's published public card exactly (205/231 correct; I 77.5 vs the repo's kit-parity 77.6).

| Model (board price class) | Arm | I | C | S | K | Composite A | Correct /231 | Think tokens (mean / median) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | one-pass | 37.0 | 58.3 | 94.1 | 66.5 | 31.3 | 156 | — |
| Qwen3.5-2B | think | 68.8 | 72.5 | 59.2 | 30.8 | 19.5 | 190 | 1,894 / 2,459 (103 truncated) |
| **Qwen3.5-4B** | one-pass | 54.3 | 66.0 | 93.1 | 61.2 | 66.0 | 170 | — |
| **Qwen3.5-4B** | **think** | **84.1** | **87.2** | 62.4 | 29.1 | 18.3 | **214** | 1,410 / 619 (67 truncated) |
| Qwen3.5-9B | one-pass | 61.2 | 75.8 | 93.4 | 45.5 | 53.3 | 178 | — |
| Qwen3.5-9B | think | 83.3 | 88.4 | 61.1 | 14.3 | 3.0 | 218 | 1,311 / 609 |
| H2O-Lightning-4B v1.1 | one-pass | 77.5 | 82.8 | 93.4 | 61.2 | 76.9 | 205 | — |
| H2O-Lightning-4B v1.1 | think | 83.7 | 86.6 | 80.0 | 36.9 | 34.5 | 210 | 716 / 8 |

Paired (same items, B = 1,000, seed 16): stock-4B think − stock-4B one-pass ΔI **+29.9 [+21.1, +38.0]**; stock-4B think − H2O one-pass ΔI +6.3 [−4.6, +15.7], ΔC +6.2 [−1.3, +13.4], ΔCapability +6.3 [−2.0, +14.2]. The hard tier carries it: 60 → 94 of 111 correct (judge_hard 10→17/17, multi_hop 10→16, temporal_numeric 2→8/15, probability 4→8, long_policy 10→14).

Reading (INF):
- **The capability gap to the leaders is a format gap, not a base gap.** A stock Apache-2.0 4B hybrid already "knows" more than H2O's single pass on these items; our one-pass training never tapped it.
- **Deliberation cannot ship raw:** p50 2.4 s raw (adjusted 5 s), 1,410 output tokens at the board's output price → Composite 18 and outside both Capability caps. 9B adds nothing over 4B when thinking (83.3 vs 84.1). Thinking only on low-confidence items (gate sweep on these outputs) keeps p50 at one-pass speed but output tokens still push Cost down (H2O + gate at τ 0.6: Capability 80.2 → 83.5 locally, cost $0.046/1k inside the cap, Composite 76.9 → 73.3).
- **So the lever is deliberation→single-pass distillation** (the model's own thinking, or a stronger open teacher's, as soft labels on train-split items), with a small, calibrated deliberation budget for what does not distill (temporal arithmetic). This is W2's "teacher-authored, thinking-verified" recipe, made cheap: the teacher is the Apache-2.0 base itself; labelling 20k rows ≈ 1–2 H100-hours.
- Caveats: n = 231, one seed, public→board gap unknown for this arm; temporal_numeric still fails half the time even with 3k-token thinking; how much of the +30 survives distillation is the open question the first experiment answers.

## 4. "Can we just take Quyet and post-train and/or quantize it?"

| Question | Answer | Evidence |
|---|---|---|
| Allowed? | Yes: Apache-2.0, ungated since 2026-10-10; keep the NOTICE | HF API, card (EXT) |
| What is Quyet? | Gemma-4-31B-it + merged rank-16 LoRA, letter-readout prompt, refit temperatures: the base does the work | Quyet card (EXT) |
| Does it fit SSDM? | No: dense attention, no SSM; conflicts with the hybrid-SSM main line and the operator's 2026-10-11 directive | GOAL.md, BOARD directive |
| Composite (GOAL #1) | Hard at 31B: Cost K 50.5 is priced from the base's $0.09/M whatever the precision. At Quyet's C 90 / S 86.9 it needs board I ≈ 78.5 to pass H2O's 72.5 (best I on the board: 73.4) and ≈ 86 to pass Sage's 74.0; even with FP8 serving (S ≈ 90) and C 92 it lands ≈ 72.2, i.e. it needs +1–2 I to tie H2O inside board noise | SIM from board JSON |
| Capability (open headline) | Quyet already leads (81.7). A better-calibrated Quyet could edge it, but that is Quyet's result, not ours | board JSON |
| Quantize? | Changes nothing on JevBench Cost (base-model reference pricing: René-1 31B FP8 K 46.0; decisio's FP8 priced as Gemma-4-31B). Speed gain ≤ +0.6 Composite. Useful for *our* real serving bill and cheap dev hardware, not for the score | board JSON (EXT), W4 SIM |
| Expected gain from continued training | Quyet: +1 I (80% range −3 to +5) at $40–100/cell; H2O: ≈0 (our two runs: −0.41, −0.97) | W2 estimate, MEAS |
| Where it is useful | As a **control arm**: run our recipe once on a leader's weights to measure what the recipe adds | INF |

## 5. What "far more aligned" should mean (W3, adopted with edits)

A decision is aligned when it follows the **trusted** policy/rubric and evidence — not wording, option names, order, or instructions planted in untrusted content — carries a probability you can act on, and escalates instead of executing silently when it cannot. Measured as a **Decision Alignment Card**, always with an evidence-retention control (otherwise "ignore the state" wins every robustness metric):

| Axis | Metric | Target for "far more aligned" |
|---|---|---|
| A1 Steering | adaptive natural-context steer ASR on clean-correct units | ≤ 5% (CI upper ≤ 8%) and ≥ 10 pp below each comparator we can run |
| A3 Evidence retention | benign evidence that should flip the answer still does | ≥ 90% of comparator |
| A4 Consistency | 8-operator flip rate; reorder/relabel bitwise | ≤ 1.3% (MD1's reported level), reorder/relabel exactly 0 |
| A5 Rubric fidelity | definition-level rebinding flips | ≤ 1% |
| A6 Act/defer | cluster-valid certified act-rate at ε = 5% on an unseen split; confident-wrong rate | ≥ best comparator |
| A7 Harm vs over-refusal | operating curve on moderation items | no regression |
| Honesty | raw (unfloored) probabilities reported beside any benchmark-compatible floor | always |

Claims to stop making (W3 audit): "injection-robust conformal" as a general property (with state untrusted it is a prior-only set: mean 3.1 of 4 labels in simulation); the G16 veto as model alignment (it is a host-side mask any vendor can apply); "0 flips like MD1" (our 8-op rate is 3.6–4.1%); static-template robustness as "more robust than Clef".

## 6. Pathways

| | Pathway | What it is | Tranche cost | Time to first verdict | What success proves | Verdict |
|---|---|---|---|---|---|---|
| **P1** | **Deliberation-distilled 4B** (capability core) | Qwen3.5-4B hybrid; answer-slot one-pass readout with exact invariance (JEV3-LM-S lineage); LoRA on train-split rows relabelled by the base's own thinking (k samples → soft labels), later a stronger open teacher; calibrated gate to a bounded think budget | $40–80 | 1 week | One-pass I closes most of the +30 → H2O parity or better at H2O's price | **Do** |
| **P2** | **Steer-bounded, consistent decisions** (alignment core) | DAC battery from public attack suites + evidence-retention controls; steer negatives + retention positives in training with νPI ON/OFF (capability-regression constraint); trusted-stream variant only on agent-control workloads where trusted fields carry the decision | $40–80 | 2 weeks | A measured, paired steering gap vs H2O/Clef at equal capability; exact consistency | **Do, coupled to P1** |
| P3 | Fast-follower on an open leader | Continue-train Quyet or H2O; quantize | $40–100/cell | 1 week | A leaderboard number that is mostly someone else's model | **Control arm only** (1 cell, if the directive allows) |
| P4 | Attention-free research line | RWKV7-G1k 2.9B/7.2B for mechanism work (stability, νPI, interpretability), never the product | ≤ $20 | — | Science, not standings | **Minimal / frozen** |
| P5 | Assurance product | jevkit + DAC battery + certification as a vendor-neutral audit of decision models | ≈ $0 training; comparator runs $5–30 | 1–2 weeks | Market need for audits (no evidence yet of willingness to pay) | **Opportunistic** (needs publication approval) |

Recommended program = **P1 + P2 as one system**: "a System 1 that knows when to think — and cannot be talked into a decision." The same calibrated-uncertainty signal routes hard items to deliberation and suspicious items to deferral, so capability and alignment share one mechanism instead of trading off.

## 7. First two weeks (gated; ≈ $150–250 total)

| Step | What | $ | Gate / kill rule |
|---|---|---|---|
| 0 | Unblock data: training draws (jev1/2/3), `jev-heldout` and artifacts live only on the operator's laptop (`ssdm-data`), not in git — get a copy or rebuild from public train splits; merge the cluster-valid LTT fix; port the DAC battery v0 from public suites | 0–5 | — |
| 1 | **D2S-v0**: relabel ~20k leak-gated train rows with Qwen3.5-4B thinking (k = 4 samples → soft labels); LoRA r64 on the answer-slot readout, 1 epoch, 2 seeds; score public-231 (eval-only) + `jev-heldout` once | 25–40 | Kill if one-pass ΔI < +8 over stock on public **and** Δ ≤ 0 on the lockbox |
| 2 | Gate sweep (one-pass → bounded think on low confidence): report Composite *and* Capability-within-caps | 2–5 | Report only; no selection on public items |
| 3 | DAC steer battery v0 on stock one-pass, stock think, D2S-v0 (+ H2O/Clef if exempted) | 3–10 | Alignment thesis on hold if no ≥10 pp steer gap at equal evidence retention |
| 4 | D2S-v1: add steer negatives + retention positives with νPI ON vs OFF; stronger open teacher (DeepSeek-V4 / GLM-5.x) for items the 4B's own thinking misses | 40–80 | Capability regression > 1 I point with νPI ON ⇒ drop the constraint form |
| 5 | Confirm: 2 seeds, lockbox once, independent rescore; decide on a JevBench submission (needs approval) | 10–20 | — |

## 8. Operating model (faster, cheaper, and the organisation aligned to one goal)

- **One north star, one lane.** Chief of staff sets the board; worker agents run one hypothesis per cell; no parallel descendants of an unvalidated runner; the lead/evidence split survives only as "independent rescore before any claim".
- **Per-second GPUs** (Modal is configured on this machine; RunPod/Vast are cheaper for long cells) with the vLLM harness from today's probe: evaluation runs cost $1–5 and finish in minutes. Short LoRA cells $3–8. CPU admission tests on the real production path before any paid cell.
- **Evidence tiers:** exploratory = a one-line run card (hypothesis, kill rule, cost cap); confirmatory = preregistration, lockbox, two seeds, independent rescore. Leak gates and pinned scorers always.
- **Budget gates:** tranche of ≈ $200 with a daily cap; next tranche only on a passed gate. Apply now for Modal academic (up to $10k) and Lambda research (up to $5k) credits.

## 9. Decisions needed from you

1. **North star:** keep GOAL.md's order (Composite → Capability → MD1 → Clef), or adopt §6's "knows when to think, cannot be talked into a decision" with Composite/Capability as reported constraints *(recommended)*?
2. **Directive scope (operator's 2026-10-11 rule):** allow (a) running open leaders (H2O, Clef-flash, Quyet) and the MD1 API on our alignment battery ($1–10), (b) one fast-follower control cell on H2O weights? *(recommend a: yes; b: one control cell)*
3. **Budget and provider:** approve tranche 1 (≈ $200) on Modal or another provider; the SF Compute credit is exhausted.
4. **Freeze list:** Track L, Q9-AF, RWKV-13.3B, Decision-Index-vs-Clef chase, H2O-host data cells.
5. **Data access:** ask Apuroop for `ssdm-data` (training draws, lockbox, artifacts), or approve rebuilding from public train splits.

## 10. What would change this plan

- D2S-v0 recovers < 25% of the thinking gain on the lockbox → single-pass distillation fails at 4B; capability shifts to bounded deliberation within the Capability caps or to a stronger external open teacher; the Composite target is declared unreachable at this budget with a gap attribution.
- The steer battery shows H2O/Clef already ≤ 10% adaptive steer ASR → the alignment wedge narrows to consistency + act/defer (T2/T3).
- A board submission shows the public→sealed gap erases local gains → measurement-first: build a larger private lockbox before training more.

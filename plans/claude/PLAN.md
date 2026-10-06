# Two-project plan: attention-free System-1 decision model → controlled LLM

Date: 2026-10-05. Budget: $5,000 total. Evidence: `reviews/01–04` (Fable 5.1 fact-check, technical pressure test, prior-lab synthesis, session triage). All costs below are estimates from live RunPod prices (H100 SXM $2.69/h community, H200 $3.59/h, RTX PRO 6000 $1.69/h).

## Bottom line first

1. **The goal is reachable in a sharper form than v1/v2 assumed.** Clef's own backbones (Qwen3.5-9B, Qwen3.8-27B) are already 75% recurrent: 3 Gated-DeltaNet layers per attention layer (24+8 and 48+16). An "attention-free Clef" means converting **8 attention layers**, not building a new model. That converts "beat Clef" from a pretraining problem (impossible at $5k) into a layer-distillation problem (≈$500–1,500).
2. **Honest targets.** Official board (Decision Index 0.2.1): Jev 57.91, stock Gemma-4-31B 57.33, Clef-flash ≈57.07 (secondary source), Clef 61.98 (pending PR), Torchcast-27B 65.00 (pending, claimed). No SSM/RWKV/xLSTM entrant exists.
   - **Must-hit:** first attention-free entrant to beat Jev (>57.9) and Clef-flash, with best-on-board calibration (ECE < 0.014) and a measured injection-steering rate.
   - **Stretch:** beat Clef (>62) via the 27B conversion. Beating 65.0 is not promised.
3. **v1's from-scratch 10M–300M ladder is dead.** Every sub-1B entrant scores 6–28. Kept only as the small, fully glass-box end of the Pareto map.
4. **"No tradeoff" is pursued, not assumed.** Every config reports the same four numbers together: Index score, ECE, injection-steer rate, and interpretability measures. The plan picks interpretability mechanisms that are exact *and* free in capability where they exist, and measures the tax where they aren't.

## What the reviews killed or corrected (so we don't rebuild it)

| v1/v2 claim | Verdict | Replacement |
|---|---|---|
| "Calibration is a property of the integrator architecture" | Wrong (simulated: ECE 0.14 with true LLRs under redundant evidence) | LS-CE + Brier (Clef recipe) + post-hoc per-question temperature with a length feature; debiased ECE |
| Forked continuations are permutation-equivariant and adequate | Breaks with questions-first ordering; lossy on retrieval/tools (~38% of Index weight) | Snapshot forks (fork options from S chunk states, logsumexp); Clef cross-attention head only as a *control* arm |
| "Head reads only h_sem ⇒ provable interpretability" | Vacuous | Concept-supervised block + leakage test + causal probes + completeness test on h_free |
| Per-mode "injection half-life" certificate | Correct per layer, useless as safety (16K memory needs ≥16K half-life; per-step bounds fail for switching systems — prior lab: ρ=0.99/step, ×8,994 in 100 tokens) | Provenance-gated writes + measured steer/hijack flip rates |
| PID loop on ECE, setpoint 0.02 | Below estimator noise (perfect model: 0.057±0.015 at batch 256) | Post-hoc temperature + debiased ECE on large held-out sets |
| LLC/patterning-MPC as control sensors | Prior lab results: SGLD biased 1.3–3.3×, inestimable at 1.7B; policy "log, never gate" | H-SLT-1 kept as a $0 CPU side paper (corrected formula below) |
| Hard grounding theorem = "no hallucination" | Sound syntactic lemma only | State the guarantee exactly; measure selection/omission errors separately |
| CBF filter on latent state | Math right; can shift logits by 0 | Output masks/caps (exact, free) for the decision model |
| Distil Clef-flash then beat it | Distillation ceiling | Ground-truth train splits + multiple temperature-scaled teachers (Clef-27B, Jev API) |
| Jev 444.6× cheaper; 132,422 decisions; equal-weight areas | Wrong edition/number | 244.6×; 0.2.1 = 120,340 rows, 38 benchmarks, area weights knowledge 25.8 / language 25.8 / retrieval 20 / tools 18.3 / arts 10 |
| "Index is zero-shot" | Not a rule | Train-split fine-tuning is allowed; prompt tuning, truncation, option pruning and test-item training are not |

Key prior-lab evidence the plan builds on (chl-typed-control, 17 models): output masks are exact (0 attacker-tool picks), but **within-mask steering** (attacks moving the model to a wrong *allowed* option) runs 31–69% with zero alarms. Clean-only LoRA *increases* hijackability. Injection-augmented LoRA cut hijacking to 0.3%. Steering, not schema violation, is the safety metric that matters.

## Definition: "attention-free SSM"

No softmax over positions, no KV cache that grows with length, fixed-size recurrent state. Diagonal selective SSMs (Mamba-1/3), delta-rule matrix-state recurrences (Gated DeltaNet, RWKV-7) and mLSTM all qualify. A diagonal-only arm (Falcon-Mamba-7B) is kept so we can report what the glass-box-friendly class costs.

---

## Project 1: SSM-1, an attention-free System-1 decision model

**Output contract:** drop-in Jev/Clef `POST /v1/systemone` (noul / choice / score, per-question softmax, `confidence`, `probabilities`), so it runs in the Decision Index harness and the TypeSafe evals unchanged.

### Architecture

1. **Backbone (chosen by Stage A):**
   - (a) **Q9-AF**: Qwen3.5-9B with its 8 attention layers distilled into Gated-DeltaNet layers. This is the Clef-flash backbone made attention-free.
   - (b) **RWKV7-G1k 13.3B** (pure recurrent, Apache-2.0).
   - (c) Falcon-Mamba-7B, the diagonal arm.
   - Adapters are rank-256 LoRA, matching Clef.
2. **Ordering:** schema and options → state → questions re-read. Options known in advance let the head keep O(K) running statistics instead of storing O(L) features. This is the fundamental trade-off the review identified.
3. **Decision head, attention-free, 3 variants compared:**
   - **Integrator bank** with structural λ=1: each option's logit is an exact sum of per-token increments, so per-token attribution is exact by construction.
   - **Snapshot forks:** S chunk-state snapshots, each option scanned from each, combined by logsumexp. This fixes the compressed-before-read loss.
   - **Final-state fork** as the cheap baseline.
   - Clef's 128M cross-attention routing head is used as a *control*, to price what attention in the head buys.
4. **Typed outputs:** sigmoid for noul, softmax for choice, a cumulative-link head with monotone thresholds for score. A CRF with hard schema factors handles multi-field records; the review found the exact-inference code sound but low leverage, so it is used only where fields interact.
5. **Calibration:** LS-CE + Brier, post-hoc per-question-type temperature with a length feature, and a 3-member LoRA ensemble for epistemic signal if budget allows.
6. **Alignment / control, built in from step one:**
   - **Provenance channels:** tokens from `state` (untrusted data) vs. `schema/instructions` (trusted) get a learned channel embedding. The head's option-scoring increments from the untrusted channel are gated by a trusted-channel-computed gate. Trained with injection-augmented data (prior lab: 0.3% hijack vs 25% clean-only).
   - **Output masks:** the schema plus allow-lists make invalid actions impossible. This is exact, so we state it as such and nothing more.
   - **Selective deployment mode, off-Index:** LTT/CRC threshold on confidence for act/abstain. Report risk-coverage separately, because the Index scores abstentions as wrong.
7. **Interpretability deliverables:**
   - per-token, per-option evidence traces (exact for the integrator);
   - a concept-supervised semantic block with leakage and completeness tests;
   - per-layer Gramians/HSVs and decay spectra as descriptive telemetry, not certificates;
   - a layer-by-layer "glass-box conversion" log (Stage D).

### Stages, gates, spend

| Stage | Weeks | What | Gate / decision | Est. cost |
|---|---|---|---|---|
| **A. Backbone triage** | 1 | Stock readout (option-count temperature, as in the Gemma 57.33 entry) on a stratified Index dev slice for Qwen3.5-9B, RWKV7-13.3B, Falcon-Mamba-7B, xLSTM-7B. Set up the harness and Clef-flash locally. | Rank backbones by area. Knowledge gap → backbone problem; retrieval/tools gap → head problem (attribution rule, replaces v2's 10-point stop) | $250 |
| **B. Attention removal** | 1–3 | Distil Qwen3.5-9B's 8 attention layers into GDN layers, MOHAWK-style: layer-wise output matching, then end-to-end KL, ~1–2B tokens. Run in parallel with B′: LoRA+head on RWKV7-13.3B | Q9-AF within 2 pts of stock Qwen3.5-9B readout on the dev slice and on a long-context retrieval probe; otherwise fall back to RWKV7 | $700 |
| **C. Head × backbone factorial** | 3–5 | 2 backbones × 3 attention-free heads (+ Clef-head control), Clef's training recipe on train splits + multi-teacher soft labels (Clef-27B local, Jev API ≈$100) + injection augmentation | Pick a config. Pre-registered must-hit: dev-slice score > Jev-equivalent, ECE < 0.02, steer rate below the stock-backbone baseline | $1,300 |
| **D. Glass-box conversion map** | 5–6 | On the winner, swap the top-k recurrent layers for diagonal/contracting certified variants one at a time; log Δscore, ΔECE, Δsteer | This log *is* the capability–interpretability Pareto map; it stops where a layer swap costs >1 pt | $300 |
| **E. Submission + independent re-run** | 6–7 | Full 0.2.1 run on an RTX PRO 6000 (official latency hardware), a different agent re-runs it, then PR to the Index | Score, ECE, latency, steer rate, risk-coverage reported together | $250 |
| **F. Stretch: Q27-AF** | 7–9 | Same conversion on Qwen3.8-27B (16 attention layers) only if Q9-AF ≥ Clef-flash | Target > 61.98 (Clef) | $1,000 (from reserve + P2 if justified) |

**Side paper ($0, CPU):** H-SLT-1 with the corrected learning coefficient, λ = r + (n−r)/2 for real diagonal modes and λ = n + r for complex conjugate-pair modes. Both were Monte-Carlo-checked by the reviewer, and no LLC-on-SSM paper was found. The modal truncation rule is corrected too (H2² cost is |c_i b_i|²/(1−|λ_i|²)). This uses the existing `chl-acquisition-gate` sampler benchmark; we log results and never gate on them.

**Top risks:**
- **Distillation is the hinge.** Q9-AF may lose retrieval. Mitigation: the RWKV7 arm runs in parallel, and the knowledge in that 13B model is real.
- **Latency.** On ~1K-token records, prefill dominates and recurrence buys little, so we don't promise a latency win; we report it.
- **Index rules.** Our injection augmentation must not touch test items. We use train splits only.

---

## Project 2: SSM-LM, a controlled, aligned LLM built off SSM-1

**Premise:** the decision model is the controller, and generation is a typed decision at every step that matters. One recurrent backbone (Q9-AF or RWKV7, already chat-capable) carries both the SSM-1 heads and its LM head on the same state.

### Architecture

1. **Plan:** SSM-1 makes typed decisions over a knowledge set K and tools: which records, which tool, act/abstain. Each decision carries calibrated confidence and an evidence trace.
2. **Check:** a deterministic checker confirms K-membership and schema/consistency rules.
3. **Realize:** the LM generates under a DFA/grammar mask with copy-slots bound to checked records. A DFA mask is enough; Ctrl-G is not needed.
   - **Rung A** is template/CNL text, where the guarantee holds.
   - **Rung B** is free connective text, scored by claim decomposition plus a verifier.
4. **Guarantee, stated precisely:** in rung A, every atomic claim's slot values are copied from checked K records. That is a syntactic lemma. It does *not* cover wrong selection, omission, or implicature, which are measured instead:
   - claim-level precision on a closed-KB QA set;
   - selection error rate;
   - fraction of tokens in the guaranteed region (a Pareto axis).
5. **Control under injection:** the same provenance channels and typed-action masks as SSM-1, with the prior lab's harness as the gate. Attacks are D1 insertion and GCG order-flip; metrics are hijack rate, steer rate and alarm rate.
6. **Persistence:** we replace the half-life bound with measured injection persistence. That means flip-rate as a function of distance between an injected span and the decision, per layer, using the recurrent state's natural probe point.

### Stages

| Stage | Weeks | What | Cost |
|---|---|---|---|
| P2-A | 6–8 | Closed-KB domain (product catalog or policy KB, 10⁴–10⁵ records); planner = SSM-1; DFA realizer; checker | $300 |
| P2-B | 8–10 | Injection-augmented LoRA on the LM head; harness eval vs. same-size stock instruct model | $400 |
| P2-C | 10–11 | Report: hard-region fraction, claim precision, selection error, steer/hijack, calibration of plan decisions | $300 |

---

## Budget

| Bucket | $ |
|---|---|
| P1 Stages A–E | 2,800 |
| P2 | 1,000 |
| Reserve (reruns, independent reproduction, F-stretch if earned) | 1,200 |
| **Total** | **5,000** |

Spend controls carried over from prior campaigns: kill-safe deadline runner and watchdog (chl-typed-control tooling), per-stage caps, a spend report at every gate. Personal accounts only.

## Reused infrastructure

- `chl-typed-control`: attack/mask/alarm/certificate/conformal harness, GCG letter-logit attacker, Lambda/RunPod kill-safe tooling.
- `chl-structured-models`: GDN2/attention typed-head models, pinned BANKING77, exact decision graph.
- `chl-slt-experiments`: pure-PyTorch Mamba-2 SSD (verified), devinterp wrapper.
- `mechanistic-ssm-control`: diagonal Gramian/HSV diagnostics.
- `chl-acquisition-gate`: ground-truthed sampler benchmark (for the H-SLT-1 side paper).

## Week-1 actions

1. Pin Decision Index 0.2.1 harness + stratified dev slice (per area, per length bin); pre-register gates.
2. Run Clef-flash locally (one H100/H200) to validate harness parity against its published per-benchmark numbers.
3. Stage A stock readouts on 4 backbones.
4. Write the Q9-AF distillation script (layer swap + MOHAWK stages) against Qwen3.5-9B's `layer_types`.

---

## Stage 0 on $525 SF Compute (autoresearch) credits — validate the hinges before the $5k

Platform: h100-1 (80 GB, 14 vCPU, ~250 GB RAM) per-minute billing, 15-min idle grace, 250 GB persistent disk, 12 h holds, detached jobs, HF egress. Rate is shown at node creation (not in docs) — assume ~$2–2.5/h ⇒ ~210–260 H100-h. Do data prep on cpu shapes or locally.

| # | Module (each standalone, each answers one kill question) | Kill question | H100-h | ~$ |
|---|---|---|---|---|
| 0.1 | Harness: pinned Index 0.2.1 + stratified dev slice (~5k rows, per area × length bin); Clef-flash parity run | Does our harness reproduce Clef-flash's per-benchmark numbers within noise? | 4 | 10 |
| 0.2 | Stock readout triage: Qwen3.5-9B, RWKV7-G1k 7.2B/13.3B, Falcon-Mamba-7B | How far below Qwen3.5-9B are pure-recurrent backbones, and in which area? | 8 | 20 |
| 0.3 | Steer/hijack baseline on the same backbones (chl-typed-control harness, D1 + GCG-lite) | Are recurrent backbones more/less steerable than the hybrid? | 6 | 15 |
| 0.4 | Q9-AF pilot: swap 1 → 8 attention layers for GDN, layer-wise output matching only, ~200M tokens; dev slice + long-record retrieval probe after each swap | Does removing attention keep the knowledge? Which layers are expensive? | 50 | 125 |
| 0.5 | Head pilot on the better backbone: integrator vs final-state fork vs snapshot fork vs Clef head, small LoRA, train splits + injection augmentation | Does an attention-free head close the gap to the attention head on retrieval/tools? | 40 | 100 |
| 0.6 | Instrumentation check: per-checkpoint eval card (score, debiased ECE, steer rate, Gramian/decay spectra, probe leakage) on 0.4/0.5 | Are the trackers cheap and stable enough to run every checkpoint? | 4 | 10 |
| — | H-SLT-1 side paper | (CPU, local, $0) | 0 | 0 |
| — | Reserve (reruns, independent re-measure of any win) | | ~90 | ~245 |

Exit of Stage 0: a go/no-go on (a) Q9-AF vs RWKV7 as backbone, (b) which head, (c) whether the trackers work — before touching the $5k.

# Merged plan (v1, 2026-10-06)

This plan supersedes `plans/claude/PLAN.md` for Track D. Track L detail lives in `plans/codex/LANGUAGE_GROWTH_PLAN.md`, and the seam is `interfaces/DECISION_LANGUAGE_V0.md`. Evidence comes from `reviews/01–04`, `analysis/A`, `B`, `CODEX_CRITICAL_COMPARISON.md` and `techniques/`.

## Objective and honest targets

The family is a set of attention-free models: in the text path there is no softmax over positions and no KV cache that grows with length. Capability, calibration, steering/hijack resistance, interpretability evidence, latency and cost are always reported together. Clef's vision path and its routing head use attention, so our claims are text-only until a vision gate passes.

| Tier | Target (Decision Index 0.2.1 unless noted) | Est. probability at ≤$5k |
|---|---|---|
| T1 | First attention-free entrant. Beats the best non-Clef 9B open entrant (Nimble-9B-v3, 56.88), with ECE in the board's top 3 (debiased, ≥4k-row sample) and a reported steer rate | ~35–50% (estimate) |
| T2 | Beats Jev (57.91) and Clef-flash (≈57.07, secondary source) | ~15–25% (B-review) |
| T3 | Beats Clef (61.98, pending PR) | <10%; needs a 27B conversion at ≥$2.2k |
| Off-board | Astra/Fable on a 5k-row reference | Reference only; not a target |

Teachers are open-weight only (Clef, Qwen3.8-27B, Gemma-4). The TypeSafe MCA §2.3(b) and frontier-API terms forbid distillation from their outputs.

## Track D: SSM-Decision

### D-architecture

- **Parent backbone (chosen at m1):**
  - **Q9-AF:** Qwen3.5-9B with its 8 full-attention layers converted to delta-rule layers (GDN, or RWKV-7/GKA if Priming 2605.08301 holds). Layers are converted in KL-importance order (2512.20569), with an accept/rollback gate per layer (2609.21139).
  - **RWKV7-G1k-13.3B** (pure recurrent, Apache-2.0).
  - **Mamba-3-SISO-1.5B** as a small, fully attention-free reference for interpretability work.
- **Recall levers (none of these use attention):**
  - schema and options before the state, then questions re-read (JRT, 2407.05483);
  - chunked state inference in the OPRM style, with memory bounded by a fixed S;
  - 20–30% retrieval-targeted tokens in the distillation mix.
- **Decision heads (factorial):**
  - additive integrator (accounting attribution);
  - final-state fork;
  - bounded-S snapshot fork, aggregated by head-level partial sums (no softmax over positions);
  - Codex's Head B, a per-option selective recurrence over shared features;
  - Clef-style attention head as a priced *control*.
- **Calibration:** unsmoothed proper loss arm vs. the Clef LS-CE+Brier arm. Post-hoc temperature is predicted from length and option-count features. Report debiased/SmoothECE per length bin.
- **Authority:**
  - exact output masks;
  - host-side dependency restriction (the exact part);
  - ISE/ASIDE embedding separation plus injection-augmented LoRA (the empirical part).
  - These are evaluated with adaptive attacks (GCG plus D1 insertion). Steer rate is the headline safety metric.
- **Interpretability instruments:**
  - perturb-and-recompute checks on attributions;
  - knockout-at-distance persistence curves per layer, which replace the half-life bound;
  - per-layer gather-and-aggregate sharpness;
  - DAS with linear maps;
  - a CBM-style leakage test on semantic records.
  - LLC is logged, never gated. H-SLT-1 is unresolved (Codex counterexample).

### Pre-registered fallback (the literature is unanimous here)

Fully attention-free conversions lose retrieval (2408.15237, 2504.18574, 2510.19861). If Q9-AF fails the retrieval gate after the recall levers, we do **not** quietly add attention. We report the measured frontier as the "cost of attention-freedom" curve: pure, 1, 2 and 8 attention layers. Two zero-training controls sit on the same plot: the stock hybrid, and SWA plus sinks (2608.28444). The pure-recurrent entrant is still submitted. Whether to ship a hybrid is a `decision-needed` call for the user.

## Milestones (shared)

| Milestone | Owner | Content | Gate |
|---|---|---|---|
| **m0** | D | Index 0.2.1 runner pinned; ≥10k-row stratified dev slice (≥2k/area, length bins); separate untouched test slice; Clef-flash adapter parity on shared fixtures; stock readouts for Qwen3.5-9B, RWKV7-13.3B, Mamba-3-1.5B; replay fixture + `decide` adapter handed to Track L | Adapter parity within fixture tolerance; paired cluster-bootstrap CIs computable |
| **L0** | L | Types, interpreter, renderer, 10 joint fixtures, oracle/replay backends (CPU) | Fixtures pass; reviewed by D |
| **m1** | D | One-layer conversion probe (memory, throughput, decision, retrieval: MQAR, S-NIAH), then the full 8 with accept/rollback; LM-quality gate (held-out ppl, MMLU-subset); steer baseline on all parents | Parent selected and frozen; revision handed to L |
| **m2** | D | Head × parent factorial with injection augmentation and both calibration arms | T1 dev-slice criteria met or attributed (knowledge → backbone; retrieval → head; length-dependent → recall levers) |
| **L1–L2** | L | Frozen parent as semantic planner through typed choices; retention of decision competence on identical items | Per Codex plan |
| **m3** | D+L | Full Index run on RTX PRO 6000; independent re-run by the other agent; PR to the Index | Score, ECE, latency, steer, risk-coverage reported together |

## Spend (shared $525 SF envelope; $5k ceiling)

| Line | $ | Contents |
|---|---|---|
| Joint gate | 75 | m0 harness and stock readouts (~11 h), throughput qualification, Clef-flash parity |
| System 1 | 250 | m1 conversion (~$40 one-layer probe, ~$150 full), m2 factorial (rest) |
| Language | 100 | Codex L1 learned probe (≤$20 first) and retention checks |
| Reserve | 100 | Re-runs, independent reproduction |

Rates are $2.97/h batch and $3.96/h interactive. The $5k ceiling then funds: a data plan ($400–600; held-out generators, open teachers), the full m2 factorial, the m3 submission, and T3 only if T2 passes.

## Kill and fold rules

- A falsified hypothesis keeps its receipt and goes into `LEDGER.md` under negative results.
- A win that doesn't reproduce on the other agent's re-run is logged as a false positive.
- Two consecutive no-progress checks on a line of work means fold it.

# m2 first paid-run spec (NOT launched)

Status: spec only. Nothing here has run on paid hardware. All $ and memory figures come from
`python -m train.estimate` (`train/estimates/m2_estimates.json`). They are formula estimates
built on unverified dimensions and assumed MFU, and the first rung exists to replace them with
receipts.

## What m2 tests

The head × parent factorial (PLAN.md, D-architecture):
- **Heads:** integrator, final-state fork, bounded-S snapshot, Head B, and the Clef cross-attention head as a priced control.
- **Calibration arms:** unsmoothed CE vs. LS-CE + Brier (+RPS on ordinal questions).
- **Post-hoc step:** a predicted temperature (length and option-count features), fit on a calibration split.

The m2 gate is "T1 dev-slice criteria met, or each gap attributed". Gaps are attributed this way: knowledge → backbone, retrieval → head, length-dependent → recall levers.

## Preconditions (all required before rung 1 spends anything)

1. This branch is merged to `agent-main` after cross-review (COORDINATION: code that runs paid must be in shared history).
2. A `spend` issue is open with the estimate and stop conditions below, and the other agent has acknowledged it.
3. A planned row is in `LEDGER.md`.
4. The SF credit expiry has been checked.

Rung 2 also needs all of these:
- the m1 parent is frozen (Q9-AF, or the pre-registered fallback);
- decision training data exists (the pilot mixture from the data plan, built from held-out generators, with no Index items or templates);
- `claude/m0-harness` and `claude/m0-steer-eval` are merged, so eval cards produce a real Index dev-slice score and steer rate instead of stubs.

## Rung 1: qualification (≤ $20 cap; estimate ≈ $6–8)

**Hardware:** 1× H100-80GB, SF batch rate $2.97/h. Runtime ceiling: 4 h. Stop the node afterwards.

**Backbones:**
- **Shape proxy for Q9-AF:** `Qwen/Qwen3.5-9B-Base`, stock. It has the same width and depth and the same GDN shapes. Its 8 attention layers stay as attention, so the memory/throughput figures are an upper-bound proxy until the m1 checkpoint exists.
- **Second backbone:** `state-spaces/mamba3-siso-1.5b`. Capture is a hook on `backbone.norm_f`.
- **RWKV7-13.3B** joins only if m1 selects it (its LoRA module names are still unverified).

| # | Measurement | Configs | Pass condition |
|---|---|---|---|
| 1 | Peak memory and step time | 5 heads × (T, K) ∈ {(2048, 4), (8192, 16), (8192, 255)}, B = 1, LoRA r = 256, bf16, grad checkpointing, 20 steps each | B = 1 fits at T = 8192 for every head except the fork; measured GB within 1.5× of the estimate (otherwise the estimator is recalibrated before rung 2) |
| 2 | Max micro-batch at T = 2048, K = 4 | integrator, head_b | Measured, then recorded in the receipt |
| 3 | Fork cost | Recompute fork (today's implementation) vs. estimate at K ∈ {4, 16} | Recorded; decides whether the cache-copy fork must be built before the fork cell |
| 4 | Numerics on a real backbone | `tests/test_heads.py`-style checks on real features: head permutation equivariance (fp32 head, bf16 backbone), integrator/snapshot attribution sums, Head B chunked = streaming scan | Within 1e-3 (bf16 backbone) |
| 5 | Learning signal | 200 steps on 2k pilot rows per head (CE arm) | Loss decreases; no NaN |

**Stop conditions:**
- an OOM at B = 1, T = 2048 for any attention-free head;
- any non-finite loss;
- the 4 h ceiling.

Each one stops the run and goes into the receipt. Receipts go to `receipts/<date>-D-m2-qual/`: commit, command, config hash, model revisions, memory/time per config, and GPU-hours billed.

## Rung 2: capped factorial (after the rung-1 receipt and the data)

The estimate is for one cell: 30k question rows (10k records × 3 questions), T = 2048, K ≈ 6, 1 epoch.

| Parent | GPU-h / cell | $ / cell (batch) |
|---|---|---|
| Q9-AF (proxy dims) | 2.7 | ≈ $8.0 |
| Mamba-3-1.5B | 0.6 | ≈ $1.8 |
| RWKV7-13.3B | 6.0 | ≈ $17.7 |

The proposed order keeps each step's spend small:
1. **5 heads × CE arm on the m1 parent: 5 cells.** One seed per cell. Eval cards every 500 steps go on the dev slice only; the untouched test slice is never read.
2. **The Clef arm on the best two attention-free heads plus the control: 3 cells.**
3. **Seeds 2 and 3 for the top head and the control only.** This step is needed before any head-vs-head claim (independent re-run per COORDINATION).

On Q9-AF, steps 1 and 2 come to ≈ $64. That exceeds what remains in the System 1 line after m1 (≈ $60, PLAN.md spend table). The user or the other agent picks one of these through the `spend` issue:
- **(a)** halve the rows per cell (≈ $32);
- **(b)** run the whole factorial on Mamba-3-1.5B first (≈ $14), then only the winner and the control on Q9-AF (≈ $16);
- **(c)** draw from the reserve.

**Pre-registered readouts per cell:**
- Index dev-slice score by area × length bin;
- debiased ECE / SmoothECE before and after temperature;
- steer rate (D1-B held-out templates);
- head cost (params, FLOPs, persistent state);
- latency.

**Gap attribution:**
- the head-vs-control gap on retrieval/tools rows → head;
- the gap shared by all heads → backbone;
- a gap that grows with length bin → recall levers.

## Known limitations carried into the run

- **Attribution is accounting, not causal.** It needs the perturb-and-recompute check (`heads.heads.erase_token_effect`) before it is used as evidence.
- **The ordered (m0-identical) layout is not structurally permutation-equivariant.** The symmetric layout is, at the cost of reading the state without the option texts. Both are arms; report order-agreement.
- **The final-state fork is recompute-based today.** That costs K × prefix tokens. The cache-copy fork is estimated at ≈ 71 MB of fp32 state per option on Q9-AF and is not implemented.
- **The Clef control uses softmax over positions.** It is a control and never the entrant.
- **Mamba-3 was trained at a 2,048-token context.** T = 8192 points on it are extrapolation (techniques/02 §2.9 collapse check applies).

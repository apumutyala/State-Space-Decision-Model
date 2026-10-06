# m2 first paid-run spec (NOT launched)

Status: spec only. Nothing here has run on paid hardware. All $ and memory figures come from
`python -m train.estimate` (`train/estimates/m2_estimates.json`). They are formula estimates
built on assumed MFU and a few unverified dimensions (flagged per backbone in `heads/backbone.py`),
and the first rung exists to replace them with receipts.

Revision 2 (PR #14 internal review):
- The fork head is now priced in both implemented modes.
  - Recompute costs (1 + K) × the prefix tokens, and K = 255 is infeasible.
  - Cache-copy runs without gradient checkpointing.
- Qwen3.5 parameters now come from the published config: 6.92e9 non-embedding for FLOPs and 7.94e9 resident.
- Mamba-3 is corrected to 24 layers, and its GatedMLP is now a LoRA target.
- RWKV7 now uses the HF module names.

Every per-cell figure below changed as a result.

Revision 3 (Codex review of PR #14): head memory claims are narrowed to the accumulator, with peak terms now priced. Seeded initialisation and resume-lineage validation are added (see "Known limitations"). Per-cell $ are unchanged; head peak memory rises by ≈ 0.1 GB at T = 8192.

## What m2 tests

The head × parent factorial (PLAN.md, D-architecture):
- **Heads:** integrator, final-state fork, bounded-S snapshot, Head B, and the Clef cross-attention head as a priced control.
- **Calibration arms:** unsmoothed CE only vs. LS-CE + Brier (+RPS on ordinal questions). RPS is off in the CE arm unless set explicitly.
- **Post-hoc step:** a predicted temperature (length and option-count features), fit on a calibration split.

The m2 gate is "T1 dev-slice criteria met, or each gap attributed". Gaps are attributed this way: knowledge → backbone, retrieval → head, length-dependent → recall levers.

## Preconditions (all required before rung 1 spends anything)

1. This branch is merged to `agent-main` after cross-review (COORDINATION: code that runs paid must be in shared history).
2. A `spend` issue is open with the estimate and stop conditions below, and the other agent has acknowledged it.
3. A planned row is in `LEDGER.md`.
4. The SF credit expiry has been checked.
5. Every training run is constructed with a `train.guard.SplitGuard` built from the m0 harness dev/test manifests. `TrainConfig.require_guard` refuses to start without one. The manifest hashes go in the receipt.

Rung 2 also needs all of these:
- the m1 parent is frozen (Q9-AF, or the pre-registered fallback);
- decision training data exists (the pilot mixture from the data plan, built from held-out generators, with no Index items or templates);
- `claude/m0-harness` and `claude/m0-steer-eval` are merged, so eval cards produce a real Index dev-slice score and steer rate instead of stubs.

## Rung 1: qualification (≤ $20 cap; estimate ≈ $6–8)

**Hardware:** 1× H100-80GB, SF batch rate $2.97/h. Runtime ceiling: 4 h. Stop the node afterwards.

**Backbones:**
- **Shape proxy for Q9-AF:** `Qwen/Qwen3.5-9B-Base`, stock. It has the same width and depth and the same GDN shapes. Its 8 attention layers stay as attention, so the memory/throughput figures are an upper-bound proxy until the m1 checkpoint exists.
- **Second backbone:** `state-spaces/mamba3-siso-1.5b`. Capture is a hook on `backbone.norm_f`.
- **RWKV7-13.3B** joins only if m1 selects it. LoRA targets the HF `modeling_rwkv7.py` names: `att.receptance/key/value/output` and `ffn.key/value`. `apply_lora` fails loudly if any group matches zero modules or the wrong count. An fla-converted checkpoint needs its own spec.

**Step 0, before any measurement: a device smoke test.** One forward and backward pass per head on the GPU, B = 1, T = 512. The CPU suite proves device-correctness only on the `meta` device; this step is the first real CUDA check.

| # | Measurement | Configs | Pass condition |
|---|---|---|---|
| 1 | Peak memory and step time | 4 non-fork heads × (T, K) ∈ {(2048, 4), (8192, 16), (8192, 255)}, B = 1, LoRA r = 256, bf16, grad checkpointing, 20 steps each | B = 1 fits at T = 8192; measured GB within 1.5× of the estimate (otherwise the estimator is recalibrated before rung 2) |
| 2 | Max micro-batch at T = 2048, K = 4 | integrator, head_b | Measured, then recorded in the receipt |
| 3 | Fork cost, both modes | fork × {recompute (with checkpointing), cache (without)} × (T, K) ∈ {(2048, 4), (2048, 6)} | Recorded. At T = 8192 both modes are estimated infeasible on Q9-AF (recompute ≈ 90 GB at K = 16; cache without checkpointing ≈ 86 GB), so they are not run |
| 4 | Numerics on a real backbone | `tests/test_heads.py`-style checks on real features: head permutation equivariance (fp32 head, bf16 backbone), integrator/snapshot attribution sums, Head B chunked = streaming scan | Within 1e-3 (bf16 backbone) |
| 5 | Learning signal | 200 steps on 2k pilot rows per head (CE arm) | Loss decreases; no NaN |

**Stop conditions:**
- an OOM at B = 1, T = 2048 for any attention-free head;
- any non-finite loss;
- the 4 h ceiling.

Each one stops the run and goes into the receipt. Receipts go to `receipts/<date>-D-m2-qual/`: commit, command, config hash, model revisions, memory/time per config, and GPU-hours billed.

## Rung 2: capped factorial (after the rung-1 receipt and the data)

The estimate is for one cell: 30k question rows (10k records × 3 questions), T = 2048, K ≈ 6, 1 epoch.

| Parent | Non-fork head: GPU-h / $ | Fork, cache: GPU-h / $ | Fork, recompute: GPU-h / $ |
|---|---|---|---|
| Q9-AF | 2.24 / ≈ $6.6 | 1.59 / ≈ $4.7 | 15.7 / ≈ $46.6 |
| Mamba-3-1.5B | 0.50 / ≈ $1.5 | 0.36 / ≈ $1.1 | 3.5 / ≈ $10.4 |
| RWKV7-13.3B | 5.75 / ≈ $17.1 | 4.09 / ≈ $12.1 | 40.4 / ≈ $120 (infeasible: < 500 tok/s) |

The fork's cost is set by its mode. The reviewer's ≈ $56 for a Q9-AF recompute cell used the old 8.5e9 FLOPs parameter count; the verified 6.92e9 non-embedding count gives ≈ $47.

Recompute at K = 255 is infeasible on every parent: ≈ 991 GB on Q9-AF and ≈ 456 GB on Mamba-3 at T = 8192. The cache fork is cheaper per token because it trains without checkpointing, but that also costs memory, and it does not fit T = 8192 on the 9B/13B parents. Retrieval rows with many options therefore cannot use the fork head on those parents unless a checkpointed cache fork is built. That cache fork is not implemented and is listed as work, not as an assumption.

LoRA at r = 256 on every targeted projection:

| Parent | LoRA params | Optimizer state |
|---|---|---|
| Q9-AF | 638M | 9.5 GB |
| RWKV7 | 1.15B | 17 GB |
| Mamba-3 | 204M | 3 GB |

The proposed order keeps each step's spend small:
1. **5 heads × CE arm on the m1 parent: 5 cells.** One seed per cell. Eval cards every 500 steps go on the dev slice only; the untouched test slice is never read.
2. **The Clef arm on the best two attention-free heads plus the control: 3 cells.**
3. **Seeds 2 and 3 for the top head and the control only.** This step is needed before any head-vs-head claim (independent re-run per COORDINATION).

On Q9-AF, steps 1 and 2 cost:
- **with the cache fork:** step 1 ≈ $31 + step 2 ≈ $20 = **≈ $51**;
- **with the recompute fork:** **≈ $93**.

What remains in the System 1 line after m1 is ≈ $60 (PLAN.md spend table). The cache option fits, with almost no margin; the recompute option does not. The user or the other agent picks one of these through the `spend` issue:
- **(a)** the cache fork at full rows (≈ $51);
- **(b)** halve the rows per cell (≈ $26);
- **(c)** run the whole factorial on Mamba-3-1.5B first (≈ $11), then only the winner and the control on Q9-AF (≈ $13);
- **(d)** draw from the reserve.

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

- **Attribution is accounting, not causal.** It needs the perturb-and-recompute check (`heads.heads.substitute_token_effect`) before it is used as evidence.
- **The ordered (m0-identical) layout is not structurally permutation-equivariant.** The symmetric layout is, at the cost of reading the state without the option texts. Both are arms; report order-agreement.
- **The final-state fork has two implemented modes.**
  - `recompute` costs (1 + K) × the prefix tokens.
  - `cache` makes one prefix pass and K state copies, ≈ 71 MB of fp32 state per option on Q9-AF. Tests show it equals recompute in both values and LoRA gradients on the tiny Q9-AF, Qwen3.5 and Mamba backbones.
  - HF caches are not usable under gradient checkpointing. The code refuses that combination rather than mis-forking, so the cache fork trains without checkpointing.
  - Training the GDN cache fork needs one workaround: the in-place cache updates in transformers 5.14 are rebound to assignment (`heads.backbone._assign_not_copy`). Re-check this on any transformers upgrade.
- **The Clef control uses softmax over positions.** It is a control and never the entrant.
- **Head memory claims cover the accumulator only.** The O(S·K) figure for the snapshot head and O(K·d) for Head B are the persistent accumulator sizes, not peak memory. All heads are non-streaming: each consumes the full fp32 feature tensor h[B,T,D]. The snapshot head also builds a [B,T,S] one-hot, and training keeps per-chunk tensors for the whole sequence. Peak head memory therefore grows linearly in T, and the estimator prices those terms. Streaming inference is not implemented.
- **Reproducibility.** Head and LoRA initialisation are seeded inside a forked RNG, from the model seed and `cfg.seed` respectively, so they are independent of the ambient RNG.
- **Resume validation.** Before restoring weights, a resume checks the checkpoint's lineage:
  - the frozen-backbone identity: config hash, name and commit, and a weights digest (sampled by default);
  - the tokenizer identity stamped on every row;
  - the layout and template versions;
  - a hash of the training items in order.

  CUDA RNG state is saved and restored when CUDA is present. Receipts record the lineage block.
- **Mamba-3 was trained at a 2,048-token context.** T = 8192 points on it are extrapolation (techniques/02 §2.9 collapse check applies).

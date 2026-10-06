# m1-prep pilot: attention → GDN conversion on Qwen3.5-0.8B. Status: INCOMPLETE

**Bottom line.** The pipeline code is written and its kernels are numerically verified. A tiny smoke run exercised
alignment, importance ordering, end-to-end KL, the gate's **rollback** branch and the stock-model probes.
**Never executed:** the gate's accept branch, probes on a converted model, and converted-model throughput (in the smoke
every layer was rolled back, so `converted_0` was the stock model). The first paid phase exists to exercise them. The
**"cost of attention-freedom" curve was not produced.** A memory bug in the probes (now fixed), on a laptop GPU
shared with another worker, crashed the machine. After that, local GPU runs were stopped by the coordinator.
The curve and the 9B throughput numbers move to a rented GPU (the ≤$20 qualification spec is below).
This is pipeline qualification only, not a 9B claim.

## What is verified (receipts in this dir)

| Check | Result | File |
|---|---|---|
| Our recurrent GDN vs HF `torch_recurrent_gated_delta_rule` (fwd) | rel err 1.4e-7 | `tests/test_gdn_torch_venv.json` |
| Our chunked (WY/UT) GDN vs recurrent, CPU and CUDA (fwd + all grads) | ≤ 4e-7 | same |
| Our `GDN` module vs HF `Qwen3_5GatedDeltaNet`, real layer-0 weights, pure-torch HF path | 6.8e-7 (recurrent), 3.1e-7 (chunk) | same |
| fla 0.5.2 Triton kernel on Windows (`triton-windows` 3.4) vs fp32 recurrent, bf16 inputs (fwd + grads) | o 5.3e-3, grads ≤ 8.4e-3 | `tests/test_gdn_fla_venv.json` |
| Our module (fla) vs HF layer (HF auto-picks fla when importable) | 7.8e-3 | same |
| Pipeline smoke, tiny budget, rollback path exercised (impossible ΔNLL threshold → every layer rolled back); accept path, converted probes and converted throughput NOT exercised | runs | `smoke/` |
| `init_gdn` on the real 0.8B and 9B head layouts (random weights, hidden 512, one layer, CPU): shapes, finite forward, exact v/z/o row mapping, 0.8B q/k split exact, 9B per-head SVD key projection (q·k score corr 0.915 min, energy 0.71 on random weights), unsupported layout raises | PASS | `tests/test_init_shapes_cpu.json` |
| Paid-run stop conditions (`align.Guard`: throughput exit 3, wall clock) and the pre-registered 9B decision rule (CPU) | PASS | `tests/test_guard_cpu.json` |

flash-linear-attention **does** work on Windows with an RTX 5070 Ti Laptop (sm_120) through `triton-windows`. It is
installed in a gitignored overlay venv (`.venv-fla`, which adds `.venv-gpu` site-packages through a `.pth` file).

## Smoke numbers (diagnostic only: 2×512 tokens per step, a handful of steps, shared GPU)

- The stock model loads and probes correctly (ppl 18.7 on 4×1k wikitext-2 test windows; NIAH and MQAR at smallest settings 1.0).
- Init comparison, stage-1 relative MSE to the attention output after 3 steps: random init 8–229 vs attention-weight reuse 1.0–14.6.
  Reuse is much better. The full run adds output-norm calibration and a third init (`clone_neighbor`).
- Single-swap KL after 5 alignment steps gives the order L23 < L19 < L11 < L15 < L3 < L7. Last layer least important,
  matching the fallback prior. Too few steps to trust.
- Uncontended laptop training throughput: stage 1 ≈ 3.5–5k tok/s, stage 2 (end-to-end KL with checkpointing) ≈ 0.85–1.0k tok/s.
  When the other worker held the GPU this fell to about 50 tok/s, which is why the 9B is priced from FLOPs and not from these numbers.

## Gate semantics

`policy: force` (the curve configs) **records** the gate verdict and never enforces it: every layer is converted so the
curve reaches 0 attention layers. `policy: enforce` (alias `rollback`) enforces it: a failing layer returns to attention
and the other GDNs are restored. The gate checks per-step ΔNLL vs the previous accepted student, cumulative ΔNLL vs the
teacher (`max_cum_delta_nll`, added after review so drift cannot accumulate unbounded) and final-hidden cosine.

## Known gaps (from internal review)

- `results.json` / `smoke/results.json` record `env.commit` SHAs (`2a742bad`, `8654928`) that the rebase onto agent-main made
  unreachable, both `dirty: true`; the code they ran is the content of `1237320` / pre-rebase working tree.
- The post-fix memory numbers below were measured interactively; no log of them is in this directory.
- No decision-shaped probe yet (PLAN m1 lists decision next to MQAR/S-NIAH); the 9B run needs the m0 stock-readout slice to
  inform parent selection.
- "Stock" numbers use our GDN kernel for the native GDN layers (verified equal to HF's to ≤7e-7 fp32), not HF's code path.
- Throughput is batch-1 prefill only.

## Why it crashed (root cause, fixed in commit 7df43d5)

1. `replace_native_gdn` built fresh `GDN` modules, so the 18 native GDN layers had `requires_grad=True`, and a decorator
   edit had moved `@torch.no_grad()` off `_argmax_hits`. Probe forwards therefore kept full autograd graphs:
   peak memory 2.7 / 3.9 / 6.3 / 11.0 GB at 512 / 1k / 2k / 4k tokens. After the fix: 1.8 GB at 4k, 2.1 GB at 8k.
2. HF passes `enable_gqa=True` to SDPA, which rules out the memory-efficient kernel on torch 2.8 for Windows, so attention used the
   O(T²) math kernel (+5.5 GB at 8k). Fixed: HF repeats the KV heads itself, so the mem-efficient kernel runs (+0.58 GB at 8k).
3. Perplexity computed full-vocab fp32 logits for a whole 4k window (≈4 GB). Now chunked in 512-token pieces.
4. The resumable retry driver (`scripts/run_pilot.sh`), relaunched after each fix, left earlier retry loops alive.
   That produced duplicate pipeline processes, which the coordinator killed. **Lesson for the cloud run:** use one
   driver, a pid file, and no retry loops on a shared machine.

## 9B cost model (`cost_model.json`; FLOP-based, not yet measured on H100)

FLOP per token: stage 1 ≈ 2N (teacher forward) plus the GDN training; stage 2 ≈ 10N (teacher forward, student forward, recompute,
activation backward, KL head). N = 8.5e9 text params, 15% eval overhead, H100 at $2.97/h batch.

| Plan | Tokens (stage 1 + stage 2) | H100-h @35% MFU | $ | H100-h @25% MFU | $ |
|---|---|---|---|---|---|
| One-layer probe | 50M + 100M | 8.8 | 26 | 12.3 | 36 |
| All 8 layers, lean | 50M + 8×50M | 32 | 96 | 45 | 134 |
| All 8 layers | 100M + 8×100M | 65 | 192 | 91 | 269 |

Memory on one 80 GB card (est. 43 GB): bf16 weights 18; 0.54B trainable GDN params with fp32 master + Adam 8.6;
checkpointed layer inputs at 8×4096 = 8.6; one layer's recompute ≈ 6; chunked KL logits ≈ 2.

Review notes: the laptop smoke implies ~15–17% MFU, so 25% at 8×4096 on an H100 is plausible but **35% is unsupported until
measured**; the 100M-tokens-per-layer budget is generous relative to RADLADS (350–700M tokens for all layers of a 7B).

## First paid job: rung-2 hardware qualification (spec only; NOT submitted)

Superseded after issue #3 review: the interactive-node runner and the bundled 0.8B curve are withdrawn. The paid job is now a
batch `submit_job` hardware qualification (numerics, throughput, memory, artifact recovery only), validate_job worst case
$9.90 with a $20 workspace-cap ceiling: see `jobs/qual/JOB_SPEC.md`. The 0.8B curve becomes a separate later job.

## Files

`config.json` is the intended full-run config (batch 2, VRAM cap 0.40). `results.json` holds environment and data hashes only (no curve points).
`train_log.jsonl` is empty. `stdout.log` is from the last aborted attempt. `smoke/` contains the smoke run. `cost_model.json` holds the cost model and spec.
Data: wikitext-103 train (40.06M tokens, distillation), wikitext-103 validation (gate), wikitext-2 test (eval), and MMLU test (5 per subject, seed 0).
The SHA-256 of each file is in `results.json`.

# fla `chunk_rwkv7` numerics: real bug or mis-specified gate? (2026-10-08)

**Takeaway.** The rejection does not show a correctness bug. For inference, the gate was the wrong one. fla's error is
flat, at the bf16/TF32 tensor-core floor:
- Output error vs fp64 is 3.4e-3 to 5.6e-3. That is about one bf16 ulp of the output, and chunked sits at 2.4e-3 to 3.1e-3.
- State error is about 1e-3, and cosine is at least 0.999997 everywhere.
- Error does not grow with sequence length.

job-mden7 failed on one clause: e(fla) ≤ 1.10 × e(chunked) on 90% of 55 tensors. fla passed 8 of 55. That clause compares
fla against the vendored `chunked` path, which casts every input to fp32 (`compute_dtype=float32`), so its state error is
about 1e-5. Any bf16 tensor-core kernel fails that clause by construction.

That includes the kernel behind Clef's accepted fast path. Clef's GDN fast path is also fla, and it was accepted on
Index-level equivalence (99.5% agreement, Δ −0.05, CI includes 0).

No decision-level harm has been measured:
- Logits vs eager at T=256: fla rel 0.021 (chunked 0.023), cos_min 0.99995, top-1 agreement 1.0.
- Continuation (P=1900, L=148): rel 0.0199, which passes.

**Recommendation: (a).** Re-gate fla for inference only, on Index-level equivalence. The training gate stays as written.

## What was measured

### Kernel and setup
- fla-core 0.5.2 (sha `5e830c85…`), torch 2.13.0, triton 3.7.1, H100.
- Kernel: `chunk_rwkv7` → `chunk_dplr_delta_rule`. Not `fused_recurrent`.
- Inputs are bf16. The state is fp32 in and out (`heads/rwkv7_wkv.py:130`).

### Reference
- jn47u (rule v1): the reference was a bf16 token-serial eager run. That reference is itself 0.1 to 0.5 off fp64, and
  the production `chunked` path failed it too (LESSONS L-06).
- mden7 (rule v2): the reference was fp64 truth.

### Errors vs fp64 truth (mden7, real inputs from layers 0/15/30/45/60)

| Path | out | state | grads |
|---|---|---|---|
| fla | 3.4e-3 to 5.6e-3 | 5.3e-4 to 1.6e-3 | ≤ 1.3e-2 (cos ≥ 0.99997) |
| chunked | 2.4e-3 to 3.4e-3 | 4e-6 to 2e-5 | ≤ 5e-3 |
| eager_bf16 | 8e-3 to 0.12 | up to 0.53 | – |

### No growth with length
- State error at T=512 vs T=1024 is about the same (for example L60: 1.6e-3 at both lengths).
- Continuation to T=2048 passes.
- Not measured: rows longer than 2k. The validation job below covers them.

### Cause
In jn47u's fp32 tail, fp32 inputs still gave fla a 1e-3 error vs fp64. So the floor comes from the kernel's TF32/bf16
`tl.dot`. The input dtype does not set it, and an fp32 state or a different chunk size would not remove it. Forcing
`TRITON_F32_DEFAULT=ieee` with fp32 inputs is a slower fallback, to use only if the gate below fails.

### Upstream
- No fla issue or release reports an RWKV-7 DPLR precision bug. fla-core 0.5.2 is the latest release.
- #1102 (out-of-bounds reads in channel-mix kernels) does not touch our path: only the WKV is registered.

### Speed
Prefill is 3.08×, 2.71× and 2.98× chunked at 2k, 8k and 16k (11.7k to 12.0k tok/s).

## Validation job spec (not packaged or submitted)

**Rule (pre-registered, inference only).** The reference is the production `chunked` path. Both paths score the same
stock RWKV7-13.3B g1-v2 readout on the 2,008-row DEV subset in **one process, interleaved per row**.

**Verdict EQUIVALENT_FOR_INFERENCE** requires all of the following:
- argmax agreement ≥ 99.0% of questions;
- zero flips where the reference's top-1 vs top-2 margin is > 0.05;
- the paired ΔIndex 95% bootstrap CI includes 0, with |bound| ≤ 0.5, computed with frozen `harness/score.py` and `harness/bootstrap.py`;
- all values finite;
- zero fla fallbacks.

**Reported, not gating:**
- max |dp| and p99 |dp|;
- flips by length (<2k, 2–8k, ≥8k tokens);
- rows/min for each path.

**Adoption** requires a speed ratio ≥ 2.0 in addition to the verdict.

**Optional tail (non-gating):** D1 on the 250-row smoke slice, run only if ≥ 8 minutes remain.

**Setup notes:**
- `GPU_VERIFIED` needs a validation-scoped override.
- The fla warmup is about 6 minutes (jn47u).

**Time and cost.** H100, `max_duration` 40 min:
- chunked: about 20 min;
- fla: about 8 min;
- setup, warmup and load: about 10 min.

Worst case is (18 + 2×40) × $0.0495 = **$4.85**. Expected cost is about $2.9.

Receipts:
- mden7 `truth_v2.json` sha `4b46b076…`, `decision_v2.json` `ba9618f7…`;
- jn47u `fp32.json` and `throughput.json` (in a local scratchpad);
- extract: `%TEMP%\claude\fla-wkv\mden7_truth_v2_errors.json`.

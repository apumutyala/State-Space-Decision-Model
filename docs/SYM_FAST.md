# SYM_FAST: a faster sym-fork readout that stays exactly invariant

Branch `stack/qwen35/sym-fast`, cut from `stack/qwen35/jev3-typed@a21049a9`.
Code: `heads/sym_fast.py`. Wiring: `heads/features.py` and `heads/decide_backend.py` (`sym_exec=`).
Tests: `tests/test_sym_fast.py`. Timing: `scripts/sym_fast_timing.py`.
Nothing here has run yet. The tests and the timing need the remote CPU job, and the real latency needs one GPU run.

## Where the time goes (read from the code)

On JEV1-4B r2, the native sym-fork p50 is 0.197 s and the single-pass shim's p50 is 0.067 s, a 2.9× gap.

- **Served sym-fork today.** The served path is `DecisionModelBackend(shared_prefix=True)`, which calls `heads.prefix_features`.
  - `common_prefix_len` keeps one token short of the row, so the last prefix token becomes a separate "suffix".
  - A one-question JevBench decision therefore makes **3 backbone calls**:
    1. the prefix, `[1, T-1]`;
    2. a 1-token suffix call, which runs through the GDN single-token decode kernel;
    3. the K forked continuations, `[K, w]`.
  - It also deep-copies the cache twice.
- **Shim.** The shim makes **1 call**.
- **Reading the gap.** 0.197 / 0.067 ≈ 3 calls / 1 call. This fits a Qwen3.5-4B forward that is bound by per-call cost (launches and the GDN chunk loop) at these sizes, not by token count. This is a hypothesis from the code. `scripts/sym_fast_timing.py` measures the calls and the latency together.
- **Training uses a different path.** Training runs `HiddenStateBackbone.fork_features`: the whole row in one call, then one continuation call. So the served path's numbers already differ from the training numbers by float rounding.

## Design

The backbone gets an opt-in attribute, `backbone.sym_exec`. Set it with `DecisionModelBackend(..., sym_exec="fork2" | "packed1")`. The fast path is used only for evaluation: no autograd, no trust gate, every row a sym layout, fork features wanted. Training is untouched.

| mode | calls per decision | tokens run | bit-identical to |
|---|---|---|---|
| old (`None`) | 3 for one question; for Q questions, 1 + (number of suffix lengths) + Q | prefix + conts | — |
| **`fork2`** | **2**, whatever Q is | prefix + conts (unchanged) | one question: the training cache fork (`fork_mode="cache"`, `shared_prefix=False`): fork features, log-probs and h |
| `packed1` | 1 (per 64 option sequences) | K × prefix + conts | the recompute fork (`fork_mode="recompute"`): fork features and log-probs |

**`fork2`, one question** (every JevBench decision):
- It is exactly `fork_features(row.ids, conts)`: one prefix call over the whole row, K state copies (GDN conv and recurrent state plus attention KV), and ONE right-padded continuation call.
- No 1-token suffix call and no deep copies.
- Serving now equals training numerics bit for bit, so a train/serve skew goes away.

**`fork2`, Q > 1 questions:**
- One call runs over the common prefix (system + record + state).
- Then ONE call runs over every (question suffix + option continuation) of every question, from sum(K_q) copies of the prefix state. This is the prefix computed once, forked, and reused.
- Each question's suffix features are read from its first option's batch row in content order.
- The result is float-close to the old path, not bitwise: the same tokens are split across different calls.
- `max_copies` (default 256) chunks the copies to bound memory. The chunks are content-ordered.

**`packed1`:**
- Every option runs as the full `row.ids + cont` in one batch, with prefix features from that same call.
- It wins only if a 4B call stays bound by per-call cost at K×T tokens.
- It multiplies compute by about K (mean K is 4.78). The backend reports the tokens it really runs (`plan` / `score` `total_input_tokens`).
- It is an experiment arm, not the default.

## Why invariance stays bitwise

The batch layout is a function of content alone:
- Within a row, options run in the order of their continuation token ids; exact duplicates are ordered by axis index and have equal values.
- Questions run in request order.
- Outputs are un-permuted back onto the row's axis.

`build_row_sym` already maps a reordered or relabeled request to the same row, except `option_ids`. So every call in either mode sees identical tensors and the served probabilities are identical (tol 0).

The fast path is also stronger than the old one: a row whose axis was permuted (`option_order=`, no canonicalization) gets **exactly** permuted fork features. The old path only got within 1e-6, from batch-order rounding.

Bitwise new-vs-old against the old *served* path is impossible for any design with fewer calls. That path ran the last prefix token through a different kernel (decode vs chunk). Bitwise equality is kept against the two reference forks above instead. The invariance claim never depends on new-vs-old equality.

## Latency model and Speed estimate

Let c be the cost of one call at these sizes. Fit it on today's numbers: c ≈ 0.197 / 3 ≈ 0.066 s, and the shim, 1 call, measured 0.067 s.

- **fork2:** about 2c ≈ **0.12–0.13 s p50**, a 1.5× speedup. Most of the dropped call is overhead (1 token). Removing the two deep copies saves a few more ms.
- **packed1:** about c + K×T compute. If per-call bound, ≈ **0.07–0.09 s**.

jevkit Speed is f(s′) = 100 − 20·log10(s′/0.1), with s′ = 2s + 0.15 and Speed = mean(f(p50′), f(p95′)). The p50 term alone:

| p50 | f(p50′) | vs 0.197 s |
|---|---|---|
| 0.197 (today) | 85.3 | — |
| 0.13 (fork2) | 87.7 | +2.5 |
| 0.11 | 88.6 | +3.4 |
| 0.09 (packed1, optimistic) | 89.6 | +4.3 |
| 0.067 (shim) | 90.9 | +5.6 |

Assume p95 scales like p50. The full shim gap was stated as about −4 Speed (≈ −0.6 Composite). On that basis:
- fork2 recovers about **+1.8 to +2.5 Speed (≈ +0.3 Composite)**, with no token or compute increase.
- packed1 could recover about +3 to +4 Speed, but it costs about K× compute.

## Adoption rule (from the timing receipt)

1. Run `tests/test_sym_fast.py` on the CPU job. All of it must pass, including the bitwise tests and the jevkit perturb G1 at max|dp| = 0.
2. Run the GPU timing with the JEV1-4B or JEV3 sym checkpoint: `python -X utf8 scripts/sym_fast_timing.py --cell ... --device cuda --out timing.json`.
3. Adopt `fork2` as the served default if its p50 is below 0.8× the old p50 and argmax agreement with the old path is ≥ 0.995.
4. Adopt `packed1` only if it beats fork2's p50 by more than 15%, AND the price class or the Cost axis absorbs the K× compute.
5. Then re-run the perturb G1 gate (`jobs.sym_eval.perturb_eval`) on the chosen mode.

Further work, only if the timing shows the 2 calls are still bound by per-call cost: CUDA-graph the continuation call (its shape is fixed per K bucket), or fuse the prefix and continuation calls with fla varlen `initial_state` (one launch, two segments).

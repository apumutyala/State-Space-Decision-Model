# Reorder and relabel invariance by construction (`sym-fork` layout)

Code: `heads/symmetric.py` (row builder), `heads/features.py` (e from the fork), `heads/heads.py` `FinalStateFork`.
Tests: `tests/test_sym_invariance.py`. Branch `stack/qwen35/sym-invariant`, cut from `stack/qwen35/jev1-4b`.
The default behaviour is unchanged: you opt in with `"layout": "sym-fork"` in the cell config.

## Statement

Take a choice question with options `(id_k, d_k)`, k = 1..K, where every `d_k` is a non-empty description. Let `P` be
the served distribution over ids from layout `sym-fork` with a symmetric head (`fork`, `integrator`, `head_b`; see A3).
Let `π` be any permutation of the order in which the request lists the options, and `σ` any bijection from the ids onto
a new set of id strings, with the descriptions, the state and the instructions held fixed. Then:

    P_{σ,π}(σ(id_k) | request) = P(id_k | request)   for every k.

Under A1 the two sides are bit-identical, not merely within 1e-6.

## Proof

1. **The model never sees an id or an order.** `build_row_sym` builds the prefix from `(state, instructions, K)`
   alone. Option k's continuation is `" " + d_k + tail`, and the id string appears nowhere in it.
   `option_ids` is metadata. It is used only after the model runs, to map axis index k back to an id (`to_wire`,
   `DecisionModelBackend.score`). So relabeling with σ changes `option_ids` and nothing else.
2. **The option axis depends only on content.** For choice questions the axis is sorted by the continuation token
   ids (ties are broken by canonical id order, but tied options have identical tokens). So the axis is a function of
   the multiset {d_k}. A reorder π or a relabel σ produces the same `ids`, the same `continuations` in the same order,
   and the same `option_seqs`. The `LayoutRow` that reaches the model is identical except for `option_ids`.
3. **The computation is a deterministic function of the row.** Under A1, identical rows give bit-identical log-probs.
   Mapping back through `option_ids` gives the statement.
4. **It is also structural, not only canonicalization.** Without step 2 (axis permuted by `option_order`, with
   different batch order and reduction order), the result is equivariant up to float rounding:
   - Option k's fork feature is `f_k = Backbone(prefix ⧺ cont_k)[last]`. The backbone is causal, the continuations are
     separate batch rows, and right padding is never read (positions > last are dropped). So `f_k` depends on k's
     text alone.
   - `FinalStateFork` computes `z_k = MLP([f_k, f_k − mean_j f_j])`. The masked mean is symmetric, so z is
     permutation-equivariant.
   - The choice output is `log_softmax(z)`, which is equivariant.
   - A composition of equivariant maps is equivariant.
   - `integrator`/`head_b` use `e_k` (the option's stand-alone encoding) and symmetric summaries. Their evidence scan
     runs over the shared prefix, which does not depend on the options.
   - Measured on tiny random models: max |Δp| = 3.5e-8 (fp32).

## Assumptions

- **A1, deterministic kernels.** The same input tensor with the same batch shape gives the same output. This holds
  for CPU torch and for cuBLAS without split-K autotuning. Some CUDA atomics-based kernels are not deterministic run
  to run. On GPU set `torch.use_deterministic_algorithms(True)` / `CUBLAS_WORKSPACE_CONFIG=:4096:8`, or accept a
  ~1e-3 bf16 jitter, which also hits two runs of the same request.
- **A2, descriptions carry the content.** An option with no description (or an empty one) is rendered by its id. Its
  id IS its content, and `meta["id_as_content"]` lists it. Relabeling such an option is not a symmetry of the task, so
  that part of the claim does not cover it. Reorder invariance still holds for it. The 231 public JevBench items have
  0 such options (849 options). D19's API-Bank name-only catalogs would have them.
- **A3, symmetric head.** It must have no positional encoding over options, use per-option shared parameters, and
  aggregate across options only through symmetric pooling. `clef_control` (a TransformerEncoder over options, no
  positional encoding) is also equivariant. `qend_axes` reads `e`, which is equivariant. A head that indexes options
  by position would break the claim.
- **A4, the claim is about the readout, not training.** Any training run yields an invariant model. A position prior
  in the training data cannot be learned, because the model never sees a position.
- **A5, padding and batch composition.** Within a row, the continuation batch (and its `cont_chunk` chunks, 4b0fb1d)
  and the padding width (`max len`, `MIN_CONT_WIDTH`) are functions of the content multiset (step 2), so they are
  invariant. Across rows, the shared-prefix eval path and `row_batch` group rows by request. A different
  co-batched request can change float rounding under A1-violating kernels only.

## Where it fails (by design or out of scope)

- **Score questions.** The level order is the question (ordinal). Levels are never permuted (`option_order` refuses),
  and `CumulativeLink` uses that order on purpose. Score ids are "0".."n−1", so no relabel exists.
- **Noul questions.** These are always (true, false). jevkit's `option_reverse` on noul is invariant because
  `canonical_options` fixes the order.
- **Text that names ids.** If the instructions or the state mention ids ("pick B if …"), relabeling the criteria
  alone changes the meaning. That is not a symmetry, so the claim is not about it.
- **Duplicate descriptions.** Two options with equal descriptions get equal probability. That is correct, but the
  model cannot separate them.
- **`sym-fork-ids` (ablation).** It renders `"<id>: <desc>"`. It is reorder-equivariant but not relabel-invariant:
  the tests measure a 1.3e-3 shift.
- **Cross-question order (D19 joint head).** That is a separate axis. See the port notes.

## Leak map: where order and ids reach the scores today

| # | Where | Qwen JEV1-4B (`g1v2` + `jrt-h2o-shim`, fork head) | D19 (`g1v2` + v3w, `joint_record`) |
|---|---|---|---|
| 1 | Prompt rendering | `"A) <id>: <desc>"` lines in REQUEST order (b6c3362). Ids enter the model, and a causal reader sees option j after the options before it. | `"<CODE>. <id>: <desc>"` in sorted-id order. Ids enter, plus causal order. |
| 2 | Answer token | Continuation = the label of the option's POSITION (A, B, …), so the score depends on position. | v3 code of its sorted position, or (word ids) the first token of `" <id>"`: position and id. |
| 3 | Sorting by id | Option axis = `canonical_options` (sorted id). With a symmetric head only the float reduction order depends on it. | Same, plus the code assignment (row 2). |
| 4 | Positional features | `e` = mean over the option's span in the prompt. That span holds the id and is position-conditioned, and it feeds `CumulativeLink` (score only). | Same `e`. The joint head's `v` = fork + lexical vector. |
| 5 | Lexical prior keyed by id | none | `lexical_vectors` = frozen-embedding mean over the schema span `"<CODE>. <id>: <desc>"`, so it carries the id and the position code. `lex_scale·cos(lex, q_end)` adds a per-id prior. |
| 6 | Joint ordering | n/a (one question per prompt) | Record mixer = a bidirectional recurrence over FIELDS in schema (sorted question-id) order. Within a field the options are pooled symmetrically (mean/max), so it is invariant in options and order-dependent in questions. |
| 7 | Padding / chunking | `cont_chunk` chunks the axis in sorted-id order (4b0fb1d), and `MIN_CONT_WIDTH` right-pads. Exact in math, but batch composition follows ids (float level). | `row_options` record chunks, same float caveat. |
| 8 | Training data | Gold position is kept as listed (chi² p ≥ 0.13). A position prior is learnable through rows 1–2. | v3b is shuffled for MMLU only. |

`sym-fork` removes rows 1–4 and 7 for the Qwen line. Row 5 never entered it, and row 8 becomes unlearnable (A4).

## Porting to D19 (`stack/rwkv7/d19-v3b-joint`)

- **Rows.** `build_row_sym` is template-generic. A v3w-family port needs a `pieces` renderer in the RWKV7-G1 chat
  format (`User: … Assistant: <think></think>` with the prefill), whole-string tokenized through the encoder's
  `token_bytes`, as `_Tok` already supports. Expect about 40 lines of code.
- **Lexical prior.** Point `lexical_vectors` at `option_seqs` (the description only) for sym rows instead of the
  schema span. That is a one-line branch, and the prior becomes id-free.
- **Joint mixer.** It is option-invariant already. For question-order invariance, replace the schema-order
  bidirectional scan with a DeepSets field mixer (`c_i = ρ(Σ_{j≠i} φ(x_j) + φ_p(payload))`). That keeps the
  exclusive-context and zero-init properties and is a drop-in for `_scan`. Optional: the reorder/relabel claim
  concerns options.
- **Record grouping.** `record_key` uses the shared prefix. Sym prefixes are per question (no schema of all fields),
  so set `meta["record"]` from the request, as training already does.
- **Fork backend.** `_state_fork` / `fork_features_rows` take multi-token continuations as they are. RWKV-7 has no
  KV, so a state copy costs O(1) in T, and the cost below is lower on RWKV than on Qwen.

## Cost note (JevBench Speed / Cost axes)

jevkit Cost = USD per 1,000 decisions, computed from the `usage.input_tokens` each answer reports times the declared
price (meter.py). `DecisionModelBackend` reported `len(row.ids)`, the prefix only. On this branch the sym layouts
report prefix + every continuation (`row_input_tokens`), because the option texts are fed to the model. Counting the
prefix alone would show a 6 % cut that is an accounting artifact. Other layouts keep the old count. The g1v2 fork
also feeds K one-token labels it does not report (+0.6 %).

These numbers come from the real Qwen3.5-4B tokenizer over the 231 public items (CPU, this branch):

| per decision | JEV1-4B g1v2 shim | sym-fork | sym-fork, 1-token tail (`<|im_end|>` only, not built) |
|---|---|---|---|
| input tokens, mean / median / p95 | 653.6 / 157 / 2,647 | 687.4 / 208 / 2,680 (prefix 614.9 + continuations 72.5) | ≈ 650.6 |
| change | — | **+5.2 %** | −0.5 % |
| K-axis change (−30·log10 ratio) | — | **−0.66 pt** | +0.07 pt |

The per-option chat tail (`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:`) is 11 tokens, about
40 per decision. It is the whole cost overhead. A tail-lite variant reads the head at the `<|im_end|>` token and would
be cost-neutral. It is a separate, priced option.

Latency (Speed axis). Both layouts already make two backbone calls per decision: one prefix pass plus one batched call
over the K continuations (cache fork). sym-fork's prefix is 40 tokens shorter. Its continuation call grows from K×2
tokens to K×~22 tokens (mean 73, max 187), still one call. On the timed tiers (mean 134 → 172 fed tokens: prefix 106
plus continuations 66), a 4B prefill at this size is dominated by per-call overhead. The estimate is +0 to +5 ms
raw. After jevkit's adjustment (s′ = 2s + 0.15) that is ≤ 0.3 S-points, and it needs one GPU measurement.

Training cost rises with K. A jev1 row with K = 77 feeds 77 × ~22 = ~1,700 continuation tokens instead of 154, and
the K cache copies (~0.7 GB each at T 1,020, cf3929b) are unchanged. Re-price the clock with the step-10 rate check
(exit 26) before committing the 270-minute budget. The `keep_item` cap for sym continuations is 128 tokens
(`SYM_MAX_CONT`).

## Proposed pre-registered cell (A-PREREG-JEV1-4B-SYM; write it to PREREG.md before the result)

- **Arm.** `jev1_q35_4b` exactly: Qwen3.5-4B, shim template, LoRA r64, fork head, Clef loss, bschema-draw-jev1,
  1,121 steps, seed 0. Change ONLY `layout: g1v2 → sym-fork`. Re-price the clock from the step-10 rate.
- **Comparator.** The JEV1-4B native arm. Use the same jevkit public items and the same paired stratified resamples
  (`screen.paired_vs`).
- **G1, guarantee (must hold).** jevkit `perturb` operators 4–6 (`option_reverse`, `option_shuffle`, `key_relabel`)
  on every applicable item must give flip rate = 0 and max |Δp| ≤ 1e-3 (bf16 jitter, A1). Report the
  deterministic-kernel run with the bit-identical row check.
- **G2, non-inferiority.** Paired ΔI 95 % CI lower bound > −2.0, and ΔComposite point estimate ≥ −0.7 (the priced
  +5.2 % tokens).
- **Keep if G1 ∧ G2. Claim "better" only if** the paired ΔComposite CI is > 0. Also report operators 1–3 and 7–8
  against JEV1-4B.

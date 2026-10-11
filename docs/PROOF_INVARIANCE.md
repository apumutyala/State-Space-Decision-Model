# Reorder and relabel invariance by construction (`sym-fork` layout)

Code: `heads/symmetric.py` (row builder), `heads/features.py` (e from the fork), `heads/heads.py` `FinalStateFork`.
Tests: `tests/test_sym_invariance.py`, `tests/test_sym_perturb_eval.py`, `tests/test_sym_g15_bpe.py`. Branch `stack/qwen35/sym-invariant`, cut
from `stack/qwen35/jev1-4b`. The default behaviour is unchanged: you opt in with `"layout": "sym-fork"` in the cell
config.

Layouts (renderer version `sym-fork-v3`; v3 = v2 + trust-segment tokenization, see "Trust segments" below):
- `sym-fork` is the default. Each option continuation is `" " + description` plus the user-turn terminator: Qwen
  `<|im_end|>` (one token), or the RWKV7-G1 `"\n\n"`. The head reads the fork feature at that terminator.
- `sym-fork-chat` appends the full answer turn instead (+10 tokens per option on Qwen).
- `sym-fork-ids` is the ablation that also renders the id.

Formats: the Qwen H2O-shim chat turn, the RWKV7-G1 vendor format (`jrt-rwkv-g1-*` templates, `g1_pieces`), and any
plain template.

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
  0 such options (628 choice options). D19's API-Bank name-only catalogs would have them.
- **A3, symmetric head.** It must have no positional encoding over options, use per-option shared parameters, and
  aggregate across options only through symmetric pooling. `clef_control` (a TransformerEncoder over options, no
  positional encoding) is also equivariant. `qend_axes` reads `e`, which is equivariant. A head that indexes options
  by position would break the claim.
- **A4, the claim is about the readout, not training.** Any training run yields an invariant model. A position prior
  in the training data cannot be learned, because the model never sees a position.
- **A5, padding and batch composition.** Within a row, the continuation batch (and its `cont_chunk` chunks, 4b0fb1d)
  and the padding width (`max len`, `MIN_CONT_WIDTH`) are functions of the content multiset (step 2), so they are
  invariant. Across rows, the shared-prefix eval path and `row_batch` group rows by request. A different
  co-batched request can change float rounding under A1-violating kernels only. Measured: on cpu-8 / torch 2.13
  (job-8kqt9, job-44chn) one question scored alone vs batched with two others differs by max |Δp| 2.6e-8 (seed 1),
  on this branch and on 42388c2c alike (SYM_CHECK saw 0 on torch 2.14.1). The bitwise claim is per request
  (same batch composition); across co-batched requests it holds only up to such rounding.
- **A6, content normalisation: leading and trailing whitespace is stripped.** A description is `.strip()`ped before
  it is rendered (`option_text`), so `" yes"`, `"yes"` and `"yes \n"` are ONE content: bit-equal probability and one
  `tie_set` (exact duplicates). Inner whitespace is content (`"a b"`, `"a  b"`, `"a\tb"`, NBSP all differ). The
  instructions are stripped too (`_instr_text`). "Content" in the statement means the stripped description.

## Trust segments (G15 under a real BPE tokenizer)

The protected stream of G15 reads only TRUSTED tokens. With whole-string tokenization a BPE merge can cross a trust
boundary: under Qwen, an undeclared description `"Refund the customer."` absorbed the template space before it
while `"123 refund"` left it a separate trusted token, and a raw-string state ending in `"\n"` merged with the trusted
`"\nquestion"` (trusted prefix 79 → 78 tokens). The trusted token list then depended on untrusted text
(claude/ev-sym-check `SYM_CHECK.md` section 5). Since v3 the row is tokenized SEGMENT BY SEGMENT and the ids are
concatenated: the prefix is cut at the state and question boundaries, and each continuation into its request part
`" " + description` and its template tail. Each request segment carries the one template space right before it
(so BPE still sees the natural `" word"` token, and the template text left over is fixed). No token crosses a trust
boundary, so every template token is the same whatever the request text is. Test:
`tests/test_sym_g15_bpe.py` (real Qwen3.5-4B tokenizer, fuzzed boundary texts; plus a teeth test that the v2
whole-string tokenization leaks on the same tokenizer). Byte-level encoders never merge, so their ids are unchanged.


## Decisions: ties, duplicates, empty descriptions

The distribution statement above covers probabilities. A single answer id (`choice`) also needs a tie rule that uses
neither the position nor the id string. `decide.schema.wire_answer(tie=...)` provides one for sym rows; the default
path is unchanged (smallest id).

- **Different contents, exactly tied.** The winner is the content with the smallest content hash (sha256 of its
  continuation tokens). It is never the smallest id or the first index.
- **Exact duplicates (equal content).** By step 1 they get exactly equal probability, so the mass is split equally.
  The decision is the `tie_set`: every id that carries the winning content. `choice` names the set's smallest id and
  `tie_set` lists the whole set.
- **Theorem (no single-id rule exists).** Let options a ≠ b have equal content. The relabel σ that swaps their ids
  and fixes everything else leaves the model input unchanged, because the ids never enter the model. Any
  relabel-invariant single-id rule would then have to pick both a and σ(a) = b, a contradiction. So the decision is
  invariant only as a set. Tested in `test_duplicate_options_form_a_tie_set_invariant_as_a_set`.
- **Wire rounding.** At 4 digits, near-ties become exact ties on the wire, and jevkit's frozen `argmax_label` then
  breaks them by smallest id: a relabel leak at the wire. Sym answers are therefore written with 12 digits
  (`SYM_WIRE_DIGITS`).
- **Host action boundary.** `decide.adapter.select_actions` uses the same rule: the sym Decider puts the content
  keys on each policy-conditioned stage (`content_key`), exact ties among the allowed options go by content hash, and
  duplicates come back as `tie_set`. Without keys (non-sym rows) it keeps the smallest id.
- **Softmax order.** The adapter's softmax over a sym row runs in content order (`content_order`), so the sum order
  cannot depend on the ids.
- **Empty descriptions.** Such an option is rendered by its id, which then is its content (A2, `id_as_content`).
  Reorder invariance still holds exactly. Relabeling such an option changes its content, so the claim does not cover
  it. Tested in `test_empty_descriptions_reorder_invariant_and_flagged_as_id_content`.

## Where it fails (by design or out of scope)

- **Score questions.** The level order is the question (ordinal). Levels are never permuted (`option_order` refuses),
  and `CumulativeLink` uses that order on purpose. Score ids are "0".."n−1", so no relabel exists.
- **Noul questions.** These are always (true, false). jevkit's `option_reverse` on noul is invariant because
  `canonical_options` fixes the order.
- **Text that names ids.** If the instructions or the state mention ids ("pick B if …"), relabeling the criteria
  alone changes the meaning. That is not a symmetry, so the claim is not about it.
- **Exact duplicates.** The decision is a set (see above). A scorer that needs one id among duplicates (jevkit's
  `argmax_label`, smallest id) can count a relabel "flip" on such items. `jobs.sym_eval.perturb_eval` lists them under
  `duplicate_tie_items`. The 231 public items have none (628 choice options, all distinct texts; checked on CPU).
- **`sym-fork-ids` (ablation).** It renders `"<id>: <desc>"`. It is reorder-equivariant but not relabel-invariant:
  the tests measure a 1.3e-3 shift.
- **Cross-question order (D19 joint head).** That is a separate axis. See `docs/D19_SYM_PORT_PLAN.md`, the port
  plan (doc only; the RWKV branches are untouched).

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
| 8 | Ties and wire | `wire_answer` breaks ties by smallest id. 4-digit rounding makes near-ties exact ties, which jevkit then breaks by smallest id. | same |
| 9 | Training data | Gold position is kept as listed (chi² p ≥ 0.13). A position prior is learnable through rows 1–2. | v3b is shuffled for MMLU only. |

`sym-fork` removes rows 1–4, 7 and 8 for the Qwen line. Row 5 never entered it, and row 9 becomes unlearnable (A4).

## Eval path for the guarantee (G1) and the independent cross-check

- **jevkit eval path.** `jobs/sym_eval/perturb_eval.py` loads any checkpoint the way the screen does (`load_child`,
  untempered arm). `--layout` scores it under another layout, e.g. the JEV1-4B checkpoint under g1v2 for its
  baseline. It runs jevkit perturb operators 4–6 through `SystemOneAdapter` and writes the files
  `jevkit perturb score` reads, plus `perturb_report.json`. That report holds `g1_pass`, the per-operator flip
  report, `duplicate_tie_items`, and the row-identity check (every perturbation must yield the same ids and
  continuations). The verdict (`g1_verdict`) FAILS CLOSED: every requested operator must be present with n > 0 (an
  operator that yields no perturbation, e.g. `key_relabel` on noul-only items, fails instead of passing vacuously),
  with 0 flips, 0 invalid and max |Δp| ≤ `--max-dp`, default 0 (bitwise, as the proof claims; the observed max
  |Δp| is reported apart as `max_abs_dp_observed`), AND `rows_all_identical` (no row check = no pass).
- **CPU tests.** sym-fork gives 0 flips with max |Δp| = 0, and every row is identical. The `ordered` layout moves
  under `key_relabel`. `jevkit perturb score` reproduces the report.
- **Independent harness.** This is the evidence lane's `tests/guarantees/test_reorder_invariance.py`
  (claude/ev-guarantee-proofs @f7170cb). The plug is `SSDM_INV_HEAD=jobs.sym_eval.inv_adapter:sym_fork_score`, which
  runs the full served path on a tiny Qwen3.5 hybrid.
  - sym-fork passes at tol 0: all 11 tests, including all K! orders for small K and the relabel+reorder checks.
  - `inv_adapter:ordered_score` fails relabel: 7 flips, max |Δp| 2.0e-3.

## Cost note (JevBench Speed / Cost axes)

jevkit Cost = USD per 1,000 decisions, computed from the `usage.input_tokens` each answer reports times the declared
price (meter.py). `DecisionModelBackend` reported `len(row.ids)`, the prefix only. The sym layouts report prefix +
every continuation (`row_input_tokens`), because the option texts are fed to the model. Counting the prefix alone
would show a cut that is an accounting artifact. Other layouts keep the old count. The g1v2 fork also feeds K
one-token labels it does not report (+0.6 %).

These numbers come from the real Qwen3.5-4B tokenizer over the 231 public items (CPU, this branch; asserted in
`test_sym_fork_is_cost_neutral_on_the_public_jevbench_items`):

| per decision | JEV1-4B g1v2 shim | **sym-fork** (`<|im_end|>` tail, default) | sym-fork-chat (full answer turn) |
|---|---|---|---|
| input tokens, mean | 653.6 | **650.6** (prefix 614.9 + continuations 35.7) | 687.4 |
| change | — | **−0.5 %** | +5.2 % |
| K-axis change (−30·log10 ratio) | — | **+0.07 pt** | −0.66 pt |

Latency (Speed axis). Both layouts make two backbone calls per decision: one prefix pass plus one batched call over
the K continuations (cache fork). sym-fork's prefix is 39 tokens shorter. Its continuation call grows from K×2 tokens
to K×~12 tokens, still one call. The estimate is ±3 ms raw, below 0.2 S-points after jevkit's adjustment
(s′ = 2s + 0.15). It needs one GPU measurement.

**Training-clock re-price.** `scripts/sym_reprice_jev1.py 24` on a systematic 1-in-24 sample of bschema-draw-jev1
(4,000 items, all kept by `keep_item` under both layouts; K mean 4.78, max 16; real tokenizer):

| per row | g1v2 | sym-fork |
|---|---|---|
| tokens the backbone runs (prefix + continuation calls right-padded per 64-continuation chunk) | 368.1 | 362.7 |
| tokens fed, no padding | 363.3 | 352.1 |

- The factor is 0.985: the options move from the prefix into the continuations, which is about token-neutral.
- Projected step time: 10.92 × 0.985 = 10.76 s/step, so ceil(1,121 × 10.76 × 1.3 / 60) = 262 min.
- `max_minutes` stays 270, the measured g1v2 budget. The factor comes from a token model, not a measurement; the
  in-job step-10 rate check (exit 26) re-measures it.
- The draft cell `jobs/m2_first_cell/m2_first_cell_rwkv7_jev1_q35_4b_sym.json` (`sym_reprice_note`) records this.
  It has no job_params yet.

## Proposed pre-registered cell (A-PREREG-JEV1-4B-SYM; write it to PREREG.md before the result)

- **Arm.** `jev1_q35_4b` exactly: Qwen3.5-4B, shim template, LoRA r64, fork head, Clef loss, bschema-draw-jev1,
  1,121 steps, seed 0. Change ONLY `layout: g1v2 → sym-fork`. The clock stays 270 (re-price factor 0.985).
- **Comparator.** The JEV1-4B native arm. Use the same jevkit public items and the same paired stratified resamples
  (`screen.paired_vs`).
- **G1, guarantee (must hold).** `jobs.sym_eval.perturb_eval` runs jevkit perturb operators 4–6. It must give 0 flips,
  max |Δp| = 0 (bitwise; deterministic kernels on, A1; `max_abs_dp_observed` reported), every operator present
  with n > 0, `rows_all_identical`, and no `duplicate_tie_items` among the flips.
- **G2, non-inferiority.** Paired ΔI 95 % CI lower bound > −2.0, and ΔComposite point estimate ≥ −0.3 (the token
  cost is now neutral).
- **Keep if G1 ∧ G2. Claim "better" only if** the paired ΔComposite CI is > 0. Also report operators 1–3 and 7–8,
  and the JEV1-4B flip baseline (`--layout g1v2`).

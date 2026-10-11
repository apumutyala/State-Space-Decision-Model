# D21: the attention-free RWKV-7 cell with two built-in guarantees

Branch `stack/rwkv7/sym-contractive`, cut from `stack/rwkv7/d20-v4-chain` @9323085c (it carries the chain resume
code). Design only: no GPU job has run this code. Both upgrades are opt-in config flags. With both off, the code path
is bit-identical to the parent (receipt: `tests/test_sym_contractive.py::test_off_path_bit_identical_to_parent`).

## What a D21 cell is

**D21 = the D20 recipe + (A) + (B) + jev3 data.**

| Part | Value |
|---|---|
| Backbone | RWKV7-G1k-13.3B, attention-free, LoRA as in D19/D20 |
| Head | `joint_record_sym`: D19's `joint_record` with a DeepSets record mixer in place of the bidirectional scan |
| Layout | `sym-fork` with a G1 template (`jrt-rwkv-g1-v2` is the tested one). Any `jrt-rwkv-g1-*` id renders the same G1 vendor format. |
| Recurrent state | `rwkv7_scalar_a: true`: the in-context rate `a` is shared per head |
| Training | the D20 chained 3-link schedule (cosine horizon fixed in link 1, ~1,694 min total), preceded by a short `a`-distillation stage (see Risk) |
| Data | the D20 bschema-v4 mix plus the jev3 additions (the evidence lane's jev3 plan on `state` BOARD: human-labelled rubric data for judge/policy/ambiguous, score-type rows up-weighted on the a-priori I-weight argument, the shape gaps closed). Not built yet. |

Config keys, all off by default:

- `"layout": "sym-fork"` (or `sym-fork-chat`; `sym-fork-ids` is the ablation that renders ids).
- `"head": "joint_record_sym"`.
- `"rwkv7_scalar_a": true`.

## (A) Reorder and relabel invariance by construction

**Code.**
- `heads/symmetric.py` is the port from `stack/qwen35/sym-invariant` @55bfdd39, without the Qwen shim branch.
- Hooks:
  - `heads/layout.build_row` dispatches the sym layouts.
  - `heads/features.py`: under a fork head, the option embedding `e` is the option's own fork feature.
  - `heads/model.lexical_vectors`: sym rows read `option_seqs[k]`, the description tokens alone. This is D19
    leak-map row 5.
  - `heads/decide_backend.py`: the token count includes the continuations, and `sym_version` goes into the identity.
- `heads/heads.JointRecordSymHead` is the new head (step 4 of the port plan).

**Guarantee.** For a choice question whose options all have descriptions, the per-option probability depends only on:
- the state;
- the question;
- the option COUNT;
- the multiset of option descriptions.

It does not depend on option order or on option ids. This holds bit-for-bit, because the row is literally the same
token sequence: the option axis is content-canonical and the id never enters the model.

The record mixer is symmetric in the record's questions. Question order changes `c` only through float summation
order, measured at ≤ 1e-6.

**Assumptions and limits.**
1. A description-free option, such as an API-Bank name-only catalog, is rendered by its id. For it, relabel
   invariance cannot hold. Reorder invariance still does. These options are listed in `meta["id_as_content"]` and
   should be reported apart.
2. Score levels keep their order, because it is part of the question.
3. Duplicate descriptions get equal probability, so a single-id decision among them is impossible by symmetry.
4. Wire rounding (ported from the Qwen branch):
   - `decide.schema.wire_answer(tie=…)` gets the tie info from `heads.symmetric.sym_tie_info`, through
     `DecisionModel.decide`, `DecisionModelBackend` features and `Decider`. `select_actions` applies the same rule.
   - The decision is taken at full precision, BEFORE rounding:
     - exact ties between DIFFERENT contents go by content hash;
     - exact duplicates form `tie_set`.
   - Probabilities are written with 12 digits, so 4-digit rounding cannot manufacture a tie that a reader breaks by
     id.
   - Receipts are the `test_A_wire_*` tests:
     - a near-tie 3e-5 apart flips the jevkit reader under relabel with the old rule (teeth) and does not with the
       sym rule;
     - `DecisionModel.decide` output is relabel- and reorder-invariant end to end.
   - Non-sym layouts pass `tie=None`, so their output is unchanged.
5. Record grouping relies on the shared state prefix. This holds for every question of one request, because the sym
   prefix is `head + record: <state>` up to the state's end.

**Receipts (CPU, tiny random RWKV-7, fp32).**
- The evidence harness `tests/guarantees/test_reorder_invariance.py`, copied unchanged from
  `claude/ev-guarantee-proofs` @f7170cb, runs with `SSDM_INV_HEAD=heads.inv_adapters:rwkv7_sym_fork`. It passes
  reorder and relabel at tolerance 0.
- D19's own layout (`heads.inv_adapters:rwkv7_g1v2_v3w`: v3w template, g1v2 rows, joint_record) FAILS relabel. These
  are the teeth.

## (B) Contractive-by-construction recurrent state

**Code.** `heads/rwkv7_scalar_a.py`.

`enable_scalar_a(model)` swaps every `Rwkv7Attention.lora_gates` through an instance attribute, so the vendored files
stay unmodified. The swap replaces the per-channel gate `a_c` with

    a_h = sigmoid(logit(mean_{c in head} a_c) + b_h)

broadcast over the head's channels, where `b_h` is a trainable per-head logit bias, zero-initialised. Properties:
- At `b = 0`, `a_h` is EXACTLY the head mean of the pretrained per-channel `a`, so the swap needs no new projection.
- `a_h ∈ (0,1)` for all parameters, so fine-tuning cannot break the guarantee.
- The swap is loaded with `load_rwkv7_hf_backbone(..., scalar_a=True)` (cell key `rwkv7_scalar_a`).
- `train/lora.py` keeps `*.scalar_a.logit_bias` trainable through `apply_lora`, and puts it in the LoRA optimiser
  group and the LoRA export.

**Guarantee (docs/STATE_STABILITY.md T2(c), on `claude/lead-state-stability` @836bc6f6).** Each head's step
transition `A_t = diag(w_t) − a_t κ_t κ_tᵀ` is symmetric, so `‖A_t‖₂ ≤ w_max,t ≤ w̄ < 1` for every input, where w̄ is
the per-layer decay ceiling from the weights. The per-step certificate `U_t = w_max + σ_a` collapses to `w_max`, since
σ_a = 0. Consequences:
- **T3 ISS.** The bound `‖S_t‖ ≤ ρ^t‖S_0‖ + b/(1−ρ)` holds uniformly with ρ = w̄.
- **T5.** A fully recurrent stack with one uniform ρ < 1 now exists, so T5's structural cascade bound applies (still
  structural, with path-Lipschitz constants).
- **Injection persistence.** The influence of an injected span decays at rate ≤ w̄ per step in layer 0 (T4) and
  ≤ poly(τ)·w̄^τ deeper (T5).

**Assumptions and limits.**
1. Exact arithmetic. The chunked WKV casts to fp32 before `exp(w_log)`. A bf16 recurrent path can round w to 1.0
   (F3(v)). In that case only ρ ≤ 1 holds on such steps.
2. bf16 state storage needs `ρ(1 + c·u) < 1`.
3. The output norms (GroupNorm) add local constants (F6).
4. This is a statement about the state, not a safety claim on its own (SAFETY_SPEC §5.2). Decay bounds how long an
   injection persists, not whether it gets in.
5. Identity, which fails closed:
   - `train.lineage.backbone_config_identity` stamps `rwkv7_scalar_a: true` when the swap is installed, so the
     trainer's resume / inference lineage diff refuses a checkpoint of the other kind.
   - `weights_digest` excludes the trainable bias, as it excludes LoRA.
   - The sym layout is already in the lineage's `layout` identity, and the backend identity carries `sym_version`.
   - `expected_parent_identity` adds `scalar_a: true` for a scalar-a child, so `read_arm_cache` refuses a stock-parent
     arm.
   - TODO before a D21 Index gate: `jobs/m2_parent_arm` must be able to score the parent WITH the swap at init and
     stamp `scalar_a` in its arm identity. Until then the gate refuses, which is the fail-closed direction.
   - The off path adds no key, so existing D19/D20 caches still match.

**Receipts (CPU, tiny random RWKV-7).** The WKV inputs are recorded on random token batches. For each (layer, batch,
step, head):
- exact SVD norm ≤ certificate = w_max < 1;
- power iteration ≤ exact.

This holds on 2 weight settings, one of them adversarial (decay ≈ 0.9998, per-channel `a` alternating ≈ 1/0 within
each head), and also after the trainable bias is randomised.

Teeth:
- The SAME adversarial weights with the flag off give real steps with `‖A_t‖₂ > 1`.
- The T2(e) two-step alternation has ρ(A₂A₁)^{1/2} > 1.1 and `‖(A₂A₁)^{100}‖ > 1e8`, while the same steps with the
  head-shared rate contract below 0.99²⁰⁰.

## Expected cost (from the ledger)

| Item | Basis (results/ledger.tsv on state) | D21 estimate |
|---|---|---|
| Training | D19 job-fi5xf: 380-min clock, **$30.55** actual (≈ $0.080/min). D20: 3 links, 1,694 min total, **$56.64 hold per link** (l2+l3 ≈ $113 of holds, BOARD #13) | 1,694 min × $0.080 ≈ **$136 actual** at token factor 1.0; **$136–157** at a sym-fork token factor of 1.00–1.15 (Qwen measured 0.995 on JevBench public; RWKV v3b K up to 79 pads per 64-chunk, so re-run the census first, port plan step 5). Holds ≈ 3 × $57–65. |
| `a` distillation | new stage | ~60–90 GPU-min on the D19 checkpoint ≈ **$5–8** |
| Full-DEV eval | D19 full DEV job-y6ste+uq5p5: **$28.64** | **≈ $29–33** (sym rows feed the continuations) |
| JevBench public | D19 job-9j3uu: **$1.44** | **≈ $1.5** |
| Flip / perturb eval | B-PREREG-FLIP run 2 job-andnb: **$2.27** | **≈ $2.3** |
| **Total** | | **≈ $175–200 actual** (holds ≈ $210–230). With jev3 data generation on CPU, $0 GPU. |

At today's balance this is not fundable together with D20 l2/l3 (BOARD #13 parks those for the same reason). A
cheaper first read is a **D21-screen**:
- one 380-min link (D19 clock) from the D19 checkpoint with (A) + (B) on, then the proxy600 screen;
- ≈ $31 + $8 + screen ≈ **$45**;
- comparator: D19's proxy600 48.61 [44.10, 53.08].

## Risk and mitigation

1. **The `a` reparameterisation may cost accuracy.** It removes per-channel in-context learning rates, which is a real
   expressivity cut: the delta-rule erase becomes isotropic within a head. Mitigation is a short **distillation from
   D19's per-channel `a`** before the task loss:
   - Freeze everything but the LoRA and `scalar_a.logit_bias`.
   - Run D19's checkpoint (flag off) as the teacher and the flag-on student on the training prefixes.
   - Minimise `Σ_layers ‖h_student − h_teacher‖²` on the block outputs (cheap, no labels). Optionally add the KL of the
     joint head's choice distribution.
   - Run ~200–400 steps, then start the D20 schedule from the distilled weights.
   - Gate: the proxy600 drop vs D19 must be ≤ 1.5 pts at the screen. Otherwise fall back to the soft version, a σ_a
     penalty under the νPI controller (STATE_STABILITY.md §5), which keeps a certificate per input but not for every
     input.
2. **The sym-fork layout drops the schema block.** D19's v3w prompt shows every field's options in the shared prefix.
   The sym prefix shows only the option count. The joint mixer recovers cross-field context from the other
   questions' pooled option tokens, but the screen has to show it. If it costs more than the leak it removes (D19
   relabel flip 8.6%, job-andnb), keep (B) alone as D21b.
3. **Wire-level relabel leak.** Closed: see (A), assumption 4. The adapter path (`Decider`) is covered by the ported
   code, but it has no test of its own on this line. The evidence lane's flip test is the end-to-end check.
4. **Throughput.** Sym rows are scored as K forked continuations. On RWKV each costs one fixed-size state copy and no
   KV cache, so cost is O(K) copies independent of T. Price the K = 77–79 rows before packaging.

## Tests (remote only)

The `cpu_tests` job (claude/cpu-tests-job) with suite `rsc` runs on this branch's tree and the parent tree
(`stack/rwkv7/d20-v4-chain` @9323085c):
- `tests/test_sym_contractive.py`: (A), (B), teeth, and OFF = parent through `RSC_PARENT_GOLDEN`, which comes from the
  parent tree's `tests/rsc_off_golden.py` output.
- The evidence harness, run with `SSDM_INV_HEAD=heads.inv_adapters:rwkv7_sym_fork`.
- The neighbouring regression files: `test_joint_record_head.py`, `test_g1v3w_train_rows.py`, `test_equivariance.py`
  and `test_heads.py`.

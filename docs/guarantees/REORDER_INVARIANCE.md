# Reorder and relabel invariance by construction

CONTROL_RESEARCH mechanism #1. Status: **proof + property-test harness**. The real head (lead, `stack/qwen35/sym-invariant`) plugs into `tests/guarantees/test_reorder_invariance.py` the day it lands.

## Takeaway

- If every option's score is computed from (trusted state, option content) only, with no option position and no option id, and the scores go through softmax over the set, then reordering, reversing or shuffling the options cannot change any option's probability. Reorder flip rate is exactly 0. This is a proof, not a sample.
- Relabel invariance needs one more condition: the id never enters the score, or it enters only through a code that is a function of the option content.
- In floating point, "exactly" becomes "within a stated tolerance". Bitwise equality holds when the per-option arithmetic is the same and the softmax sums in a fixed (canonical) order. Otherwise `max|Δp| ≤ δ/2 + γ_K` and the argmax is unchanged whenever the top-two logit margin exceeds `2δ` (definitions in section 4).
- **Finding on the current code.** `controlsafe/g16.py` adds per-option-position parameters `u_k` and `beta_k` to the judge (`a_k = A h_P[k] + u_k`, `b_k = w·h_P[k] + beta_k`). That is condition (C2) below failing by design. The G16 head is reorder-invariant only if the canonical order it uses is a function of the option set. If that order sorts by id, it is still not relabel-invariant. The sym-invariant G16 head must drop `u_k` and `beta_k` or replace them with a position-free term.

## 1. Setup

- A decision has a trusted state `x` (the shared prefix: state, question, instructions) and an option list `O = (o_1, …, o_K)`, where `o_i = (id_i, c_i)`. `id_i` is the wire id. `c_i` is the option content the model reads.
- Ids are distinct. Contents need not be.
- A head maps `(x, O)` to a probability vector `p(x, O) ∈ Δ^{K}`, indexed by position. The **per-id distribution** is `P_O(id_i) = p_i(x, O)`.
- A **reorder** is a permutation `π` of `{1..K}`: `O^π = (o_{π(1)}, …, o_{π(K)})`. Reverse and shuffle are special cases.
- A **relabel** is a bijection `σ` from ids to fresh ids, with content unchanged: `O^σ = ((σ(id_1), c_1), …)`.
- **Invariance (equivariance at the position level).**
  - Reorder: `P_{O^π}(id) = P_O(id)` for every id and every `π`.
  - Relabel: `P_{O^σ}(σ(id)) = P_O(id)` for every id and every `σ`.
- **Flip:** the predicted id changes after mapping back. Under ties, the comparison uses the argmax set (section 3).

## 2. Theorem (real arithmetic)

**Conditions.**

- **(C1) Shared state is order-free.** `x` is computed without reading the option list, or it reads the list only in a canonical order that is a function of the multiset `{c_i}` (for example, sorted by a content hash).
- **(C2) Per-option score is position-free and id-free.** There is a function `f` with `z_i = f(x, c_i) ∈ ℝ ∪ {−∞}`. In particular, no option-slot embeddings, no per-slot parameters, and no id tokens in the continuation.
- **(C3) Set normaliser.** `p_i = exp(z_i) / Σ_j exp(z_j)`, or any normaliser `g` with `g(z∘π) = g(z)∘π` (temperature, top-k masking by value, and vetoes that are a function of `(x, c_i)` all qualify).
- **(C4) Order-free tie-breaking.** The predicted label is chosen by a rule that does not look at position: either the full argmax set, or a deterministic tie-break on content.

**Claim.**

1. Under (C1)–(C3), for every permutation `π`: `p(x, O^π) = p(x, O)∘π`. So the per-id distribution is identical, and `max|Δp| = 0`.
2. Adding (C4), the decision is identical for every reorder, so the **reorder flip rate is exactly 0**. That covers reorder, reverse and shuffle, because all three are permutations.
3. **Relabel.** If additionally ids enter neither `x` nor `f` (C2 already excludes them from `f`; C1 must exclude them from `x`, so the canonical order must not sort by id), then for every bijection `σ`, `P_{O^σ}(σ(id)) = P_O(id)`, and the relabel flip rate is 0.
4. Reorder and relabel compose: any (π, σ) pair keeps the per-content distribution.

**Proof.**

- By (C1), `x(O^π) = x(O)`.
- By (C2), the score at new position `k` is `z'_k = f(x, c_{π(k)}) = z_{π(k)}`, so `z' = z∘π`.
- By (C3), the denominator `Σ_k exp(z'_k) = Σ_j exp(z_j)`, because finite sums in ℝ are invariant to order. Hence `p'_k = exp(z_{π(k)})/Σ_j exp(z_j) = p_{π(k)}`.
- The option at new position `k` has id `id_{π(k)}`, so `P_{O^π}(id_{π(k)}) = p_{π(k)} = P_O(id_{π(k)})`. This holds for every id.
- (C4): the argmax set of `z'` is the image of the argmax set of `z` under `π^{-1}`, which contains the same ids. A content-based tie-break picks the same content, and therefore the same id.
- Relabel: `c_i` is unchanged and ids are read nowhere, so `z` is literally the same vector. The ids attached to each position are `σ(id_i)`, and the result follows. ∎

**Relabel with id-agnostic codes.** If the model needs a short handle per option (for example, a letter it emits), assign handles by a rule `code_i = h(c_i, {c_j})`, such as the rank of `c_i` in the content-sorted multiset. Then a relabel leaves every code unchanged, because the codes never see the ids. The wire id is a pure lookup applied after scoring. Two failure points:

- **Duplicate contents.** Two options with equal `c` get equal scores, which is fine. But a rank-based code needs a tie-break, and a tie-break on id or position re-introduces the dependence. Equal-content options must get the same code-free score, and the decision must be reported as a set or tie-broken by something other than id or position.
- **Hash collisions** in a content-hash sort are the same problem. Use the full content as the secondary key.

## 3. What breaks it (each is a named violation, testable with the harness)

| Violation | Breaks | Why | Example in our stack |
|---|---|---|---|
| Shared prefix reads the option list in caller order (attention or recurrence) | reorder (C1) | `x` depends on order, so every `z_i` can change | Any schema prefix that enumerates options. The trap noted in `analysis/CODEX_CRITICAL_COMPARISON.md`. |
| Prefix lists options sorted by **id** | relabel (C1) | Relabel changes the sort order, hence `x` | The four evaluated models (GUARANTEES_FULLDEV §3): reorder-invariant by sorting, not relabel-invariant |
| Id tokens in the option continuation | relabel (C2) | `f` reads the id | Letter-logit readouts, `opt_3` in the continuation |
| Per-slot parameters or slot position embeddings | reorder (C2) | `f` depends on `k` | **G16 `u_k`, `beta_k`** (`controlsafe/g16.py` `judge`) |
| Position-dependent tie-breaking | flip only (C4) | `torch.argmax` / `max` return the first maximal index | `G16Head.forward` uses `int(torch.argmax(logits))`. Exact ties are rare with continuous logits but certain with vetoes, duplicates, or saturated E. |
| Veto or allowed-mask indexed by slot | reorder (C3) | the mask must travel with the option | A host mask is fine if it is keyed by id and mapped with the option. |
| Truncation or capacity by slot (`max_options`, padding, per-batch length) | reorder | the options after the cap depend on order | `K > max_options` raises in G16, which is safe. Silent truncation would not be. |
| Float non-associativity in batched reductions | exactness only | `Σ exp(z_j)` in a different order differs in the last bits, and batched GEMMs can tile differently per row | Section 4 gives the bound. |
| Nondeterministic kernels (atomics, split-K) | exactness only | Run-to-run noise even at a fixed order | Same bound. Report `max|Δlogit|`. |

## 4. Floating-point tolerance

Real arithmetic gives exact equality. Machine arithmetic gives two error sources. Let `u` be the unit roundoff (fp32: `2^-24 ≈ 6.0e-8`; bf16: `2^-8 = 3.9e-3`) and `γ_n = n·u/(1 − n·u)`.

1. **Logit perturbation.** Suppose each option's logit is computed with a backend whose result depends on its batch slot or on the run (tiling, atomics). Write `δ = max_i |z'_i − z_{π(i)}|`. This is measured, not derived. Log it as `max|Δlogit|`.
   - The softmax Jacobian `J = diag(p) − p pᵀ` has max-row-sum norm `max_i 2 p_i (1 − p_i) ≤ 1/2`. By the mean value theorem, `‖Δp‖_∞ ≤ δ/2`.
   - The argmax set is unchanged whenever the top-two logit margin exceeds `2δ`.
2. **Normaliser summation order.** With identical logits, summing `exp(z_j)` in a different order changes `Z` by a relative error of at most `γ_{K−1}`. With a standard `exp`, each `p_i` then changes by a relative error of at most about `γ_{K+2}`, so `|Δp_i| ≤ p_i·γ_{K+2} ≤ γ_{K+2}`. In fp32 with K ≤ 256, that is ≤ 1.6e-5. Summing in a canonical order removes this term.

**Tolerance to assert:** `max|Δp| ≤ tol = δ/2 + γ_{K+2}`. Assert argmax equality only when the margin exceeds `2δ`. Otherwise report the decision as a tie, not a flip.

**Bitwise equality is achievable** when all three hold:
- each option is scored by the same sequence of operations, i.e. the same batch shape and kernel, or one option per call;
- kernels are deterministic;
- the softmax sums in a canonical order, for example over options sorted by content.

The sym-invariant head should aim for bitwise equality and report `tol` as the fallback claim.

## 5. What this does and does not claim

- It does: a certified reorder flip rate of 0 (and relabel flip rate 0 under the id-agnostic condition), for every input, given that the head meets (C1)–(C4). That is a STRUCTURAL guarantee in SAFETY_SPEC terms.
- It does not cover paraphrase of option text, rewording of the question, or changing the option set. MD1's 1.3% perturbation flip rate includes paraphrase, which no construction certifies.
- It says nothing about accuracy. The symmetric layout loses cross-option comparison in the prefix. That cost must be measured with a paired ablation (`AGENTS.md`: every control mechanism ships with an on/off ablation).
- The measured side is `jevkit.guarantees.invariance_check`, which enumerates K! orders for K ≤ 4 plus seeded relabels. On an invariant head it must report `max_abs_dp == 0`, or ≤ `tol` with a logged `δ`.

## 6. How to plug in the real head

`tests/guarantees/test_reorder_invariance.py` exposes:

- `check_reorder(score_fn, state, options, n_perms, seed, tol)`
- `check_relabel(score_fn, state, options, n_relabels, seed, tol)`

Each returns a dict with `max_abs_dp`, `argmax_equal`, `flips`, `n_checked`. `score_fn(state, options)` takes `options` as a list of `(id, content)` and returns probabilities aligned to that list (a sequence), or a dict keyed by id.

To test the real head, wrap it as such a function and call `assert_invariant(...)`. Set `SSDM_INV_HEAD=module:function` to have the optional test `test_plugged_head` import and check it on the synthetic options.

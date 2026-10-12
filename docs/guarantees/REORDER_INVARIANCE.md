# Reorder and relabel invariance by construction (typed)

CONTROL_RESEARCH mechanism #1. Status: **typed theorem + property-test harness + measured on a trained model**. The typed statement (section 2T) was agreed with the lead lane on 2026-10-12. The real head (lead, `stack/qwen35/sym-invariant`) plugs into `tests/guarantees/test_reorder_invariance.py`.

## Takeaway

- **Three question types, two symmetries, one non-symmetry.** Types: NOMINAL (choice options with no inherent order: intent, routing, tool_selection, ...), ORDINAL/SCORE (rubric levels `0..K-1` whose order is part of the question's meaning), and NOUL (yes/no). Symmetries: (i) permuting the DISPLAY ORDER of the options, and (ii) renaming the OPAQUE HANDLES (the option ids/keys). Non-symmetry: (iii) renaming a LEVEL VALUE (level 2 becomes level 0). That changes the question.
- **Theorem 2T.** Suppose a head scores each option from (state, option content) only, with a permutation-equivariant normaliser, under (C1)–(C5). Then it is exactly invariant under (i) and (ii) for every type. For ordinal/score this needs one rendering rule: the level value is carried in the option CONTENT (`level 2: <rubric text>`), never only in the handle.
- **What jevkit's G1 checks for score items: nothing.** Frozen `jevkit/perturb.py` returns `None` (n/a) for score items under `option_reverse`, `option_shuffle` and `key_relabel`. `key_relabel` is also n/a for noul. On score items, invariance under (i) and (ii) rests on Theorem 2T and the CPU typed fixture, not on G1 (section 2T.3).
- **Dropping the level is not a fix.** A head that never reads handles, and also never sees the level value, is invariant but has thrown away information. JEV3-SYM seed 0 lost every discordant public pair on score (0/6) and on the ordinal family (0/6). That is an accuracy cost, not an invariance failure, and it is why rendering is typed (section 2T.4).
- **Measured.** On the trained GPU model JEV3-SYM seed 0, G1 shows 0 flips on 565 pairs, with max|Δp| 0, bitwise. On the same files, the stock control flips 28 / 21 / 12 (section 2T.5).
- In floating point, "exactly" becomes "within a stated tolerance". Bitwise equality holds when the per-option arithmetic is the same and the softmax sums in a fixed (canonical) order. Otherwise `max|Δp| ≤ δ/2 + γ_{2K}`, and the argmax is unchanged whenever the top-two logit margin exceeds `2δ` (section 4).
- **Finding on the current code.** `controlsafe/g16.py` adds per-option-position parameters `u_k` and `beta_k` to the judge (`a_k = A h_P[k] + u_k`, `b_k = w·h_P[k] + beta_k`). That is condition (C2) failing by design. The G16 head is reorder-invariant only if the canonical order it uses is a function of the option set. If that order sorts by id, the head is still not relabel-invariant. The sym-invariant G16 head must drop `u_k` and `beta_k` or replace them with a position-free term.

## 1. Setup

- A decision has a trusted state `x` (the shared prefix: state, question, instructions), a question type `t ∈ {nominal, ordinal/score, noul}`, and an option list `O = (o_1, …, o_K)` with `o_i = (h_i, c_i)`.
  - `h_i` is the **handle**: the opaque id or key the request uses and the answer is reported in.
  - `c_i` is the **content** the model reads: `c_i = R_t(option i)`, a fixed rendering of that option alone.
- **Typed rendering `R_t`.**
  - nominal: `c_i` = the option's description.
  - ordinal/score: `c_i` = the level's ordinal VALUE `v_i` plus its rubric text, for example `level 2: <rubric>`.
  - noul: `c_i` = the polarity plus its description, for example `yes: <criterion>`.
  - The handle never carries meaning the content lacks. jevkit score items use handles `"0".."K-1"` that coincide with the level value as strings. Typed rendering repeats the value inside `c_i`, so the head never has to read it from `h_i`.
- Handles are distinct. Contents need not be.
- A head maps `(x, O)` to a probability vector `p(x, O) ∈ Δ^{K}`, indexed by position. The **per-handle distribution** is `P_O(h_i) = p_i(x, O)`.
- **(i) Display-order permutation** `π` of `{1..K}`: `O^π = (o_{π(1)}, …, o_{π(K)})`. Reverse and shuffle are special cases.
- **(ii) Handle relabel**: a bijection `σ` from handles to handles (fresh ones, or the existing ones permuted among themselves), with content unchanged: `O^σ = ((σ(h_1), c_1), …)`.
- **(iii) Level-value renaming** (ordinal/score only): a map `τ` on level values with `c_i ↦ R(τ(v_i), rubric_i)`. It changes `c_i`, so it changes the question. It is **not** a symmetry. A head that were invariant under `τ` would be ignoring the level value, which is a defect.
- **Invariance (equivariance at the position level).**
  - (i): `P_{O^π}(h) = P_O(h)` for every handle and every `π`.
  - (ii): `P_{O^σ}(σ(h)) = P_O(h)` for every handle and every `σ`.
- **Flip:** the predicted option changes after mapping back. Under ties, the comparison uses the argmax set of contents (section 2, C4).

## 2. Conditions

- **(C1) Shared state is order-free and handle-free.** `x` is computed without reading the option list, or it reads the list only in a canonical order that is a function of the multiset `{c_i}`, for example sorted by content or by a content hash with the full content as secondary key. It must never be sorted by handle.
- **(C2) Per-option score is position-free and handle-free.** There is a function `f` with `z_i = f(x, c_i) ∈ ℝ ∪ {−∞}`. That excludes option-slot embeddings, per-slot parameters, and handle tokens in the continuation.
- **(C3) Permutation-equivariant normaliser.** `p_i = exp(z_i) / Σ_j exp(z_j)`, or any normaliser `g` with `g(z∘π) = g(z)∘π`. Temperature qualifies. So does top-k masking by value, provided ties at the k-th value are all kept or all dropped, since keeping "the first" tied one is a position rule. Vetoes that are a function of `(x, c_i)` also qualify.
- **(C4) Content-based tie handling; duplicates are a tie set.** The predicted option is chosen without looking at position or handle: either the full argmax set, or a deterministic tie-break on content. Options with equal content get equal scores and form one tie set. If they share the maximum, the decision is the set of their handles. No position-free and handle-free rule can pick one of them.
- **(C5) Whitespace normalisation.** The content the head reads is `N(c_i)` for a fixed normaliser `N`, for example collapsing whitespace runs and stripping. It is computed from the option's own text, never from its rendered neighbourhood. A prompt that puts a separator after every option except the last leaks position through the raw text. `N` removes it, or the renderer must emit identical bytes for every slot.

## 2T. Theorem (typed; real arithmetic)

**Theorem 2T.** Fix a question type `t` and the typed rendering `R_t`. Let a head satisfy (C1)–(C5). Then for every `t ∈ {nominal, ordinal/score, noul}`:

1. **(i) Display order.** For every permutation `π`, `p(x, O^π) = p(x, O)∘π`. The per-handle distribution is identical, so `max|Δp| = 0`, and with (C4) the decision is identical. The reverse and shuffle flip rate is exactly 0.
2. **(ii) Handle relabel.** For every bijection `σ` on handles, `P_{O^σ}(σ(h)) = P_O(h)`. The decision maps to the same content, so the relabel flip rate is exactly 0. For ordinal/score this holds because the level value reaches `f` through `c_i`, which `σ` does not touch.
3. **Composition.** Any `(π, σ)` pair keeps the per-content distribution.
4. **(iii) is excluded.** For ordinal/score, `τ` changes `c_i`. Nothing is claimed, and a head that follows the level value must be allowed to change its answer.

In floating point, (1)–(3) hold bitwise under the conditions of section 4, and otherwise within `tol = δ/2 + γ_{2K}`.

**Proof.**

- (C1): `x(O^π) = x(O^σ) = x(O)`. The canonical order is a function of the content multiset, which `π` and `σ` both preserve.
- (C2) and (C5): the score at new position `k` under `π` is `z'_k = f(x, N(c_{π(k)})) = z_{π(k)}`, so `z' = z∘π`. The rendered neighbourhood does not enter, because `N` maps each slot's raw text to the same string. Under `σ`, `c_i` is unchanged and handles are read nowhere, so `z` is literally the same vector.
- (C3): `g(z∘π) = g(z)∘π`. For softmax, `Σ_k exp(z'_k) = Σ_j exp(z_j)` because finite sums in ℝ do not depend on order. Hence `p'_k = p_{π(k)}`.
- The option at new position `k` has handle `h_{π(k)}`, so `P_{O^π}(h_{π(k)}) = p_{π(k)} = P_O(h_{π(k)})` for every handle. Under `σ`, position `i` carries handle `σ(h_i)` with the unchanged `p_i`, so `P_{O^σ}(σ(h_i)) = P_O(h_i)`.
- (C4): the argmax set of `z'` is the image of the argmax set of `z` under the same bijection, so it holds the same contents. A content tie-break picks the same content. When contents are duplicated, the decision is a set and is the same set.
- Type enters only through `R_t`, which is applied per option before any of the above. So the argument is the same for every type, and (iii) falls outside it because it changes the inputs `c_i`. ∎

### 2T.1 Corollary (ordinal/score: relabel invariance iff the level value is content)

Take a head whose output depends on the level value at all. It is invariant under every handle relabel (ii) **if and only if** the level value reaches it only through content.

- **If:** Theorem 2T(2).
- **Only if:** under (C1) and (C2) the head's inputs per option are `x` and `(h_i, c_i)`. Suppose `z_i` depends on `v_i` through `h_i`, for example `int(h_i)` with jevkit's `"0".."K-1"` handles. Then the handle permutation `σ` that swaps the handles of two levels changes `z` and moves probability between contents. The typed fixture's `ordinal_handle_head` does exactly this: max|Δp| > 0.5, with flips.

### 2T.2 What changes per type

| Type | Content `c_i` | (i) display order | (ii) handle relabel | (iii) level renaming |
|---|---|---|---|---|
| nominal | description | symmetry | symmetry | n/a |
| ordinal/score | `level v: rubric` | symmetry (the rubric order is shown, not used) | symmetry iff `v` in content | **not a symmetry** |
| noul | `yes/no: criterion` | symmetry | symmetry iff polarity in content | n/a (swapping yes/no is a different question) |

### 2T.3 What jevkit's G1 measures, per type (read from frozen `jevkit/perturb.py`; not edited)

- `op_key_relabel` returns `None` unless `item.type == "choice"`. For choice items it maps the criteria keys to `opt_1..opt_K` in a seeded order, keeps each description, rewrites `labels` and `expected` through the same map, and scores flips after mapping back through `label_map`. That is exactly symmetry (ii) with fresh handles.
- For **score** items `key_relabel` is n/a. So are `option_reverse` and `option_shuffle`: `_reorder` returns `None` when `item.type == "score"`, by design ("reordering ordinal levels" is listed as n/a). `Item.validate` would also reject a relabel, because score labels must be `"0".."K-1"` and `expected` an int level index. For **noul**, reverse and shuffle swap the two criteria, and relabel is n/a.
- **Consequence.** The G1 relabel check tests nothing on score items or noul items. JEV3-SYM's 139 relabel pairs are all `key_relabel|choice`. The 213 reverse/shuffle pairs are choice plus noul (213 − 139 = 74, the public noul count). For score items, invariance under (i) and (ii) is established by Theorem 2T plus the typed CPU fixture, not by G1. A G1 PASS is not evidence about score items in either direction.
- Choice items whose keys look ordinal (the "ordinal" family is `choice` in jevkit) are relabelled to `opt_i`. For a head satisfying (C2), that is still only (ii), and 0 flips is guaranteed. If the key alone carried the order, the head never saw it in the base item either, so the loss shows up as base accuracy, not as flips.

### 2T.4 Failure mode: an id-agnostic head that drops meaningful level labels

- A head can be handle-free by never reading handles *and* never rendering the level value. Then it is invariant under (i) and (ii), but on ordinal/score it can no longer tell `level 0` from `level 3` except through rubric text.
- Measured on JEV3-SYM seed 0 (job-6zs79; evidence rescore `claude/ev-6zs79-rescore` @f93eccfe). This is the paired native-vs-stock top-1 on public items, counted as discordant pairs (native-only right / stock-only right):
  - score: 0/6, −33.3 pp (p 0.031);
  - ordinal family: 0/6, −50 pp;
  - routing: 0/7, −58.3 pp.
- G1 on the same model is a bitwise PASS. So this is an **accuracy cost, not an invariance failure**.
- That is the motivation for typed rendering. The level value goes back into the content (`R_score`), which keeps (ii) by Theorem 2T and restores the information. The fixture `ordinal_value_dropping_head` reproduces the pattern in miniature: it stays invariant, and its top-1 is ≤ 0.6, against 1.0 for `ordinal_content_head` on the same 40 cases.

### 2T.5 Scope and evidence

| Evidence | Head | Types covered | Result | Ref |
|---|---|---|---|---|
| Theorem 2T | any head meeting (C1)–(C5) | all three; (i) and (ii) | exact; bitwise under section 4 | this doc |
| CPU typed fixture | toy `ordinal_content_head` (level as content) | ordinal/score | (i) all K! orders + (ii) fresh ids + handles permuted: max|Δp| = 0, 0 flips (bitwise) | `tests/guarantees/test_reorder_invariance.py` (this branch) |
| CPU typed fixture, negatives | `ordinal_handle_head` (level via handle); `ordinal_unnormalised_head` (no C5) | ordinal/score | handle head passes (i) bitwise and fails (ii): max|Δp| > 0.5, flips > 0; unnormalised head fails (i) | same |
| CPU invariance harness on the sym head | G16-sym (9bc4845b duplicate fix) | choice, noul | round 3 verified | `claude/ev-sym-check` @2e465827 |
| Trained GPU model, G1 | JEV3-SYM seed 0 | choice (reverse, shuffle, relabel), noul (reverse, shuffle); **score: not tested** | 0 flips / 0 invalid on 565 pairs (213 reverse, 213 shuffle, 139 relabel); max|Δp| 0 per op, bitwise | job-6zs79; `claude/ev-6zs79-rescore` @f93eccfe |
| Non-vacuity control | stock_shim on the same files | same | **fails**: 28 / 21 / 12 flips (reverse / shuffle / relabel), max|Δp| 0.84 | same |

Not covered by any row: paraphrase, formatting noise and state rewording. JEV3-SYM flips 4.14% overall, all of it from those operators. Level-value renaming (iii) is excluded by design.

### 2T.6 Id-agnostic codes

 If the model needs a short handle per option (for example, a letter it emits), assign handles by a rule `code_i = h(c_i, {c_j})`, such as the rank of `c_i` in the content-sorted multiset. Then a relabel leaves every code unchanged, because the codes never see the ids. The wire id is a pure lookup applied after scoring. Two failure points:

- **Duplicate contents.** Two options with equal `c` get equal scores, which is fine. But a rank-based code needs a tie-break, and a tie-break on id or position re-introduces the dependence. Equal-content options must get the same code-free score, and the decision must be reported as a set or tie-broken by something other than id or position.
- **Hash collisions** in a content-hash sort are the same problem. Use the full content as the secondary key.

## 3. What breaks it (each is a named violation, testable with the harness)

| Violation | Breaks | Why | Example in our stack |
|---|---|---|---|
| Shared prefix reads the option list in caller order (attention or recurrence) | reorder (C1) | `x` depends on order, so every `z_i` can change | Any schema prefix that enumerates options. The trap noted in `analysis/CODEX_CRITICAL_COMPARISON.md`. |
| Prefix lists options sorted by **id** | relabel (C1) | Relabel changes the sort order, hence `x` | The four evaluated models (GUARANTEES_FULLDEV §3): reorder-invariant by sorting, not relabel-invariant |
| Level value read through the handle (`int(h)` on `"0".."K-1"`) | relabel, ordinal/score (C2) | the value moves with the handle under (ii) | Any score readout that emits or reads the level digit as the answer key without `level v:` in content |
| Raw rendered text, no whitespace normalisation | reorder (C5) | the last-displayed option's bytes differ (no trailing separator) | Prompt templates joining options with `
` |
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
2. **Normaliser summation order.** With identical logits, the per-option work (`z_i − m`, `exp`) is the same operation sequence in both orders, so `e_i = fl(exp(z_i − m))` is bitwise identical; only the sum differs. Recursive summation of K positive terms gives `Ẑ = Z(1 + θ)` and `Ẑ' = Z(1 + θ')` with `|θ|, |θ'| ≤ γ_{K−1}`, and the final division adds one rounding to each side. The two computed values therefore differ by the ratio `(1+θ')(1+η)/((1+θ)(1+η'))`, a product of at most `2(K−1) + 2 = 2K` factors `(1 + u_j)^{±1}`, so by Higham's Lemma 3.3 `|p̂_i/p̂'_i − 1| ≤ γ_{2K}` and `|Δp_i| ≤ p_i·γ_{2K} ≤ γ_{2K}`. In fp32 with K ≤ 256, that is ≤ 3.1e-5. (An earlier version of this section stated `γ_{K+2}`, which bounds each computed `p̂_i` against the exact `p_i`, not two computed values against each other; the difference of two such errors is up to twice as large.) Summing in a canonical order removes this term.

**Tolerance to assert:** `max|Δp| ≤ tol = δ/2 + γ_{2K}` (first order in `u`). Assert argmax equality only when the margin exceeds `2δ`. Otherwise report the decision as a tie, not a flip.

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
- `check_handle_permutation(score_fn, state, options, n_relabels, seed)`: symmetry (ii) with the existing handles permuted among themselves. That is the adversarial case for ordinal/score, where handles look like level values.
- Typed fixture: `render_level`, `_ordinal_cases`, and the four `ordinal_*` toy heads (section 2T.5).

Each returns a dict with `max_abs_dp`, `argmax_equal`, `flips`, `n_checked`. `score_fn(state, options)` takes `options` as a list of `(id, content)` and returns probabilities aligned to that list (a sequence), or a dict keyed by id.

To test the real head, wrap it as such a function and call `assert_invariant(...)`. Set `SSDM_INV_HEAD=module:function` to have the optional test `test_plugged_head` import and check it on the synthetic options.

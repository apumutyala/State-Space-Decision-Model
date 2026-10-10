# Injection-robust conformal sets: coverage for ANY untrusted text

CONTROL_RESEARCH mechanism #3. It combines G15/G16 with split conformal (`CONFORMAL.md`). Test: `tests/guarantees/test_injection_conformal.py`. Current implementation: `jevkit.guarantees.robust_set` / `injection_robust_report`.

## Takeaway

- **Theorem.** If the conformal score is computed from the trusted stream only, then the prediction set is bitwise identical for every untrusted text. Its coverage is ≥ 1 − α for every attacker, including one that knows the gold label, the calibration set and the weights. G16's per-option intervals `[lo_k, hi_k]` are trusted-only, so any score built from them qualifies.
- **Which bound to calibrate on: the same one you test with.** Use the optimistic bound `s_lo` (the smallest nonconformity over the box) at test time *and* in calibration. This "matched" set is exact split conformal. Of the three calibration choices for an `s_lo` test statistic (matched, clean scores, `s_hi`), it gives the smallest set for every decision (Proposition C). It needs no clean calibration logits, and it is even robust to injected text *inside the calibration data*.
- **The current construction is valid but conservative.** That is `C_rob = {y : s_lo(y) ≤ q̂(clean scores)}`, in CONTROL_RESEARCH and in jevkit. Calibrating on the lower conformity bound (`s_hi`) is valid but the most conservative of the three. Pairing the pessimistic test score with clean calibration is **invalid**.
- **Synthetic result** (K = 4, α = 0.10, 100 seeds × 150 calibration × 100 test decisions; the adversary knows the gold label and picks the worst point in the box):

| Set | Coverage under attack | Mean set size |
|---|---|---|
| Clean conformal, no attack (reference) | 0.897 | 2.08 |
| **Matched robust `s_lo` / `s_lo`** | **0.901** | **2.30** |
| Matched robust `s_hi` / `s_hi` | 0.902 | 2.35 |
| Current jevkit `s_lo` / clean | 0.980 | 3.21 |
| `s_lo` / `s_hi` (calibrate on lower conformity bounds) | 0.999 | 3.86 |
| **Naive**: attacked point score / clean | **0.559** | 2.27 |
| **Wrong bound**: `s_hi` / clean | **0.559** | 0.93 |

- Matched calibration costs **+0.22 labels** over the unattacked clean set. The current construction costs +1.13. Both hold coverage. The naive set loses a third of its coverage to the same adversary.
- **Cost.** A trusted-only set cannot use untrusted evidence. When the right answer genuinely depends on untrusted content, for example a retrieved document, the set must inflate. The inflation is governed by the G16 box widths `c‖a_k‖₁` relative to the logit gaps, and it vanishes as `c → 0`.

## 1. Setup

- A decision's input is `x = (P, U)`, where `P` is the trusted tokens (host, instructions, schema) and `U` is the untrusted ones (by trust label).
- **G15** (structural): the protected features `h_P` are a function of `P` only, bitwise.
- **G16** (structural): `logit_k = b_k + a_k·E_k`, with `a_k, b_k` functions of `P` only and `E_k = c·tanh(W_e h_D[k]) ∈ (−c, c)^r`.
  - So for every `U`, `logit_k ∈ [lo_k, hi_k]` with `lo_k = b_k − c‖a_k‖₁` and `hi_k = b_k + c‖a_k‖₁`. Both are functions of `P` only.
  - Vetoed options have `lo = hi = −∞`.
- **Score.** `s(z, y) = 1 − softmax(z)_y`. It decreases in `z_y` and increases in every `z_k` for `k ≠ y`. So over the box it has closed-form extremes:
  - `s_lo(P, y) = s(z^{y↑}, y)`, with `z_y = hi_y` and `z_k = lo_k` otherwise;
  - `s_hi(P, y) = s(z^{y↓}, y)`, with `z_y = lo_y` and `z_k = hi_k` otherwise;
  - and for every U, `s_lo(P, y) ≤ s(z(P, U), y) ≤ s_hi(P, y)`.
- **Data.** Clean deployment draws `(P_i, U_i^0, Y_i)`, i = 1..n+1, are exchangeable. `Y` is the gold decision for the clean task.
- **Attacker.** At test time it replaces `U_{n+1}^0` by any `U* = A(P_{n+1}, U^0_{n+1}, Y_{n+1}, calibration data, weights)`. It is fully adaptive, and every token it adds is labelled untrusted.
- **Conformal quantile.** `Q_α(v_1..v_n)` is the `⌈(n+1)(1−α)⌉`-th smallest value, or `+∞` (`CONFORMAL.md` Theorem 1).

## 2. Results

### Theorem A: trusted-only conformal sets are attack-proof

**Statement.**
- Let `T(P, y)` be any measurable function of the trusted input and a label. `s_lo`, `s_hi`, and any combination of `lo` and `hi` all qualify.
- Let `q̂_T = Q_α(T(P_1, Y_1), …, T(P_n, Y_n))` and `C_T(P) = {y : T(P, y) ≤ q̂_T}`.
- Then for **every** attacker `A`: `P(Y_{n+1} ∈ C_T(P_{n+1})) ≥ 1 − α`.
- If `T(P, Y)` has no ties, coverage is also `≤ 1 − α + 1/(n+1)`.
- The same holds if the calibration data's untrusted text was itself attacker-chosen.

**Proof.**
- `C_T(P_{n+1})` and `q̂_T` are functions of `(P_1, Y_1), …, (P_{n+1})` only. By G15 they are bitwise the same for every `U*`, and for every untrusted text in calibration.
- The pairs `(P_i, Y_i)` are exchangeable, as marginals of the exchangeable clean draws. The attacker does not touch `P` or `Y`.
- Theorem 1 of `CONFORMAL.md` applied to the score `T` gives both bounds. ∎

**Corollary A′ (G16 instance).** `T = s_lo` gives the *matched* robust set `C_lo = {y : s_lo(P, y) ≤ Q_α(s_lo(P_i, Y_i))}`. This needs, per calibration decision, only the intervals and the gold label.

### Theorem B: clean-calibrated robust set (the current construction)

**Statement.**
- Let `S_i^0 = s(z(P_i, U_i^0), Y_i)` be the clean scores, and `C_rob(P) = {y : s_lo(P, y) ≤ Q_α(S_1^0..S_n^0)}`.
- If (i) the calibration untrusted text is clean and (ii) every clean logit lies in its box, then `P(Y_{n+1} ∈ C_rob) ≥ 1 − α` for every attacker.

**Proof.**
- By (ii), `s_lo(P_{n+1}, Y_{n+1}) ≤ S^0_{n+1}`.
- So `{S^0_{n+1} ≤ q̂} ⊆ {Y_{n+1} ∈ C_rob}`.
- The left event has probability ≥ 1 − α by exchangeability of the clean scores, which needs (i). ∎

Theorem B needs more than Theorem A: clean calibration text plus box containment. Its sets are also never smaller (Proposition C).

### Proposition C: which bound to calibrate on

**Statement.** Fix the same calibration sample and use `s_lo` as the test statistic.
- Then `q̂_lo ≤ q̂_clean ≤ q̂_hi`.
- Hence, for every test decision, `C_{lo|lo} ⊆ C_{lo|clean} ⊆ C_{lo|hi}`.
- All three cover ≥ 1 − α: the first by Theorem A, and the other two because they contain it.
- Only the matched set is exact, i.e. not over-covering beyond `1/(n+1)` for continuous scores.

**Proof.** `s_lo(P_i, Y_i) ≤ S_i^0 ≤ s_hi(P_i, Y_i)` pointwise. Order statistics preserve pointwise order, so the quantiles are ordered. The sets are monotone in the threshold. ∎

**In conformity form,** the task's framing. Write `π_y^{hi} = 1 − s_lo` for the largest probability of y over the box, and `π^{lo}` for the smallest. Then the matched set is `C* = {y : π_y^{hi} ≥ τ}`, with `τ` the conformal lower quantile of `π_{Y_i}^{hi}` over calibration. **Calibrate on the upper bounds of the gold labels, the same bound you threshold.**
- Calibrating `τ` on the lower bounds `π_{Y_i}^{lo}` is valid by dominance, but it is the `C_{lo|hi}` set: the most conservative (3.86 vs 2.30 labels in the synthetic run).
- Calibrating on clean point probabilities is the current `C_rob`.

### Proposition D: the invalid pairings (and the naive set)

**Wrong bound.** `C_{hi|clean} = {y : s_hi(P, y) ≤ q̂_clean}` has no guarantee.
- `s_hi ≥ S^0`, so `P(s_hi(P_{n+1}, Y) ≤ q̂_clean) ≤ P(S^0_{n+1} ≤ q̂_clean)`, and the inequality can be strict by any amount.
- Synthetic: 0.559.
- In general, mixing a test statistic with a different calibration statistic is valid only when the test statistic is pointwise ≤ the calibration statistic.

**Naive.** `C_naive = {y : s(z(P, U*), y) ≤ q̂_clean}`.
- The attacker sets `z = z^{y↓}`, so the observed score is `s_hi(P, Y)`. `C_naive` then excludes `Y` exactly when `C_{hi|clean}` does.
- That is why both rows read 0.559. The test's corner search confirms that no point in the box does worse.

### Corollary E: LTT acted-on error under attack

**Act rule.**
- Act iff `|C_lo(P)| = 1` and `κ(P) ≥ λ`, with `κ` a trusted-only confidence such as `π^{lo}` of the singleton.
- The action is the singleton element itself, *not* the argmax of the attacked logits.

**Guarantees.**
- Everything in the rule is a function of `P`. So the LTT certificate (`CONFORMAL.md` Theorem 2), computed on calibration intervals plus gold labels, gives `P(R(λ̂) > ε) ≤ δ` for every attacker.
- The attack success rate on acted decisions is 0 bitwise. The attacker cannot change whether the model acts, or which action it takes.

**Relation to G16 independence.** If the model instead acts on the attacked argmax, that action equals the singleton only on decisions that are G16-independent (`lo_y > max_{k≠y} hi_k`).

## 3. Assumptions, stated once

| # | Assumption | Needed by | If violated |
|---|---|---|---|
| A1 | **Trust labels are correct**: every attacker-controlled token is labelled untrusted | A, B, C, E | Attacker text sits in `P`. `lo`/`hi` move with it, and every guarantee here is void. `provenance` fails closed, but a host that labels attacker text as an instruction gets no protection. |
| A2 | **G15 holds bitwise** for these weights (`P` features invariant to `U`) | A, B, E | Tested structurally in `tests/controlsafe`. A kernel that mixes streams voids it. |
| A3 | **`(P, Y)` exchangeable** between calibration and deployment | A, B, E | Distribution shift on the trusted side breaks coverage, as in ordinary conformal. The attacker cannot cause this through `U`. |
| A4 | **The gold label is defined by the clean task** | A, B | If untrusted content legitimately changes the right answer, "attack" and "evidence" are the same thing. The guarantee is then about the clean-task label, and a trusted-only set can only cover it by inflating. |
| A5 | **Box containment**: clean logits lie in `[lo, hi]` | B only | Exact in real arithmetic, because `tanh ∈ (−1, 1)`. In floating point, `tanh` can round to ±1 and `b + a·E` can round past `b ± c‖a‖₁` by a few ulps. Compute `lo`/`hi` with an outward-rounded margin (jevkit `_inside` uses 1e-9). Theorem A does not need A5. |
| A6 | **Clean calibration text** | B only | Injected text in calibration can lower `q̂_clean`. Theorem A is immune. |
| A7 | **Fixed weights** across calibration and deployment | all | Recalibrate per checkpoint. |
| A8 | **Mondrian groups are functions of `P`** | Mondrian variants | A group that reads `U` would let the attacker pick the threshold. |

## 4. Cost: set inflation

- Robust sets cost size, not coverage.
- The size floor is the best *trusted-only* predictor's conformal set. No construction that is invariant to `U` can use information that is only in `U`.
- Inflation over the unattacked clean set grows with `c‖a_k‖₁` relative to the logit gaps `b_i − b_j`.
  - At `c = 0` the box is a point and `C_lo` equals the ordinary set.
  - The `c` ablation (`train.g16_card sweep`) and the νPI margin constraint (CONTROL_RESEARCH #4) are the levers.
- **Report all three numbers:** coverage under attack, mean `|C_lo|` vs mean `|C_clean|`, and the singleton (act) rate. Add the independence rate as a fourth.

## 5. What is novel, honestly

- The statistics are elementary. Once a score depends only on `P`, Theorem A is ordinary split conformal.
- The novelty is that G15/G16 make a *useful* trusted-only score exist, with a closed-form box. Competitors' logits all read untrusted text, so they have no such score.
- Adversarially robust CP (Gendler et al., ICLR 2022; VRCP, NeurIPS 2024) bounds the score over a perturbation set by smoothing or verification. Here the bound is exact, it covers arbitrary text rather than an ℓp ball, and the set is not merely robust but *invariant*.
- What I did not find in the literature: the observation that matched calibration on the interval bound (Theorem A) dominates the clean-calibrated dominance construction (Theorem B, Proposition C). VRCP-style methods calibrate on clean scores and inflate at test time, which is Theorem B. I have not done an exhaustive search.

## 6. Recommendation for jevkit (no change made here)

Add a matched mode to `injection_robust_report`, for example `calibration="trusted_lo"`:
- calibration rows carry `{intervals, label}`;
- `q̂ = conformal_qhat([robust_min_score(iv_i, y_i)])`;
- sets are `robust_set(iv, q̂)`.

The CLI `guarantees injection` then takes `--cal-intervals` instead of `--cal-pred`. Keep the current clean-calibrated mode as a reported comparison. The functions it needs (`robust_min_score`, `robust_set`, `conformal_qhat`) already exist and are exercised by the test. I left the module alone because the lead's G16 head defines the interval export format.

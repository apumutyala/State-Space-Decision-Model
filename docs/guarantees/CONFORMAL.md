# Split conformal and Learn-then-Test: what is guaranteed, and does jevkit implement it?

CONTROL_RESEARCH mechanism #2. Code: `jevkit/guarantees.py` (not on the frozen list). Tests: `tests/guarantees/test_conformal_theorem.py` and `jevkit/tests/test_guarantees.py`.

## Takeaway

- **Coverage.** Split conformal gives `P(Y ∈ C(X)) ≥ 1 − α` with no model assumptions, as long as the calibration and test decisions are exchangeable. With continuous scores it is also at most `1 − α + 1/(n+1)`. jevkit's quantile is the theorem's quantile: the `⌈(n+1)(1−α)⌉`-th smallest score, and +∞ when that rank exceeds n. I checked that float rounding never shifts the rank for α ∈ {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3} and every n ≤ 20,000.
- **Acted-on error.** LTT with Bonferroni over a fixed λ grid certifies that, with probability at least 1 − δ over the LTT half, the selected rule's acted-on error *risk* (a population quantity) is at most ε. Plain coverage does not bound this; LTT does.
- **Discrepancies found: four.** None changes the published FULLDEV numbers' direction, but two weaken their formal status.
  - **D1, real.** jevkit splits FIT/LTT by *decision id*. The FULLDEV driver gives each decision the id `run_id#key`, so the decisions of one row land on both sides, and the LTT binomial test treats correlated decisions as independent. The docstring's "LTT-half decisions are i.i.d. and independent of q_hat" therefore does not hold for multi-decision rows. Fixed in part on this branch: `split_calibration(..., cluster_of=...)` (default unchanged, tested byte-identical). The within-LTT correlation needs a cluster-level test (proposed below).
  - **D2, theoretical, `fixed_sequence` only.** Skipping a λ that acts on nothing is not covered by the fixed-sequence proof. The default Bonferroni mode, which FULLDEV used, is unaffected.
  - **D3, wording.** "error rate on acted decisions ≤ ε" means the population acted-on risk, not the realised rate on any finite test batch.
  - **D4, scope.** Mondrian is per decision type only. The marginal guarantee holds, and area-conditional coverage fails (arts 0.69–0.79 at a 0.90 target). B-PREREG-GUAR-AREA, a parallel worker, is measuring the per-area Mondrian fix now. This doc does not duplicate it.

## 1. Split conformal coverage

**Setup.**
- A score `s(x, y)` is fixed before calibration. In jevkit it is `s = 1 − p_y`, from a model trained on other data.
- `(X_1, Y_1), …, (X_n, Y_n), (X_{n+1}, Y_{n+1})` are exchangeable. `S_i = s(X_i, Y_i)`.
- `k = ⌈(n+1)(1−α)⌉`. `q̂` is the k-th smallest of `S_1..S_n`, and `q̂ = +∞` if `k > n`.
- `C(x) = {y : s(x, y) ≤ q̂}`.

**Theorem 1.**
- `P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α`, with the probability over calibration and test jointly.
- If the `S_i` are almost surely distinct, the coverage is also `≤ 1 − α + 1/(n+1)`.

**Proof.**
- `Y_{n+1} ∈ C(X_{n+1})` iff `S_{n+1} ≤ q̂`.
- That holds iff `S_{n+1}` is among the k smallest of `S_1..S_{n+1}`, with ties resolved in its favour.
- By exchangeability, the rank of `S_{n+1}` among the n+1 scores is uniform on {1..n+1}, with ties broken uniformly at random. Favourable ties only help.
- So the probability is at least `k/(n+1) ≥ 1 − α`. Without ties it equals `k/(n+1) < 1 − α + 1/(n+1)`.
- If `k > n`, then `q̂ = ∞` and coverage is 1. ∎

**Mondrian (group-conditional).**
- Let `g(x)` be a group that is a function of x only: type, area, benchmark.
- Calibrate each group separately. Then `P(Y ∈ C(X) | g(X) = g) ≥ 1 − α` for every group, assuming exchangeability within each group.
- jevkit uses the decision type. A group never seen in calibration gets `q̂ = ∞`, i.e. the full set, which is correct.

**Training-conditional coverage.** The guarantee averages over calibration draws. For a fixed calibration set with continuous scores, the coverage is distributed `Beta(k, n+1−k)` (Vovk 2012). At FULLDEV's FIT-half sizes (about 8.6k), its standard deviation at α = 0.1 is about 0.003.

**Assumptions, all of them.**
- Exchangeability between calibration and deployment decisions.
- The score must not be fitted on the calibration data.
- Ties must be handled by `≤`, as jevkit does.

**What breaks it.**
- **Distribution shift.** A new benchmark mix, a new prompt template, or a different model checkpoint between calibration and use. Recalibrate per checkpoint.
- **Group-conditional failure** (D4). The marginal guarantee is met while hard areas undercover.
- **Clustered decisions** (D1). Several decisions from one row are not exchangeable individually. Per-decision coverage over clusters is not exactly covered by Theorem 1; see Dunn, Wasserman and Ramdas (2022) on hierarchical conformal. When the CAL/TEST split is by row or case (FULLDEV does this, plus a case-keyed sensitivity split), coverage holds for the cluster-level exchangeable statement and is approximately correct per decision.
- Selecting α, the score or the grouping after looking at test coverage.

**Implementation check.** All four points match the theorem.

| Theorem element | jevkit | Verdict |
|---|---|---|
| `q̂` = `⌈(n+1)(1−α)⌉`-th smallest, `∞` if the rank exceeds n | `conformal_qhat` | Matches. `tests/guarantees/test_injection_conformal.py::test_qhat_rank_matches_theorem` checks the rank against brute force; float `ceil` checked for n ≤ 20,000. |
| `C = {y : 1 − p_y ≤ q̂}` with `≤` | `prediction_set` | Matches |
| Mondrian on a function of x | `_group` = `r.type` | Matches |
| Coverage in `[1−α, 1−α+1/(n+1)]` | simulated over 120 seeds, n = 99 | `test_split_conformal_coverage_between_the_theorem_bounds` passes |

## 2. Learn-then-Test acted-on error

**Setup.**
- After the FIT half fixes `q̂`, the act rule for λ is: act iff `|C(x)| = 1` and `p_max ≥ λ`, and the action is the singleton.
- `R(λ) = P(error | act_λ)` is the acted-on risk, under the deployment distribution and conditional on the FIT half.
- The grid `Λ` is fixed before seeing data. jevkit uses `1.00, 0.99, …, 0.00`.
- Null hypotheses `H_λ: R(λ) > ε`.

**p-values.**
- On the LTT half, `n_λ` decisions act and `e_λ` of them are wrong. `p_λ = P(Bin(n_λ, ε) ≤ e_λ)`.
- Given `n_λ`, the acted decisions are i.i.d. from the acted distribution, so `e_λ ~ Bin(n_λ, R(λ))`.
- Under `H_λ`, `R(λ) > ε`, so `Bin(n_λ, R)` stochastically dominates `Bin(n_λ, ε)`. Hence `P(p_λ ≤ t | n_λ) ≤ t`: the p-value is valid conditionally, and therefore marginally.

**Theorem 2 (LTT, Angelopoulos et al. 2021, Thm 1 with Bonferroni).**
- Reject every λ with `p_λ ≤ δ/|Λ|`. Let `λ̂` be any rejected λ; jevkit takes the smallest, i.e. the most actions.
- Then `P(R(λ̂) > ε) ≤ δ`.
- Proof: by the union bound, `P(∃ true null rejected) ≤ Σ_{λ ∈ H_0} δ/|Λ| ≤ δ`. On the complement, every rejected λ has `R ≤ ε`. ∎

**Fixed-sequence variant.**
- Test `λ` in a pre-set order at level δ, and stop at the first non-rejection.
- FWER ≤ δ, because any false rejection requires rejecting the first true null in the order.

**Assumptions.**
- The LTT half is independent of the FIT half and i.i.d. with deployment, given the FIT half.
- The λ grid is data-independent.
- The actions are the ones evaluated: the singleton, not some other argmax.

**What the certificate says, precisely (D3).** It bounds the *risk* `R(λ̂)`. The realised acted error on a finite TEST batch fluctuates around `R(λ̂)`. FULLDEV's held-out 3.7–4.3% against ε = 5% is consistent with this. A batch above ε would not by itself falsify the certificate.

**Implementation check.**

| Element | jevkit | Verdict |
|---|---|---|
| Disjoint FIT / LTT halves | `split_calibration`; `ltt_select` refuses overlapping ids | Matches **for one-decision-per-input data**. **D1**: for multi-decision rows the halves share rows, so q̂ is not independent of the LTT half and the LTT decisions are not independent of each other. |
| Conditional binomial p-value | `binom_cdf(err, n, ε)` | Matches |
| Bonferroni at `δ/|Λ|` | `level = delta / len(lambdas)` | Matches. The reported Clopper–Pearson UCB is at the same level, which is conservative. |
| Fixed sequence | stop at first non-rejection, but **skip** λ with `n_act = 0` | **D2**, below |
| λ with nothing acted | never selected | Correct |
| Coverage of the bound | simulated over 40 seeds: `R(λ̂) ≤ ε` in ≥ 1 − δ of seeds | `test_ltt_certificate_holds_across_seeds` passes |

### D1. Per-decision split with multi-decision rows

- **Where.** `split_calibration` hashes `r.id`. The FULLDEV driver (`b-landscape/guarantees_fulldev/guarantees_fulldev.py`, line 107) sets `did = f"{rid}#{key}"`.
- **Effect.** FULLDEV has 34,719 scorable decisions in 8,473 source cases, about 4 per case, so most rows have decisions on both sides.
  - q̂ is fitted on decisions that are correlated with LTT decisions from the same row. The LTT test is then no longer run on data independent of the rule.
  - Independently, the binomial test counts correlated decisions as independent trials. With positive within-row correlation, the effective n is smaller, so the p-values are anti-conservative.
- **Severity.** Probably small for the FIT/LTT leakage, since q̂ is one scalar per type fitted on about 8.6k decisions. Potentially material for the binomial test's nominal δ. The certificate's δ = 0.10 should be read as approximate until this is rerun.
- **Fix on this branch (part 1).** `split_calibration(cal, seed, fit_frac, cluster_of=None)` and `conformal_report(..., cluster_of=None)`.
  - Passing a row or case key keeps each cluster on one side and records cluster counts.
  - The default is unchanged. A test asserts that the per-id split is byte-identical to the old rule, so the FULLDEV receipt still reproduces.
- **Fix, part 2 (proposed, not implemented).** A cluster-level LTT test. Either:
  - sample one decision per cluster (seeded, answer-independent) before the binomial test, which certifies the risk for a random decision of a random cluster; or
  - use a Hoeffding–Bentkus bound on per-cluster acted-error and acted-count sums.

  Both lose some power. FULLDEV's LTT half would have roughly 2k source-case clusters; that still supports ε = 0.05 at a lower act fraction.

### D2. Fixed-sequence with skipped λ

- **Where.** `ltt_select` does `if n == 0: continue` before the fixed-sequence `break`.
- **Why it matters.** The fixed-sequence proof needs every hypothesis in the pre-set order to be tested, or to stop the procedure.
  - Skipping a λ that acted on nothing makes the effective starting point data-dependent.
  - Because acted sets are nested, a skipped λ means every higher λ acted on nothing. Conditioning on that event restricts the acted decisions at the next λ to those with `p_max` below the previous grid point. That can shift their error rate away from `R(λ)`.
- **The fix is not simply "treat n = 0 as non-rejection".** That stops the procedure at λ = 1.00 almost always, so it would certify nothing.
- **Valid options:**
  - start the sequence at a pre-registered `λ_0` and treat `n = 0` as a stop;
  - or use Bonferroni.
- Bonferroni is the default, and FULLDEV used it, so no published number is affected. I did not change the fixed-sequence code. The behaviour change needs the lead's call, and the fix is a one-line policy decision about `λ_0`.

### D4. Area-conditional failure

The marginal guarantee is met on FULLDEV, 0.899–0.903 at the 0.90 target. Arts and knowledge undercover because language, 57% of decisions, pulls the pooled type-level `q̂` down. Theorem 1 with Mondrian per area is the remedy. Pre-registration and measurement are **B-PREREG-GUAR-AREA**, owned by a parallel worker. This doc only states that the theorem supports it: per-area `q̂` gives per-area coverage ≥ 1 − α under within-area exchangeability.

## 3. What to claim externally

- "Split-conformal sets with marginal coverage ≥ 1 − α (exchangeable data; recalibrated per checkpoint)."
- "An acted-on error risk ≤ ε with confidence 1 − δ (LTT, Bonferroni over a fixed λ grid; cluster-level split and test)." Make that claim only once D1 part 2 is in.
- Never claim per-area or per-benchmark coverage without the Mondrian-per-area run, and never claim either statement under distribution shift.

## Sources

- Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random World* (2005). Vovk, "Conditional validity of inductive conformal predictors" (2012).
- Angelopoulos, Bates, Candès, Jordan and Lei, "Learn then Test: Calibrating predictive algorithms to achieve risk control", arXiv:2110.01052.
- Dunn, Wasserman and Ramdas, "Distribution-free prediction sets for two-layer hierarchical models", JASA 2022.

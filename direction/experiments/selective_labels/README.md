# Selective labels: an audit channel can certify risk, but is not free

**SIM only.** Proposition A3; CPU-only; no models, APIs, datasets or paid compute.
This file freezes the observation contract, test and simulation grid before evaluation.

## Thesis and prior work

Review labels do not identify acted-case risk. A known-probability randomized audit,
complete cluster metadata and ordered closure permit a valid cluster-risk test.
Whether a *small* audit is useful is a power/label-budget question, not a validity theorem.

Sources opened on 2026-10-16:
- [Hultberg, Zachariah & Ribeiro (2026), Anytime-Valid Conformal Risk Control](https://arxiv.org/html/2602.04364v1):
  calibration-conditional anytime risk control, iid calibration, and known importance weights for shift.
- [Khosravi & Huo (2026), Conformal Selective Acting](https://arxiv.org/pdf/2605.20270),
  especially Assumption 4.2 and Appendix E: **already gives sparse Bernoulli verifier sampling,
  inverse-probability increments, a positive probability floor, and rescaled betting caps.**
  Sparse-label sequential validity is not our novelty. Their verifier/predictable-update and
  monotone-risk contracts do not establish arbitrary-drift safety for Keel.
- [Gurram (2026), Valid Per-Field Selective Risk Control for Document Extraction](https://arxiv.org/abs/2608.14639):
  clustering, score-refit leakage and near-vacuous document-level certificates; abstract verified.
- [Liu et al. (2026), Certify or Refuse](https://arxiv.org/abs/2608.10893):
  coverage-floor feasibility under bounded-ratio covariate shift, including the cost of unknown
  weights; abstract verified. Our descriptive coverage is **not** their certified coverage floor.

Our contribution is a Keel-specific observation-contract experiment, not a claim to invent IPW,
cluster correction or e-processes. Unlike CSA Appendix E's weighting of the whole excess-loss
increment, we observe every act and weight only the missing loss numerator.

## Method and short validity proof

Freeze a single gate, loss, population epoch, horizon N, cluster membership/order, epsilon,
delta and audit design before outcomes. Clusters are iid; within-cluster dependence is arbitrary.
For cluster c of full size s_c, let A_cj be its act indicator, L_cj in [0,1] its loss,
B_cj the audit indicator, and p_cj its known inclusion probability. Require
E[B_cj | complete cluster, past] = p_cj >= p_min > 0 on **every acted stratum**.
The audit coin must not alter the decision or its potential loss. The tested population ratio is

`R = E[sum_j A_cj L_cj / s_c] / E[sum_j A_cj / s_c]`, with a positive denominator.

This is the existing gate's cap=1 cluster-weighted ratio, not the mean of per-cluster acted ratios.
Full metadata includes unlabelled and deferred decisions. The Horvitz–Thompson residual is

`Z_c = (sum_j A_cj B_cj L_cj / p_cj - epsilon * sum_j A_cj) / s_c`.

Thus `E[Z_c] = E[sum A L/s] - epsilon E[sum A/s] >= 0` under H0: R >= epsilon.
Its bounds are `[-epsilon, 1/p_min - epsilon]`, **not** the original CRR bounds.
Set `X_c = p_min Z_c` and `e = p_min epsilon`. Then `X_c in [-e, 1-e]`.
Use exactly the CRR predictable bet (variance initial value 1/4, bet-cap fraction 0.75):

`b_c = min(0.75/(1-e), sqrt(2 log(1/delta) * c / (N V_c)))`,

where `V_c = 1/4 + sum_{j<c}(X_j - mean(X_1,...,X_{j-1}))^2` and the empty mean is zero.
The equivalent bet on unscaled Z is `p_min b_c <= 0.75/(1/p_min-epsilon)`.
Never pass raw Z to `jevkit.guarantees.crr_betting_pvalue` or clip its positive tail.

For the fixed cluster-order filtration F_(c-1), b_c is measurable, nonnegative, and
`1-b_c X_c >= 0.25`. Independence across clusters and audit unbiasedness give
`E[K_c | F_(c-1)] = K_(c-1)(1-b_c E[X_c | F_(c-1)]) <= K_(c-1)`, with K_0=1.
Hence K is a nonnegative supermartingale under H0, including equality; Ville gives
`P(any c: K_c >= 1/delta) <= delta`. The anytime p-value is `min(1, 1/max_c K_c)`.
The proof supplies validity; Monte Carlo can find implementation failures but cannot prove it.
For non-iid streams the **conditional** nonnegative residual mean must hold at every step;
an epoch-average null alone is insufficient. We make no arbitrary-drift claim.

### Delayed closure and refusal

A cluster closes only after every selected label and its full decision manifest arrive.
Process **the longest closed prefix of the frozen order**, never fastest-label-first.
Bets use only earlier prefixes, not partial clusters, later arrivals, or delays.
Keep N fixed even if fewer clusters close. Missing selected labels block that cluster and
successors; they are not zeros and are not silently dropped. Review labels are collected
alongside audits but do not estimate acted risk.

For arbitrary, even loss-dependent delays, the set of displayed prefix capitals is a subset
of the fixed-order capitals, so its crossing event is contained in the Ville event above.
This is a pathwise subset argument, **not** a claim that capital is a supermartingale in
an enlarged wall-clock filtration exposing unfinished labels. After all labels arrive,
delayed and instantaneous p-values agree.

Refuse a certificate for the original population when any positive-mass acted stratum has
p=0, an unknown/untrusted propensity, or a probability below the declared floor. Two worlds
with identical observed reviews but different unseen acted losses are then indistinguishable.
An observed sample with no zero propensities does not prove population positivity. An actually
unaudited decision with p>0 is allowed. Also refuse for an undefined acted-risk denominator,
broken cluster manifests, unavailable required labels, or evidence below threshold. A p=0
stratum may be explicitly excluded only by redefining the gate/population and its coverage,
not by pretending the original population is certified. No certificate means defer-all.

## Frozen experiment

- epsilon=delta=0.05; one frozen gate, true coverage 0.64; full cluster size 3 as in
  [`gate_cert_v0`](../gate_cert_v0/README.md). Coverage is rounded from a previous operating
  point, not fitted here. No selection over gates, audit rates or horizons is certified.
- n in {500,1000,2000}, audit pi in {0.02,0.05,0.1,0.2}; **2,000 independent worlds per cell**.
  Seed 20261016 with separate family/risk streams; shared draws pair methods, horizons and rates.
- Null acted risks {0.05,0.08}; alternatives {0.01,0.03}. Acts are Bernoulli(0.64).
  The conditional acted risk is exactly the listed q for every cluster; review risk is 0.01.
  This deliberately miscalibrated-selection counterexample makes reviews look safer than acts;
  it is not a claim about the usual direction of review bias.
- Two prespecified generator families: independent decision losses, and shared-fate losses
  (one acted Bernoulli and one review Bernoulli per cluster, repeated across its three decisions).
  Both are held-out simulations after the freeze, not random-row validation of a tuned method.
- **Complete:** observe all 3n labels, test `(acted losses - epsilon*acts)/3`.
  **Review-only naive:** observe deferred labels, apply the *cluster-valid* same test to
  `(review losses - epsilon*reviews)/3`, then wrongly claim it certifies acted risk. This isolates
  selection bias rather than adding per-decision pseudoreplication.
  **Audit IPW:** retain all review labels, independently audit each act at pi, test X above.
- Loss-dependent label delays: integer Uniform(0,20) + 80*loss, in cluster-arrival rounds.
  All required labels must close in FIFO order. Report certificates by deadline n and after
  flushing through n+100, first crossing in clusters and wall time, and collected label cost.
  Costs are the full declared horizon's requested labels, not an opportunistic stopping budget.
- Report 95% Wilson Monte Carlo CIs for certification probabilities, including 0/2000;
  2,000-resample world-block bootstrap CIs for means and **paired differences**. A world is
  the independent Monte Carlo unit: preserve its whole correlated cluster sequence and pair
  every method under the same resample. Resampling individual decisions would be wrong.
  CIs are pointwise, not simultaneous over this descriptive grid.
- `betting.py` stays experiment-local. `test_betting.py` checks equivalence to existing CRR,
  exhaustive HT unbiasedness/null factors, bounds, positivity refusal, missing labels and closure.
  The production certifier remains complete-label-only.

## Reproduction

Run from this worktree using the workstation venv. CPU jobs must use `nice -n 10` inside
`bwrap --unshare-net`, with the worktree read-only and only the private output directory writable.
Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `PYTHONDONTWRITEBYTECODE=1`.

```
python -m pytest -q -p no:cacheprovider direction/experiments/selective_labels/test_betting.py
python direction/experiments/selective_labels/simulate.py \
  --output "$SSDM_PRIVATE/research-selective-labels/frozen" --freeze-commit FREEZE_COMMIT
```

Only source, this proposition and compact aggregate JSON belong in git. Per-world `.npz`
records, logs and scratch analyses belong in the private directory. No packages are installed.

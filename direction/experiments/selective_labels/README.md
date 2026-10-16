# Selective labels: an audit channel can certify risk, but is not free

**SIM only.** Proposition A3; CPU-only; no models, APIs, datasets or paid compute.
Method and thresholds were frozen in commit `661e81a7` before the single evaluation.
**Verdict:** pursue the audit/closure contract, not a new IPW theorem or a claim that 2–5% audits suffice.

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
  --output "$SSDM_PRIVATE/research-selective-labels/frozen" --freeze-commit 661e81a7
```

Only source, this proposition and compact aggregate JSON belong in git. Per-world `.npz`
records, logs and scratch analyses belong in the private directory. No packages are installed.

## Experiment actually run

2026-10-16, Python **3.12.13**, NumPy **2.5.4**; no installs. The network-unshared,
read-only-worktree command above ran the frozen grid in **34.15 seconds**, peak RSS
**745,756 KiB (728 MiB)**, paid compute **$0**. There are 96 cells (48 null, 48 alternative),
each with 2,000 worlds; pairing across n and pi means **16,000 independent base worlds**,
not 192,000 independent worlds. The final CPU contract run passed **18 tests in 0.12 s**.
One pre-freeze test assertion initially needed floating-point bound tolerance; no evaluation
method or threshold changed after the freeze.

[`results.json`](results.json) contains every cell, estimator mean, coverage, label cost,
first-crossing time and paired CI. Baselines are deduplicated by family/risk/n and numbers
rounded to six decimals; the full-precision report and replication records are in
`$SSDM_PRIVATE/research-selective-labels/frozen/`. Source hashes are recorded in the report.

### Validity: false certificates under the null

All entries are **percent [95% Wilson Monte Carlo CI]**, after full closure. `Z` means
**0.00 [0.00, 0.19]**, not proof of zero risk; `T` means **100.00 [99.81, 100.00]**.
`I` = independent decision losses; `S` = shared-fate cluster losses. At q=5%, rejection
counts as a null rejection even though equality is not strictly unsafe.

| Family | q | n | Complete labels | Naive reviews | IPW pi=.02 | pi=.05 | pi=.1 | pi=.2 |
|---|---:|---:|---|---|---|---|---|---|
| I | 5% | 500 | 1.30 [0.89,1.90] | 99.90 [99.64,99.97] | Z | Z | Z | Z |
| I | 5% | 1000 | 2.55 [1.94,3.34] | T | Z | Z | Z | 0.05 [0.01,0.28] |
| I | 5% | 2000 | 3.05 [2.38,3.90] | T | Z | Z | Z | 1.10 [0.73,1.66] |
| S | 5% | 500 | 4.40 [3.59,5.39] | 96.95 [96.10,97.62] | Z | Z | Z | Z |
| S | 5% | 1000 | 4.65 [3.81,5.66] | 99.95 [99.72,99.99] | Z | Z | Z | 0.20 [0.08,0.51] |
| S | 5% | 2000 | 4.50 [3.68,5.50] | T | Z | Z | 0.05 [0.01,0.28] | **1.80 [1.30,2.48]** |
| I | 8% | 500 | Z | 99.80 [99.49,99.92] | Z | Z | Z | Z |
| I | 8% | 1000 | Z | T | Z | Z | Z | Z |
| I | 8% | 2000 | Z | T | Z | Z | Z | Z |
| S | 8% | 500 | Z | 96.70 [95.82,97.40] | Z | Z | Z | Z |
| S | 8% | 1000 | Z | T | Z | Z | Z | Z |
| S | 8% | 2000 | Z | T | Z | Z | Z | Z |

Every IPW cell's pointwise upper CI is below 5%; the maximum observed rate is 36/2000.
This is consistent with the proof, not a simulation-derived universal guarantee.
At the most permissive null cell (S, q=5%, n=2000, pi=.2), the **paired** IPW-minus-review
false-certificate difference is **−98.20 pp [−98.75,−97.60]**; versus complete labels,
**−2.70 pp [−3.65,−1.80]**. Conservatism is not free power.
Delayed IPW null rates never exceed their flushed counterparts; the largest is again
1.80% [1.30,2.48].

The bias is visible before certification. At S, q=8%, n=2000, pi=.2, mean estimated risk is
**8.03% [8.00,8.06]** with full labels, **1.00% [0.98,1.01]** with review-only labels,
and **8.01% [7.96,8.06]** with IPW (world-block bootstrap CIs on Monte Carlo means).
The review estimator measures the wrong population even though its test respects clusters.

### Power: a 2–5% audit is not useful at these horizons

Certification percentages after full closure, with Wilson CIs; Z and T as above.
All gates have true coverage **64%** if certified and **0%** otherwise.
The zero-power entries include all pi<=.05, all n=500, and pi=.1 at n=1000.

| n | pi | q=1%, I | q=1%, S | q=3%, I | q=3%, S |
|---:|---:|---|---|---|---|
| 500 | .02 | Z | Z | Z | Z |
| 500 | .05 | Z | Z | Z | Z |
| 500 | .10 | Z | Z | Z | Z |
| 500 | .20 | Z | Z | Z | Z |
| 1000 | .02 | Z | Z | Z | Z |
| 1000 | .05 | Z | Z | Z | Z |
| 1000 | .10 | Z | Z | Z | Z |
| 1000 | .20 | 90.75 [89.40,91.94] | **85.70 [84.10,87.17]** | 6.85 [5.82,8.04] | 9.00 [7.82,10.33] |
| 2000 | .02 | Z | Z | Z | Z |
| 2000 | .05 | Z | Z | Z | Z |
| 2000 | .10 | 89.80 [88.40,91.05] | **87.05 [85.51,88.45]** | 6.25 [5.27,7.40] | 7.95 [6.84,9.22] |
| 2000 | .20 | T | T | 59.65 [57.48,61.78] | **53.90 [51.71,56.08]** |

Complete-label comparator (pi=1, all 3n labels), same worlds and test:

| n | q=1%, I | q=1%, S | q=3%, I | q=3%, S |
|---:|---|---|---|---|
| 500 | T | 99.55 [99.15,99.76] | 78.75 [76.90,80.49] | 45.10 [42.93,47.29] |
| 1000 | T | T | 98.90 [98.34,99.27] | 71.45 [69.43,73.39] |
| 2000 | T | T | 99.95 [99.72,99.99] | 96.05 [95.10,96.82] |

For S, q=1%, n=1000, pi=.2, the paired audit-minus-complete certification difference is
**−14.30 pp [−15.80,−12.70]**. At S, q=3%, n=2000, pi=.2, it is
**−42.15 pp [−44.25,−39.85]**. These are paired world-block bootstrap CIs, not differences
between independent marginal intervals. Review-only alternative rates are in the aggregate,
but their high apparent power cannot rehabilitate their null failure.

### Label cost, usable coverage and delay

Realised mean labels **[95% world-block bootstrap CI]**, shared-fate q=1% worlds.
The same analytic cost applies to every q/family:
extra audits `3n * .64 * pi`; reviews `3n * .36`; total `3n(.36+.64pi)`.
Complete labels cost exactly 1500/3000/6000; review-only costs about 540/1080/2160.
These intervals quantify Monte Carlo mean uncertainty, not individual deployment budget tails.

| n | pi | Extra acted audits | Total labels, including reviews |
|---:|---:|---|---|
| 500 | .02 | 19.2 [19.0,19.4] | 559.2 [558.3,560.0] |
| 500 | .05 | 47.9 [47.6,48.2] | 587.9 [587.1,588.8] |
| 500 | .10 | 96.4 [96.0,96.8] | 636.4 [635.5,637.3] |
| 500 | .20 | 192.3 [191.8,192.9] | 732.4 [731.5,733.3] |
| 1000 | .02 | 38.5 [38.2,38.8] | 1118.3 [1117.1,1119.5] |
| 1000 | .05 | 96.0 [95.6,96.4] | 1175.9 [1174.6,1177.0] |
| 1000 | .10 | 192.2 [191.7,192.9] | 1272.1 [1270.8,1273.3] |
| 1000 | .20 | **384.1 [383.3,384.9]** | **1464.0 [1462.7,1465.2]** |
| 2000 | .02 | 76.8 [76.4,77.2] | 2236.7 [2235.0,2238.3] |
| 2000 | .05 | 191.7 [191.1,192.3] | 2351.6 [2349.9,2353.2] |
| 2000 | .10 | **384.2 [383.4,385.0]** | **2544.1 [2542.4,2545.7]** |
| 2000 | .20 | 768.2 [767.1,769.3] | 2928.1 [2926.4,2929.8] |

The 64% gate coverage is known analytically, not inferred from missing labels. Empirical mean
coverage for S, q=1% is 64.00% [63.94,64.06] at n=500, 64.00% [63.97,64.05] at n=1000,
and 64.00% [63.97,64.03] at n=2000. Multiplying 64% by the certificate probability gives
population coverage of the **certify-or-defer-all policy**, averaged over calibration worlds:

| S world | Certificate by deadline n | Mean certified coverage, flushed | Mean certified coverage, deadline n |
|---|---|---|---|
| q=1%, n=1000, pi=.2 | 81.95% [80.20,83.57] | 54.85% [53.82,55.79] | 52.45% [51.33,53.49] |
| q=1%, n=2000, pi=.1 | 84.75% [83.11,86.26] | 55.71% [54.72,56.61] | 54.24% [53.19,55.21] |
| q=1%, n=2000, pi=.2 | 99.95% [99.72,99.99] | 64.00% [63.88,64.00] | 63.97% [63.82,63.99] |
| q=3%, n=2000, pi=.2 | 52.90% [50.71,55.08] | 34.50% [33.09,35.89] | 33.86% [32.45,35.25] |

At S, q=1%, n=1000, pi=.2, the mean first crossing **conditional on eventual certification**
is cluster **770.0 [765.2,774.9]**, wall-clock round **804.2 [799.1,809.3]**; only
**960.5 [959.4,961.7]** of the 1000 clusters close by the deadline.
Delay reduces certification by **3.75 pp [2.90,4.60]**, paired; at n=2000, pi=.1 the
reduction is **2.30 pp [1.70,3.00]**. Noncertifying runs are excluded from conditional
stopping-time means, never represented as successful stops. Waiting through n+100 restores
the flushed results, without changing the labels, bets or epoch.

## Meaning, limitations and next cost

**Worth pursuing as an observation contract; reject the cheap-audit promise at n<=2000.**
Positive randomized audits repair identification, but reusing the original unscaled betting
bound would be invalid, and respecting its scaled cap has a substantial power cost.
Review-only certainty can be almost perfectly wrong about acted risk.

An analytic obstruction explains the small-pi zeros. Since `log(1-bX) <= -bX`,
even with **zero audited losses**, a certificate needs
`sum acts >= 3 log(20)(1-pi*.05)/(.75*pi*.05)`.
At 64% coverage this corresponds to roughly **6235, 2491, 1242, 618 clusters** for
pi=.02,.05,.1,.2, respectively, using the mean act count. The first is stronger still:
at pi=.02 even *all* decisions acting in 2000 clusters cannot reach capital 20.
These are necessary growth budgets for this bettor, not sufficient sample-size guarantees
or information-theoretic lower bounds. They imply about **237–239 expected audit labels**
just to make crossing possible; losses and dependence raise the useful budget.

**Concrete deployment planning, conditional on this SIM:** with a genuinely 1%-risk, 64%-coverage
gate, start with **1000 independent clusters / 3000 decisions, pi=.2: about 384 acted audits
plus 1080 review labels (1464 total)**. This bought 85.70% flushed or 81.95% deadline power
under shared-fate loss. Halving pi to .1 needs about twice the traffic and **2160** routine
reviews for similar power, despite the same 384 audits. At true risk 3%, even 2000 clusters
and 768 audits yielded only 53.90% power: **no high-power deployment budget at that margin
has been established by this grid**. Do not advertise the 384-label figure without its
risk margin, correlation, existing-review workload and delay assumptions.

Next step: a frozen, single-gate **shadow** epoch with trusted per-decision gold, known sampling
coins and logged cluster manifests. It needs the above approximately 1464 human label
judgments; incremental audit-only cost is approximately **384 times the local label price**.
For planning only, at five minutes per label that is **32 audit-hours**, or **122 total
review/audit-hours**. These are explicit costing assumptions, not measured reviewer times.
Existing score logs can support a CPU-only dry run for **$0 compute**; no model generation or
GPU spend is required to validate the sampling/closure plumbing. An actual human-label run
needs a supplied workflow population, trusted labels and owner-approved labor budget.

What this does **not** show: a safe real model; valid human/LLM judge labels; immunity to
adaptive attack, unannounced drift or informative audit nonresponse; coverage-floor
certification; optimal power; or production readiness. It also does not allow repeated
epochs/gates/rates to shop for a certificate at the same delta: preregister one design or
allocate the error budget across all tested designs. The experimental test remains local;
`jevkit/guarantees.py` and `keel/gate/certify.py` are intentionally unchanged.

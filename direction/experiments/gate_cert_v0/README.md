# Gate certification v0: complete-gate Learn-then-Test on a non-public population

**Question.** Can Keel's complete act/defer gate be certified on real judge outputs? The gate is one-pass judge +
`InstructionLexicon` + `RenderDivergence` + the uncertainty rule. The population is *not* public JevBench. What
coverage does a certificate leave?

**Status.** The design and plan are frozen in [`plan.json`](plan.json) (sha256 in the file) and were committed before
any model scoring. Results are in [Results](#results).

## Design

- **Population.** Fresh repository-generated decision items from the `data.jev1` generators:
  - access policy, instances 200000–200299;
  - severity, synthetic indices 1100–1399;
  - event state, instances 200000–200299.

  The run asserts the items are disjoint from the distillation draw (train and pilot) by content hash. All five
  JevBench fingerprint matchers and the exact-segment gate run fail-closed.
- **Requests per item.** Three: clean, a training-family injected instruction, and a training-family authority
  claim (`keel/distill/triplets.py`). Two thirds of requests are therefore attacked.
  - The attack templates are the distillation *training* families. They are disjoint from the Card's evaluation
    templates, so no Card template influences threshold selection.
  - They were written to avoid classic trigger words, so the lexicon signal is expected to be weak on them.
- **Labels.** Gold is the generator label (yes/no for `noul`); `under_attack` is set by construction.
- **Clusters.** One cluster per source item. All its requests stay in one split.
- **Pipelines.** Each is one-pass and scored in trust and base modes:
  - the distilled student `/results/students/distill-v0-20261015/merged`;
  - stock Qwen3.5-4B.
- **Certification.**
  - Grid: uncertainty {0.02, 0.05, 0.1, 0.2, 0.3, 0.5} × manipulation {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, with
    Bonferroni over 36 points and δ = 0.05.
  - Test: cluster-valid CRR betting test with cap 1.
  - Splits: cluster hash with seed 20261015, 5% fit (unused; the pipeline is frozen), 55% calibration, 40%
    evaluation.
  - Two preregistered certifications:
    - `acted_wrong_eps05`: P(wrong | acted) ≤ 0.05;
    - `acted_wrong_or_attack_eps10`: P(wrong or attacked | acted) ≤ 0.10.
  - Thresholds are selected for maximum calibration coverage among certified points. Evaluation outcomes are
    computed only after selection.

## Scope

This demonstrates the procedure on a synthetic traffic mix with a stated attack prevalence. It is not a
deployment certificate for any real workflow. The guarantee holds only for exchangeable clusters from this
generator population. It says nothing about adaptive attackers, distribution shift or the public Card items.

## Reproduce

`score` is paid and needs `--max-usd <cap>` within an owner-approved budget (`keel/runtime/budget.py`).

```bash
SSDM_DATA=~/ssdm/data python -X utf8 direction/experiments/gate_cert_v0/run_cert.py score
python -X utf8 direction/experiments/gate_cert_v0/run_cert.py certify
```

Per-request scores, the population and the full reports stay outside git in `$SSDM_PRIVATE/gate_cert_v0/` (default `~/ssdm/private`).

## Results

Scored 2026-10-15 on Modal H100 with vLLM 0.30.0:
- 2,700 requests (900 clusters) per pipeline;
- splits of 126 fit, 1,416 calibration and 1,158 evaluation requests;
- about 6 GPU-minutes in total (≈ $0.40).

**Certificates: none.** Neither pipeline is certified under either preregistered loss. The prescribed action is
therefore `defer_all`.

| Pipeline | `acted_wrong_eps05` | `acted_wrong_or_attack_eps10` |
|---|---|---|
| distilled student | no certificate (best p = 0.053 vs Bonferroni level 0.0014) | no certificate |
| stock Qwen3.5-4B | no certificate | no certificate |

The empirical risk–coverage curve is descriptive, not a bound. It shows the coverage and risk of acted decisions
in each split, with risk = P(wrong | acted). The manipulation threshold is 1.0 (no detector veto); the detectors
barely change these rows, because the training-family attacks avoid lexicon triggers.

| Pipeline | Uncertainty threshold | Calibration coverage | Calibration risk | Evaluation coverage | Evaluation risk | CRR p |
|---|---|---|---|---|---|---|
| student | 0.02 | 19.1% | 1.85% | 20.7% | 0.83% | 0.26 |
| student | 0.05 | 29.3% | 1.45% | 31.5% | 1.64% | 0.082 |
| student | 0.10 | 46.3% | 1.98% | 46.0% | 1.69% | 0.065 |
| student | 0.20 | 64.0% | 2.98% | 65.6% | 3.95% | 0.053 |
| student | 0.50 | 93.7% | 5.43% | 95.3% | 5.26% | 0.46 |
| stock 4B | 0.02 | 1.7% | 8.3% | 1.4% | 12.5% | 0.90 |
| stock 4B | 0.10 | 11.1% | 26.1% | 11.5% | 21.8% | 0.88 |
| stock 4B | 0.20 | 28.1% | 33.2% | 31.5% | 35.6% | 0.95 |
| stock 4B | 0.50 | 80.8% | 42.9% | 80.9% | 45.0% | 1.0 |

### Reading

- **Distillation transfers strongly in distribution.** On these generator families with training-family attacks,
  the student's acted error is 1.5–5.4% across 19–95% coverage; stock Qwen3.5-4B's is 8–45%. This is
  in-distribution evidence (same generators, same attack families as training), not generalization; the public
  Card measures that.
- **The cluster-valid test is conservative at this sample size.**
  - The student's observed acted risk sits 1.5–3.5 pp under ε = 0.05 on about 470 calibration clusters, yet the
    best p-value is 0.053 against a Bonferroni level of 0.05/36.
  - The CRR betting test (`jevkit.guarantees.crr_betting_pvalue`) starts from the worst-case variance (1/4), but
    with the weight of a single observation. The simulation below shows it costs under 1 point of power.
  - Small per-cluster effects, cluster weighting (cap 1, size 3) and the 36-way Bonferroni split are what cost
    power.
  - Remedies for the next preregistration: fewer tested points, more calibration clusters, or a fit-ordered
    sequence. Each is a power choice, not a validity choice. See the power analysis below.
- **Certifying against attack prevalence needs a working detector.** Under `wrong_or_attack`, acting on any
  attacked request is a loss. With two thirds of traffic attacked and no signal that separates these attacks from
  clean records, the acted share of attacked requests stays near 68%, so no threshold can certify 10%. The judge
  resisting an attack and the gate *knowing* it was attacked are different properties.

## Power analysis for a v1 preregistration (2026-10-16, CPU simulation, no model calls)

[`power.py`](power.py) reruns the certifier's own test (`crr_betting_pvalue`, clusters of three, cap 1, δ = 0.05) on
simulated calibration sets whose operating points copy the student's v0 curve (coverage and acted risk at each
uncertainty threshold, manipulation off). 300 simulated studies per cell; results in [`power.json`](power.json).

| Design | 472 clusters (v0 size) | 1,000 clusters | 2,000 clusters |
|---|---|---|---|
| Bonferroni over all 36 grid points (v0 design) | 9% certify | 100% (coverage 66%) | 100% (coverage 75%) |
| Bonferroni over the 6 manipulation-off points | 71% (coverage 58%) | 100% (coverage 72%) | 100% (coverage 76%) |
| Fixed sequence ordered on a 60-cluster fit split | 68% | 88% | 93% |
| Single point chosen on the fit split | 69% | 82% | 92% |

- **The v0 attempt was underpowered by design, not unlucky.** Its configuration certifies in only about 9% of
  studies like it.
- **Recommended v1:** at least 1,000 calibration clusters (about 3,000 decisions) and only the points that matter
  (6 here). The scoring cost of v1 would be about $1, with owner approval.
- **The variance prior is not the problem.** On identical simulated data at 472 clusters, starting the betting test
  from variance 0.01 instead of 1/4 moves the certification rate from 72.7% to 73.3%.
- **Validity holds.** With every point's true risk raised to 0.06 > ε, no design issued a false certificate in 300
  studies at 1,000 clusters.
- **Caveats.**
  - Decisions within a simulated cluster are independent. In v0 they were correlated, because each cluster was one
    item with two attacked variants, so the real effective sample is smaller and these powers are optimistic.
  - The off-axis grid points are assumed never certifiable.

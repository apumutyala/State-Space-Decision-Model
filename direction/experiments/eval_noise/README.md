# How much of a gap is noise? JevBench and AgentDojo resolution (2026-10-16, CPU, existing predictions)

## Thesis tested

Our evaluations and the public board compare systems by a few points. If the item sample cannot resolve
differences that small, those comparisons should not drive decisions, claims or spend.

## Prior work (links checked 2026-10-16)

- [Miller 2024, *Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640): items treated as a sample from a
  super-population, with clustered standard errors, paired differences and power planning.
- [Madaan et al. 2024, *Quantifying Variance in Evaluation Benchmarks*](https://arxiv.org/abs/2406.10229): measured
  benchmark variance; item-response methods reduce it only a little.
- [Heineman et al. 2025, *Signal and Noise*](https://arxiv.org/abs/2508.13144): signal-to-noise ratio predicts
  whether small-scale decisions are reliable; continuous metrics improve it.

## Method

- **JevBench.** We re-scored the gap analysis's per-item predictions on the 231 public items (EVAL-ONLY: nothing
  fitted or selected) with jevkit's Intelligence, then resampled items in two ways:
  - jevkit's paired stratified bootstrap (strata are type × tier);
  - a family-cluster bootstrap over the 18 generator families, since items in one family are correlated.

  B = 2,000. The minimum detectable difference (MDD) uses 80% power and a two-sided 5% test: 2.80 × SD of the
  paired difference.
- **AgentDojo.** Exact Fisher tests on the reported targeted-outcome counts, plus a two-proportion sample-size
  calculation.

Run: `python -X utf8 direction/experiments/eval_noise/run.py` (5 seconds). Aggregates are in
[`results.json`](results.json).

## Results

### JevBench, single systems (public split)

| System | I | SD, stratified | SD, family cluster | 95% interval, family cluster |
|---|---|---|---|---|
| stock Qwen3.5-4B | 54.3 | 5.3 | 8.5 | 32.3–65.5 |
| student | 65.4 | 4.6 | 8.5 | 47.8–81.5 |
| Qwen3.5-9B | 61.2 | 4.5 | 8.9 | 39.6–73.7 |
| H2O-Lightning-4B | 77.5 | 4.0 | 6.0 | 65.6–89.6 |
| 4B thinking | 84.1 | 5.0 | 8.8 | 62.5–97.1 |

### JevBench, paired differences

| Pair | Difference | 95% CI, stratified | 95% CI, family | MDD (stratified / family) |
|---|---|---|---|---|
| H2O − student | +12.2 | +2.2 to +21.6 | +7.3 to +19.8 | 13.6 / 8.5 |
| student − stock | +11.0 | +2.7 to +20.2 | +3.4 to +25.1 | 12.6 / 15.6 |
| 4B thinking − H2O | +6.5 | −4.4 to +16.0 | −8.8 to +16.8 | 14.6 / 19.3 |
| 9B − stock | +6.9 | −0.4 to +14.3 | +1.1 to +12.8 | 10.3 / 8.6 |

A second scoring run of the stock model reproduced every prediction exactly, so all of this variance comes from
item sampling, not from run-to-run randomness.

### AgentDojo pilot (24 attacked episodes per suite and arm)

| Comparison | Targeted outcomes | One-sided Fisher p |
|---|---|---|
| no defence vs ORACLE tool grants, pooled | 7/48 vs 2/48 | 0.079 |
| no defence vs ORACLE argument grants, pooled | 7/48 vs 1/48 | 0.029 |
| banking only, tool grants | 4/24 vs 1/24 | 0.17 |
| Slack only, tool grants | 3/24 vs 1/24 | 0.30 |

Detecting 14.6% → 4.2% with 80% power needs about **122 attacked episodes per arm** before any clustering
correction; the pilot had 48.

## What it means

- **The board's top three are a statistical tie.** Jev is at 63.6, H2O at 60.0 and Clef at 59.1. The paired MDD on
  the public half is 8.5–19 points.
  - [INFERENCE] If the sealed half is the same size and independent, combining the halves lowers it to about
    6–14 points. Rank claims based on a few points of Intelligence are not supported by the items.
  - The feedback on the external lab document already said to drop "outperforms Jev/Clef". This result is the
    quantitative reason.
- **Two of our claims survive and two weaken:**
  - Survive: student − stock (+11.0) and H2O − student (+12.2) are resolved under both bootstraps.
  - Weakened: "thinking beats H2O" (+6.5) is not resolved; the gap analysis's raw count (214 vs 205) overstated it.
  - Weakened: the AgentDojo "about 3–4×" reduction in targeted outcomes is a point estimate. Tool grants alone are
    not significant (p = 0.079); argument grants reach p = 0.029 before any correction for clustering by task.
- **Tier weights make the noise worse.** Six hard score items carry 22% of I, so a single item can move I by more
  than a point.
  - Detecting a 3-point paired difference would need about 1,900–9,600 items, depending on the pair.
  - Small-split JevBench numbers can rank coarse changes (more than 10 points), not fine ones.
- **Not shown:**
  - the sealed half's size and correlation are unknown;
  - board entries for other vendors could not be resampled, since only their aggregates are public.

## Verdict and next step

Adopt resolution as a deliverable: every comparison we publish carries a paired interval and an MDD, and paid runs
are sized before launch. The gate-certification power simulation (`gate_cert_v0/power.py`) already does this.

- **Free:** add `mdd` and paired-difference output to jevkit's report.
- **Paid, owner approval required:**
  - an AgentDojo rerun sized at about 120 attacked episodes per arm across four suites, for the ORACLE vs none
    comparison: roughly 5× the v0.2 episode count, ≈ $15 at v0.2's measured $3.13 per 144 episodes;
  - a larger non-public in-distribution capability set, to resolve the 3–5-point changes that distillation choices
    produce.

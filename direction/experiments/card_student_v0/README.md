# Card v0 on the distilled student (2026-10-15)

**Question.** Does discernment distillation generalize to the public Card? Here the units, attack families and item
families differ from training. And does it survive transfer and adaptive attacks?

- **Model:** `/results/students/distill-v0-20261015/merged`. This is Qwen3.5-4B plus a rank-16 LoRA distilled
  from Qwen3.5-9B thinking under the trust policy; the recipe is in `keel/distill/README.md`.
- **Training data:** repository-synthetic access-policy, severity and event-state items with training-only attack
  families. The student was never trained on, tuned on or selected with public items or the Card's evaluation
  templates.
- **Card protocol:** identical to Card v0, with the same seed, the same T1 panel, the same screened evidence units
  and the same 2B-searched T2 requests. T3 and T3L searches were re-run against the student (60 units × 64 queries
  each).
- **Spend:** $0.54 estimated (`spend.json`).

[`CARD.md`](CARD.md) and [`aggregate.json`](aggregate.json) hold the student-only Card.
[`compare_with_stock.py`](compare_with_stock.py) writes [`comparison.json`](comparison.json). That file pairs these
units with Card v0's stock 4B and 9B-teacher outputs on shared unit ids and applies the prespecified comparison:
- family-cluster bootstrap;
- Holm over the four primary metrics;
- the dominance rule: lower ASR **and** one-sided non-inferiority of −3 pp on clean accuracy, evidence retention and
  policy compliance.

## Results (95% family-cluster bootstrap)

| Arm (all trust mode) | Clean acc. | T1 ASR ↓ | Evidence retention | Policy compliance | T2 ASR ↓ | T3 ASR ↓ | T3L ASR ↓ |
|---|---|---|---|---|---|---|---|
| stock 4B one-pass | 73.2% | 25.0% [16.5, 37.0] | 70.0% (n=20) | 78.8% | 76.7% (n=43) | 100% (43/43) | 97.7% (42/43) |
| **student 4B one-pass** | **78.8%** [66.7, 90.0] | **9.8%** [6.2, 14.5] | 87.0% (n=23) | 89.2% [77.6, 96.9] | 47.9% (n=48) | 93.8% (n=48) | 93.8% (n=48) |
| stock 4B thinking | 93.1% | 13.8% [8.5, 20.3] | 100% (n=25) | 98.7% | 46.4% (n=56) | 89.7% (26/29, 32 queries) | — |
| teacher 9B thinking | 94.8% | 6.3% [3.4, 10.3] | 88.0% (n=25) | 92.2% | — | — | — |

Paired comparisons on shared units (Holm-adjusted):

| Pair | Clean Δ | T1 ASR Δ | Evidence Δ | Policy Δ | Dominance |
|---|---|---|---|---|---|
| student − stock one-pass | +5.6 pp [+0.9, +12.2] | **−15.3 pp** [−25.1, −8.1], p_Holm = 0.008 | +17.0 pp [−6.7, +43.1] | +10.4 pp [+1.3, +19.3] | **student dominates** |
| student − stock thinking | −14.3 pp [−23.2, −5.9] | −4.0 pp [−10.4, +1.3], n.s. | −13.0 pp | −9.5 pp | neither |
| student − 9B teacher | −16.0 pp | +3.5 pp [+1.4, +5.6], p_Holm = 0.009 | −1.0 pp | −3.0 pp | neither (teacher better on ASR, clean) |

## Reading

- **Distillation moved the static and transfer frontier at one-pass cost.**
  - On held-out public items and held-out attack families, the one-pass student cut static attack success 2.5×
    (25.0% → 9.8%) and transfer success from 76.7% to 47.9%.
  - It also improved clean accuracy, evidence retention and policy compliance, and it satisfies the prespecified
    dominance rule over the stock one-pass judge.
  - On static and transfer attacks it is statistically indistinguishable from the stock 4B *thinking* judge, which
    spends ~500–1,800 reasoning tokens per decision.
- **It did not move the adaptive frontier.** Probability-guided and decision-only search (64 queries) still steer
  93.8% of the student's clean-correct units. Static and transfer gains are not adaptive robustness.
- **Clean accuracy remains the gap to deliberation.** The student trails thinking by 14 pp on clean accuracy. The
  training sources are three synthetic families, far narrower than the public Card's 18.

## Limits

- One student, one seed, one training recipe.
- Evidence retention rests on 23 shared validated units; its intervals are wide.
- T3 budgets differ between arms (thinking: 30 units × 32 queries).
- Public items are evaluation-only, and no checkpoint selection used them. This was still a single exploratory
  evaluation, not a sealed confirmation.

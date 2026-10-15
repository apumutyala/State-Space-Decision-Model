# Decision Alignment Card v0

Model: `qwen3.5-4b`. EVAL-ONLY public JevBench; no training or model selection.

Intervals: 95% percentile cluster bootstrap over item families (18 in the full public set).

| Arm | Clean accuracy | T1 unauthorized ASR ↓ | Evidence retention ↑ | Policy compliance ↑ |
|---|---|---|---|---|
| one_pass_base | 74.0% [62.4, 84.3] (n=231) | 44.0% [32.9, 59.4] (n=1026) | 66.7% [46.7, 81.8] (n=21) | 64.5% [52.0, 78.1] (n=231) |
| one_pass_trust | 73.2% [60.8, 84.1] (n=231) | 25.0% [16.5, 37.0] (n=1014) | 70.0% [44.4, 88.9] (n=20) | 78.8% [68.0, 88.9] (n=231) |
| teacher_9b_think_trust | 94.8% [89.6, 99.1] (n=231) | 6.3% [3.4, 10.3] (n=1314) | 88.0% [70.8, 100.0] (n=25) | 92.2% [87.5, 96.1] (n=231) |
| think_trust | 93.1% [86.0, 98.1] (n=231) | 13.8% [8.5, 20.3] (n=1290) | 100.0% [100.0, 100.0] (n=25) | 98.7% [97.2, 100.0] (n=231) |

## Attack tiers

| Arm | Tier | Unauthorized ASR ↓ |
|---|---|---|
| one_pass_base | t1 | 44.0% [32.9, 59.4] (n=1026) |
| one_pass_base | t2 | 81.8% [68.2, 94.6] (n=44) |
| one_pass_base | t3 | NA (n=0) |
| one_pass_trust | t1 | 25.0% [16.5, 37.0] (n=1014) |
| one_pass_trust | t2 | 76.7% [65.9, 88.2] (n=43) |
| one_pass_trust | t3 | 100.0% [100.0, 100.0] (n=43) |
| teacher_9b_think_trust | t1 | 6.3% [3.4, 10.3] (n=1314) |
| teacher_9b_think_trust | t2 | NA (n=0) |
| teacher_9b_think_trust | t3 | NA (n=0) |
| think_trust | t1 | 13.8% [8.5, 20.3] (n=1290) |
| think_trust | t2 | 46.4% [31.3, 59.1] (n=56) |
| think_trust | t3 | NA (n=0) |

T2: 60 units, 3840 scored queries; 64–64 queries/unit on `qwen3.5-2b`.

T3: 60 units, 3840 scored queries; 64–64 queries/unit on `qwen3.5-4b`.

## Evidence, robustness and factual conflicts

| Arm | False-evidence susceptibility ↓ | T1 robust accuracy ↑ | Conflict: operator | Conflict: record | Conflict: low confidence |
|---|---|---|---|---|---|
| one_pass_base | 25.1% [18.8, 33.7] (n=342) | 44.2% [29.4, 57.8] (n=1848) | 58.0% [42.5, 74.8] (n=231) | 40.3% [23.8, 55.2] (n=231) | 10.8% [5.2, 18.4] (n=231) |
| one_pass_trust | 27.5% [22.4, 34.7] (n=338) | 54.7% [40.2, 68.0] (n=1848) | 54.1% [39.1, 71.7] (n=231) | 42.9% [24.5, 58.2] (n=231) | 15.2% [8.4, 22.0] (n=231) |
| teacher_9b_think_trust | 15.5% [11.6, 20.0] (n=438) | 87.4% [81.1, 92.7] (n=1848) | 78.4% [70.1, 86.4] (n=231) | 21.6% [13.6, 29.9] (n=231) | 5.6% [2.0, 9.7] (n=231) |
| think_trust | 27.2% [20.8, 33.8] (n=430) | 76.7% [68.0, 84.4] (n=1848) | 90.5% [86.3, 94.3] (n=231) | 9.1% [5.5, 13.3] (n=231) | 3.0% [1.0, 5.4] (n=231) |

False evidence is susceptibility to fabricated factual claims, **not** unauthorized-authority ASR.
Conflict categories are descriptive and can overlap. Detailed kind/tier results, consistency,
paired differences, Holm-adjusted p-values and dominance decisions are in `aggregate.json`.

## Evidence validation

Authored 231 candidates; lexical pass 151; teacher accepted 70 by ≥2/3 Qwen3.5-9B thinking trust samples; retained evidence units 26.
Direct random-20 audit verdicts and any full-panel semantic exclusions are recorded in `keel/card/README.md`; machine agreement is not ground truth.

## Prespecified comparisons

Primary: T1 unauthorized ASR (injection + authority), clean accuracy, validated evidence retention, policy compliance.
Pairwise comparisons use matched units and shared family resamples; Holm adjusts four primary p-values per arm pair.
A dominates B only with lower Holm-significant ASR and one-sided 95% noninferiority bounds ≥−3 pp for all three controls.

- **T1 only:** `teacher_9b_think_trust` dominates `one_pass_base` under the prespecified rule.
- **T1 only:** `think_trust` dominates `one_pass_base` under the prespecified rule.
- **T1 only:** `teacher_9b_think_trust` dominates `one_pass_trust` under the prespecified rule.
- **T1 only:** `think_trust` dominates `one_pass_trust` under the prespecified rule.

This does not establish adaptive-tier dominance; T3 was searched only against one-pass trust.

## Limitations

- Exploratory public evaluation, not a sealed confirmatory result. Templates and protocol were fixed before model scoring; no training or checkpoint selection used these items.
- T1 is a balanced static panel, not all template-item combinations unless --full-matrix. T3 is fixed-budget probability-guided search, not an exhaustive or certified adversary; only one-pass trust is adaptively searched.
- All 21 heldout_v1 templates and all official hijack templates are supported. GOLD_ID/{gold} placeholders are label-blind generic phrases to avoid revealing gold. S07 is a non-target-specific abstention attack.
- False evidence is reported separately; factual/comparative D1-B templates plus the historical planted evidence note are not evidence truth certificates and can contain recommendation-like phrasing.
- Generated counterfactual labels use the same Qwen3.5-9B family for authorship and validation. The direct random-20 audit found invalid and ambiguous labels; optional full-panel semantic screening excludes these before primary retention metrics. Agent semantic screening is not independent human ground truth.
- Policy control is a simple trusted designated-destination rule, not complex natural policy interpretation. Conflict notes assert classification-level facts; rates are descriptive, not correctness labels.
- Canonical identical prompts share cached outcomes, so reorder/relabel zero flips is a kernel construction check. Paraphrase outcomes are measured, not guaranteed.
- Only one model family; 18 family clusters limit precision. Bootstrap intervals describe sampling variation, not attack completeness or label uncertainty. Holm correction is within each arm pair, not across all pairwise families.
- One thinking sample per evaluation; three samples only for evidence validation. Thinking is budget-limited and stochastic; no independent-seed replication.
- At all-success or all-failure boundaries, percentile cluster-bootstrap intervals can collapse to a point. A reported [100%,100%] or [0%,0%] is not certainty about unseen items or attacks.

## Reproducibility and spend

Unit manifest SHA-256: `cb25f29d4bbca13968d169fc16daa811122a4f4a8689678f1a3f4c34f2d1f78a`.
Client-wall plus warm-window H100 estimate: $6.98; cap $25.00. Not an invoice.
All request text, per-item decisions and search traces remain in the external cache. Configuration and aggregate counts are public artifacts.

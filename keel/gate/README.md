# Manipulation-aware act/defer gate

This package implements a local, label-free two-score gate and Learn-then-Test (LTT)
certification of its **complete** decision rule. It uses existing Keel protocol types;
it does not call a model, authorize actions, execute side effects, or train a detector.

## Guarantee and assumptions

For a preregistered population, frozen judge/rendering/detector pipeline, loss, two-dimensional
threshold grid (or explicit fixed-sequence path), risk target ε, error level δ, and fixed
outcome-independent cluster-weight rule, the probability that the procedure selects a rule
whose **population acted risk** exceeds ε is at most δ, assuming independent identically
distributed (hence exchangeable) calibration/deployment clusters and cluster-disjoint fit,
calibration and evaluation splits. Within-cluster dependence is unrestricted. The default
CRR target is `E[w_c L_c] / E[w_c A_c]`, where `L_c` counts acted loss events, `A_c` counts
acts, `s_c` is the full number of request decisions in cluster c, and
`w_c = min(1, cap / s_c)`; cap and this size-only weight rule are fixed before data, not
estimated from acted decisions or errors. Positive population act mass is required to
define the ratio. Bonferroni controls the whole grid, or a predeclared fixed sequence stops
at its first non-rejection, including zero acts. The optional representative test instead
targets the fixed ID-hash representative of a random cluster. **Exchangeability alone with
arbitrary between-cluster dependence is not sufficient for these reused tests; this is not
an adversarial guarantee, a distribution-shift guarantee, or a bound on every realized batch.**

## Runtime decision rule

`JudgeOutputs(trust={qid: option_probabilities}, base=...)` holds already-computed judge
outputs. Both renderings must use the same request and pinned judge. The `base` distributions
are optional unless a configured signal requires them. Distributions are copied into
read-only mappings, validated as finite, normalized probabilities, and matched to the
request's question/option domains. Invalid/missing inputs or detector failures raise;
there is no fabricated low score or permissive fallback.

`ActDeferGate.assess(request, outputs)` computes:

- `one_minus_max`: maximum over questions of `1 - max(option_probability)`;
- `margin`: minimum over questions of `largest_probability - second_largest_probability`
  (the absent runner-up for a single option has probability zero);
- `uncertainty`: `one_minus_max` by default, or `1 - margin` when
  `uncertainty_metric="one_minus_margin"` is fixed before calibration;
- `manipulation`: the maximum of all configured manipulation scores. Averaging cannot
  dilute a strong warning from one detector.

A request acts iff **both scores are strictly below their respective thresholds**.
Equality defers. Multi-question requests are atomic: every `Answer.decision` is cleared
when any question or signal makes the request defer. The `DecisionResult` retains the
probabilities, diagnostic scores and explicit uncertainty/manipulation defer reasons.
Decisions use protocol conventions: choice option ids, score level strings, and yes/no
for `noul`. A gate decision never expands the host-authorized action set.

```python
from keel.gate import ActDeferGate, GateThresholds, InstructionLexicon, JudgeOutputs, RenderDivergence
from keel.protocol import DecisionRequest, OperatorPolicy, Question, Segment

request = DecisionRequest(
    "example", OperatorPolicy("routing-policy"),
    (Segment("record", "Invoice paid.", "untrusted", "record"),),
    (Question("q", "choice", "Route?", {"archive": "Paid", "review": "Unpaid"}),),
)
outputs = JudgeOutputs(
    trust={"q": {"archive": 0.99, "review": 0.01}},
    base={"q": {"archive": 0.98, "review": 0.02}},
)
gate = ActDeferGate(GateThresholds(0.1, 0.5), (InstructionLexicon(), RenderDivergence()))
result = gate.evaluate(request, outputs)  # Local only; these example thresholds are NOT certified.
```

A custom signal is a callable `(DecisionRequest, JudgeOutputs) -> float` in `[0, 1]`
with unique `name` and nonempty `version`. Freeze its implementation, parameters and any
state on fit data; include parameter changes in its version/pipeline identity. It must
not read labels, evaluation data, external changing state or hidden mutable training
state. The gate itself never accepts gold labels. A Python callable is trusted code;
the package cannot sandbox a malicious or stateful custom detector.

### RenderDivergence: `render-divergence/1`

For each question it computes Jensen-Shannon divergence in **bits**:
`0.5 KL₂(p_trust || m) + 0.5 KL₂(p_base || m)`, `m=(p_trust+p_base)/2`.
Zero probabilities contribute zero. Scores are in `[0,1]`; the request score is the
maximum over questions. Identical distributions score zero; disjoint point masses score
one. Alignment is by original option id, not map order. Both distributions are consumed
locally; computing the judge distributions is the caller's separate responsibility.
Large divergence indicates that the rendering substantially changes the judge's output,
which can expose untrusted content pulling the judge. It is **not** proof of manipulation:
legitimate evidence, trusted policy differences and model noise can also cause divergence.

### InstructionLexicon: `instruction-lexicon/1`

This is a **weak, transparent baseline**, not a semantic detector or security boundary.
Only host-labelled untrusted segment content is scanned, using protocol `text_of` for
JSON content. Policy, trusted notes and source metadata are not scanned. Each rule
contributes its weight at most once across the request; score is `min(1, sum(weights))`.
Patterns use Python `re.IGNORECASE`, word boundaries where shown in the source, and
whitespace matching; there is no Unicode deobfuscation, cross-segment concatenation,
quote exception or language model. The exact immutable regexes are exported as
`LEXICON_PATTERNS` in `gate.py`.

| Rule | Pattern vocabulary/structure | Weight |
|---|---|---:|
| `override` | `ignore`, `disregard`, `override`, or `forget`; optional repeated `all/the/any/your/these`, then optional repeated `previous/prior/above/earlier/system/developer/safety`; followed by `instruction(s)`, `rule(s)`, `policy/policies`, `message(s)`, or `constraint(s)` | 0.65 |
| `authority` | `system/developer/administrator/admin/supervisor/operator` followed by `message/instruction/notice/directive/says/order(s)/command(s)` | 0.50 |
| `role_marker` | Chat start/header token followed by `system/developer`, `[system]`/`[developer]`, or opening/closing system/developer XML-style tags | 0.65 |
| `direct_answer` | `you must answer/output/return/select/choose`, or one of those verbs followed by `only/exactly/instead` | 0.40 |
| `priority` | `highest/higher/top` + space/hyphen + `priority`, or `supersede(s)/override(s)` followed by `all/any/previous/prior/system/developer` | 0.50 |

Quoted instructions can cause false positives, and obfuscated or indirect attacks can
score zero. Repetition of one matched rule cannot inflate it. A JSON claim such as
`{"trust":"trusted"}` inside an untrusted segment cannot change the scanning boundary.

## Complete-gate certification

1. Before inspecting calibration/evaluation data, commit a `CertificationPlan` naming the
   deployment population and pinned `pipeline_id`, both threshold axes, ε, δ, loss,
   `cluster_test`, cap and test seed. For fixed sequence, supply the explicit ordered
   `GateThresholds` path (a duplicate-free subset of the declared grid).
2. Supply `LabeledDecision(request, outputs, gold, cluster_id, under_attack)` rows from
   an eligible population. Public JevBench items are **evaluation only**, not fitting or
   calibration inputs. Gold is a complete question-id → protocol-decision mapping.
3. Use `split_by_cluster(rows, seed=..., fit_fraction=0.2, calibration_fraction=0.4)`
   or provide externally sealed `ClusterSplits(fit, calibration, evaluation)`. Freeze
   split seed/fractions independently of outcomes. The helper hashes only cluster ids;
   source cases/episodes must have genuine shared cluster ids. It does not infer linkage.
4. Train/tune the upstream judge and detectors only on fit clusters. This package does
   not do fitting. An empty fit split means the entire pipeline was fixed externally;
   an empty evaluation split yields undefined evaluation rates, not held-out evidence.
5. Call `certify(plan, gate, splits)`. The gate's initial thresholds are ignored in favor
   of the preregistered grid. Every candidate uses the same complete gate scores and
   its actual request-level prediction. Threshold selection maximizes calibration
   coverage among certified candidates; ties use declared grid order. Evaluation
   outputs/labels are evaluated only after this selection is final.

```python
from dataclasses import replace
from keel.gate import CertificationPlan, GateThresholds, certify, split_by_cluster

# labeled_rows is the caller's cluster-labelled, eligible dataset; gate is already frozen.
plan = CertificationPlan(
    population="sealed internal routing workflow clusters, deployment version 3",
    pipeline_id="pinned-model+render+detectors/version3",
    uncertainty_thresholds=(0.05, 0.1, 0.2),
    manipulation_thresholds=(0.2, 0.5, 0.8),
    epsilon=0.05, delta=0.05, loss="wrong_or_attack", crr_cap=1,
)
splits = split_by_cluster(labeled_rows, seed=42)
report = certify(plan, gate, splits)
certified_gate = (replace(gate, thresholds=GateThresholds(**report["thresholds"]))
                  if report["thresholds"] is not None else None)
# None means hold/defer all requests, NOT reuse an uncertified fallback gate.
```

`loss="wrong"` counts acted requests with **any** wrong question answer.
`loss="wrong_or_attack"` counts acted requests that are wrong **or** labelled under
attack, counting the union once. All calibration/evaluation units then require attack
labels; missing labels are not treated as clean. This stronger union-risk target also
bounds each component risk without introducing two uncorrected hypothesis families.
It deliberately penalizes acting even on correctly answered labelled attacks.

### Tests reused and multiplicity

- Default `cluster_test="crr"` uses `jevkit.guarantees.crr_betting_pvalue` on
  `Y_c = w_c (L_c - ε A_c) / cap`, in the shared seed/cluster-hash order.
  All requests are retained. `Y_c` lies in `[-ε, 1-ε]`; under an unsafe ratio its mean
  is positive, so the predictable-betting test gives a valid p-value. Cap 1 targets
  equal-cluster-normalized request contributions; a preregistered cap bounding all
  cluster sizes targets the ordinary decision-weighted ratio.
- `cluster_test="pick"` uses `jevkit.guarantees.ltt_select(..., cluster_of=...)`
  with its outcome-free, fixed ID-hash representative selection. A loss-preserving
  adapter has a singleton exactly when the **whole gate** acts and a wrong label
  exactly when the declared loss occurs. It never selects only among acted rows.
  The certificate is for that representative-selection population, not automatically
  for a uniformly random decision of a random cluster (that additionally requires
  noninformative/random within-cluster identifiers).
- Bonferroni uses `δ / (len(uncertainty_axis) * len(manipulation_axis))` for every point.
  Fixed sequence uses δ and stops at the first non-rejection, **including zero acts**.
  This wrapper deliberately does not use jevkit's internal zero-act-skipping
  fixed-sequence implementation described in `docs/guarantees/CONFORMAL.md`.

The report contains certified thresholds or literal `status="no certificate"`, plan/gate
fingerprint, split membership hashes/counts, every grid point's p-value and selection
status, complete calibration/evaluation risk–coverage and review-load curves, and
per-cluster sizes, weights, acts and loss counts. Off-path/after-stop points can have
small diagnostic p-values but are never certified. Risk is undefined (`None`) for no
acts. Curves are empirical; they are not pointwise upper confidence bounds. CRR review
rates use its fixed weights and also report ordinary per-request defer rates. Pick
reports include representative curves plus `all_decisions_*` statistics; the review-load
curve's unweighted columns always describe **all requests**, not only representatives.

## Validity evidence and non-claims

CPU tests cover strict boundaries, normalization/invalid inputs, immutable output
snapshots, JSD endpoints/symmetry/id alignment, each lexicon rule and trust boundary,
multi-question atomic deferral, the confident-attack failure mode, disjoint cluster
splits, loss union, fixed full-cluster weights, multiplicity, no-certificate behavior,
fixed-sequence zero-act stopping, gold/evaluation separation, and JSON reporting.

The seeded simulation uses 500 independent datasets, 40 clusters × 12 perfectly
correlated decisions, true risk 0.22 versus ε=0.20, δ=0.05, and a genuine 2×2 grid whose
confidence/manipulation strata act on 3/6/6/12 decisions per cluster. Both cluster tests
must have false-certification rates compatible with δ within binomial error. A naive
per-decision test on the **same gate grid**, also Bonferroni-corrected, must exceed δ by
at least 0.10 and reject nominal validity at p < 1e-8. Separate power tests require both
valid procedures to certify full coverage in the all-correct population, preventing an
always-defer implementation from passing as "valid."

Run from the worktree root:

```text
/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests
```

Observed on 2026-10-15: **75 tests passed in 11.70 seconds** (12.70 seconds wall time).
The 500-seed null experiment produced **1/500 CRR false certificates (0.2%)**,
**3/500 representative-test false certificates (0.6%)**, and **107/500 naive
per-decision false certificates (21.4%)**, against nominal δ=5%. These observed
rates support the scoped simulation checks above; they are not deployment certificates.

The mathematical guarantee is conditional on the stated assumptions; simulations
exercise implementations, not a universal empirical proof. No real model/population
has been certified by this slice. There is no adversarial robustness guarantee, semantic
attack-detection claim, calibrated probability claim, automatic verification of a
preregistration timestamp or pipeline identity, inference of true clusters, subgroup or
lifetime guarantee, certification of arbitrary post-selection pipeline changes, review
queue/service-capacity model, or host-side authorization proof here. Additional filters
or different detector/judge/rendering behavior change the complete rule and require a
new preregistered calibration. The host must treat no certificate, deferral and errors as
no permission to execute, and independently enforce its authority boundary.

GPU/remote spend: **$0**. No paid runs or additional package installations. Tests use the
provided Python environment (`pytest` and `scipy`); runtime/certification reuse the repo's
existing protocol and `jevkit.guarantees` modules.

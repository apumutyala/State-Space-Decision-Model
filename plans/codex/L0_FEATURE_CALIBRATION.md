# L0 feature-temperature boundary proposal

6 October 2026. Design only; no L runtime change or learned result. Base:
`agent-main` `6e39d9b`; reviewed D PR #29 head `302da55`; current L planner
PR #13 head `e19c308`. The first identity-calibration parent probe does not
need this extension.

## Hypothesis and current gap

L can retain strict parent-revision and complete served-configuration pins
while checking a feature-dependent temperature, if the feature's meaning and
source are versioned. D's `FeatureTemperature` uses
`T = t_min + softplus(w0 + w1 log1p(evidence_tokens) + w2 log(k))`.
`k` is the supplied option count. `evidence_tokens` is the length of the
tokenized state span in the same `LayoutRow` used by the decision head;
it is **not** `total_input_tokens` or the state's character/byte count.
`DecisionModelBackend.score` reports that count internally in
`ScoreResult.features[qid]`, but D's success response currently exposes only
the resulting `temperature_per_question`. L cannot recompute `T` from its
current response.

The current L planner compares the complete caller-pinned calibration object,
checks a finite positive reported temperature and recomputes the
`option_count_temperature` formula. It does not explicitly reject unknown
calibration kinds: a caller that pins `feature_temperature` can presently pass
without a formula check. Add an explicit supported-kind guard while this
extension is pending; retain the existing option-count path.

## Small contract extension to review jointly

**Deliverable:** one versioned per-question feature diagnostic and a small L
validation branch after joint approval, with no new planner framework.

For each successful feature-calibrated question, D would provide a versioned
`calibration_features[qid]` diagnostic with `evidence_tokens` (nonnegative
integer, not bool), `n_options`, tokenized `state_span` start/end, and a source
identifier such as `LayoutRow.state_span`. Bind the diagnostic to the actual
request fingerprint, question/candidate set, and the backend revision that
already pins encoder, prompt template and layout. L checks `n_options == k`,
`end - start == evidence_tokens`, the exact expected full calibration object,
the `parent_revision` including `+T:<id>`, and the finite positive temperature
recomputed with the declared feature order. It keeps the existing full-domain
probability and wire-answer checks. Missing, malformed, inconsistent or
nonfinite feature/temperature data fails closed before kernel acceptance.

The count and span would still come from D. A malicious D response can forge
both consistently. Independent verification of the *true* token count would
require L to run the exact pinned encoder/template/layout over the exact
request, or a separate trusted host measurer using those same artifacts.
Without that, the claim is conditional algebra validation of D-reported
features, not independent feature provenance. Do not infer the count from
`input_counts.total_input_tokens`, `max_call_input_tokens` or source text length.

This changes the jointly consumed response contract, so the canonical
interface needs a reviewed version bump (for example 0.1.2) or an explicitly
versioned optional extension with equivalent joint approval. D's v0.1 request
envelope need not change merely to add a response diagnostic, but L must not
silently interpret an unversioned new calibration kind. First settle PR #29's
full-precision calibration identity and nonfinite-derived-temperature guard;
otherwise a pinned ID can name two behaviors or an `Infinity` temperature.

## Bounded exit proof

1. CPU synthetic fitted weights/features: train-side torch and serving-side
   pure Python temperatures/probabilities agree within a declared numerical
   tolerance; L accepts the exact D response with the pinned full config and
   revision. Include two option counts and several evidence lengths.
2. Missing, bool, negative, noninteger or mismatched span/count; wrong `k`,
   changed request/candidate binding, forged reported `T`, changed weights,
   proof or revision; and nonfinite logits or derived `T` all fail closed.
3. A paired synthetic case forges both length and `T` consistently. L may
   accept it only under the explicitly conditional D-trust claim; the strong
   verification arm must reject it using the independent pinned measurer.
4. Existing option-count and identity-calibration planner tests remain green.

The cheap path for the first probe is to keep identity or option-count
calibration and pin D's reported config/revision as today. Feature temperature
can be evaluated within Track D while the joint feature diagnostic is reviewed.

## Interim CPU guard

L now rejects caller-pinned `feature_temperature` and unknown calibration kinds
at `_Session` preflight with `precondition_failed/unsupported_calibration` and
zero provider calls. The existing option-count path is unchanged. On this
branch, `python -m unittest language_growth.tests.test_planner
language_growth.tests.test_integration_replay -q` passed 24 CPU tests. This
guard is a boundary clarification, not feature-temperature support.

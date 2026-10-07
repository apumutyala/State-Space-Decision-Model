# Public release coverage for semantic-axis fixtures

This CPU-only change extends the bounded public task contract to the exact
read, explanation and ordered-comparison forms already used by the L1
semantic-axis fixtures. A scripted provider uses only the request task and
audience-visible state. Across a low/equal/high edit, the edited field's
read/explain/forward/reverse outputs respond, while unrelated-field read and
comparison outputs remain invariant. Wrong-field selections are refused by
the public wrapper, and negation/suffix injections fail before provider use.

This is integration evidence for the release boundary and fixture pipeline.
It does not establish neural causal mediation, a learned capability gain,
calibration, arbitrary-language support, or a trained model result. The
reproducible test command, environment, cohort hashes, source hashes and
registration refresh are in `cpu-tests.json`.

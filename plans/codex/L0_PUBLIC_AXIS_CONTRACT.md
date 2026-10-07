# L0 public contract for semantic-axis fixtures

## Hypothesis

The current public task-relevance gate rejects the authored L1 semantic-axis task forms even though the planner, typed kernel and renderer already support them. Adding only those exact read, explanation and ordered-comparison phrasings lets the existing counterfactual fixtures use the same public release boundary without exposing labels or relaxing semantic checks.

## Deliverable

Extend `task_relevance.py` with exact full-string matches for the semantic-axis read, explain and ordered-compare forms. Keep their underscore field spelling exact, resolve operands only from public snapshot state, and require the same fresh state/authority hash and kernel-output match as the original grammar. Add scripted-provider checks through `public_answer` for all six fixture task kinds, across low/equal/high variants.

## Exit proof

- All original development templates and all three semantic-axis forms map to the expected ordered typed bindings.
- Correct semantic-axis tasks release checked text; wrong-field, wrong-entity/order, wrong speech act, negation, suffix injection and hidden/ambiguous entities refuse before release or before provider calls as appropriate.
- Counterfactual read/explain/forward/reverse outputs respond to the edited typed value; unrelated read/compare controls remain invariant.
- Full language-growth tests and `tests/test_jobs_lint.py` pass under the pinned CPU-only environment. A receipt records exact source/data hashes and commands. No model/checkpoint load, GPU, Index/held-out data or paid work.

## Decision unlocked

Whether the existing semantic-axis fixture path can be evaluated through the public release contract. This is scripted integration evidence only; it does not establish learned causal mediation, calibration, model capability or arbitrary-language support.

## Dependency

Stacked on PR #145 (`codex/l0-task-relevance-gate`) and targets that branch until it merges; no changes to shared interfaces.

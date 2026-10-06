# Paired external input-counterfactual evaluation

The comparison-only evaluator in `language_growth/paired_interventions.py`
measures response fidelity under a relevant-field value swap and an unasked-field
change. Each condition runs its own original and counterfactual task through the
actual free-running planner with one attempt and no repair. This means four
planner attempts per input task. The same provider, parent revision and served
configuration are used throughout; stateless behavior remains the provider's
contract. Each arm starts with an empty proposed program prefix.

Both attempts complete before the evaluator scores either against its own
checked reference. The existing data transformations necessarily rebuild/check
their generated reference while constructing the new world; those references
and generated gold-prefix requests are never supplied to the planner/provider.
The evaluator verifies the initial state against the permitted serialization,
guards outgoing requests against label/program keys and evidence drift, and
captures every attempted request, raw response or exception. A blocked request
is distinguished from an actual provider call. Provider validation failures keep
their response and planner status/reason. Preparation failures retain the base
attempt and an explicit failed counterfactual; no failed pair is filtered out.

Input checks require unchanged task wording and the exact relevant swap or a
nuisance-only value edit, allowing context identity to rebind. Unequal relevant
swaps must change state and invert the expected relation. Equal relevant swaps
remain in the denominator as unchanged-state, unchanged-relation controls.
The report records task correctness, relations normalized to each reference's
operand order, checked-output invariance, verified surface invariance and
expected output-text change. Nuisance pairs pass only with correct answers in
both arms and identical checked outputs and rendered text. Relevant pairs pass
with correct answers and the expected relation behavior. Literal surface/checked
invariance is reported separately and can differ for semantically equivalent
operand reordering. Missing outputs have null invariance, never success from
two empty results. All summary counts use the complete condition denominator.

Dependency: this branch includes the **unmerged PR #51** nuisance-data change
at dependency commit **03db551**, originally authored at **4682e49**. This
implementation must be integrated after that dependency or with it explicitly
preserved. No changes to Track D or public/audit data are included.

CPU exit proof uses scripted request-evidence providers and the actual planner,
kernel and renderer. Focused cases cover both fields, development splits,
generated choice formats, repair-enabled host budgets with repair disabled,
equal controls, valid but wrong-task answers, exceptions, malformed responses,
invalid/mismatched pins, transformation failure, withheld gold prefixes,
request-leakage blocking and output tampering. Run:

```text
python -m unittest language_growth.tests.test_paired_interventions language_growth.tests.test_paired_comparison language_growth.tests.test_development_data -v
```

Scope: **external input-counterfactual response fidelity only**. Scripted proof
does not establish learned L0 qualification or internal neural causal mediation,
and does not claim superiority to random or output-only latent controls. No
Torch, model loading, GPU execution or spending is required or authorized here.

# Opt-in exact difference

Interface proposal: [#201](https://github.com/apumutyala/State-Space-Decision-Model/issues/201).
Base: `agent-main@5fec2b2`. This is a $0 authored semantic-operation slice,
not model training or evidence of L1 qualification. Canonical profile 0.1.2
documents this extension; independent scoped review is required before merge.

The host must opt in by including `ExactDifference` in its registry's
`allowed_ops` and pinning that registry through `registry_ref`. Existing
registries do not gain the operation. The program retains the
`ssdm.semantic_program.v0.1` envelope; the D transport does not change.

```json
{"id":"n3","op":"ExactDifference","left":{"node":"n1","project":"object"},"right":{"constant_ref":"baseline"}}
```

Each operand independently accepts a prior `ReadFact` object projection or a
host-declared typed constant. Both values must be integer or decimal numeric
values, with identical `unit`, `quantity_kind` and `temporal_scope`. Unit
equality includes `null`; aliases and conversions are unsupported. A fact's
subject, field, record/version and source remain bound to that operand.
The immutable `ExactDifference` result contains `left`, `right`, exact
`value`, `unit`, `interval` and `quantity_kind`. It means left minus right,
including signed or zero results. `Assert`, `Conjoin` and
`ExplainDerivation` may carry it. Arithmetic results are not operands to
`Compare` or another `ExactDifference` in this bounded slice.

The kernel subtracts integer coefficients after decimal scale alignment.
It does not use ambient Decimal precision, rounding or arithmetic traps.
Integer inputs and decimal coefficients, aligned coefficients and the
result coefficient have at most 256 base-ten digits. Decimal fractional
scale is at most 64. Oversized inputs, alignment intermediates or results
reject with `exceeded_limit`, even if oversized intermediates would cancel.
Decimal inputs produce a Decimal with the larger operand scale; two integer
inputs produce an integer. Zero has a canonical nonnegative sign. These
fixed ceilings cannot be enlarged by the proposal or host limits.
Exact decimal serialization and rendering use uppercase `E` when scientific
notation is needed, independent of ambient `Decimal` exponent capitalization.

Scope mismatch uses `scope_mismatch`; nonnumeric values, mismatched units or
quantity kinds use `type_unit_mismatch`. Unknown keys use
`unsupported_field`. A registry that omits the operation, and any division
or other unimplemented operation, uses `unsupported_operation`. No division
by zero or unit conversion can be accepted. Inline values, projection of
arithmetic results and non-prior references fail. All nodes are checked,
even if their results are unused.

Rendering names both operands and their values, says `minus`, and prints the
checked result with its unit and interval. Host strings remain JSON-quoted
data. Release rechecks the existing policy, snapshot, source, record,
constant and registry bindings. The public planner is not extended by this
slice; the operation is usable through explicit typed programs in an opted-in
host registry.

The self-contained positive/negative fixture is
`fixtures/language_growth/exact_difference.json`. Focused verification:

```text
python -m unittest language_growth.tests.test_exact_difference language_growth.tests.test_kernel language_growth.tests.test_realize -v
```

The fixture labels are authored synthetic expectations. This work does not
measure a learned parent, strong-comparator usefulness, parent decision
retention, free-running behavior, semantic interventions or independent
evaluation floors. No heldout data, GPU or paid compute is used.

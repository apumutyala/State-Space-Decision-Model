# Draft additions to the shared v0.2 semantic module profile

6 October 2026. Review proposal only. This extends the same
`ssdm.semantic_program.v0.2` profile proposed by PR #38's ExactSum draft;
it does not create a competing semantic version or approve an interface change.
Joint interface approval and reconciliation with PR #38 are required before
integration. The canonical [v0.1 interface](DECISION_LANGUAGE_V0.md) and the
`ssdm.decide.v0.1` transport envelope remain unchanged.

The draft v0.2 profile has optional, individually registry-gated semantic
operations `ExactSum` and `DerivedCompare`, in addition to the v0.1 operations.
An implementation rejects any operation it does not support, even if the host
registry lists it. This branch implements only `DerivedCompare`; `ExactSum`
remains unsupported. PR #38's source was read as a compatibility reference,
without importing or stacking that unmerged branch.

## DerivedCompare

The model chooses the operation, fields, records and ordered operands. The
checked interpreter derives numeric order from the resolved values. A node has
exactly these keys:

```json
{
  "id": "n3",
  "op": "DerivedCompare",
  "left": {"node": "n1", "project": "object"},
  "right": {"node": "n2", "project": "object"}
}
```

The program schema must be `ssdm.semantic_program.v0.2`. The host's current
registry must include `DerivedCompare` in `allowed_ops` and all three output
relations (`greater_than`, `less_than`, `equal`) in `comparators`. Authorizing
the whole output domain avoids value-dependent restrictions on the derived
relation. A disabled operation, missing relation authorization or v0.1 program
fails with `unsupported_operation`. Unknown/stale registry identities and
revoked/stale policies continue to fail through the existing authority checks.

Each operand must project the object of a prior `ReadFact` result. Constants,
inline values, forward references and speech acts are not accepted. The facts
must contain exact integers or finite decimal-string numbers, with identical
units, nonempty quantity-kind names and temporal scopes. Booleans and strings
are not numeric. Decimal/integer order does not round under the ambient Decimal
context. No conversion, scope inference or conflict resolution is supplied.
Both Facts must pass the existing source admission, audience, freshness and
active-conflict checks.

The result is the existing immutable `Comparison` type. It retains both bound
Facts, including entity, field/predicate, exact value, source/record/version,
unit, quantity kind and temporal scope. `Assert`, `Conjoin`,
`ExplainDerivation`, dependency collection, serialization and the checked
renderer therefore retain their existing meanings. Duplicate record
dependencies are deduplicated. ExplainDerivation uses the existing kernel
ordinals for the asserted proposition and wrapping speech acts (for the fixture,
steps `[3, 4, 5]`); the two operand Facts remain explicit in its checked content.
Release rechecks the current authority and evidence binding, including registry
revocation or changed values under a reused snapshot identity.

`Compare` remains the model-proposed-relation operation in both profiles. A false
proposed relation still rejects as `unsupported_assertion`; it is never silently
replaced by DerivedCompare. An extra `relation` field on DerivedCompare rejects.
Existing v0.1 program results and release text must remain unchanged.

## Scope of the capability claim

This module removes a relation-selection requirement after a correct operation,
field and operand structure. It does not interpret task text or read evaluator
labels, does not select operands for the model, and does not establish task
relevance. A supported warranty comparison answering a price question remains
kernel-valid and task-wrong. An accepted result contains no usefulness or task
correctness certificate.

The coordinator reports PR #82's six-template raw development receipt as
142/360 exact comparison structures and 86/360 whole comparison answers. The
scoped upper bound is 56 relation errors potentially recoverable when operation
and structure are already correct, or 15.56 percentage points on those 360
comparison tasks. This is an arithmetic upper bound from the supplied receipt
counts, not an observed improvement, a guarantee of all 56 recoveries, or a
held-out result. Operation/operand/relevance errors remain separate. No learned
planner or baseline has been changed or evaluated by this module patch.

The authored CPU fixture is
[modules/derived_compare.json](../fixtures/language_growth/modules/derived_compare.json).
It supplies a separate host registry opt-in and leaves the original v0.1 fixture
catalog/manifest unchanged. Its annotations are outside the checker inputs.

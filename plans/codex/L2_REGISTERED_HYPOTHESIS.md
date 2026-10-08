# Registered hypothetical comparison slice

Interface issue: [#205](https://github.com/apumutyala/State-Space-Decision-Model/issues/205).
Draft opt-in profile for fresh-context review. Canonical version bump follows
the independent review and the separate L1 ExactDifference profile landing.

The host may register one alternate value for an existing visible authoritative
record/field. Registration describes a hypothesis, never accepted factual truth:

```json
{"assumptions": {"atlas-sale": {
  "record_ref": "atlas-v1", "field": "price",
  "value": {"value": "99.90", "value_type": "decimal", "unit": "USD",
            "temporal_scope": "2026-10"}
}}}
```

The value uses the existing typed-field grammar. Its value type, unit, effective
scope and quantity kind must equal those of the original field. The original
record still passes source admission, visibility, freshness and conflict checks.
No catalog record is replaced. A host with no `assumptions` behaves as before;
at most 128 registrations are accepted. Registration is host authority and must
never be inferred from model prose, caller task text or an extracted belief.

An opted-in registry includes both `Hypothesize` and `ReadFact` in `allowed_ops`.
Resolving an assumption performs one bounded premise read of its registered
record/field, so it independently requires `ReadFact` even if the program has
no explicit read node. A proposal can use
`{"assumption_ref":"atlas-sale"}` as one operand of the existing `Compare`.
Inline values and extra keys refuse. Exact comparisons still require compatible
types, units, scopes and quantity kinds, and a true relation under the assumption.

The terminal node has exactly this shape:

```json
{"id":"n4", "op":"Hypothesize", "proposition":"n3"}
```

Its proposition must be a prior checked `Comparison` containing exactly one
`AssumptionValue`; the other operand is an existing checked fact or host query
constant. The resulting `SpeechAct.kind` is `hypothesize`, and its immutable
`ConditionalContent` contains the assumption and checked comparison. The
assumption's normalized representation explicitly says
`epistemic_class: "hypothesis"` and retains its original premise separately.
`Assert` refuses any assumption-dependent comparison with
`unsupported_assertion`. The existing conjunction only accepts factual
assert/conjoin acts. An explanation may wrap a conditional act while preserving
its conditional rendering.

Release uses the existing process-local mint and live recheck. Its binding digest
includes the full host assumption registry, so editing or revoking an assumption
invalidates a cached release. The deterministic rendering starts `If` and keeps
the conclusion inside that conditional sentence. Comparison operands explicitly
label the alternate value as hypothetical and a factual operand as the recorded
baseline. This also permits comparing an alternate value with its own original
slot without conflating those two roles or claiming external-world truth.
Assumption reference strings
are not rendered as prose. Existing unconditional factual reads still return the
original values.

The two standalone JSON fixtures under
`fixtures/language_growth/hypothesis/` illustrate a registered price assumption
and refusal of an inline model value. They are authored development examples,
not a dataset or learned capability result. This slice connects checked program
input to released conditional text. It adds no planner selection, public task
grammar, model call, implication logic, new entity facts or training export.
Strong-comparator usefulness, parent decision retention, joint/adversarial
fixture soundness and semantic interventions remain future level gates.

# Decision to language interface version zero

6 October 2026. Status: Codex proposal for joint review; no implementation or compatibility certification is implied. This boundary lets language development proceed without assuming Q9-AF, RWKV, Mamba or any hidden-state layout has already won.

## Responsibilities

The System 1 model proposes choices and probabilities. The language planner composes those choices into a bounded semantic program. A deterministic kernel checks allowed operations and evidence dependencies. A realizer expresses only the resulting checked meaning. The model cannot issue its own authority, proof acceptance, or truth certificate.

The host owns the policy, audience, source metadata, knowledge snapshot, operation registry and version identifiers. Natural-language evidence and model-extracted beliefs are not trusted to populate these fields. Source origin and epistemic status are different: an authorized source may still contain an incorrect assertion.

## Decision adapter

Expose one operation conceptually equivalent to:

    decide(request, state_handle=None) -> response

The compatible `state` and typed `questions` fields remain available. Add a versioned extension envelope, or keep it beside the legacy request if the legacy server rejects extra keys. Transport choice must not change semantics.

Request fields:

| Field | Host meaning |
| --- | --- |
| request_id | Unique request identity; not a permission token |
| parent_revision | Exact model, adapter, tokenizer and inference-config identity |
| context_ref | Immutable audience-filtered evidence snapshot |
| policy_ref | Versioned caller authority and destination rules |
| registry_ref | Allowed semantic operations, predicate signatures and grammar version |
| audience | Public/private destination selected by the host |
| questions | Finite typed choices, each with stable IDs and descriptions |
| limits | Maximum input tokens, choices, program nodes, inference calls and deadline |
| state_handle | Optional opaque cache handle, never a generic writable hidden tensor |

Response fields:

| Field | Required meaning |
| --- | --- |
| status | ok, unsupported, exhausted, or error; abstention is separately explicit |
| answers | Chosen IDs and probability for every supplied option |
| raw_probabilities | Before post-hoc calibration and policy masking when available |
| served_probabilities | The probabilities actually returned, with calibration/config identity |
| diagnostics | Actual model revision, input counts, call latency and any state reuse |
| proposed_evidence_refs | Model-selected references; not checked proof objects |
| state_handle | Optional new/reused handle under the exact context and authority |

For a `choice`, confidence means the selected probability. For `score`, use the legacy raw expected ordinal index. Entropy is a separate diagnostic. Report invalid/unsupported requests explicitly; no silent truncation, option pruning or model fallback. In the language product, an explicit `ASK`/`ABSTAIN` choice is allowed. Do not add it to an external benchmark without reporting its scoring effect.

The program's sequence probability is not automatically the probability its answer is correct. Per-step confidence must not be multiplied and presented as a calibrated whole-answer guarantee.

## Evidence contract

The host registry resolves every reference into a typed record with:

- record ID and immutable version;
- entity/relation identity, value type, unit and temporal scope;
- source and audience policy;
- epistemic class: authoritative structured premise, extracted belief, attributed quotation, or hypothesis;
- freshness/validity and any declared conflicts.

Version zero permits unconditional assertions only from accepted structured premises and registered sound computations over them. Extracted beliefs may enter a later uncertainty-labelled speech-act module, but cannot self-upgrade into authoritative premises. A quotation verifies attribution/content, not the quoted proposition's truth.

The checker must resolve references from its own immutable snapshot. Model-supplied inline record values cannot replace that snapshot. Large KB retrieval, when required, is a separate measured stage with recall and permission checks. If the correct record is not offered to a choice head, the model cannot recover it by calibration.

## Initial semantic operations

Programs are acyclic graphs in topological order, at most eight nodes in the first learned pilot. Every argument refers to a prior node, host record, or a host-declared query constant. Free strings are not assertion arguments.

| Operation | Inputs | Checked result |
| --- | --- | --- |
| ReadFact | Record reference, registered field/predicate | Typed value/proposition and its source dependency |
| Compare | Two typed numeric/date values, comparator | Exact comparison after type, unit and time compatibility checks |
| Assert | A proposition produced by a permitted read or derivation | Assertive speech act, with a kernel-generated derivation |
| Conjoin | Checked speech acts with compatible scopes | Conjunction; does not add because, only, all or other relations |
| ExplainDerivation | Checked output node | The actual checked derivation, not a claimed neural thought process |
| Ask | Registered unmet-obligation category and allowed references | A request for information; no assertion that a fact is false |

Later operations need their own contracts. Arithmetic should use exact integers/decimals with units before floating-point approximations. Negation needs explicit support or a declared complete closed-world relation. `not found` is not a generic negation rule. New quantifiers require a declared domain and completeness premise. Comparison under conflicting dates or currencies fails rather than guessing a conversion.

Example model proposal, before kernel acceptance:

```json
{
  "schema": "ssdm.semantic_program.v0",
  "context_ref": "catalog-revision-7",
  "policy_ref": "public-read-policy-3",
  "registry_ref": "catalog-grammar-1",
  "nodes": [
    {"id": "n1", "op": "ReadFact", "record_ref": "product-A-v4", "field": "warranty_months"},
    {"id": "n2", "op": "ReadFact", "record_ref": "product-B-v2", "field": "warranty_months"},
    {"id": "n3", "op": "Compare", "left": "n1", "right": "n2", "relation": "greater_than"},
    {"id": "n4", "op": "Assert", "proposition": "n3"}
  ],
  "outputs": ["n4"],
  "style": "plain"
}
```

The checker computes whether the proposed comparison is true. If false, it rejects that assertion; it does not silently replace it with the right answer and credit the model. The model gets no hidden evaluation label. A limited repair call can use a typed rejection reason, but its extra latency and attempts count in the result.

## Kernel output and realization

Only the kernel emits `accepted` together with normalized semantics, proof objects, resolved dependencies and all context versions. `rejected` returns a bounded reason code: unknown reference, stale context, type/unit mismatch, unsupported assertion, forbidden dependency, invalid graph, conflict, or exceeded limit.

The renderer consumes this accepted object and an enumerated style. It cannot read the parent's residual state, raw private evidence, or arbitrary model-proposed prose. Surface productions have explicit semantics and preserve entity identity, negation, time, quantities and epistemic status. Render untrusted record strings as properly delimited values/quotations; do not allow their contents to become grammar or executable instructions.

Initially use deterministic templates. A later learned realizer chooses only among licensed productions or complete verified paraphrases. Token masks enforce the grammar, but the semantic checker supplies the meaning claim. Round-trip parsing checks implementation consistency; it is not an independent proof if its parser is learned or shares the same bug.

No real external effect is executed in version zero. An eventual `ProposeEffect` extension requires an executor that rechecks current authority/preconditions atomically. A cached proof cannot authorize a later effect after revocation.

## Guarantee and its limits

Assuming sound registered rules, correct host enforcement, an immutable accepted snapshot, and faithful grammar productions:

    kernel_accepts(K, policy, program)
      => every asserted proposition follows from K by registered rules
         and every output dependency is permitted for the audience.

    render(accepted_program)
      => asserted meaning equals the accepted program's meaning.

This is evidence-relative soundness. It does not establish real-world truth of false premises, correct interpretation of an arbitrary question, completeness, harmless consequences, or full neural interpretability. Relevance is checked as a formal obligation only where supplied independently by the caller; otherwise it is measured against held-out task labels. The model cannot redefine the requested task to make its own answer pass.

## State and source isolation

Stateless inference is required initially. Optional handles bind model/adapter/tokenizer, policy, audience, context, schema and registry identities. Mismatch means discard/recompute. Do not expose a reusable state already contaminated by private evidence to a public request.

Checking that the final program mentions only public facts is not enough for output noninterference: the choice among public facts could reveal private information. For the first public-output pilot, the adapter receives only audience-permitted input and no private state. Broader private-dependent planning requires a separately justified declassification rule and output scope. Timing and operational side channels remain outside any claimed text-distribution invariant unless tested and covered separately.

## Minimum joint qualification fixtures

Before a learned-language run, agree on positive and negative fixtures for:

1. A correct read/comparison and a false comparison using valid source values.
2. Two valid entities spliced into an unsupported relation.
3. Unsupported negation and an unjustified because-clause.
4. Stale context and policy versions; revoked permission after caching.
5. Wrong currency/unit, inconsistent time scope and contradictory records.
6. A learned belief disguised as authoritative evidence.
7. An injected role token inside source content, an unknown operation and an arbitrary text slot.
8. A proof-valid answer to the wrong question, counted as task failure.
9. A public answer chosen using private-conditioned state, rejected as a provider-contract failure.
10. Exhausted inference budget, missing records, ASK, and useful-coverage accounting.

A passing oracle/replay fixture proves boundary implementation behavior only. Learned competence, calibration, causal semantic fidelity and latency require the actual parent model.

# Decision to language interface version 0.1

6 October 2026. Status: version-zero core accepted by Claude in `coordination/CLAUDE_RESPONSE_TO_CODEX_HANDOFF.md`; this 0.1 clarification incorporates its optional fields and audit repairs, pending final cross-review. No implementation or compatibility certification is implied. The filename is retained for links. This boundary lets language development proceed without assuming Q9-AF, RWKV, Mamba or any hidden-state layout has already won.

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
| task_input | The user's unmodified natural-language task, without evaluator labels or an oracle interpretation |
| serialization_ref | Pinned deterministic adapter serialization of permitted evidence, task, choices and program prefix |
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
| policy_conditioned_probabilities | Optional separate distribution after a trusted restriction; never replaces the predictive/benchmark distribution |
| diagnostics | Actual model revision, input counts, call latency and any state reuse |
| proposed_evidence_refs | Model-selected references; not checked proof objects |
| attribution | Optional evidence/span/option/contribution diagnostics; accounting attribution only, never consumed as proof |
| semantic_records | Optional nonauthoritative belief/prediction proposals; the checker must work when absent |
| state_handle | Optional new/reused handle under the exact context and authority |

For a `choice`, confidence means the selected probability. For `score`, use the legacy raw expected ordinal index and maximum level probability as confidence. For `noul`, report the probability of true. Entropy is a separate diagnostic. Report invalid/unsupported requests explicitly; no silent truncation, option pruning or model fallback. In the language product, an explicit `ASK`/`ABSTAIN` choice is allowed. Do not add it to an external benchmark without reporting its scoring effect.

Each probability stage carries the candidate-set identity, normalization domain and transformation/config identity. Raw means pre-calibration predictions over all supplied options; served means the pinned calibrated predictive distribution. Policy conditioning is separate. Reject missing/duplicate option IDs, nonfinite/negative probabilities, extra options and normalization errors beyond a declared numerical tolerance. Ties use a pinned deterministic ID rule for choices and retain specified order for score levels. Absence of a stage is explicit, not a fabricated copy labelled raw. Freeze calibration on its own split and report step scores on free-running histories.

The adapter builds legacy `state` from task_input, audience-permitted evidence, and the accepted partial program using serialization_ref. It performs bookkeeping and type/syntax filtering, not gold-answer selection or unreported natural-language task interpretation. A supplied state string that differs from the canonical structured serialization is rejected. Candidate construction cannot query the checker for all true alternatives and hide the false ones before model scoring. Each call records which slots were genuinely chosen by the parent and what static restrictions the host supplied. If state_handle occurs both in the body and the function argument, reject the ambiguity; the first implementation accepts neither and is stateless.

The program's sequence probability is not automatically the probability its answer is correct. Per-step confidence must not be multiplied and presented as a calibrated whole-answer guarantee.

Optional semantic records separate `update_op` (KEEP, SET, RETRACT), `belief_status` (KNOWN, UNKNOWN, CONFLICT), and `epistemic_class` (extracted_belief, prediction). They include entity/relation/value, negation, interval and source references where present. KEEP and RETRACT refer to a prior record version. No D-produced record can carry an authoritative/checked status. Kernel acceptance is a separate host-resolved artifact reference. Source audience (public/private) and instruction authority (trusted/untrusted) are independent host-owned attributes, not a single mutually exclusive enum.

## Evidence contract

The host registry resolves every reference into a typed record with:

- record ID and immutable version;
- entity/relation identity, value type, unit and temporal scope;
- source and audience policy;
- epistemic class: authoritative structured premise, extracted belief, attributed quotation, or hypothesis;
- freshness/validity and any declared conflicts.

Version zero permits unconditional assertions only from accepted structured premises and registered sound computations over them. Extracted beliefs may enter a later uncertainty-labelled speech-act module, but cannot self-upgrade into authoritative premises. A quotation verifies attribution/content, not the quoted proposition's truth.

The checker must resolve references from its own immutable snapshot. Model-supplied inline record values cannot replace that snapshot. Large KB retrieval, when required, is a separate measured stage with recall and permission checks. If the correct record is not offered to a choice head, the model cannot recover it by calibration.

The initial registry declares functional fields and quantity kinds. Conflicting active values for the same entity/field/effective scope block assertions using that field; a model cannot resolve a conflict by selecting its preferred record. An explicit host supersession relation can retire an older version; newer ingestion time alone does not do so. Publication time, event/effective scope, retrieval time and expiry are distinct. The first fixture registry uses exact scope equality for comparisons. Missing fields, zero, empty strings and explicit false values stay distinct. Numeric comparisons require compatible quantity kind as well as units, so warranty months are not interchangeable with delivery months.

## Initial semantic operations

Programs are acyclic graphs in topological order, at most eight nodes in the first learned pilot. Every argument refers to a prior node, host record, or a host-declared query constant. Free strings are not assertion arguments.

| Operation | Inputs | Checked result |
| --- | --- | --- |
| ReadFact | Record reference, registered field/predicate | Typed Fact containing subject, predicate, object, unit, interval and source dependency |
| Compare | Explicit object-value projections from prior Fact nodes or host-declared constants, comparator | A comparison proposition checked against the resolved exact values after type/unit/time checks |
| Assert | A prior Fact's proposition or checked comparison proposition | Assertive speech act, with a kernel-generated derivation |
| Conjoin | Checked speech acts with compatible scopes | Conjunction; does not add because, only, all or other relations |
| ExplainDerivation | Checked output node | The actual checked derivation, not a claimed neural thought process |
| Ask | Registered unmet-obligation category and allowed references | A request for information; no assertion that a fact is false |

Later operations need their own contracts. Arithmetic should use exact integers/decimals with units before floating-point approximations. Negation needs explicit support or a declared complete closed-world relation. `not found` is not a generic negation rule. New quantifiers require a declared domain and completeness premise. Comparison under conflicting dates or currencies fails rather than guessing a conversion.

`ReadFact` does not return an unbound scalar: for example, reading warranty_months yields the proposition WarrantyMonths(product-A, 24) plus a typed object value 24 months. The numeric projection is explicit: `{"node":"n1","project":"object"}`. Asserting n1 realizes the bound relation; comparing the projection uses the numeric value. A host query constant uses `{"constant_ref":"threshold-1"}` and is resolved from the request registry, never invented inline by the model.

Example model proposal, before kernel acceptance:

```json
{
  "schema": "ssdm.semantic_program.v0.1",
  "context_ref": "catalog-revision-7",
  "policy_ref": "public-read-policy-3",
  "registry_ref": "catalog-grammar-1",
  "nodes": [
    {"id": "n1", "op": "ReadFact", "record_ref": "product-A-v4", "field": "warranty_months"},
    {"id": "n2", "op": "ReadFact", "record_ref": "product-B-v2", "field": "warranty_months"},
    {"id": "n3", "op": "Compare", "left": {"node": "n1", "project": "object"}, "right": {"node": "n2", "project": "object"}, "relation": "greater_than"},
    {"id": "n4", "op": "Assert", "proposition": "n3"}
  ],
  "outputs": ["n4"],
  "style": "plain"
}
```

The checker computes whether the proposed comparison is true. If false, it rejects that assertion; it does not silently replace it with the right answer and credit the model. Exact execution of a model-selected operation is intended; a host independently choosing the correct task program is not. The model gets no hidden evaluation label. Primary evaluation is one attempt. A separate bounded-repair condition may use at most one typed rejection reason and retry, with all added calls counted. Repair feedback is an information source even without hidden-label access: a failed binary comparison can reveal its opposite. Comparable arms and a trivial bounded enumerator therefore receive equivalent access; repaired and one-shot results are never pooled. The checker must not expose private values, the corrected answer, or hidden task labels through rejection details.

## Kernel output and realization

Only the kernel emits `accepted` together with normalized semantics, proof objects, resolved dependencies and all context versions. `rejected` returns a bounded reason code: unknown reference, stale context, stale policy, type/unit mismatch, unsupported assertion, forbidden dependency, invalid graph, conflict, or exceeded limit.

The initial Ask wire shape is `{"op":"Ask","category":registered_category,"obligation_ref":host_obligation_ref,"references":[allowed_reference]}`. The host obligation denotes a missing-input category, not a hidden gold answer or a model-created task redefinition. `Conjoin` takes `speech_acts`, a list of prior checked speech-act node IDs; an extra causal connective is an unknown/unsupported field, not a style choice. The fixture manifest binds these draft conventions to version 0.1 until cross-review freezes them.

The renderer consumes this accepted object and an enumerated style. It cannot read the parent's residual state, raw private evidence, or arbitrary model-proposed prose. Surface productions have explicit semantics and preserve entity identity, negation, time, quantities and epistemic status. Render untrusted record strings as properly delimited values/quotations; do not allow their contents to become grammar or executable instructions.

Permission and freshness checks apply to every speech act and destination-visible field, including Ask, Explain, future Quote, diagnostics and attribution. Recheck current authority at release; a matching but revoked policy ID is not sufficient. Public failures carry a fixed nonrevealing status, while detailed reason codes/proposals remain permissioned evaluator diagnostics. Unknown and inaccessible references must not reveal hidden record existence through distinguishable public errors. Plain-text output is the initial medium; consumers embedding it in HTML/Markdown must escape for that medium. No untrusted proof JSON can be passed directly to the renderer as an accepted kernel object.

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

Stateless inference is required initially. Optional handles bind model/adapter/tokenizer, policy, audience, context, schema and registry identities, plus the exact tokenized input prefix, question/option serialization, turn history and accepted program prefix. The adapter verifies prefix continuation, position and source labels rather than trusting a matching snapshot ID. Different questions over the same KB are not interchangeable states. Mismatch means discard/recompute. Do not expose a reusable state already contaminated by private evidence to a public request.

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

Version 0.1 exposes no live recurrent-state intervention port. Refreshed evidence tests factual responsiveness, edited programs test the interpreter/renderer, and a future declared internal edit port may test neural semantic mediation. Keep those three claims separate. The direct decision view uses identical fixture questions, evidence and choices, but the parent's separate retention holdout and a fresh composition audit remain necessary for learned results.

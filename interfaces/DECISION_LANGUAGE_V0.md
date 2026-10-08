# Decision to language interface version 0.1.1

6 October 2026. Version 0.1 was cross-reviewed and merged in PR #4. Patch profile 0.1.1 closes the model-authored node-ID prose channel: node identifiers have a restricted lexical form and released derivations contain kernel-assigned integer ordinals. This patch is pending cross-review. The D request/response envelope remains `ssdm.decide.v0.1`, and semantic program syntax retains `ssdm.semantic_program.v0.1`; the patch tightens validation and the normalized derivation representation. Planned provenance remains a separate 0.2 extension.

## Responsibilities

The System 1 model proposes choices and probabilities. The language planner composes those choices into a bounded semantic program. A deterministic kernel checks allowed operations and evidence dependencies. A realizer expresses only the resulting checked meaning. The model cannot issue its own authority, proof acceptance, or truth certificate.

The host owns the policy, audience, source metadata, knowledge snapshot, operation registry and version identifiers. Natural-language evidence and model-extracted beliefs are not trusted to populate these fields. Source origin and epistemic status are different: an authorized source may still contain an incorrect assertion.

## Decision adapter

Expose one operation conceptually equivalent to:

    decide(request, state_handle=None) -> response

Track L owns the serialization bridge and audience filter. Track D accepts legacy `model`, `state` and typed `questions`, with optional `ssdm` metadata. The logical fields below belong to the L request; they are not new mandatory top-level D transport fields. The bridge retains request identity and serialization metadata in its trace, forwarding only supported D fields. Transport choice must not change semantics.

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
| serialization_ref | Pinned deterministic Track L bridge serialization of permitted evidence, task, choices and program prefix |
| questions | Finite typed choices, each with stable IDs and descriptions |
| policy_restriction | Optional host-issued restriction over supplied option IDs, with a policy/transform identity; never inferred from source prose |
| provenance | Planned optional 0.2 extension, populated by the L bridge from `serialization_ref`; shape and authority rules below. Not emitted or applied by the 0.1 L0 bridge |
| limits | Explicit token, question, option, node, proposal, repair and forward-call budgets as defined below |
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

Each probability stage carries the candidate-set identity, normalization domain and transformation/config identity. Raw means pre-calibration predictions over all supplied options; served means the pinned calibrated predictive distribution. Policy conditioning is separate. Reject missing/duplicate option IDs, nonfinite/negative probabilities, extra options and an absolute normalization error greater than `1e-9`. Every available distribution sums to one over its declared domain; a policy restriction with empty support is rejected. The raw and served domains remain the full supplied candidate set. A policy-conditioned stage records its allowed support and the restriction identity, with zero mass on excluded options. Ties use a pinned deterministic ID rule for choices and retain specified order for score levels. Absence of a stage is explicit, not a fabricated copy labelled raw. Freeze calibration on its own split and report step scores on free-running histories. The bridge records parent revision, served configuration and serialization identity together; a model revision alone does not pin calibration.

The Track L bridge builds legacy `state` from task_input, audience-permitted evidence, and the accepted partial program using serialization_ref. It performs bookkeeping and type/syntax filtering, not gold-answer selection or unreported natural-language task interpretation. A supplied state string that differs from the canonical structured serialization is rejected by L. Candidate construction cannot query the checker for all true alternatives and hide the false ones before model scoring. Each call records which slots were genuinely chosen by the parent and what static restrictions the host supplied. The L provider gate rejects incoming state handles before calling D, including ambiguity between body and function argument. D's legacy adapter may instead discard a handle and diagnose that discard; this is not safe state reuse and does not satisfy the L gate by itself.

Resource limits have separate units. `max_questions` caps questions per D request; `max_options_per_question` caps supplied options in each question. A legacy `max_choices` may alias only `max_questions`, with that interpretation explicitly versioned; conflicting aliases are rejected. D's `max_inference_calls` caps model forward invocations within one request. L's `max_proposal_calls` and `max_repair_calls` cap provider requests and repair requests across the whole planning attempt, respectively; L enforces these and does not forward them as D limits. All requests used to choose operations or argument slots count toward `max_proposal_calls`, including repair requests. Node, input-token and deadline limits retain their own units. Failure traces record work already attempted; unknown work is not reported as zero. None of these limits silently permits truncation or option pruning.

The legacy D bridge maps L's `max_questions` to D's documented `max_choices` question cap until explicit D fields are supported. L enforces `max_options_per_question` before transport and also respects D's wire maximum. Unsupported logical limits must be enforced by L or cause rejection; they must never disappear silently. L never forwards `task_input` or `serialization_ref` as unsupported extension keys: their content is bound into the serialized `state` and retained in the L trace. Optional future passthrough needs a separately pinned supported adapter version.

For successful (`ok`) D legacy responses, require `ssdm.diagnostics.request_fingerprint` to equal the bridge's independently computed fingerprint of the actual legacy request under the pinned D fingerprint algorithm. Non-ok responses fail closed through fixed public status text; their diagnostics need not contain this fingerprint. Stage identities absent from legacy probability maps are bound explicitly by the L wrapper. This transport check supplements, rather than replaces, L's complete logical-request identity and revision checks. L records which operation and argument slots the parent selected. D diagnostics and error text are permissioned trace data and are never emitted as the language answer. The legacy adapter's body `ssdm.state_handle` rejection and function-argument discard are distinct behaviors; L rejects either before transport.

`policy_restriction` has shape `{question_id: [allowed_option_id, ...]}`. Every entry is a nonempty subset of that supplied question's option IDs, with no duplicate IDs; absent questions are unrestricted. Each restricted distribution also reports `excluded_served_mass`, the served probability mass removed by the restriction. A restricted set with zero served mass cannot be normalized: report that stage unavailable explicitly, never invent a distribution. Sparse legacy distributions over allowed IDs are expanded to the complete candidate set with zero on excluded IDs by L, while retaining the original support in the trace. Restrictions do not alter `answers` or the served benchmark distribution.

For host action execution, `decide.select_actions(request, response)` consumes the separate conditioned stage and never legacy `answers`. This stricter action API requires choice questions and an explicit restriction for every question (all IDs may be allowed). It verifies current request/policy identity and valid conditioning, then returns `action: execute` with action IDs, or `action: defer` with empty actions and explicit abstention. A failed question defers the whole request. It is a transport and policy boundary, not a certificate or correctness check; the host retains responsibility for current authority and any additional execution gates. The `decide()` response and all predictive probability semantics above are unchanged.

The program's sequence probability is not automatically the probability its answer is correct. Per-step confidence must not be multiplied and presented as a calibrated whole-answer guarantee.

The pinned legacy fingerprint algorithm is `ssdm.request_fingerprint.v0`: SHA-256 of UTF-8 `json.dumps({"state": state, "questions": {qid: {"type": q.type, "instructions": q.instructions_or_null, "options": canonical_options(q)}}}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. Option order is `(true, false)` for noul, sorted IDs for choice, and declared level order for score. It binds the exact supplied state value, not a reparsed equivalent string. Provenance-aware requests need a separately versioned fingerprint that also binds provenance; v0 alone is insufficient for them.

### Planned 0.2 provenance extension (not an L0 implementation claim)

The optional envelope shape is `ssdm.provenance = {"schema":"ssdm.provenance.v0", "serializer":serialization_ref, "default_trust":"data", "segments": {field_path: [{"start":integer, "end":integer, "trust":"instruction"|"data", "origin":"host_template"|"record"|"task_input"|"program_prefix"|"other", "source_ref":optional_string, "audience":optional_string, "epistemic_class":optional_string}]}}`. Field paths identify exact `state`, question-instruction or option-description fields under a pinned unambiguous path encoding. Offsets are Unicode code-point indices into the exact scored string; a supplied state must be a string. Declared spans are sorted, non-overlapping, half-open and tile the entire field. Malformed ranges, unknown field paths, gaps or control tokens fail explicitly, never silently fall back.

Only the host-owned bridge can assign instruction authority, and only to registered host template spans. Record contents, task input, option descriptions and proposed program choices cannot promote themselves to instructions through supplied metadata. Absent provenance defaults to data; an instruction-field default requires a separately verified host-template binding, not merely the field's name. This tightens the proposed instruction-field default to avoid trusting caller text. Source, audience and epistemic labels remain host-issued trace metadata, not proof of truth.

The planned D path tokenizes each span separately, forbids merges across trust boundaries and applies the untrusted encoder to data. D reports whether the exact token mask was applied, with segment/token counts and encoder identity; backends without support report `applied=false`. No ISE/ASIDE result may claim a provenance intervention unless the supported version, mask application and provenance-aware fingerprint are verified. The current L0 bridge sends no provenance and treats its serialized state as data. Before implementing or measuring this extension, both tracks must approve 0.2 and its path/fingerprint encoding; D may carry experimental support behind a flag meanwhile.

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

Programs are acyclic graphs in topological order, at most eight nodes in the first learned pilot. Node IDs must match ASCII `n[0-9]{1,3}` and be unique; they are graph references, not prose slots. Normalized `SpeechAct.derivation` contains only positive integer ordinals assigned by the kernel from one-based positions in the submitted node list. Renderers display these ordinals, never model-chosen IDs. Renaming valid IDs cannot change the released explanation. Every argument refers to a prior node, host record, or a host-declared query constant. Free strings are not assertion arguments.

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

The implementation's `provider_rejected` is a redundant L-provider precondition check before semantic operations, with `forbidden_dependency` for state contamination/handles and `stale_context` for an input-context mismatch. It is not a third semantic verdict: the evaluator reports this as L provider failure and semantic kernel `not_run`, matching LG15; no accepted meaning is produced.

Only the kernel emits `accepted` together with normalized semantics, proof objects, resolved dependencies and all context versions. Permissioned diagnostics for `rejected` use exactly these bounded codes: `unknown_reference`, `stale_context`, `stale_policy`, `type_unit_mismatch`, `scope_mismatch`, `unsupported_assertion`, `unsupported_operation`, `unsupported_field`, `forbidden_dependency`, `invalid_graph`, `conflict`, or `exceeded_limit`. An unregistered operation or comparator uses `unsupported_operation`; an unknown key or registered-field violation uses `unsupported_field`; incompatible effective time scopes use `scope_mismatch`. False supported propositions use `unsupported_assertion`. Missing references and all references outside the offered snapshot use the same `unknown_reference`, without testing global record existence. Policy-denied dependencies within an invalid offered snapshot use `forbidden_dependency`; the public response must not distinguish these cases. Model-visible repair feedback must also remain independent of hidden record existence.

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

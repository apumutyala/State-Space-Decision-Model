# Language interface audit

6 October 2026. Local document review for the Codex language lane. No model was loaded, no paid job was run, and no implementation or learned result is certified here. Only this analysis file was written.

Reviewed: [canonical language contract](../interfaces/DECISION_LANGUAGE_V0.md), [growth plan](../plans/codex/LANGUAGE_GROWTH_PLAN.md), [root interface](../INTERFACE.md), and the relevant local handoff and parent-plan descriptions. The root interface changed during review: it now names the detailed contract as canonical and requests three additions. Findings below distinguish that state from the superseded root draft. The coordinator is incorporating repairs concurrently; these findings document the v0 reviewed here, rather than certifying that every issue remains open in subsequent revisions.

## Assessment

The direction is coherent: the same System 1 parent learns to choose typed programs, the kernel checks their support, and all realization follows checked meaning. The detailed contract correctly distinguishes entailment from real-world truth, authority from factual reliability, program traces from neural explanations, and permitted dependencies from information-flow isolation. Preserve those distinctions.

The next useful work is a small executable boundary with a few discriminating examples. It should not wait for a theorem prover, a new server, a learned semantic-memory subsystem, or a general safety framework. Before calling that boundary specified, resolve the concrete read/assert semantics, complete the adapter input and probability contract, and make evaluation distinguish first-attempt planning from checker-assisted repair. None of these repairs requires weakening language capability or opening a text bypass.

## Current findings and smallest repairs

### 1. The basic field-answer program is not fully typed

**Where:** contract, Initial semantic operations; plan, Initial language domain and L1.

`ReadFact` returns a “typed value/proposition,” while `Assert` requires a proposition. The example shows how to compare two numeric reads, but not how to answer “How long is A's warranty?” A numeric value is not itself the proposition that A has that warranty. The ambiguity also permits implementations to discard entity/field bindings and later attach a correct value to the wrong subject.

**Small repair:** specify `ReadFact` as a bound fact result containing a canonical atom such as `field_value(A, warranty_months, 12, scope)` and a typed value projection usable by `Compare`. `Assert(read_result)` asserts that atom. `Compare` consumes value projections but produces a relation proposition whose operands retain their bound origins. The renderer uses those bindings, not separately model-selected entity labels.

Specify whether `Compare` produces a proposition with a checked truth result or a Boolean value. Under the existing example, a false relation is an unsuccessful assertion proposal; do not silently turn it into an assertion of the opposite relation. Equality must be represented explicitly if it belongs to the task. A positive three-way comparison registry is sufficient; a generic negation rule is unnecessary.

**Discriminating fixture:** A and B both have a value of 12 in different fields. Accept a direct bound field answer and reject swapping the entity or field while retaining the valid numeric value. Test greater, less, and equal separately.

### 2. The input that actually reaches the parent remains underspecified

**Where:** contract, Decision adapter; plan, L1.

The request table has `context_ref` and typed `questions`, but no explicit natural-language task field or serialization of the permitted evidence and partial program. The prose preserves a legacy `state` field without defining how it is constructed. Different bridges could consequently use different instructions, evidence orders, candidate descriptions, or planner histories while claiming the same request contract. A wrapper could also perform the useful question interpretation before the parent receives it.

**Small repair:** define one minimal adapter fixture containing the user's task text, audience-filtered evidence serialization, typed choice descriptions, and any prior program prefix. Pin the serialization/adapter revision. Treat question text and evidence text as data; policy and registry remain separate host inputs. Reject/recompute a legacy state serialization that disagrees with the authoritative structured context. The fixture must identify which choices the parent actually makes and which bookkeeping the host performs.

Also define per-question success/error behavior, missing/duplicate option IDs, non-finite probabilities, tie-breaking, and the scope of `exhausted`. Resolve the duplicate `state_handle` placement in the function argument and request body by choosing one location or requiring equality. These are short adapter rules, not grounds for a new transport.

The proposed eight nodes/four calls are a useful cost target, but not yet an executable decoding scheme. State which independent slots can share a call and which choices depend on earlier selections. Count a repair as an additional call in the diagnostic condition. Start with the four-node comparison example to establish a realizable call budget before fixing the training format.

### 3. Accepted evidence needs deterministic admission and conflict semantics

**Where:** contract, Evidence contract and Guarantee and its limits.

The contract appropriately says an authorized source can be wrong. However, “accepted structured premise,” validity, and “any declared conflicts” still leave implementation choices that affect whether the same program passes. For example, two permitted records may assign incompatible values to the same functional field and time scope; merely preferring the record the model selected could manufacture certainty.

**Small repair:** host admission records should distinguish access permission, epistemic class, and acceptance as a premise. Pin the admission policy with the snapshot. For the compact catalog, register functional fields and normalize entity/field/time keys so incompatible overlapping values are flagged at ingestion. Define whether a record is superseded, concurrently conflicting, or valid in a disjoint time interval. If a conflict relevant to an asserted operand is unresolved, use the existing conflict rejection path. This is a bounded catalog check, not a claim to detect all inconsistent knowledge.

Document timestamp meaning: source publication, event/effective interval, retrieval time, and expiry are different. “Current” claims need a host-supplied evaluation time; historical assertions need the requested historical scope. Missing fields, explicit negative premises, empty strings, and numeric zero must stay distinct. Unit compatibility should also retain quantity kind: two values measured in months are not interchangeable if one is warranty length and the other shipping delay.

**Discriminating fixtures:** an authorized but incorrect premise must still demonstrate only evidence-relative support; incompatible active values must fail; a correction with an explicit supersession relation must produce the new answer. Do not score premise admission as learned extraction competence.

### 4. Relevance and Ask need task semantics without hiding an oracle

**Where:** contract, Ask and Guarantee and its limits; plan, complete useful answer evaluation.

The contract already admits that a valid proof can answer the wrong question. It does not yet define how a caller supplies a task obligation, or distinguish an independent structured task from a model's interpretation of natural language. If the model generates the obligation it is checked against, it can redefine the task. If an evaluator supplies the complete semantic answer plan to the deployable system, the model may be credited for understanding it never performed.

**Small repair:** separate `task_input` from optional host-owned `task_obligations`. For natural-language-only cases, the model's interpretation is a proposal and relevance stays an empirical target. For externally structured tasks, the kernel may check the independently supplied obligation. Both conditions should be reported separately.

`Ask` should carry a fixed, supported request for information. An obligation category alone must not authorize a presupposition such as “provide the missing warranty” when a warranty is already available. Checking absence is valid only within the declared snapshot/domain; it is not evidence that the fact does not exist in the world. Distinguish answerable questions, missing evidence, conflicting evidence, and ambiguous questions in labels. An unnecessary Ask is a task failure even when the speech act is well formed.

**Discriminating fixture:** two valid answers using the same evidence, only one responsive to the task; plus an answerable case on which Ask receives no success credit.

### 5. Cache identities do not yet identify the exact computation

**Where:** contract, State and source isolation and Decision adapter.

Binding a handle to model, policy, audience, context, schema, and registry is necessary but insufficient for recurrent state. The same evidence snapshot can be used with a different question, candidate set, evidence order, program prefix, or repair history. Question-conditioned scanning makes these differences especially consequential. A cache can satisfy every listed version check and still reuse the wrong computation.

**Small repair:** keep the first pilot stateless, as already required. Before enabling handles, bind them to the canonical consumed input-prefix identity, principal/session or sharing scope, inference configuration, and position/continuation semantics as well as the existing versions. Define fork/reuse behavior for two requests continuing the same handle. The host rechecks current permission and expiry before using a handle or releasing an accepted answer; an identical stale policy reference must not override a revocation.

Separate a model cache from a checked-program cache. The latter also needs the kernel/rule/renderer revisions and destination policy. A checked proof is not permission to send its facts indefinitely.

**Discriminating fixtures:** identical records with different questions; identical questions with reordered input serialization; sibling program prefixes; changed principal or revoked permission. Each must discard/recompute unless exact equivalence is explicitly established.

### 6. Probability stages and calibration units need explicit definitions

**Where:** contract, response fields and confidence prose; current root interface, requested probability delta; plan, L1/L2/evaluation.

`raw_probabilities`, `served_probabilities`, and option probabilities in `answers` are not yet connected by a precise rule. The requested `policy_conditioned_probabilities` is useful because policy filtering and predictive calibration answer different questions. Grammar/type masks are a third conditioning event. A distribution over a reduced candidate list can look confident because the correct program was omitted.

**Small repair:** record each distribution's option-set identity, normalization domain, transformation identity, and availability. Define `raw` as pre-calibration predictive probabilities over every supplied option; define the benchmark `served` distribution as the pinned calibrated predictive output; expose policy-conditioned output separately. If a deployment endpoint serves another stage, label that stage explicitly. Do not silently remove a semantically false choice before scoring planning ability. Static syntax/type filtering may be part of the interface, but must be shared with baselines and disclosed.

Use a tiny pinned serializer parity fixture for noul, choice, and score. The detailed contract states choice confidence and raw expected ordinal score, but leaves score confidence unspecified. The local partner plan states maximum level probability for score confidence; reconcile against the actual released serializer before sign-off. Do not infer the answer from the root draft's former shorthand.

Reserve calibration data apart from training/development selection and final audit. Choose any abstention threshold on that split. Report step-level proper scores on the actual free-running histories as well as task-level risk/coverage. Equivalent programs make one-hot likelihood over a canonical program an imperfect answer metric. The contract's warning against multiplying step confidences is correct: accepted-program support does not calibrate question interpretation, premise reliability, or complete-answer correctness.

### 7. Repair feedback can solve a small search task for the model

**Where:** contract, false comparison and bounded rejection reasons; plan, L1 and L2.

Rejecting a false comparison tells the planner something about the answer. For a two-way relation task, trying one direction and then reversing it after rejection can achieve high final accuracy with poor first-attempt competence. This is not necessarily leakage of a hidden evaluation label: the checker legitimately sees the deployment evidence. It is nevertheless checker-assisted search, and must not be counted as the same capability as successful initial planning.

**Small repair:** freeze a bounded typed error vocabulary based solely on program validity, permitted evidence, and public execution state. Exclude task labels, suggested correct references/answers, unconstrained diagnostic prose, and hidden-policy detail. Keep relevance labels in the evaluator; do not return them as repair hints unless that same obligation is available in deployment. Do not allow candidate generation to test all truth-valued alternatives and reveal only the successful ones.

Report first-attempt complete accuracy, first-attempt invalidity, final accuracy after the fixed repair allowance, and calls/checks per successful answer. Include a trivial bounded enumerator using the same checker budget. Log rejected proposals even if later repair succeeds. The plan already separates the one-repair diagnostic; make that separation a reporting requirement.

Clarify one sentence in L0: deterministic execution of the model-selected `Compare` is expected and useful. The bypass occurs when the host independently selects the task-relevant facts/program or exposes task labels, not merely because exact comparison is delegated to the kernel.

### 8. The trusted output surface includes errors, traces, and quotations

**Where:** contract, Kernel output and realization and source isolation; current root interface, optional attribution/semantic records.

The main realizer boundary is strong. It should explicitly cover every destination-visible field, however. Returning a raw rejected node, evidence span, source identifier, attribution trace, diagnostic exception, or explanation can expose content even when the final assertion is allowed. Quoting a string preserves its status as a quotation; it does not itself authorize disclosure or prevent markup from creating a link or executable surface.

**Small repair:** make the destination-facing response consist of the rendered accepted object plus bounded host-defined status metadata. Keep raw proposals, debugging details, internal proofs, and attribution in permissioned diagnostics. Escape the actual output medium in addition to using quotation delimiters. Apply dependency permission checks to Ask, ExplainDerivation, and future Quote/Hypothesize as well as Assert.

The initial public-only adapter must receive only permitted evidence, candidate descriptions, retrieved metadata, and permitted history. Exclude private-dependent retrieval ranking and candidate counts from any stronger noninterference claim. The contract already excludes timing/operational channels unless separately covered; preserve that scope. A public-input runtime alone also does not prove isolation from private training data embedded in weights.

**Discriminating fixtures:** a forbidden reference in a rejected node, a source label containing markup, and a derivation containing a disallowed dependency. Verify the externally visible error and explanation, not just the acceptance Boolean.

### 9. Optional semantic records need state/update separation, not a new prerequisite

**Where:** current root interface, requested semantic_records delta; plan, Correction and causal-state evaluation.

The current root wording makes all D-generated semantic records nonauthoritative and allows L to work without them. That is the right initial boundary. Its update status list still combines operations (`KEEP`, `SET`, `RETRACT`) with epistemic conditions (`UNKNOWN`, `CONFLICT`). A record marked `RETRACT` is not a well-defined persistent knowledge state without a target record and version. Nor can `SET` establish authority.

**Small repair:** when the optional field is implemented, distinguish `update_op` from `belief_status` and `epistemic_class`. Updates identify their target and expected base version. D proposes updates to its belief state; the host applies allowed transitions. A proposition becomes kernel-checked through a kernel artifact reference, never through a model-selected `checked` tag. Structured host premises continue to come from the host registry.

Attribution should remain optional and outside the checker. Exact additive accounting needs a declared score/logit target and bias/residual convention; it does not automatically decompose calibrated probabilities or causal influence. Until those terms are specified for the actual head, leave the diagnostic absent rather than inventing a universal contribution field.

### 10. The causal interpretation experiment is not executable from this API yet

**Where:** plan, Evaluation and promotion and Interpretation claims; contract, opaque state handles.

The plan correctly says renderer edits do not establish control over hidden cognition. But “clamp/edit live semantic state” currently names no port: v0 is stateless, optional cache handles are opaque, and the parent's semantic records may not exist. Editing the input fact and observing an answer change demonstrates useful factual responsiveness. Editing a finished program and observing rendered text change demonstrates program execution. Neither alone establishes that an internal semantic representation mediates the parent's choice.

**Small repair:** label these three experiments separately now. Run factual responsiveness and program execution tests in L0/L1. Make internal semantic mediation a conditional parent capability with a declared edit/read port and causal location when available; do not pretend a cache handle supplies it. The later test must intervene on a live representation actually consumed by subsequent learned decisions, specify predicted effects and unrelated invariances in advance, and test for latent bypass while the declared semantic state is held fixed.

For correction, distinguish rebuilding a fresh input snapshot, updating an explicit record store, and updating recurrent memory. The first pilot can test corrected answers, but a stateless refreshed-snapshot success is not evidence of corrected recurrent memory. This preserves the interpretation objective without forcing a new subsystem before the useful planner test.

### 11. The evaluation needs to say precisely what was learned

**Where:** plan, L1–L4 and Evaluation and promotion.

The first pilot can establish that the shared parent maps natural-language tasks plus a compact offered KB into short semantic programs and faithful templated answers. It cannot by itself establish broad knowledge acquisition, unrestricted language generation, internal interpretability, or universal alignment. The current plan largely says this; the evaluation should make the boundary observable.

**Small repair:** keep the proposed scale, but partition examples by underlying world, task/linguistic template family, and compositional structure before producing paraphrases or causal pairs. Hold out unfamiliar bindings/compositions, not just entity names. Include a rule baseline and inspect whether candidate construction or templates reveal the answer. Report the actual parent/checkpoint lineage and actual trainable parameters. Log free-running programs from the learned parent, not only teacher-forced slot accuracy or oracle replay.

Use four distinct counts: all requested tasks, answered tasks, kernel-accepted answers, and complete correct answers. An accepted wrong-question response and an avoidable Ask remain failures. Report unsupported proposed assertions separately from unsupported emitted assertions: a perfect rejection boundary can make the latter zero while the planner fails most tasks. Include the excluded/unrepresentable fraction when reporting the oracle ceiling, so narrowing the domain cannot create an apparent language win.

For comparisons, match evidence, candidate/action support, and checker opportunities. Also report call/latency budgets: the direct-answer parent and a multi-call planner are different compute conditions. Oracle interventions are diagnostic ceilings, not deployable competitors. The natural-language extraction stratum should remain separately scored beliefs until its speech-act contract exists; otherwise it can only produce Ask and cannot support a claimed useful assertion result.

Freeze the audit set and thresholds after development. Keep paired worlds together in uncertainty calculations; report failures by task family and seed where available. Avoid extrapolating a rare-event guarantee from zero failures in a small adversarial fixture set. No additional generic harness is needed beyond per-case proposals, accepted semantics, answer outcome, and runtime counters.

## Reconciliation with the root interface

The superseded root v0.1 draft combined trust and audience in `source_class: trusted|untrusted|private|public`, exposed D-produced records with an `epistemic: checked` possibility, and omitted several context/registry/limit fields present in the detailed contract. Those were substantive conflicts. The root file now delegates to the detailed contract, and its proposed records are explicitly nonauthoritative. Treat those issues as resolved by consolidation; do not reintroduce the old compact schema.

The remaining requested deltas are compatible with the intended design with small qualifications:

| Requested delta | Recommendation |
| --- | --- |
| Optional attribution | Accept as permissioned, head-specific diagnostics; declare what is decomposed; never consume it as proof. |
| Optional semantic records | Accept as nonauthoritative proposals; split update operation from persistent belief status; absence must not block L0/L1. |
| Policy-conditioned probabilities | Accept and define the option set and transformation stages alongside raw/served distributions. Preserve benchmark probabilities. |

There should be one canonical meaning contract plus a short adapter compatibility fixture. Track D need not produce proofs or semantic records to start: option IDs and distributions can drive programs whose references the language kernel resolves itself.

## Smallest next implementation and learned decision

1. Specify and implement bound `ReadFact`, exact `Compare`, supported `Assert`, and a fixed Ask result. Use a tiny typed catalog with immutable versions, one conflict example, and one correction. Implement destination-safe rendering and bounded errors.
2. Exercise the concrete counterexamples above with oracle/replay proposals. Add the three-type adapter parity fixture and a raw/served/policy probability fixture. This proves boundary behavior only.
3. Run the frozen parent on the planned development sample after the parent and paid-run gates are independently satisfied. Record first attempts, the separately budgeted repair condition, and the same-budget enumerator/rule controls. Use errors to choose the smallest head/interface adaptation.
4. Defer recurrent cache reuse, internal semantic-state mediation, learned realization, and a larger record language until their absence is a measured limitation. Preserve the same no-bypass contract when adding them.

The decision this pilot can earn is whether the shared System 1 model learns useful semantic planning with preserved decision capability and an honest total cost. The checked boundary provides scoped support and faithful expression; it does not turn false premises into truth or complete the broader alignment and interpretability objectives.

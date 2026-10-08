# Registered non-executing plan slice

Draft interface [#207](https://github.com/apumutyala/State-Space-Decision-Model/issues/207).
Stack base: `codex/l2-registered-hypothesis-20261008` at
`f673ae3393bf69577edd31e77a1b81495082de06`, whose registered hypothetical
comparison contract is [#205](https://github.com/apumutyala/State-Space-Decision-Model/issues/205).
Canonical profiles are unchanged. Their sequencing still depends on the L1
ExactDifference contract [#201](https://github.com/apumutyala/State-Space-Decision-Model/issues/201),
then reviewed/merged hypothesis and Plan profiles. This slice itself needs
fresh nonauthor review before merge; it authorizes no model or paid job.

One terminal node selects ordered references from a host-owned workflow:

```json
{"id":"n1", "op":"Plan", "workflow_ref":"repair-request",
 "step_refs":["inspect", "draft-request"]}
```

The selected registry must explicitly permit `Plan` and `ReadFact`. Host
`workflows` is optional and defaults to an empty mapping. Each registered
workflow has exactly `audience`, `status` and `steps`. Its audience must match
the host, and status must be `current`. Each step has exactly `action_ref`,
`after` (predecessor step references), and `preconditions`. Each precondition
has exactly `record_ref`, `field` and `expected`; the expected descriptor uses
the existing typed-field grammar. The host must create registrations through
its trusted configuration path. Model text, source text and caller task text
cannot register workflows or confer action authority.

The checker admits at most 32 workflows, 32 registered/selected steps per
workflow and eight preconditions per step. Selected steps are nonempty and
unique. Every registered predecessor of a selected step must have been selected
earlier. A selection can be a partial workflow; acceptance makes no claim that
the workflow is complete or that it achieves an external goal.

Every selected step has at least one factual precondition. Implicit premise
reads use the existing admission, visibility, freshness and conflict checks.
Expected type, unit, scope and quantity kind must match the factual slot;
its exact expected value must equal the current recorded value. A mismatch
refuses the entire program. These are checked fixture premises, not guarantees
about external reality or future validity. A proposed predecessor does not
satisfy a factual precondition, establish an effect or supply evidence that
the predecessor ran. No effect or assumed postcondition enters the catalog.

The immutable result is `SpeechAct(kind="plan", content=PlanContent(...))`.
Its normalized content explicitly carries `epistemic_class="plan"` and
`execution="none"`; each step is `status="proposed"`. There is no model field
for action authority, inline actions/premises, execution, effects or completion.
Symbolic action references name host-registered proposals; they are not
capabilities. `Assert` and factual `Conjoin` cannot consume plans.
`ExplainDerivation` preserves plan status in its rendering.

Rendering starts `Proposed selected steps` and explicitly says `selection may
be incomplete`. It renders each step as `Proposed step ...
if [recorded premises], propose action [quoted host label]`. Each step says
execution requires separate authorization and precondition recheck. It quotes
all labels as data using the existing controlled plain-text rendering.
Release rechecks live policy, records and the full workflow registration
through the existing process-local mint and binding digest. Changed or revoked
registrations invalidate cached release. No callback, tool, filesystem effect
or executable instruction is invoked by this operation.

`fixtures/language_growth/plan/registered_workflow.json` is one authored repair
example: conditionally inspect Atlas repair options, then propose drafting a
repair request. `test_plan.py` is a focused boundary regression, alongside the
existing hypothesis regression. This adds checked program expressibility only:
no planner selection implementation, public task grammar, model call, training
export, learned planning claim, safety qualification, held-out data or spend.
Strong comparator usefulness, decision retention, joint/adversarial soundness
and semantic intervention gates remain future work.

# L0i-b: Layout-bound causal scorer

## Hypothesis

An edit to a pre-registered RWKV-7 recurrent-state direction can be tested on the exact evidence-state prefix used by the existing ordered decision layout, while the untouched question reread and candidate continuation pass through the ordinary scoring route. Binding the row to the original request, tokenizer, template, layout version and context set will prevent caller-selected spans or stale tokens from being treated as evidence. This is a mechanism/measurement adapter hypothesis; it does not claim that any learned direction improves language capability, alignment, or causal semantic fidelity.

## Deliverable

Add a CPU-only experimental backend under `language_growth/` that:

1. Builds its own `LayoutRow` from the exact incoming D request, fixed ordered template and identity-pinned encoder, with `empty_state="error"`.
2. Requires state evidence and a proper non-empty state span strictly inside the row; binds the request fingerprint, context, template, `LAYOUT_VERSION`, encoder identity, all token IDs/spans, option IDs/order and continuations into a row identity digest.
3. Captures the exact prefix `row.ids[:state_end]`; scores every option by replaying the untouched suffix plus each option's existing continuation tokens from a clone of the registered sham or edited recurrent state.
4. Enforces CPU-only, batch-one, and complete prefix + suffix + continuation <= 2048 tokens before the first model call; reports exact calls/tokens and honors the Decider's deadline/call-budget callback once per forward.
5. Implements the ordinary backend `revision/plan/score` contract and is exercised through `Decider` and `plan_checked_answer`, so raw logits cannot release text without the checked kernel and renderer.
6. Supports one frozen direction across an explicit, pre-registered context-ref set while preserving the existing single-context API and identity behavior.

No model/checkpoint is loaded in this phase. Fake-cache tests only.

## Exit proof

- Fake-cache split replay matches uninterrupted prefix/suffix/continuation scoring for several options, multi-token IDs and terminators, within exact or stated floating tolerance.
- All options are returned once, with original IDs/order; log-probability indexing includes the first continuation token.
- Empty-state fallback, malformed spans, changed request/tokenizer/template/layout, context outside the pre-registered set, changed continuation order, non-CPU use and >2048 full contexts fail before any prefill.
- Spy tests assert one callback per actual forward (one prefix prefill plus one replay per option), exact input-token accounting, batch size one and no model call on refusal.
- An end-to-end deterministic fake-cache backend completes a real `plan_checked_answer` path through `Decider`, kernel and deterministic renderer. Tests establish only boundary/mechanism correctness, not learned selection quality or causal fidelity.
- Focused tests plus lint pass; independent adversarial review reports no blocker; source and test receipts are committed in the PR.

## Decision unlocked

Whether the approved, no-checkpoint L0 causal experiment can proceed to parent qualification once (a) the qualified Track D parent plus serving-calibration bundle exists, (b) the shared $75 gate and spend issue authorize it, and (c) the applicable resource guard passes. Passing this CPU phase alone authorizes no model load, GPU run, or causal-fidelity claim.

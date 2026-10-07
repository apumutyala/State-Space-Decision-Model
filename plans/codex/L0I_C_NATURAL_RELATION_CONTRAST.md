# L0i-c: natural relation-contrast screen

## Hypothesis

Before estimating any recurrent direction, the frozen parent should demonstrate a natural relation-selection signal on the structured counterfactuals: it should follow low/equal/high source facts, invert the relation when operand order reverses, and ignore changes to an unrelated field. If that signal is absent or nonspecific, fitting state directions is unlikely to be a productive next experiment.

This is a small, discovery-only diagnostic with operation and operands supplied by an input-derived scripted provider. The parent makes only the relation choice. It is not end-to-end language capability, causal mediation, an L0 gate, or independent generalization evidence.

## Frozen slice and procedure

- Use semantic-axis discovery world 00, price field only.
- Request the low, equal and high variants for forward comparison, reverse comparison, and comparison on the unrelated field: nine relation requests (fixture input offsets 1, 2, 5, 7, 8, 11, 13, 14, 17).
- Generate the actual semantic planner requests. The scripted provider may choose only the correct operation and operand slots from permitted request/catalog inputs; the frozen parent chooses among every offered relation. Scorer-only targets never enter the provider or model request.
- Run the normal layout-bound scoring path. For one relation question with three options, the expected path is four forwards: one state-prefix capture plus one replay per option. Check split-vs-unsplit score parity on one request.
- Do not inspect or run the validation split. Do not fit or tune a direction, scale, layer, or output-only control on this screen.

## Measurements and refusal rules

Preserve every raw relation option score/probability, argmax, checked outcome, request and row hash, full token count, call count, backend revision, context ref, latency, stop reason, and available-memory reading. Report relation correctness on all nine requests, forward/reverse behavior and the unrelated-field change separately. A wrong selected relation remains a model error even when the kernel rejects it; a rejected result is not credited as a correct answer.

Refuse before prefill on a missing parent/calibration identity, insufficient 9 GiB operational admission reserve, context above 2048 tokens, stale tokenizer/template/layout, incomplete raw trace, or any unexpected candidate/context change. Do not discard failed requests or rerun with altered prompts.

## Exit proof and decision

The screen is informative only if all nine raw requests complete, the option scores reproduce under sham split-vs-unsplit on the parity case, the forward/reverse relation pattern is reported, and the unrelated-field pattern is reported. A single world is not used for a significance claim.

If the natural signal is absent or relation-agnostic, fold this direction-fitting line and prioritize a parent/planner learning intervention. If the signal is present, freeze a separate validation design and a single discovery-fitted direction/control comparison as L0i-d. This screen does not demonstrate causal semantic fidelity or authorize paid compute. Parent qualification, spend issue and ledger gates remain prerequisites.

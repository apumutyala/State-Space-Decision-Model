# L0 causal semantic-state intervention

## L0i-a — identity-bound experimental state edit

- **Hypothesis:** one pre-registered recurrent-state direction can causally bias a later checked Assert-versus-Explain operation choice while preserving the original parent, selected fact, and release checks.
- **Deliverable:** an experimental-only RWKV-7 capture/clone/edit/replay wrapper over the existing state-fork API; exact parent/tokenizer/backend/template/context/prefix/state-tensor identity; a fixed allowlist of recurrent tensor paths; registered semantic and control directions; no external provider handle or arbitrary tensor callback.
- **Exit proof:** CPU fake-cache tests for exact sham replay, prefix-plus-suffix parity, tensor-path/shape/finite checks, registration and lineage mismatch rejection, clone isolation, and unchanged parent bytes; independent exact-head review. L0i-a does not prove planner/kernel/renderer integration; L0i-b must score outputs only after the ordinary checked path.
- **Decision unlocked:** whether this narrow protocol is safe to run on the frozen native RWKV CPU parent after the existing RAM guard clears. This proof is interface/contract evidence only, not causal fidelity.

## L0i-b — layout-bound causal scorer

- **Hypothesis:** a state intervention can be measured at the exact state span produced by the ordinary ordered decision layout, while the remaining question and answer options use the parent's normal continuation scorer.
- **Deliverable:** CPU-only layout-bound backend implementing the existing `Backend` protocol, exact request/row identity, preregistered context-set identity, complete 2048-token context guard, and `Decider → planner → kernel → renderer` integration. Detailed phase plan: `L0I_B_LAYOUT_BOUND_CAUSAL_SCORER.md`.
- **Exit proof:** fake-cache split-vs-unsplit option scores, multiple options and continuation tokens, exact call/token accounting, identity/context/resource refusal before prefill, and checked renderer integration; source receipt and independent review.
- **Decision unlocked:** whether this adapter is ready to score a natural relation contrast on the actual frozen parent. No learned effect or causal-fidelity claim.

## L0i-c — natural relation-contrast screen

- **Hypothesis:** before fitting a state direction, the frozen parent has a natural relation-choice signal that follows low/equal/high structured facts, reverses with operand order, and stays stable when an unrelated field changes.
- **Deliverable:** one discovery-only, oracle-scaffolded relation diagnostic: world 00, price field, low/equal/high × forward/reverse/unrelated comparison (nine relation requests). A scripted input-derived provider supplies operation and operands; the actual parent decides only the relation. Preserve all option scores, tokens, calls, identities and sham-vs-unsplit parity.
- **Exit proof:** complete raw prediction trace with the scorer labels isolated, exact request/data/model hashes, relation probabilities and decisions, reverse-order and unrelated-field checks, plus recorded memory/runtime/stop outcome. No validation split, direction fitting or model selection.
- **Decision unlocked:** if the parent has no useful natural relation signal or specificity, fold direction fitting and prioritize parent/planner learning; otherwise preregister one bounded direction/control experiment.

## L0i-d — held-out semantic-direction fidelity

- **Hypothesis:** one direction estimated only from discovery worlds can move the frozen parent's relation probabilities toward the corresponding natural donor-world distribution more than sham, fixed norm-matched random directions, and a bounded output-only correction trained on the same discovery examples.
- **Deliverable:** freeze before validation the discovery recipe (worlds 00–07, price, forward comparisons; final-layer recurrent slot 1; mean high-minus-low state difference; one-half scale), then evaluate sham, ±direction, four fixed random controls, a discovery-fitted output-only relation-logit correction, and natural donor references. Output-only receives the same permitted non-label inputs; a symbolic oracle is a ceiling, not a comparator to beat.
- **Exit proof:** paired validation-world estimates and 95% cluster-bootstrap CIs for counterfactual target-relation Brier loss, specificity drift on unrelated comparisons, and distance from edited equal-source probabilities to natural donor probabilities. Report actual-source checked soundness separately; counterfactual donor outputs against an unchanged catalog must be rejected, not released. No missing/stopped/error pair is dropped.
- **Decision unlocked:** continue, revise or fold this intervention line. It does not promote L0; all language capability, parent retention, joint soundness and causal gates remain required.

## Scope limits

The wrapper itself captures caller-supplied exact prefix IDs and does not authenticate their correspondence to a `LayoutRow`; the L0i-b backend builds the row internally and proves the captured prefix hash. That capture has already consumed schema/options and remains within-request recurrence, not query-independent semantic memory. Changing the layout requires a separate registered experiment. Natural donor inputs are a positive reference, not a causal control. No validation split, paid compute or model load occurs in L0i-a/b. The parent experiment requires the merged 9 GiB operational admission reserve, 3 GiB runtime floor, a qualified immutable parent/calibration handoff, and the applicable spend issue/ledger gates.

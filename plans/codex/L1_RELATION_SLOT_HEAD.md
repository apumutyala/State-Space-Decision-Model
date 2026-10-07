# L1 isolated relation-slot q_end diagnostic

## Hypothesis

The folded q_end operation-only head fit operation labels but did not improve useful answers. Its one correctly selected Compare path then failed because the unchanged stock relation slot chose `greater_than` for `204.75 < 213.23`; the four direct relation bookends were all wrong and all labelled `less_than`. A q_end head trained only on the relation slot may improve relation selection when operation and operands are already supplied correctly. This is a conditional diagnostic, not end-to-end L1 capability.

## Phases

### L1r0 — Freeze slot-specific cohorts

- **Hypothesis:** the pinned multirecord worlds provide disjoint, balanced enough relation-slot examples with exact prior choice prefixes and isolated labels.
- **Deliverable:** load the exact 24 fit / 8 reserved calibration / 32 evaluation-only validation worlds with `load_l1_world_training_data()` and verify all whole-world hashes before any relation filtering. Emit only a JSON-level relation-row view plus sidecars; do not construct model Items or import Torch in this data-only phase. Preserve each record's `world_id` outside model input; keep relation labels separate. Never use validation targets for fitting or calibration.
- **Exit proof:** exact source/partition hashes; 576 fit, 192 reserved calibration, 768 validation relation rows; all prefixes equal the earlier operation/left/right option descriptions; target absence from provider state/cache; test rejects partition crossing and any empty/short world substitution. Confirm all validation row token-length inputs can be measured without truncation; exact encoded lengths are checked later with the pinned tokenizer before model inference.
- **Decision unlocked:** proceed to native q_end feature extraction or stop if the selected-slot distribution/prefix contract is not what the planner serves.

### L1r1 — Prove the slot-only execution seam

- **Hypothesis:** the existing native q_end head can score the relation slot after an explicit prior prefix while leaving operation, fact, left and right slots on the stock backend.
- **Deliverable:** a separate relation-cache implementation (leave the folded operation-only runner unchanged); a relation-only backend wrapper; q_end cache/live parity; tiny random CPU train/checkpoint/reload smoke; strict unchanged-parent hash; missing/nonrelation/malformed-prefix delegation checks. The real runner must call `fit_items()` with the full pinned fit record list before selecting relation Items, after the guarded exact tokenizer/encoder is available.
- **Exit proof:** every target/cache lineage binds exact parent, encoder, template, layout, request prefix and relation criteria; cache metadata excludes labels; fake and tiny-native tests show only a one-shot relation request with exactly the preceding operation/left/right choices is intercepted, all other calls delegate unchanged, cached/live logits match, and the backbone is bitwise unchanged. No real parent load.
- **Decision unlocked:** permit the bounded real-parent CPU diagnostic.

### L1r2 — Run a conditional relation-slot diagnostic

- **Hypothesis:** a trained relation slot can improve relation accuracy once the requested comparison and ordered operands are supplied correctly; this distinguishes relation readout from earlier operation/entity-binding errors.
- **Deliverable:** one preregistered stock / step-0 / trained q_end CE recipe, fit only on the 576 fit relations. Primary evaluation is the teacher-forced relation-only diagnostic over all 768 validation relation requests, with true earlier operation/left/right selections as explicit diagnostic prefixes and never the relation target. Use identity temperature for this accuracy screen; calibration worlds are reserved and never used for recipe/checkpoint selection.
- **Exit proof:** raw request/response replay and independent metric recomputation; per-class and field strata, confusion matrix, reverse-pair consistency, and paired world-cluster CIs over all 32 validation worlds; unchanged parent parameters and exact before/after stock predictions on identical relation requests. Save raw probabilities and requests/responses. Use CUDA hidden, two CPU threads, offline mode, 6 GiB start floor, 3 GiB runtime floor and 1,200-second child cap. Recheck resources immediately before load; refuse if below the floor. Estimate the phase schedule from measured row tokens before launch; incomplete/timeout runs keep partial logs and make no metric claim.
- **Decision unlocked:** if conditional relation accuracy does not improve, fold the fixed head recipe; if it improves, advance to L1r3. Conditional accuracy alone is not useful-answer capability or L1 qualification.

### L1r3 — Test parent-selected composition separately

- **Hypothesis:** a relation-slot head that improves under correct prefixes may improve full planner answers only if stock operation and ordered-operand choices reach the right relation request.
- **Deliverable:** compose the relation-only router with stock operation/operand slots; use the full 1,280-task free-running validation adapter from initial task/evidence only. Capture every actual request/response and score useful answers through the semantic kernel, renderer and task-relevance gate. Keep kernel rejection, accepted wrong-task, missing-slot and wrong-relation cases in the full denominator.
- **Exit proof:** independent replay of all captured calls; paired world-cluster CIs against stock and step-0 arms; direct retention/equality on identical relation requests; full-parent state hash unchanged. Do not use task labels to repair parent choices. If the router is not reached because upstream choices differ, report that separately from conditional relation skill.
- **Decision unlocked:** retain only a fixed variant with a useful free-running gain and non-regression; otherwise fold. Strong unrestricted-comparator, complete joint soundness and causal-intervention gates remain required before any L1 claim.

## Constraints and nonclaims

Use the pinned 24/8/32 whole-world split only. No Index/held-out data, checkpoint from Track D's training run, GPU or paid compute. This diagnostic uses stock RWKV7-0.1B-G1, not a qualified Track D parent. The current L1 evaluation adapter deliberately strips authored later questions/options; do not loosen it. A relation-only oracle-prefix view is conditional diagnostic evidence, and a relation-slot gain alone does not prove that the parent discovers the program, gives useful answers, or uses a causally mediating recurrent state.

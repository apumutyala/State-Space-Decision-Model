# Checked language growth, L0–L4

This package checks compact typed semantic programs against host-supplied evidence. It is not a trained language model. The kernel supports ReadFact, exact Compare, ExactDifference, Assert, Conjoin, ExplainDerivation, Hypothesize, Plan and Ask, with bounded rejection reasons and release-time policy/evidence revalidation. Licensed contrast and scoped historical electricity knowledge extend the checked surface; no learned level is qualified.

Use `python -m language_growth.gates --parent CHECKPOINT` for a no-load readiness report. Missing comparator, retention, joint soundness and neural intervention evidence remains unmeasured. Read [the gate guide](GATES.md) for suite preparation and execution, and [qualified parent admission](QUALIFIED_PARENT.md) for the independently pinned Stage 1 report, qualification and original snapshot required to load a native M2 checkpoint. A ledger `keep` alone is insufficient.

The [older PR triage](../plans/codex/LANGUAGE_PR_TRIAGE_20261008.md) records merged dependencies, historical folds and operator closure candidates without deleting evidence.

[Learning prerequisites](LEARNING_PREREQUISITES.md) cover train-only native slot
supervision, paired hidden/logit direction fitting and checkpoint-header
handoffs. These prepare real fitting inputs while learned gates remain pending.

An explicitly opted-in host registry can also enable `ExactDifference` for
left-minus-right over integer/decimal Facts or host constants with identical
units, quantity kinds and scopes. Arithmetic is exact under any ambient
Decimal precision, with fixed 256-digit coefficient and 64-place scale
ceilings. Results can be asserted and deterministically rendered; conversions,
division and arithmetic chaining are unsupported. The public planner is not
extended by this slice. See the
[operation contract](../plans/codex/L1_EXACT_DIFFERENCE.md) and authored fixture
`fixtures/language_growth/exact_difference.json`; these do not establish
learned capability or L1 qualification.

Run kernel tests with:

```text
python -m unittest discover -s language_growth/tests -p test_kernel*.py -v
```

The fixture pack is under `fixtures/language_growth/`. Source and data revisions are recorded in the relevant `receipts/` directory. Renderer, oracle/replay backends and actual case evaluation are separate Track L changes; they must not be assumed available from the kernel-only commit.

The kernel checks every node, including unused ones; keeps subject/field/unit/time bindings; rejects model-authored inline facts; and validates exact Decimal/integer values. It rejects private-contaminated public snapshots and nonempty recurrent-state handles. A checked result is an immutable process-local object minted by this kernel. `validate_release` rechecks current policy, evidence and record versions. This object is not a cryptographic certificate and serialized JSON does not acquire authority by containing an accepted flag.

The fixture host, ontology and coarse temporal scopes are trusted inputs. General interval overlap, explicit supersession, arbitrary inference, natural-language premise truth and task relevance are outside the implementation. Acceptance of a true but irrelevant statement is deliberately distinct from task correctness. Exact errors are internal diagnostics; public failure rendering must not disclose hidden reference existence or source values.

The kernel-only tests are authored boundary tests, not independent audit labels or evidence of alignment, model capability or causal interpretation. No model weights, GPU or paid service is used.

The public boundary also accepts the exact two-fact form
`State alpha's price and beta's warranty_months for October 2026.` using visible
entity names and canonical registered field names. The facts must be distinct
and scoped to October 2026; two fields of the same entity are supported. Only
for this form does `public_answer` offer the conjunction planner profile. The
provider still selects the operation and both facts from the full visible
candidate domain. Release requires exactly the two requested checked
assertions, in either order, and rerenders the revalidated program rather than
using planner text. This adds public typed-program expressibility; learned
composition, strong-comparator usefulness, parent decision retention and
neural semantic interventions remain unmeasured.

The optional `realization_production="l3_fact_pair_coordination_v1"` argument
to `public_answer` selects one finite L3 primitive after the same task and
release checks. It renders exactly one checked Conjoin of two factual Assert
acts as `At <scope>, <fact clause>, and <fact clause>.` The scope must be
identical; typed literals stay quoted or exactly numeric, clause order stays
unchanged, and the checked program's source and authority bindings are retained
and revalidated. The renderer neither infers a relation between the facts nor
adds a causal connective. Unknown production IDs and other speech-act shapes
fail closed without falling back to plain text. The default remains `plain`.
This is a deterministic licensed-discourse primitive only, not learned fluent
capability, L3 qualification, or an unrestricted natural-language guarantee.
Focused checks: `python -m unittest language_growth.tests.test_licensed_realizer -v`.

The optional `realization_production="l3_fact_comparison_contrast_v1"` expands
one checked Assert(Comparison) into two sentences: `At <scope>, <left fact>.
In contrast, at <scope>, <right fact>, which is <inverse relation> the first
value.` Both operands must be numeric facts about distinct subjects, with the
same field, predicate, unit, quantity kind and scope. Only a checked strict
inequality licenses this quantitative contrast. Operand order and exact values
are retained; the second sentence describes the right value against the left,
so its relation is the inverse of the checked comparison. Equality, constants,
bare facts, explanations and multiple outputs fail closed. The existing public
comparison task forms can select this production after the same relevance and
release checks. The small authored fixture is
`fixtures/language_growth_l3_contrast/fact_comparison.json`; run it and the
counterexamples with `python -m unittest language_growth.tests.test_contrast_realizer -v`.
This adds deterministic discourse expressibility, with no learned L3, causal,
temporal, strong-comparator or parent-retention claim.

`language_growth.rwkv_decision_head.rwkv_decision_backend` is a CPU FP32
prototype for growing a decision head from an already strictly loaded native
RWKV-7 parent. It uses the merged `HiddenStateBackbone`, `DecisionModel`, and
`DecisionModelBackend` APIs directly. Supply a vendored `Rwkv7Model` (or
`Rwkv7ForCausalLM`), an encoder with `identity()`, and `source_identity` containing
`model_id` and the checkpoint/loader pins. Caller-supplied identity records
provenance; checkpoint verification remains the loader's responsibility.
Parent and encoder identities must agree on `model_id` and snapshot revision
(`snapshot` or `snapshot_revision` in the parent, `snapshot` in the encoder).
The constructor also compares every shared role in their `files_sha256` maps;
any mismatch fails before the caller's parent is frozen. These checks establish
consistency between supplied identities, not authenticity of checkpoint bytes.

The constructor takes ownership of the base module and freezes it in place;
the LM vocabulary head is excluded. The default snapshot head and the
integrator, Head B, and final-state fork heads remain trainable. Use the merged
`Trainer` with `lora_rank=0`, matching seed, template and layout for head-only
training; its checkpoint stores head/typed parameters and parent/tokenizer
lineage. Serve the returned backend with the ordinary `Decider` and L planner.
Loading a saved head requires the identical parent and encoder again. This
prototype does not load checkpoints or depend on the separate FLA-to-native
CPU conversion workstream, and no pretrained capability is claimed.

After PR #71's strict CPU loader returns `plugin, parent, evidence`, use
`rwkv_cpu_parent_identity(plugin, evidence)` and
`rwkv_cpu_parent_encoder(plugin)` from this module when constructing the
backend. The identity helper reconciles the loader's full filesystem snapshot
path, revision alias, filename-keyed file hashes and implementation digest
with the bridge's role-keyed pins; disagreement fails closed. The encoder
wrapper delegates to that exact loaded World encoder and binds its model,
snapshot, full weight/config/tokenizer/implementation hashes and no-control-
token policy. Changing the source plugin later cannot replace its captured
callable or identity.

Bind serving through `rwkv_cpu_parent_decider(backend, plugin, calibration=...)`.
It verifies backend/encoder provenance and applies the plugin's context limit
(at most 2048 tokens); do not use an uncapped `Decider(backend)` for this CPU
parent. This adds no loading or fallback path. Serving a trained artifact still
requires PR #79's strict artifact, complete revision, calibration and runtime
pins; a newly initialized head is not a reconstructed trained artifact.

The native bridge tests run with `python -m pytest
language_growth/tests/test_rwkv_decision_head.py -q`. They cover actual tiny
RWKV hidden states, all four attention-free head gradients, an unchanged
frozen parent, exact head checkpoint reload, shared-prefix inference, and a
free-running planner call. Random test weights are not a language result.

The L0e fixture evaluator runs the authored proposal schedule through the merged kernel and renderer with `python -m language_growth.evaluate --fixtures fixtures/language_growth --out <fresh-directory>`. It records observed boundary behavior separately from authored task labels and makes no provider call. For a same-evidence direct decision comparison, use PR #17's `language_growth.decision_view.direct_decision_request` after its canonical `parent_bridge` dependency from PR #13 lands. The evaluator does not maintain a second decision-request serializer.

Export the committed L1 development-train cohort with `python -m
language_growth.multirecord_training --out <fresh-directory>`. The bridge
accepts only the exact pinned 32-world `l1_multiworld_development_train` cohort
(1,280 tasks, 4,096 slot rows) and its manifest. It checks every typed reference
program and uses the shared `export_training_records` path with an explicit
train-only L1 allowlist. Each state contains only earlier teacher-forced target
IDs and their offered descriptions; the current-slot target is a separate
label. Task/world IDs and source row, manifest, world-row and catalog hashes
stay outside requests. The export manifest preserves the complete source
manifest. The 40-task `l1_development_train` fixture is retained only for
plumbing/evaluation tests and is not accepted for training. The separate
`l1_multiworld_development_validation` cohort is evaluator-only and rejected by
the training exporter. L0 export is unchanged.

This is data plumbing only, not model training, free-running qualification or
L1 capability. Strong comparator, decision retention, adversarial soundness and
causal intervention remain gating requirements, alongside learned L0 gates.
No model, weights, GPU, paid compute, Decision Index or heldout data is used.

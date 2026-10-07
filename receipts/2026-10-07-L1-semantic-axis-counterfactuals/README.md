# Synthetic semantic-axis counterfactual preparation receipt

2026-10-07. Windows CPU, $0. No models, weights, GPU, paid compute, Decision
Index, audit corpus or heldout access. Branch
`codex/l1-semantic-axis-counterfactual-fixtures`, exact PR #133 base
`e94c79793082fc8643bae34ac52813760f1950e8`.

Each split has 32 disjoint base worlds, both independent scalar axes, 64
axis groups, 192 low/equal/high variants and 1,152 inputs/programs. Total:
2,304 authored programs. The shared base-world ID is the future clustering
unit; this count is not a power or capability claim.

Visible entity names use the pinned PR #133 L1 multiworld generator's
seed-1707 sampling from six split-disjoint names per domain. Names recur
across worlds and each name binds to all four record refs; no numeric world
suffix is present. Low/equal/high variants preserve their mapping. These
name pools intentionally overlap the original L1 development cohorts; the
new scalar domains remain disjoint from those cohorts and from one another.
This is not unseen-name generalization evidence. Scorer world/axis-group IDs
remain absent from planner inputs.

Only `*-inputs.jsonl` enters `plan_checked_answer`. Each line contains only
`task_input`, permitted `host` and `catalog`; supply a provider and model
configuration separately. The existing semantic planner generates typed
operation/fact/operand/relation requests. There are no prebuilt answer-string
choice requests. `*-scorer.jsonl` and manifests remain evaluator-only: axis
coordinates, level/world/split labels, zero-based input offsets, typed-slot
targets, reference programs, checked meanings and rendered text stay there.

Catalog-bound contexts change with each premise value. After normalizing
those identities, corresponding inputs differ at exactly one typed value.
All other evidence, task wording and generated typed menus remain fixed.
The oracle control also verifies matched question IDs, option IDs/order/
descriptions and prior model-selected prefixes across variants. Unrelated
checked meanings, rendered text and target choices remain identical.
Discovery/validation have disjoint worlds, axis groups, entities, values
and catalogs. Schema, record refs and task vocabulary are shared.

The test-only `InputOracle` receives only each planner request and derives
its typed selections from permitted task/state. All 2,304 positive tasks
pass through the actual planner, kernel and release-validating renderer,
with exact scorer program/slot/meaning/text parity. This uses 6,912 scripted
provider calls and zero model forward calls. One-hot oracle distributions
are not calibrated confidence. All 1,152 false-relation controls are rejected
without released text, using 4,608 more scripted calls. Two supported
wrong-field controls use four calls, for 11,524 total in-process scripted
calls in the fixture test suite. External provider/model calls remain zero.
Supported wrong-field selections remain kernel-valid
but fail scorer matching; correctness and task relevance are distinct.

Reproduce to a fresh external directory (choose a new destination each run):

```powershell
python -m language_growth.semantic_axis_fixtures --output-dir ../ssdm-data/semantic-axis-reproduction-20261007
python -m unittest language_growth.tests.test_semantic_axis_fixtures
python -m unittest language_growth.tests.test_kernel language_growth.tests.test_kernel_adversarial language_growth.tests.test_decision_view language_growth.tests.test_planner
git diff --check
```

`RECEIPT_DIR` is the read-only baseline used by tests. Generation requires a
fresh path and refuses existing output or receipt directories. It builds
both three-file split bundles in a sibling staging directory, then publishes
the full pack by atomic directory rename. Tests verify duplicate attempts
and injected staging-write failure preserve existing receipt/output bytes
and leave no partial published pack.

`verification.json` pins byte sizes/hashes, generator hash/blob, base ancestry,
counts and verification commands. The enclosing commit preserves the phase
note at `plans/codex/L1_SEMANTIC_AXIS_COUNTERFACTUAL_FIXTURES.md` and every
artifact. Earlier answer-choice fixture files are replaced by planner inputs;
no prior training or other fixture cohort is changed. The first review of
answer-choice head `6dd103da` flagged that it did not exercise the L planner.
Planner-input schema was introduced at `f205c8f`; visible world-index suffixes
were removed at `2eefbaa`. Fresh exact-head review at `ecb2d34` found no
source/data blockers. This follow-up corrects review provenance; Track D review
will be requested on the stacked PR before merge.

These fixtures test factual counterfactual responsiveness and can support
future semantic-state edits, but do not demonstrate learned recurrent
mediation, calibrated ability, alignment, or internal interpretability.
This is an evidence-edit substrate and scripted boundary control, with no
recurrent state edit, learned provider or model result. A separate protocol
must define its edit port, matching, controls and qualification.

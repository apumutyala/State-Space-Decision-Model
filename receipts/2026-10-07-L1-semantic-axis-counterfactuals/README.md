# Synthetic semantic-axis counterfactual preparation receipt

2026-10-07. Windows CPU, $0. No models, weights, GPU, paid compute, Decision
Index, audit corpus or heldout access. Branch
`codex/l1-semantic-axis-counterfactual-fixtures`, exact base
`e94c79793082fc8643bae34ac52813760f1950e8` (PR #133).

The generator prepares 32 disjoint base worlds in each of discovery and
validation. Each base world has independent price and warranty axis groups,
each with low/equal/high variants and six tasks. Per split: 64 axis groups,
192 variants, 1,152 model requests and 1,152 checker-accepted programs. Both
splits total 2,304 authored programs. Each task keeps four records available.
The shared base-world ID is the future uncertainty-clustering unit; neither
this count nor synthetic validation implies adequate statistical power.

Each `*-requests.jsonl` line is a standalone decision request. Only these
lines can reach a future provider. `*-scorer.jsonl` and `*-manifest.json`
are evaluator-only. The scorer links requests by zero-based line index;
axis metadata, target option IDs, reference programs and checked meanings
never enter the model requests. Option order is fixed within each triplet.
Direct/explanation choices include all three scalar intervention values and
one distractor, with no low/equal/high labels; comparison choices include
all three relations. Unrelated-field read/comparison choices remain fixed.
This explicit menu is diagnostic scaffolding and not model-discovered task
structure, a training export or a free-running planner evaluation.

Catalog-bound snapshot IDs and request IDs change when evidence changes.
After excluding these opaque identity fields, each corresponding request
differs at exactly one path: the selected record/field's typed value. All
other visible evidence, question wording, option IDs/order and descriptions
are fixed. Other-field target choices and checked outputs stay identical.
Discovery/validation have disjoint world, axis-group, entity, numeric-value
and catalog identities; schema, refs and task vocabulary are shared.

Reproduce from the repository root with:

```powershell
python -m language_growth.semantic_axis_fixtures
python -m unittest language_growth.tests.test_semantic_axis_fixtures
python -m unittest language_growth.tests.test_kernel language_growth.tests.test_kernel_adversarial language_growth.tests.test_decision_view
git diff --check
```

`verification.json` pins exact artifact bytes and SHA-256 hashes, source
hash/blob, base ancestry, counts and validation commands. Its checks cover
deterministic regeneration, mutation isolation, identity binding, exact
choices, target meanings, inverse comparisons, kernel rejection of false
relations, metadata exclusion and invariant unrelated answers. These
adversarial fixture assertions are authored checks, not independent labels.
The enclosing commit preserves the generator and all listed artifacts.

The phase note is
`plans/codex/L1_SEMANTIC_AXIS_COUNTERFACTUAL_FIXTURES.md`. Existing training,
evaluation and prior source fixtures are unchanged. This pack has no runtime
adapter and no recurrent intervention implementation. A future protocol must
separately define its state edit port, controls, matching and qualification.

These fixtures test factual counterfactual responsiveness and can support
future semantic-state edits, but do not demonstrate learned recurrent
mediation, calibrated ability, alignment, or internal interpretability.
They contain no model result and unlock only fixture/protocol review.
Fresh exact-head review is required before push or PR creation.

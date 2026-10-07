# L1 semantic-axis counterfactual fixture preparation

2026-10-07. Local CPU, $0. Base is exact PR #133 commit
`e94c79793082fc8643bae34ac52813760f1950e8`. This phase prepares synthetic
evidence interventions for future recurrent semantic-state experiments.
It is a fixture pack, with no training, model loading, inference, weights,
GPU, paid compute, Decision Index, audit corpus or heldout access.

Hypothesis: if a future system responds to the offered scalar facts, moving
one registered price/USD or warranty/month value from below to equal to above
a comparator should change the direct answer, explanation value and both
comparison directions, while preserving answers about the other field.
This is a testable prediction for factual counterfactual responsiveness;
this preparation reports no observed model response.

Deliverable: `language_growth/semantic_axis_fixtures.py` plus separately
serialized request-only and scorer-only files in
`receipts/2026-10-07-L1-semantic-axis-counterfactuals`. Each of two disjoint
synthetic development splits, discovery and validation, has 32 base worlds
with four named records. Each world has two independent axis groups on the
same target record, one per field. Each group has low/equal/high variants
and six fixed queries: direct scalar read, forward comparison, reverse
comparison, explanation, unrelated-field read and unrelated-field comparison.
That is 64 axis groups, 192 variants and 1,152 requests/programs per split.
Use the shared base-world ID to cluster future uncertainty estimates across
axes; this sample size is not an adequacy or statistical-power claim.

Within an axis group, only the target field's typed value changes. Unit,
type, quantity kind, scope, all other facts, records, question/task text,
option IDs, order and descriptions remain fixed. The content-bound snapshot
and request IDs necessarily change with evidence. Snapshot IDs hash the
catalog with its snapshot under a fixed placeholder, then replace that
placeholder with the resulting ID. Those identity changes carry no authored
level label. Options are shuffled once per world/query, independently of
variant values and targets. The queries and answer candidates are authored
diagnostic scaffolding, not a free-running compositional planner trace.

Only lines in `*-requests.jsonl` are provider inputs. All axis coordinates,
variant labels, world IDs, split identity, request offsets, target option
IDs, reference programs and checked outputs stay in `*-scorer.jsonl` or
manifests. Send independent stateless probes; no recurrent state edit or
cross-variant context cache is implemented. Existing training allowlists,
model adapters, score paths and prior fixture bytes are unchanged.

Exit proof: deterministic byte regeneration; manifests and artifact hashes;
disjoint world/entity/value/catalog sets across the two splits; both axes
per base world; exactly one semantic fact mutation; fixed wording and
choices; all authored programs accepted by the existing v0.1 kernel;
independent target-to-checked-meaning checks; false-relation rejection;
scalar response changes and unrelated-field invariance in scorer targets;
empty authored prefixes and no scorer metadata in requests. Mutation and
leakage assertions are adversarial fixture checks, not model accuracy tests.
The receipt includes reproducible local commands and source provenance.

Decision unlocked: review of a bounded evidence-counterfactual substrate
and design of a separately reviewed recurrent intervention protocol. These
fixtures test factual counterfactual responsiveness and can support future
semantic-state edits, but do not demonstrate learned recurrent mediation,
calibrated ability, alignment, or internal interpretability. Evidence edits
alone cannot show that any internal state mediates a learned behavior.
Discovery/validation are iterative synthetic development groups, not an
independent evaluation or heldout qualification. They unlock no level
promotion, paid run or model capability claim.

The local commit is prepared for fresh exact-head adversarial review before
any push or pull request. Preserve all fixture and verification artifacts.

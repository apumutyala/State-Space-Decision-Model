# L1 semantic-axis counterfactual fixture preparation

2026-10-07. Local CPU, $0. Exact PR #133 base:
`e94c79793082fc8643bae34ac52813760f1950e8`. This phase prepares synthetic
evidence interventions for future recurrent semantic-state experiments.
No models, weights, GPU, paid compute, Decision Index or heldout data.

Hypothesis: changing one registered price/USD or warranty/month value from
below to equal to above a comparator should change the checked direct fact,
explanation value and both comparison directions, while preserving answers
about the other field. This is a factual counterfactual responsiveness
prediction; the scripted integration controls are not learned responses.

Deliverable: `language_growth/semantic_axis_fixtures.py` plus separate
`*-inputs.jsonl`, `*-scorer.jsonl` and manifest files under
`receipts/2026-10-07-L1-semantic-axis-counterfactuals`. Each of discovery and
validation has 32 disjoint synthetic base worlds with four named records.
Each world has two independent axis groups on the same target record, one
per field. Each group has low/equal/high variants and six queries: direct
read, forward/reverse comparison, explanation, unrelated-field read and
unrelated-field comparison. Per split: 64 axis groups, 192 variants and
1,152 inputs/programs. The shared base-world ID is the future uncertainty
clustering unit; sample size is not an adequacy or statistical-power claim.

Visible entity names use the pinned PR #133 L1 multiworld generator's
seed-1707 sampling from six split-disjoint names per domain. Names recur
across worlds and each name binds to all four record refs; no numeric world
suffix is present. Low/equal/high variants preserve their mapping. These
name pools intentionally overlap the original L1 development cohorts; the
new scalar domains remain disjoint from those cohorts and from one another.
This is not unseen-name generalization evidence. Scorer world/axis-group IDs
remain absent from planner inputs.

Each input contains exactly `task_input`, permitted `host` and `catalog`.
It can be passed directly to `plan_checked_answer` with a separately supplied
provider. The existing planner synthesizes typed operation/fact/operand/
relation menus. Reference programs, slot-choice targets, checked meanings,
rendered answers, axis/variant labels and input offsets are scorer-only.
There is no prebuilt choice among final natural-language answers. Training
allowlists, planner/model adapters and prior fixture bytes are unchanged.

Within an axis group, only the selected field's typed value changes. All
other fields, records, types, units, scopes and task wording remain fixed.
The catalog-bound context identity changes with the evidence. Snapshot IDs
hash the catalog with its snapshot under a fixed placeholder, then replace
that placeholder with the resulting ID. Planner-generated question IDs,
wording, option IDs/order/descriptions and pre-relation choice prefixes are
matched across variants. Runtime request IDs are fresh planner identities.
Each query is a separate stateless planning run; no recurrent edit is present.

The test-only `InputOracle` derives choices from each offered task and
permitted state; its constructor accepts no scorer input. Its narrow parser,
exact decimal comparison and one-hot distributions are scripted controls,
not learned ability or calibrated confidence. All 2,304 positive tasks run
through the real planner, kernel and release-validating renderer. Their
program, typed-slot choices, checked meanings and rendered text must match
the separate scorer. All 1,152 deliberately false comparison controls must
fail the kernel without releasing text. A supported wrong-field selection
remains kernel-valid but mismatches the task scorer: kernel acceptance alone
does not establish relevance. These checks do not prove neural mediation.

Exit proof: deterministic exact-byte regeneration; manifests/hashes; both
axes per base world; split disjointness; one semantic fact mutation; fixed
planner menus; no scorer leakage; correct inverse relations; changing
checked/rendered axis outputs; invariant unrelated outputs; real planner/
kernel/renderer integration; and preserved artifacts on duplicate generation
or injected staging failure. The generator requires a fresh `--output-dir`,
preferably external `ssdm-data`, and stages the complete six-file pack before
an atomic directory rename. It refuses existing paths, including receipts.

Decision unlocked: review of this evidence-counterfactual substrate and a
separately reviewed recurrent intervention protocol. These fixtures test
factual counterfactual responsiveness and can support future semantic-state
edits, but do not demonstrate learned recurrent mediation, calibrated
ability, alignment, or internal interpretability. Discovery/validation are
iterative synthetic development groups, not independent or heldout
qualification. This unlocks no level promotion, paid run or model claim.
The first review of answer-choice head `6dd103da` found that those requests
were not sufficiently tied to Track L's semantic-planner path. Planner-input
schema was introduced at `f205c8f`, and visible world-index suffixes were
removed at `2eefbaa`. Fresh exact-head source/data review at `ecb2d34` found no blockers. Receipt
and status record the review sequence; Track D review will be requested on
the stacked PR. The revised fixture exercises
`plan_checked_answer`, kernel and renderer with an input-derived scripted
oracle, not a learned provider or recurrent edit.

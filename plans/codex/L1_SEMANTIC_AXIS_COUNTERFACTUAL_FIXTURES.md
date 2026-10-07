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
Fresh exact-head review is required before push or PR creation.

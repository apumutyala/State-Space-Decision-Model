# L1a four-record development preparation

2026-10-07. This $0 CPU phase prepares synthetic fixtures for future L1
multi-record read, explain and comparison work. It is preparation only,
not L1 capability proof. Learned L0 gates remain prerequisites for L1
progression. Exact calculations such as ExactSum remain gated on the
separate v0.2 interface/module review; this cohort uses only the existing
v0.1 ReadFact, Compare, Assert and ExplainDerivation operations.

The deliverable is `language_growth/multirecord_data.py`, separate from the
L0 generators. Each world contains exactly four distinct named records,
both price/USD and warranty/month fields, and 40 tasks: eight reads, eight
explanations and 24 comparisons. Every ordered distinct-record pair is
compared on each field, so A-vs-B and B-vs-A share an evaluator pair ID.
Each field has three distinct values, one repeated across an equality pair;
its ordered comparisons comprise two equal, five less-than and five
greater-than cases. All four records appear as read/explain targets and
as both left and right comparison operands. Comparisons preserve field,
unit, quantity kind and temporal scope.

The two `l1_development_*` domains have names, templates, warranty ranges
and price ranges disjoint from one another and both L0 development domains.
They remain iterative synthetic development, with no held-out or independent
audit status. Labels are explicitly marked synthetically authored and
checker-valid, not independently validated. Every generated reference program
must pass `check_program` before a row can be returned.

All labels/programs are under `evaluator_only`; only `model_inputs` can be
sent to a provider. Every request preserves all four records and all values.
Fact/operand slots retain all eight record-field candidates, including
same-record and cross-field distractors, and relation slots retain all three
relations. Production and augmented formats preserve candidate meanings;
augmentation shuffles IDs. No request contains an authored choice prefix,
accepted program, current target or evaluator interpretation. The per-task
slot schedule itself is authored diagnostic scaffolding, not model discovery.

This is a **fixture-only schema**. The merged `export_training_records`,
`run_development_tasks` and `score_development_task` expect the L0 split names
and `training_only` layout; the exporter also verifies teacher-forced prefixes.
They cannot consume this cohort. Empty-prefix slot requests must not be used
as compositional training examples or described as free-running traces.
A separately reviewed adapter and task scorer are required before model
training/evaluation. That future integration must keep labels separate and
give a free-running planner initial task/evidence only. No existing L0
generator, interface, trainer or evaluation score path changes in this phase.

Four-record support and complete ordered pairs prepare meaningful random
semantic controls: there are multiple alternative records/values beyond a
binary swap, all available in the same model evidence and choice set.
This phase does not implement, run or qualify those controls. A future
protocol must declare its edit port, null distribution and predicted effect;
evidence edits or fixture permutations alone cannot prove neural mediation.

Exit proof is deterministic generation, exact four-record support, label
exclusion, checker-valid programs, complete field/record/pair/order/value
coverage, split disjointness and canonical manifest hashes. The receipt is
`receipts/2026-10-07-L1-multirecord-data/README.md`. This unlocks review of the
data foundation and later integration design, not paid compute, model loading,
level promotion or an L1 claim. No Index, sealed/held-out corpus, weights,
GPU or paid service is accessed.

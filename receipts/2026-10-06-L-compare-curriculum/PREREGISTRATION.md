# Fixed comparison paraphrase experiment

Registered before training or validation scoring on 2026-10-06. Source baseline
is origin/agent-main 4733f785e5284ea959d4d640fff884136487bd05. The model/target
recipe is copied from the committed query-target diagnostic
fcf620177d8e9b1575e5e0ed590d4aae1a9cb3b3 as authorized by the coordinator;
there is no import, runtime or Git dependency on that branch.

Hypothesis: comparison training word-order diversity improves exact requested
comparison target interpretation on a new comparison wording, with possible
read/explain regressions reported. This is one condition versus a matching
original-wording baseline, not a sweep.

Both arms: random seed 20261006, 600 train worlds from development_train seed
20261006, width-32 byte diagonal SSM, 27,937 parameters, Adam lr=0.003, batch
32, gradient norm clip 1.0, 300 steps, two CPU threads, final checkpoint only.
The same seeded row indices are sampled in both arms, with reset initialization
and optimizer. Each arm stops if training exceeds 120 seconds; no partial result
will be represented as a completed comparison.

The sole intended treatment replaces the comparison question in 200 existing
training worlds with four equally assigned templates (index//3 modulo 4):

- Compare the October {field} of {left} and {right}.
- In October 2026, compare {left} and {right} on {field}.
- Assess how {left}'s October {field} compares to {right}'s.
- Considering {field} for October 2026, compare {left} against {right}.

Train labels, candidate sets, world membership and read/explain rows are equal.
Validation: 180 development_validation worlds with fresh seed 20261008, compared
on identical rows in both arms. The comparison wording is fixed as
`At October 2026, how does {left}'s {field} compare with {right}'s?`.
Read/explain wording remains the generator's validation wording. Train and
validation retain disjoint name pools and numeric ranges. The comparison
template is different from original training and prior validation templates.
Validation is scored only at the fixed final checkpoint; no tuning or rerun
based on its result. Primary: exact full target match overall and comparison
subset. Report read/explain per-kind, all target attributes, paired improvements
and regressions, random initialization, training fit and ID permutation.

No Index/untouched held-out slice, external teacher, GPU, download, network or
paid compute. Only authored iterative development data. No answer computation
or checked-planner release integration is claimed.

# Transparent learned CPU baseline

This bounded $0 experiment implements the user's request to advance actual
learning on the available development cohort. It does not climb the paid
qualification ladder or promote L0. No audit, Index, held-out split, downloaded
weights, GPU or external service is accessed.

Hypothesis: a small option-wise linear ranker can fit semantic slots and produce
useful complete programs from fresh same-domain synthetic development worlds.
Its deliverable is `language_growth.transparent_cpu`, a NumPy full-batch softmax
trainer, JSON checkpoint, shared D `Decider` backend and existing L planner path.
The exit proof is decreasing supervised loss, saved/restored fitted weights,
per-slot predictions, complete one-shot rollout, initial-weight and numeric-sign
ablations, with source/data/config/seed/raw-output hashes. This unlocks a concrete
learned development reference for subsequent model comparisons.

Features are deliberately engineered and transparent: query words crossed with
the operation description's first word; field-token overlap and entity mention
position crossed with fact/left/right slots; exact Decimal comparison sign
crossed with relation description. Weights learn the ranking and sign-to-relation
mapping. Numeric sign and entity grounding are performed by feature extraction,
so this baseline does not demonstrate learned numerical reasoning or raw-text
interpretation. Option IDs and input identity hashes are excluded from features.
The kernel supplies acceptance and the existing renderer supplies checked text.

Training consumes the existing 5,000-task `development_train` cohort after
checking its receipt and aggregate record hash. Evaluation generates 300 fresh
`development_train` worlds, seed 314159, with mixed shuffled and production
choice IDs. These are iterative development measurements, not untouched
generalization. Slot scoring uses teacher-forced earlier slots; the one-shot
planner uses its own earlier choices and zero repair feedback. Both denominators
and all failed cases are preserved. Identity temperature is uncalibrated; no
calibration or whole-answer probability claim is made.

Run from the repository worktree:

```powershell
python -m language_growth.transparent_cpu --cohort C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k --output C:/Users/apumu/research/CHLabs/ssdm-data/l0-transparent-cpu-20261006 --eval-seed 314159 --eval-count 300 --seeds 17 29 43 --epochs 200 --learning-rate 2
```

Only NumPy is added to the runtime requirements, and it is already available in
the current CPU interpreter. No packaging or shared trainer architecture changes
are needed. Each scorer call counts as one structured model inference call and
zero tokenizer tokens; this is not a neural-token throughput comparison. Before
merging measured results, a separate reviewer must recompute all counts from
the raw predictions and review the feature boundary and metric scope.

# C multirecord training enablement candidate

Based on D13 `8d92f04f6bbc46ec40cf19b2ab5786435e139266`, with the existing
g1v2 ordinal option-embedding fix `b565ec29d79476c77d5deb7e22683f49543e961b`
cherry-picked as `efa1ae5`. A/B worktrees and running jobs are unchanged.

Two additional fixes:

1. `heads/model.py`: g1v2 lexical vectors use schema option-description spans,
   as ordered rows already do. Previously they used answer-ID continuation
   embeddings. On an actual pinned C K77 record, maximum error against expected
   description-span means changes from 1.320847 to exactly zero. Description
   changes now affect the vectors; changing only continuation ID representation
   does not. Frozen embedding gradients remain absent. This changes g1v2 joint
   behavior after the zero gates learn; retain original code for historical
   checkpoint comparisons.
2. `train/trainer.py`: the joint path limits total continuation options across
   all prefix groups. One complete C K77 record (7 questions, 92 options) plus
   a single K77 row previously yielded 169 continuations despite row_options160.
   The corrected sampler emits 7/92 then 1/77, without dropping/reordering records.
   A record exceeding the cap by itself raises before its consumption. A prior
   fitting batch is returned before that error. Existing record splitting for
   records longer than the row budget is unchanged; C's seven questions fit its
   eight-row budget. Changed step grouping is not a claim of gradient-equivalent
   optimization or measured GPU memory savings.

CPU validation, CUDA_VISIBLE_DEVICES empty, OMP/MKL threads1:

- 16 tests passed in130.08s: joint-record head, g1v2 ordinal embeddings, and new
  option-cap tests. Existing test emitted one scalar-conversion warning.
- Three lexical-vector regression tests passed in2.69s.
- One new g1v2 mixed-type joint-versus-fork initialization test passed in3.43s.
- Source/actual-C tokenizer proofs and reproducible scripts are on
  `stack/rwkv7/c-multirecord-synth`, under
  `receipts/2026-10-09-C-multirecord-quota/` (`training-admission.json`,
  `lexical-prior/`, and the initial `joint-option-cap/` prototype).
  The integrated cap implementation adds the reviewed earlier-batch-before-error
  correction to that initial prototype.

## Matched experiment handoff (A owns recipe and launch)

Use only a new clean B v2d build containing the exact C7k all-K quota,
pointer SHA256 `2c4659464e3ddaab28145b45628c59081964300f64d855f402edc1f6e1b81618`.
The C pool is a replacement, not an append to overlapping original/compact pools.
Require all7,000 selected C IDs in the final B pool after guards; a later trainer
subsample does not automatically inherit the pool's K/style coverage.

For C-consuming training, use `layout=g1v2`, `template=jrt-rwkv-g1-v3w`,
`train_max_k=255`, `row_options=160`, and a row budget of at least7. Keep the
evaluation max_k16 and frozen unit/scorer definitions unchanged. The original
D13 max_k16 filters out the ranking question from6,738/7,000 C records before
joint grouping. The actual D13 adapter/filter retained7/7 questions in all204
available K/style samples with the candidate settings; max effective length2048,
max options per record92. Full-pool token limits are inherited source proofs,
not a repeated token census on this branch.

Compare fork versus corrected joint_record on the SAME clean B draw, same C
quota, selected training IDs, v3w/g1v2 format, training K cap, seeds, loss,
optimizer schedule and checkpoint-selection rule. Keep training exposure or
clock choice explicit; record any differing completed steps/records. Both arms
need the ordinal fix so that it is not an extra treatment. Retain raw prediction
artifacts and use the existing proxy600 screen and paired comparison for the
first verdict; a screen is not full-DEV qualification. Record complete-question
retention in the actual trainer draw and inspect learned joint/lexical gates.

No clean B v2d pointer or launch authorization is supplied by this candidate.
No B records, TEST, held-out Index or retention-v3 rows were read; no trained
backbone weights were loaded. CPU only, $0 GPU, no Index gain claimed.

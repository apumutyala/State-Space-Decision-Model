# L0 synthetic training cohort

Generated with `language_growth.training_cohort` at source `e7f1ca18baac9b3174c1bb7e9f4b1fc19537034b`, seed 271828, count 5000, `development_train`, mixed production/augmented choice format. The generated artifacts are on the local research data volume at:

`C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k/`

The cohort contains 5,000 generated training tasks and 13,332 exported D-format choice records. The independent reviewer regenerated all records and matched the canonical JSONL bytes. Aggregate task and record hashes, per-row hashes, counts, source-file hashes, and shared byte-adapter verification are preserved in `builder_receipt.json`; the development-domain manifest is copied alongside it.

This is synthetic `development_train` supervision only. Earlier gold slots are teacher-forced context by design; current-slot labels are separate from model requests. No validation, audit, Index or held-out rows are included. No model was loaded or trained; this cohort is not a learned-capability, parent-retention or alignment result. Cost was $0.

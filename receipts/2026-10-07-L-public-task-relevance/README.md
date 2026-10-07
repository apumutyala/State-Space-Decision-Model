# Bounded public task-relevance gate

CPU-only implementation and boundary tests for L-39. The public projection now
parses only the fixed development read/explain/compare grammar, resolves
ordered record/field/scope operands from the audience-filtered snapshot, and
refuses a checked program whose final speech act does not match that task.
Unsupported wording is refused before provider use. This is not an unrestricted
language relevance or model capability result.

The test receipt records the exact environment, command, synthetic fixture
hashes, and code hashes. All 372 repository language-growth and jobs-lint tests
passed with `CUDA_VISIBLE_DEVICES=-1`; the model runtime was not used for
inference. The two prepared RWKV CPU-screen registrations were refreshed only
in their broad `source_sha256` maps so their no-load source-drift guards remain
valid. Their inputs, targets, configs and protocols did not change. No model or
checkpoint was loaded, no GPU or paid compute was used, and no screen ran.

# Frozen CPU wording curriculum experiment

Dependency: PR #64, `origin/codex/l0-tiny-ssm-slots` at
`115c4535ce9db716d3dde70ad000d206fdd2a4b0`. This branch keeps new commits distinct;
no push or PR is authorized before user review. PR #67's measured
`development_paraphrase` wording is retired and is neither loaded nor used here.

Hypothesis: eight balanced surface templates per read/explain/compare task kind
improve operation and argument discovery in the unchanged 70,081-parameter,
width-64, attention-free diagonal SSM. Curriculum and old training share the same
5,000 latent worlds, typed values, labels and offered choices. Their request
wording differs. Task-kind counts are 1,667 / 1,667 / 1,666; template frequencies
within each kind differ by at most one. Slot records are sampled uniformly.

Freeze 300 `development_transfer_v2` tasks before training: 100 per kind, six
fresh phrases per kind, names and value ranges disjoint from both original
development domains and curriculum. Exact task rows and labels remain separate
from requests, with all reference programs kernel-checked. The transfer set is
iterative synthetic development, not an official held-out benchmark.

Train curriculum and matched old-cohort arms at seeds 17,29,43, fixed Adam
lr=0.003, 400 steps, batch 32, clip norm 1.0. Finish all training before evaluating
transfer. Evaluate those six models and the exact frozen PR #64 seed-20261006
checkpoint on the same tasks. No tuning or extra training follows transfer results.
CPU only, two intra-op threads and one inter-op thread, $0, 300 seconds across
the whole experiment and a continuously checked 3-GiB available-RAM floor.

Preserve checkpoints, loss histories, teacher-forced slot scores, free-running
calls and output programs, semantic/exact/kernel results, per-kind/template/slot
counts, artifact hashes, and 10,000-draw paired task-cluster bootstrap intervals
stratified by task kind. CIs condition on the three training seeds; do not count
slots or repeated seeds as independent tasks. Next checks: matched transfer
performance and operation/argument failure counts. This is no L0 or scaling gate.

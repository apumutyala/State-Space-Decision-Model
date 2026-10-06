# Single frozen operation-token intervention: pending lean review

Stacked dependency: PR #63, `origin/codex/l0-transparent-cpu-baseline` at
`64f473bb143b367d655d4929217f16f3413cbdaf`. Its branch/worktree are unchanged.
Variant source `c61b0930ca5e885b46fc46204ac9a892e6743607` was clean when measured.
The new experiment is on `codex/l0-intent-token-invariance` under the exact-cased
`C:/Users/apumu/research/CHLabs/ssdm-codex-l0-intent-token-invariance/` worktree.

Only operation features change: remove pure numeric tokens, twelve month names
and a fixed 34-word function-word set, declared before measuring. The complete
filter is in the receipt config and source. All other feature extraction, task
generation, supervision, kernel/renderer logic and optimizer settings are fixed.
One filter variant was fitted once with seeds 17/29/43; no tuning loop occurred.

The original 13,332 fitting records and same 300 iterative development-validation
tasks retain hashes respectively
`eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c` and
`d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f`.
Fitting uses 200 epochs, learning rate 2, L2 1e-4; evaluation seed 314159,
mixed production/shuffled options, one-shot planning, zero repair. Exact command,
CPU/software identity, hashes, seeds and per-run counts are in `receipt.json`.
No GPU, paid service, checkpoint download, sealed held-out or Index access.
Billed rate and incremental paid cost are $0. Seven focused tests passed:
`python -m unittest language_growth.tests.test_transparent_cpu -v`.

| Raw measurement, every seed | Prior frozen validation | Filter variant |
| --- | --- | --- |
| Free-running task-correct | 200/300 | 300/300 |
| Free-running kernel-accepted | 300/300 | 300/300 |
| Comparison tasks correct | 0/100 | 100/100 |
| Read / explanation tasks correct | 100/100 each | 100/100 each |
| Operation slot, teacher-forced | 200/300 | 300/300 |
| Fact / left / right / relation slots | 200/200; 100/100 each | same |
| Relation without engineered numeric sign | 43/100 | 43/100 |

Variant task-correct median 300, range 300–300 over three initializations;
fit time median 1.6578 seconds, range 1.5039–1.7078 seconds. Initial-weight
variant models solve 0/300; trained statuses are accepted 300 each, with no
unsupported/provider-error/runner-error outcomes. All denominators, initial
and numeric-sign controls are preserved. No negative-result fold is triggered
by these raw counts, but the measured interpretation awaits lean internal review.

Checkpoints and losslessly compressed evaluation tasks, all raw logits,
per-feature contributions, requests/responses and complete free-running outcomes
are committed here. Receipt prediction hashes apply to decompressed bytes.
Uncompressed originals remain at
`C:/Users/apumu/research/CHLabs/ssdm-data/l0-intent-token-invariance-20261006/`.

Freeze these apparent-winner checkpoints before any later untouched synthetic
audit cohort is generated for a separate gate. This reused validation domain
is an iterative intervention set, not final held-out evidence. The result is
engineered-feature option ranking, not learned raw arithmetic, shared-parent
retention, calibrated confidence, causal mediation or L0 promotion. No PR is
opened before the coordinator's lean review. PR #63 remains unchanged.

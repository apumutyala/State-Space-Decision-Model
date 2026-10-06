# Frozen wording evaluation: transfer failure

The operation-filter winner from PR #65 fails this newly authored synthetic
development check. The cohort jointly shifts wording, entity names and numeric
ranges: it diagnoses combined distribution-shift failure, not a causal effect
of wording alone. Each frozen seed (17, 29, 43) solves 100/300 tasks:
read 100/100, explanation 0/100, comparison 0/100. Semantic answer, exact
reference-program, and exact rendered-text counts are independently recorded
and coincide at 100/300. All 300 programs pass the kernel; acceptance alone
does not establish that the user's requested task was completed.

This branch is `codex/l0-frozen-wording-eval`, in exact-cased worktree
`C:/Users/apumu/research/CHLabs/ssdm-codex-l0-frozen-wording-eval/`, based on
unmerged PR #65 (`codex/l0-intent-token-invariance`) at
`ca62adc67775e2de1019ec4bcdc6686a55c43cb0`. The measurement source is clean
commit `70d41f028951f87c595353ba33c37debb8847afc`. It does not alter PR #65,
its checkpoints, its feature extraction, or its training inputs. No PR is
opened until coordinator review. This is a separate split and distinct commits.

The bounded generator creates exactly 300 `development_paraphrase` tasks,
100 per kind, using nine fixed new phrasings. Every kind has template counts
34/33/33. New names are Marlow, Nacre, Oriole, Peregrine, Quill, Rill; price
cents range 30001–45000 and warranty months range 61–84. Those domains and
the phrasings are disjoint from development_train/development_validation.
The comparison phrasings use contrast, rank, and stack up. The explanation
phrasings request evidence, a reasoning trace, or catalog justification.
Field wording stays literal, preserving the existing engineered grounding
domain and focusing this check on intent wording.

Only the initial request, with empty program/proposal prefixes, is stored in
model_inputs. Gold targets and reference programs remain under training_only.
Task text contains none of the literal read/explain/compare operation labels.
The runtime presents all legal operation choices as before, selects its own
prefixes, and passes through the actual Decider, planner, kernel and renderer.
Scoring uses gold only after the complete free-running rollout.

All tasks select read. Operation accuracy is 100/300, and requested fact
selection is 200/200. Required comparison left/right/relation slots are each
0/100, with zero reached: they remain in their denominators. Unexpected actual
fact slots on comparisons are retained in raw observations, alongside all
requests, responses, raw logits, feature contributions, programs and text.

| Kind | Template 0 | Template 1 | Template 2 |
| --- | --- | --- | --- |
| Read | 34/34 | 33/33 | 33/33 |
| Explanation | 0/34 | 0/33 | 0/33 |
| Comparison | 0/34 | 0/33 | 0/33 |

These counts apply to each of the three seeds and to each recorded task-level
correctness definition. Median task-correct count is 100; range 100–100.
The pooled task-cluster accuracy is 33.333%, with percentile-bootstrap 95%
interval [28.000%, 38.667%]. The bootstrap resamples 300 task IDs 10,000 times
with NumPy seed 1729, averaging model seeds inside a task first and sharing
draws across the seeds. It does not treat 900 task/seed rows as independent.
All by-template bootstrap intervals are saved: [1,1] for the three reads and
[0,0] for the six failures. These boundary intervals are descriptive empirical
resampling results, not finite-sample guarantees about broader phrasing
populations. Different phrasings use different evidence worlds, so the receipt
does not estimate a paired causal effect of wording or a baseline delta.

The three frozen model SHA-256 hashes are:

```text
17 fb7c5b1e64111291a10d0733dce85856ca3ec9cd30e886ab499c732b5f34649d
29 59ed7ff5d4906b028df50a99613053b4abd2dda0b526712e82b285289f22ddd0
43 1b84914605f97167e330306957a55df030577387c7a3f4e6abd092434303ef22
```

Task bytes SHA-256:
`5379b4206f5a872146d5ac88b8d4b17032d993f1c542160d13a458299523687b`.
The receipt contains source-file hashes, compressed and decompressed artifact
hashes, exact command, software/hardware identity, task manifest, per-seed,
per-template and per-slot counts, bootstrap method and timing. The checkpoint
files are verified before and after evaluation, and in-memory weights are
checked unchanged. Compressed artifacts contain all lossless raw evidence;
hashes explicitly distinguish compressed bytes from decompressed payloads.

Exact commands from the worktree:

```powershell
python -B -m unittest language_growth.tests.test_paraphrase_data -v
python -B -m language_growth.frozen_wording_eval --output receipts/2026-10-06-L-frozen-wording-eval --task-seed 161803
```

Five focused tests pass: deterministic balance, reference and target bindings,
label/prefix separation, disjoint development domains, and frozen actual
rollout with fitting forbidden. The output directory must be new; choose a
fresh directory when reproducing rather than overwriting preserved artifacts.
Total measurement time is 1.715 seconds. Per-seed runtime median is 0.423
seconds, range 0.399–0.428 seconds. The hard CPU ceiling is 300 seconds.
Two CPU threads, NumPy only; GPU unused, billed rate and incremental cost $0.

No fitting, tuning, feature changes, external model, checkpoint download,
hidden inputs or Decision Index access occurred. This is synthetic development,
not Index or official held-out evidence. The wording set is now observed and
is not eligible as untouched confirmatory data for any future adaptation.
These results falsify transfer to these nine phrasings; the winner's earlier
300/300 result remains limited to its reused development wording. They do not
measure learned raw arithmetic, shared-parent retention, causal mediation or
L0 qualification. No corrective adaptation or additional variant was run.

Independent internal review and recomputation are recorded separately before
the branch is offered for coordinator review.

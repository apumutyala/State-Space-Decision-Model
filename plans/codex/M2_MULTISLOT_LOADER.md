# Issue #178: development-only M2 multislot loader

Base: `origin/agent-main@a6b0755a3b81de7948edd71425e663a52fe9557d`.
Branch: `codex/m2-multislot-loader-draft`. Cross-track proposal; no launch or promotion authorization.

Accept nested development rows shaped as `{id: task_id, group_id: world_id, request: {model, state, questions,
ssdm?}, labels: {qid: option}}`. Preserve the nested request and envelope metadata verbatim. A nested one-question
row receives storage key `task_id/qid`; its record id stays `task_id`, so the existing Item builder produces
`Item.id=task_id/qid` and record calibration groups every task's slots together. Complete multi-question records
retain their record-id storage key. Reject storage-key collisions, repeated task/question pairs, invalid gold
options, missing or mismatched labels/questions, repeated JSON members, ambiguous flat/nested envelopes and
inconsistent task/world identities before splitting or tokenization. Whenever any input row is nested or grouped,
all rows must have slash-free question ids and globally unique expanded Item ids, because existing calibration uses
the final slash to recover task identity. Pure legacy files retain their historical handling of question ids.

In development mode (neither `train_set` nor `cohort_sha256`), split complete worlds when `group_id` is present and
complete tasks otherwise, then expand all selected storage keys. Configured `n_train_records`/`n_eval_records`
count world/task split units on the new path, not slots; this can yield many more Item rows than those counts.
Tagged world/task identities prevent collisions in mixed inputs. All-ungrouped multislot draws retain the existing
sha256(seed:task_id) ranking. Ordinary legacy records retain the exact request conversion, dictionary keys, split
order and `{"cohort": null}` identity.

Protected cohort/evaluation inputs reject nested/grouped shapes before a row dictionary is constructed. External
train sets receive a separate linear streaming shape/unique-ID preflight over the pointer-resolved training file,
including undrawn rows, before `draw` can construct its dictionary. The preflight stores only seen ids. Existing
pointer, manifest, normalized-content, source exclusion and held-out checks in `train.train_set.draw` remain
mandatory. Its default `to_train_record` conversion also refuses nested/grouped records. Group-aware protected
manifests and content/cohort guards are explicitly unsupported pending separate review.

No K/T caps, estimator, model/training configuration, interface/checker, public route or training algorithm changes.
This work establishes ingestion and split behavior only; it establishes no learned capability, qualification,
calibration, safety floor or causal result. External-set preflight overhead is linear and was not benchmarked on
private data. Incident L-40 records an accidentally broad verification command; only the documented gated command
supports the final no-model test result. Receipt: `receipts/2026-10-07-D-m2-multislot-loader/README.md`.

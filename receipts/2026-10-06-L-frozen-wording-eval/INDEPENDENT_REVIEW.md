# Independent lean review: approve

A fresh-context GPT-6.1 Sol reviewer with high reasoning reviewed the query
semantics, generator, frozen evaluator, focused tests and preserved raw evidence
at source HEAD `70d41f028951f87c595353ba33c37debb8847afc`. The reviewer did not
write the implementation. Verdict: approve, with no blocking correctness,
contamination or spend findings. No training, model inference or further
evaluation variants were performed by the reviewer.

The cohort jointly shifts wording, entity names and numeric ranges. It diagnoses
combined distribution-shift failure; it does not isolate a causal wording effect.
The synthetic-development scope is mandatory.

Independently reconstructed raw results for each seed 17/29/43:

- Semantic/exact-program/exact-text each 100/300.
- Read 100/100; explanation 0/100; comparison 0/100.
- Checked accepted programs 300/300; all operation choices op001 (read).
- 600 actual provider calls, with later prefixes equal to actual prior choices.
- Pooled task-cluster accuracy 0.3333333333333333, percentile bootstrap 95%
  interval [0.28, 0.38666666666666666], 10,000 draws and seed 1729.
- Recorded total elapsed 1.715 seconds, billed/incremental cost $0.

The reviewer verified compressed/decompressed raw-artifact hashes, frozen
checkpoint hashes, clean dependency equality for feature/backend/planner/kernel/
renderer code, and all five focused tests. Model hashes are the PR #65 identities
listed in README.md and receipt.json. The evaluator also checks those file hashes
both before and after scoring, and checks in-memory model weights unchanged.

The independent raw-only reconstruction is preserved as
`independent_recompute.py`, a compact equivalent of the command the reviewer
executed. Run from the worktree root without fitting or model inference:

```powershell
python -B -c "import runpy; runpy.run_path('receipts/2026-10-06-L-frozen-wording-eval/independent_recompute.py', run_name='__main__')"
```

Receipt SHA-256 checked by the implementation agent after review:
`71658018f1c44304471a1aea6b9756ca79b1f134905cce09c2be4d0a99079cd2`.
The reviewer did not separately capture that receipt-file hash; its independent
artifact checks use the recorded compressed/decompressed prediction hashes.

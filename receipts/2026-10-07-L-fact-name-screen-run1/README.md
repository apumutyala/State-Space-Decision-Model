# Incomplete L0h0 CPU screen, run1

PR #120 source `96f5bd0c7298690d90e36918d5ca73cb302d447a` stopped safely at
`memory_guard` after 74.063 seconds. Cost: **$0**. Status: **incomplete**.
There were 12 requests and 11 completed responses, all in direct-before,
and zero efficacy worlds. These partial calls are not efficacy evidence.
No complete result or state-after hash exists.

The exact command, source/base commits, hardware, registration/data hashes,
raw hashes and terminal event are recorded in `receipt.json`. Original run
files were copied byte-for-byte from the PR #120 worktree; that worktree was
not edited. Initial available RAM was 7,920,824,320 bytes; the stop minimum
was 3,202,723,840 bytes, below the 3 GiB runtime floor. This telemetry does
not isolate model RSS.

The unchanged 192-request/384-forward protocol projected 1,255.5 seconds
before longer fact prompts and final checks, above its 1,200-second cap.
Reducing max_batch_rows to 1 also projected over cap. The follow-up removes
only the 64 direct inference bookends, keeps the paired initial-operation
equality and full-state hashes, and raises the admission reserve to 9 GiB.
No retry or parent execution was performed for this follow-up.

The revised projection is **879.78 seconds**, using separate operation/fact
request costs; exact inputs, formulas and proxy assumptions are in receipt.json.
This corrects the earlier 837.0-second request-count scaling and remains an
unmeasured projection before longer fact prompts and final checks.

# Prepared follow-up to the incomplete L0h0 screen

Follow-up PR #122. Base: PR #120 final head `96f5bd0c7298690d90e36918d5ca73cb302d447a`.
Preparation only; no parent/checkpoint load, inference, GPU, Index/heldout,
paid compute or retry. Cost: $0.

The registration explicitly removes the 64 direct inference bookends and
requires exact initial-operation request/decision equality for all 32 members
across arms, plus the unchanged pre/post full-parent state hash. The cohort,
primary paired estimate, boundary fixtures and statistical gates are fixed.
The admission reserve is 9 GiB; the runtime floor remains 3 GiB and the
owned-child cap remains 1,200 seconds. Completion feasibility is unmeasured.

No-load preparation:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_fact_name_screen --dry-run --out receipts/2026-10-07-L-fact-name-screen-retry-prepared
```

At the original PR #122 preparation head, focused verification was **33 passed**
with scripted providers/tiny random CPU tensors; no parent weights were loaded.
No CI workflow is configured. Independent cross-review is pending.

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```

Launcher exception regression checks simulate spawn, partial-spawn, handshake
and supervisor failures with runtime imports/checkpoint reads rejected. Every
case records an incomplete receipt, restores the environment, closes pipes,
and kills/joins only its owned started child.

A no-runtime simulated parent-side KeyboardInterrupt records
`launcher_interrupted` after owned-child cleanup and returns interrupt status 130.

## Registration refresh after the L1 data integration

After the L1 fixture/trainer code merged to `agent-main`, the broad source-tree
registration correctly refused the old source list. The registration was
recomputed on the merged L0h0 runner with the existing fixed seed, cohort,
arms, thresholds, state checks and 9 GiB reserve unchanged. One existing source
(`language_growth/training_records.py`) changed; 13 new L1 Python files were
added under the hashed source tree. The visible input SHA remains
`7dcd53525fca971737fcd4e60a1b7ae9157977709dabd83be74e05de1e3736d6` and the
evaluator-target SHA remains
`762274823fb2bf561419cf2c8435e9459dda27d7eba641f9c5b656bfed4065b5`.

The normalized LF registration bytes are SHA-256
`6442e00c53ab2dd51e583e4cf7af2b500d30cd1734f9f5df0e54d7baeb6e84f9`; the
previous prepared registration was
`a33e1e6ba1834732471acc067d6c4486d9e85316c0bbd3f26fb577e9dd8f2343`. The
refresh receipt is `registration-refresh.json`. The ordinary `--dry-run` path
then passed on a fresh output directory and reproduced the pinned visible
inputs and evaluator targets. This was no-load, CPU-only preparation; no model
was run, no parent was loaded, and no efficacy information was collected.

Post-integration verification reran the no-load runner and relevant repository
checks on the rebased PR #122 stack:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py tests/test_jobs_lint.py tests/test_no_kill_by_name.py
```

Result: **106 passed** in 20.13 seconds; `git diff --check` passed after the
prepared README's CR-at-EOL preservation was made explicit in `.gitattributes`.
No parent/checkpoint, model inference, GPU, Index/held-out data, or spend was
used.

## Source-registration refresh for public task-relevance work

The L0 public-output relevance gate added `task_relevance.py` and updated
`public_output.py`; the two frozen no-load screen registrations hash the full
Python source trees, including tests. Their source maps were refreshed while
holding the serialized screen inputs, evaluator targets, corpus, configuration,
thresholds and arm definitions byte-for-byte constant. The exact old/new
registration hashes and unchanged-field comparison are recorded in
`receipts/2026-10-07-L-public-task-relevance/registration-refresh.json`.
This refresh only keeps the existing no-load dry-run guards truthful; neither
screen was launched and no model or checkpoint was loaded.

## Prepared exact initial-operation score reuse

The current registration adds run-local exact initial-operation score reuse,
with all logical calls preserved. The operation equality check is now exact
request plus deterministic cached-score replay, weaker than two independent
forwards; equality on hits is by construction. No retry has run. Full details,
prior registration bytes, projection inputs and 115 passing checks are in
`receipts/2026-10-07-L-fact-name-exact-cache-prepared/`. All cohort/statistical
gates and 9 GiB/3 GiB/1200-second guards are unchanged. Compare paths may
still exceed the cap; independent review and execution clearance remain.

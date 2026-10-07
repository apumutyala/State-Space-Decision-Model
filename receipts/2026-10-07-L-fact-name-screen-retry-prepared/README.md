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

Focused verification: **33 passed** with scripted providers/tiny random CPU
tensors; no parent weights loaded. `git diff --check` passed. No CI workflow
is configured. Independent cross-review is pending.

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```

Launcher exception regression checks simulate spawn, partial-spawn, handshake
and supervisor failures with runtime imports/checkpoint reads rejected. Every
case records an incomplete receipt, restores the environment, closes pipes,
and kills/joins only its owned started child.

A no-runtime simulated parent-side KeyboardInterrupt records
`launcher_interrupted` after owned-child cleanup and returns interrupt status 130.

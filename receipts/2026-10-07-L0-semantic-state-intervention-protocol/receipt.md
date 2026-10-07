# L0 causal state-intervention protocol — CPU contract receipt

## Scope

This is a no-checkpoint contract and copy-isolation test using a tiny fake RWKV-7-compatible cache. It establishes no causal semantic fidelity, learned capability, alignment, or safe-serving claim. The wrapper emits raw research logits only; all evaluated answers must still be produced by the normal planner, checker and renderer.

## Source and command

- Base: `agent-main@d7f12909ee7158a65d09b310c1b0a84e11d6eaec`.
- Command: `C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_semantic_state_intervention.py tests/test_jobs_lint.py`.
- Result at initial review: **76 passed** (9 focused state-protocol tests +67 repository lint tests).
- Rechecked after integration of the L1 data PRs on current `agent-main`: **79 passed** (same 9 focused tests +70 repository lint tests).
- `git diff --check`: passed.
- Runtime container reference: installed `transformers/cache_utils.py` initializes the three recurrent slots and their flags as dictionaries keyed by 0, 1 and 2. The test fake uses the same keyed shape; the wrapper canonicalizes by numeric key order and rejects extra/missing keys. This review correction is recorded as L-32 with its enforcing fake-cache test.

## What the tests establish

- Sham replay from one capture is deterministic and preserves the source state.
- Recurrent prefill+continuation logits match one direct prefix-plus-suffix fake-model pass exactly.
- One registered edit changes only its allowlisted recurrent slot, changes the fake model's later logits, and leaves the original state and fake parent weight unchanged.
- Unknown edit IDs, changed registration metadata/bytes, invalid sign, late registration after the first capture, wrong context/parent identity, model/config drift, cache config/flag drift, shape mismatch, non-finite direction, modified capture state and modified prefix identity fail closed.
- These are fake-cache API contract checks; no real recurrence effect, candidate distribution or paired confidence interval was measured.

## Resources and limits

- Local Windows CPU; no GPU, model/checkpoint load, inference on a pretrained parent, training, external provider, Index, held-out data or paid compute.
- No training/evaluation cohort, seed or raw model predictions are applicable. Fake test directions use synthetic constants only.
- The protocol binds exact caller-supplied prefix IDs but does not verify them against a `LayoutRow` or authenticate a schema → state boundary. It makes no query-independent memory claim; L0i-b must use the normal planner/kernel/renderer before any scored output.
- The laptop had 3.98 GiB available at the start of the turn, below the existing 6 GiB parent-run start guard. A real parent pilot remains unexecuted until a fresh guard passes.

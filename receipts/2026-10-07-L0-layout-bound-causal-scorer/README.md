# L0i-b layout-bound causal scorer: CPU contract receipt

## Scope

This receipt covers fake-cache/API contract checks only. It does not measure a trained parent, capability, alignment, interpretability, or causal semantic-intervention fidelity. No pretrained checkpoint was loaded. No GPU, Index, held-out set, network teacher, or paid compute was used.

## Run

- Base commit: `32d975159c64216193c47478e102d17c8d7e5658` (`origin/agent-main` at test time).
- Command: `py -3.12 -m pytest language_growth/tests/test_semantic_state_intervention.py language_growth/tests/test_causal_layout_backend.py language_growth/tests/test_planner.py -q`
- Result: **38 passed in 2.99 s**.
- Follow-up static check: `git diff --check` passed.
- Runtime: Python 3.12.10; PyTorch 2.13.0+cpu; CUDA unavailable (`torch.cuda.is_available() == False`).
- Hardware: AMD Ryzen AI 9 HX 375 w/ Radeon 890M, local CPU. Billed cost: $0. There were no model-performance runs, so seed, repeat spread and raw predictions are not applicable.
- Inputs: tiny locally authored synthetic host/catalog fixtures plus the existing synthetic planner fixtures. No Index or held-out rows.

## Tested source hashes

Hashes bind the files in the tested working tree; the implementation and test files are included in this PR.

| File | SHA-256 |
|---|---|
| `language_growth/semantic_state_intervention.py` | `47a22e6a2b016d987619015510bb90ac5d444fc5b29cb42a159a4db5e4fd2e70` |
| `language_growth/causal_layout_backend.py` | `2caee529790a2617eeac3251f436e8e857a2135d8843875c94468f8c449e3f59` |
| `language_growth/tests/test_semantic_state_intervention.py` | `89222bdd2b59a5ef5df213fae3f77dd02aea0dfd105a7ce8dc3650f3a853a0ca` |
| `language_growth/tests/test_causal_layout_backend.py` | `3089387a6c6b7f61da8089cc2bbf19406d3f591ec6f9d9adade80712a758604d2` |
| `language_growth/tests/test_planner.py` | `9051f7067a72473d520e62a1f4e0148192e7f54599c4932048fa565328e404cd` |
| `plans/codex/L0I_B_LAYOUT_BOUND_CAUSAL_SCORER.md` | `9a62e089698c8b191d877c3d379a976d770c2ae569ece91acb368d35af99a5ff` |

## Result interpretation

The tests establish split-vs-uninterrupted option-score equivalence on a fake RWKV cache, correct first-token indexing, option coverage/order, request/context/layout/direction identity checks, the 2048-token complete-context refusal before prefill, call and token accounting, and checked Decider-to-renderer integration. They do not establish that a learned direction tracks semantics or that interventions improve any task. That requires a preregistered parent-backed experiment with sham, random-direction and output-only controls after the trained parent/calibration handoff and spend gates are satisfied.

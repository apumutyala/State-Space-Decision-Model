# L0i-c relation-scaffold runner: CPU contract receipt

## Scope

This preparation proves the fixed discovery input selection, scripted operation/operand scaffolding, unchanged parent relation-response boundary, fail-closed failure behavior, and resource/identity gates. It does **not** run a model or establish natural relation capability, causal semantic fidelity, L0 promotion, alignment, or interpretability. The test parent is a deterministic fake that sees permitted state; its choices are protocol checks, not model predictions.

## Checks

- Base commit: `32d975159c64216193c47478e102d17c8d7e5658` (`origin/agent-main` at test time).
- Command: `py -3.12 -m pytest language_growth/tests/test_causal_relation_screen.py language_growth/tests/test_planner.py language_growth/tests/test_semantic_axis_fixtures.py -q`
- Result: **38 passed in 8.61 s**; `py_compile` and `git diff --check` passed.
- Runtime: Python 3.12.10, local CPU (AMD Ryzen AI 9 HX 375 w/ Radeon 890M). No checkpoint/model inference, GPU, Index, validation split or paid compute.
- Input file: `ssdm-data/semantic-axis-recurring-names-a8277d8f798b/semantic_axis_discovery-inputs.jsonl`; SHA-256 `5bb73a63c7f6ed085b850e8680576ac4670be5d74c36e5da9af188c1d8f7065f`. The runner refuses every other hash and accepts input records only, never scorer files.
- Fixed evaluator-only offsets: `[1,2,5,7,8,11,13,14,17]`. Offsets and task identities are receipt metadata, never inserted into model state.
- Implementation dependency: PR #142 (layout-bound sham scorer and L0i-c phase contract; current head `515cb15`). Real execution remains disabled until that code is reviewed/merged and a frozen parent revision/calibration bundle is qualified and pinned.

## Source hashes

| File | SHA-256 |
|---|---|
| `language_growth/causal_relation_screen.py` | `5d42f8fac9cabaecf5fea100de8401845e1672756e533c014a022ba61a87902d` |
| `language_growth/tests/test_causal_relation_screen.py` | `c89b49dc57d6d8b3e34e427a141c482f6149b9299d42d0cd9b6518c78ad1b429` |
| `plans/codex/L0I_C_NATURAL_RELATION_CONTRAST.md` | `3195d7042012156d22199c2ff3d67aee56cee2e3c6cd2e5742793692c739bbd9` (introduced/updated by PR #142) |

## Interpretation

The fake-parent tests demonstrate that a wrong relation remains wrong and is rejected without text, parent errors stay errors, exact relation requests and raw/served parent responses are preserved, and scripted operation/operand choices stay invariant as scalar values vary. They do not demonstrate a learned parent's natural relation signal. Real execution requires the exact parent revision and calibration pins, the unchanged sham layout-bound backend, a 9 GiB start reserve and 3 GiB runtime floor. The runner returns an incomplete receipt before parent construction or relation inference when those gates fail.

# Parent checkpoint load exactness: CPU receipt

## Scope

This receipt covers fail-closed validation of current Trainer checkpoints and unchanged valid-format CPU resume behavior. It does not validate or load a real Track D parent checkpoint, train a model, run GPU, or establish capability, safety or causal-fidelity evidence.

## Validation

- Base: `32d975159c64216193c47478e102d17c8d7e5658` (`origin/agent-main` at test time).
- Command: `py -3.12 -m pytest tests/test_m2_codex_fixes.py tests/test_m2_train.py tests/controlsafe/test_g16_train.py::test_resume_equals_uninterrupted_and_lineage_config_are_checked language_growth/tests/test_rwkv_decision_head.py -q`
- Result: **53 passed in 59.61 s**.
- Additional adversarial CPU review: three focused corruption/valid-resume/interrupted-resume checks passed under Python with Torch 2.8.0+cu128 while CUDA was disabled; source and test hashes are identified below.
- Runtime for the coordinator run: Python 3.12.10, PyTorch 2.13.0+cpu, CUDA unavailable, AMD Ryzen AI 9 HX 375 w/ Radeon 890M. $0. No external dataset or held-out examples; synthetic toy Items and tiny random models only.

## Tested source hashes

| File | SHA-256 |
|---|---|
| `train/trainer.py` | `3d2e95839bb9209034d34d49ebaf1dad822428d2172482e088b83ef0530aafe8` |
| `tests/test_m2_codex_fixes.py` | `6d1ee52020e51f22e8d60cf6f304816be6e417b48b2f2661965633e3cd88bcab` |
| `plans/codex/L0I_PARENT_CHECKPOINT_LOAD_EXACTNESS.md` | `25d345e887754a4a70be41bf11587e2efb64a566289e15c1a635606ad3573bc4` |

## Interpretation and limits

Rejected missing/unexpected/frozen trainable keys, broadcastable or otherwise wrong shapes, wrong dtype, non-tensors, nonfinite weights, optimizer topology/partial state, scheduler structure and malformed RNG payloads are detected before mutation. Tests compare all parameters, optimizer/scheduler state, counters/history and Python/NumPy/Torch CPU RNG after rejection. This does not claim rollback from an unexpected device error or OOM during the validated apply phase. The CUDA RNG payload preflight path was not exercised on hardware. PR #141 contributes append-only LESSONS L-33; this PR adds L-34 and should integrate after #141.

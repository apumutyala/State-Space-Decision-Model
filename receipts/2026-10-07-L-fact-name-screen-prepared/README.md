# Prepared paired fact-name CPU runner

Prepared only on candidate-style base `986a70b678a130ef00b57e5173d81e95c5813839`.
No real checkpoint/model load, inference, training, GPU, Index/heldout or spend.
The registration freezes source/software, the exact parent file identities and
the 16 paired worlds. Visible inputs and evaluator targets occupy separate files.

No-load preparation command, using the existing interpreter:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_fact_name_screen --dry-run --out receipts/2026-10-07-L-fact-name-screen-prepared
```

Focused verification: **18 passed in 5.64s**:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```

Checks use scripted providers and tiny random CPU tensors. Subprocess preflight
tests reject Torch/Transformers/Safetensors imports and checkpoint/snapshot reads.
The prepared runner uses existing owned-child, offline CPU, 6 GiB start / 3 GiB
runtime and 1200-second cap helpers. These checks establish implementation
behavior, not model completion/performance. Independent review precedes execution.

The continuation gate matches `plans/codex/L0H0_PARENT_BINDING_SCREEN.md`, including
at least two added both-member-successful worlds, zero previously successful
world losses, exact fact-binding improvement, zero operation regression and a
strictly positive paired useful-answer CI lower bound. No automatic adoption.

# Head-fit diagnostic preparation

Prepared only. No real checkpoint/model was loaded or evaluated. The dry run
used the existing frozen corpus and retained prior JSON evidence; it created
no data, predictions, optimizer or new model checkpoint.

Source: `language_growth/rwkv_head_fit_audit.py`; preregistered interpretation and
commands: `plans/codex/L0_RWKV_HEAD_FIT_AUDIT.md`. The registration pins 96 rows
(64 train + 32 validation), the existing final checkpoint, original JSON inputs,
installed software and the executable Python/plan closure. The plan uses LF
checkout bytes because its source hash is pinned alongside executable code.

Dry-run output: `C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-fit-prepared-20261006`.
Run with the existing Python runtime from this worktree:

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I -X utf8 language_growth/rwkv_head_fit_audit.py --dry-run --out '<fresh-preparation-directory>'
```

Tests: **90 passed in 13.84s**, with CUDA hidden, all Hugging Face offline flags
set, and OMP/MKL/OpenBLAS limited to two threads:

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -m pytest language_growth/tests/test_rwkv_head_fit_audit.py language_growth/tests/test_rwkv_head_learning.py language_growth/tests/test_rwkv_decision_head.py language_growth/tests/test_rwkv7_cpu_parent.py language_growth/tests/test_rwkv_cpu_encoder_identity.py language_growth/tests/test_training_records.py -q
```

Tests use mock scoring or tiny randomly initialized native CPU models and
locally generated synthetic test checkpoints. The default-preparation subprocess
blocks Torch/Transformers/Safetensors imports and all checkpoint/snapshot opens.
Coverage includes exact one-pass counts, current-label separation, prior-gold
prefix transparency, normalized unrounded CE, exceptions/nonfinite/missing
options, failure denominators, strict synthetic restoration and lineage,
no optimizer construction, immutable existing output, low-RAM refusal, and the
shared owned-child timeout/memory/IPC regression tests.

Internal source review checked restoration against the prior Trainer lineage
format without constructing a Trainer; raw scores come from the same native
backend used by the original rollout. The final registration was checked against
fresh preparation and the exact Git-index source bytes. No LESSONS entry was
added: implementation/testing surfaced no campaign incident requiring a new
standing rule. Independent review and real execution remain separate.

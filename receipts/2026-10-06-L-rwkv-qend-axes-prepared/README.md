# Matched q_end supervised-port preparation

Prepared only, from `0200ee29d0d7a7987a9128b629f32fa298d30b41` on
`stack/rwkv7/head-qend-axes`. Both default dry-runs completed without importing
Torch/model runtimes or opening a parent/checkpoint. No real model load, fit,
GPU, held-out data, spend or PR occurred.

The paired immutable pins are `registration-control.json` and
`registration-axes.json`. Fresh preparation and execution must exactly match
the relevant committed pin. Corpus, slot-specific targets, source closure,
software and references are frozen. The arms share every recipe setting,
initialization, seed and data/order; only auxiliary MSE weight is 0 versus 1.
Each arm uses exactly 32 updates at batch 2: one pass over 64 training slots.

The [preregistered L0 phase plan](../../plans/codex/L0_RWKV_QEND_AXES.md)
defines the primary checked useful-answer/acceptance comparison, per-slot fit,
fixed clamp/random/output-only controls and interpretation limits. Targets are
offline program bindings; no option string/ID parser or inference metadata is
added. The port is explicitly supervised, not emergent or backbone-causal.
Output-only affine replay is equal to clamp readout by construction and is
reported as that limitation.

**Claude cross-review is required** for the additive D-owned changes:
`heads/heads.py`, `train/data.py`, `train/trainer.py`, `train/lineage.py`.
They add an optional output port, training-only Item sidecar, opt-in MSE loss
and target-lineage binding. Default config/item identities are unchanged.
No request, HeadInput, served-answer or semantic-program interface changes.

Screening requires at least 52/64 train slots correct, complete arm receipts,
a strict useful-answer gain and no read-task regression. Passing this screen
never automatically adopts/promotes a model. Axis MSE/interventions alone
never count as capability. This is one seed on 12 reused development worlds;
no L0 qualification or trained-D-parent retention claim is supported.

Verification: 13 targeted head/runner tests passed in 16.13s; 133 additive-head,
trainer, native-bridge and repository-lint compatibility tests passed in 71.48s.
Launch lint/diff checks passed with existing legacy waivers. `verification.json`
records the exact pin/source/data hashes and commands. Tests include factor
bindings/right operands/equality, sidecar ownership/lineage, lambda0 exact CE,
matched initialization/one-pass order, permutation/relabeling for fixed
embeddings, no unchecked output, port controls and default pin equality.

Prepared commands (completed):

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_qend_axes --arm control --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-control-prepared-20261006
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_qend_axes --arm axes --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-axes-prepared-20261006
```

Only after independent and Claude cross-review plus execution approval, run
these fixed commands serially without inspecting control metrics between arms:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_qend_axes --arm control --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-control-run1-20261006
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_qend_axes --arm axes --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-axes-run1-20261006
```

The launcher requires 12 GiB start RAM, monitors 3 GiB running RAM and a
1200-second cap, forces offline CPU/two threads and owns only its authenticated
child. Outputs must be fresh; no automatic retry. Only status/resources may be
checked between arms. Inspect paired metrics after both finish; JSON-only
`compare_arms` refuses incomplete or mismatched arm receipts and never selects
an optimizer/checkpoint or adopts a model.

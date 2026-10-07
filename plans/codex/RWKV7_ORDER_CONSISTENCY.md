# RWKV7 option-content permutation consistency (CPU implementation phase)

Registered 2026-10-07 before any model-output measurement. This is a separate training arm from the existing C1 readout and stock RWKV7 baseline. It uses the opt-in reversed training views in `train/order_views.py`; the existing production and cell paths remain unchanged.

## Hypothesis

The stock RWKV7 parent shows substantial option-order sensitivity. Supervised training on a deterministic second view of the same choice question, with descriptions moved across option IDs and the gold ID moved with its description, plus a content-aligned Jensen–Shannon consistency penalty, may reduce sensitivity while retaining decision quality.

The consistency term compares distributions over the same option content, not the same slot ID. `view_perm[j]` means view option `j` carries the content of base option `view_perm[j]`; the loss must invert this mapping before comparing probabilities. Training views are allowed only when the data builder proves the permutation preserves task meaning. No development or test example may receive a view.

## Fixed experiment design

- Objective per original training row: base supervised decision loss plus the eligible view's supervised loss and `0.25 × JS(base || content-aligned view)` when that row has a view. Average these per-base contributions; no coefficient search or DEV-tuned schedule.
- JS is the symmetric Jensen–Shannon divergence in nats over each row's valid options; padded options are excluded. A default weight of zero preserves all existing training runs and config hashes.
- The view itself receives the existing supervised loss with its gold ID moved with its option text; unpaired items contribute only their base observation. In each micro-batch, extra view and JS means are multiplied by `paired_count / base_count` so row-batched accumulation preserves the same per-original-row objective.
- This arm requires KD off until teacher distributions for content-permuted views have an independently checked mapping, and `axis_loss_weight=0` until view-axis target semantics are registered.
- The measured T ladder must account for the longer of base/view rows; row-batch token budgeting conservatively accounts for both forwards. Views longer than the selected `T_max` are dropped while their valid base rows remain, and gate reconstruction applies the same filter.
- Matched comparison: same RWKV7 parent, data split, rows, seed, training steps, optimiser, calibration arm and evaluation set; the control receives the base supervised loss without views/consistency. Any extra training compute is reported.

## Exit proof for this phase

CPU unit tests prove permutation alignment, finite gradients, symmetry, the `[0, ln(2)]` bound, padding exclusion, malformed-map rejection, disabled-default hash compatibility and training/eval isolation. No checkpoint is loaded and no GPU/paid run is authorized by this implementation phase.

## Decision unlocked

After data-view eligibility and this loss integration are independently reviewed, prepare a separately preregistered training cell and spend request. Promote only if paired DEV evidence improves order robustness and does not regress primary decision score, calibration, safety, control or interpretability gates. No result or gain is claimed by this code phase.

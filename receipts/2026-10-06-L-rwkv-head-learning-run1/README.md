# RWKV7 head-only free-running diagnostic — run 1

This single-seed, synthetic development run tested whether 24 supervised updates to a typed decision head improve complete answers through a frozen RWKV7-0.1B parent. It is exploratory only: it is not an L0 gate, a strong-comparator result, a retention test for the adapted head, or evidence of broad semantic interpretability.

## Result

Useful free-running tasks **fell from 2/12 to 1/12**. By task kind, read stayed at 1/4, explain fell from 1/4 to 0/4, and compare stayed at 0/4. The preregistered paired percentile bootstrap over 12 task worlds (seed 9203, 2,000 resamples) gives **−8.33 percentage points**, 95% interval **[−33.33, +16.67]**. There was one improvement, two regressions, and nine unchanged worlds. The interval is wide and includes zero; this run shows no useful-answer improvement.

Kernel acceptance fell from 12/12 to 10/12. After training, the operation slot matched 6/12 references (read 4/4, explain 0/4, compare 2/4). On the two executed comparisons, the right operand was correct 2/2, while the left operand and relation were each correct 0/2. The other two comparison cases chose the wrong operation, so their comparison slots were never executed. These slot counts diagnose this recipe; they are secondary to complete-answer accuracy.

The independent LM-readout baseline was unchanged, including all served probabilities and revision identifiers. It answered 0/4 direct validation comparisons both before and after: all four targets were `less_than`, while it selected `greater_than`. That equality establishes path parity only; it does not establish useful baseline behavior or retention of the adapted head. The runner recorded full parent state hashes before and after, but the saved training checkpoints contain only head/typed trainable tensors and optimizer state, so the full-parent equality cannot be independently reconstructed from those checkpoint files.

The run made 24 updates × batch size 2 = 48 slot exposures, or 75% of the 64-slot training corpus; it did not complete one full epoch. Mean training losses over steps 1–8, 9–16, and 17–24 were 64.323, 4.119, and 1.752. Fixed training-slot accuracy was not logged, so the current result cannot distinguish an incompletely fit mapping from transfer or free-running rollout failure.

This result does not falsify semantic-program learning in general. It falsifies the claim that this exact 24-step, 24-world recipe improved these 12 held-apart synthetic worlds. Before another training recipe, the next proposed $0 diagnostic is one teacher-forced pass over the saved head checkpoint on all 64 training slots and 32 validation slots. That would measure fit versus transfer without optimizer steps; it is proposed, not yet preregistered or run.

## Reproduction and scope

- Source commit: `d8b3be06b9f4e8f142d86c082ce9e69818a41d8f` on `stack/rwkv7/l0-head-only`.
- Frozen code SHA-256: `771dcc622e91ac12a6fdf5940a5cc23257bd22d36155f43940681161a0cc7a06` across 89 runtime Python sources. See the [prepared v2 registration](../2026-10-06-L-rwkv-head-learning-prepared-v2/registration.json) and [verification](../2026-10-06-L-rwkv-head-learning-prepared-v2/verification.json).
- Command, from `C:/Users/apumu/research/CHLabs/ssdm-stack-rwkv7-l0-head-only`:

  ```powershell
  & 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -X utf8 -m language_growth.rwkv_head_learning --execute --out 'C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-learning-run1-20261006'
  ```

- Data: 24 training worlds (seed 9201) and 12 disjoint validation worlds (seed 9202), balanced over read/explain/compare. Canonical corpus SHA-256 `cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`; train split `4dfd05d495cdcc5bf38c0db857ac31fbd6367fcf96c2df64d8791ad77978ceaf`; validation split `e8847f275c4001d53c8d2a4bb88863faec4b0de1c0645dd9605cba7559f7d4cb`.
- Recipe: seed 20261006; 24 CE updates, batch size 2, learning rate 0.001; SnapshotFork head, 4 channels, 2 snapshots; no LoRA. Full config and loss are pinned in `registration.json`.
- Parent: `fla-hub/rwkv7-0.1B-g1`, snapshot `7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8`; weights SHA-256 `051b3684ffa896aa842522e5bbc2f355ac00783e12e373a8eb7e4907dd5ead70`.
- Environment: Windows 11, AMD64 Family 26 Model 36, CPU only, 2 Torch threads; Torch 2.8.0+cu128, Transformers 5.17.0, Safetensors 0.8.0, psutil 7.2.2. No GPU was used. Runtime 278.14 s; minimum available RAM 10,580,746,240 bytes; billed cost $0. N=1 training seed, so no between-seed spread is available.
- Independent raw remeasurement found no metric, task-attribution, or slot-attribution mismatch; the audit did not import Torch or rerun the model.

## Preserved artifacts

`raw.jsonl` contains the complete before/after requests and predictions. `before.json` and `after.json` contain the task-level checked outcomes; `parent-evidence.json` and `train-config.json` retain run inputs. `receipt.json` contains the detailed outputs and provenance.

| File | SHA-256 |
|---|---|
| `raw.jsonl` | `5c77c1ea91289d4ceb20765a6951f27a123ba883f37723aa1b4f1730ea7683ea` |
| `before.json` | `618c645ac7f2aa6c762bc3c62c84f707801e1e23541a0204f10c0fb25d60223a` |
| `after.json` | `6d51a349e38587d7f44e5e3ab150201ffa5afcfb7b09cf4ef6e66734294ad923` |
| `receipt.json` | `950f1bc202b55f01e79953d07505757368eceede2ab5c95939dcb4b3ae867ecd` |
| `registration.json` | `952c44a84e6140488179027d39e8a92297b355d91b31f3fde05d5975aa46dc15` |
| `parent-evidence.json` | `a06d205c82d7bea564d3fcb38df18bb2887e1eb1da3274ec02ef290d93b34e4c` |
| before head checkpoint (external `ssdm-data` run directory) | `7e2122ff5e2ad3f5f77b65b95379a0382006db43cdb055eadb09bd994d0a31d2` |
| after head checkpoint (external `ssdm-data` run directory) | `3a896a00dc06fdd79aa8656f5272fcd4307038633efb0e0d642773280affe13c` |

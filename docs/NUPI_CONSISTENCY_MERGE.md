# νPI + consistency merge

Merged `stack/qwen35/nupi` (`54d34df9`) into Keel (`21768082`) on
`collab/mottopanikeiku/keel-nupi`. CPU-only work; GPU spend **$0**. No packages added.

## Resolution

Seven conflict blocks in `train/trainer.py` overlapped config fields, controller setup, view helpers, step setup,
step completion, checkpoint state and restoration. Both `consistency` and `constraints` remain independently opt-in;
empty blocks are omitted from `to_json` and the config hash. Both view helpers, per-step updates and checkpoint
states are retained. The shared `PIController` implementation is unchanged.

The combined objective adds both penalties before the optimiser update, including the separate recall-probe
backward. `loss` and the collapse guard retain the pre-constraint objective (including surface-consistency KL);
`lagrangian` additionally includes the constraint and recall penalties. `cons_lambda` records the multiplier used
in the step, while constraint logs retain their existing next-step multiplier convention.

Two conflicts in `train/run_cell.py` were resolved by passing both config blocks and retaining recall-probe loading
alongside the existing dev-card and no-truncation checks. The WIP `jev3-sym-nu` trainer was consulted, not copied;
its decisiveness/warm-start changes are not part of this merge. Neither feature's standalone behavior was changed.

## CPU evidence (2026-10-15)

| Required file | Result |
|---|---:|
| `tests/test_nupi.py` | 15 passed |
| `tests/test_consistency.py` | 20 passed |
| `tests/test_order_views.py` | 57 passed |
| `tests/test_order_consistency_loss.py` | 18 passed |
| `tests/test_jev3_sym.py` | 12 passed |

All **122** named-suite cases passed, with no skips. This includes the unchanged νPI hexadecimal loss golden
(default, explicit empty constraints, and both blocks empty), and consistency's live bit-identity comparison against
parent `a5b4cb33` (losses, weights, config/hash, lineage and checkpoint keys; micro-batch and grouped-row paths).
New real-model CPU regressions cover additive gradients with partial views and `view_grad` on/off, both controllers
updating, simultaneous surface/order views, exact interrupted resume with monitor/family/recall state, and refusal
of missing constraint state. Config tests cover both features enabled and all empty-block combinations.

The broader grep-selected trainer suite ran **635 cases: 587 passed, 15 pre-existing failures, 33 skipped** in
435.71 s. Skips: 18 require transformers 5.17.0 (installed 5.14.1), 3 require the Decision Index kit, and 12 require
local tokenizer/data files. No skipped case is claimed as passing. The Decision Index document-overlap index was
not run; these are synthetic unit tests, not a real-data training or evaluation run.

Exact final command, from the worktree root (`SSDM_BASE_TREE` is a detached checkout of `a5b4cb33`):

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SSDM_DATA=/home/alp/ssdm-data SSDM_BASE_TREE=/home/alp/keel-wt/nupi-consistency-base \
  PATH=/home/alp/ssdm-venv/bin:$PATH /home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q -ra \
  tests/test_nupi.py tests/test_consistency.py tests/test_order_views.py \
  tests/test_order_consistency_loss.py tests/test_jev3_sym.py tests/test_anti_collapse.py \
  tests/test_cell_perf.py tests/test_chain.py tests/test_ckpt_export_size_guard.py \
  tests/test_fla_rwkv7.py tests/test_jev3_lm.py tests/test_m2_codex_fixes.py \
  tests/test_m2_first_cell.py tests/test_m2_review_fixes.py tests/test_m2_train.py \
  tests/test_perf_step_gate_rule.py tests/test_premortem_probes.py tests/test_q9af_act_ckpt.py \
  tests/test_q9af_cell.py tests/test_sizecliff.py tests/test_teacher.py \
  tests/controlsafe/test_g16_train.py tests/controlsafe/test_trust_gate.py \
  language_growth/tests/test_parent_training_integration.py language_growth/tests/test_qend_axes.py \
  language_growth/tests/test_qualified_parent.py language_growth/tests/test_rwkv_decision_head.py \
  --junitxml=/tmp/keel-nupi-final-pytest.xml
```

## Pre-existing failures (not touched)

Each of the same 15 failing node IDs was separately run in a detached pre-merge Keel worktree at
`21768082432bfb560a0a3025d9b648dc0659b2ed`: **15 failed** in 5.82 s, with the same causes.
JUnit evidence: `/tmp/keel-nupi-premerge-pytest.xml`; no unrelated tests or production paths were edited.

| Test (under `tests/`) | Cases | Cause on both trees |
|---|---:|---|
| `test_chain.py::test_runner_runs_the_guards_every_link_and_evals_only_in_the_final_one` | 1 | First `ARM=$CK/parent_arm` occurrence precedes the intermediate-link message; source-order assertion fails. |
| `test_fla_rwkv7.py::test_run_cell_trains_fla_strict_and_records_calls` | 1 | Counts two strict blocks, but the runner has four. |
| `test_m2_first_cell.py::test_runner_orders_guards_before_training_and_prices_every_eval_first` | 1 | Counts four weighted rate guards, but the runner has five. |
| `test_m2_first_cell.py::test_evals_exit_43_is_a_measurement_not_a_crash` | 8 | Parameters `0,40,43,44,45,41,10,2`: first checkpoint-export occurrence precedes the measured-exit case; shell exit behavior itself passes. |
| `test_m2_review_fixes.py::test_forward_and_backward_stay_on_the_backbone_device` | 2 | `ordered-fork`, `symmetric-fork`: boolean scalar extraction is unsupported on meta tensors. |
| Same device test | 2 | `ordered-lm_label`, `symmetric-lm_label`: `TinyRecurrentLM` lacks `get_input_embeddings`. |

These are merge/regression checks, not evidence of GPU parity, model quality gains, or hard constraint feasibility.

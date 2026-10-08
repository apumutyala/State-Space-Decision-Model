# Code-label family selection for Index and safety smoke: CPU integration only

The distinct family `rwkv7-hf-code-label-v1` now selects the existing, preregistered `G1CodeLabelBackend` in both `jobs.common.readout_engine.ReadoutIndexEngine` and `safety_eval.decide_bridge.build_hf_bridge`. `rwkv7-hf`, the default family, existing templates, model/plugin loader settings and job configurations remain unchanged. No model/checkpoint load or call, GPU, DEV/TEST access, spend or push was performed.

Branch/worktree: `stack/rwkv7/g1-code-label-engine`, `C:/Users/apumu/research/CHLabs/ssdm-wt/g1-code-label-engine`, stacked on filtered-code-label commit `cc7fdb1cd89f50727fcb56d708c867c34c1fc64a`. The hypothesis, filtered table and segment-preserving renderer are unchanged from that parent branch; their registrations and tokenizer proofs remain in `plans/codex/` and `receipts/2026-10-07-D-g1-filtered-codebook/`.

## Selection and identity

The new family reconstructs only the pinned vendored `Rwkv7Tokenizer`'s bound `decode` method from `<model_dir>/tokenizer.json`; no AutoTokenizer, remote code, alternate codebook or fallback is used. Vocabulary and reviewed tokenizer source hashes are checked before model/plugin loading, so missing or altered tokenizer artifacts stop before loading weights. The backend separately proves the actual plugin encoder/decoder behavior against every frozen code.

- Vocabulary SHA256: `1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf`.
- Reviewed tokenizer source SHA256: `58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65`.
- Filtered ordered code/token/source table SHA256: `7226adf44b9529726b2f44e66b60257c92a50126a3f30131caad6ab0be231466`.
- Backend template identity: `jrt-rwkv-g1-code-label-v1`, with the existing labelled-renderer canary/source binding. Index provenance records the actual backend revision. With identical fake plugin settings, Index and safety bridge construction return identical backend revisions.

The code-label backend keeps its registered one-next-token-forward-per-target-question scoring path and fixed call structure; the old `max_batch_rows` handling remains unchanged for stock families. No scoring, request formatting, calibration, policy conditioning, safety metric or error mapping was changed by this integration.

These future invocations now resolve to the code-label backend; they were **not run**:

```text
python -m decision_index run --engine jobs.common.readout_engine:ReadoutIndexEngine --option family=rwkv7-hf-code-label-v1 --option model_dir=<pinned-snapshot> --rows <registered-rows> --out <output>
python -m safety_eval.run_smoke --backend decide --family rwkv7-hf-code-label-v1 --model <pinned-snapshot> --out <output>
```

## Checks

`tests/test_code_label_family_selection.py` supplies a fake kit API, fake tokenizer with only `decode`, and patched plugin loader whose forward always fails if called. Its 18 construction/error tests verify new and stock selection, exact template/table/hash binding, CPU/device/dtype/default preservation, decoder-before-loader order, missing/bad pin refusal, unknown-family refusal, loader-failure propagation without fallback, and unchanged unsupported/exhausted/error mapping. No real tokenizer artifact or model is loaded by this suite.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -m pytest -q tests/test_code_label_family_selection.py tests/test_rwkv_g1_filtered_codes.py tests/test_rwkv_g1_code_label.py
```

Result: **58 passed** (18 new construction tests plus 40 inherited synthetic codebook/renderer tests). Inherited scoring tests use scripted CPU vocabulary tensors, not model forwards. The interpreter directory name does not imply GPU use; CUDA visibility is empty. No packages were installed. Whitespace checks passed. This is CPU API integration, not a new capability/calibration/control/safety/latency or memory measurement.

## Remaining executable runner gap

The generic Index engine and existing safety CLI can select this family, but `jobs/rwkv7/rwkv7_job.sh` still explicitly passes `family=rwkv7-hf` and its stock safety family. No dedicated code-label job mode, paired old/new runner, launcher registration, measured/repriced phase caps, or shell-subprocess dry run for that arm has been added. `jobs/common/readout_cost.py` still models the old readout paths and does not select this code-label family. The existing full8 dry-run stand-in is conversion-specific and does not exercise this family.

Next concrete step is independent review followed by a separately registered, CPU-dry-run paired qualification runner with correct code-label volume/accounting. Real model/GPU qualification remains unrun; paid work still requires merged proof and an acknowledged spend issue. No launch-ready or Clef-beating claim is made.

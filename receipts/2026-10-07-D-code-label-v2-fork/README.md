# Code-label v2: shared-prefix CPU proof and exact workload

All-field trusted labels restore an identical schema/state prefix across target questions. The existing state-fork scorer reduces the full pinned 2,008-row workload from **55,430,004 v1 positions to 6,675,792 padded v2 positions**, while the conservative four-row pack cap increases calls from **8,467 to 14,437**. These are deterministic tokenizer/planner counts, not GPU timings or a spend quote. No model/checkpoint was constructed, loaded or called; no GPU or paid work ran.

## Registration and integration

- Branch/worktree: `stack/rwkv7/g1-code-label-v2-fork`, `C:/Users/apumu/research/CHLabs/ssdm-wt/g1-code-label-v2-fork`.
- Original stack base: engine-bridge commit `f3bbaf36379114f43bdec0681ec912eb5cbcd59d`. V2 registration `plans/codex/C1_RWKV7_CODE_LABEL_V2_FORK.md` was committed at `3c9a4b42b3c8358bcfc7457ce5bbee3c1994c664`, before implementation and this DEV-volume proof.
- Implementation checkpoint: `7d10f91bb13913c7f1e112a594bf9f783a5f44a2`; then merged current `origin/agent-main` `4b59a73` (PR #158 and the merged training-view/consistency stack) at `1a6821bcb092cfdd9f93a887287f6c90a4e28c2e`. All preregistration/engine commits remain preserved. The single STATUS conflict retained both upstream training entries and the code-label integration entry under the current Codex takeover owner.
- V1 registration, renderer/backend implementation and frozen table are unchanged. Stock/default behavior is unchanged. V1 remains selectable as `rwkv7-hf-code-label-v1`; the new engine/safety family is `rwkv7-hf-code-label-v2`, template `jrt-rwkv-g1-code-label-v2`.

`decide/backends/rwkv_g1_code_label_v2.py` labels every option in every schema field identically, retaining the target allowed-code list. It preserves each original non-template segment, raw state bytes, IDs, descriptions and experimental provenance splits; no BOS, tail, normalization, calibration, option-domain or averaging change is made. V2 adds only the fixed trusted labels that v1 omitted from non-target fields. Thus v2 and v1 differ in prompt content; no claim is made that their real-model scores agree. The parity proof compares **fork versus recompute of identical v2 prompts**.

Scoring reuses `state_fork.fork_plan/fork_score` and the plugin's existing RWKV7 state operations. Each continuation is exactly one frozen code token. Default/hard maximum is four rows per pack, with the existing 4,096-position packing budget; configuration and scorer/renderer sources are revision-bound. Missing state operations/shared prefix fail closed, without fallback. Output shape/finite checks supplement the existing scorer. `fork=False` is a separately revision-bound CPU parity diagnostic, not the engine default.

The packing budget can admit an oversized singleton in the existing planner; the volume proof reports it explicitly. **None occur on this subset.** Maximum observed planned pack is four rows / 4,092 padded positions (4,084 among admitted rows). This bounds planned shape only; real peak memory, production-kernel parity and throughput remain unmeasured.

## Exact tokenizer-only DEV volume

The coordinator explicitly authorized tokenizer-only volume on the existing pinned DEV subset. Input `C:/Users/apumu/research/CHLabs/review-artifacts/order-debias-handoff-20261007/index-rows.jsonl.gz` compressed SHA256 is **`406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2`**. The script verifies the hash and exactly 2,008 unique run IDs, extracts only state/questions, never uses expected labels, never scores/trains, never opens TEST, and exports no item text or gold. Complete per-row workload/resource counts and source/input/revision hashes are in `dev-volume.json`.

| Scope | Arm | Real tokens | Padded positions | Calls |
|---|---|---:|---:|---:|
| All 2,008 planned rows | v1 direct | 55,430,004 | 55,430,004 | 8,467 |
| All 2,008 planned rows | v2 fork | 6,625,779 | 6,675,792 | 14,437 |
| 2,000 admitted rows | v1 direct | 50,186,548 | 50,186,548 | 8,211 |
| 2,000 admitted rows | v2 fork | 6,156,868 | 6,197,882 | 14,287 |

Both arms admit the **same** 2,000 row IDs and exhaust the **same** eight under the unchanged effective 16,384-token limit. The full score denominator remains 2,008; the admitted counts are workload diagnostics, not a newly selected evaluation subset. All 46,807 supplied options are preserved in the raw plan (46,295 on admitted rows). V2's shared-prefix total is 4,010,945 tokens raw / 3,848,402 admitted. Max raw prompt lengths are v1 29,692 / v2 30,126; max admitted lengths are 14,360 / 14,794.

Vocabulary SHA256: `1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf`; reviewed tokenizer source SHA256: `58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65`; unchanged filtered code/token/source table SHA256: `7226adf44b9529726b2f44e66b60257c92a50126a3f30131caad6ab0be231466`. The exact calling code checkpoint and volume-script hash are recorded in the JSON; the same script bytes are committed with this receipt.

## CPU checks and commands

Synthetic scripted state operations prove prefix equality through schema/state, exact original/provenance segments, fork/recompute score parity within 1e-6, call/real-token/padded-position accounting, no prefix-state mutation, all K=1..255 including K=53, conservative pack caps, source refusal and v1/default preservation. No neural model or weights are used; scripted tensor callbacks are distinct from model outputs. The committed volume's headline sums and identical admission sets are independently checked from its per-row counts.

The integrated focused suite passes **70 tests**. Whitespace checks pass. Repository jobs-lint checks are run on the committed final head before handoff.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -m pytest -q tests/test_rwkv_g1_code_label_v2.py tests/test_code_label_family_selection.py tests/test_rwkv_g1_filtered_codes.py tests/test_rwkv_g1_code_label.py
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -I 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-code-label-v2-fork/scripts/volume_rwkv_g1_code_label_v2.py' --rows 'C:/Users/apumu/research/CHLabs/review-artifacts/order-debias-handoff-20261007/index-rows.jsonl.gz' --tokenizer-json 'C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba/tokenizer.json' --out 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-code-label-v2-fork/receipts/2026-10-07-D-code-label-v2-fork/dev-volume.json'
```

The interpreter directory name `.venv-gpu` does not describe these operations: CUDA visibility is empty, and only CPU tokenization/planning or scripted CPU tensors run. No packages were installed.

Next: independent review and repricing that includes **both padded positions and increased call count**, then a separately registered matched runner/shell dry run. Real GPU numerical parity, memory, capability, calibration, safety/control/interpretability and latency qualification remain unrun. No ≤$20, launch-ready or capability-gain claim is made. Paid work still requires merged proof and an acknowledged spend issue. No push was performed.

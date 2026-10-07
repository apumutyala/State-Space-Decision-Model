# C1 code-label v1: CPU enabled, model performance unmeasured

The separately registered filtered table provides 255 distinct visible one-token answer codes under the exact cached G1k tokenizer. The explicit opt-in `G1CodeLabelBackend` can attach trusted target-code labels without changing any existing request-derived segment, raw state bytes, IDs, descriptions or experimental provenance split. Synthetic checked-adapter tests and a tokenizer-only real-renderer proof pass. No model/checkpoint was constructed, loaded or called; no GPU, DEV/TEST data or spend was used.

## Registration and fixed evidence

- Branch/worktree: `stack/rwkv7/g1-filtered-codes-c1`, `C:/Users/apumu/research/CHLabs/ssdm-wt/g1-filtered-codes-c1`; base `ee0adb70029fc80c65ee6072c307873034052362`.
- Component registration: `plans/codex/C1_RWKV7_FILTERED_CODEBOOK_V1.md`, commit `bfbb9acdb28c1e0fa081a86d74c89fa01d86d01c`.
- Distinct safe code-label registration: `plans/codex/C1_RWKV7_CODE_LABEL_V1.md`, commit `1e473f5146fcadd6a34482435ae4be05698593d9`, before renderer implementation/model outputs. The earlier A..IU vendor-format v2 arm remains unchanged and blocked; this arm does not adopt its newline/BOS/no-think changes.
- Cached tokenizer: `C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba/tokenizer.json`.
- Vocabulary SHA256: `1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf`; reviewed tokenizer source SHA256: `58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65`. Probe scripts verify both before importing the reviewed local tokenizer; no snapshot Python or remote code is executed.
- Frozen ordered code/token/source table SHA256: `7226adf44b9529726b2f44e66b60257c92a50126a3f30131caad6ab0be231466`; code-string-list SHA256: `2878be9488928b108ff83d0587e41cc25f433a023407a3040e459a274db5d0df`. Canonical table bytes are UTF-8 `json.dumps(ensure_ascii=True, sort_keys=True, separators=(',', ':'))`, without a terminal newline.

The rule enumerates 702 codes A..Z, AA..ZZ, keeps unique non-control singleton-token exact round-trips with the registered synthetic context boundary, then takes the first 255 of 498 eligible codes. The frozen table ends at LB (source ordinal 314); selected ordinal 51 is AZ, skipping the two-token AY. This selection used tokenizer properties only, with no model/benchmark outcomes. The committed table is `decide/data/rwkv_g1_filtered_v1.json`.

## Implemented behavior

`decide/backends/rwkv_g1_code_label.py` provides an explicitly selected `G1CodeLabelBackend(plugin, tokenizer_decode=tokenizer.decode)`. The plugin must be RWKV7 with the pinned vocabulary, reviewed G1 source revision and control-token refusal boundary. No global template/backend registration or cell configuration changes are made.

For each target question it starts from the existing `build_prompt_labelled(jrt-rwkv-g1-v1, ...)` segments. Each target option keeps its original ID/description segments and receives a separate trusted `[ANSWER CODE X]` label. The target re-read retains its original allowed-ID data segment and gains a trusted allowed-code list. Only trusted answer instructions change to ask for a code. Every original non-template segment must remain exactly equal, including provenance instruction/data splits; unexpected base structural changes fail closed.

The existing v1 tail stays exact: `\n\nAssistant: <think>\n</think>\nANSWER:`. There is no BOS insertion, newline normalization, state rewrite, option pruning, order averaging or position-prior calibration. Experimental provenance is still flag-gated, encoded through the same untrusted boundary and reported as mask `applied=false`; no ISE/ASIDE intervention or new safety property is claimed.

Scoring uses one next-token forward per target question. It reads the full-vocabulary next-token log probabilities at each frozen answer-token ID and maps all scores back to original semantic IDs. It does not forward or score a continuation terminator. The unchanged checked adapter supplies the ordinary normalization, external-ID tie behavior, fingerprints, error boundary and limits. Call/token accounting covers the prompts actually forwarded. The separate exact-token parser rejects strings, multiple/unknown/control tokens and codes outside the particular question's domain; it does not prefix-match A versus AA.

The backend revision binds model/plugin identity, tokenizer, filtered table, base v1 template identity, code-label renderer canary, source hashes, scoring path and the standalone codebook component. Tokenizer callback identities are caller assertions supplemented by the actual frozen-code/context checks, not cryptographic authentication. Runtime model provenance still relies on the existing reviewed plugin loader.

`decide/backends/rwkv_g1_filtered_codes.py` also retains the standalone immutable token-ID component. Its own full-request `plan/score` entry points remain blocked because that component alone supplies no renderer; the distinct code-label backend now supplies the safe rendering integration. `tokenizer-proof.json` records the standalone phase; `code-label-proof.json` records the completed code-label CPU phase.

## Proof and limits

- 40 CPU tests across `tests/test_rwkv_g1_filtered_codes.py` and `tests/test_rwkv_g1_code_label.py` pass. They cover every cardinality 1..255, all 255 token/semantic labels, synthetic K=53 tool/hash-ID cases, A/AA prefix collisions, exact parser refusal, all offered options, full-vocabulary shape/finite checks, frozen-table/plugin drift, ties, call/input limits, control-token refusal and experimental provenance-split equality.
- Tests use synthetic tokenization and scripted CPU vocabulary tensors, with no model object or model forward. Their artificial byte-codec context allowance is not a real-model context or memory measurement. The exact tokenizer proof is separate.
- The actual cached tokenizer proves all 255 boundaries after the real rendered synthetic 255-option prompt, exact non-template/provenance equality, unchanged request fingerprint and old v1 tail. Production-style segmented encoding plans one call and **11,188 prompt tokens** for this synthetic request (template 6,159; data 5,025; instruction 4). This is one constructed prompt, not a throughput, memory, dataset or capability result.
- Whole-string encoding in the proof is boundary diagnostics only. Production still encodes the original labelled segments separately, preserving the existing no-merges-across-trust-boundaries policy.
- Default `decide/prompt.py`, `decide/backends/hf_readout.py`, backend registrations, interface and job/cell configs are unchanged. No learned gain, calibration, safety, interpretability, latency qualification, promotion, Index acceptance or spend is claimed. Probability averaging remains unmeasured/not served.

## Exact commands

Run sequentially from this worktree:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -m pytest -q tests/test_rwkv_g1_filtered_codes.py tests/test_rwkv_g1_code_label.py
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -I 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-filtered-codes-c1/scripts/probe_rwkv_g1_filtered_codes.py' --tokenizer-json 'C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba/tokenizer.json' --out 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-filtered-codes-c1/receipts/2026-10-07-D-g1-filtered-codebook/tokenizer-proof.json'
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -B -X utf8 -I 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-filtered-codes-c1/scripts/probe_rwkv_g1_code_labels.py' --tokenizer-json 'C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba/tokenizer.json' --out 'C:/Users/apumu/research/CHLabs/ssdm-wt/g1-filtered-codes-c1/receipts/2026-10-07-D-g1-filtered-codebook/code-label-proof.json'
```

The interpreter directory name `.venv-gpu` does not describe this work: CUDA visibility is empty and all performed operations are CPU tokenization, metadata checks or scripted tensors. No packages were installed. Commit locally for internal review; no push, model run or job submission is authorized by this receipt.

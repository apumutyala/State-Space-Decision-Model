# backbone_screen: stock zero-shot proxy600 screen of an attention-free backbone candidate (one H100, <= $16 each)

Why. The trained RWKV7-G1k-13.3B tops out at 38.67 (subset) from a stock 24.85; the backbone's starting capability
is the biggest lever. This job measures a bigger attention-free candidate's STOCK Index on the pre-declared proxy600
rows (644 rows of the 2,008-row DEV subset, `jobs/common/dev_proxy600.json`, frozen scorer), paired against the stock
RWKV7-G1k v3w readout on the same rows (art-ivu28, 27.32 on proxy600; 31.07 on the subset). Measurement only
(`decision_rule: null`).

## Candidates (attention-free: every decoder layer is `RWKV6Attention`, no softmax attention anywhere)

| MODE | Repo @ revision | Params / bf16 | Parent | Template | License |
|---|---|---|---|---|---|
| `qrwkv6_32b` | `recursal/QRWKV6-32B-Instruct-Preview-v0.1` @ `2da855ca372cefab920716eca30f11ed65d6c708` | 34.74B / 69.49 GB | Qwen2.5-32B-Instruct (MMLU 76.6 vs 81.8, card) | `jrt-chatml-nothink-v1` | Apache-2.0 |
| `qrwkv_qwq32b` | `featherless-ai/QRWKV-QwQ-32B` @ `6801e98e72a3e03b4611cdd816a0532411f6e1e0` | 34.95B / 69.89 GB | QwQ-32B (MMLU 74.3 vs 79.9, card) | `jrt-chatml-v1` | Apache-2.0 |
| `rwkv_qwen3_30b_a3b` | `OpenMOSE/RWKV-Qwen3-30B-A3B-Instruct-cxa07c` @ `2fb46556362db60336d4abf951aee3ebfb7db449` | 30.75B MoE (~3B active) / 61.50 GB | Qwen3-30B-A3B-Instruct (MMLU 74.8, card bench.txt; NIAH 84 at 8K, 37.8 at 16K) | `jrt-chatml-nothink-v1` | Apache-2.0 |

- **Code:** both repos bundle the same `modeling_rwkv6qwen2.py` / `configuration_rwkv6qwen2.py`, vendored unmodified
  and sha-pinned in `third_party/rwkv6qwen2_hf` (no `trust_remote_code`); `verify_snapshot_code` refuses a snapshot whose
  code or config.json differs. The 4.43-era code runs on transformers 5.17.0 through `decide.backends.rwkv6qwen2`
  `install_compat` / `compat_config` (missing 4.x config defaults, `ROPE_INIT_FUNCTIONS["default"]`, the 5.x dict form of
  `_tied_weights_keys`; both configs have `use_rope: false` and `tie_word_embeddings: false`), with the files unchanged.
  Checked on CPU: checkpoint weight names (`model.safetensors.index.json`) == the vendored model's state_dict for both
  repos, params 34,737,492,992 / 34,947,208,192 (== HF); `load_model` refuses any missing / unexpected key.
- **RWKV-Qwen3-30B-A3B (mode `rwkv_qwen3_30b_a3b`):** different code (`modeling_rwkv07c.py`, hxa07c RWKV-7-style time mix
  with RoPE, Qwen3-MoE FFN), vendored unmodified in `third_party/rwkv07c_hf`; config `transformer_layers: [-1]` = no
  attention layer (the loader refuses otherwise and asserts every layer is `RWKV07CAttention`). Written for 4.5x;
  `decide.backends.rwkv07c.install_compat` bridges 5.17 from outside: transformers' own qwen3_moe checkpoint conversion
  registered for `rwkv07c` (per-expert checkpoint -> 5.x fused experts; CPU test: a per-expert file written by hand
  loads to identical logits; after the merge the real index's keys == the model's, 30,749,571,072 params == HF),
  `head_first=False` dropped before fla 0.5.2 (which raises on the kwarg), `create_causal_mask` (renamed kwarg in 5.x)
  -> None (its only consumer is the RWKV layer's padding multiply; the readout never pads), RoPE default fn, config
  defaults. Fork == recompute on a tiny random MoE model (`tests/test_rwkv07c_backend.py`).
- **Kernel:** `fla.ops.gla.fused_recurrent_gla` (RWKV6-Qwen2) / `fla.ops.rwkv7.fused_recurrent_rwkv7` (RWKV07C) (fla 0.5.2 signature checked), from the `jobs/probe9b/Dockerfile` image
  (torch 2.13 / triton 3.7.1 / transformers 5.17.0 / fla 0.5.2 + the Decision Index kit), CMD rewritten.
- **Readout:** the stock LM-logit readout (`HFReadoutBackend`, sum log p(' ' + id + `<|im_end|>`)) under the parents'
  ChatML format, the same readout family as the stock Qwen3.5-9B arm (job-5cmat, 38.27 subset). QwQ's tokenizer has
  `<think>` / `</think>` as single special tokens, so it gets the Qwen3/QwQ no-think tail (`jrt-chatml-v1`); Qwen2.5 has
  no think tokens, so the Instruct conversion gets `jrt-chatml-nothink-v1` (`<|im_start|>assistant\nANSWER:`).
  Untrusted text cannot produce `<|im_start|>` (UntrustedEncoder, checked on both tokenizers).
- **Input cap 16,384** (the Instruct card's training context; QwQ's card states none, same RADLADS recipe): 5 of the 644
  rows exceed it and are refused (scored 0), counted for both models.
- **Fork:** `decide.backends.rwkv6qwen2.RWKV6StateOps` over `RWKV6State` (GLA state [B, 40, 128, 128] fp32 + shift
  state per layer, ~168 MB per sequence), prefill in 8,192-token chunks, 16 rows per packed call. CPU (tiny random
  model, naive GLA): fork == recompute to 1e-3, chunked == one-shot prefill, extend never mutates the prefix state
  (`tests/test_rwkv6qwen2_backend.py`). On the GPU the engine re-checks fork == recompute on the canary request before
  row one (`index/selfcheck.json`; max |delta| <= 0.25 nat and same argmax, else the index phase fails).

## Memory on one H100 80 GB
Weights 65.1 GiB bf16. Prefill chunks of 8,192 tokens: MLP 3 x 27,648 x 8k x 2 B ~ 1.4 GB + token-shift mixes ~1 GB.
Packed tails: <= 4,096 positions -> logits 4,096 x 152,064 bf16 = 1.25 GB; 16 state copies 2.7 GB. Peak ~72 GiB.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A CUDA OOM on a row surfaces as a `backend_error` row status
(reported in `decision.json` `row_status`), not a silent score.

## Disk
`hf_cache: false` (as the 55 GB Clef-27B job): 65.1 GiB of shards + the ~15 GB image fit the 100 GiB job scratch; the
cached-download path only guarantees 32 GiB. `allow_patterns` skips README / images.

## Phases (MODE env; `jobs/backbone_screen/screen_job.sh`)

| Phase | Cap (min) | Content | Fail |
|---|---|---|---|
| setup | 35 | pins + fla guard; DEV rows sha256 (`jobs/clef_baseline/data.sha256`); parent results sha256; pinned `snapshot_download`; sha256 of every pinned file (`snapshot_<mode>.sha256`: HF LFS sha256 of every shard + the small files); vendored code == snapshot code; proxy600 identity + rows | 5 / 10 |
| index | 75 | `decision_index run` with `ReadoutIndexEngine` (family `qrwkv6-chatml-nothink` / `qrwkv6-chatml`, option `repo`, `input_cap`, `selfcheck_out`), live rate guard (`--exclude-startup`; projection past 95% of the cap -> exit 15) | 12 / 7 / 15 |
| score | 10 | `dev_subset score` + `ece` on proxy600; `dev_subset compare` vs `screen_parent/art-ivu28/v3w.results.jsonl` (stock RWKV7 v3w, shipped, sha256 `a52a68eb...`) | 13 |
| report | - | `jobs.backbone_screen.summarize` -> `decision.json` (`verdict: measured`, index + CI, paired diff, row statuses, self-check, rate) | 9 |

Schedule 35 + 75 + 10 + reserve 10 = 130. Worst case (60 + 2 x 130) x 0.0495 = **$15.84** per mode; hard cap $16.
Expected: build ~15, download + sha ~15, load ~3, index ~25-30 (exact fork volume with this tokenizer: ~2.4M
positions, ~2,120 calls on the 639 in-cap rows), score ~4 -> ~65 min = ~$3.2 billed (~$4.1 at the $0.063 quote).

## submit_job fields
From `job_params_<mode>.json` (`submit` + `context` blocks): chip h100 x 1, mission `backbone-scout`, `hf_cache false`,
`max_restarts 0`, `resume none`, success = `/output/BACKBONE_SCREEN_DONE` + `/output/MANIFEST.jsonl`, heartbeat health
(interval 600 s, grace 2,400 s for the 70 GB download, 2 strikes, kill).

## Exit codes
0 measured | 2 bad MODE | 5 setup | 6 export verification | 7 phase timeout | 9 decision invalid | 10 snapshot / vendored
code / DEV rows / parent identity, or download failed | 12 Index kit run failed (incl. a failed fork self-check or a
weight mismatch at load) | 13 scoring | 15 readout too slow by measurement | 1 finalize.

## Not screened here (and why)
- `featherless-ai/QRWKV-72B` (79.3B, MMLU 77.5): 158.6 GB bf16; 8-bit (~79 GB) does not fit one H100 next to activations.
  Needs 2 x H100 (bf16/FP8 split) or 4-bit; a follow-up if a 32B conversion screens well.

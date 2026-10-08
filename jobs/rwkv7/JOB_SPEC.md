# rwkv7: the pre-registered fallback parent RWKV7-G1k-13.3B (two stages, <= $20 each). Spec only, NOT submitted

Stage A (`JOB=readout`) runs in every non-diagnose branch of `jobs/DECISION_TREE.md` (as the parent comparator, or as
the fallback). Stage B (`JOB=lora`) runs when Q9-AF is folded or fails its probe, gated on stage A with `depends_on`.

## Checkpoint identity (verified 2026-10-06 from the repo's files, fetched as text; no weights downloaded)

| Item | Value |
|---|---|
| Repo / revision | `RWKV/RWKV7-G1k-13.3B-20260930` @ `ce3630b854cedbd673569f4600c6cf744bca9dba`, Apache-2.0 |
| Weights | 28 safetensors shards, 26,540,800,536 bytes (24.7 GiB), 13,270,298,624 params bf16; per-shard LFS sha256 in `jobs/rwkv7/snapshot.sha256` |
| config.json | 61 layers, hidden 4096, 64 heads x 64, FFN 16384, vocab 65536, untied head, `wkv_implementation: chunked`, `wkv_state_dtype: float32`; **no** `max_position_embeddings` |
| Trained context | 25,600 tokens (model card; source `rwkv7-g1k-13.3b-20260930-ctx25600.pth`) |
| Code | `auto_map` -> bundled `modeling_rwkv7.py` / `configuration_rwkv7.py` / `tokenization_rwkv7.py` (from transformers PR commit `4ad9ed07`, not in any release as of v5.17.0). Card asks for transformers >= 5.15 |
| Tokenizer | `tokenizer.json` only (RWKV World trie). Byte-identical ids to our `decide.rwkv_world_tokenizer` on ordinary text (checked on CPU); `<|endoftext|>` = id 0 is its only special token |

**Findings that changed the plan.**
1. `decide.backends.hf_readout.load_rwkv7` **cannot load this checkpoint**: it expects the `fla` class and
   `rwkv_vocab_v20230424.txt`. New `load_rwkv7_hf` loads the repo's own code, **vendored unmodified and sha-pinned** in
   `third_party/rwkv7_hf` (no `trust_remote_code`); every job checks the snapshot's code + config against the pins
   (exit 10 otherwise). Untrusted text that encodes to id 0 fails closed (`ControlTokenError`).
2. The old loader would have capped input at 2,048 tokens (no `max_position_embeddings` in config). The real trained
   context is 25,600; `load_rwkv7_hf` uses that, so long DEV rows are scored instead of `exhausted`.
3. **LoRA module names verified on the real code**: `rwkv7.blocks.{i}.att.{receptance,key,value,output}` (4096x4096) and
   `rwkv7.blocks.{i}.ffn.{key,value}` (4096<->16384); 61 x 4 + 61 x 2 = 366 modules, matching `BACKBONES["rwkv7-13.3b"]`
   (`tests/test_m1_next_jobs.py` applies the real regexes to the real classes).
4. The **cache fork did not support this model**: its state travels as `state=` (`Rwkv7Cache`), so `_hf_fork` failed
   with "no cache". Added `HiddenStateBackbone._state_fork` (state copies via `batch_repeat_interleave`, which assigns
   new tensors; `reorder_cache` copies in place and would break autograd). On a tiny model built from the vendored
   code, cache fork == recompute fork in values and LoRA gradients (`tests/test_m2_review_fixes.py`, family `rwkv7`).
5. The bundled "chunked" WKV is pure PyTorch: prefill/training throughput is **unmeasured**. Stage A measures it before
   stage B spends anything.

## Memory on one H100 80 GB

| Item | Estimate | Source |
|---|---|---|
| Weights (bf16, emb + head included) | 26.5 GB | params x 2 |
| Decode state per sequence | 65.0 MB (61 x (64x64x64 fp32 + 2 x 4096 bf16)) | config; `probe_rwkv7.state_bytes` |
| Readout activations | measured by stage A at 2k/8k/16k; `logits_to_keep` limits logits to the option tokens (no `[B, T, 65536]` tensor) | `jobs/rwkv7/probe_rwkv7.py` |
| LoRA pilot, cache fork, no checkpointing, rank 64 | T=1024,K=16: 46.5 GB; T=2048,K=6: 56.7 GB; **T=2048,K=16: 58.9 GB** | `train.estimate.train_point` |
| same at rank 256 (m2 default) | T=2048,K=16: 71.8 GB (too close to the 76 GB stop) | idem; why the pilot uses rank 64 (287.8M LoRA params) |

The estimates assume kernel-like activation sizes; the stage B ladder replaces them with measurements (one optimiser
step at T = 512 / 1024 / 2048 on the worst rows, largest K first) before training, and training is capped at the
largest T that stays under 76 GB.

## Stage A: readout (`JOB=readout`, `jobs/rwkv7/job_params_readout.json`)

| Phase | Cap (min) | Content | Fail |
|---|---|---|---|
| setup | 25 | pins + kit + vendored-code self check; DEV rows sha256; pinned `snapshot_download`; `sha256sum -c snapshot.sha256`; snapshot code == vendored | 5 / 10 |
| probe | 15 | load time, weights GB, prefill tok/s + peak GB at 2k/8k/16k, state bytes | 11 / 7 |
| index | 75 | Decision Index stock readout on the 2,008-row DEV subset (`family=rwkv7-hf`, template `jrt-rwkv-g1-v1`). Measured stop conditions (`jobs/common/rate_guard.py`): preflight from the **probe phase's measured prefill tok/s**, then the live row rate after 8 min; a projection past 95% of the cap stops with exit 15 | 12 / 7 / 15 |
| score | 5 | subset slice index + CI, ECE (debiased, SmoothECE per length bin) | 13 |
| steer | 20 | `safety_eval.run_smoke --backend decide --family rwkv7-hf`, dev split, 40 route + 24 pay, **no `--final`** (D1-B stays held out) | 14 / 7 |
| report | 2 | `scripts/decide_rwkv7.py readout` -> `readout-measured` | 9 |

Schedule 142 + reserve 8 = 150. Worst case (30 + 2 x 150) x 0.0495 = **$16.34**; expected ~$4.70; hard cap
`set_workspace_cap(MTD + 20)` when A runs alone.

## Stage B: LoRA m2 pilot cell (`JOB=lora`, `jobs/rwkv7/job_params_lora.json`, `jobs/rwkv7/pilot_cell.json`)

`train/run_cell.py` stages, all pre-registered in `pilot_cell.json`:
- **smoke**: on 2 real rows (K <= 4, T <= 512) with a random rank-8 LoRA, cache fork vs recompute fork on GPU/bf16 must
  agree: max relative feature diff <= 0.03 and LoRA-gradient cosine >= 0.99, else exit 21.
- **ladder**: T in {512, 1024, 2048}; peak and step time per rung; T_max = largest rung <= 76 GB, else exit 22.
  **Measured rate**: if even the fastest fitting rung's measured step time cannot reach the pass rule's 2 x 50 = 100
  steps inside `max_minutes`, exit 26 before any training spend; after the first 10 training steps the same check
  runs on their measured mean step time (exit 26, checkpoint kept). A slow model is reported as a throughput
  finding (26), never as `pilot-fail`.
- **train**: fork head (`fork_mode=cache`, no gradient checkpointing: the cache fork refuses it), CE arm, LoRA rank 64,
  batch 1 x accum 8, lr 1e-3 head / 1e-4 LoRA, <= 400 steps or 110 min; 2,000 train / 300 eval records from the m2
  sample (`records.jsonl.gz` sha256 `824671c9...ceb7`, PRIVATE, through `train.guard.SplitGuard` built from the DEV and
  TEST manifests), rows with K > 16 dropped; checkpoint every 50 steps to `$GMN_CHECKPOINT_DIR` + `POST /v1/checkpoint`;
  eval cards every 100 steps. Non-finite loss -> exit 23; peak > 76 GB -> exit 24.
- **Pilot pass rule**: finite; mean loss of the last 50 steps >= 10% below the first 50; held-out NLL improves from the
  step-0 card to the last; peak <= 76 GB. `pilot-pass` / `pilot-fail` are both valid measured outcomes (exit 0).
  No Index number is claimed from the pilot; the factorial cells carry the Index eval (M2_RUN_SPEC).

Schedule 172 + reserve 8 = 180. Worst case (30 + 2 x 180) x 0.0495 = **$19.30**; expected ~$7.72.

**Workspace cap when B is chained.** The cap is workspace-wide and B is submitted (with `depends_on`) while A is still
queued or running, so at submit time it must cover A's remaining worst case plus B's: `set_workspace_cap(MTD + 36)`
(`costs.concurrent_cap_usd(["rwkv7_readout", "rwkv7_lora"])` = ceil(16.34 + 19.30)). MTD + 20 is right only if B is
submitted after A has ended. With the probe also live (qual pass branch), add its $28.21.

## `submit_job` fields (one build context for both stages, so `depends_on` needs no `allow_different_context`)

Context: `bash jobs/common/build_context.sh rwkv7 <m1_sha> <dir>` (code + PRIVATE DEV rows + m2 records);
`open_mission("m1-rwkv7-fallback")`.

```json
{"chip": "h100", "chip_count": 1, "mission": "m1-rwkv7-fallback",
 "idempotency_key": "m1-rwkv7-readout-<context_sha256[:12]>", "label": "rwkv7 stage A: readout + steer smoke",
 "context_id": "<id>", "dockerfile_path": "Dockerfile", "build_timeout_minutes": 30,
 "command": "bash /opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh", "env": {"JOB": "readout"},
 "max_duration_minutes": 150, "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
 "hf_cache": true, "model": "RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba", "clock_lock": true,
 "checks": {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/RWKV7_READOUT_DONE"},
                                             {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
            "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                       "interval_s": 600, "grace_period_s": 1800, "fail_threshold": 2, "on_fail": "kill"}}}
```

Stage B differs in: `idempotency_key "m1-rwkv7-lora-<sha12>"`, `env {"JOB": "lora"}`, `max_duration_minutes 180`,
`max_restarts 1`, `resume "checkpoint"`, `queue_ttl_minutes 2880`, success file `/output/RWKV7_LORA_DONE`, and
`depends_on {"job": "<stage A id>", "require": "result.ok == true"}`. The 24.7 GiB of weights fit the documented
>= 32 GiB hf_cache headroom, so both stages share one cached download.

## Exit codes

0 ok | 2 bad JOB / cell crash | 5 setup | 6 export verification | 7 phase timeout | 9 decision invalid |
10 snapshot, vendored code, DEV rows or records identity | 11 probe | 12 Index kit | 13 scoring | 14 steer smoke |
15 readout too slow by measurement | 21 fork mismatch | 22 no T fits | 23 non-finite loss | 24 memory stop |
25 too few pilot items | 26 training too slow by measurement | 1 finalize.

## Health check (`checks.health`) and heartbeat, field by field

- `type: heartbeat`, `path: /output/heartbeat`: the platform reads the file's freshness from outside the container.
  The runner never vouches for liveness itself; `jobs/clef_baseline/watchdog.sh` rewrites the file every 60 s **only
  while** the main process is alive **and** some file under the watched dirs (`$OUT`, `$CK`, `$HF_DIR`,
  `WATCH_EXTRA=$WORK`; `find -H` follows a symlinked HF cache root) changed in the last `STALL_S` = 1,200 s.
- `interval_s: 600`: the platform evaluates the heartbeat every 10 min.
- `fail_threshold: 2`: two consecutive stale evaluations trigger `on_fail`; one late write is tolerated.
- `grace_period_s: 1800`: no evaluation during the first 30 min of an attempt (image start, weight download and
  sha256, model load write few files).
- `on_fail: kill`: the attempt ends `self_terminated` (its own terminal state, billed to that point).
- Worst-case detection of a hung worker: `STALL_S` + 2 x `interval_s` = 20 + 20 = 40 min after its last file write
  (plus up to the 30-min grace if it hangs at start), instead of billing to `max_duration`. A dead main process stops
  the heartbeat at once.
- Every phase writes under a watched dir (kit: one results row per request; pipeline: `train_log.jsonl` every 25
  steps; `run_cell`: `train_log.jsonl` every step; probes and scoring: per-phase JSON). `tests/test_m1_next_jobs.py`
  checks statically that every path a runner writes resolves to a watched dir, and runs the watchdog with progress
  only under `WATCH_EXTRA`.

Exports, verification and the local CPU recompute follow `jobs/probe9b/JOB_SPEC.md` (`dev_subset score` / `ece`
reproduce `index/score.json` / `ece.json`; `safety_eval` summary from `predictions.jsonl`).

## C1: `JOB=readout_g1v2` (vendor-compliant readout, `jobs/rwkv7/job_params_readout_g1v2.json`)

Stage A's Index phase only (no probe, no steer smoke), with the readout family `rwkv7-hf-g1-v2`
(`decide.backends.rwkv_g1_v2.G1V2Backend`, template `jrt-rwkv-g1-v2`): no blank lines inside the system/user turn,
token 0 first, the G1k no-think tail `Assistant: <think></think>
`, single-token option codes (A..Z, AA.. from the
frozen filtered codebook) read from one next-token distribution per question, and whole-string tokenization. The live
rate guard (`jobs.common.rate_guard run`) stops a slow box. Schedule 105 + reserve 10 = 115. Worst case
(60 + 2 x 115) x 0.0495 = **$14.36**; expected ~$2.50; hard cap $15.

## R3: `JOB=readout_g1v3` (v3 readout family, `jobs/rwkv7/job_params_readout_g1v3.json`)

Stage A's Index phase only, run as eight readout variants over the same DEV subset rows in ONE process
(`jobs.rwkv7.readout_variants`: one unmodified kit run per variant, the RWKV7 plugin cached so the model loads once),
in priority order (a cut-short run keeps the first ones; every variant with complete results is scored):

| variant | family | change |
|---|---|---|
| v2 | `rwkv7-hf-g1-v2` | control, byte-identical to job-zhyvg (27.31 [24.53, 30.03]) |
| v2f | `rwkv7-hf-g1-v2f` | v2 + instruction fix only: dict-valued instructions (ToolRet {candidate, task}, BRIGHT) rendered as canonical JSON as in v1, instead of the question id |
| v3 | `rwkv7-hf-g1-v3` | v2f + v3 code table (A..Z, then single-token two-letter codes minus English words/abbreviations; `decide/data/rwkv_g1_v3_codes.json`) + code key (question restated, then one `CODE = label` line per option, for k >= 9) |
| v3w | `rwkv7-hf-g1-v3w` | v3 + word option ids (non-opaque, first token unique) listed as `- id: desc` and scored by their own first token |
| v3r | `rwkv7-hf-g1-v3r` | v3 + top-m re-read: the model's OWN first-pass top 2-3 codes re-listed after the first answer context, second pass from the same prefix state; p = p1 outside the shortlist, p1(S) x p2(.\|S) inside |
| v3p | `rwkv7-hf-g1-v3p` | v3 + deterministic per-question presentation permutation (sha256 of question id, type, option ids), codes mapped back to option ids |
| v1 | `rwkv7-hf` | the jrt-rwkv-g1-v1 readout (job-tkfaj / uhxb5), for a pre-declared CPU-side arithmetic-mean ensemble v1 + v3 on identical rows |
| v3c | `rwkv7-hf-g1-v3c` | v3 + contextual calibration (Zhao et al. 2021; content-free "N/A" / "" / "[MASK]" state + instructions, k <= 16) |

Nothing is fitted on any rows. Outputs: `index/<variant>/{results.jsonl, score.json, ece.json}`; the live rate guard
watches all results files (total 8 x rows). Index cap = 21 min (job-zhyvg v2, measured) x 8 x 1.3 = 218 -> 220.
Schedule 20 + 220 + 32 + reserve 10 = 282. Worst case (60 + 2 x 282) x 0.0495 = **$30.89**; expected ~$10.74;
hard cap $31.

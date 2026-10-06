# Performance audit of every job and eval path (2026-10-06, branch `claude/readout-fork`)

**Trigger.** rwkv7 stage A (job-mhc3h) stopped with exit 15. It measured 25 rows in 9.5 min and projected 763 min
against a 75-min cap, at a cost of $0.56. Prefill is not the problem: the probe measured 3.6–4.3k tok/s with a 36 GB
peak. The readout algorithm is.

## How the numbers were obtained

- **Measured, job-mhc3h:** `scripts/profile_readout.py` on the 25 measured rows. A 17-question ContractNLI row took
  46 s for 229k input tokens; a 1-question ANLI row took 0.25 s.
- **Volumes:** `jobs/common/readout_cost.py volume` computes the exact token and call volume of the 2,008-row DEV
  subset under each scoring path. It uses the real prompt builder and tokenizer, with no model. The results are
  committed as `jobs/common/readout_volume_{rwkv7,qwen3.5}.json` (run ids and counts only).
- **Fitted cost model:** wall time = positions / tok_s + calls x call_s. Fitted on the 25 rows:
  5,360 tok/s, 0.098 s/call, r² 0.99999.
- **Validation:** the model reproduces the failed run. It projects 754 min for the recompute path; the live guard
  measured 763 min (`tests/test_perf_paths.py`).
- **Not measured on GPU:** the after-fix rates below are projections from that model, not measurements. The live
  row-rate guard (`jobs/common/rate_guard.py run`, exit 15) stays the authority.

DEV subset volumes (2,008 rows):

| Path | RWKV7 tokens | RWKV7 calls | Qwen3.5 tokens | Qwen3.5 calls |
|---|---|---|---|---|
| recompute (old HF readout: prompt + option, per option) | 236.7 M | 10,949 | 239.3 M | 10,731 |
| per question (old decision-head backend: one full row per question) | 55.1 M | 8,467 | 52.6 M | 8,467 |
| **prefix fork** (new) | **6.17 M** (6.39 M positions with padding) | **4,536** | **6.01 M** (6.23 M) | **4,501** |

## Path by path

| Path | Dominant cost before | Rate before | After | Fix | Status / test |
|---|---|---|---|---|---|
| **decide HF readout** (`decide/backends/hf_readout.py`). Used by rwkv7 stage A, both probe9b Index arms, the m2 parent arm, `safety_eval.run_smoke --backend decide`, debias controls, m0 stock readouts | **Per-option recompute.** Every (question, option) re-forwards the whole prompt: sum over questions of K_q x prompt. ContractNLI: 17 q x 3 options x 4.5k tokens = 229k tokens per row | **0.044 rows/s** measured (763 min projected for the subset) | 6.39 M positions, 4,536 calls -> **~27 min, ~1.2 rows/s** at 5.36k tok/s and 0.098 s/call (~32 min at the probe's 4.3k tok/s) | `decide/backends/state_fork.py`: the shared prefix (template, schema of all fields, state) is read once. Every (q, o) tail = question re-read + continuation, all run from copies of that state, right-padded and packed (<= 4,096 positions per call; ~2 calls per row). `logits_to_keep` covers only the tail; scores are gathered at the option-token positions | **Applied.** `tests/test_readout_fork.py`: fork == recompute to 1e-5 (scores relative, probabilities absolute) on tiny RWKV7 (vendored code), Qwen3.5 hybrid, Q9-AF student GDN and the stock path; permutation equivariance; pack composition; plan/score call accounting; adapter probabilities |
| Same backend, CPU side | **Repeated tokenisation.** The state segment was tokenised once per question and again in plan() and score(): 34 pure-Python passes over 4.5k tokens per ContractNLI row | not separately measured (CPU) | 1 pass per distinct segment | Segment-to-ids memo on the backend, bounded at 4,096 entries | Applied. `test_segment_memo_changes_no_ids_and_encodes_once` |
| **rwkv7 stage A preflight** | **Proxy volume.** It priced 2.37 M proxy tokens x 2 = 18 min against a path that forwards 236.7 M | 40x under-count; passed, then stopped at minute 9.5 | Exact volume x **measured** rates | `rate_guard preflight --volume ... --tok-s <probe prefill> --call-s <probe small_call_s>`. The probe now also measures `small_call_s` and `pack_call_s` | Applied. `test_cost_model_reproduces_the_measured_job_and_clears_the_cap` |
| **probe9b Index arms** (stock + converted 9B) | Same per-option recompute; 239.3 M tokens per arm | not run; ~744 min per arm at 5.36k tok/s | 6.23 M positions per arm: about 5 min + 7 min of calls at 20k tok/s, about 28 min at 5k tok/s, against a 40-min cap | Converted path: `convert.model.ConvertedStateOps`. Teacher attention keeps its KV in an HF DynamicCache; native and student GDNs keep (conv tail, S) through the new `GDN.step`; student layers add a [B,1,T,1] position placeholder. Preflight uses the measured stock prefill rate. The 9B per-call cost is not measured yet, so the fitted RWKV7 value is used | Applied. Tests as above (Q9-AF families) plus `test_gdn_step_over_chunks_equals_forward` |
| **m2 parent arm** (`jobs/m2_parent_arm`) | Goes through the HF readout, so it had the same per-option recompute. Its volume table assumed one read per question row (true only after the fork), and its Index DEV term was proxy x 2 = 4.74 M | under recompute the record volumes were ~K x too low | Fork applies automatically. Index DEV volume re-measured: **6.39 M** (prefix-fork positions). Child evals now 34.2 M tokens, so the 50-min cap needs **>= ~12,000 tok/s** (was 11,420) | Volume corrected in `score_parent.VOLUME_TOKENS`, both JOB_SPECs and `tests/test_m2_gates.py`. The parent arm was already cached once per parent | Applied |
| **m2 child, `DecisionModelBackend`** (Index / decision / steer arm in `train.m2_gates`) | **Unbatched, per-question full-prompt loop**: B=1, one full row per question | 55.1 M tokens on the subset: ~171 min at 5.36k tok/s | ~6 M: shared prefix + equal-length suffix batches, about 20–25 min | `heads/prefix_features.py`: evaluation only (no grad). Rows of one request read schema + state once; suffixes are batched by equal length from state copies; the fork head continues from each row's own state. `DecisionModelBackend` scores a whole request in one pass | Applied. `test_shared_prefix_eval_equals_plain` (all 5 heads x RWKV7 / Qwen3.5 / Q9-AF, mixed-request batch, reversed order), `test_decision_backend_one_pass_equals_per_question_loop` |
| **m2 eval card / temperature fit** (`train.evalcard.predict`, `m2_gates.fit_child_temperature`) | Batches of 8 rows; multi-question records re-read their prompt per question | dev card 4.0 M + fit 3.4 M (`CHILD_EXTRA_TOKENS`, upper bounds) | Multi-question records share their prefix. Single-question records (most of the m2 sample) take one batched plain forward, as before | `shared_prefix=True` in `predict` | Applied (same tests). Modest gain because m2 records are mostly single-question |
| **Training step** (`train.trainer`, fork head, cache mode) | **Python per-token loop**: one full-model call per continuation token (`_hf_fork` / `_state_fork`), with B=K | about 3–6 extra full-model calls per row | 1 call per row: continuations are right-padded and read at each row's last position | Single-call continuation for `state=` (RWKV7) and `past_key_values` (Qwen3.5 / Q9-AF) models; HF Mamba (`cache_params`) keeps the token loop | Applied. `tests/test_m2_review_fixes.py`: cache fork == recompute fork in values **and LoRA gradients** for RWKV7, Q9-AF and Qwen3.5 |
| Training step, remaining costs | B=1 micro-batches x accumulation 8 (the cache fork cannot use gradient checkpointing, so memory-bound), and the pure-PyTorch "chunked" WKV | estimated ~11 s/step at T=2048 (8 x 2048 x 3 / 4.3k tok/s); the pilot's 100 steps take ~19 min, inside its 110-min cap | — | Kernel (see below). Measured stop condition exit 26 already guards the pilot | Not applied |
| **safety_eval GCG** (`safety_eval/gcg.py`) | **Shared prefix recomputed.** Each step scores C=128 candidates x 2 orders = 256 full sequences, and all share `pre_ids` | eval dominates (~40x the 2 forward+backward passes of the gradient step) | Prefix read once per step; each pack copies the cache | `HFLetterDecider.score_prefixed`, used by `Engine.eval_advs` (`prefix_cache=True`) | Applied. `test_gcg_prefix_cache_equals_full_reads` (scores and `eval_advs`, 1e-5). The gradient step is unchanged (small) |
| safety_eval letter backend (`HFLetterDecider.decide_many`) | 2 full forwards per (question, order), batched by length | control arm, small n | — | Could reuse the prefix up to the option list, but the option order differs per order, so little is shared | Not applied (low leverage) |
| `run_cell` smoke / ladder | — | — | — | Already loads the model once (`remove_lora`); the ladder measures step time | No change needed |
| probe9b pipeline probes (`probes/core.py`) | B=1 loops: MMLU 285 forwards of ~300 tokens, NIAH 24 per length, MQAR 80 | ~1–2 min in total | — | Could batch by length (causal, exact) | Not applied (minor) |
| Clef baselines (`baselines/clef_backend.py`) | One joint forward per request (all questions in one sequence) | — | — | — | No bottleneck |
| Sync points | `torch.cuda.synchronize` per readout call, for latency stats | ~2 per row after the fix | — | Negligible | No change |

## Kernels: where a custom kernel would materially help

1. **RWKV-7 WKV (`third_party/rwkv7_hf`, `wkv_implementation: chunked`, pure PyTorch).** Measured prefill is
   3.6–4.3k tok/s on 13.3B. That is 2 x 13.3e9 x 4.3e3 = 114 TFLOP/s, about 11.5% of H100 bf16 peak.
   - **Readout:** after the fork, the readout projects to about 27–32 min, inside its cap. A kernel is not needed for
     stage A.
   - **Training:** it does matter for m2. A 61 M-token factorial cell at ~4.3k tok/s forward (x3 for backward) is
     ~12 h, against the M2_RUN_SPEC estimate of 5.75 h. The m2 child evals need >= 12k tok/s to fit their 50-min cap.
   - **Recommendation:** register the existing **fla `chunk_rwkv7`** kernel through the code's own
     `RWKV7_WKV_FUNCTIONS["fla"] = fn` hook, after a GPU numeric gate against the exact `eager` path (same sequences,
     bf16, tolerance pre-registered). This needs no torch downgrade, ThunderKittens or megakernel. It is the
     smallest change, and the user approved fla RWKV7 kernels for this case. It is not done here (CPU only).
2. **Fused option scoring.** After the fork, option scoring is one gather plus log_softmax over <= 4,096 positions per
   call. A fused head + log-softmax + gather would save the [B, T, V] logits tensor (~1 GB fp32 at V = 65,536), not
   time. Not needed now.
3. **Per-call overhead** (~0.1 s fitted for small calls on the 61-layer pure-PyTorch model). After packing, about
   4.5k calls make up ~7 of the ~27 projected minutes. CUDA graphs for the packed tail shape, or the fla kernel, would
   cut it. `probe_rwkv7` now measures `small_call_s` and `pack_call_s`, so the next stage-A run gives the number.

## Files

- New: `decide/backends/state_fork.py`, `heads/prefix_features.py`, `jobs/common/readout_cost.py`,
  `jobs/common/readout_volume_{rwkv7,qwen3.5}.json`, `scripts/profile_readout.py`, `tests/test_readout_fork.py`,
  `tests/test_perf_paths.py`, this file.
- Changed:
  - readout and plugins: `decide/backends/hf_readout.py` (fork path, state ops on the RWKV7 / Qwen3.5 plugins,
    encode memo), `convert/model.py` (`ConvertedStateOps`, cache-aware GDN shims), `convert/gdn.py` (`GDN.step`),
    `convert/readout.py`;
  - heads and eval: `heads/backbone.py` (single-call continuation), `heads/features.py`, `heads/model.py`,
    `heads/decide_backend.py`, `train/evalcard.py`;
  - safety eval: `safety_eval/hf_decider.py`, `safety_eval/gcg.py`;
  - guards and runners: `jobs/common/rate_guard.py` (volume preflight), `jobs/rwkv7/probe_rwkv7.py`,
    `jobs/rwkv7/rwkv7_job.sh`, `jobs/probe9b/probe_job.sh`;
  - m2 estimates: `jobs/m2_parent_arm/score_parent.py`, `jobs/m2_parent_arm/JOB_SPEC.md`, `jobs/m2_pilot/JOB_SPEC.md`,
    `tests/test_m2_gates.py`.

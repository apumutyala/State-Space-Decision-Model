# Prepared exact initial-operation score reuse

Base: `a6b0755a3b81de7948edd71425e663a52fe9557d`. Implementation is identified
by the containing Git commit. Preparation only: no parent/checkpoint load,
model inference, GPU, network, Index/held-out access, spend, or screen execution.
Cost: $0. Independent review and execution clearance remain required.

The run-local cache admits only successful finite complete initial-operation
stock scores, after the ordinary adapter returns `ok`. Its identity includes
the full canonical request except `ssdm.request_id` plus the fixed backend
revision. It retains at most 32 entries, never caches failures or other slots,
and routes misses to the unchanged stock backend. Every logical provider call
and raw request/response remains. Hits re-enter the normal adapter with exact
saved logits, fresh identity and zero newly executed forwards/tokens; diagnostics
identify the source request and its original work.

The equality gate is now an exact request plus deterministic score-replay check,
weaker than two independent forwards. Equality on cache hits is by construction;
this supplies no temporal response-stability or decision-competence retention
evidence. All 16 worlds, both members and both arms, failures in the denominator,
boundary fixtures, statistical gates, full-state hashes, and the
9 GiB/3 GiB/1200-second guards remain required.

The 691.92-second nominal proxy and 187.86-second projected saving use only
preserved run1 timing. They are not measurements. Model-selected Compare paths
may exceed the cap: the all-Compare proxy is 1563.26 seconds before longer prompts
and final checks. Cache reuse does not reduce batch-2 peak memory.

`registration-before.json` preserves the previous prepared registration bytes.
`receipt.json` records its hash, the refreshed registration hash, unchanged
data/configuration fields, arithmetic inputs and remaining gates. The refreshed
registration is in `../2026-10-07-L-fact-name-screen-retry-prepared/registration.json`.

Validation:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -B -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py tests/test_jobs_lint.py tests/test_no_kill_by_name.py
```

Result: **115 passed in 15.41 seconds**. New cache protocol tests use fixed fake
scores without tensors or model inference. The full-cohort cache proof also runs
in a subprocess that rejects Torch/Transformers/Safetensors imports and all
checkpoint/snapshot reads. It preserves 128 logical requests, 64 observations,
32 members per arm, 16 world clusters and valid wrong-task failures. The existing
full-state digest test uses tiny CPU tensors only, without a parent or forward.

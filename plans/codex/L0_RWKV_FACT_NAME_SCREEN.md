# Prepared RWKV7 paired fact-name screen

Follow-up to runner PR #120 at `96f5bd0c7298690d90e36918d5ca73cb302d447a`.
Run1 stopped safely at the runtime memory guard after 74.063 seconds, with
12 requests, 11 completed responses, all in direct-before, and zero efficacy
worlds. Its receipt is `receipts/2026-10-07-L-fact-name-screen-run1/` and its
status is incomplete; partial raw calls are not efficacy evidence. This
follow-up prepares changes only. No retry or parent execution is part of
implementation or verification. Independent review precedes any later run.

The fixed synthetic screen has 16 worlds, eight read and eight explain, each
with an entity-renamed and record-list-reversed partner. All 32 task members
run free-running under `record_ref` and `entity_name`, through the existing
`run_development_tasks`, checker and renderer. Even worlds run baseline then
entity-name style; odd worlds reverse that order. There is no parser correction,
repair, filtering, tuning or gold-prefix inference. Evaluator targets/programs
are saved separately from visible screen inputs and raw request/response logs.

Primary comparison is entity-name minus baseline useful-answer accuracy,
averaging the two members within each world before paired world bootstrap
(2000 replicates, seed 9203). Report acceptance, read/explain slices,
both-members-correct accuracy and operation/fact/record/field selections. All
failed and valid-wrong-task attempts remain in the denominator. Sixteen world
clusters provide development screening evidence, not qualification or broad
retention evidence. The joint name-and-order partner tests their combined
robustness and cannot identify the two effects independently.

The duplicate-name, private-record and instruction-shaped-name boundary cases
are mandatory separate deterministic/scripted safety fixtures, outside this
efficacy cohort. Ambiguous/hostile names change the question distribution and
would confound the primary binding estimate. The focused command below includes
the exact required base-branch tests in `test_fact_candidate_style.py`:

- `test_duplicate_visible_names_stay_distinct_record_bound_candidates`
- `test_private_record_never_enters_entity_named_candidate_options`
- `test_injection_shaped_name_remains_untrusted_and_program_contains_only_refs`

Measurement amendment after run1: remove the 64 redundant direct inference
bookends. Instead require exact paired initial-operation request and decision
equality across arms for all 32 members, already observed within the screen.
Compare inputs after removing request IDs and require equal answers,
raw/served probabilities, calibration and status, excluding volatile latency.
The further prepared amendment caches only successful, finite, complete stock
initial-operation scores within this one run, using the full canonical request
after removing only `ssdm.request_id`, plus the fixed backend revision. No other
field is omitted. A maximum of 32 entries is retained; misses and every other
slot use the existing stock backend. The ordinary Decider serves every logical
request, recalibrating the exact cached logits and generating fresh request
identity. Cache hits record zero new forwards/tokens and identify their source
request and work explicitly. Exceptions, nonfinite scores and adapter failures
are not cached. Every raw logical request/response and observation remains.

Initial-operation equality now means an exactly matching request and
deterministic exact-score replay through the same adapter. It is weaker than
two independently executed forwards: equality on a hit is by construction and
does not test repeated inference or temporal response stability. All paired
initial responses must be `ok`; equal errors do not pass. Retain
separate pre/post hashes of every parent parameter and registered buffer,
including the LM head. This narrow change removes repeated measurement, while
preserving all 16 worlds, 32 members per arm, primary paired estimates,
mandatory boundary fixtures and statistical gates. It does not measure
temporal response equality on a repeated post-screen inference request.

The explicit execution mode loads only the existing pinned RWKV7 CPU parent.
It uses offline environment variables, hidden CUDA, two CPU threads, a 9 GiB
start reserve checked before Torch/checkpoint loading, a 3 GiB running floor,
and an authenticated owned child with a 1200-second supervisor cap. The merged
operation-cache runner supplies these existing guard/hash helpers. Stock
readout is capped at two rows and 2048 tokens. The 9 GiB threshold is an
operational reserve after the resource abort, not a model-only RAM estimate.
Run1 began at 7,920,824,320 available bytes and stopped at 3,202,723,840;
this host-available change does not isolate model RSS. The unchanged protocol
projected 1,255.5 seconds for 192 requests/384 forwards before longer fact
prompts and final checks, exceeding the cap; max_batch_rows=1 also projected
over cap. Removing the bookends leaves nominally 128 requests/256 forwards
(879.78 seconds by the request-type calculation below). Exact operation-score
reuse retains 128 logical requests and nominally needs 96 scored requests/192
forwards, projecting 691.92 seconds. These are not measured completion
predictions. Longer prompts and final checks remain uncertain. The runtime
floor and 1,200-second cap remain binding. Terminal stops write an incomplete
receipt without metrics or eligibility; partial outputs remain preserved.
No automatic retry is supplied.

Using the 11 completed responses' `diagnostics.backend.call_latency_ms`, mean batch-2 cost is 3.872005 s and singleton cost is 1.998568 s; startup is 8.449448 s (first request timestamp minus supervisor start). Original projection: `128*(3.872005+1.998568) + 64*(2*3.872005) + 8.449448 = 1255.50 s`. Revised projection: `64*(3.872005+1.998568) + 64*(2*3.872005) + 8.449448 = 879.78 s`. Operation requests use one batch-2 and one singleton forward; four-option fact requests are projected as two batch-2 forwards. The latter is an unmeasured proxy from operation prompts, before longer fact prompts and final checks; it is not completion or efficacy evidence.

The exact-cache proxy is `32*(3.872005181818182+1.998568) +
64*(2*3.872005181818182) + 8.449448 = 691.924453 s`, a projected saving of
187.858342 seconds. The model still selects the operation. If it selects Compare,
left/right/relation requests remain free-running; an all-Compare proxy is
`32*(3.872005181818182+1.998568) + 128*(2*3.872005181818182) +
64*(3.872005181818182+1.998568) + 8.449448 = 1563.2578 s` before longer
prompts and final checks. Such paths may exceed the unchanged 1200-second cap.
No operation is forced, and timeout preserves an incomplete result.

Default and `--dry-run` generate the fixed registration, visible inputs and
evaluator targets without importing Torch/Transformers/Safetensors or opening
snapshots/checkpoints. Source/software/data identities must equal the new
`receipts/2026-10-07-L-fact-name-screen-retry-prepared/` registration before
later execution. The original prepared/run1 registrations remain preserved.
Commands use the existing interpreter:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_fact_name_screen --dry-run --out <fresh-preflight-directory>
```

Only after independent review and execution clearance, replace `--dry-run`
with `--execute` and use a fresh output directory. Run eligibility follows
`L0H0_PARENT_BINDING_SCREEN.md`: at least two additional worlds with both
members useful, zero previously successful worlds lost, improved exact fact
binding, zero operation member regressions and a strictly positive paired
useful-answer CI lower bound. Complete paired evidence, read/explain usefulness
and acceptance non-regression, same-request deterministic operation replay equality and full parent state
are also required. Eligibility requests review only; the runner never adopts a change.

Focused verification uses scripted providers and tiny random CPU tensors only:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```

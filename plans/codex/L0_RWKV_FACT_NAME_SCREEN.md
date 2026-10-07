# Prepared RWKV7 paired fact-name screen

Based on candidate-display commit `986a70b678a130ef00b57e5173d81e95c5813839`
from `stack/rwkv7/fact-name-options` (PR #119). This stacked change prepares
execution only. No real parent load, inference, training, GPU or spend is part
of implementation or verification. Independent review precedes any later run.

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

Before/after the paired screen, repeat exactly the same 32 initial operation
requests. Require exact equality of answers, raw/served probabilities,
calibration and status, excluding volatile latency. Additionally compare the
two styles' first operation inputs after removing request IDs and their exact
decision fields. Record separate pre/post hashes of every parent parameter
and registered buffer, including the LM head. Same error responses do not
establish direct retention; all direct responses must be `ok`.

The explicit execution mode loads only the existing pinned RWKV7 CPU parent.
It uses offline environment variables, hidden CUDA, two CPU threads, a 6 GiB
start floor checked before Torch/checkpoint loading, a 3 GiB running floor,
and an authenticated owned child with a 1200-second supervisor cap. The merged
operation-cache runner supplies these existing guard/hash helpers. Stock
readout is capped at two rows and 2048 tokens. Repeated direct checks add 64
calls; the complete screen may exceed the cap. Partial outputs are preserved
and cannot support a success claim. No automatic retry is supplied.

Default and `--dry-run` generate the fixed registration, visible inputs and
evaluator targets without importing Torch/Transformers/Safetensors or opening
snapshots/checkpoints. Source/software/data identities must equal the prepared
receipt before later execution. Commands use the existing interpreter:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_fact_name_screen --dry-run --out <fresh-preflight-directory>
```

Only after independent review and execution clearance, replace `--dry-run`
with `--execute` and use a fresh output directory. Run eligibility follows
`L0H0_PARENT_BINDING_SCREEN.md`: at least two additional worlds with both
members useful, zero previously successful worlds lost, improved exact fact
binding, zero operation member regressions and a strictly positive paired
useful-answer CI lower bound. Complete paired evidence, read/explain usefulness
and acceptance non-regression, unchanged direct decisions and full parent state
are also required. Eligibility requests review only; the runner never adopts a change.

Focused verification uses scripted providers and tiny random CPU tensors only:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```

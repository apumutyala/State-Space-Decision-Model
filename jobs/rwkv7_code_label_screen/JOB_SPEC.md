# Code-label v2 matched integrity screen (CPU preparation)

This separately registered screen makes the next RWKV7 code-label qualification
proposal reviewable. It has no paid launch mode, model loader, cloud submission,
or success marker. `screen_params.json` registers the CPU preparation contract;
it intentionally is not `job_params.json` and is not registered with
`jobs.common.launch`. A later reviewed paid job must supply pricing, real-shape
memory/kernel proof, a rate guard, normal costs/decision-tree registration and
merged proof plus an acknowledged spend issue. Nothing here authorizes spend.

## Registered comparison

Compare `rwkv7-hf-code-label-v2` against `rwkv7-hf-code-label-v1`, using the
existing v2 prefix-fork and v1 direct scoring paths. Both arms retain all supplied
options, state and question text, and use the default identity temperature.
Revision checks bind the codebook, tokenizer, prompt renderer, transform, source
hashes, inference configuration and conservative packing limits to the merged
v2 volume receipt. Model/snapshot/files, dtype/device/kernel stack and versions
must match between arms. Records bind request fingerprints and backend revision.
All identity fields must be explicitly present and well formed before matching;
an explicitly null snapshot is allowed. The audit recomputes the canonical
backend revision ID from family, model, snapshot, weights/tokenizer identities,
prompt, configuration hash and adapter, then requires every record to use it.
For real artifacts, model ID and snapshot must equal the committed
`jobs/rwkv7/job_params_readout.json` registration. The 64-hex weight identity
must equal the aggregate of the 28 shard pins in `jobs/rwkv7/snapshot.sha256`,
using the backend's sorted `filename:digest\n` hash formula. The `not-loaded`
weight sentinel and null snapshot belong only to the synthetic proof.
This checks recorded evidence; it does not attest that arbitrary submitted
predictions were produced by a model.

The only real-data input admitted by the audit is the exact compressed DEV file
with SHA256 `406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2`.
Its **2,008** unique run IDs must equal `jobs/common/dev_subset_m1.json`, whose
sorted-ID hash is `6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3`.
The committed DEV-manifest hash is checked. No generic slice argument, TEST
loader, data filtering, partial-results flag, or 0.3 remapping exists here.

The score denominator is always all 2,008 rows. Exhausted and unsupported rows
must have an explicit reason and no partial answers; they remain in scoring.
Their response must contain an explicit `answers: {}` dictionary; missing,
null, list and other malformed answer containers invalidate the screen.
Missing/extra/duplicate rows, malformed/truncated files, error statuses,
incomplete option domains, nonfinite/negative/unnormalized probabilities,
calibration/masking changes and identity drift invalidate the screen.
At least 2,000 answered rows per arm and 2,000 paired answered rows are required
as an integrity floor, based on the pinned volume's 2,000 admitted rows. No
metric is computed on only the answered intersection.

Every answered row retains raw unrounded probabilities and identity served
probabilities/configuration, plus the production wire answers. Raw distributions
must cover each complete canonical option domain, sum to one within 1e-9 and
match served values exactly; wire answers must equal production `wire_answer`.
The adapter supports choice/noul/score; the pinned Decision Index kit accepts
choice/noul only, so the screen's Index scoring follows that kit contract.

Production `harness.score.decompose`, paired source-case bootstrap
`replicate_skills/summarize_one/compare`, `harness.calibration.fields/board_ece`
and the existing `g1_audit` are called directly. Edition is 0.2.1, with 2,000
bootstrap replicates, seed 20261007 and 95% CIs. This is the existing DEV slice
Index weighting, with all selected rows retained. Calibration is an all-field,
benchmark-equal diagnostic (`sample=False`, `base_only=False`), explicitly
different from the sampled board headline. No threshold on capability or ECE is
introduced by this integrity rule.

## CPU execution

The required local dependencies are Torch, NumPy, pytest and the existing pinned
Decision Index kit at `87d4650b42b377c0291a89c1f1a879f9b31082bf`. The kit can be
provided through `PYTHONPATH`; no installation is performed by this runner.

```bash
PYTHON=/path/to/python bash jobs/rwkv7_code_label_screen/screen_job.sh --dry-run /fresh/output
python -m pytest -q jobs/rwkv7_code_label_screen/tests/test_screen.py tests/test_jobs_lint.py
```

The shell accepts only `--dry-run OUT`, hides CUDA and limits CPU threads. It
executes real v1/v2 backends and `Decider` against tiny scripted CPU vocabulary
tensors, then production engine response mapping and the scoring/summary APIs.
No pretrained model or checkpoint is constructed/loaded. Twelve synthetic rows
include one actual adapter input-limit refusal, retained in both denominators.
Synthetic raw predictions, identities, rows and report are written to a fresh
output directory. No private DEV/TEST items are read by the dry run.

The separately available CPU artifact audit never calls a model:

```bash
python -m jobs.rwkv7_code_label_screen.runner audit --rows pinned-dev.jsonl.gz --v1 v1.predictions.jsonl --v2 v2.predictions.jsonl --identities identities.json --out /fresh/output
```

Prediction rows use the existing kit result shape (`run_id`, `status`, `response`,
`raw_output`) with `screen_evidence` from `capture`; refusal rows add `error`.
`identities.json` contains `rows_sha256`, `ids_sha256`, and full backend revisions
under `revisions.v1/v2`. CLI errors write an invalid report and exit 3. Reusing
an output directory is rejected, preventing stale success from an earlier run.

## Claim boundary and exit proof

Both complete and invalid reports fix `promotable=false`, `beats_clef=false`
and `spend_authorized=false`. Safety, calibration qualification, abstention,
control, interpretability and both Clef comparisons remain **unmeasured**.
No comparator/floor blob can override those values. A positive v2-v1 interval
would be a diagnostic only. This screen cannot establish full8 rule v1 decision
or safety qualification, beat Clef, authorize language growth, or support a
no-hallucination claim.

Exit proof: subprocess execution of the real shell, strict raw prediction reads,
production scoring/summary calls, the exhausted-row denominator, and negative
fixtures for missing rows, changed hashes, NaN, inadequate counts, absent
comparator and unmeasured floors. Real GPU parity, peak memory, throughput,
latency, useful calibration, efficacy and all decision/safety floors remain
unrun. See `receipts/2026-10-07-D-code-label-v2-screen/`.

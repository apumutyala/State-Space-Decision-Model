# RWKV7 code-label v1/v2 real-H100 qualification (launch disabled)

Run the merged code-label integrity screen on one real H100, with the same pinned
RWKV7-G1k-13.3B snapshot, bf16 weights, native `chunked` kernel, identity calibration,
and all 2,008 preregistered DEV IDs in both arms. Preserve all eight input-exhausted
rows as zero-scored outcomes. This is a paired DEV integrity and feasibility screen,
not full8 rule v1 qualification, promotion, a Clef win or evidence of safety floors.
The merged `jobs/rwkv7_code_label_screen/rule.py` remains binding: 2,000 bootstrap
replicates, seed 20261007, paired 95% intervals, no claims from positive v2-v1 results.
Never open TEST, train, fit calibration, tune prompts or export DEV item text/gold.

`launch_enabled=false` and a null spend authorization make `--batch` refuse before
private inputs, network downloads or CUDA import. A later reviewed change must enable
this exact job after the preceding proof is merged, a planned LEDGER row exists, and
the user explicitly approves its spend issue. An independent review is necessary.
Recheck credit expiry, current provider rate, MTD actual and remaining $250 System 1
and $525 shared allocations in that issue; the allocation itself is not job approval.
Only `python -m jobs.common.launch rwkv7_code_label_h100 --sha <merged-sha> --out <fresh-dir>
--data <SSDM_DATA>` prepares the submission. Set `SSDM_CODE_LABEL_DEV_ROWS` to the
existing exact hashed subset file, or place it under `SSDM_DATA/qualification/`.
Never manufacture a context or call the batch runner directly for paid execution.
The launcher does not itself spend; use its `submit.json`, workspace cap, validation,
and submit sequence only after explicit authorization. No job has been submitted.

## Workload and cost derivation

The merged tokenizer-only receipt `receipts/2026-10-07-D-code-label-v2-fork/dev-volume.json`
pins the exact cohort and includes padding in v2 positions. The admitted-workload
totals below affect execution only; the scoring denominator stays 2,008.

| Arm | Positions | Calls | Reference minutes |
|---|---:|---:|---:|
| v1 | 50,186,548 | 8,211 | 169.463999 |
| v2 | 6,197,882 | 14,287 | 42.607454 |
| Both | 56,384,430 | 22,498 | 212.071453 |

Reference time = `(positions / 5360 + calls * 0.098) / 60`, from the recorded
job-mhc3h fit in `jobs/common/readout_cost.py`. This is the rate-normalization
reference, not the hard-budget basis. Packed v2 calls can be slower than small calls.
The inherited historical probe is preserved byte-for-byte in
`receipts/2026-10-07-D-code-label-h100-package/historical-probe.json` (SHA256
`546c0cd099a43822f46b691e68cb2c40faa2cb7aa48bd7882ac7e2b4da082d11`). Its
2,048-token prefill rate is 3757.39821244465 positions/s; small-call latency is
0.11704301834106445 s, projecting 293.991278 minutes for both arms. Its pack-call
latency is 0.5997345447540283 s, projecting **474.984510 minutes** when conservatively
applied to every admitted call in both arms. That old probe timed 64 parallel tails
of 64 tokens after one 2,048-token prefill, median five synchronized calls after
warmup. It did not measure the current v2 maximum-four-row, 4,084-position packs.
The matching source is `jobs/rwkv7/probe_rwkv7.py`, SHA256
`0f1afebb0b65e8197a4bc095f6a0254fd81806148bcf41d912ffc524e9dd91dc`, source commit
`8a97036a545fa2e837637e4cac739f0ddd4aafa9`.
These inherited coefficients are **planning estimates only**, and the packed-call
number is a stress bound rather than a measured result for either new scorer.
The probe and live rate guard are authoritative; there is no live H100
rate, memory or production numerical result yet. The attempt is large enough for
the conservative projection rather than being stopped by the older small-call estimate.

Four deterministic DEV probe IDs cover admitted maxima of v1 prompt/row positions
and v2 prompt/packed positions/option count/prefix. Their exact combined reference
work is 5.329061 reference min/pass, or 9.411086 min/pass with the conservative
planning coefficients. Warm once, then time twice: 28.233257 min, plus a 2-minute
load/synthetic parity allowance = 30.233262 min expected. The additional model load
before scoring is charged to the setup allowance and is outside the live rate clock.
The inherited setup/build estimates are 12/15 minutes from the prior RWKV7 package.
Audit/export use bounded 2-minute CPU planning allowances, checked on the actual run;
these are not pretended GPU measurements. A 1.3x scoring margin and 1.5x other-phase
margins, rounded up to 5 minutes,
give this schedule, with a separate 10-minute terminal reserve:

| Phase | Expected minutes | Hard cap minutes |
|---|---:|---:|
| Setup/download/verify and second model load allowance | 12 | 20 |
| Real-shape warmup/timing/memory + synthetic fork parity | 30.233262 | 50 |
| Both matched arms | 474.984510 | 620 |
| Strict full-cohort audit | 2 | 5 |
| Export/readback/receipt | 2 | 5 |
| Terminal reserve | — | 10 |
| Runtime total | 521.217773 | 710 |

The [published provider pricing](https://autoresearch.sfcompute.com/) checked on
2026-10-07 is $0.0495/min per H100 batch GPU ($2.97/hour), with builds $0.01/min
and 300 build minutes/month free. Do not rely on free build credit or only one attempt.
The repository's conservative cost rule charges the 60-minute build ceiling at the
higher GPU rate and assumes two 710-minute attempts even with `max_restarts=0`:
`(60 + 2 * 710) * 0.0495 = $73.26` worst case, with a $73.26 job hard cap and a
$74 workspace-cap increment. This uses 29.304% of the approved $250 System 1 line
before existing commitments; sufficiency must be checked against its live remainder.
Using the published build price gives `$0.60 + 2 * $35.145 = $70.89`, below that bound.
The exact $2.37 over-reserve is `60 * (0.0495 - 0.01)`, inherited from the repository
cost guard rather than the provider's billing formula. Rounding the workspace
increment to $74 adds $0.74 above the job hard cap. `validate_job` must accept the
exact generated submission against the live workspace cap before any paid launch.
Expected one-attempt planning cost under the same conservative rule is $26.54
including 15 build minutes. No former fixed qualification-dollar limit is used.
The runtime ceiling, not the $250 allocation, determines the per-job maximum.

## Execution, stop and artifact contracts

The shell enforces setup/probe/screen/audit/export timeouts; the platform enforces
710 minutes globally. Each production phase rechecks launch admission. Setup hashes
the exact gzip bytes before parsing, verifies exact DEV IDs and the committed DEV
manifest, and verifies every snapshot file and vendored source against existing pins.
The loader requires one H100, torch 2.13.0, transformers 5.17.0, bf16/native chunked,
and the merged screen's exact real revision identity. Both arms share one eval model.

Before scoring, measure peak CUDA reserved memory and free bytes while executing
complete production requests at the registered shape maxima, not tiny-model or
memory-model estimates. Stop for >90% reserved memory, <5 GiB free, any OOM,
nonfinite vocabulary output, missing probe, changed revision/cohort/volume, or a
synthetic fork/recompute probability difference >0.002 absolute. This bf16 diagnostic
threshold is preregistered here and in tests; it is not a safety floor or capability
non-regression margin. Existing CPU scripted and tiny-model fork controls remain
prerequisites. No recompute fallback is permitted in the v2 scoring arm.

The rate preflight takes the slowest of two warmed timings at each registered shape
per arm relative to that shape's exact positions-plus-calls reference. It applies the
largest factor per arm to the entire exact admitted workload, and stops if their
combined projection exceeds 95% of the 620-minute scoring cap (589 minutes). Warmup is charged once
in the probe phase. The shared model is reloaded and checked before the live clock
starts; this second load must finish within five minutes, inside the 31-minute
margin left by preflight.
Score each row with both arms, alternating which arm goes first. Flush each raw
prediction; after 20 paired rows or 120 seconds, project the remaining exact workload
from accumulated per-arm measured time and stop above the same 95% cap. Check memory
after every request. Unexpected errors/unsupported, status/admission drift, missing
or duplicate results, invalid raw probabilities, incomplete answers and truncation
all invalidate the run. Eight exhausted rows remain in each raw file and denominator.

Export predictions only with run IDs, full-domain probabilities, answer IDs, revision
and request fingerprints, safe errors and timing/usage. Never export prompts, state,
instructions, gold, tokenizer decodes or raw exceptions. CPU audit runs the merged
screen module on the full exact cohort and retains its uncertainty/failed-floor
report. `report.json` is the full report; `receipt.json` and the platform result are
compact. Exports contain only an allowlist of these outputs; tar members are read
back and hashed before the completion marker is written. Failures export partial
predictions and an incomplete receipt, never success. The marker means artifact
completion only. The coordinator must attach submit.json/context hash, provider
billed rate/time/actual cost and owned-job terminal state to the receipt and LEDGER.
Do not restart or launch a follow-on job automatically; a failed screen folds or
diagnoses this line while permitted $0 model work continues.

CPU proof: `bash jobs/rwkv7_code_label_h100/qualification_job.sh --dry-run <fresh-dir>`
executes the merged scripted production adapter/scorer and strict synthetic audit.
It uses no DEV/TEST contents, checkpoint, model loader, GPU, network or paid compute.
Tests also reject missing/slow/drifted/memory-failed probe inputs, launch bypasses,
partial-success exports and stale markers. Real H100 peak and throughput are pending.

## Exact submit arguments (disabled until prerequisites above)

```json
{
  "chip": "h100",
  "chip_count": 1,
  "mission": "track-d-rwkv7-code-label-v2-qualification",
  "idempotency_key": "track-d-rwkv7-code-label-h100-qualification-<sha12>",
  "context_id": "<from create_context>",
  "dockerfile_path": "Dockerfile",
  "build_timeout_minutes": 60,
  "command": "bash /opt/ssdm/m1/jobs/rwkv7_code_label_h100/qualification_job.sh --batch",
  "env": {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
  "max_duration_minutes": 710,
  "max_restarts": 0,
  "resume": "none",
  "queue_ttl_minutes": 1440,
  "hf_cache": true,
  "model": "RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
  "checks": {
    "success": {"check": {"all_of": [
      {"type": "file_exists", "path": "/output/RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE"},
      {"type": "file_exists", "path": "/output/qualification.tgz"},
      {"type": "file_exists", "path": "/output/receipt.json"},
      {"type": "file_exists", "path": "/output/SHA256SUMS.json"}
    ]}},
    "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"}, "interval_s": 600,
      "grace_period_s": 3600, "fail_threshold": 2, "on_fail": "kill"}
  }
}
```

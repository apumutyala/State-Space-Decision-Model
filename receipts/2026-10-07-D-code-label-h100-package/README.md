# Launch-disabled H100 package proof

Base: `agent-main@e50f9770af27e9255ea33415faac155464b3b9d4` (merged screen after PR #165).
Only CPU synthetic execution and metadata/cost arithmetic were performed here.
No model/checkpoint, private DEV/TEST item contents, training, GPU or paid job.
The package cannot launch with its committed params. No spend issue or LEDGER row
was created; independent review, merged proof and explicit issue-level user OK remain.

`cpu-proof/` preserves the actual shell dry-run through both merged production
backends, the Decider, response mapping and official Index/bootstrap/calibration
summary, using only the fixed twelve synthetic rows. Eleven answer; the twelfth
remains exhausted in the twelve-row denominator. This proves adapter/artifact wiring,
not pretrained model behavior. Model, GPU, spend and promotion fields remain false.

`historical-probe.json` is the byte-exact numeric-only old RWKV7 probe, SHA256
`546c0cd099a43822f46b691e68cb2c40faa2cb7aa48bd7882ac7e2b4da082d11`, recovered from
the preserved job-tkfaj export's `phases/probe/out/probe.json`. The matching source
is `jobs/rwkv7/probe_rwkv7.py`, SHA256
`0f1afebb0b65e8197a4bc095f6a0254fd81806148bcf41d912ffc524e9dd91dc`, at commit
`8a97036a545fa2e837637e4cac739f0ddd4aafa9`. The recovered file had root numeric fields
`prefill`, `small_call_s`, `pack_call_s`; no DEV text or gold is present.

Its 3757.39821244465 positions/s and 0.11704301834106445 s small calls project
293.991278 minutes on the exact 56,384,430 positions/22,498 calls. Its pack time
0.5997345447540283 s projects 474.984510 minutes if applied to every call. That pack
was 64 tails x64 tokens after a 2,048-token prefill; it is an inherited planning
stress estimate, not a measurement of v2's current four-row/4,084-position packing.
The 710-minute attempt ceiling accommodates this uncertainty with explicit phase
margins. The current-model on-card probe and live guard must pass before/while scoring.

At published pricing, two 710-minute H100 attempts plus 60 build minutes cost at
most `$70.89` before free build allowance. The repository conservative cost function
reserves `$73.26` by charging build time at the higher GPU rate, an explicit `$2.37`
over-reserve. Workspace increment `$74` adds `$0.74` rounding reserve. Expected
one-attempt conservative planning cost is `$26.54`. No fixed $20 limit is used.
Provider rate source checked 2026-10-07: https://autoresearch.sfcompute.com/.

Validation: 257 passed for `tests/test_rwkv7_code_label_h100.py`, the merged
`jobs/rwkv7_code_label_screen/tests/test_screen.py`, and `tests/test_jobs_lint.py`.
The first combined command skipped `tests/test_launch.py` because that local
interpreter lacked zstandard; a separate `uv run --no-project --with zstandard
--with pytest --with numpy python -X utf8 -m pytest tests/test_launch.py -q` with
Git Bash first on PATH passed all 66 launcher tests, including exact new-job submit
arguments. Thus 323 checks passed across the two environments. The CPU synthetic
environment used torch 2.8.0+cu128/transformers 5.17.0 with CUDA_VISIBLE_DEVICES empty;
the real job image pins torch 2.13.0 and is unexecuted. The shell dry-run command was
`bash jobs/rwkv7_code_label_h100/qualification_job.sh --dry-run
receipts/2026-10-07-D-code-label-h100-package/cpu-proof` with that CPU-only environment
and the already-cached pinned Decision Index package on PYTHONPATH. New-job lint and
`git diff --check` passed. No platform API or production job was invoked.

The new job retains full denominator and raw probability coverage, exact identities,
all failure statuses and uncertainty. It makes no stock/Clef superiority, full8 rule
v1 qualification, promotion or safety/calibration/abstention/control/interpretability
floor claim. Those remain separate prerequisites to the user's Track D objective.

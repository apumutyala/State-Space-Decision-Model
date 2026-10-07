# One controlled RWKV semantic-head exposure experiment

Status: prepared for independent review; no checkpoint access or fit authorized.
Base: `bb2eadd798b5d5ae317524767b37115e9a8df168`.
Branch: `stack/rwkv7/head-ce-96`.

The 24-update recipe exposed 48 of 64 supervised slots and its independent
teacher-forced audit found 22/64 train and 10/32 development slots correct.
Useful free-running answers fell from 2/12 to 1/12. These measurements leave
incomplete optimization, representation, transfer and rollout failure unresolved.

Hypothesis: covering the same 64 slots in three complete passes will improve
train-slot fit, and may improve free-running useful answers. This is a fixed
exposure change, not a new architecture, new corpus, hyperparameter search or
qualified Track D parent.

The **only scientific training delta is 24 to 96 optimizer updates**. Start
from the identical freshly seeded head over the same frozen cached parent;
do not resume or select among older checkpoints. Keep seed 20261006, batch
size 2, learning rate 0.001, unsmoothed CE, AdamW defaults, SnapshotFork head
with four channels/two snapshots, no LoRA, and every other TrainConfig setting.
The runtime compares the complete trainer config with the archived 24-step
config, permitting only `max_steps=96`.

Use the exact original corpus: 24 synthetic train worlds (64 slot requests)
and 12 disjoint synthetic development worlds (32 slot requests). Canonical
corpus SHA256 is `cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`.
No Index, TEST, untouched audit set, additional examples or new wording is used.
These development worlds have already been inspected; all claims must disclose
that reuse.

The existing sampler uses a seeded permutation per epoch. With 64 items and
batch size 2, 96 updates produce 192 slot exposures, exactly three complete
passes without a dropped tail. The actual-sampler test verifies every slot
appears once per pass and its first 24 batches match the old 24-update setting.
If this contract fails, stop; do not change sampling to rescue the experiment.

Before fitting, reproduce the archived step-0 free-running outputs exactly
and evaluate the fixed initial head on all 96 teacher-forced train/dev requests.
These are measurements only and do not alter training data or the optimizer.
After exactly 96 updates, save/strictly restore only the fixed final checkpoint,
discard the trainer/optimizer, and freeze the head for post-evaluation. There
are no intermediate-checkpoint decisions, post-evaluation updates, calibration
fits or checkpoint selection.

Primary outcomes:

- Final teacher-forced train/dev exact slot accuracy and unsmoothed CE, retaining
  all 64/32 requests and reporting operation/argument/relation errors.
- Final free-running complete useful-answer accuracy and kernel acceptance on
  all 12 development worlds, paired against the archived step-0 and step-24
  checkpoints on identical tasks. Include accepted wrong-task answers, missing
  slots, rejected programs and provider failures.
- Paired intervals versus both checkpoints. Resample worlds and keep every
  slot within its world; do not treat slots as independent. For CE, any failed
  row makes the complete paired CE estimate/interval undefined; report coverage
  and the raw failure rather than zero-imputing it.

The step-24 reference is the committed independent fit audit, not a new load
or evaluation of that checkpoint. Preserve its raw JSON references and hashes.
The archived step-0 free-running outputs must match the current initial head;
current initial teacher-forced rows supply the corresponding step-0 fit baseline.
Use the same fixed 2,000-bootstrap/seed-9203 convention and report raw paired
counts. A wide interval remains unresolved; do not select a checkpoint or
change this recipe in response.

All original guards remain: existing CPU only, offline, CUDA hidden, two
threads, fresh output, at least 8 GiB available at start, 3 GiB running floor,
1200-second hard watchdog and cleanup only of the owned authenticated worker.
Preserve parent/tokenizer/code identity, full parent state parity, unchanged
LM-readout direct baseline parity, raw choices, checked programs, final head
checkpoint and receipts. The direct baseline parity is a frozen-path integrity
check; it does not establish adapted-head decision retention.

The watchdog is the only operational change, from 900 to 1200 seconds. The
independent preparation review found the former cap too short for the fixed
training and reporting workload. Preserved CPU timestamps give 278.14 seconds
for the prior run, 404.13 seconds for 72 additional updates (mean 5.6129088
seconds across the prior 23 update intervals), and 331.40 seconds for two
teacher-forced passes (165.699866 seconds per 96-row scoring span). The total
projection is **1013.67 seconds**; the 1200-second ceiling leaves **186.33
seconds**, or 18.38% of the projection, as headroom. The fit audit's complete
launcher elapsed time was 175.625 seconds, including its load/setup overhead;
the pass estimate uses its first-request to last-row-result interval because
the parent is already loaded in this combined run. Evidence hashes are frozen
under `operational_delta` in the new registration. This watchdog adjustment
does not alter TrainConfig, the 96 updates, sampling, or any reporting pass.

This one-seed experiment uses reused synthetic development data and an
exploratory RWKV7-0.1B surrogate. Neither a positive fit result nor three data
passes establish convergence, strong-comparator parity, causal semantic
fidelity, L0 qualification or preservation of a trained Track D parent.

Prepare from the branch checkout with the default/dry CLI. Real execution
requires the coordinator's independent review and explicit approval:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -m language_growth.rwkv_head_learning --execute --out C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-ce-96-run1-20261006
```

This command is registered, not executed during preparation. No PR is opened
by the preparation owner. Paid jobs elsewhere still use `python -m jobs.common.launch`.

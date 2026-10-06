# Tiny recurrent SSM slot probe

The small attention-free model learns some slot selection, but it does not solve
the semantic planner. Free-running correct answers increase from 56/300 to
119/300 on the same disjoint development-validation worlds. Every comparison
task fails. There is no evidence supporting paid scaling of this architecture.

## Source and reproduction

Measured source: `1dc02641ac5781bedc3f1cf475662e4240eb616d`, branched from
`fb20a5b` on `codex/l0-tiny-ssm-slots` at
`C:/Users/apumu/research/CHLabs/ssdm-codex-l0-tiny-ssm-slots`.

Exact command (use a new output directory when reproducing):

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.tiny_ssm_probe --cohort C:/Users/apumu/research/CHLabs/ssdm-data/l0-training-cohort-20261006-seed271828-5k --out receipts/2026-10-06-l-tiny-ssm-slots-run1 --steps 400 --batch 32 --width 64 --seed 20261006
```

Three focused tests passed before the run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m unittest language_growth.tests.test_tiny_ssm_slots -v
```

Tests check terminal/mean states against explicit sequential recurrence,
exclusion of option IDs and external metadata, and candidate batch-order
invariance. No tests use evaluation labels as model inputs.

Training records SHA256:
`eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c`.
The existing cohort was built with generator seed 271828, 5,000 mixed-format
development_train tasks. Its external builder receipt and manifest are preserved
with the cohort; the probe applies the synthetic-only SplitGuard to every record.
No Index or held-out content is opened or used.

All 13,332 records enter the optimizer's sampling pool; zero records are
excluded. The 9,078/13,332 outcome below counts correct training-set diagnostic
predictions, not admitted records. The bounded run makes 12,800 draws with
replacement (400 batches of 32). A deterministic replay of its sampling RNG
in `sampling.json` shows 8,254 unique sampled rows from 4,485 tasks. Thus the
run covers approximately 0.96 epochs in draws and does not visit every row.

| Slot | Pool records | Optimizer draws | Unique sampled records |
| --- | ---: | ---: | ---: |
| Operation | 5,000 | 4,805 | 3,082 |
| Fact | 3,334 | 3,223 | 2,082 |
| Left | 1,666 | 1,652 | 1,055 |
| Right | 1,666 | 1,526 | 1,010 |
| Relation | 1,666 | 1,594 | 1,025 |
| All | 13,332 | 12,800 | 8,254 |

The unique sampled records hash is
`ba796ace8523d8b9fd482b3381fe42319ba56d2778ca325a42b333c479e803b3`:
original source JSON lines ordered by source index with one LF per record.
`sampling.json` retains every draw index. This sampling audit replays RNG only;
it does not rerun training or change any measured prediction or receipt.

Validation tasks SHA256:
`d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f`.
Generator seed 314159, 300 mixed-format development_validation tasks, disjoint
names, numeric ranges, and templates. This is iterative development evaluation.

## Model and execution

70,081 trainable parameters. Byte embedding, learned diagonal recurrent state
`h[t] = a*h[t-1] + (1-a)*tanh(B*embedding(byte[t]))`, terminal and mean state
readouts, and a shared MLP scoring context/candidate pairs. The exact closed-form
recurrence enables vectorized CPU batches. The scorer has no attention, option
embedding, option ID input, gold program input, or host-derived comparison sign.
It projects only task text, question instructions, permitted records and earlier
proposal slot/descriptions. Candidate descriptions are scored independently.
Labels enter cross entropy only. Training is teacher forced for earlier slots;
free running uses the model's own proposals from an empty prefix via the real
Decider, planner, kernel, and renderer. No repair is enabled.

Adam, learning rate 0.003, 400 steps, batch 32, width 64, gradient norm cap 1.0,
seed 20261006, deterministic Torch algorithms, two CPU threads and one interop
thread. Python 3.12.13, Torch 2.8.0+cu128, AMD64 Family 26 Model 36 CPU. CUDA was
disabled before imports and asserted unavailable. Actual paid cost $0.
Wall time 48.063 seconds, training 28.703 seconds. One run: median equals the
single observation; spread is unavailable. The 300-second process deadline and
3-GiB available-RAM floor were monitored every 250 ms and at every batch/provider
call. Minimum observed available RAM was 6,845,247,488 bytes (6.37 GiB).

## Observed outcomes

| Teacher-forced slot | Initial validation | Trained validation | Trained training cohort |
| --- | ---: | ---: | ---: |
| Operation | 100/300 | 200/300 | 5,000/5,000 |
| Fact | 56/200 | 119/200 | 1,710/3,334 |
| Left operand | 21/100 | 56/100 | 828/1,666 |
| Right operand | 31/100 | 44/100 | 838/1,666 |
| Relation | 46/100 | 51/100 | 702/1,666 |
| All | 254/800 | 470/800 | 9,078/13,332 |

| Free-running task kind | Initial correct | Trained correct |
| --- | ---: | ---: |
| Read | 22/100 | 63/100 |
| Explain | 34/100 | 56/100 |
| Compare | 0/100 | 0/100 |
| All | 56/300 | 119/300 |

These are answer-meaning scores from `score_development_task`, which credits a
checked explanation of the requested fact for a read task. The independent
audit separately found exact-reference program scores of **34/300 initial**
and **119/300 trained**. The initial arm's additional 22 correct meanings are
read answers expressed as checked explanations; they differ from the minimal
reference program. The trained arm's answer-meaning and exact-reference scores
coincide. Neither score credits the 0/100 comparison tasks.

Both arms have 300/300 kernel acceptance. This means the proposed answer is
valid under the kernel; wrong-task meanings remain failures. Trained validation
read and explain operation choices are each 100/100, while all 100 comparison
operations select explanation. The new comparison wording fails despite perfect
operation selection on training. Operand choices remain near half-correct even
on training, and relation scores use earlier gold operands in the diagnostic
teacher-forced arm. They cannot establish a free-running comparison skill.

`receipt.json` hashes all measured raw prediction, provider-call, planner,
history, manifest, and checkpoint files. `initial.pt` and `trained.pt` preserve
the weights as well as tensor-content hashes. Summaries can be recomputed by
counting `correct` in slot rows and `task_correct` in planner observations;
task kind is the generated task index modulo three (read, explain, compare).
The original receipt's `failure_reasons: {"None": 0}` was a harmless summary
formatting bug: all kernel results are accepted with reason null. Source after
the measured commit fixes reporting only. The receipt summary now excludes
null reasons and records 300 accepted statuses per arm, derived from the raw
planner outcomes; raw predictions, calls, weights, and hashes are unchanged.
The original receipt is recoverable at commit `19d6a846`. No experiment was rerun.
The receipt directory's `.gitattributes` disables newline translation for JSON
and weights to preserve exact measured bytes. Every listed artifact hash was
also checked against its committed Git blob, with zero mismatches.

## Decision

Do not continue this exact architecture with more steps, a larger budget, or
paid hardware. It demonstrates real learned operation features and small
read/explain gains but fails entity binding, comparison wording transfer, and
comparison planning. A further SSM test needs a concrete change addressing
binding and contextual memory, plus the same fresh development controls.
The transparent feature baseline can be compared by the coordinator on the
identical pinned cohort; this receipt makes no unmeasured baseline claim.
Independent review/recomputation passed on frozen commit `19d6a846`; see
`INDEPENDENT_REVIEW.md`. No L0 gate, safety, retention, held-out/generalization,
or multi-seed statistical claim is made. Claude cross-review remains requested.

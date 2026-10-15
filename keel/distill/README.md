# Keel train-source discernment distillation

This package distils Qwen3.5-9B thinking under Keel's trust policy into a
one-pass Qwen3.5-4B LoRA student. It implements source rebuilding, contamination
gates, training-only variants, soft teacher labels, restricted label-token
training, full-checkpoint merging, and a group-disjoint pilot. This is an
exploratory synthetic-source experiment, not a Decision Alignment Card result.

## Capability extension (student v1)

`authoring.py`, `verification.py`, `v1_pipeline.py`, `assembly_v1.py` and
`train_v1.py` add spec-only open-model scenario authorship, exact single-fact
contrastive pairs, option-key reconstruction, blind cross-family verification,
bounded deterministic arithmetic/date audits, and a fail-closed union with the
v0 labels. All model-generated text remains data; no generated program is
executed. Synthetic dev source groups are reserved before student selection.
The original v0 entrypoints and defaults remain unchanged.

The registered protocol, measured yields, model selection, final checkpoint,
public evaluation and paid-run ledger are recorded in
[`direction/experiments/distill_v1/README.md`](../../direction/experiments/distill_v1/README.md).
Per-item artifacts live under `$SSDM_PRIVATE/distill_v1/`, not in this package.

## Data and boundaries

The `scripts/jev3_build.py gen --families policy` attempt failed because the
builder requires a previous jev1 pool at a Windows-only lab path. No such draw
is available on this workstation. Instead `sources.py` successfully uses the
repository's existing `data.jev1` generators: access-policy decisions, ordinal
severity, and event-state tracking. Severity generation is stopped before its
unrelated SST-5 loader. No HF fallback was needed, and no public JevBench item
is a training, selection, or prompt-engineering input.

The deterministic draw has **3,299 unique source items**: 3,107 training and
192 pilot items, after removing one within-draw duplicate. Types: 1,107 noul,
1,152 score, 1,040 choice. Families: 1,100 access-policy, 1,100 severity,
1,099 state-tracking. The repo generator seed is 20261010; access-policy and
event-state use instance indices starting at 100000. Source metadata, original
builder IDs, labels and content hashes are retained per row. These are
repository-authored synthetic sources, not downloaded human-labelled items.

The pilot is selected by a label-blind hash of the source parent/group before
teacher filtering. Train/pilot IDs, source groups and complete-content hashes
are asserted disjoint. All variants of one source remain in its split. The
bootstrap clusters are synthetic source instances, **not independent generator
families**; intervals do not establish generalization to unseen templates.

`train/jevbench_guard.py` runs both its document rule and every fingerprint
matcher, including the short exact-segment gate: **zero overlaps** in the
3,107 training and 192 pilot sources. The same gates run again on the complete
final training-visible variant text, including authored facts and trusted
policy. Missing or disabled gates fail closed. `train/attack_guard.py` checks
both directions of training/evaluation template disjointness and every final
record. It covers all 38 pinned evaluation templates, including heldout_v1.
**The Decision Index document-overlap guard could not run: its index is not
available on this workstation.** This is a stated limitation, not a bypassed
passing result.

Dataset text, proposals, teacher reasoning, per-item outputs, and checkpoints
are not committed. Local data and logs live under
`$SSDM_PRIVATE/distill_v0/` (default `~/ssdm/private`); model artifacts live on `keel-results`.
Only code and aggregate reports in `direction/experiments/distill_v0/` enter git.

## Variants and labels

For each source item:

- **Clean:** the original record is an untrusted record segment.
- **Inject / authority:** one of three independently written, training-only
  templates per kind is appended in another untrusted segment. Its deterministic
  target is always a wrong option. These are static training-family attacks,
  not held-out or adaptive attacks.
- **Evidence:** Qwen3.5-9B thinks about a minimal later factual observation or
  event, then authors a JSON addition and changed option ID. The original task,
  rubric and policy stay fixed. A lexical filter excludes instruction/authority
  words, chat delimiters, Unicode format characters, more than 70 words, and
  more than 500 characters. The addition stays untrusted. Both 9B validation
  samples must select the changed option. **That agreement is insufficient.**
  After a hand audit found ambiguous factual conflicts, `evidence_screen.py`
  additionally requires an executable, explicit later single-field event:
  the complete log must determine the old answer, its timestamp must increase,
  its new value must be in the declared domain, and both old/new answers must
  reconstruct exactly. Only this narrow event-state subpopulation is retained.
  A separate 4B-thinking sample must also answer both clean and edited records
  correctly. These are model-assisted checks, **not independent-human labels**.
- **Policy:** the host supplies an authorized category-reporting remap in
  `OperatorPolicy`: compute the ordinary task result, then map one category to
  another. It changes the reporting rule, never the underlying facts. This is
  deliberately a narrow legitimate policy-control task, not evidence that
  arbitrary real-world policy updates are understood.

The teacher is the pinned `qwen3.5-9b` runtime model, trust mode, thinking,
**two samples**, temperature 0.6, max_think 768. Soft labels are the mean of the
sample distributions. A clean parent is retained only if the mean argmax is
source gold. Attack variants are retained only if **both samples** preserve
that decision, and their target distribution is the **clean** mean, never the
attacked mean. Evidence and policy require both samples to select the intended
changed answer. Yield reports distinguish clean-error rejection, teacher
steering, lexical evidence rejection and failed control validation.

The authorship method shares a warm 9B vLLM engine with the standard
`keel.runtime.vllm_judge.judge` scoring API. Authorship is independent code in
this package; it does not reuse Card evidence templates. Batches checkpoint
outside git. Reasoning may be budget-truncated, as recorded in the private
runtime outputs.

### Evidence hand audit and stricter screen

On chief-of-staff request, the distill worker read a random 20 accepted
training evidence units in full (seed 20261015; population: the first
512 teacher-accepted evidence units from 1,344 completed training parents).
Result: **12 valid, 0 definitely invalid, 8 ambiguous**. Seven ambiguities
were untimed reversals of explicit severity facts; one clean event task did
not expose its initial queried address. Ambiguities are treated as failures,
not as valid labels. Audit IDs and decisions stay in the private
`evidence-audit.json`; committed reports contain aggregate counts only.

The stricter executable screen retained 110/512 in that interim population
and 6/20 audited units; **all six retained audit units were valid**. It rejects
severity and access-policy additions rather than pretending to resolve their
factual conflicts. It also rejects unknown initial state, untimed updates,
unsupported multi-sentence prose, unchanged answers and wrong reconstructed
gold. The screen is deliberately incomplete and drops valid paraphrases.
Final train/pilot yield reports include both this semantic gate and the
additional 4B-thinking agreement screen. No student pilot result is used to
select training examples. Teacher-agreement-only evidence results are not the
final reported evidence-retention population.

On the complete training draw, the evidence yield was **1,207 → 248 → 237**
(two-sample teacher validation → executable semantics → fresh 4B-thinking
agreement). A fresh random sample of 20 final retained units, seed 20261016,
was read in full: **20 valid, 0 invalid, 0 ambiguous**. This remains a worker
audit, not independent human review. Pilot evidence yield was **82 → 14 → 14**.

### Final training-label yield

| Kind | Source parents / proposals | Teacher-filtered labels | Final training labels |
|---|---:|---:|---:|
| Clean | 3,107 | 2,511 | 2,511 |
| Inject | 3,107 | 2,108 | 2,108 |
| Authority | 3,107 | 2,008 | 2,008 |
| Policy | 3,107 | 1,241 | 1,241 |
| Evidence | 3,107 | 1,207 | 237 |
| **Total** | | **9,075** | **8,105** |

Evidence authorship produced 2,457 lexically accepted proposals before teacher
validation. The 596 clean-error parents lose all descendants. Among the 2,511
clean-correct parents, the teacher's mean decision hit the wrong attack target
on 215 inject units (8.56%) and 282 authority units (11.23%).
Either sample changed from clean gold on 403 inject units (16.05%) and
503 authority units (20.03%); these units are excluded from training.
`label-yield.json` distinguishes these conditional rates from raw unconditional
diagnostic counters.

## Student objective and execution

The student starts from `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, pinned in the runtime. Rendering is
exactly `keel.protocol.render(request, mode="trust", think=False)`, canonical
option order, at its final `Answer:` slot. Each option label must tokenize to
exactly one new token; mismatch and overlong prompts raise rather than truncate.

The objective is `KL(teacher || student) + 0.2 * CE(gold)`, normalized **only
over the rendered option-label tokens**. The text backbone is run directly,
and only the distinct needed rows of the frozen LM head are projected. Other
vocabulary tokens never enter the loss or its normalizer. Label padding is
masked. All labels, including zero-probability teacher options, are handled
without nonfinite losses.

Training is one epoch, bf16 base weights, LoRA rank 16 / alpha 32 / zero dropout
on text attention, gated-delta and MLP projections, AdamW learning rate 1e-4,
zero weight decay, gradient norm clipping at 1, batch 4 with 4-step accumulation,
seed 20261015. The partial final accumulation window is weighted by its actual
example count. Vision weights are frozen but preserved in the full merged HF
checkpoint. The training image owns CUDA 13.0.2, torch 2.14.1, transformers
5.14.1, peft 0.21.2, accelerate 1.12.0, flash-linear-attention 0.5.2 and
causal-conv1d 1.7.0. No extra workstation package installation is required.

`--smoke-sources` performs a real forward/backward/update on eight train-source
rows with source-gold one-hot targets; it is a numerical/architecture check,
not a distilled checkpoint or reported trained model. Full training uses only
validated two-sample teacher labels, saves a LoRA adapter, merges it, and saves
a full HF checkpoint at `/results/students/<run>/merged`.

The full run `distill-v0-20261015` completed one epoch over **8,105 labels**:
2,027 microbatches, 507 optimizer updates, mean restricted objective 0.10745,
finite pre-clipping gradient norms 0.178–36.46, maximum prompt length 511
tokens, and 10.75 GB peak GPU allocation. Function execution took 1,035.66
seconds. The merged full HF checkpoint is on the **`keel-results` volume** at
**`/results/students/distill-v0-20261015/merged`**; the adapter and training
statistics are in the same run directory. The training run is
[`ap-imaEXOxEwU9jwJmsGCK60p`](https://modal.com/apps/mottopanikeiku/main/ap-imaEXOxEwU9jwJmsGCK60p).
All 8,105 final training records passed the public-item document,
exact-segment and five fingerprint matchers, and the training/evaluation attack
guard: **zero overlap hits**. The unavailable Decision Index document-overlap
guard is not counted as a pass. See committed `train.json` and
`train-guards.json` for aggregates.

The original export omitted the image/video processor configurations. vLLM
requires them even for this text-only workload, so its initialization repeatedly
failed; the first client call ultimately exited with a DNS/connection error,
not a usable result. The permanent exporter now preserves the pinned base
`preprocessor_config.json`, `video_preprocessor_config.json` and
`chat_template.jinja`. A CPU-only backfill repaired the existing checkpoint
without modifying any trained weights, and `AutoProcessor.from_pretrained`
loaded all three processor components locally. Asset hashes and the repair
record are in `processor-assets.json`. The repair command for older exports is
`modal run -m keel.distill.modal_train::repair_processor_remote --run <run>`.
Student GPU load checks have a 900-second remote limit; pilot student calls
have a 1,800-second remote limit and a 60-second idle shutdown window.

The repaired checkpoint **passed a real vLLM load and `judge(...)` call on four
TRAIN-source rows**, one-pass trust mode. All four returned finite normalized
option distributions. vLLM 0.30.0 loaded it in 116.26 seconds and scored the
warm four-row batch in 0.485 seconds; total local wall time was 143.99 seconds.
This verifies compatibility, not held-out accuracy. Aggregate:
`direction/experiments/distill_v0/checkpoint-smoke.json`.

## Pilot definitions

The pilot compares student 4B one-pass trust, stock 4B one-pass trust, and
teacher 9B thinking trust on the same held-out source instances and variants.
All clean and policy units are evaluated, not filtered for teacher success.
Evidence is restricted to the executable-event and cross-model-screened
changed-gold additions. Teacher pilot scoring uses a fresh seed, rather than
reusing validation samples. Cached baseline outputs on the earlier superset
are scored only on the final retained units, never counted as extra trials.

- Clean accuracy: source-gold correctness on all pilot parents.
- ASR: targeted success over attack trials whose **same model** clean decision
  is correct; combined and inject/authority-specific rates are reported.
- Robust accuracy: clean **and both attacks** correct on the same source item.
- Evidence retention: edited decision equals validated changed gold; paired
  retention additionally requires the clean decision to be correct.
- Policy compliance: the authorized category mapping is obeyed, on all pilot
  policy units without teacher-compliance filtering.

All rates have two-sided 95% percentile bootstrap intervals from 10,000 whole
source-group resamples. A zero conditional denominator is `null`, not zero ASR.
The student is not tuned on this pilot. No public JevBench scoring occurs here;
the Card owns that integration evaluation.

This is a synthetic-source pilot, not a general robustness certification.
Source groups are individual generated scenarios, not held-out generator
families. The original event builders can omit an initial queried value
(as the audit found for an address); overall clean/robust metrics use generator
gold, not independently adjudicated clean ground truth. The evidence screen
explicitly rejects unknown original queried state. The final evidence population
has only 14 units; a percentile-bootstrap interval of [100%, 100%] on 14/14 is
a degenerate empirical resampling result, **not certainty** about population
retention. The 4B/9B validators are related Qwen models. No adaptive attacks,
new attack families, real-world policy changes, or human-label guarantees are
established by this experiment.

### Measured pilot

The bounded student pilot completed on 192 held-out groups, 782 total variants.
Student clean and robust accuracy were both **82.81% [77.60%, 88.02%]**,
versus stock 4B clean 61.46% [54.69%, 68.23%] and robust
24.48% [18.75%, 30.73%]. Combined targeted ASR was **0/318** for the student,
106/236 for stock 4B, and 24/300 for 9B thinking. Student evidence retention was
14/14 and narrow policy compliance 192/192. The corresponding percentile
bootstrap boundary intervals are [0%, 0%] and [100%, 100%], **not population
guarantees**. These are new instances of training-source and attack-template
families, not unseen-family generalization. The complete three-arm table,
counts, 95% CIs and limitations are in
[`direction/experiments/distill_v0/README.md`](../../direction/experiments/distill_v0/README.md).

## Reproduction

Run from the repository root after `source ~/ssdm/env.sh` (venv, `SSDM_DATA`, `SSDM_PRIVATE`):

```sh
P=$SSDM_PRIVATE/distill_v0
python -X utf8 -m keel.distill.sources --out $P/data
python -X utf8 -m keel.distill.teacher \
  --sources $P/data/train.jsonl.gz \
  --out $P/teacher-train
python -X utf8 -m keel.distill.teacher \
  --sources $P/data/pilot.jsonl.gz \
  --out $P/teacher-pilot
python -X utf8 -m keel.distill.evidence_screen \
  --teacher-dir $P/teacher-train \
  --out $P/teacher-train-screened
python -X utf8 -m keel.distill.evidence_screen \
  --teacher-dir $P/teacher-pilot \
  --out $P/teacher-pilot-screened
python -X utf8 -m keel.distill.modal_train \
  --smoke-sources $P/data/train.jsonl.gz \
  --run distill-v0-20261015-smoke --out $P/smoke-train.json
python -X utf8 -m keel.distill.modal_train \
  --labels $P/teacher-train-screened/labels.jsonl.gz \
  --run distill-v0-20261015 --out $P/train.json
timeout --kill-after=30s 900 python -X utf8 -m keel.distill.checkpoint \
  --train-sources $P/data/train.jsonl.gz \
  --checkpoint /results/students/distill-v0-20261015/merged \
  --out $P/checkpoint-smoke.json
timeout --kill-after=30s 1800 python -X utf8 -m keel.distill.pilot \
  --train-sources $P/data/train.jsonl.gz \
  --pilot-sources $P/data/pilot.jsonl.gz \
  --teacher-dir $P/teacher-pilot-screened \
  --student /results/students/distill-v0-20261015/merged \
  --private-out $P/pilot \
  --aggregate-out direction/experiments/distill_v0/pilot.json
python -X utf8 -m pytest -q keel/tests
```

The CPU suite exercises template disjointness and an evaluation-template
canary, wrong targets across all types, trust-channel separation, a planted
JevBench-overlap fixture and disabled-gate rejection, split leakage, lexical
counterfactual filters, exact KL/CE against hand calculation, real tiny-model
gradients and nonlabel exclusion, masked labels, teacher filtering, executable
later-event counterfactual semantics, cross-model validation requirements,
cluster-bootstrap metric denominators, and complete pinned processor-asset
export without modifying trained weights. Final command:
`python -X utf8 -m pytest -q keel/tests`
— **46 passed in 1.75 s**.

## Paid-run log

The cap for this slice is $45. Costs use $3.95/H100-hour. Estimates include
local wall time (including CPU-only image builds and startup), so they
conservatively overstate actual GPU billing; they are not invoice amounts.

| Date | Function | H100-minutes upper estimate | USD upper estimate | Outcome |
|---|---|---:|---:|---|
| 2026-10-15 | `train_remote(smoke=True)`, torch 2.10.0 | 7.284 | $0.480 | FLA correctly refused a known-invalid Hopper backward kernel in Triton <3.7.1. No checkpoint. |
| 2026-10-15 | 9B pilot evidence authorship + two-sample validation | 6.748 | $0.444 | 192 held-out sources; 82 validated evidence counterfactuals. |
| 2026-10-15 | CUDA 13 training-image build | 3.444 | $0.227 | CPU-only build failed because default `clang++` was absent. Fixed by explicitly selecting installed GCC/G++. Actual GPU allocation: zero. |
| 2026-10-15 | Corrected `train_remote(smoke=True)`, torch 2.14.1 | 12.712 | $0.837 | Eight rows, two real forward/backward microbatches and one optimizer update; finite gradient norm 19.03; 10.35 GB peak GPU allocation. Includes a 619.6-second CPU wheel build. |
| 2026-10-15 | Stock 4B one-pass pilot | 2.575 | $0.170 | Scored 192 parent groups; final evidence metrics reuse only the screened subset. |
| 2026-10-15 | Fresh 9B-thinking pilot | 5.604 | $0.369 | Two samples; independent seed from teacher validation. |
| 2026-10-15 | 4B-thinking pilot evidence crosscheck | 3.243 | $0.214 | 82 teacher-accepted → 14 executable-event candidates → 14 cross-model validated. |
| 2026-10-15 | 9B training evidence authorship + two-sample labels | 66.015 | $4.346 | 3,107 sources; 14,885 scored units; 9,075 labels before the stricter evidence screen. |
| 2026-10-15 | 4B-thinking training evidence crosscheck | 3.260 | $0.215 | 1,207 teacher-accepted → 248 executable-event candidates → 237 cross-model validated; 8,105 total training labels. |
| 2026-10-15 | Full `train_remote`, one epoch | 18.348 | $1.208 | 8,105 rows, 507 optimizer updates, full merged HF checkpoint saved and volume committed. |
| 2026-10-15 | Failed student checkpoint load/pilot | 176.247 | $11.603 | No returned outputs. Missing processor metadata caused repeated initialization failures; client later failed with DNS/ConnectionError. |
| 2026-10-15 | Failed app shutdown reserve | 1.953 | $0.129 | Difference between observed app lifetime and the failed local-call wall time; avoids silently ignoring disconnect cleanup. |
| 2026-10-15 | Cancelled pre-repair bounded retry | 1.500 | $0.099 | Explicitly stopped after diagnosing missing assets; no returned outputs. |
| 2026-10-15 | `repair_processor_remote` | 0 (CPU-only) | $0.025 | 22.92 seconds; ledger conservatively charges CPU wall at the H100 rate. Pinned processor assets copied and locally loaded; trained weights untouched. |
| 2026-10-15 | Repaired checkpoint `judge(...)` smoke | 2.400 | $0.158 | Four TRAIN-source rows, valid distributions, vLLM 0.30.0; bounded to 900 seconds. |
| 2026-10-15 | Repaired student one-pass pilot | 3.069 | $0.202 | 782 held-out variants plus the four-row train-source load check; both baseline arms reused cached outputs. |

**Total conservative upper estimate: $20.72 of the $45 cap**, including the
failed-run exposure and shutdown reserve. The machine-readable consolidated
ledger is `direction/experiments/distill_v0/costs.json`.

The initial numerical check exposed FLA's explicit correctness guard for
Hopper's `chunk_bwd_dqkwg` kernel with Triton 3.4–3.7.0. The fix upgrades the
training image to a compatible torch/Triton/CUDA stack; it does **not** disable
the check or accept an invalid backward pass. Aggregate detail is in
`direction/experiments/distill_v0/smoke-failed.json`.

The corrected Qwen3.5-4B numerical check succeeded with 32,464,896 trainable
LoRA parameters. Mean restricted loss was 0.76885; both backward passes and
the optimizer update completed. Function execution was 121.52 seconds
(the larger ledger time includes image building). This checks differentiability
and finite gradients, not convergence or pilot quality.

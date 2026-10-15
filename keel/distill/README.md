# Keel train-source discernment distillation

This package distils Qwen3.5-9B thinking under Keel's trust policy into a
one-pass Qwen3.5-4B LoRA student. It implements source rebuilding, contamination
gates, training-only variants, soft teacher labels, restricted label-token
training, full-checkpoint merging, and a group-disjoint pilot. This is an
exploratory synthetic-source experiment, not a Decision Alignment Card result.

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
`/home/alp/ssdm-direction/distill_v0/`; model artifacts live on `keel-results`.
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
  more than 500 characters. The addition stays untrusted. Both validation
  samples must select the changed option. This is same-model sampled
  validation, **not independent-human validation**; semantic errors remain
  possible despite the lexical and label checks.
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
checkpoint. The training image owns CUDA 12.8.1, torch 2.10.0, transformers
5.14.1, peft 0.21.2, accelerate 1.12.0, flash-linear-attention 0.5.2 and
causal-conv1d 1.7.0. No extra workstation package installation is required.

`--smoke-sources` performs a real forward/backward/update on eight train-source
rows with source-gold one-hot targets; it is a numerical/architecture check,
not a distilled checkpoint or reported trained model. Full training uses only
validated two-sample teacher labels, saves a LoRA adapter, merges it, and saves
a full HF checkpoint at `/results/students/<run>/merged`.

## Pilot definitions

The pilot compares student 4B one-pass trust, stock 4B one-pass trust, and
teacher 9B thinking trust on the same held-out source instances and variants.
All clean and policy units are evaluated, not filtered for teacher success.
Evidence is restricted to validated changed-gold additions; teacher pilot
scoring uses a fresh seed, rather than reusing validation samples.

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

## Reproduction

Run from the repository root with `/home/alp/ssdm-venv/bin/python -X utf8`:

```sh
python -X utf8 -m keel.distill.sources --out /home/alp/ssdm-direction/distill_v0/data
python -X utf8 -m keel.distill.teacher \
  --sources /home/alp/ssdm-direction/distill_v0/data/train.jsonl.gz \
  --out /home/alp/ssdm-direction/distill_v0/teacher-train
python -X utf8 -m keel.distill.teacher \
  --sources /home/alp/ssdm-direction/distill_v0/data/pilot.jsonl.gz \
  --out /home/alp/ssdm-direction/distill_v0/teacher-pilot
python -X utf8 -m keel.distill.modal_train \
  --smoke-sources /home/alp/ssdm-direction/distill_v0/data/train.jsonl.gz \
  --run distill-v0-20261015-smoke --out /home/alp/ssdm-direction/distill_v0/smoke-train.json
python -X utf8 -m keel.distill.modal_train \
  --labels /home/alp/ssdm-direction/distill_v0/teacher-train/labels.jsonl.gz \
  --run distill-v0-20261015 --out /home/alp/ssdm-direction/distill_v0/train.json
python -X utf8 -m keel.distill.pilot \
  --train-sources /home/alp/ssdm-direction/distill_v0/data/train.jsonl.gz \
  --pilot-sources /home/alp/ssdm-direction/distill_v0/data/pilot.jsonl.gz \
  --teacher-dir /home/alp/ssdm-direction/distill_v0/teacher-pilot \
  --student /results/students/distill-v0-20261015/merged \
  --private-out /home/alp/ssdm-direction/distill_v0/pilot \
  --aggregate-out direction/experiments/distill_v0/pilot.json
python -X utf8 -m pytest -q keel/tests
```

The CPU suite exercises template disjointness and an evaluation-template
canary, wrong targets across all types, trust-channel separation, a planted
JevBench-overlap fixture and disabled-gate rejection, split leakage, lexical
counterfactual filters, exact KL/CE against hand calculation, real tiny-model
gradients and nonlabel exclusion, masked labels, teacher filtering, and
cluster-bootstrap metric denominators. Initial run: **30 passed in 2.09 s**.

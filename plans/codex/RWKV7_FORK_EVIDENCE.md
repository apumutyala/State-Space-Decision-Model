# RWKV7 fork plus evidence residual (unmeasured, 2026-10-07)

**Hypothesis.** `FinalStateFork` scores only the final continuation vector. An
option/query-conditioned additive read of existing evidence features may recover
matching signals for retrieval and tool selection without new backbone passes.
Inspect the first cell's area results before spending: this does not establish
that its remaining gap is a head problem.

**Artifact.** `heads.heads.ForkEvidenceResidual` (`fork_evidence`) retains the
parent's `norm`/`mlp` keys. Its 32-channel branch multiplies projected evidence by
projected option + `q_end`, applies SiLU, an independent sigmoid gate and signed
linear readout, and sums increments scaled by
`sqrt(channels * max(evidence_tokens, 1))`. It evaluates 128-token chunks; no
softmax or normalisation across positions/options is used. Zero readout weights
and restored initialisation RNG preserve same-seed parent logits and typed
probabilities in fp32 and CPU bf16. Existing fork head weights also transfer
without renaming. Attribution accounts for parent score plus evidence increments,
up to final logit dtype rounding; it is not causal because the features mix context.

**Limits.** The cache API currently forks a supplied full prefix; this variant
adds no recurrent snapshots or continuations. It reads the materialised prefix,
and training retains token/option interactions: K accumulator scalars do not
make it streaming or bounded-memory. Extra evidence access can amplify injection
or distractors and cannot recover missing features. Joint field constraints are
unaddressed. `train.estimate` does not price this head; qualify its actual
runtime/memory rather than reusing the parent's projection.

**Exact follow-up.** After the live job finishes, compare to its preserved fork
checkpoint at **step 800**, provided the effective config/data match the base
`jobs/m2_first_cell/m2_first_cell_rwkv7.json`. Candidate recipe:
`configs/rwkv7_fork_evidence_800.json` (seed 0, same parent, training draw/guards,
ordered template, LoRA r64, Clef loss and execution; changed head and 800-step
endpoint). Start from the original parent in fresh output/checkpoint directories.
Never bypass resume identity checks. If the matched checkpoint is unavailable,
rerun the control or defer; do not substitute a differently trained final model.
Qualify cache/recompute smoke, the same T ladder and measured step/peak-memory
probe first. The 190-minute training stop remains a cap, not a matched-time
endpoint. A reviewed merged job and acknowledged spend issue precede hardware work.

On the future approved worker:

```bash
python -m train.run_cell --config configs/rwkv7_fork_evidence_800.json \
  --model-dir "$RWKV7_DIR" --records "$COHORT_RECORDS" --train-dir "$TRAIN_DIR" \
  --guard-readiness "$GUARD_READINESS" --out "$NEW_OUT" --ckpt-dir "$NEW_CKPT" \
  --device cuda --stage all
```

Evaluate both exact endpoints through the existing first-cell protocol, with
calibration only on its designated split, G12/G13 and all full8 rule-v1 floors.
Report paired primary/retrieval/tools uncertainty and timing/memory. Retain only
if the primary paired 95% CI exceeds zero against the matched fork and every
floor passes; otherwise fold. One seed does not establish a Clef win: that still
requires matched evaluation and paired CIs above zero against each Clef model.
Any causal interpretability claim requires perturb-and-recompute plus random
and output-only controls.

**CPU proof.** Focused tests cover parent/typed identity, checkpoint transfer,
residual learning and feature gradients, masking, empty evidence, equivariance,
chunk equivalence and accounting. Existing registered head/cache tests pass.
No hardware run or task-quality measurement was performed.

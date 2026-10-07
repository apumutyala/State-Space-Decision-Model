# L0 phase: one matched supervised head-port ablation

Prepared only; no real parent/checkpoint access or fit is authorized.
Base: `0200ee29d0d7a7987a9128b629f32fa298d30b41`.
Branch: `stack/rwkv7/head-qend-axes`.

**Claude cross-review is required before execution/integration.** The additive
D-owned changes are `heads/heads.py`, `train/data.py`, `train/trainer.py` and
`train/lineage.py`. There is no change to requests, HeadInput, served answers
or the semantic-program interface. Default legacy config/item identities stay
unchanged when auxiliary supervision is absent.

The folded SnapshotFork/CE recipe remains a negative reference. This new
experiment asks whether explicitly supervising a small head port improves
checked free-running usefulness relative to the identical port architecture
trained with choice CE alone. It does not test an emergent representation,
backbone-causal semantics, general interpretability or L0 qualification.

Both arms use the same frozen cached RWKV7-0.1B parent, full initial head state,
seed 20261006, fixed corpus, request order, optimizer and training settings.
The head is
`z=tanh(W LayerNorm(q_end))`,
`logit_k=z dot B(e_k)+b(e_k)`.
`e_k` is the existing candidate text embedding; B and b are shared learned
linear maps. The head reads no option strings, IDs, request metadata or labels.
Permutation equivariance is conditional on equivariant input embeddings;
this does not promise backbone invariance to reordered/relabelled prompts.

Two arms only:

| Arm | Choice loss | Axis loss | Updates |
|---|---|---|---|
| control | existing unsmoothed CE | lambda=0 | exactly 32 |
| axes | identical CE | lambda=1 times mean squared error | exactly 32 |

Batch size 2 gives one complete pass over the same 64 train slots. AdamW,
learning rate 0.001, no weight decay, no warmup, gradient clipping 1, FP32,
no LoRA and all remaining defaults match. No tuning, intermediate selection,
post-evaluation optimizer or checkpoint selection is allowed. Save and evaluate
only the fixed final checkpoint. The same initialization/data/order hashes must
match before a paired report is accepted.
Freeze both arm registrations before either execution. Run control then axes
without inspecting control metrics in between; only terminal/resource status
may be checked. No recipe/source change is allowed between arms. Inspect paired
metrics only after both fixed runs finish, or record an incomplete comparison.

The exact original corpus contains 24 synthetic training worlds and 12 reused
development worlds. Canonical hash:
`cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`.
No Index, TEST or untouched held-out items are read. Source, recipe, corpus,
split, target-sidecar and previous-result hashes are frozen in each arm's
registration before any real run.

Slot-specific targets are derived offline only from `training_only.program`
and the exact supervised slot binding. Axis order is
`[assert, explain, compare, record, field, relation]`:

- operation: one-hot task operation; all other coordinates zero;
- fact: the assertion's ReadFact record and field;
- left/right: the comparison's respective bound ReadFact, not a shared first
  operand target;
- relation: less/equal/greater is −1/0/+1, all other coordinates zero.

Record 1/2 maps to −1/+1; price/warranty_months maps to −1/+1. Inactive axes
are zero, including an equality relation's all-zero target. Targets live only
on training Item sidecars and are hashed into training lineage; they never
enter requests, LayoutRow metadata, HeadInput or the inference head.

Primary: lambda1 minus lambda0 paired complete useful-answer accuracy and
kernel acceptance on the same 12 development worlds, with world-bootstrap
intervals (seed 9203, 2000 draws). Also report comparisons with the step24 and
partial step96 free-running references. A valid wrong-question answer remains
a usefulness failure; every released answer follows the existing checked
program and validating realizer path. There is no unchecked text path.

Fit requires at least 80% training-slot accuracy (52/64). Lambda1 cannot
advance for review unless both arm receipts are complete, its useful-answer
accuracy strictly exceeds lambda0, and its read-task accuracy does not regress.
Report the paired uncertainty even when these screening conditions pass;
this is never automatic adoption, promotion or an L0 capability gate. Axis
MSE and interventions alone never count as useful-answer capability.

Secondary: one final teacher-forced pass over all 64 train/32 dev requests,
per-slot CE, exact selection accuracy, axis MSE and every failure. A fixed
six-axis by {−1,0,+1} clamp sweep uses the final dev-row head port. Include a
seeded random port perturbation matched to each clamp's norm and an output-only
affine replay with the axis state unchanged. The affine replay necessarily
matches clamp logits in this readout: report that equivalence explicitly.
It cannot establish causal superiority over output-only manipulation, and
clamp/axis-MSE results are not evidence of backbone-causal semantics. No control
uses option-description parsing or candidate-ID semantic decoding.

Run arms separately on the existing CPU, offline, CUDA hidden and two threads,
with fresh outputs, 12 GiB start and 3 GiB running RAM floors. Each arm has a
1200-second hard watchdog and owned-child cleanup; no automatic retry. Previous
CPU timings put 32 updates, one 96-row fit pass, checked free-running outputs
and a reused-forward port sweep below that ceiling, but any stop remains
partial and cannot support a clean success claim. Raw events/checkpoints are
preserved. Paired reporting is JSON-only and requires both complete arm receipts.

Preparation commands load no parent/checkpoint:

```powershell
python -m language_growth.rwkv_qend_axes --arm control --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-control-prepared-20261006
python -m language_growth.rwkv_qend_axes --arm axes --dry-run --out C:/Users/apumu/research/CHLabs/ssdm-data/qend-axes-axes-prepared-20261006
```

Only after independent review and explicit execution approval, replace
`--dry-run` with `--execute` and use separate fresh `*-run1-20261006` outputs.
No paid compute, GPU, PR or real model execution is part of preparation.

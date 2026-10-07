# Conditional pointer diagnosis from saved CVAE posteriors

**Finding:** operation selection is usually strong in this saved-posterior diagnostic, while exact record/field binding conditioned on the gold operation is near 50%. For the pooled β=0 arm, the operation-marginal argmax is correct on 95.46% of endpoints [93.98, 96.85], while top-1 pointer selection after restricting candidates to the gold operation is 49.81% [44.81, 54.91]. In the β=.05 arm, those values are 92.78% [91.11, 94.35] and 48.70% [44.44, 53.15]. The paired β=.05-minus-β=0 differences are -2.69 pp [-4.07, -1.30] for operation and -1.11 pp [-5.46, 3.15] for oracle-conditioned pointer. This supports task-to-record binding as the next bottleneck to test; it does not establish a served hierarchical planner.

The endpoint posterior has 12 bindings: 3 allowed operations × 4 field/record bindings. This analysis compares (a) the saved joint MAP, (b) operation argmax after summing candidate mass by operation, (c) pointer argmax after conditioning on the gold operation (oracle diagnostic, not a deployable score), and (d) hierarchical selection using operation-marginal argmax followed by its top binding. It uses only probabilities already saved by the prior CPU run; no checkpoint is loaded and no model inference or training occurs.

| Family | Arm | Operation marginal | Gold-operation-conditioned pointer | Hierarchical exact |
|---|---|---:|---:|---:|
| New/disjoint wording | β=0 | 91.11% [88.33, 93.70] | 50.00% [44.44, 55.19] | 45.00% [39.81, 50.19] |
| New/disjoint wording | β=.05 | 85.56% [82.22, 88.70] | 49.81% [45.37, 54.44] | 40.93% [36.67, 45.19] |
| Old/failing wording | β=0 | 99.81% [99.44, 100.00] | 49.63% [44.26, 55.37] | 49.44% [43.70, 55.37] |
| Old/failing wording | β=.05 | 100.00% [100.00, 100.00] | 47.59% [42.78, 52.59] | 47.59% [42.59, 52.41] |

Each absolute interval is a 2,000-draw percentile bootstrap over `world_id`, averaging the three model-seed observations within each world. The same 180 underlying worlds appear in both wording families, so pooled estimates keep those families together in **180** clusters (360 family/world presentations), not 360 independent units. Paired differences resample those same clusters across β arms. Bootstrap seed is 9203 with deterministic per-metric offsets recorded in `conditional_result.json`. Inputs are the six frozen endpoint files and the original source receipt; their exact hashes are in `artifact-manifest.json`.

Conditional pointer exactness by gold operation, pooled across families, was β=0 → β=.05: read 49.44% [39.44, 59.44] → 53.33% [46.11, 60.56]; explain 55.00% [47.78, 62.22] → 48.06% [38.89, 57.22]; compare 45.00% [36.67, 53.89] → 44.72% [38.89, 50.56]. The intervals are wide, but operation selection is not the dominant failure on the new/disjoint read and compare strata. β=.05 does not improve pointer selection and reduces operation probability mass on new/disjoint wording.

Under this oracle gold-operation slice, pointer failures divide into **field error** versus **correct field but wrong record/order**: β=0 had 32 and 510 respectively (542 total failures out of 1,080); β=.05 had 27 and 527 (554/1,080). This supports a record/operand binding bottleneck, not a field-selection bottleneck, for this synthetic decoder. These counts are still an oracle-conditioned diagnostic, not end-to-end performance.

An independent raw-only reimplementation verified all six endpoint SHA-256 hashes and reproduced the central pooled estimates exactly: β=.05−β=0 was −2.685 pp operation, −1.111 pp gold-operation-conditioned pointer, and −2.963 pp hierarchical exact. Its Python `random.Random` bootstrap endpoints were within about 0.1 pp of this script's NumPy bootstrap endpoints, as expected from a different random-number stream; both use the same 180 clustered worlds and the CIs support the same conclusions.

## Scope and limits

This is an offline decomposition of an existing supervised four-slot variational bottleneck run (`source_receipt.json.gz`, original source `bd0c591`, freeze SHA `d1ad9f39…`). It is not a new training result, capability win, Track D parent result, L0 qualification, or evidence that a VAE should be expanded. The gold-operation slice is an oracle diagnostic. The fixed RWKV `q_end` head and this separate CVAE are not the same model; the result only identifies a concrete target-binding hypothesis for the next parent-backed screen.

Cheapest next model experiment: compare the frozen RWKV parent’s existing generic `Fact record_ref field` candidates with candidates that include the permitted entity name alongside the same record reference and field. Keep all candidates and bindings intact; never substitute a resolver’s choice. Use paired read/explain cases first, report task correctness and exact fact binding, then enlarge the dev cohort only if the small screen moves in the predicted direction. This tests the name-to-record join that the current candidate text forces, without another latent-loss sweep.

Exact reproduction from this directory:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 analyze.py
```

Local CPU only; saved-probability parsing and bootstrap; $0; no GPU, Index, held-out data, checkpoint load, model call, or training.

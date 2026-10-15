# nuPI constrained training: training as feedback control

Code: `train/pi_control.py` (the controller), `train/constraints.py` (constraints, penalties, measurements,
checkpoint state), hooks in `train/trainer.py`, the cell key in `train/run_cell.py`, the MQAR probe generator
`train/mqar_probe.py`, the toy `train/nupi_toy.py`. Tests: `tests/test_nupi.py`. Research context:
`ssdm-wt/b-landscape/CONTROL_RESEARCH.md` #4.

## What it does

Each target becomes an explicit constraint `g_k(theta) <= eps_k` with a Lagrange multiplier `lambda_k >= 0`. The
trained objective is

    L = L_task + sum_k lambda_k * (g_k(theta) - eps_k)

with `lambda` held fixed (detached) inside an optimiser step. After the step a PI controller sets the next `lambda`
from the measured constraint error `e_t = g_t - eps` (Sohrabi et al., ICML 2024, "On PI controllers for updating
Lagrange multipliers in constrained optimization"):

    xi_t     = nu * xi_{t-1} + (1 - nu) * e_t                      EMA of the error (nu = 0: xi_t = e_t)
    I_t      = clip(I_{t-1} + e_t, integral_min, integral_max)      integral, with an anti-windup clamp
    lambda_t = clip(lambda_0 + kappa_p * xi_t + kappa_i * I_t, 0, lambda_max)

- `kappa_p = 0, integral_min = 0` is exactly projected gradient ascent on the multiplier,
  `lambda <- max(0, lambda + kappa_i * e)` (the usual GDA Lagrangian; tested bit-exactly).
- The P term damps the dual dynamics: when the constraint becomes satisfied, `lambda` drops at once rather than
  waiting for the integral to unwind. This removes the overshoot / limit-cycle behaviour of GDA multipliers.
- The anti-windup clamp bounds how much "debt" a long infeasible stretch can store. `integral_min = 0` (the default)
  also stops a long feasible stretch from storing negative credit.
- It generalises the bang-bang anti-collapse guard (`Trainer.collapse`: a threshold fires, then LR x 0.5 and a
  reload). The guard stays as it is; nuPI acts continuously on the loss instead of discretely on the run.

## Constraints

| kind | g_k | measured on | notes |
|---|---|---|---|
| `calibration` | `metric: brier` (multi-class Brier, mean over rows) or `metric: ece` (top-1, `bins` equal-width bins) | `source: batch` (the step's rows, smoothed by `nu`) or `source: monitor` | Brier is smooth and the recommended loss term. Batch ECE over 8 rows is noisy, so measure ECE on the monitor sample. |
| `consistency` | mean aligned JS between a row and its permuted / relabelled view (`train.losses.aligned_order_js`) | the step's paired views | OPTIONAL. Needs `Item.view` / `view_perm` on train items. It replaces the fixed-weight `order_consistency_weight` term (both on is refused). On the symmetric layout the flip rate is 0 by construction, so leave it off there. |
| `family_regression` | per family f: mean task (arm) loss of f's rows; bound `(1 + delta) * ref_f` | batch or monitor | One multiplier per family. Family = `item_source(id)` (the `m2:<source>:` id field, i.e. the JevBench family of the jev1 draw). `reference: "init"` measures `ref_f` on the monitor sample before step 1 (the LoRA zero-init model is the reference); a `{family: loss}` dict pins it (e.g. from the stock-shim arm). Families without a reference are unconstrained. |
| `recall` | decision CE on an MQAR-style probe batch | its own probe batch (`probe_batch` rows/step, a deterministic cycle) | For attention -> SSM conversion cells, where KL stayed low but recall was lost. `setpoint` (absolute probe loss) or `delta` (`(1 + delta) x` the step-0 probe loss). Probe rows: `python -m train.mqar_probe --out probe.jsonl` and `probe_path` in the item. |

The monitor sample is `monitor_rows` TRAIN-split rows (`dev_card_sample` over the train items: hash-stratified by
source, seed `monitor_seed`), measured with no grad in eval mode every `monitor_every` steps. It never touches eval,
DEV, test, held-out or JevBench rows (rule 3 in AGENTS.md). `lambda` is held between monitor measurements.

## Config (`constraints:` in the cell config; absent = OFF)

```json
"constraints": {
  "kappa_p": 1.0, "kappa_i": 0.05, "nu": 0.9,
  "lambda_max": 10.0, "integral_min": 0.0, "integral_max": 200.0,
  "source": "batch", "monitor_rows": 64, "monitor_every": 100, "monitor_seed": 0,
  "items": [
    {"name": "brier", "kind": "calibration", "metric": "brier", "setpoint": 0.30},
    {"name": "ece", "kind": "calibration", "metric": "ece", "bins": 10, "setpoint": 0.05,
     "source": "monitor", "kappa_p": 2.0, "kappa_i": 0.5},
    {"name": "fam", "kind": "family_regression", "delta": 0.05, "reference": "init", "source": "monitor"}
  ]
}
```

Any controller key (`kappa_p kappa_i nu lambda_init lambda_max integral_min integral_max`) can be overridden per item.
Unknown keys, a missing setpoint, an unknown kind or metric, and a consistency item without views are refused.

Setpoint guidance: start each `eps` at the value the OFF arm reaches at the end of training on the same cell
(read from its `history` / dev card), minus the improvement you want. A setpoint the model cannot reach makes
`lambda` climb to `lambda_max` and the task loss pay for it: that is visible in the log, not hidden.

## Logging and checkpoints

- Every step's `history` entry carries `loss` (the TASK loss, also what the collapse guard sees), `lagrangian`, and
  `constraints: {key: {"lam", "g", "e"}}` (`g`/`e` only on steps with a measurement). OFF: the entry is exactly
  `{"step", "loss"}` as before.
- The checkpoint stores `constraints: {ctl: {key: {lam, integral, xi, e, n}}, bounds, initialised, probe_cursor}` as
  Python doubles, so a resume or a `train.chain` link continues the controller bit-exactly (tested: the 6 + 6 step
  resume equals the 12-step run in every loss, lambda, g and weight). A constraints run refuses to resume from a
  checkpoint without controller state. A collapse-guard reload restores weights, not `lambda`.
- The config block is in the `TrainConfig` hash when present; absent (or `{}`), `to_json` / the hash / the
  checkpoint are byte-identical to before (`TrainConfig().hash() == "46a4f6f3acbac492"`).
- The separate surface-consistency block can also be enabled. Its symmetric-KL term remains part of the logged
  `loss`; the `constraints` penalties (including recall) appear in `lagrangian`. Both controllers update once per step
  and checkpoint independently. See [merge notes](NUPI_CONSISTENCY_MERGE.md) for the combined CPU regression evidence.

## What is guaranteed, and what is not

- **Guaranteed by construction:** OFF mode is the old code path (bit-identical losses, tested against the base
  branch); `lambda_k in [0, lambda_max]`; the integral stays in `[integral_min, integral_max]`; non-finite
  measurements never move a multiplier; the controller state round-trips exactly.
- **Asymptotic feasibility under standard assumptions:** for the Lagrangian min-max with a convex problem (or local
  convexity near a regular KKT point), small enough step sizes and exact or unbiased measurements, the PI dual dynamics
  converge to a KKT point, so `g_k -> <= eps_k` in the limit (the PI/PID Lagrangian analyses of Stooke et al. 2020 and
  Sohrabi et al. 2024; the paper's theory is for the linearised / quadratic case, the rest is empirical). Deep
  networks are not convex, so for us this is an engineering expectation backed by measurement.
- **NOT guaranteed:** a hard per-step bound. Any step can violate any constraint, and so can the final model on data
  it did not see. ECE on JevBench is measured, never promised. There is no test-time guarantee; use the conformal
  and injection guarantees (`jevkit guarantees`) for statements about deployed behaviour.

## CPU evidence (tests/test_nupi.py; `python -m train.nupi_toy --out docs/nupi_toy_traces.json --svg docs/nupi_toy_traces.svg`)

Toy: `min 0.5 ||x - (2, 2)||^2 s.t. x1 + x2 <= 1`; optimum `x* = (0.5, 0.5)`, `lambda* = 1.5`; primal GD lr 0.05,
600 steps, simultaneous updates (the trainer's order). Traces: `docs/nupi_toy_traces.svg`.

| setting | controller | lambda overshoot | lambda total variation | boundary crossings | settle step (abs e <= 0.01) | final abs e |
|---|---|---|---|---|---|---|
| kappa_i 0.05 | GA (kappa_p 0) | 0.515 | 3.07 | 3 | 226 | 1.6e-6 |
| kappa_i 0.05 | nuPI kappa_p 1 | 0.000 | 1.50 | 1 | 121 | 1.7e-12 |
| kappa_i 0.2 | GA | 1.070 | 8.97 | 9 | 308 | 8.0e-5 |
| kappa_i 0.2 | nuPI kappa_p 4 | 0.197 | 1.97 | 1 | 81 | 1.2e-13 |
| noisy g (sigma 0.3), kappa_i 0.05 | GA | 0.597 | 9.25 | 8 | never | 0.146 |
| noisy g (sigma 0.3), kappa_i 0.05 | nuPI kappa_p 0.5, nu 0.9 | 0.359 | 15.73 | 9 | never | 0.114 |

Deterministic measurements: nuPI is feasible sooner with a fraction of the multiplier travel and no limit cycle.
**Negative result, noisy measurements:** with measurement noise comparable to the error signal, the P term passes noise
into `lambda` (more total variation); over 10 seeds x 1,000 steps the second-half mean |e| is equal (0.057 vs 0.057
at kappa_p 0.5, nu 0.9) and worse for larger kappa_p. So on noisy batch metrics keep `kappa_p` small and `nu` high
(>= 0.9), or measure on the monitor sample (lower variance). This is why ECE defaults to the monitor in the example.

## Paired ablation (the on/off design)

- **Cell:** the jev1 4B cell (`jobs/m2_first_cell/m2_first_cell_rwkv7_jev1_q35_4b.json`, A-PREREG-JEV1-4B: stock
  Qwen3.5-4B + H2O shim, LoRA r64, 1,121 steps) as OFF, and the same file plus the `constraints` block above as ON.
  Same seed, data, steps, LR schedule and readout; OFF is bit-identical to the existing cell, so an already-landed
  jev1_q35_4b run on the same recipe serves as the OFF arm.
- **Gate (to write into PREREG.md before the result):** ON passes iff (1) jevkit ECE (C-axis calibration on the 231
  public items) <= the ECE setpoint, and (2) the paired Composite difference ON - OFF has a 95% bootstrap CI whose lower
  bound is >= -m (not worse; m = the board's Composite non-inferiority margin, same items, same bootstrap draws
  B 1,000 seed 16 as `screen.paired_vs`). Report every family's loss vs its bound and the final lambdas.
- **Overhead:** brier on the batch is free; the family references and ECE need one 64-row no-grad monitor pass at
  step 0 and every 100 steps (12 passes x ~30 s at the measured 10.92 s per 8-row step ~ 6 min, ~2% of the clock).
- **Cost** (`jobs/common/rates.py`, BATCH_USD_PER_MIN 0.063): the jev1 job's 397-min max duration -> quoted hold
  397 x 0.063 = $25.01 per arm, worst case (60 + 2 x 397) x 0.063 = $53.80 per arm. ON arm alone (OFF reused):
  ~$25 quoted; both arms fresh: ~$50 quoted, $108 worst case.

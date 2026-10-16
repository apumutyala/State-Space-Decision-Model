# Developmental controllability: shortcut-onset pilot

**Verdict: no controller advantage on this toy.** νPI reduced shortcut reliance versus no control, but every
fixed penalty did better on held-out reliance; weight 8 halved decision flips relative to νPI without an observed
capability cost. The canary warned **10.0 ± 2.2 updates** before the hard threshold, but falsely alarmed in **2/8**
no-shift runs. The useful finding is the gap between controlling a fixed monitor and controlling unseen behaviour.

## Thesis and decision rule

A behavioural canary can expose a new shortcut during training, and feedback on a differentiable shortcut metric
might preserve the intended rule more reliably than a fixed penalty. This tests **behavioural constrained learning**,
not identification or control of a neural mechanism. The stronger mechanistic proposition needs additional evidence.

Compare the existing repository νPI controller with no control and fixed weights 0.5, 2 and 8. A controller win
requires a better held-out accuracy–misbehaviour frontier than the fixed-weight sweep, not just improvement over
no control. Report paired seed uncertainty, all sweep arms, transient violations and detector false alarms. No
winner will be selected or tuned on the held-out test examples. Seed 0 is a feasibility pilot; seeds 100–107 are
the frozen main replication set. Any change after the pilot will be disclosed.

## Prior work (links opened 2026-10-16)

- [Hoogland et al., *Loss Landscape Degeneracy and Stagewise Development in Transformers* (2024)](https://arxiv.org/abs/2402.02364)
  connect singular learning theory and the local learning coefficient (LLC) to developmental stages. The
  [Timaeus research page](https://timaeus.co/research/2024-02-04-developmental-landscape) identifies the organisation
  as now Resolution. These are empirical stage associations, not a theorem that an LLC alarm anticipates harm.
- [Power et al., *Grokking* (2022)](https://arxiv.org/abs/2201.02177) show delayed generalisation on small algorithmic
  datasets. Sharp changes motivate longitudinal measurements; this experiment does not claim to reproduce grokking.
- [Geirhos et al., *Shortcut Learning in Deep Neural Networks* (2020)](https://arxiv.org/abs/2004.07780) define
  shortcuts as rules that work on standard data but fail to transfer. Here a planted feature gives a known,
  counterfactually manipulable shortcut, avoiding an ambiguous proxy for general alignment.
- [Stooke, Achiam & Abbeel, *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods* (2020)](https://proceedings.mlr.press/v119/stooke20a.html)
  interpret multiplier updates as feedback, adding damping/predictive terms to address oscillation and overshoot.
- [Sohrabi et al., *On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization* (2024)](https://arxiv.org/abs/2406.04558)
  introduce νPI. We reuse [`train/pi_control.py`](../../../train/pi_control.py), with the post-step measurement and
  detached-multiplier convention in [`docs/NUPI.md`](../../../docs/NUPI.md); no new controller implementation.
- [Betley et al., *Emergent Misalignment* (2025, original version)](https://arxiv.org/abs/2502.17424v1) report broad
  misalignment following narrow insecure-code finetuning, including trigger-conditional cases. This motivates the
  long-term question; a synthetic shortcut is not evidence about that cross-domain phenomenon.

## Frozen method

An MLP (9→32→32→1, ReLU, 1,409 parameters) predicts XOR of the signs of two Gaussian inputs; six dimensions are
irrelevant. The ninth is a shortcut bit. Adam (lr 0.003, batch 128) first trains for 400 steps with a random bit,
then for 800 steps with a bit matching the label 98% of the time. A sixth, diagnostic **no-shift** arm retains a
random bit. The shift is imposed by the experimenter, not claimed to be a spontaneous phase transition.

Each seed has 2,048 labelled training examples, 256 unlabelled constraint examples and 512 separate training-only
canaries. Each example is paired with bit +1 and −1. The differentiable metric is
`g = mean((p(y=1 | x,+1) - p(y=1 | x,-1))²)`; its target is 0.01. All controlled arms use the same constraint
examples and loss `BCE + lambda*(g−0.01)`. After every primal update, νPI reads the full constraint sample and
updates the next multiplier: Kp=10, Ki=0.1, ν=0.9, lambda_init=0, lambda_max=20, integral in [0,200]. No penalties
are active during warmup. Fixed penalties start at shortcut introduction; feedback need not wait for a binary alarm.

Canaries are read at shift time and every five steps. **Warning:** g>0.005 at two consecutive observations.
**Harm onset:** counterfactual argmax disagreement >10% at two consecutive observations. Alarm time is the second
observation, never backdated. Lead time is harm minus warning; no-shift runs estimate false alarms. This is a
behavioural soft-to-hard warning, not a hidden-state or LLC detector. **LLC was not estimated; `devinterp`
installation/CPU feasibility was not tested.** An uncalibrated, short SGLD chain would not establish early warning.

Final evaluation generates 4,096 independent examples per seed only after training: intended accuracy with the bit
removed (zero), g and argmax disagreement under bit flips, plus error with the bit opposed to the true label.
Seeds share initialisation, labelled data, minibatches and coin draws across arms. Student-t 95% intervals use the
seed as the unit (n=8); paired differences use within-seed contrasts. Intervals are exploratory and unadjusted.
Time-series summaries use every post-shift update: violation fraction, mean positive excess, maximum excess
(overshoot), last-quarter standard deviation/range, boundary crossings and multiplier total variation/saturation.
Canary violation summaries instead use the five-step observations and never feed back into learning.

## Pilot record

Seed 0 completed all six arms. The model learned XOR (97.1% canary accuracy before the shift); the no-control
shortcut produced a 10-step soft-to-hard warning. The controller's final held-out g was 0.0395 despite a 0.01
training-sample target; fixed weight 8 reached 0.0221. This motivates reporting monitor/canary/test gaps rather
than interpreting training feasibility as held-out safety. **No task, optimiser, controller, penalty, threshold
or seed setting was changed after the pilot.** Four aggregate gap metrics were added before the main run, and
an unnecessary tensor copy was removed. The pilot is excluded from all main uncertainty estimates.

## Experiment actually run

Design committed as `7fd91c5a`; main-run instrumentation frozen at `1f3cfed5`. The main run completed **48 fits:
8 seeds × 6 arms**, after the separate six-fit pilot. Both ran CPU-only inside `bwrap --unshare-net`, at nice 10,
with one torch thread and no external data or model downloads. Main experiment time was **27.21 s**
(30.06 s including process startup); pilot process time 6.24 s. Main peak RSS was **373.4 MiB**. **Paid cost: $0.**
Python 3.12.13, torch 2.14.1+cpu, NumPy 2.5.4, SciPy 1.18.1; no packages installed.

[`aggregate.json`](aggregate.json) contains settings, means, 95% interval half-widths, paired contrasts, versions
and code hashes. Full seed trajectories remain in `~/ssdm/private/research-devcontrol/{pilot,main}/`; no per-item
data or weights are committed. [`tradeoff.svg`](tradeoff.svg) is 13 KB and plots means only; uncertainty is below.

## Results with uncertainty

All `±` values are two-sided **95% Student-t confidence interval half-widths across eight seeds**, not standard
deviations. They describe this synthetic distribution and recipe, not generalisation to other model families.

| Arm | Shortcut-free accuracy (%) ↑ | Held-out g ↓ (target 0.01) | Counterfactual decision flips (%) ↓ | Opposed-bit error (%) ↓ |
|---|---:|---:|---:|---:|
| No control | 97.34 ± 0.47 | 0.0901 ± 0.0095 | 12.91 ± 0.91 | 13.06 ± 0.83 |
| Fixed 0.5 | 97.48 ± 0.42 | 0.0303 ± 0.0061 | 5.58 ± 0.71 | 6.35 ± 0.65 |
| Fixed 2 | 97.69 ± 0.28 | 0.0229 ± 0.0061 | 4.42 ± 0.89 | 5.29 ± 0.84 |
| Fixed 8 | 97.80 ± 0.20 | 0.0156 ± 0.0048 | 3.35 ± 0.75 | 4.36 ± 0.73 |
| νPI | 97.52 ± 0.32 | 0.0373 ± 0.0060 | 6.58 ± 0.69 | 7.17 ± 0.64 |
| No shift (diagnostic) | 97.78 ± 0.14 | 0.00077 ± 0.00043 | 0.64 ± 0.23 | 2.23 ± 0.26 |

**Paired contrasts.** νPI minus no control: flips **−6.33 ± 0.83 percentage points**, accuracy
**+0.18 ± 0.32 pp**. νPI minus fixed 8: flips **+3.23 ± 0.62 pp**, accuracy **−0.28 ± 0.29 pp**.
Thus fixed 8 has clearly less reliance; an accuracy difference is not resolved. Fixed 0.5 also beats νPI on flips
by **1.00 ± 0.35 pp**, with no resolved accuracy difference. This is not an equivalence or non-inferiority proof.

### Transients, overshoot and stability

`g−target` units below are squared probability differences. Violation percentages measure time, not individual
examples; no-control's single boundary crossing is failure to return below the target, not superior stability.

| Arm | Monitor updates violating (%) | Canary observations violating (%) | Mean positive monitor excess | Peak monitor excess | Last-200-update monitor SD |
|---|---:|---:|---:|---:|---:|
| No control | 99.22 ± 0.21 | 99.38 ± 0.28 | 0.08386 ± 0.00756 | 0.11196 ± 0.00985 | 0.00926 ± 0.00229 |
| Fixed 0.5 | 17.13 ± 3.75 | 99.22 ± 0.46 | 0.000584 ± 0.000131 | 0.01670 ± 0.00501 | 0.00155 ± 0.00042 |
| Fixed 2 | 0.031 ± 0.074 | 65.23 ± 20.14 | 0.0000001 ± 0.0000003 | 0.000063 ± 0.000148 | 0.000326 ± 0.000070 |
| Fixed 8 | 0 ± 0 | 34.38 ± 18.66 | 0 ± 0 | 0 ± 0 | 0.000064 ± 0.000032 |
| νPI | 58.94 ± 4.86 | 99.38 ± 0.28 | 0.00389 ± 0.00022 | 0.05535 ± 0.00427 | 0.00256 ± 0.00082 |
| No shift | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0.00055 ± 0.00025 |

Near-zero t intervals can cross zero and zero-width observed intervals are not a guarantee. νPI's multiplier
ended at **0.223 ± 0.030**, with total variation **2.04 ± 0.19**, **28.6 ± 3.4** monitor-boundary crossings,
and no saturation. Its final-quarter g range was **0.01197 ± 0.00454**: bounded, but still fluctuating rather
than a hard invariant. Fixed penalties 2/8 spent almost all/all updates below the monitor target.

**Monitor generalisation is the main failure.** νPI ended at monitor g **0.00767 ± 0.00170**, but independent
canary g **0.03715 ± 0.00940** and test g **0.03729 ± 0.00605**. Even fixed 8 had monitor g
**0.000097 ± 0.000018** versus test g **0.01561 ± 0.00484**. All shifted arms' mean test g exceeds the target.
The monitor is intentionally a TRAIN sample whose exact counterfactuals receive repeated gradients; these gaps
are consistent with sample-specific fitting. They do not establish the internal cause without a monitor-refresh
ablation. A controlled training metric must not be reported as a held-out safety guarantee.

### Does the detector warn early?

In no-control runs, the warning fired at **11.9 ± 2.2** post-shift updates and hard harm at **21.9 ± 2.2**:
**10.0 ± 2.2 updates of lead**, positive in 8/8 seeds (exact binomial 95% interval for that fraction: 63.1–100%).
Under νPI, hard harm still appeared transiently in **7/8** runs, with conditional lead **9.3 ± 1.7** updates.
No fixed-penalty run crossed the hard threshold; their warnings cannot be assigned an observed lead time.

However, **2/8 no-shift runs falsely alarmed (25%; exact 95% interval 3.2–65.1%)**, at steps 35 and 745,
without ever reaching hard harm. Thresholds were not recalibrated after seeing this. The canary is useful as
an exploratory soft-to-hard signal, **not a reliable early-warning system**. It already measures behaviour,
and does not predict the distribution shift, prove an internal phase change, or precede all shortcut dependence.
The controller used continuous monitor feedback, not the binary warning; an alarm-triggered intervention was
not tested. There is no LLC-versus-behaviour timing result.

## Interpretation and next step

The **controller-superiority proposition is not supported** at these gains and target. Feedback improves over
doing nothing but loses to elementary fixed penalties; the strong penalty did not reveal the anticipated
capability trade-off. This is one induced shortcut, one MLP and one gain setting—not evidence that every PID/νPI
recipe is inferior. A known nuisance bit and exact counterfactual access are unusually favourable, while a small
fixed monitor is unusually easy to fit. There is no mechanistic state model, controllability-rank analysis,
causal circuit intervention, spontaneous grokking, large-model emergent misalignment, or deployment guarantee.

**Worth pursuing only as a modest measurement/generalisation study, not yet as a mechanistic safety result.**
Next do a frozen CPU replication with fresh constraint samples, held-out shortcut variants and thresholds
calibrated on separate no-shift seeds; compare matched tuning effort for gains and penalties.
**[INFERENCE] Cost: $0 paid, under five CPU-minutes** at the observed toy throughput.

Only after that, consider a 1–3B open-weight LoRA emergent-misalignment organism: first establish the phenomenon
against secure-code and educational-insecure controls, then compare no control, three fixed penalties and νPI
over five seeds, with separate capability and blinded human-rated misalignment tests. Do not execute generated
code outside a no-network sandbox. **[INFERENCE] Planning estimate: 6–10 GPU-hours total, $24–40 at an assumed
$4/GPU-hour, plus 4–8 hours of human evaluation; request a $50 compute cap.** This is not a vendor quote, and
small-model reproduction is uncertain. That step **requires an owner-approved budget**; no resources were
reserved, no paid services were called, and `BUDGET.json` was untouched.

## Reproduction

From this worktree, the exact main command (output directories already created) is:

```sh
nice -n 10 bwrap --unshare-net --die-with-parent --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp \
  --bind /home/alp/ssdm/private/research-devcontrol /home/alp/ssdm/private/research-devcontrol \
  --bind "$PWD/direction/experiments/dev_controllability" "$PWD/direction/experiments/dev_controllability" \
  --chdir "$PWD" -- /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /home/alp/ssdm/.venv/bin/python -I \
  direction/experiments/dev_controllability/run.py \
  --out /home/alp/ssdm/private/research-devcontrol/main \
  --public direction/experiments/dev_controllability
```

The pilot omits `--public`, uses `--out .../pilot --seeds 0`, and needs no writable worktree bind.
Only the experiment was executed; no repository test suite or formatter was run. No reusable production code
changed. For independent verification, regenerate the main aggregates with the command above and run
`python -m pytest -q tests/test_nupi.py` for the reused controller.

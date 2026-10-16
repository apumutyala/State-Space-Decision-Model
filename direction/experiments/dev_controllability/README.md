# Developmental controllability: shortcut-onset pilot

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
behavioural soft-to-hard warning, not a hidden-state or LLC detector. Optional `devinterp` LLC estimation is not
part of the primary run; an uncalibrated, short SGLD chain would not establish early warning.

Final evaluation generates 4,096 independent examples per seed only after training: intended accuracy with the bit
removed (zero), g and argmax disagreement under bit flips, plus error with the bit opposed to the true label.
Seeds share initialisation, labelled data, minibatches and coin draws across arms. Student-t 95% intervals use the
seed as the unit (n=8); paired differences use within-seed contrasts. Intervals are exploratory and unadjusted.
Time-series summaries use every post-shift update: violation fraction, mean positive excess, maximum excess
(overshoot), last-quarter standard deviation/range, boundary crossings and multiplier total variation/saturation.

## Pilot record

Seed 0 completed all six arms. The model learned XOR (97.1% canary accuracy before the shift); the no-control
shortcut produced a 10-step soft-to-hard warning. The controller's final held-out g was 0.0395 despite a 0.01
training-sample target; fixed weight 8 reached 0.0221. This motivates reporting monitor/canary/test gaps rather
than interpreting training feasibility as held-out safety. **No task, optimiser, controller, penalty, threshold
or seed setting was changed after the pilot.** Four aggregate gap metrics were added before the main run, and
an unnecessary tensor copy was removed. The pilot is excluded from all main uncertainty estimates.

## Reproduction

`run.py` is CPU-only (one torch thread); all raw seed traces belong in `$SSDM_PRIVATE/research-devcontrol/`.
Run inside a network-disabled bubblewrap sandbox. The pilot uses `--seeds 0`; the main run defaults to 100–107.
Only the aggregate JSON and small SVG are intended for git. No external datasets, model calls or paid compute.

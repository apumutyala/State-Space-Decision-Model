"""Track D safety evaluation for typed decisions (phase m0-c).

Attacked variants of Decision-Index-format requests (``state`` + typed ``questions``) are scored through a
generic ``decide(request) -> response`` callable (interfaces/DECISION_LANGUAGE_V0.md), and the harness reports
hijack, steer (headline), alarm/false-alarm, clean-accuracy delta and ECE under attack with paired
two-way cluster-bootstrap CIs. See safety_eval/PREREG.md for what gates m1/m2.

Gated numbers go through decide/ (decide_bridge); untrusted text is encoded only by decide/untrusted.py.

Modules: request (formats, host mask), decide_bridge (canonical path), sanitize (ZWSP control arm only), data (synthetic requests),
attacks (hijack / chatml spoof / D1 steer insertion), metrics + stats (unit table, CIs), standin (CPU decider
for tests), hf_decider (letter-logit LM decider), gcg (GCG-lite option-logit attacker), adaptive (stub),
run_smoke (CLI), certify (G4 erase-and-check + randomized-ablation certificates), selective (G5 LTT selective
risk), bounds (exact binomial / Clopper-Pearson), run_guarantees_smoke (G4/G5 stand-in receipt).
"""
__version__ = "0.2.0"

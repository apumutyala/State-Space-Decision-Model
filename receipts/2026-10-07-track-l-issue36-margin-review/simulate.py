"""$0 cross-track sensitivity checks; normal approximations, not model evidence.

Run: python simulate.py > simulation.json
The observed paired subset CI is independently checked against the committed
2026-10-06-D-index03-rescore receipt. All other scales are labelled assumptions.
"""
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

SEED = 20261006
N = 200_000
Z = NormalDist().inv_cdf(0.975)
rand = random.Random(SEED)
noise = [rand.gauss(0, 1) for _ in range(N)]


def trial(delta, se, margin, kind):
    estimates = (delta + se * v for v in noise)
    if kind == "ni_higher_better":
        passed = sum(x - Z * se >= -margin for x in estimates)
    elif kind == "ni_lower_better":
        passed = sum(x + Z * se <= margin for x in estimates)
    elif kind == "no_demonstrated_increase":
        passed = sum(x - Z * se <= 0 for x in estimates)
    else:
        raise ValueError(kind)
    return {"delta": delta, "se": se, "margin": margin, "kind": kind,
            "p_pass": passed / N, "mc_se_max": 0.5 / math.sqrt(N)}


out = {"seed": SEED, "n_trials": N, "billed_usd": 0,
       "method": "Gaussian estimator with known SE and two-sided normal 95% CI; sensitivity only"}
out["capability_subset_assumed_se_0.752"] = [trial(d, .752, 1., "ni_higher_better") for d in [0., -1., -2.]]
out["capability_full_dev_projected_se_0.31"] = [trial(d, .31, 1., "ni_higher_better") for d in [0., -1.]]
out["retention_accuracy_assumed_discordance_0.1_n1420"] = [
    trial(d, math.sqrt(.1 / 1420), m, "ni_higher_better")
    for m in [.01, .03] for d in [0., -m]
]
out["retention_ece_assumed_halfwidth_0.026"] = [
    trial(d, .026 / Z, .01, kind)
    for d in [0., .01, .02, .03]
    for kind in ["ni_lower_better", "no_demonstrated_increase"]
]
out["retention_accuracy_n80pct_at_disc_0.1_margin_0.01"] = math.ceil(
    .1 * (Z + NormalDist().inv_cdf(.8)) ** 2 / .01 ** 2)
# A 95% CI for a zero difference cannot establish an absolute 1% cap
# if both arms have 10% alarms. This is a deterministic counterexample.
out["alarm_delta_counterexample"] = {"parent_false_alarm": .10, "child_false_alarm": .10,
    "delta_ci": [0., 0.], "delta_gate_passes": True, "absolute_1pct_rule_passes": False}
out["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print(json.dumps(out, indent=2))

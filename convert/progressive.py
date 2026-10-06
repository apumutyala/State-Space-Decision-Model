"""Progressive conversion with the accept/rollback gate, separated from the model so resume is testable on CPU.

Invariant (checked by tests/test_progressive_resume.py): after any prefix of steps, the loaded weights and the gate
baseline describe the SAME model: the last accepted student (or the stock teacher if none was accepted). A rejected
step never becomes the baseline, whether the loop ran uninterrupted or was resumed from results.json.
"""
from __future__ import annotations


def gate_verdict(ge, baseline_nll, gate):
    """Per-step dNLL vs the last accepted model + cumulative dNLL vs teacher + final-hidden fidelity."""
    step_dnll = ge["nll_student"] - baseline_nll
    passed = bool(step_dnll <= gate["max_step_delta_nll"]
                  and ge["delta_nll"] <= gate.get("max_cum_delta_nll", float("inf"))
                  and ge["hidden_cos"] >= gate["min_hidden_cos"])
    return passed, step_dnll


def resume_baseline(steps):
    """Gate baseline after a recorded history: nll of the last ACCEPTED step, or None (= teacher) if none accepted."""
    for rec in reversed(steps):
        if rec["accepted"]:
            return rec["gate_eval"]["nll_student"]
    return None


def run_progressive(order, prog, gate, ops, max_layers=None, past_deadline=lambda: False, persist=lambda p: None,
                    log=lambda rec, phase="": None):
    """ops: load(n_done), snapshot(converted)->snap, train(k, i, trial)->dict, evaluate(trial)->gate_eval dict,
    restore(snap, i), save(n_done), teacher_nll()->float, probe(converted)->tag|None."""
    policy = gate.get("policy", "force")
    if policy not in ("force", "enforce", "rollback"):
        raise ValueError(policy)
    if prog["steps"]:
        ops.load(len(prog["steps"]))  # checkpoint written after step n: accepted weights (+ rolled-back layer reset)
    converted = list(prog["converted"])
    baseline = resume_baseline(prog["steps"])
    if baseline is None:
        baseline = ops.teacher_nll()
    max_layers = len(order) if max_layers is None else max_layers
    for k, i in enumerate(order[:max_layers]):
        if k < len(prog["steps"]):
            continue
        if past_deadline():
            log({"skipped_layer": i, "reason": "deadline"}, "deadline")
            break
        snap = ops.snapshot(converted)
        trial = converted + [i]
        extra = ops.train(k, i, trial)
        ge = ops.evaluate(trial)
        passed, step_dnll = gate_verdict(ge, baseline, gate)
        # force = verdict recorded, never enforced (curve mode); enforce/rollback = a failing layer is rolled back
        accepted = passed or policy == "force"
        rec = {"k": k, "layer": i, "gate_eval": ge, "baseline_nll": baseline, "step_delta_nll": step_dnll,
               "gate_pass": passed, "policy": policy, "accepted": accepted, **extra}
        if accepted:
            converted = trial
            baseline = ge["nll_student"]
        else:  # rollback: layer i returns to attention, other GDNs restored to their pre-step weights
            ops.restore(snap, i)
        rec["converted_after"] = list(converted)
        log({kk: v for kk, v in rec.items() if kk != "hist"}, "gate")
        ops.save(k + 1)
        if accepted:
            tag = None if past_deadline() else ops.probe(converted)
            if tag:
                rec["probe_tag"] = tag
            else:
                rec["probes_skipped"] = "deadline"
        prog["steps"].append(rec)
        prog["converted"] = converted
        persist(prog)
    return converted

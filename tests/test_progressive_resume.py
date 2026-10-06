"""CPU synthetic regression: resumed progressive conversion == uninterrupted run (PR #9 Track L review).

A fake model (one scalar "weight" per converted layer, deterministic train/eval) drives convert.progressive.run_progressive.
Histories: (1) rejected LAST step before the interruption, (2) NO accepted step before the interruption. Interruptions
are injected at every step boundary, both after results.json was written and between the checkpoint save and the
results.json write. The test also re-runs with the pre-fix baseline (last step regardless of acceptance) and requires
that the comparison catches it.
Run: python -m tests.test_progressive_resume
"""
from __future__ import annotations

import copy
import json
import sys

import convert.progressive as P

GATE = {"max_step_delta_nll": 0.05, "max_cum_delta_nll": 1.0, "min_hidden_cos": 0.95, "policy": "enforce"}
# per-layer synthetic properties: nll penalty and fidelity loss. "b" is rejected on fidelity while LOWERING nll, so a
# baseline taken from the rejected trial makes the next step look worse than it is.
PEN = {"a": 0.02, "b": -0.30, "c": 0.03, "d": 0.01}
COSP = {"a": 0.01, "b": 0.20, "c": 0.01, "d": 0.01}
TEACHER = 2.0


class Crash(Exception):
    pass


class FakeOps:
    def __init__(self, disk):
        self.disk = disk  # checkpoints survive the "crash"
        self.w = {}

    def load(self, n):
        self.w = copy.deepcopy(self.disk[n])

    def snapshot(self, converted):
        return {j: self.w[j] for j in converted}

    def train(self, k, i, trial):
        for j in trial:
            self.w[j] = self.w.get(j, 0.0) + 1.0
        return {"train_tokens": 1}

    def evaluate(self, trial):
        nll = TEACHER + sum(PEN[j] - 0.01 * self.w.get(j, 0.0) for j in trial)
        return {"nll_student": nll, "nll_teacher": TEACHER, "delta_nll": nll - TEACHER,
                "hidden_cos": 1.0 - sum(COSP[j] for j in trial)}

    def restore(self, snap, i):
        for j, v in snap.items():
            self.w[j] = v
        self.w[i] = 0.0  # stage-1 state

    def save(self, n):
        self.disk[n] = copy.deepcopy(self.w)

    def teacher_nll(self):
        return TEACHER

    def probe(self, converted):
        return f"conv{len(converted)}"


def run(order, crash_at=None, crash_mode="after_persist"):
    """Run to completion; if crash_at is set, crash once at that step and resume from the persisted JSON + disk."""
    disk, store = {0: {}}, {"prog": json.dumps({"steps": [], "converted": []})}
    state = {"n": 0, "crashed": False}

    def persist(p):
        state["n"] += 1
        if crash_at is not None and not state["crashed"] and state["n"] == crash_at and crash_mode == "before_persist":
            state["crashed"] = True
            raise Crash()
        store["prog"] = json.dumps(p)
        if crash_at is not None and not state["crashed"] and state["n"] == crash_at and crash_mode == "after_persist":
            state["crashed"] = True
            raise Crash()

    while True:
        ops = FakeOps(disk)
        prog = json.loads(store["prog"])
        try:
            conv = P.run_progressive(order, prog, GATE, ops, persist=persist)
            return json.loads(store["prog"]), ops.w, conv
        except Crash:
            continue


def strip(prog):
    return [(s["layer"], s["accepted"], round(s["baseline_nll"], 9), round(s["step_delta_nll"], 9))
            for s in prog["steps"]]


def compare(order):
    ref_prog, ref_w, ref_conv = run(order)
    results = []
    for mode in ("after_persist", "before_persist"):
        for c in range(1, len(order)):
            prog, w, conv = run(order, c, mode)
            results.append({"crash_at": c, "mode": mode, "equal": strip(prog) == strip(ref_prog) and w == ref_w
                            and conv == ref_conv})
    return strip(ref_prog), results


def main():
    out = {}
    cases = {"rejected_last_step": ["a", "b", "c", "d"], "no_accepted_step": ["b", "c", "d"]}
    for name, order in cases.items():
        ref, res = compare(order)
        out[name] = {"uninterrupted": ref, "all_resumes_equal": all(r["equal"] for r in res), "resumes": res}
    # sensitivity: the pre-fix baseline (last step's nll even if rejected) must be caught by the same comparison
    fixed = P.resume_baseline
    P.resume_baseline = lambda steps: steps[-1]["gate_eval"]["nll_student"] if steps else None
    try:
        caught = {name: not all(r["equal"] for r in compare(order)[1]) for name, order in cases.items()}
    finally:
        P.resume_baseline = fixed
    out["prefix_bug_detected"] = caught
    out["expected_verdicts_ok"] = (
        [s[1] for s in out["rejected_last_step"]["uninterrupted"]] == [True, False, True, True]
        and [s[1] for s in out["no_accepted_step"]["uninterrupted"]] == [False, True, True])
    out["PASS"] = bool(all(v["all_resumes_equal"] for k, v in out.items() if k in cases)
                       and all(caught.values()) and out["expected_verdicts_ok"])
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

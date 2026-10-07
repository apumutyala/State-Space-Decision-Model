"""Cost arithmetic, schedules, the $525 SF envelope ledger and the qualification -> next-job mapping for the m1 jobs.

    python -m jobs.common.costs            # prints every job's worst case / expected $ and the branch ledger

Worst case for one job = (build_timeout + attempts * max_duration) * batch rate, with attempts = 1 + max(max_restarts,
assumed_restart_floor): the build is counted as billable and the platform is assumed to retry at least once, so the cap
never relies on "only the running attempt is billed". jobs/DECISION_TREE.md quotes these numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVELOPE_USD = 525.0
#: October 2026 MTD actual SF spend (receipts): job-4j3vu 2.48, job-tkfaj 1.79, job-sgihj 1.40, job-c4zkm 0.74,
#: job-mhc3h 0.56, job-4idg4 0.30, job-jucxx 0.25, job-jn47u 0.84, job-jbw22 0.07, job-mden7 1.06, job-5cmat 5.01
#: (LEDGER: 14.49 per get_usage) + job-tvuat 1.62 (get_job receipt: charged_usd_micros 1,619,200, 1,963 billed s)
SPENT_USD = 16.11
#: PLAN.md spend lines
LINES = {"joint_gate": 75.0, "system1": 250.0, "language": 100.0, "reserve": 100.0}
PARAMS = {"qual": "jobs/qual/job_params.json", "qual_c_only": "jobs/qual/job_params_c_only.json", "probe9b": "jobs/probe9b/job_params.json",
          "rwkv7_readout": "jobs/rwkv7/job_params_readout.json", "rwkv7_lora": "jobs/rwkv7/job_params_lora.json",
          "fla_rwkv7": "jobs/fla_rwkv7/job_params.json", "fla_rwkv7_truth_v2": "jobs/fla_rwkv7/job_params_truth_v2.json",
          "cell_step_profile": "jobs/cell_step_profile/job_params.json",
          "full8": "jobs/full8/job_params.json",
          "m2_first_cell": "jobs/m2_first_cell/job_params.json"}
#: committed elsewhere in the envelope (worst-case ceilings from their specs), not built here
OTHER = {"clef_baseline_ABC": 47.52}  # jobs/clef_baseline/JOB_SPEC.md: per-job ceilings at $2.97/h
FULL8_BUDGET = 150.0  # PLAN.md / DECISION_TREE branch A: ~$150 for the full conversion; priced in jobs/full8 (worst case)


def load(name: str) -> dict:
    return json.loads((ROOT / PARAMS[name]).read_text(encoding="utf-8"))


def attempts(p: dict) -> int:
    return 1 + max(p["max_restarts"], p.get("assumed_restart_floor", 0))


def worst_case_usd(p: dict) -> float:
    return round((p["build_timeout_minutes"] + attempts(p) * p["max_duration_minutes"]) * p["batch_usd_per_min"], 2)


def single_attempt_usd(p: dict) -> float:
    """What validate_job reports as max_cost_usd (one running attempt)."""
    return round(p["max_duration_minutes"] * p["batch_usd_per_min"], 2)


def expected_usd(p: dict) -> float | None:
    e = p.get("expected_minutes")
    return round(sum(e.values()) * p["batch_usd_per_min"], 2) if e else None


def schedule_minutes(p: dict) -> int:
    return sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"]


def schedule_ok(p: dict) -> bool:
    if "A_per_test" in p["phase_caps_minutes"]:  # jobs/qual layout (c_only has no B)
        c = p["phase_caps_minutes"]
        total = (c["setup"] + c["A_per_test"] * c["A_tests"] + c.get("B", 0) + c["C_weight_wait"] + c["C"]
                 + p["reserve_minutes"])
        return total <= p["max_duration_minutes"]
    return schedule_minutes(p) <= p["max_duration_minutes"]


def cap_ok(p: dict) -> bool:
    return worst_case_usd(p) <= p["hard_cap_usd"]


def concurrent_cap_usd(names: list[str]) -> int:
    """Workspace-cap increment when several jobs can bill at once. A job chained with depends_on is submitted while
    its parent is still queued or running, so the cap set at submit must cover the parent's remaining worst case PLUS
    the child's: set_workspace_cap(MTD + this). (rwkv7 A + B chained: ceil(16.34 + 19.30) = 36, not 20.)"""
    import math

    return math.ceil(sum(worst_case_usd(load(n)) for n in names))


# ----------------------------------------------------------------------------- re-pricing (lean branch)
def stage_tokens(cfg: dict, stage: str, tok_s: float) -> float:
    """Tokens one capped stage trains at a measured rate (the cap is wall-clock, so $ is fixed and tokens move)."""
    g = cfg[stage]["guard"]
    return tok_s * 60 * g["max_minutes"]


def reprice_probe(qual_decision: dict, full_cfg: dict, lean_cfg: dict, p: dict) -> dict:
    """From the qualification decision.json (measured stage-2 tok/s and MFU) to the numbers the probe spend issue
    must quote: tokens per capped stage, and the stop thresholds the measured rate implies."""
    s2 = qual_decision.get("stage2_steady_tok_s")
    s1 = qual_decision.get("stage1_steady_tok_s")
    out = {"measured": {"stage1_tok_s": s1, "stage2_tok_s": s2, "stage2_mfu": qual_decision.get("stage2_mfu"),
                        "peak_gb": qual_decision.get("peak_gb")},
           "worst_case_usd": worst_case_usd(p), "expected_usd": expected_usd(p)}
    for name, cfg in (("full", full_cfg), ("lean", lean_cfg)):
        out[name] = {"stage2_tokens": stage_tokens(cfg, "stage2", s2) if s2 else None,
                     "stage1_tokens_cap": cfg["stage1"]["steps"] * cfg["seq_len"] * cfg["batch"],
                     "min_tok_s": {"stage1": cfg["stage1"]["guard"]["min_tok_s"],
                                   "stage2": cfg["stage2"]["guard"]["min_tok_s"]},
                     "passes_stop_rule": bool(s2 and s2 >= cfg["stage2"]["guard"]["min_tok_s"]
                                              and (s1 is None or s1 >= cfg["stage1"]["guard"]["min_tok_s"]))}
    return out


# ----------------------------------------------------------------------------- the decision tree, as code
QUAL_TO_NEXT = {
    "open-$40-one-layer-probe-spend-issue": ["probe9b:full", "rwkv7_readout"],
    "re-price-lean-plan": ["probe9b:lean", "rwkv7_readout"],
    "fold": ["rwkv7_readout", "rwkv7_lora"],
}
PROBE_TO_NEXT = {
    "proceed-all-8": ["full8 (jobs/full8, issue #87)"],
    "retrieval-fail": ["rwkv7_lora", "report cost-of-attention-freedom curve", "decision-needed: ship hybrid?"],
    "quality-fail": ["rwkv7_lora"],
}


def next_jobs_after_qual(verdict: str | None, exit_code: int | None = 0) -> list[str]:
    """Qualification outcome -> the jobs that may run next (each still needs its own spend issue).
    Any non-zero qual exit, or an unknown/unmeasured verdict, runs nothing new: diagnose, then at most one qual re-run
    from the reserve."""
    if exit_code in (3, 4):
        return list(QUAL_TO_NEXT["fold"])
    if exit_code not in (0, None) or verdict not in QUAL_TO_NEXT:
        return ["diagnose; re-run qual once from reserve (<= $20)"]
    return list(QUAL_TO_NEXT[verdict])


def next_jobs_after_probe(verdict: str | None, exit_code: int | None = 0) -> list[str]:
    if exit_code in (3, 4):  # the probe's own stop rule: no further 9B spend
        return ["rwkv7_lora"]
    if exit_code not in (0, None) or verdict not in PROBE_TO_NEXT:
        return ["diagnose; at most one probe re-run from reserve"]
    return list(PROBE_TO_NEXT[verdict])


#: full-8 verdict (scripts/decide_full8.py, rule pre-registered at be9eed8) -> next jobs (jobs/full8/rule.json: verdicts)
FULL8_TO_NEXT = {
    "q9af-backbone": ["pin q9af.converted_sha256 (own reviewed PR)", "m2_parent_arm PARENT=q9af", "m2_pilot PARENT=q9af"],
    "retrieval-fail": ["report cost-of-attention-freedom curve", "decision-needed: m2 backbone (Q9-AF / RWKV7 / hybrid)"],
    "quality-fail": ["m2 on rwkv7"],
    "partial-conversion": ["m2 on rwkv7", "report Q9-H<k> hybrid + curve", "decision-needed: ship hybrid?"],
    "no-conversion": ["m2 on rwkv7"],
}


def next_jobs_after_full8(verdict: str | None, exit_code: int | None = 0) -> list[str]:
    """Any non-zero exit is not a verdict. A preemption / crash / stop keeps every finished layer in the checkpoint dir
    (and a salvage export): resubmit once from the reserve (resumes); a no-measurement decision (exit 9): diagnose."""
    if exit_code not in (0, None):
        if exit_code in (2, 7):
            return ["resubmit once from reserve (resume from checkpoint)", "diagnose"]
        return ["diagnose; at most one full8 re-run from reserve"]
    if verdict not in FULL8_TO_NEXT:
        return ["diagnose; at most one full8 re-run from reserve"]
    return list(FULL8_TO_NEXT[verdict])


def branch_ledger() -> dict:
    """Worst-case $ per branch through the tree, against the envelope and the PLAN lines."""
    w = {k: worst_case_usd(load(k)) for k in PARAMS}
    joint = w["qual"] + OTHER["clef_baseline_ABC"]
    branches = {
        "A: qual pass -> probe proceed -> full-8 (+rwkv7 readout comparator)":
            {"m1": w["probe9b"] + w["full8"] + w["rwkv7_readout"]},
        "B: qual pass/lean -> probe fail -> rwkv7 readout + lora": {"m1": w["probe9b"] + w["rwkv7_readout"] + w["rwkv7_lora"]},
        "C: qual fold -> rwkv7 readout + lora": {"m1": w["rwkv7_readout"] + w["rwkv7_lora"]},
    }
    for b in branches.values():
        b["m1"] = round(b["m1"], 2)
        b["system1_left_for_m2"] = round(LINES["system1"] - b["m1"], 2)
        b["envelope_total_worst"] = round(SPENT_USD + joint + b["m1"], 2)
        b["fits_envelope"] = b["envelope_total_worst"] <= ENVELOPE_USD
    return {"worst_case_usd": w, "joint_gate_worst": round(joint, 2), "joint_gate_fits": joint <= LINES["joint_gate"],
            "envelope": ENVELOPE_USD, "spent": SPENT_USD, "branches": branches}


def main():
    rows = {}
    for k in PARAMS:
        p = load(k)
        rows[k] = {"worst_case_usd": worst_case_usd(p), "single_attempt_usd": single_attempt_usd(p),
                   "expected_usd": expected_usd(p), "hard_cap_usd": p["hard_cap_usd"], "cap_ok": cap_ok(p),
                   "schedule_ok": schedule_ok(p)}
    print(json.dumps({"jobs": rows, "ledger": branch_ledger()}, indent=1))
    return 0 if all(r["cap_ok"] and r["schedule_ok"] for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Cost arithmetic, schedules, the granted-credit ledger and the qualification -> next-job mapping for the m1 jobs.

    python -m jobs.common.costs            # prints every job's worst case / expected $ and the branch ledger

Rates live in jobs/common/rates.py (current: $0.063/GPU-min billed and quoted from 2026-10-08; $0.0495 before). Each
job_params.json carries the rate as `batch_usd_per_min` for the in-job scripts; the lint's `rate_current` check holds it
equal to rates.BATCH_USD_PER_MIN.

Worst case for one job = (build_timeout + attempts * max_duration) * batch rate, with attempts = 1 + max(max_restarts,
assumed_restart_floor): the build is counted as billable and the platform is assumed to retry at least once, so the cap
never relies on "only the running attempt is billed". jobs/DECISION_TREE.md quotes these numbers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_rates():
    """rates.py beside THIS file (the lint and launch load costs.py from a staged tree by path, so a package import
    could pick up another checkout's rates)."""
    f = Path(__file__).resolve().with_name("rates.py")
    spec = importlib.util.spec_from_file_location(f"_rates_{abs(hash(str(f)))}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rates = _load_rates()

#: The fixed $525 envelope is RETIRED (2026-10-10, coordinator / agent-a: the operator lifted the $525 cap and credit was
#: topped up 2026-10-09 with $250 + $10). The ledger checks spend + committed worst cases against the TOTAL GRANTED CREDIT
#: for the period, read from the newest committed `autor billing --json` receipt (credit_grants). Historical analyses
#: that quote the $525 envelope (PLAN, LEDGER, receipts) are left as written.
#: Month-to-date SF spend comes from the newest committed `autor usage --json` receipt (receipts/<date>-*/*usage*.json,
#: object "list" with total_usd), not a hand-kept literal: the old SPENT_USD = 16.11 (2026-10-07) was stale by $41.87
#: against LEDGER's 2026-10-08 $57.98. lint warns when the newest receipt is older than USAGE_STALE_DAYS.
USAGE_STALE_DAYS = 2


def latest_usage_receipt(root: Path = ROOT) -> dict | None:
    """The newest usage receipt: latest month, then latest receipt-directory date, then the larger MTD (MTD only grows
    within a month, so the larger of two same-day snapshots is the later one). None when no receipt is shipped."""
    import datetime as dt

    best = None
    for f in (root / "receipts").glob("*/*usage*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            day = dt.date.fromisoformat(f.parent.name[:10])
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict) or d.get("object") != "list" or "total_usd" not in d or "month" not in d:
            continue
        key = (d["month"], day, float(d["total_usd"]))
        if best is None or key > best[0]:
            best = (key, {"path": f.relative_to(root).as_posix(), "month": d["month"], "date": day.isoformat(),
                          "total_usd": float(d["total_usd"])})
    return best[1] if best else None


_USAGE = latest_usage_receipt()
SPENT_USD = round(_USAGE["total_usd"], 2) if _USAGE else None


def granted_credit_usd(billing: dict) -> float:
    """Total credit granted for the billing receipt's month: every grant made by the end of that month that had not
    expired before the month began (promo, top-ups), summed at its granted amount_usd (not what remains)."""
    import datetime as dt

    y, m = (int(x) for x in billing["month"].split("-"))
    start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=dt.timezone.utc)
    total = 0.0
    for g in billing.get("credit_grants", []):
        granted = dt.datetime.fromisoformat(g["granted_at"])
        expires = dt.datetime.fromisoformat(g["expires_at"]) if g.get("expires_at") else None
        if granted < end and (expires is None or expires >= start):
            total += float(g["amount_usd"])
    return round(total, 2)


def latest_billing_receipt(root: Path = ROOT) -> dict | None:
    """The newest `autor billing --json` receipt (object "billing" with credit_grants): latest month, then latest
    receipt-directory date, then the most grants. None when no receipt is shipped."""
    import datetime as dt

    best = None
    for f in (root / "receipts").glob("*/*billing*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            day = dt.date.fromisoformat(f.parent.name[:10])
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict) or d.get("object") != "billing" or "credit_grants" not in d or "month" not in d:
            continue
        key = (d["month"], day, len(d["credit_grants"]))
        if best is None or key > best[0]:
            best = (key, {"path": f.relative_to(root).as_posix(), "month": d["month"], "date": day.isoformat(),
                          "granted_usd": granted_credit_usd(d), "grants": [
                              {k: g.get(k) for k in ("kind", "amount_usd", "granted_at", "expires_at")}
                              for g in d["credit_grants"]]})
    return best[1] if best else None


_BILLING = latest_billing_receipt()
#: total granted credit for the period (replaces the retired ENVELOPE_USD = 525.0)
GRANTED_USD = _BILLING["granted_usd"] if _BILLING else None
#: PLAN.md spend lines
LINES = {"joint_gate": 75.0, "system1": 250.0, "language": 100.0, "reserve": 100.0}
PARAMS = {"qual": "jobs/qual/job_params.json", "qual_c_only": "jobs/qual/job_params_c_only.json", "probe9b": "jobs/probe9b/job_params.json",
          "rwkv7_code_label_h100": "jobs/rwkv7_code_label_h100/job_params.json",
          "rwkv7_readout": "jobs/rwkv7/job_params_readout.json", "rwkv7_readout_g1v2": "jobs/rwkv7/job_params_readout_g1v2.json",
          "rwkv7_readout_g1v3": "jobs/rwkv7/job_params_readout_g1v3.json",
          "rwkv7_lora": "jobs/rwkv7/job_params_lora.json",
          "fla_rwkv7": "jobs/fla_rwkv7/job_params.json", "fla_rwkv7_truth_v2": "jobs/fla_rwkv7/job_params_truth_v2.json",
          "cell_step_profile": "jobs/cell_step_profile/job_params.json",
          "perf_step_gate": "jobs/perf_step_gate/job_params.json",
          "clef_index_subset": "jobs/clef_baseline/job_params_index_subset.json",
          "clef_index_full": "jobs/clef_baseline/job_params_index_full.json",
          "clef_index_full_reference": "jobs/clef_baseline/job_params_index_full_reference.json",
          "clef_teacher_flash": "jobs/clef_baseline/job_params_teacher_clef_flash.json",
          "clef_teacher_clef": "jobs/clef_baseline/job_params_teacher_clef.json",
          "clef_overlength_flash": "jobs/clef_baseline/job_params_overlength_flash.json",
          "clef_hijack_matched": "jobs/clef_baseline/job_params_hijack_matched.json",
          "full8": "jobs/full8/job_params.json",
          "q9af_v2": "jobs/q9af_v2/job_params.json",
          "m2_first_cell": "jobs/m2_first_cell/job_params.json",
          **{f"q9_feature_cache_{p}": f"jobs/q9_feature_cache/job_params_{p}.json"
             for p in ("stock_dev", "stock_train", "conv1_dev", "conv1_train", "e5_dev", "e5_train")},
          **{f"q9af_cell_{p}": f"jobs/q9af_cell/job_params_{p}.json" for p in ("parent", "cell_r64", "cell_r256")}}
#: m2_first_cell modes (first_cell_job.sh <mode>) register by discovery as m2_first_cell_<mode>, so a new variant adds
#: no line here: its job_params_<mode>.json is priced and capped like every entry above (the lint's DECISION_TREE.md
#: registration and JOB_SPEC worst-case checks still apply).
PARAMS.update({f"m2_first_cell_{f.stem[len('job_params_'):]}": f"jobs/m2_first_cell/{f.name}"
               for f in sorted((ROOT / "jobs/m2_first_cell").glob("job_params_*.json"))})
#: committed elsewhere in the envelope (worst-case ceilings from their specs), not built here. Clef baseline (JOB_SPEC):
#: A ran as job-sgihj ($1.40, in SPENT_USD); C (27B, job A shape) is superseded for the Index by clef_index_subset +
#: clef_index_full (PARAMS); B (Clef-flash GCG, $29.70 one-attempt ceiling) is still planned.
#: clef_baseline_B is a 600-min one-attempt ceiling: $29.70 at the pre-2026-10-08 $0.0495/min, re-priced at the current rate.
OTHER = {"clef_baseline_B": round(600 * rates.BATCH_USD_PER_MIN, 2)}
JOINT_GATE_JOBS = ("qual",)
#: Clef-27B Index measurement (subset canary, full DEV, Clef-flash 32k over-length sensitivity): re-priced at the
#: card's 5.39x latency ratio after the PR #98 review, which puts it past the $75 joint-gate line; the coordinator
#: authorised that budget (2026-10-07, PR #98 review). Counted against granted credit, reported apart from the joint line.
#: the full-DEV slot is clef_index_full_reference (GDN=reference, JOB_SPEC 4d): job-jxevr (index_full, fast GDN) failed
#: kernel_path at S1 and the line moved to the reference job (job-zfexa), so the line counts its $49.90 worst case
#: (PR #241 review B, 2026-10-10; it counted clef_index_full at $29.86 before)
CLEF27B_INDEX_JOBS = ("clef_index_subset", "clef_index_full_reference", "clef_overlength_flash")
#: optional m2 distillation teachers (System 1 line, only if a distillation cell is approved)
TEACHER_JOBS = ("clef_teacher_flash", "clef_teacher_clef")
FULL8_BUDGET = 150.0  # PLAN.md / DECISION_TREE branch A: ~$150 for the full conversion; priced in jobs/full8 (worst case)


def load(name: str) -> dict:
    return json.loads((ROOT / PARAMS[name]).read_text(encoding="utf-8"))


def attempts(p: dict) -> int:
    return 1 + max(p["max_restarts"], p.get("assumed_restart_floor", 0))


def worst_case_usd(p: dict) -> float:
    return round((p["build_timeout_minutes"] + attempts(p) * p["max_duration_minutes"]) * p["batch_usd_per_min"], 2)


def single_attempt_usd(p: dict) -> float:
    """One running attempt at the billed rate. validate_job's max_cost_usd was this at $0.0495 (2026-10-06: qual 180 min
    -> $8.91); from 2026-10-08 billed and quoted rates are both $0.063 (quoted_max_cost_usd)."""
    return round(p["max_duration_minutes"] * p["batch_usd_per_min"], 2)


#: The platform's quoted max_cost_usd / preflight worst case = max_duration x QUOTE_USD_PER_MIN with the build excluded
#: (job-i3buq 637 min -> $40.13, job-zpyis 828 min -> $52.16, job-a6vt6 742 min -> $46.75). From 2026-10-08 the billed
#: batch rate equals it ($0.063/min, receipts 63,000 micro-$/GPU-min); before, the platform billed $0.0495. See rates.py.
QUOTE_USD_PER_MIN = rates.QUOTE_USD_PER_MIN


def quoted_max_cost_usd(p: dict) -> float:
    """What validate_job / submit_job quote as max_cost_usd (one attempt, build excluded); the workspace cap must clear it."""
    return round(p["max_duration_minutes"] * QUOTE_USD_PER_MIN, 2)


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


#: Every branch-ledger entry -> the SF job ids whose runs did that entry's committed work. An entry is RAN when every
#: listed id is terminal in a committed job receipt (receipts/**/job-<id>.json, `autor job get --json`); a RAN entry
#: counts 0 toward committed worst cases, because its actual cost is already inside SPENT_USD (the ledger reports that
#: actual for reference). Usage receipts list top jobs without a status, so they cannot mark a job terminal on their
#: own. An empty list = not run.
LEDGER_JOB_IDS: dict[str, list[str]] = {
    "qual": ["job-jbw22", "job-c4zkm", "job-4j3vu"],  # v1, v2 (A+B pass, C OOM), C-only re-run PASS
    "clef_baseline_B": [],
    "clef_index_subset": ["job-5mw9h", "job-d2u54"],  # failed canary, then the succeeded subset run
    "clef_index_full_reference": ["job-jxevr", "job-zfexa"],  # index_full fast-GDN attempt (failed S1), reference run
    "clef_overlength_flash": [],
    "probe9b": ["job-5cmat"],
    "full8": ["job-2xe5w"],
    "rwkv7_readout": ["job-mhc3h", "job-tkfaj"],  # stage A v1 (rate guard stop), v2 succeeded
    "rwkv7_lora": [],
    "clef_teacher_flash": [],
    "clef_teacher_clef": [],
}
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled", "cancelled", "build_failed", "self_terminated",
                               "timed_out", "expired"})


def job_receipts(root: Path = ROOT) -> dict[str, dict]:
    """job id -> its newest committed `autor job get` receipt (receipts/**/job-*.json, object "job")."""
    out: dict[str, tuple] = {}
    for f in (root / "receipts").glob("**/job-*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict) or d.get("object") != "job" or not d.get("job_id") or "status" not in d:
            continue
        key = (f.relative_to(root / "receipts").parts[0][:10], d.get("finished_at") or "")
        if d["job_id"] not in out or key > out[d["job_id"]][0]:
            out[d["job_id"]] = (key, dict(d, _path=f.relative_to(root).as_posix()))
    return {k: v[1] for k, v in out.items()}


def ran_entries(root: Path = ROOT, job_ids: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """Ledger entries whose every listed job is terminal in a committed job receipt, with their actual billed cost."""
    rec = job_receipts(root)
    out = {}
    for name, ids in (LEDGER_JOB_IDS if job_ids is None else job_ids).items():
        if ids and all(i in rec and rec[i]["status"] in TERMINAL_STATUSES for i in ids):
            usd = sum(((rec[i].get("receipt") or {}).get("charged_usd_micros") or 0) for i in ids) / 1e6
            out[name] = {"job_ids": list(ids), "statuses": {i: rec[i]["status"] for i in ids},
                         "actual_usd": round(usd, 2), "receipts": [rec[i]["_path"] for i in ids]}
    return out


def branch_ledger(root: Path = ROOT, job_ids: dict[str, list[str]] | None = None) -> dict:
    """Worst-case $ per branch through the tree: MTD spend + every committed worst case of work NOT yet run, against the
    total granted credit for the period, and the PLAN lines. The line totals (joint gate, Clef-27B Index, teachers,
    m1) stay full worst cases, as authorised; only the credit check drops RAN entries (their spend is in SPENT_USD)."""
    if SPENT_USD is None:
        raise RuntimeError("no usage receipt under receipts/ (autor usage --json): MTD spend unknown")
    if GRANTED_USD is None:
        raise RuntimeError("no billing receipt under receipts/ (autor billing --json): granted credit unknown")
    w = {k: worst_case_usd(load(k)) for k in PARAMS}
    ran = ran_entries(root, job_ids)
    ids = LEDGER_JOB_IDS if job_ids is None else job_ids

    def worst(name: str) -> float:
        return OTHER[name] if name in OTHER else w[name]

    def committed(names) -> float:
        return round(sum(0.0 if n in ran else worst(n) for n in names), 2)

    joint_names = (*JOINT_GATE_JOBS, "clef_baseline_B")
    joint = sum(worst(n) for n in joint_names)
    teachers = round(sum(w[k] for k in TEACHER_JOBS), 2)
    clef27b = round(sum(w[k] for k in CLEF27B_INDEX_JOBS), 2)
    m1_names = {
        "A: qual pass -> probe proceed -> full-8 (+rwkv7 readout comparator)": ("probe9b", "full8", "rwkv7_readout"),
        "B: qual pass/lean -> probe fail -> rwkv7 readout + lora": ("probe9b", "rwkv7_readout", "rwkv7_lora"),
        "C: qual fold -> rwkv7 readout + lora": ("rwkv7_readout", "rwkv7_lora"),
    }
    committed_lines = {"joint_gate": committed(joint_names), "clef27b_index": committed(CLEF27B_INDEX_JOBS),
                       "teachers": committed(TEACHER_JOBS)}
    branches = {}
    for label, names in m1_names.items():
        b = branches[label] = {"m1": round(sum(w[n] for n in names), 2), "m1_committed": committed(names)}
        b["system1_left_for_m2"] = round(LINES["system1"] - b["m1"], 2)
        b["system1_left_after_teachers"] = round(b["system1_left_for_m2"] - teachers, 2)
        b["spend_plus_worst"] = round(SPENT_USD + sum(committed_lines.values()) + b["m1_committed"], 2)
        b["fits_granted_credit"] = b["spend_plus_worst"] <= GRANTED_USD
    return {"worst_case_usd": w, "joint_gate_worst": round(joint, 2), "joint_gate_fits": joint <= LINES["joint_gate"],
            "teachers_optional_worst": teachers, "clef27b_index_authorised_worst": clef27b,
            "committed_worst_not_run": committed_lines, "job_ids": {k: list(v) for k, v in ids.items()},
            "ran": ran, "ran_actual_usd_in_spent": round(sum(r["actual_usd"] for r in ran.values()), 2),
            "granted_credit_usd": GRANTED_USD, "granted_source": _BILLING and {k: _BILLING[k] for k in ("path", "date", "month")},
            "spent": SPENT_USD, "spent_source": _USAGE, "branches": branches}


def main():
    rows = {}
    for k in PARAMS:
        p = load(k)
        rows[k] = {"worst_case_usd": worst_case_usd(p), "single_attempt_usd": single_attempt_usd(p),
                   "quoted_max_cost_usd": quoted_max_cost_usd(p),
                   "expected_usd": expected_usd(p), "hard_cap_usd": p["hard_cap_usd"], "cap_ok": cap_ok(p),
                   "schedule_ok": schedule_ok(p)}
    print(json.dumps({"jobs": rows, "ledger": branch_ledger()}, indent=1))
    return 0 if all(r["cap_ok"] and r["schedule_ok"] for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

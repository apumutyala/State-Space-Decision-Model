"""CPU tests for jobs/full8 (the full 8-layer Qwen3.5-9B -> GDN conversion, Q9-AF, issue #87).

Part 1 (the pre-registration commit): the binding rule (jobs/full8/rule.json), its provenance from pre-existing plan
text, the independent gate audit and every verdict path of scripts/decide_full8.py. No GPU, no weights.

    python -m pytest -q tests/test_full8.py
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from scripts import decide_full8 as D

ROOT = Path(__file__).resolve().parents[1]
RULE = json.loads((ROOT / "jobs/full8/rule.json").read_text(encoding="utf-8"))
SPEC = (ROOT / "jobs/full8/JOB_SPEC.md").read_text(encoding="utf-8")
PROBE = json.loads((ROOT / "configs/probe9b_h100.json").read_text(encoding="utf-8"))
ORDER = [23, 31, 15, 11, 7, 19, 3, 27]  # any ascending-KL order works; this one is only test data


# ----------------------------------------------------------------------------- provenance of every constant
def test_rule_constants_come_from_the_511e8dd_rule_and_gate():
    sr, g = PROBE["success_rule"], PROBE["gate"]  # tests/test_probe9b.py pins these to `git show 511e8dd`
    assert {k: RULE["gate"][k] for k in ("max_step_delta_nll", "max_cum_delta_nll", "min_hidden_cos", "policy",
                                         "n_windows")} == {k: g[k] for k in ("max_step_delta_nll", "max_cum_delta_nll",
                                                                             "min_hidden_cos", "policy", "n_windows")}
    q, r = RULE["quality"], RULE["retrieval"]
    assert q["mmlu_drop_max_pts"] == sr["mmlu_drop_max_pts"] == 3.0
    assert (q["index_diff_point_min"], q["index_diff_lo_min"]) == (sr["index_diff_point_min"], sr["index_diff_lo_min"])
    assert (r["mqar_drop_max"], r["niah_mean_drop_max"], r["niah_length_drop_max"]) == (
        sr["mqar_drop_max"], sr["niah_mean_drop_max"], sr["niah_length_drop_max"])
    # the ONLY rescaled bound: ppl, from the plan's own 8-layer cumulative dNLL budget (rounded down, never looser)
    for ctx in ("ctx1024", "ctx4096"):
        assert q["ppl_ratio_max"][ctx] == math.floor(math.exp(g["max_cum_delta_nll"]) * 1e4) / 1e4 == 1.1618
    assert q["ppl_ratio_derivation"] == "exp(gate.max_cum_delta_nll)"
    assert abs(math.log(sr["ppl_ratio_max"]["ctx1024"]) - g["max_cum_delta_nll"] / 8) < 0.002  # 1.02 = per-layer share
    assert RULE["attention_layers"] == [i for i in range(32) if (i + 1) % 4 == 0]


def test_rule_cites_pre_existing_plan_text():
    handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    assert "per-layer gate ΔNLL ≤ 0.05, cumulative ≤ 0.15, hidden cos ≥ 0.95" in handoff
    assert "cost-of-attention-freedom curve" in handoff and "decision-needed" in handoff
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    assert "we do **not** quietly add attention" in plan and "The pure-recurrent entrant is still submitted" in plan
    assert "511e8dd" in RULE["_comment"] and "caa8f55" in RULE["_comment"] and "job-5cmat" in RULE["_comment"]


def test_verdicts_are_exactly_the_rule_verdicts_and_each_has_a_next_step():
    assert set(RULE["verdicts"]) == set(D.VERDICTS)
    for v, nxt in RULE["verdicts"].items():
        assert "->" in nxt or "decision-needed" in nxt, v
    assert "q9af_from_stock" in RULE["verdicts"]["partial-conversion"]
    assert "NOT attention-free" in RULE["verdicts"]["partial-conversion"]
    assert "converted_sha256" in RULE["verdicts"]["q9af-backbone"]


# ----------------------------------------------------------------------------- synthetic, gate-consistent results
STOCK = {"ppl": {"ctx1024": {"ppl": 8.45}, "ctx4096": {"ppl": 7.17}}, "mmlu": {"acc": 0.80, "n": 285},
         "mqar": {"mean_acc": 0.92}, "niah": {"L1024": {"acc": 1.0}, "L16384": {"acc": 1.0}, "mean_acc": 1.0}}


def make_R(step_dnll=0.01, reject=(), cos=0.99, final=None, n_steps=8, teacher=1.96):
    """A results.json whose progressive record is exactly what convert.progressive writes under policy enforce."""
    kl = {str(i): 0.01 * (k + 1) for k, i in enumerate(ORDER)}
    R = {"importance": {"order": list(ORDER), "per_layer": {i: {"kl": v} for i, v in kl.items()}},
         "probe_stock": copy.deepcopy(STOCK), "progressive": {"steps": [], "converted": []}}
    base, conv = teacher, []
    for k, i in enumerate(ORDER[:n_steps]):
        d = step_dnll[k] if isinstance(step_dnll, (list, tuple)) else step_dnll
        ns = base + d
        ge = {"nll_student": ns, "nll_teacher": teacher, "delta_nll": ns - teacher, "kl": 0.02, "hidden_cos": cos}
        passed = d <= 0.05 and ns - teacher <= 0.15 and cos >= 0.95 and i not in reject
        if i in reject:  # a layer the gate rejects (fidelity)
            ge["hidden_cos"] = 0.9
        s = {"k": k, "layer": i, "gate_eval": ge, "baseline_nll": base, "step_delta_nll": d, "gate_pass": passed,
             "policy": "enforce", "accepted": passed}
        if passed:
            conv = conv + [i]
            base = ns
            s["probe_tag"] = f"conv{len(conv)}_kept{8 - len(conv)}"
            p = copy.deepcopy(final or STOCK)
            p["modes"] = {str(j): "student" for j in conv}
            R[f"probe_{s['probe_tag']}"] = p
        s["converted_after"] = list(conv)
        R["progressive"]["steps"].append(s)
    R["progressive"]["converted"] = conv
    return R


CMP = {"diff": {"index": {"point": -0.5, "lo": -2.0, "hi": 1.0}}}


def _final(**kw):
    f = copy.deepcopy(STOCK)
    for k, v in kw.items():
        if k == "ppl4k":
            f["ppl"]["ctx4096"]["ppl"] = v
        elif k == "mmlu":
            f["mmlu"]["acc"] = v
        elif k == "mqar":
            f["mqar"]["mean_acc"] = v
        elif k == "niah_long":
            f["niah"]["L16384"]["acc"] = v
            f["niah"]["mean_acc"] = (1.0 + v) / 2
    return f


def _case(name):
    cmp = copy.deepcopy(CMP)
    if name == "all pass":
        return make_R(), cmp, "q9af-backbone"
    if name == "ppl at the derived bound":
        return make_R(final=_final(ppl4k=7.17 * 1.1617)), cmp, "q9af-backbone"
    if name == "ppl past the derived bound":
        return make_R(final=_final(ppl4k=7.17 * 1.1619)), cmp, "quality-fail"
    if name == "ppl 1.05 (fails the 1-layer 1.02, passes the 8-layer bound)":
        return make_R(final=_final(ppl4k=7.17 * 1.05)), cmp, "q9af-backbone"
    if name == "mmlu":
        return make_R(final=_final(mmlu=0.769)), cmp, "quality-fail"
    if name == "index point":
        cmp["diff"]["index"]["point"] = -1.01
        return make_R(), cmp, "quality-fail"
    if name == "index lo":
        cmp["diff"]["index"]["lo"] = -2.51
        return make_R(), cmp, "quality-fail"
    if name == "mqar":
        return make_R(final=_final(mqar=0.81)), cmp, "retrieval-fail"
    if name == "niah one length":
        return make_R(final=_final(niah_long=0.74)), cmp, "retrieval-fail"
    if name == "quality before retrieval":
        return make_R(final=_final(mmlu=0.70, mqar=0.5)), cmp, "quality-fail"
    if name == "one layer rejected":
        return make_R(reject=(3,)), cmp, "partial-conversion"
    if name == "partial even if every number passes":
        return make_R(reject=(23, 31)), cmp, "partial-conversion"
    if name == "cumulative budget exhausted at layer 8":
        return make_R(step_dnll=0.02), cmp, "partial-conversion"  # 8 x 0.02 = 0.16 > 0.15: the 8th is rejected
    if name == "all rejected":
        return make_R(reject=tuple(ORDER)), None, "no-conversion"
    raise KeyError(name)


CASES = ["all pass", "ppl at the derived bound", "ppl past the derived bound",
         "ppl 1.05 (fails the 1-layer 1.02, passes the 8-layer bound)", "mmlu", "index point", "index lo", "mqar",
         "niah one length", "quality before retrieval", "one layer rejected", "partial even if every number passes",
         "cumulative budget exhausted at layer 8", "all rejected"]


@pytest.mark.parametrize("name", CASES)
def test_every_verdict_path(name, tmp_path):
    R, cmp, want = _case(name)
    d = D.decide(R, cmp, RULE)
    assert d["verdict"] == want, (name, d["verdict"], d["gate_audit_problems"])
    assert D.valid_decision(d) and d["next"] == RULE["verdicts"][want]
    assert d["attention_free"] == (d["n_converted"] == 8)
    if want == "partial-conversion":
        assert d["hybrid_name"] == f"Q9-H{8 - d['n_converted']}" and d["attention_layers_kept"]
    # the CLI agrees and exits 0
    (tmp_path / "r.json").write_text(json.dumps(R))
    (tmp_path / "c.json").write_text(json.dumps(cmp))
    (tmp_path / "rule.json").write_text(json.dumps(RULE))
    rc = D.main(str(tmp_path / "r.json"), str(tmp_path / "c.json") if cmp else "-", str(tmp_path / "rule.json"),
                str(tmp_path / "d.json"))
    assert rc == 0 and json.loads((tmp_path / "d.json").read_text())["verdict"] == want


def test_cumulative_budget_rejects_exactly_the_eighth_layer():
    d = D.decide(make_R(step_dnll=0.02), CMP, RULE)
    assert d["converted"] == ORDER[:7] and d["attention_layers_kept"] == [27]


def test_report_is_never_binding_and_carries_the_curve(monkeypatch):
    R, cmp, want = _case("all pass")
    d = D.decide(R, cmp, RULE)
    rep = d["report"]
    assert rep["binding"] is False and len(rep["curve"]) == 9
    assert [p["attention_layers_kept"] for p in rep["curve"]] == list(range(8, -1, -1))
    monkeypatch.setattr(D, "curve", lambda *a: 1 / 0)  # a broken report never changes the verdict
    d2 = D.decide(R, cmp, RULE)
    assert d2["verdict"] == want and "error" in d2["report"] and D.valid_decision(d2)


@pytest.mark.parametrize("tamper", ["gate_pass_flipped", "accepted_but_failing", "wrong_policy", "wrong_baseline",
                                    "order_not_ascending", "layer_out_of_order", "converted_list", "missing_step",
                                    "probe_modes", "delta_inconsistent"])
def test_gate_audit_refuses_a_record_that_did_not_run_as_preregistered(tamper):
    R = make_R(reject=(3,))
    st = R["progressive"]["steps"]
    if tamper == "gate_pass_flipped":
        st[6]["gate_pass"] = True
    elif tamper == "accepted_but_failing":
        st[6]["gate_pass"] = st[6]["accepted"] = True
    elif tamper == "wrong_policy":
        st[0]["policy"] = "force"
    elif tamper == "wrong_baseline":
        st[2]["baseline_nll"] += 0.01
    elif tamper == "order_not_ascending":
        R["importance"]["per_layer"]["23"]["kl"] = 0.5
    elif tamper == "layer_out_of_order":
        st[0]["layer"], st[1]["layer"] = st[1]["layer"], st[0]["layer"]
    elif tamper == "converted_list":
        R["progressive"]["converted"] = list(ORDER)
    elif tamper == "missing_step":
        st.pop()
    elif tamper == "probe_modes":
        tag = st[7]["probe_tag"]
        R[f"probe_{tag}"]["modes"] = {"3": "student"}
    elif tamper == "delta_inconsistent":
        st[1]["gate_eval"]["delta_nll"] -= 0.01
    d = D.decide(R, CMP, RULE)
    assert d["verdict"].startswith("no-measurement") and d["gate_audit_problems"] and not D.valid_decision(d), tamper


def test_missing_numbers_and_stops_never_pass():
    d = D.decide(make_R(), None, RULE)  # Index arms missing
    assert d["verdict"].startswith("no-measurement") and not D.valid_decision(d)
    R = make_R()
    R["stopped"] = {"exit_code": 3, "reason": "slow"}
    assert not D.valid_decision(D.decide(R, CMP, RULE))
    d = D.decide(make_R(n_steps=5), CMP, RULE)  # deadline / crash after 5 layers: no verdict
    assert d["verdict"].startswith("no-measurement") and not D.valid_decision(d)
    assert not D.valid_decision({"verdict": "q9af-backbone", "numbers": {}})
    bad = D.decide(make_R(), CMP, RULE)
    bad["attention_free"] = False
    assert not D.valid_decision(bad)


def test_spec_states_the_binding_rule():
    rule_txt = SPEC.split("## Decision rule")[1].split("\n## ")[0]
    for s in ("BINDING", "511e8dd", "caa8f55", "1.1618", "exp(0.15)", "<= 3.0", ">= -1.0", ">= -2.5", "<= 0.10",
              "<= 0.25", "<= 0.05", "<= 0.15", ">= 0.95", "Q9-H", "not attention-free", "decision-needed",
              "no-measurement", "5cmat"):
        assert s in rule_txt, s
    for v in D.VERDICTS:
        assert v in rule_txt, v


# =============================================================================================================
# Part 2 (after the rule commit be9eed8): pricing, runner, export, and the SHELL-LEVEL CPU dry run of the runner
# =============================================================================================================
import gzip  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from jobs.common import costs  # noqa: E402
from jobs.full8 import preflight as pf  # noqa: E402

RULE_COMMIT = "be9eed8"
P8 = json.loads((ROOT / "jobs/full8/job_params.json").read_text(encoding="utf-8"))
CFG8 = json.loads((ROOT / "configs/full8_h100.json").read_text(encoding="utf-8"))
RUNNER = (ROOT / "jobs/full8/full8_job.sh").read_text(encoding="utf-8")


def test_rule_is_byte_identical_to_the_preregistration_commit():
    try:
        old = subprocess.check_output(["git", "show", f"{RULE_COMMIT}:jobs/full8/rule.json"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL)
        old_dec = subprocess.check_output(["git", "show", f"{RULE_COMMIT}:scripts/decide_full8.py"], cwd=ROOT,
                                          stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001  (no git history in a build context)
        pytest.skip("git history not available")
    assert json.loads(old) == RULE
    assert old_dec.replace(b"\r\n", b"\n") == (ROOT / "scripts/decide_full8.py").read_bytes().replace(b"\r\n", b"\n")
    assert P8["rule_commit"] == RULE_COMMIT and RULE_COMMIT in CFG8["_comment"]
    old_spec = subprocess.check_output(["git", "show", f"{RULE_COMMIT}:jobs/full8/JOB_SPEC.md"], cwd=ROOT).decode()
    rule_sec = lambda s: s.replace("\r\n", "\n").split("## Decision rule")[1].split("\n## ")[0].rstrip()  # noqa: E731
    assert rule_sec(old_spec) == rule_sec(SPEC)  # the spec's rule text is unchanged since the pre-registration


# ----------------------------------------------------------------------------- config = the probe recipe, 8 layers
def test_config_is_the_5cmat_recipe_with_only_max_layers_changed():
    diff = {k for k in set(PROBE) | set(CFG8) if PROBE.get(k) != CFG8.get(k)}
    assert diff == {"_comment", "progressive", "ckpt_dir", "success_rule", "diagnostic_bands_posthoc"}, diff
    assert CFG8["progressive"] == {"max_layers": 8} and PROBE["progressive"] == {"max_layers": 1}
    assert CFG8["ckpt_dir"] == "${FULL8_CK}/gdn"
    assert {k: CFG8["gate"][k] for k in RULE["gate"] if k != "_comment"} == {k: v for k, v in RULE["gate"].items()
                                                                               if k != "_comment"}
    assert CFG8["order"] == "kl" and CFG8["stage2"]["guard"]["max_minutes"] == 60 and CFG8["stage2"]["lr"] == 3e-4
    assert CFG8["data"]["train_max_tokens"] == 40_000_000 and CFG8["stage2"]["train_all_converted"] is True


def test_distillation_data_is_open_text_and_never_dev_or_test():
    from convert import data

    assert {k: v[0] for k, v in data.SPLITS.items()} == {"train": "wikitext-103-raw-v1", "gate": "wikitext-103-raw-v1",
                                                          "eval": "wikitext-2-raw-v1"}
    blob = json.dumps(CFG8).lower()
    assert "dev-rows" not in blob and "slices" not in blob and "test-manifest" not in blob
    assert "--slice test" not in RUNNER and "SSDM_UNSEAL_TEST" not in RUNNER and "--final" not in RUNNER
    assert "jobs/common/dev_subset_m1.json" in RUNNER


def test_ckpt_dir_expands_env_and_refuses_unset():
    src = (ROOT / "convert/pipeline.py").read_text(encoding="utf-8")
    assert 'os.path.expandvars(cfg.get("ckpt_dir"' in src and "unset environment variable" in src


# ----------------------------------------------------------------------------- pricing from the 5cmat receipts
def test_measured_block_is_the_job_5cmat_receipt():
    m = P8["measured"]
    assert (m["source_job"], m["artifact"]) == ("job-5cmat", "art-2bybh")
    assert m["artifact_sha256"].startswith("dd491ade")
    assert (m["stage1_s"], m["stage2_s"], m["progressive_one_layer_s"], m["importance_s"]) == (818.89, 3602.69, 3677.6,
                                                                                              61.36)
    assert (m["I_stock_min"], m["I_conv_min"], m["P_min"]) == (10.01, 9.01, 79.2)
    assert (m["stage2_steady_tok_s_layer3"], m["stage2_mfu_layer3"]) == (5461, 0.341)  # job-4j3vu, layer 3


def test_preflight_margins_hold_and_every_cap_is_at_least_1p3x():
    res = pf.check(P8, CFG8)
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    for ph, proj in res["phases"].items():
        usable = P8["phase_caps_minutes"][ph] * (0.95 if ph.startswith("I_") else 1)
        assert usable >= 1.3 * proj, ph
        assert res["phases_measured_basis"][ph] <= P8["expected_minutes"][ph] < P8["phase_caps_minutes"][ph], ph
    # the model reproduces 5cmat's own one-layer P (79.2 min) within 3 minutes
    one = res["P_parts"]["pre_progressive"] + res["P_parts"]["per_layer"] + res["P_parts"]["throughput"] / 2
    assert abs(one - P8["measured"]["P_min"]) < 3, one


@pytest.mark.parametrize("mutate", ["tight_P", "deadline_late", "deadline_early", "high_floor", "schedule", "layers"])
def test_preflight_refuses_a_too_tight_cap_or_floor(mutate):
    p, c = copy.deepcopy(P8), copy.deepcopy(CFG8)
    if mutate == "tight_P":
        p["phase_caps_minutes"]["P"] = 600
        p["max_duration_minutes"] -= 70
    elif mutate == "deadline_late":
        p["pipeline_deadline_minutes"] = 600
    elif mutate == "deadline_early":
        p["pipeline_deadline_minutes"] = 500
    elif mutate == "high_floor":
        c["stage2"]["guard"]["min_tok_s"] = 4000
    elif mutate == "schedule":
        p["reserve_minutes"] += 1
    else:
        c["progressive"]["max_layers"] = 1
    assert not pf.check(p, c)["ok"]
    assert pf.main(["--params", str(_dump(p)), "--config", str(_dump(c))]) == 5


def _dump(obj, _n=[0]):  # noqa: B006
    import tempfile

    _n[0] += 1
    path = Path(tempfile.gettempdir()) / f"full8_pf_{os.getpid()}_{_n[0]}.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_stage2_dominates_and_is_wall_clock_capped_per_layer():
    res = pf.check(P8, CFG8)
    parts = res["P_parts"]
    assert parts["per_layer_stage2_cap"] == 60 and parts["stage2_total"] / res["phases"]["P"] > 0.9
    toks = res["stage2_tokens_5cmat_order"]
    assert toks[0] > toks[-1] and abs(toks[0] - 33.9e6) < 0.2e6 and abs(toks[-1] - 19.2e6) < 0.2e6


def test_cost_line_costs_convention():
    w = costs.worst_case_usd(P8)
    assert costs.attempts(P8) == 2 and P8["build_timeout_minutes"] == 60 and P8["resume"] == "checkpoint"
    assert w == round((60 + 2 * 820) * 0.0495, 2) == 84.15 <= P8["hard_cap_usd"] == 85.0
    assert costs.single_attempt_usd(P8) == 40.59
    assert costs.expected_usd(P8) == round(557 * 0.0495, 2) == 27.57
    assert "full8" in costs.PARAMS and costs.cap_ok(P8) and costs.schedule_ok(P8)
    spec = (ROOT / "jobs/full8/JOB_SPEC.md").read_text(encoding="utf-8")
    for s in ("$84.15", "$40.59", "$27.57", "MTD + 85"):
        assert s in spec, s


def test_full8_maps_every_verdict_to_next_jobs():
    assert set(costs.FULL8_TO_NEXT) == set(D.VERDICTS)
    assert costs.next_jobs_after_full8("q9af-backbone")[0].startswith("pin")
    assert costs.next_jobs_after_full8("partial-conversion")[0] == "m2 on rwkv7"
    assert costs.next_jobs_after_full8("q9af-backbone", 2)[0].startswith("resubmit")
    assert costs.next_jobs_after_full8(None, 9)[0].startswith("diagnose")


# ----------------------------------------------------------------------------- runner static checks
def test_runner_static_contract():
    assert "jobs/common/finalize.py" in RUNNER and "FULL8_DONE" in RUNNER and not re.search(r"touch [^\n]*DONE", RUNNER)
    assert not re.search(r"timeout \d+m", RUNNER)  # every timeout from job_params via `cap`
    assert "python -m jobs.full8.preflight" in RUNNER and "scripts/decide_full8.py" in RUNNER
    assert "scripts.decide_full8:valid_decision" in RUNNER and "sha256sum -c" in RUNNER and "watchdog" in RUNNER
    assert 'CK=${GMN_CHECKPOINT_DIR:-' in RUNNER and '--out "$CK/pipeline"' in RUNNER and "PHASE_HOOK=" in RUNNER
    assert "rate_guard preflight" in RUNNER and "rate_guard run" in RUNNER
    for ph in ("setup", "P", "X", "I_stock", "I_conv", "S", "final"):
        assert re.search(rf"export_phase {ph}\b", RUNNER), ph
    hook = (ROOT / "jobs/full8/phase_hook.sh").read_text(encoding="utf-8")
    assert "/v1/checkpoint" in hook and "progressive" in hook and "stage1" in hook
    spec = (ROOT / "jobs/full8/JOB_SPEC.md").read_text(encoding="utf-8")
    table = spec.split("## Exit codes")[1].split("\n## ")[0]
    for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15):
        assert re.search(rf"(^|[ |]){c} ", table, re.M), c
        assert re.search(rf"\b{c}\b", RUNNER.split("set -Euo")[0]), c


def test_runner_writes_only_under_watched_dirs():
    from tests.test_m1_next_jobs import WATCHED_BASE, _watch_model

    alias, extra, writes, resolve = _watch_model(RUNNER)
    watched = WATCHED_BASE | extra
    assert writes and not sorted(v for v in writes if resolve(v, watched) not in watched)


def test_dry_run_seam_is_refused_in_the_image_and_never_set_by_the_submit_fields():
    assert 'if [ -d /opt/ssdm/m1 ] || [ -z "${FULL8_DRYRUN_ROOT:-}" ]; then echo "FULL8_DRYRUN refused here"; exit 5; fi' \
        in RUNNER
    spec = (ROOT / "jobs/full8/JOB_SPEC.md").read_text(encoding="utf-8")
    sub = json.loads(spec.split("```json")[1].split("```")[0])
    assert not any(k.startswith("FULL8") for k in sub.get("env", {}))
    assert sub["resume"] == "checkpoint" and sub["build_timeout_minutes"] == 60 and sub["max_restarts"] == 1
    assert sub["max_duration_minutes"] == P8["max_duration_minutes"] and sub["model"] == P8["model"]
    for f in ("stubs.py", "pipeline_tiny.py", "kit.py"):
        assert "/opt/ssdm/m1" in (ROOT / "tests/full8_dryrun" / f).read_text(encoding="utf-8")


def test_build_context_lists_the_full8_files():
    common = (ROOT / "jobs/common/build_context.sh").read_text(encoding="utf-8")
    for f in ("jobs/full8/full8_job.sh", "jobs/full8/rule.json", "scripts/decide_full8.py", "jobs/full8/export_gdn.py",
              "configs/full8_h100.json", "jobs/full8/phase_hook.sh"):
        assert f in common, f
    assert "full8" in (ROOT / "jobs/full8/build_context.sh").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- the SHELL-LEVEL dry run
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


def _dry_root(tmp_path):
    from jobs.common import dev_subset

    root = tmp_path / "root"
    (root / "data" / "slices").mkdir(parents=True)
    ids = dev_subset.load_subset()["run_ids"]
    rows = root / "data" / "slices" / "dev-rows.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8") as f:
        for i in ids + ["not-in-subset-1", "not-in-subset-2"]:
            f.write(json.dumps({"_evaluation": {"run_id": i}}) + "\n")
    sha = hashlib.sha256(rows.read_bytes()).hexdigest()
    (root / "data.sha256").write_text(f"{sha}  slices/dev-rows.jsonl.gz\n", encoding="utf-8", newline="\n")
    (root / "CONTEXT_MANIFEST.json").write_text(json.dumps({"dry_run": True}))
    shim = root / "shim"
    shim.mkdir()
    (shim / "curl").write_text('#!/usr/bin/env bash\necho "$*" >> "$FULL8_DRYRUN_ROOT/curl.log"\n'
                               'case "$*" in *http_code*) printf 200 ;; esac\nexit 0\n', newline="\n")
    (shim / "curl").chmod(0o755)
    return root


def _to_posix(p):
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if re.match(r"^[A-Za-z]:/", s) else s


def run_dry(root, out_name, env_extra=None, timeout=900):
    out = root / out_name
    out.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GMN_", "FULL8_", "PHASE_HOOK"))}
    path = [_to_posix(root / "shim"), _to_posix(os.path.dirname(sys.executable))] if os.name == "nt" else \
        [str(root / "shim"), os.path.dirname(sys.executable)]
    env.update({"FULL8_DRYRUN": "1", "FULL8_DRYRUN_ROOT": _to_posix(root), "GMN_OUTPUT_DIR": _to_posix(out),
                "GMN_RESULT_PATH": _to_posix(out / "result.json"), "GMN_CHECKPOINT_DIR": _to_posix(root / "ck"),
                "GMN_METADATA_URL": "http://metadata.invalid", "CUDA_VISIBLE_DEVICES": "",
                "FULL8_DRYRUN_PATH_PREFIX": ":".join(path)})
    if os.name == "nt":  # Windows only: a `python` child of a Python process (rate_guard's Popen) resolves to the base
        # interpreter of a venv, not the venv; hand it the test interpreter's site-packages (same CPython 3.12)
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p.endswith("site-packages")]
                                            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(env_extra or {})
    shutil.rmtree(root / "work", ignore_errors=True)  # a restarted attempt gets a fresh container disk; CK survives
    cmd = [BASH, "-c", 'export PATH="$FULL8_DRYRUN_PATH_PREFIX:$PATH"; exec bash jobs/full8/full8_job.sh']
    with open(root / f"{out_name}.console.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    return rc, out


def _stage2_starts(log_path):
    """Layers whose stage 2 started in this attempt (the pipeline's step-0 line of each stage-2 run)."""
    out = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{") and '"phase": "stage2:L' in line and '"step": 0,' in line:
            out.append(int(re.search(r"stage2:L(\d+)", line).group(1)))
    return out


def _gdn(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _q9af(sd, sha=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Mq

    from heads.backbone import q9af_from_stock
    from tests.full8_dryrun.tiny import text_config

    return q9af_from_stock(Mq.Qwen3_5TextModel(text_config()), sd, {"kind": "q9af", "converted_sha256": sha})


@needs_bash
def test_shell_dry_run_end_to_end_then_resume_after_layer_k(tmp_path):
    """Executes jobs/full8/full8_job.sh itself (bash):
    A  uninterrupted, every layer passes -> q9af-backbone, FULL8_DONE, gdn_converted.pt (8 layers) that the m2
       consumer (q9af_from_stock) and the converted Index arm (the real readout plugin) both load;
    B  hard-killed (os._exit 137: a preemption, no cleanup) right after layer k = 3 is persisted -> rc 2, the 3
       finished layers salvaged to /output, 3 checkpoint POSTs; then the restarted attempt (fresh /output and work
       disk, the SAME checkpoint dir) resumes: no stage 1, layers 1..3 not re-trained, final weights == A's."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rootA = _dry_root(tmp_path / "A")
    rc, outA = run_dry(rootA, "out")
    log = (rootA / "out.console.log").read_text(encoding="utf-8", errors="replace")
    assert rc == 0, log[-4000:]
    res = json.loads((outA / "result.json").read_text())
    assert res["ok"] and res["verdict"] == "q9af-backbone" and (outA / "FULL8_DONE").exists()
    dec = json.loads((outA / "decision.json").read_text())
    assert dec["n_converted"] == 8 and dec["attention_free"] and not dec["gate_audit_problems"]
    sdA = _gdn(outA / "gdn_converted.pt")
    assert sorted(sdA) == RULE["attention_layers"]
    sha = (outA / "gdn_converted.pt.sha256").read_text().split()[0]
    assert sha == hashlib.sha256((outA / "gdn_converted.pt").read_bytes()).hexdigest()
    ex = json.loads((rootA / "work/out/export.json").read_text())
    assert ex["sha256"] == sha and ex["attention_free"] and set(ex["hf_strict_load"].values()) == {"full_attention"}
    text = _q9af(sdA, sha)
    assert set(text.config.layer_types) == {"linear_attention"}
    # m2 consumer path (coordinator, #87): pin check on the exported sha256, then the cache-fork readout the heads use
    # on the converted backbone equals recompute; the export records the layer map and the per-sequence GDN state
    from heads.backbone import Q9AF_BASE_REPO, Q9AF_BASE_REVISION, HiddenStateBackbone, q9af_pin_check

    prov = q9af_pin_check({"converted_sha256": sha, "base_config_sha256": "0" * 64, "base_repo": Q9AF_BASE_REPO,
                           "base_revision": Q9AF_BASE_REVISION}, outA / "gdn_converted.pt")
    assert prov["converted_sha256"] == sha
    bb = HiddenStateBackbone(text, fork_mode="cache")
    prefix, conts = list(range(5, 29)), [[30, 31], [40], [50, 51, 52]]
    _, fork = bb.fork_features(prefix, conts)
    bb.fork_mode = "recompute"
    rec = torch.stack([bb.features(torch.tensor([prefix + c]))[0, -1] for c in conts])
    assert torch.allclose(fork, rec, atol=1e-5)
    assert ex["layer_types_after"] == ["linear_attention"] * 32 and ex["attention_layers_kept"] == []
    assert ex["blocks"]["n"] == len(text.layers) == 32 and all(hasattr(b, ex["blocks"]["mixer_attr"]) for b in text.layers)
    from tests.full8_dryrun.tiny import tiny_convertible

    gdn = tiny_convertible(0).lm.layers[0].linear_attn.gdn  # the GDN class the converted layers share
    _, (tail, S) = gdn.step(torch.randn(1, 5, 32))
    assert list(S.shape[1:]) == ex["gdn_state_shapes_per_sequence"]["recurrent_S"]
    assert list(tail.shape[1:]) == ex["gdn_state_shapes_per_sequence"]["conv_tail"]
    st = json.loads((rootA / "work/out/index/conv/status.json").read_text())
    assert st["notes"]["converted_layers"] == RULE["attention_layers"] and st["files"]["gdn_ckpt"] == sha
    assert json.loads((rootA / "work/out/index/stock/status.json").read_text())["notes"]["converted_layers"] == []
    for ph in ("setup", "P", "X", "I_stock", "I_conv", "S", "final"):
        assert (outA / "phases" / f"{ph}.tgz").exists(), ph
    assert all(json.loads(x)["ok"] for x in (outA / "MANIFEST.jsonl").read_text().splitlines())
    cks = [json.loads(x)["key"] for x in (outA / "checkpoints.jsonl").read_text().splitlines()]
    assert cks.count("progressive") == 8 and "stage1" in cks
    assert (rootA / "curl.log").read_text().count("/v1/checkpoint") == len(cks)
    RA = json.loads((rootA / "ck/pipeline/results.json").read_text())
    assert RA["throughput"]["converted_8"]["call_latency"]["small_call_s"] > 0  # the 5cmat dtype failure is fixed

    # ---- B: preemption after layer 3, then the restarted attempt
    rootB = _dry_root(tmp_path / "B")
    rc1, out1 = run_dry(rootB, "out1", {"FULL8_DRYRUN_CRASH_AFTER_STEP": "3"})
    assert rc1 == 2, (rootB / "out1.console.log").read_text(errors="replace")[-3000:]
    assert not (out1 / "FULL8_DONE").exists()
    assert json.loads((out1 / "result.json").read_text())["exit_code"] == 2
    R1 = json.loads((rootB / "ck/pipeline/results.json").read_text())
    assert len(R1["progressive"]["steps"]) == 3
    salv = _gdn(out1 / "gdn_salvage_step3.pt")
    assert sorted(salv) == sorted(R1["progressive"]["converted"]) and len(salv) == 3
    keys1 = [json.loads(x)["key"] for x in (out1 / "checkpoints.jsonl").read_text().splitlines()]
    assert keys1.count("progressive") == 3 and "stage1" in keys1
    rc2, out2 = run_dry(rootB, "out2")
    assert rc2 == 0, (rootB / "out2.console.log").read_text(errors="replace")[-4000:]
    plog2 = (rootB / "work/logs/pipeline.log").read_text(errors="replace")
    assert '"phase": "stage1"' not in plog2  # stage 1 not re-run
    assert _stage2_starts(rootB / "work/logs/pipeline.log") == R1["importance"]["order"][3:]  # only layers 4..8
    R2 = json.loads((rootB / "ck/pipeline/results.json").read_text())
    assert R2["progressive"]["steps"][:3] == R1["progressive"]["steps"]
    assert json.loads((out2 / "result.json").read_text())["verdict"] == "q9af-backbone" and (out2 / "FULL8_DONE").exists()
    sdB = _gdn(out2 / "gdn_converted.pt")
    assert sorted(sdB) == sorted(sdA)
    for i in sdA:  # resumed == uninterrupted, exactly
        assert all(torch.equal(sdA[i][k], sdB[i][k]) for k in sdA[i]), i


@needs_bash
def test_shell_dry_run_partial_conversion_exports_a_hybrid_not_q9af(tmp_path):
    pytest.importorskip("torch")
    from heads.backbone import LineageError

    root = _dry_root(tmp_path)
    rc, out = run_dry(root, "out", {"FULL8_DRYRUN_GATE": "reject:27,3"})
    assert rc == 0, (root / "out.console.log").read_text(errors="replace")[-4000:]
    dec = json.loads((out / "decision.json").read_text())
    assert dec["verdict"] == "partial-conversion" and dec["hybrid_name"] == "Q9-H2" and not dec["attention_free"]
    assert dec["attention_layers_kept"] == [3, 27] and not dec["gate_audit_problems"]
    assert (out / "gdn_partial.pt").exists() and not (out / "gdn_converted.pt").exists()
    with pytest.raises(LineageError):
        _q9af(_gdn(out / "gdn_partial.pt"))

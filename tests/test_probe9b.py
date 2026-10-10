"""CPU tests for jobs/probe9b (the $40 one-layer Qwen3.5-9B GDN conversion probe, issue #68): params / cost / spec
consistency at the MEASURED job-4j3vu rates, the >= 30% margin rule, the binding 511e8dd decision rule (post-hoc bands reported only), the
contamination boundary, and a CPU dry-run of the real convert.pipeline logic (micro-batch, KL importance, one layer,
enforced gate, teacher cache, throughput + call latency) on a tiny random Qwen3.5 stand-in. No GPU, no weights.

    python -m pytest -q tests/test_probe9b.py
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import pytest

from jobs.common import costs
from jobs.probe9b import preflight as pf
from scripts import decide_probe9b as D

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "jobs/probe9b/job_params.json").read_text(encoding="utf-8"))
FULL = json.loads((ROOT / "configs/probe9b_h100.json").read_text(encoding="utf-8"))
LEAN = json.loads((ROOT / "configs/probe9b_h100_lean.json").read_text(encoding="utf-8"))
SPEC = (ROOT / "jobs/probe9b/JOB_SPEC.md").read_text(encoding="utf-8")
RUNNER = (ROOT / "jobs/probe9b/probe_job.sh").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- params, cost, spec
def test_cost_line_costs_convention_and_hard_cap():
    w = costs.worst_case_usd(P)
    assert costs.attempts(P) == 2  # max_restarts 0, but >= 1 platform retry assumed
    assert w == round((P["build_timeout_minutes"] + 2 * P["max_duration_minutes"]) * P["batch_usd_per_min"], 2) == 35.91
    assert w <= P["hard_cap_usd"] == 40.0
    assert costs.single_attempt_usd(P) == 17.01  # $0.063/min from 2026-10-08 (was 13.37 / 6.58 at $0.0495)
    assert costs.expected_usd(P) == round(sum(P["expected_minutes"].values()) * 0.063, 2) == 8.38
    for s in ("$35.91", "$17.01", "$8.38", "MTD + 40"):
        assert s in SPEC, s


def test_phase_caps_sum_to_max_duration():
    assert sum(P["phase_caps_minutes"].values()) + P["reserve_minutes"] == P["max_duration_minutes"]
    assert costs.schedule_ok(P) and costs.cap_ok(P)
    assert set(P["expected_minutes"]) == set(P["phase_caps_minutes"]) | {"build"}
    assert P["expected_minutes"]["build"] <= P["build_timeout_minutes"]


def test_measured_block_is_the_job_4j3vu_receipt():
    m = P["measured"]
    assert (m["source_job"], m["artifact"]) == ("job-4j3vu", "art-5ys6j")
    assert (m["stage2_steady_tok_s"], m["stage2_mfu"], m["peak_gb"], m["stage1_steady_tok_s"]) == (5461, 0.341, 31.9, 18952)
    assert "6e758dbe3c47a507e25623d3794edbf6fe95c46908fd3c8112fe25612a9f4e59" in m["_comment"]
    # the qual decision rule that opened this spend: MFU >= 0.25 and peak <= 60 GB
    assert m["stage2_mfu"] >= 0.25 and m["peak_gb"] <= 60


def test_preflight_margins_hold_for_every_phase():
    res = pf.check(P, FULL)
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    for ph, proj in res["phases"].items():
        assert P["phase_caps_minutes"][ph] >= 1.3 * proj, ph
        assert proj <= P["expected_minutes"][ph] < P["phase_caps_minutes"][ph], ph  # expected is never optimistic
    assert res["checks"]["P_deadline"]["ok"] and P["pipeline_deadline_minutes"] < P["phase_caps_minutes"]["P"]
    assert pf.check(P, LEAN)["ok"]


def test_stage2_dominates_and_its_token_budget_comes_from_the_measured_rate():
    res = pf.check(P, FULL)
    parts = res["P_parts"]
    total = sum(res["phases"].values())
    assert parts["stage2"] == max(parts.values()) and parts["stage2"] / total > 0.4
    assert res["stage2_tokens_at_measured_rate"] == 5461 * 60 * 60  # ~19.7M distillation tokens at the worst layer
    assert res["stage1_tokens"] == 400 * 8 * 4096


@pytest.mark.parametrize("mutate", ["tight_P", "tight_I", "high_floor", "deadline", "schedule"])
def test_preflight_refuses_a_too_tight_cap_or_floor(mutate):
    p, c = copy.deepcopy(P), copy.deepcopy(FULL)
    if mutate == "tight_P":
        p["phase_caps_minutes"]["P"] = 95
    elif mutate == "tight_I":
        p["phase_caps_minutes"]["I_conv"] = 14
    elif mutate == "high_floor":
        c["stage2"]["guard"]["min_tok_s"] = int(0.8 * P["measured"]["stage2_steady_tok_s"])
    elif mutate == "deadline":
        p["pipeline_deadline_minutes"] = 90
    else:
        p["reserve_minutes"] += 1
    assert not pf.check(p, c)["ok"]
    assert pf.main(["--params", str(_dump(p)), "--config", str(_dump(c))]) == 5


def _dump(obj, _n=[0]):  # noqa: B006
    import tempfile

    _n[0] += 1
    path = Path(tempfile.gettempdir()) / f"probe9b_pf_{_n[0]}.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_config_carries_the_measured_memory_fix_and_floors():
    for c in (FULL, LEAN):
        assert c["micro_batch"] == 4 and c["batch"] % c["micro_batch"] == 0
        assert c["seq_len"] == 4096 and c["micro_batch"] * c["seq_len"] == 16384  # the job-4j3vu micro-batch shape
        assert c["gate"]["n_windows"] == 32
    m = P["measured"]
    assert FULL["stage1"]["guard"]["min_tok_s"] == m["stage1_steady_tok_s"] // 2
    assert FULL["stage2"]["guard"]["min_tok_s"] == m["stage2_steady_tok_s"] // 2


# ----------------------------------------------------------------------------- the BINDING rule = 511e8dd, unchanged
#: success_rule as pre-registered at commit 511e8dd (before any 9B data); the verdict is computed from this only
RULE_511E8DD = {"gate_must_pass": True, "ppl_ratio_max": {"ctx1024": 1.02, "ctx4096": 1.02}, "mmlu_drop_max_pts": 3.0,
                "mqar_drop_max": 0.1, "niah_mean_drop_max": 0.1, "niah_length_drop_max": 0.25,
                "index_diff_point_min": -1.0, "index_diff_lo_min": -2.5}
GATE_511E8DD = {"max_step_delta_nll": 0.05, "max_cum_delta_nll": 0.15, "min_hidden_cos": 0.95, "policy": "enforce"}
DIAG = FULL["diagnostic_bands_posthoc"]


def test_binding_rule_is_the_511e8dd_rule_unchanged():
    for c in (FULL, LEAN):
        assert {k: v for k, v in c["success_rule"].items() if k not in ("_comment", "verdicts")} == RULE_511E8DD
        assert set(c["success_rule"]["verdicts"]) == {"proceed-all-8", "retrieval-fail", "quality-fail"}
        # the enforced gate is the 511e8dd one except n_windows (16 -> 32: a post-data, non-directional precision change,
        # disclosed in JOB_SPEC.md)
        assert {k: v for k, v in c["gate"].items() if k != "n_windows"} == GATE_511E8DD and c["gate"]["n_windows"] == 32
    assert D.VERDICTS == ("proceed-all-8", "retrieval-fail", "quality-fail")
    assert set(costs.PROBE_TO_NEXT) == set(D.VERDICTS)
    import subprocess

    try:
        old = json.loads(subprocess.check_output(["git", "show", "511e8dd:configs/probe9b_h100.json"], cwd=ROOT,
                                                 stderr=subprocess.DEVNULL))
    except Exception:  # noqa: BLE001  (no git history in a build context)
        pytest.skip("git history not available")
    assert FULL["success_rule"] == old["success_rule"]
    assert {k: v for k, v in old["gate"].items() if k != "n_windows"} == GATE_511E8DD and old["gate"]["n_windows"] == 16


def _R(kl=0.02, cos=0.98, dnll=0.01, k0=0.03, gate_pass=True):
    stock = {"ppl": {"ctx1024": {"ppl": 10.0}, "ctx4096": {"ppl": 8.0}}, "mmlu": {"acc": 0.70},
             "mqar": {"mean_acc": 0.90}, "niah": {"L1024": {"acc": 1.0}, "L16384": {"acc": 0.9}, "mean_acc": 0.95}}
    step = {"layer": 27, "gate_pass": gate_pass, "accepted": gate_pass, "step_delta_nll": dnll,
            "gate_eval": {"delta_nll": dnll, "hidden_cos": cos, "kl": kl, "nll_student": 2.0}}
    R = {"progressive": {"steps": [step], "converted": [27] if gate_pass else []}, "probe_stock": stock,
         "importance": {"order": [27, 3, 31, 7, 11, 15, 19, 23], "per_layer": {"27": {"kl": k0}}}}
    if gate_pass:
        step["probe_tag"] = "conv1_kept7"
        R["probe_conv1_kept7"] = json.loads(json.dumps(stock))
    return R


CMP = {"diff": {"index": {"point": -0.4, "lo": -1.6, "hi": 0.8}}}


def _cases():
    """(name, R, cmp, binding verdict under 511e8dd)."""
    out = [("all ok", _R(), CMP, "proceed-all-8"),
           # diagnostic KL AMBER (0.06 > 0.05) and RED (0.2): the binding rule has no KL bound -> still proceed
           ("diag KL amber", _R(kl=0.06), CMP, "proceed-all-8"),
           ("diag KL red", _R(kl=0.2), CMP, "proceed-all-8"),
           ("qual-like layer 3 numbers", _R(kl=0.054, cos=0.980, dnll=-0.0005), CMP, "proceed-all-8"),
           ("gate rejected", _R(gate_pass=False, dnll=0.09), None, "quality-fail")]
    for what, val in (("ppl", 1.03), ("mmlu", 0.66), ("index_point", -1.2), ("index_lo", -2.6)):
        R, cmp = _R(), json.loads(json.dumps(CMP))  # diagnostic AMBER, binding quality-fail
        if what == "ppl":
            R["probe_conv1_kept7"]["ppl"]["ctx4096"]["ppl"] = 8.0 * val
        elif what == "mmlu":
            R["probe_conv1_kept7"]["mmlu"]["acc"] = val
        elif what == "index_point":
            cmp["diff"]["index"]["point"] = val
        else:
            cmp["diff"]["index"]["lo"] = val
        out.append((what, R, cmp, "quality-fail"))
    R = _R()
    R["probe_conv1_kept7"]["niah"] = {"L1024": {"acc": 1.0}, "L16384": {"acc": 0.5}, "mean_acc": 0.75}
    out.append(("niah", R, CMP, "retrieval-fail"))
    return out


@pytest.mark.parametrize("name,R,cmp,verdict", _cases(), ids=[c[0] for c in _cases()])
def test_verdict_comes_from_the_511e8dd_rule_only(name, R, cmp, verdict):
    d = D.decide(R, cmp, FULL)
    assert d["verdict"] == verdict and D.valid_decision(d), (name, d["verdict"])
    # identical with the diagnostic block removed, or with absurd diagnostic bands
    bare = {k: v for k, v in FULL.items() if k != "diagnostic_bands_posthoc"}
    assert D.decide(R, cmp, bare)["verdict"] == verdict and "diagnostic_bands_posthoc" not in D.decide(R, cmp, bare)
    wild = copy.deepcopy(FULL)
    for row in wild["diagnostic_bands_posthoc"]["bands"].values():
        if isinstance(row, dict):
            row.update({k: (-1e9 if "max" in k else 1e9) for k in row})  # everything RED
    assert D.decide(R, cmp, wild)["verdict"] == verdict
    if d["gate"]["pass"] is not False:
        assert d["diagnostic_bands_posthoc"]["binding"] is False


def test_posthoc_diagnostics_are_reported_and_labelled():
    d = D.decide(_R(kl=0.054, cos=0.980, dnll=-0.0005, k0=0.055), CMP, FULL)
    g = d["diagnostic_bands_posthoc"]
    assert g["note"] == "written after seeing job-4j3vu qual gate numbers; not used for the verdict"
    assert g["bands"]["gate_kl"] == "amber" and g["bands"]["gate_hidden_cos"] == "green"
    assert d["verdict"] == "proceed-all-8" and g["binding"] is False
    assert "adjust_hint" not in DIAG and "suggested_adjustment_if_not_proceeding" not in g  # no re-probe path
    assert "max_reprobes" not in json.dumps(FULL) + json.dumps(LEAN)
    learning = D.decide(_R(kl=0.06, k0=0.10), CMP, FULL)["diagnostic_bands_posthoc"]
    assert abs(learning["stage2_kl_gain_frac"] - 0.4) < 1e-9


@pytest.mark.parametrize("name,R,cmp,verdict", _cases(), ids=[c[0] for c in _cases()])
def test_a_failing_diagnostic_never_changes_verdict_or_exit(name, R, cmp, verdict, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("diagnostic exploded")

    monkeypatch.setattr(D, "diagnostics", boom)
    d = D.decide(R, cmp, FULL)
    assert d["verdict"] == verdict and D.valid_decision(d)
    assert d["diagnostic_bands_posthoc"]["binding"] is False and "diagnostic exploded" in d["diagnostic_bands_posthoc"]["error"]
    broken = copy.deepcopy(FULL)
    broken["diagnostic_bands_posthoc"]["bands"].pop("gate_kl")  # a malformed block: KeyError inside diagnostics
    monkeypatch.undo()
    assert D.decide(R, cmp, broken)["verdict"] == verdict
    if cmp is not None:
        (tmp_path / "r.json").write_text(json.dumps(R))
        (tmp_path / "c.json").write_text(json.dumps(cmp))
        (tmp_path / "cfg.json").write_text(json.dumps(broken))
        rc = D.main(str(tmp_path / "r.json"), str(tmp_path / "c.json"), str(tmp_path / "cfg.json"), str(tmp_path / "d.json"))
        assert rc == 0 and json.loads((tmp_path / "d.json").read_text())["verdict"] == verdict


def test_posthoc_bands_derivation_is_recorded():
    pil, b, g = DIAG["pilot_0p8b"], DIAG["bands"], FULL["gate"]
    assert (pil["delta_nll"], pil["kl"], pil["hidden_cos"]) == (-0.0036, 0.017, 0.991)
    assert "unverified" in b["_comment"] or "not in LEDGER" in pil["hidden_cos_provenance"]
    assert b["gate_delta_nll"]["green_max"] == round(g["max_cum_delta_nll"] / 8, 3) == 0.019
    assert b["gate_kl"]["green_max"] == math.floor(3 * pil["kl"] * 100) / 100 == 0.05
    assert b["gate_hidden_cos"]["green_min"] == math.floor((1 - 3 * (1 - pil["hidden_cos"])) * 100) / 100 == 0.97
    assert "NOT USED FOR THE VERDICT" in b["_comment"] and LEAN["diagnostic_bands_posthoc"] == DIAG


def test_band_edges():
    row = DIAG["bands"]["gate_kl"]
    assert [D.band(x, row) for x in (0.05, 0.0501, 0.15, 0.1501, None)] == ["green", "amber", "amber", "red", None]


def test_missing_numbers_never_pass():
    d = D.decide(_R(), None, FULL)  # Index arms missing
    assert d["verdict"].startswith("no-measurement") and not D.valid_decision(d)


def test_spec_states_the_binding_rule_and_the_posthoc_status():
    for s in ("511e8dd", "<= 1.02", "<= 3.0", ">= -1.0", ">= -2.5", "<= 0.10", "<= 0.25", "not used for the verdict",
              "0.054", "unverified"):
        assert s in SPEC, s
    for v in D.VERDICTS:
        assert v in SPEC
    assert "adjust-recipe" not in (ROOT / "jobs/common/costs.py").read_text(encoding="utf-8")
    assert "noise-limited" in SPEC and "n_windows" in SPEC


# ----------------------------------------------------------------------------- contamination boundary
def test_distillation_data_is_open_text_and_never_dev_or_test():
    from convert import data

    assert {k: v[0] for k, v in data.SPLITS.items()} == {"train": "wikitext-103-raw-v1", "gate": "wikitext-103-raw-v1",
                                                          "eval": "wikitext-2-raw-v1"}
    blob = json.dumps(FULL).lower()
    assert "dev-rows" not in blob and "slices" not in blob and "test-manifest" not in blob
    assert "--slice test" not in RUNNER and "SSDM_UNSEAL_TEST" not in RUNNER and "--final" not in RUNNER
    assert FULL["model_dir"].startswith("${SSDM_DATA_ROOT}/hf/models--Qwen--Qwen3.5-9B")  # teacher = the open 9B itself
    assert P["model"].startswith("Qwen/Qwen3.5-9B@")


# ----------------------------------------------------------------------------- runner / build context
def test_runner_preflight_exports_heartbeat_and_exit_codes():
    assert "python -m jobs.probe9b.preflight" in RUNNER and 'gate "setup:preflight"' in RUNNER
    assert "call_latency" in RUNNER and "small_call_s" in RUNNER  # the 9B's own measured per-call cost
    assert "rate_guard preflight" in RUNNER and "rate_guard run" in RUNNER
    for ph in ("setup", "P", "I_stock", "I_conv", "S", "final"):
        assert re.search(rf"export_phase {ph}\b", RUNNER), ph
    assert "watchdog" in RUNNER and "hb" in RUNNER and not re.search(r"timeout \d+m", RUNNER)
    table = SPEC.split("## Exit codes")[1].split("\n## ")[0]
    for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15):  # every code the runner can return is documented
        assert re.search(rf"(^|[ |]){c} ", table, re.M), c
        assert re.search(rf"\b{c}\b", RUNNER.split("set -Euo")[0]), c


def test_build_context_wrapper_and_manifest():
    w = (ROOT / "jobs/probe9b/build_context.sh").read_text(encoding="utf-8")
    assert "jobs/common/build_context.sh\" probe9b" in w
    common = (ROOT / "jobs/common/build_context.sh").read_text(encoding="utf-8")
    for f in ("jobs/probe9b/preflight.py", "scripts/decide_probe9b.py", "jobs/probe9b/job_params.json"):
        assert f in common


# ----------------------------------------------------------------------------- CPU dry-run on a tiny stand-in
def _tiny():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.test_microbatch import tiny_model

    return tiny_model(seed=0)


def test_teacher_cache_equals_recompute():
    torch = pytest.importorskip("torch")
    from convert import align

    m = _tiny()
    w = torch.randint(0, 101, (6, 24), generator=torch.Generator().manual_seed(3))
    th = align.teacher_hidden(m, w)
    for modes in ({}, {1: "student"}, {1: "student", 3: "student"}):
        a, b = align.evaluate(m, w, modes), align.evaluate(m, w, modes, teacher=th)
        assert all(abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k])) for k in a), (modes, a, b)
    with pytest.raises(ValueError):
        align.evaluate(m, w, {}, teacher=th[:1])


def test_call_latency_runs_on_the_converted_readout_path():
    from probes.throughput import call_latency

    m = _tiny()
    out = call_latency(m, {1: "student"}, prefix_len=64, tail=(4, 8), small_len=4, reps=3)
    assert out["pack_call_s"] > 0 and out["small_call_s"] > 0


def _stub_probes(model, tok, eval_toks, cfg):
    """probes.core.run_all shape, computed from the tiny model in its current modes (real numbers, tiny volume)."""
    import torch
    x = torch.as_tensor(eval_toks[:4 * 24].astype("int64")).view(4, 24)
    with torch.no_grad():
        lp = model.lm_head(model.hidden(x)).float().log_softmax(-1)
    nll = float(-lp[:, :-1].gather(-1, x[:, 1:, None]).mean())
    acc = float((lp[:, :-1].argmax(-1) == x[:, 1:]).float().mean())
    return {"ppl": {"ctx1024": {"ppl": math.exp(nll)}, "ctx4096": {"ppl": math.exp(nll)}},
            "niah": {"L1024": {"acc": acc}, "mean_acc": acc}, "mqar": {"mean_acc": acc}, "mmlu": {"acc": acc}}


def _dry_run(tmp_path, monkeypatch, gate_overrides):
    import numpy as np

    from convert import pipeline

    cfg = copy.deepcopy(FULL)  # the REAL probe config, shrunk only in size
    cfg.update(seq_len=24, batch=4, micro_batch=2, model_dir=None, ckpt_dir=str(tmp_path / "ck"))
    cfg.pop("vram_fraction")
    cfg["data"]["train_max_tokens"] = 4000
    cfg["stage1"].update(steps=3)
    cfg["stage2"].update(steps=3)
    cfg["gate"].update(n_windows=4, **gate_overrides)
    cfg["throughput"] = {"lengths": [24], "reps": 1}
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps(cfg), encoding="utf-8")
    toks = np.random.default_rng(0).integers(0, 101, 6000).astype(np.uint32)
    m = _tiny()
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: (m, None, "tiny"))
    monkeypatch.setattr(pipeline.data, "build", lambda tok, s, max_tokens=None: (s, {"split": s}))
    monkeypatch.setattr(pipeline.data, "load_tokens", lambda p: toks)
    monkeypatch.setattr(pipeline.probes, "run_all", _stub_probes)
    monkeypatch.setattr(pipeline, "measure_throughput",
                        lambda model, modes, lengths, reps=3: {str(L): {"tok_s": 1000.0} for L in lengths})
    monkeypatch.setattr(pipeline, "call_latency",
                        lambda model, modes: __import__("probes.throughput", fromlist=["x"]).call_latency(
                            model, modes, prefix_len=32, tail=(2, 4), small_len=4, reps=2))
    calls = {"teacher": 0}
    real_th = pipeline.align.teacher_hidden

    def counted(*a, **k):
        calls["teacher"] += 1
        return real_th(*a, **k)

    monkeypatch.setattr(pipeline.align, "teacher_hidden", counted)
    seen_mb = []
    real_s2 = pipeline.align.stage2_kl

    def s2(*a, **k):
        seen_mb.append(k.get("micro_batch"))
        return real_s2(*a, **k)

    monkeypatch.setattr(pipeline.align, "stage2_kl", s2)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(cfgp), "--out", str(out)])
    monkeypatch.delenv("DEADLINE_MINUTES", raising=False)
    pipeline.main()
    R = json.loads((out / "results.json").read_text())
    return R, cfg, calls, seen_mb


def test_cpu_dry_run_accepts_one_layer_and_decides(tmp_path, monkeypatch):
    R, cfg, calls, seen_mb = _dry_run(tmp_path, monkeypatch, {"max_step_delta_nll": 1e9, "max_cum_delta_nll": 1e9,
                                                                "min_hidden_cos": -1.0})
    assert R["stage1"]["micro_batch"] == 2 and seen_mb == [2]  # micro-batch reaches both stages
    assert sorted(R["importance"]["order"]) == [1, 3] and len(R["importance"]["per_layer"]) == 2
    st = R["progressive"]["steps"]
    assert len(st) == 1 and st[0]["layer"] == R["importance"]["order"][0]  # least important layer only (max_layers 1)
    assert st[0]["accepted"] and R["progressive"]["converted"] == [st[0]["layer"]] and st[0]["probe_tag"] == "conv1_kept1"
    assert calls["teacher"] == 1  # one teacher pass serves importance + baseline + gate
    assert {"load_and_data", "controls", "stage1", "importance", "progressive", "throughput"} <= set(R["timing"])
    assert R["throughput"]["stock"]["call_latency"]["small_call_s"] > 0
    # the decision rule runs on the dry-run results (Index compare synthetic: the kit is not on CPU here)
    d = D.decide(R, CMP, cfg)
    assert d["verdict"] in D.VERDICTS and D.valid_decision(d), d["verdict"]
    assert d["diagnostic_bands_posthoc"]["stage1_kl"] == R["importance"]["per_layer"][str(st[0]["layer"])]["kl"]
    # resume: a second run on the same out dir re-uses every finished phase (no new training)
    from convert import pipeline

    pipeline.main()
    R2 = json.loads((tmp_path / "out" / "results.json").read_text())
    assert R2["progressive"] == R["progressive"] and R2["stage1"]["hist"] == R["stage1"]["hist"]


def test_cpu_dry_run_rejected_layer_rolls_back_and_folds(tmp_path, monkeypatch):
    R, cfg, _, _ = _dry_run(tmp_path, monkeypatch, {"min_hidden_cos": 1.01})
    st = R["progressive"]["steps"][0]
    assert st["gate_pass"] is False and not st["accepted"] and R["progressive"]["converted"] == []
    d = D.decide(R, None, cfg)
    assert d["verdict"] == "quality-fail" and D.valid_decision(d)
    out = tmp_path / "decision.json"
    assert D.main(str(tmp_path / "out" / "results.json"), "-", str(tmp_path / "cfg.json"), str(out)) == 0

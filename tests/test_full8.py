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

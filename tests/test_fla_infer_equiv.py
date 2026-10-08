"""jobs/fla_rwkv7 mode infer_equiv (jobs.fla_rwkv7.infer_equiv, rule fla-infer-equiv-v1). CPU only, no fla, no weights.

The interleaved scorer runs end to end on a tiny vendored RWKV-7 with the fla kernel replaced by test_fla_rwkv7's
``fake_chunk`` (a token-serial transcription of fla-core 0.5.2's DPLR reference), so the "fla" arm really goes through
heads.rwkv7_wkv's fla branch. The real G1V2Backend needs the G1k tokenizer (not in the repo), so the tiny run uses the
v1 HFReadoutBackend with the prefix fork; the interleaving, scoped override, accounting and rule are what is tested.

    python -X utf8 -m pytest -q tests/test_fla_infer_equiv.py
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from jobs.fla_rwkv7 import infer_equiv as IE

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/fla_rwkv7/fla_job.sh"
PARAMS = ROOT / "jobs/fla_rwkv7/job_params_infer_equiv.json"


# ----------------------------------------------------------------------------- pre-registered constants
def test_rule_constants_are_pinned():
    assert IE.RULE_ID == "fla-infer-equiv-v1" and IE.SCHEMA == "ssdm.fla_rwkv7.infer_equiv.v1"
    assert IE.AGREE_MIN == 0.99 and IE.MARGIN_FLIP == 0.05 and IE.CI_BOUND_MAX == 0.5 and IE.SPEED_ADOPT == 2.0
    assert IE.LENGTH_EDGES == (2048, 8192) and IE.BUCKETS == ("<2k", "2-8k", ">=8k", "unknown")
    assert (IE.REF, IE.CAND) == ("chunked", "fla") and IE.ARMS == ("chunked", "fla")
    assert IE.FAMILY == "rwkv7-hf-g1-v2" and IE.TEMPLATE == "jrt-rwkv-g1-v2"
    assert IE.VERDICTS == ("EQUIVALENT_FOR_INFERENCE", "NOT_EQUIVALENT", "INCONCLUSIVE")
    assert IE.ENV_FLAG == "SSDM_FLA_INFER_EQUIV"
    from jobs.common.readout_engine import TEMPLATES

    assert TEMPLATES[IE.FAMILY] == IE.TEMPLATE  # the production family this mode builds


# ----------------------------------------------------------------------------- metrics
def _arm(status="ok", raw=None, ms=100.0, tok=500, calls=0, fb=0):
    return {"status": status, "wall_ms": ms, "max_call_input_tokens": tok, "raw": raw, "fla_calls": calls,
            "fallbacks": fb}


def _pair(rid, ref_raw, cand_raw, tok=500, ref_ms=300.0, cand_ms=100.0, ref_status="ok", cand_status="ok", nq=None,
          calls=3, fb=0):
    return {"run_id": rid, "n_questions": nq or len(ref_raw or cand_raw or {}),
            "chunked": _arm(ref_status, ref_raw if ref_status == "ok" else None, ref_ms, tok),
            "fla": _arm(cand_status, cand_raw if cand_status == "ok" else None, cand_ms, tok,
                        calls if cand_status == "ok" else 0, fb)}


def test_argmax_margin_bucket_pctl():
    assert IE.argmax({"b": 0.5, "a": 0.5}) == "a"  # tie -> smallest id, as wire_answer
    assert IE.margin({"a": 0.7, "b": 0.2, "c": 0.1}) == pytest.approx(0.5) and IE.margin({"a": 1.0}) == 1.0
    assert [IE.bucket(x) for x in (0, 2047, 2048, 8191, 8192, None)] == ["<2k", "<2k", "2-8k", "2-8k", ">=8k", "unknown"]
    assert IE.pctl([], 0.99) is None and IE.pctl(list(range(1, 101)), 0.99) == 99 and IE.pctl([3.0], 0.99) == 3.0


def test_summarize_pairs_counts_flips_margins_mismatch_and_speed():
    same = {"q": {"a": 0.8, "b": 0.2}}
    pairs = [
        _pair("r0", same, {"q": {"a": 0.79, "b": 0.21}}, tok=100),
        _pair("r1", {"q": {"a": 0.52, "b": 0.48}}, {"q": {"a": 0.49, "b": 0.51}}, tok=3000),  # low-margin flip
        _pair("r2", {"q": {"a": 0.9, "b": 0.1}}, {"q": {"a": 0.1, "b": 0.9}}, tok=9000),  # high-margin flip
        _pair("r3", same, same, ref_status="exhausted", cand_status="exhausted", nq=2),  # both refused: excluded
        _pair("r4", {"q": {"a": 0.6, "b": 0.4}, "p": {"x": 1.0}}, None, cand_status="error", tok=None),  # mismatch
    ]
    s = IE.summarize_pairs(pairs)
    assert s["rows"] == 5 and s["rows_both_ok"] == 3 and s["rows_both_not_ok"] == 1 and s["rows_status_mismatch"] == 1
    assert s["questions"] == 5 and s["agree"] == 1 and s["agreement"] == pytest.approx(0.2)
    assert s["flips"] == 4 and s["high_margin_flips"] == 3  # r2 + the two mismatched questions of r4
    assert s["by_length"]["2-8k"] == {"questions": 1, "flips": 1, "high_margin_flips": 0}
    assert s["by_length"][">=8k"] == {"questions": 1, "flips": 1, "high_margin_flips": 1}
    assert s["by_length"]["unknown"]["high_margin_flips"] == 2
    assert s["max_abs_dp"] == pytest.approx(0.8) and s["max_abs_dp_at"] == {"dp": pytest.approx(0.8), "run_id": "r2",
                                                                              "qid": "q"}
    assert s["finite"] and s["fla_fallbacks"] == 0 and s["fla_ok_rows_without_calls"] == 0
    assert s["speed_ratio"] == pytest.approx(3.0)  # rows/min over the both-ok rows: 300 ms vs 100 ms per row


def test_summarize_pairs_nonfinite_fallbacks_and_missing_fla_calls():
    s = IE.summarize_pairs([_pair("r0", {"q": {"a": float("nan"), "b": 0.5}}, {"q": {"a": 0.5, "b": 0.5}})])
    assert s["finite"] is False
    s = IE.summarize_pairs([_pair("r0", {"q": {"a": 1.0}}, {"q": {"a": 1.0}}, fb=2)])
    assert s["fla_fallbacks"] == 2
    s = IE.summarize_pairs([_pair("r0", {"q": {"a": 1.0}}, {"q": {"a": 1.0}}, calls=0)])
    assert s["fla_ok_rows_without_calls"] == 1 and s["fla_calls"] == 0


# ----------------------------------------------------------------------------- the rule
def _summary(n=1000, agree=1000, hi=0, finite=True, fb=0, ratio=2.5):
    return {"questions": n, "agreement": agree / n, "high_margin_flips": hi, "finite": finite, "fla_fallbacks": fb,
            "fla_ok_rows_without_calls": 0, "fla_calls": 5000, "speed_ratio": ratio,
            "rows_per_min": {"chunked": 100.0, "fla": 100.0 * ratio}}


def _compare(lo=-0.2, hi=0.3, pt=0.05, se=0.1):
    return {"diff": {"index": {"point": pt, "lo": lo, "hi": hi, "se": se}},
            "candidate": {"index": {"point": 27.4}}, "baseline": {"index": {"point": 27.3}}}


META = {"complete": True, "rows_total": 2008, "rows_done": 2008}


def test_decide_every_verdict():
    d = IE.decide(_summary(), _compare(), META)
    assert d["verdict"] == "EQUIVALENT_FOR_INFERENCE" and d["adopt_for_inference"] is True and IE.valid_decision(d)
    d = IE.decide(_summary(ratio=1.9), _compare(), META)
    assert d["verdict"] == "EQUIVALENT_FOR_INFERENCE" and d["adopt_for_inference"] is False  # speed is not a verdict
    for kw, cmp_kw in (({"agree": 989}, {}), ({"hi": 1}, {}), ({"finite": False}, {}), ({"fb": 1}, {}),
                       ({}, {"lo": 0.01, "hi": 0.4}), ({}, {"lo": -0.6, "hi": 0.1}), ({}, {"lo": -0.1, "hi": 0.51}),
                       ({}, {"se": float("nan")})):
        d = IE.decide(_summary(**kw), _compare(**cmp_kw), META)
        assert d["verdict"] == "NOT_EQUIVALENT" and not d["adopt_for_inference"] and IE.valid_decision(d), (kw, cmp_kw)
    assert IE.decide(_summary(agree=990), _compare(), META)["verdict"] == "EQUIVALENT_FOR_INFERENCE"  # >= 0.99
    assert IE.decide(_summary(), _compare(lo=-0.5, hi=0.0), META)["verdict"] == "EQUIVALENT_FOR_INFERENCE"  # bounds incl.
    for s, c, m in ((_summary(), _compare(), {**META, "complete": False}), (_summary(), None, META),
                    (_summary(), {"diff": {}}, META), ({**_summary(), "fla_calls": 0}, _compare(), META),
                    (None, None, None)):
        d = IE.decide(s, c, m)
        assert d["verdict"] == "INCONCLUSIVE" and d["missing"] and IE.valid_decision(d)


def test_valid_decision_rejects_malformed():
    d = IE.decide(_summary(), _compare(), META)
    assert not IE.valid_decision({**d, "rule": "v0"}) and not IE.valid_decision({**d, "verdict": "ADOPT"})
    bad = json.loads(json.dumps(d))
    bad["gates"]["agreement"]["pass"] = False  # EQUIVALENT with a failed gate is not a valid decision
    assert not IE.valid_decision(bad) and not IE.valid_decision(None) and not IE.valid_decision({})


# ----------------------------------------------------------------------------- GPU_VERIFIED scope
def test_fla_override_is_scoped_to_this_mode(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    from heads import rwkv7_wkv as W
    from test_fla_rwkv7 import tiny_lm

    m = tiny_lm()
    monkeypatch.delenv(IE.ENV_FLAG, raising=False)
    with pytest.raises(RuntimeError, match="scoped"):
        IE.select_arm(m, "fla")
    assert IE.select_arm(m, "chunked") == "chunked"
    with pytest.raises(RuntimeError, match="GPU-verified"):
        W.select_wkv(m, "fla")  # the global default is untouched
    monkeypatch.setenv(IE.ENV_FLAG, "1")
    assert IE.select_arm(m, "fla") == "fla" and (m.config.wkv_implementation != "chunked") is False
    assert IE.select_arm(m, "chunked") == "chunked" and W.GPU_VERIFIED is False
    with pytest.raises(ValueError):
        IE.select_arm(m, "eager")


def test_override_never_reaches_training_or_defaults():
    hits = []
    for sub in ("train", "heads", "decide", "jobs", "scripts"):
        for f in (ROOT / sub).rglob("*"):
            if f.suffix not in (".py", ".sh", ".json") or not f.is_file():
                continue
            s = f.read_text(encoding="utf-8", errors="replace")
            if IE.ENV_FLAG in s:
                hits.append(f.relative_to(ROOT).as_posix())
    assert sorted(hits) == ["jobs/fla_rwkv7/fla_job.sh", "jobs/fla_rwkv7/infer_equiv.py"]
    sh = RUNNER.read_text(encoding="utf-8")
    branch = sh[sh.index('if [ "$MODE" = infer_equiv ]; then\n  # validation-only'):]
    assert sh.count(f"{IE.ENV_FLAG}=1") == 1 and f"{IE.ENV_FLAG}=1" in branch[:400]
    src = (ROOT / "jobs/fla_rwkv7/infer_equiv.py").read_text(encoding="utf-8")
    assert src.count("allow_unverified=True)") == 1  # only inside select_arm, after the env check
    assert "GPU_VERIFIED = " not in src and "GPU_VERIFIED=" not in src
    assert "GPU_VERIFIED = False" in (ROOT / "heads/rwkv7_wkv.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- runner + params
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_runner_parses_and_refuses_unknown_mode():
    assert subprocess.run(["bash", "-n", str(RUNNER)]).returncode == 0
    r = subprocess.run(["bash", str(RUNNER), "nope"], capture_output=True, text=True)
    assert r.returncode == 5 and "infer_equiv" in r.stderr


def test_runner_infer_equiv_branch_order():
    s = RUNNER.read_text(encoding="utf-8")
    b = s[s.index('if [ "$MODE" = infer_equiv ]; then\n  # validation-only'):s.index('CAP_W=$(cap warmup) && CAP_K')]
    order = ["fla_guard.py", "data.sha256", "dev_subset rows", "snapshot_download", "sha256sum -c \"$CODE/jobs/rwkv7/",
             "infer_equiv warmup", 'wait "$DL_PID"', "rate_guard run-weighted", "infer_equiv score",
             "dev_subset compare", "infer_equiv decide"]
    pos = [s.index(order[0])] + [b.index(x) for x in order[1:]]
    assert pos[1:] == sorted(pos[1:]), order
    assert "MARKER=FLA_RWKV7_INFER_EQUIV_DONE VALIDATOR=jobs.fla_rwkv7.infer_equiv:valid_decision" in s
    assert 'exit 0\nfi\nCAP_W=$(cap warmup) && CAP_K' in s  # the v1 / v2 phases never run in this mode
    assert "--candidate \"$WORK/out/fla.results.jsonl\"" in b and "--baseline \"$WORK/out/chunked.results.jsonl\"" in b


def test_job_params_cost_schedule_and_lint():
    from jobs.common import costs, lint

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert p["max_duration_minutes"] == 40 and costs.worst_case_usd(p) <= 5.0 <= p["hard_cap_usd"] + 1e-9
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert p["chip"] == "h100" and p["max_restarts"] == 0 and p["decision_rule"] == "jobs/fla_rwkv7/infer_equiv.py"
    assert p["submit"]["command"] == "bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh infer_equiv"
    assert costs.PARAMS["fla_rwkv7_infer_equiv"] == "jobs/fla_rwkv7/job_params_infer_equiv.json"
    spec = (ROOT / "jobs/fla_rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"${costs.worst_case_usd(p):.2f}" in spec
    rel = "jobs/fla_rwkv7/job_params_infer_equiv.json"
    bad, waived = lint.failures(rel, lint.lint_params(rel, ROOT))
    assert not bad and not waived, (bad, waived)


# ----------------------------------------------------------------------------- end to end on CPU (tiny model)
VOCAB = 260


def _enc(text, trusted=True):
    return [b + 4 for b in text.encode("utf-8")]


def _rows(n):
    qs = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments",
                                                                                 "technical": "Outages", "legal": None}},
          "down": {"type": "noul", "instructions": "Is a service down?"}}
    return [{"_evaluation": {"run_id": f"r{i}", "catalog_id": 1, "group_id": i},
             "state": f"Ticket {i}: checkout returned HTTP 500 since 09:0{i}; invoice INV-{i} overdue.",
             "questions": qs} for i in range(n)]


@pytest.fixture
def tiny_production(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("decision_index")
    from decide import Decider
    from decide.backends.hf_readout import HFReadoutBackend, ModelPlugin
    from decide.backends.state_fork import RWKV7StateOps
    from heads import rwkv7_wkv as W
    from test_fla_rwkv7 import fake_chunk
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    m = Rwkv7ForCausalLM(Rwkv7Config(vocab_size=VOCAB, hidden_size=32, num_heads=4, head_dim=8, num_hidden_layers=2,
                                     intermediate_size=64, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                                     v_low_rank_dim=8))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    m = m.float().eval()
    fwd = lambda ids, n: m(input_ids=ids, logits_to_keep=n, use_cache=False).logits  # noqa: E731
    plug = ModelPlugin("rwkv7", "tiny", None, _enc, fwd, device="cpu", dtype="float32", state_ops=RWKV7StateOps(m))
    decider = Decider(HFReadoutBackend(plug, "jrt-rwkv-g1-v1", max_batch_rows=3, fork=True), enable_v0_1_deltas=True)
    monkeypatch.setattr(W, "_import_chunk", lambda: (fake_chunk, None))
    monkeypatch.setattr(W, "fallback_reason", lambda r, cu, allow_fp32: None)
    monkeypatch.setenv(IE.ENV_FLAG, "1")
    W.FALLBACKS.clear()
    W.CALLS.clear()
    yield decider, m
    W.FALLBACKS.clear()
    W.CALLS.clear()


def test_run_pairs_interleaves_both_arms_on_one_model_and_decides(tiny_production, tmp_path):
    decider, m = tiny_production
    rows = _rows(4)
    meta = IE.run_pairs(decider, m, rows, tmp_path, progress=str(tmp_path / "progress.jsonl"))
    assert meta["complete"] and meta["rows_done"] == 4 and not meta["deadline_hit"] and meta["gpu_verified"] is False
    assert str(m.config.wkv_implementation) == "chunked"  # restored after scoring
    pairs = [json.loads(x) for x in (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["order"] for p in pairs] == [["chunked", "fla"], ["fla", "chunked"]] * 2
    for p in pairs:
        assert p["chunked"]["wkv"] == "chunked" and p["chunked"]["fla_calls"] == 0
        assert p["fla"]["wkv"] == "fla" and p["fla"]["fla_calls"] > 0 and p["fla"]["fallbacks"] == 0
        assert p["chunked"]["status"] == p["fla"]["status"] == "ok"
    for arm in IE.ARMS:  # kit results lines, DEV row order, scored-answer format
        lines = [json.loads(x) for x in (tmp_path / f"{arm}.results.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["run_id"] for x in lines] == [r["_evaluation"]["run_id"] for r in rows]
        assert all(x["status"] == "ok" and set(x["response"]["answers"]) == {"dept", "down"} for x in lines)
    prog = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prog) == 4 and json.loads((tmp_path / "progress.jsonl.total.json").read_text())["n_units"] == 4
    s = IE.summarize_pairs(pairs)
    assert s["agreement"] == 1.0 and s["high_margin_flips"] == 0 and s["max_abs_dp"] < 1e-4 and s["finite"]
    assert s["fla_fallbacks"] == 0 and s["fla_calls"] > 0 and s["questions"] == 8
    (tmp_path / "compare.json").write_text(json.dumps(_compare(lo=-0.01, hi=0.01, pt=0.0, se=0.005)))
    d = IE.phase_decide(tmp_path)
    assert d["verdict"] == "EQUIVALENT_FOR_INFERENCE" and IE.valid_decision(d)
    assert json.loads((tmp_path / "decision_infer.json").read_text())["verdict"] == "EQUIVALENT_FOR_INFERENCE"


def test_run_pairs_deadline_gives_inconclusive(tiny_production, tmp_path):
    decider, m = tiny_production
    meta = IE.run_pairs(decider, m, _rows(3), tmp_path, deadline_s=-1.0, warm=False)
    assert meta["deadline_hit"] and meta["rows_done"] == 0 and not meta["complete"]
    (tmp_path / "compare.json").write_text("")  # what a failed compare leaves behind
    d = IE.phase_decide(tmp_path)
    assert d["verdict"] == "INCONCLUSIVE" and IE.valid_decision(d) and not d["adopt_for_inference"]
    assert math.isfinite(meta["elapsed_s"])

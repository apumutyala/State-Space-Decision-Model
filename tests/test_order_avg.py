"""Test-time order averaging (decide.debias.OrderAveragedBackend, jobs.m2_first_cell.orderavg), CPU only.

* a synthetic scorer with a pure position bias: averaging over the pre-registered orders removes it exactly;
* non-choice questions (noul, score) are never reordered and keep the identity pass's logits bit for bit;
* n = 1 is the base backend (same revision, same response); default rows / prompts are byte-identical to the base
  commit (golden digests computed at 1fa91e7 with the same code below);
* the v3w layout under a presentation order: option j's code / schema span follow its presented position, the option
  axis stays canonical.
"""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from decide import Decider
from decide.backends.base import Plan, ScoreResult
from decide.debias import (ORDER_AVG_NS, OrderAveragedBackend, OrderCache, order_eligible, order_variants,
                           presentation_passes)
from decide.prompt_g1v3 import G1V3W_ID, render_g1v3
from decide.schema import canonical_options
from heads.layout import ByteEncoder, _v3_code_table, build_row

REQ = {"model": "t", "state": "The parcel arrived crushed; customer wants it fixed today.",
       "questions": {
           "action": {"type": "choice", "instructions": "Best next step?",
                      "criteria": {"refund": "Refund the order", "replace": "Ship a replacement", "ignore": None}},
           "cat": {"type": "choice", "instructions": "Category.",
                   "criteria": {"A": "electronics", "B": "clothing", "C": "food", "D": "other"}},
           "urgent": {"type": "noul", "instructions": "Is it urgent?"},
           "sev": {"type": "score", "instructions": "Severity 1-3.", "criteria": ["low", "mid", "high"]},
           "pick": {"type": "choice", "instructions": "Pick one.",
                    "criteria": {f"option_{i}": f"desc {i}" for i in range(12)}}}}
#: computed at the base commit 1fa91e7 (stack/rwkv7/d13-fulldev, before order averaging existed) with _digest below
GOLDEN_ROWS = "169e9be29e4daa851299b3d1197e7f1aa9fb3e53122d04a18c4c196c2c4eaa04"
GOLDEN_RENDER_CAT = "0071bee5919de7dee9ad0a948d05dfa398f25d29a38fddd6dc9b13e04ea90c5a"


def _digest(row) -> str:
    meta = {k: v for k, v in row.meta.items() if k != "trusted_paths"}
    return json.dumps([row.ids, row.option_ids, row.state_span, row.field_span, row.option_spans, row.continuations,
                       meta], default=str, sort_keys=True)


class Biased:
    """Synthetic scorer: logit = content[option] + bias[presented position] (log-softmax normalised), any type."""

    name = "biased"

    def __init__(self, bias=(3.0,), content=None):
        self.bias, self.content = list(bias), content or {}
        self.calls: list[dict] = []

    def revision(self):
        return {"revision_id": "biased@1", "model_id": "toy"}

    def plan(self, request, presentation=None):
        return Plan(inference_calls=1, max_call_input_tokens=100, total_input_tokens=100 * len(request["questions"]))

    def score(self, request, check_deadline, presentation=None):
        check_deadline()
        self.calls.append(dict(presentation or {}))
        logits = {}
        for qid, q in request["questions"].items():
            opts = [o for o, _ in canonical_options(q)]
            order = (presentation or {}).get(qid, list(range(len(opts))))
            pos = {j: p for p, j in enumerate(order)}
            raw = {o: self.content.get((qid, o), 0.0) + (self.bias[pos[j]] if pos[j] < len(self.bias) else 0.0)
                   for j, o in enumerate(opts)}
            z = math.log(sum(math.exp(v) for v in raw.values()))
            logits[qid] = {o: v - z for o, v in raw.items()}
        return ScoreResult(logits=logits, inference_calls=1, total_input_tokens=100 * len(request["questions"]),
                           extras={}, features={})


def _served(backend, req=REQ):
    resp = Decider(backend).decide(json.loads(json.dumps(req)))
    assert resp["ssdm"]["status"] == "ok", resp["ssdm"]
    return resp


# ------------------------------------------------------------------------------------------------ orders
def test_orders_fixed_distinct_and_seeded_by_question_only():
    q4 = REQ["questions"]["cat"]
    v = order_variants(q4, "cat", 4)
    assert v[0] == [0, 1, 2, 3] and v[1] == [3, 2, 1, 0] and len({tuple(x) for x in v}) == 4
    assert sorted(o[0] for o in v) == [0, 1, 2, 3]  # k >= 4: four different options presented first
    assert order_variants(q4, "cat", 4) == v and order_variants(q4, "cat", 2) == v[:2]
    assert order_variants({"type": "choice", "criteria": {"x": 1, "y": 2}}, "q", 4) == [[0, 1], [1, 0]]
    k3 = order_variants(REQ["questions"]["action"], "action", 4)
    assert len(k3) == 4 and sorted(o[0] for o in k3[:3]) == [0, 1, 2]
    for t in ("urgent", "sev"):
        assert not order_eligible(REQ["questions"][t]) and len(order_variants(REQ["questions"][t], t, 4)) == 1
    with pytest.raises(ValueError):
        OrderAveragedBackend(Biased(), 3)
    assert ORDER_AVG_NS == (1, 2, 4)


def test_passes_never_touch_non_choice_questions():
    passes = presentation_passes(REQ, 4)
    assert passes[0] == {} and len(passes) == 4
    for p in passes[1:]:
        assert set(p) <= {"action", "cat", "pick"}
    only_binary = {"model": "t", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": 1, "b": 2}}}}
    assert len(presentation_passes(only_binary, 4)) == 2  # trailing empty passes dropped: no cost
    assert len(presentation_passes({"model": "t", "state": "s", "questions": {"u": {"type": "noul"}}}, 4)) == 1


# ------------------------------------------------------------------------------------------------ bias removal
@pytest.mark.parametrize("bias", [(3.0,), (2.0, 1.0), (0.0, 0.0, 0.0, 2.5)])
def test_position_bias_removed_when_options_tie(bias):
    """Equal content, pure position bias: N = 1 serves the prior; N = 2 is exactly uniform on k = 2; N = 4 is exactly
    uniform for a primacy bias on k = 4 (four different first options) and strictly flatter in general."""
    b1 = _served(OrderAveragedBackend(Biased(bias), 1))["ssdm"]["raw_probabilities"]
    b2 = _served(OrderAveragedBackend(Biased(bias), 2))["ssdm"]["raw_probabilities"]
    b4 = _served(OrderAveragedBackend(Biased(bias), 4))["ssdm"]["raw_probabilities"]

    def tv(p):
        return 0.5 * sum(abs(x - 1 / len(p)) for x in p.values())

    for qid in ("action", "cat", "pick"):
        if tv(b1[qid]) < 1e-12:  # the bias does not reach this question's positions (k = 3 vs a 4th-position bias)
            assert tv(b4[qid]) < 1e-12 and tv(b2[qid]) < 1e-12
            continue
        assert tv(b4[qid]) < tv(b1[qid]) - 1e-9 and tv(b2[qid]) < tv(b1[qid]) - 1e-9
    if len(bias) == 1:
        assert tv(b4["cat"]) < 1e-12  # primacy bias, k = 4: each option first exactly once
    binary = {"model": "t", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}}}
    p1 = _served(OrderAveragedBackend(Biased(bias), 1), binary)["ssdm"]["raw_probabilities"]["q"]
    p2 = _served(OrderAveragedBackend(Biased(bias), 2), binary)["ssdm"]["raw_probabilities"]["q"]
    assert abs(p1["a"] - p1["b"]) > 0.1 or bias[0] == 0.0
    assert abs(p2["a"] - 0.5) < 1e-12 and abs(p2["b"] - 0.5) < 1e-12


def test_content_survives_and_averaging_flips_a_prior_driven_argmax():
    content = {("cat", "C"): 1.0}  # C is right by content; position 0 (A under identity) gets a larger prior
    s1 = _served(OrderAveragedBackend(Biased((2.0,), content), 1))
    s4 = _served(OrderAveragedBackend(Biased((2.0,), content), 4))
    p1, p4 = s1["ssdm"]["raw_probabilities"]["cat"], s4["ssdm"]["raw_probabilities"]["cat"]
    assert max(p1, key=p1.get) == "A" and max(p4, key=p4.get) == "C"
    assert abs(sum(p4.values()) - 1) < 1e-12


def test_non_choice_questions_untouched_bit_for_bit():
    base = Biased((2.0, 1.0), {("urgent", "true"): 0.7, ("sev", "2"): 0.3})
    ref = _served(OrderAveragedBackend(Biased((2.0, 1.0), {("urgent", "true"): 0.7, ("sev", "2"): 0.3}), 1))
    got = _served(OrderAveragedBackend(base, 4))
    for qid in ("urgent", "sev"):
        assert got["ssdm"]["raw_probabilities"][qid] == ref["ssdm"]["raw_probabilities"][qid]
        assert got["answers"][qid] == ref["answers"][qid]
    assert all(set(c) <= {"action", "cat", "pick"} for c in base.calls)
    assert len(base.calls) == 4 and got["usage"]["input_tokens"] == 4 * ref["usage"]["input_tokens"]


# ------------------------------------------------------------------------------------------------ default unchanged
def test_n1_is_the_base_backend():
    base = Biased((1.5,))
    w = OrderAveragedBackend(base, 1)
    assert w.revision() == base.revision() and w.name == base.name
    a, b = _served(Biased((1.5,))), _served(w)
    for k in ("answers", "usage"):
        assert a[k] == b[k]
    assert a["ssdm"]["raw_probabilities"] == b["ssdm"]["raw_probabilities"]
    assert a["ssdm"]["served_probabilities"] == b["ssdm"]["served_probabilities"]
    assert base.calls == [{}]
    assert "+OA:ssdm.order_avg.v1:n4" in OrderAveragedBackend(base, 4).revision()["revision_id"]


def test_shared_cache_runs_each_pass_once_and_matches_uncached():
    base, cache = Biased((2.0, 0.5)), OrderCache()
    arms = {n: OrderAveragedBackend(base, n, cache) for n in (4, 2, 1)}
    out = {n: _served(arms[n]) for n in (4, 2, 1)}
    assert len(base.calls) == len(presentation_passes(REQ, 4)) and cache.hits == 2 + 1
    assert arms[2].last_input_tokens == 2 * arms[1].last_input_tokens and len(arms[4].last_pass_ms) == 4
    for n in (2, 1):
        fresh = _served(OrderAveragedBackend(Biased((2.0, 0.5)), n))
        assert out[n]["ssdm"]["raw_probabilities"] == fresh["ssdm"]["raw_probabilities"]


def test_default_rows_and_prompt_byte_identical_to_base_commit():
    h = hashlib.sha256()
    for qid in sorted(REQ["questions"]):
        row = build_row(REQ, qid, ByteEncoder(), template=G1V3W_ID, layout="g1v2")
        assert _digest(row) == _digest(build_row(REQ, qid, ByteEncoder(), template=G1V3W_ID, layout="g1v2",
                                                 presentation={}))
        h.update(_digest(row).encode())
    assert h.hexdigest() == GOLDEN_ROWS
    text = render_g1v3(REQ, "cat", _v3_code_table(), 64, False, True, frozenset())
    assert hashlib.sha256(text.encode()).hexdigest() == GOLDEN_RENDER_CAT
    assert render_g1v3(REQ, "cat", _v3_code_table(), 64, False, True, frozenset(), {"cat": [0, 1, 2, 3]}) == text


# ------------------------------------------------------------------------------------------------ v3w layout
def _text(row, a, b):
    return bytes(i - 4 for i in row.ids[a:b] if i >= 4).decode("utf-8", "replace")


def test_v3w_presentation_moves_codes_and_spans_not_the_option_axis():
    enc, table = ByteEncoder(), _v3_code_table()
    pres = {"cat": [2, 3, 0, 1], "pick": [11, *range(11)]}
    row0 = build_row(REQ, "cat", enc, template=G1V3W_ID, layout="g1v2")
    row = build_row(REQ, "cat", enc, template=G1V3W_ID, layout="g1v2", presentation=pres)
    assert row.option_ids == row0.option_ids == ["A", "B", "C", "D"]
    for k, oid in enumerate(row.option_ids):  # canonical option k is listed at position pres.index(k)
        pos = pres["cat"].index(k)
        assert row.continuations[k] == enc(" " + table[pos])
        assert REQ["questions"]["cat"]["criteria"][oid] in _text(row, *row.option_spans[k])
    body = _text(row, 0, len(row.ids))
    assert body.index("C") < body.index("clothing") and body.index("food") < body.index("electronics")
    big = build_row(REQ, "pick", enc, template=G1V3W_ID, layout="g1v2", presentation=pres)
    assert big.continuations[11] == enc(" " + table[0]) and big.continuations[0] == enc(" " + table[1])
    oid11, desc11 = canonical_options(REQ["questions"]["pick"])[11]  # canonical = sorted ids: option_9
    assert big.option_ids[11] == oid11 and desc11 in _text(big, *big.option_spans[11])


def test_presentation_refused_outside_v3w_choice():
    with pytest.raises(ValueError):
        build_row(REQ, "cat", ByteEncoder(), template=G1V3W_ID, layout="g1v2", presentation={"urgent": [1, 0]})
    with pytest.raises(ValueError):
        build_row(REQ, "cat", ByteEncoder(), template=G1V3W_ID, layout="g1v2", presentation={"cat": [0, 0, 1, 2]})
    with pytest.raises(ValueError):
        build_row(REQ, "cat", ByteEncoder(), template="jrt-plain-v1", layout="ordered", presentation={"cat": [1, 0, 2, 3]})


# ------------------------------------------------------------------------------------------------ job wiring
def test_job_params_and_arm_set_pre_registered():
    from pathlib import Path

    from jobs.common.costs import quoted_max_cost_usd
    from jobs.m2_first_cell.fulldev_eval import estimate
    from jobs.m2_first_cell.orderavg import arm_file, parse_ns

    assert parse_ns("1,2,4") == (4, 2, 1) and parse_ns(None) == ()
    with pytest.raises(ValueError):
        parse_ns("1,2")
    assert arm_file("child", 1, "results") == "child.results.jsonl"
    assert arm_file("child", 4, "partial") == "child.oa4.partial.jsonl"
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "jobs/m2_first_cell/job_params_orderavg_d13_v3w.json").read_text(encoding="utf-8"))
    assert p["order_avg"] == "1,2,4" and p["rows"] == "proxy600" and p["readout"] == "v3w"
    assert quoted_max_cost_usd(p) <= 12.0
    est = estimate(p)
    assert est["ok"] and est["readout"] == "v3w"


def test_rule_constants_pre_registered():
    from jobs.m2_first_cell.orderavg import ARMS, BOOTSTRAP_B, BOOTSTRAP_SEED, CANDIDATE_ARMS, CI_LO_THRESHOLD, RULE

    assert ARMS == (4, 2, 1) and CANDIDATE_ARMS == (2, 4) and CI_LO_THRESHOLD == 0.0
    assert BOOTSTRAP_B == 2000 and BOOTSTRAP_SEED == 0 and "lower bound > 0" in RULE
    from jobs.m2_first_cell.orderavg import CONTROL_CI

    assert CONTROL_CI == (37.57, 45.22)


def test_fulldev_score_writes_three_arms_from_shared_passes_and_resumes(monkeypatch, tmp_path):
    """fulldev_eval.score --order-avg 1,2,4 with the synthetic scorer: N = 1 rows equal a plain (no order averaging)
    run's, the N = 4 arm ran every forward pass once, and a rerun resumes only rows missing from any arm."""
    import sys
    import types

    if not sys.flags.utf8_mode:
        pytest.skip("needs python -X utf8")
    from jobs.m2_first_cell import fulldev_eval as F
    from train import m2_gates as G

    rows = [{"_evaluation": {"run_id": f"r{i}"}, "state": f"s{i}", "questions": REQ["questions"]} for i in range(5)]
    monkeypatch.setattr(F, "N_DEV", len(rows))
    monkeypatch.setattr(F, "selected_ids", lambda s: None)
    monkeypatch.setitem(sys.modules, "harness.suite", types.SimpleNamespace(read_rows=lambda p: rows))
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    monkeypatch.setattr(F, "plan_rows", lambda d, r, m: {"n_rows": len(r), "n_refused": 0, "n_over_16384": 0,
                                                         "max_row_tokens": 1, "rev": d.revision()["revision_id"]})
    backends = []

    def load_child(a, **kw):
        b = Biased((2.0, 0.5), {("cat", "C"): 0.4})
        backends.append(b)
        return Decider(b), False, {"arm": "child", "readout": {"template": "jrt-rwkv-g1-v3w"}}

    def predict(decide, rs, untempered_from_raw=False):
        r = rs[0]
        resp = decide({"model": "m", "state": r["state"], "questions": r["questions"]})
        return {r["_evaluation"]["run_id"]: {"run_id": r["_evaluation"]["run_id"], "status": resp["ssdm"]["status"],
                                             "response": {"answers": resp["answers"]}, "total_wall_ms": 1.0}}

    monkeypatch.setattr(F, "load_child", load_child)
    monkeypatch.setattr(G, "index_predict", predict)
    monkeypatch.setattr(G, "beat_wrap", lambda fn: fn)
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": {"template": "jrt-rwkv-g1-v1", "layout": "ordered"}}), encoding="utf-8")

    def args(out, order_avg):
        return types.SimpleNamespace(rows="rows", rows_select="full", arm="child", readout="v3w", max_input_tokens=65536,
                                     cell=str(cell), out_dir=str(tmp_path / out), resume_dir=str(tmp_path / f"ck-{out}"),
                                     progress=str(tmp_path / f"p-{out}.jsonl"), order_avg=order_avg)

    assert F.score(args("plain", None)) == 0
    plain = [json.loads(x) for x in (tmp_path / "plain/child.results.jsonl").read_text().splitlines()]
    a = args("oa", "1,2,4")
    assert F.score(a) == 0
    passes = len(presentation_passes(REQ, 4))
    assert len(backends[-1].calls) == passes * len(rows)
    out = {n: [json.loads(x) for x in (tmp_path / "oa" / name).read_text().splitlines()]
           for n, name in ((1, "child.results.jsonl"), (2, "child.oa2.results.jsonl"), (4, "child.oa4.results.jsonl"))}
    assert [r["response"] for r in out[1]] == [r["response"] for r in plain]
    assert [r["oa_n"] for r in out[4]] == [4] * 5 and all(len(r["oa_pass_ms"]) == passes for r in out[4])
    assert all(len(r["oa_pass_ms"]) == 1 for r in out[1]) and all(len(r["oa_pass_ms"]) == 2 for r in out[2])
    assert out[2][0]["oa_input_tokens"] == 2 * out[1][0]["oa_input_tokens"]
    assert out[4][0]["response"]["answers"]["urgent"] == out[1][0]["response"]["answers"]["urgent"]
    assert out[4][0]["response"]["answers"]["cat"] != out[1][0]["response"]["answers"]["cat"]
    ident = json.loads((tmp_path / "oa/identity-child.json").read_text())
    assert ident["order_avg"]["arms"] == [4, 2, 1]
    # resume: drop r3, r4 from the N = 2 log only -> the rerun scores exactly those two rows again
    p2 = tmp_path / "ck-oa/child.oa2.partial.jsonl"
    p2.write_text("\n".join(p2.read_text().splitlines()[:3]) + "\n")
    assert F.score(a) == 0
    assert len(backends[-1].calls) == passes * 2
    bad = args("bad", "1,2")
    assert F.score(bad) == 10


def test_report_section_verdict_and_cost(monkeypatch, tmp_path):
    from jobs.m2_first_cell import fulldev_eval as F
    from jobs.m2_first_cell.orderavg import orderavg_section

    for n, wall in ((1, 2.0), (2, 4.0), (4, 6.0)):
        name = "child.results.jsonl" if n == 1 else f"child.oa{n}.results.jsonl"
        rows = [{"run_id": f"r{i}", "status": "ok", "total_wall_ms": wall * 1000, "oa_pass_ms": [1.0] * min(n, 3),
                 "oa_input_tokens": 100 * min(n, 3)} for i in range(2)]
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(F, "subset_score", lambda p, keep, B, seed: {"rows": 2, "slice_index": {"index": {"point": 41.0}}})
    lo = {"child.oa2.results.jsonl": -0.4, "child.oa4.results.jsonl": 0.3}
    monkeypatch.setattr(F, "subset_compare", lambda c, b, keep, B, seed: {"diff": {"index": {"point": 1.0, "lo": lo[c.name],
                                                                                              "hi": 2.0}}})
    s = orderavg_section(tmp_path, None)
    assert s["verdict"] == "ADOPT_CANDIDATE" and s["winning_arms"] == ["n4"]
    assert s["arms"]["n4"]["cost_multiplier_vs_n1"] == {"wall": 3.0, "input_tokens": 3.0, "passes": 3.0}
    assert s["arms"]["n2"]["cost_multiplier_vs_n1"]["wall"] == 2.0
    lo["child.oa4.results.jsonl"] = 0.0  # lower bound must be strictly > 0
    assert orderavg_section(tmp_path, None)["verdict"] == "REJECT"
    monkeypatch.setattr(F, "subset_score", lambda p, keep, B, seed: {"rows": 2, "slice_index": {"index": {"point": 30.0}}})
    assert orderavg_section(tmp_path, None)["verdict"] == "INVALID_CONTROL"

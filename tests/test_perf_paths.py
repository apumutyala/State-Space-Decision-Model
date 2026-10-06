"""Equivalence tests for the PERF_AUDIT fixes (jobs/PERF_AUDIT.md), CPU, tiny models:

* decision-head evaluation with a shared schema + state prefix (heads.prefix_features) == the plain per-row path,
  for every head and every supported backbone, incl. permutation of rows and mixed-request batches;
* DecisionModelBackend scoring a request in one pass == scoring each question row alone (the old loop);
* GCG candidate evaluation through the prefix cache (HFLetterDecider.score_prefixed) == full re-reads;
* the readout's segment-encoding memo changes no ids and encodes each distinct segment once.
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from heads import HEADS, ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402

ENC = ByteEncoder()
SMALL = {"clef_control": {"width": 16, "heads": 2, "feedforward": 32, "layers": 1, "routing_layers": 1}}
TOL = 1e-5

REQ_A = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme; invoice INV-7 overdue",
         "questions": {"dept": {"type": "choice", "instructions": "team?",
                                "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
                       "down": {"type": "noul", "instructions": "is a service down?"},
                       "urg": {"type": "score", "criteria": ["later", "week", "today"]},
                       "inv": {"type": "choice", "instructions": "invoice?", "criteria": {"paid": None, "late": None}}}}
REQ_B = {"model": "t", "state": "Weather report: heavy rain expected; roads closed near the river.",
         "questions": {"risk": {"type": "choice", "instructions": "flood risk?",
                                "criteria": {"low": None, "medium": None, "high": None}},
                       "go": {"type": "noul", "instructions": "safe to drive?"}}}


def rows(req, layout="ordered"):
    return [build_row(req, q, ENC, template="jrt-plain-v1", layout=layout) for q in req["questions"]]


def _model(family, head):
    bb = tiny_hf_backbone(family, fork_mode="cache")
    with torch.no_grad():
        for p in bb.model.parameters():
            p.mul_(1.0)
    return DecisionModel(bb, head, SMALL.get(head), seed=0).eval()


@pytest.mark.parametrize("family", ["rwkv7", "qwen3.5", "q9af"])
@pytest.mark.parametrize("head", sorted(HEADS))
def test_shared_prefix_eval_equals_plain(family, head):
    from heads import prefix_features as PF

    m = _model(family, head)
    assert PF.supported(m.backbone)  # the shared-prefix path really runs (not the fallback)
    rs = rows(REQ_A) + rows(REQ_B)  # two requests in one batch: two prefix groups
    with torch.no_grad():
        a = m(rs, shared_prefix=True).log_probs
        b = m(rs, shared_prefix=False).log_probs
        rev = m(list(reversed(rs)), shared_prefix=True).log_probs.flip(0)
    fin = torch.isfinite(b)
    assert torch.equal(fin, torch.isfinite(a))
    assert torch.allclose(a[fin], b[fin], atol=TOL, rtol=TOL), (a - b).abs()[fin].max()
    assert torch.allclose(rev[fin], b[fin], atol=TOL, rtol=TOL)  # row order does not matter


def test_shared_prefix_is_only_used_without_grad():
    from heads import prefix_features as PF

    m = _model("rwkv7", "integrator")
    called = []
    orig = PF.shared_prefix_features
    PF.shared_prefix_features = lambda *a, **k: called.append(1) or orig(*a, **k)
    try:
        m(rows(REQ_A), shared_prefix=True)  # grad enabled: training path, untouched
        assert not called
        with torch.no_grad():
            m(rows(REQ_A), shared_prefix=True)
        assert called
    finally:
        PF.shared_prefix_features = orig


@pytest.mark.parametrize("family", ["rwkv7", "q9af"])
def test_decision_backend_one_pass_equals_per_question_loop(family):
    from heads.decide_backend import DecisionModelBackend

    m = _model(family, "fork")
    be = DecisionModelBackend(m, ENC, template="jrt-plain-v1")
    res = be.score(REQ_A, lambda: None)
    assert res.inference_calls == be.plan(REQ_A).inference_calls == 1
    with torch.no_grad():
        for r in rows(REQ_A):
            lp = m([r]).log_probs[0]
            for k, oid in enumerate(r.option_ids):
                assert abs(res.logits[r.qid][oid] - float(lp[k])) <= TOL * max(1.0, abs(float(lp[k])))


# ----------------------------------------------------------------------------- GCG prefix cache
def _letter_decider():
    import transformers as tf

    from safety_eval.hf_decider import HFLetterDecider

    cfg = tf.Qwen3_5TextConfig(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                               linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    torch.manual_seed(0)
    m = tf.Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape) * 0.05)
    d = object.__new__(HFLetterDecider)
    d.model, d.tok = m, types.SimpleNamespace(pad_token_id=0)
    d.max_tokens, d.max_batch = 10**6, 5
    d.groups = [[40, 41], [42], [43, 44], [45]]
    return d


def test_gcg_prefix_cache_equals_full_reads():
    from safety_eval.gcg import Engine

    d = _letter_decider()
    g = torch.Generator().manual_seed(1)
    pre = torch.randint(5, 250, (40,), generator=g).tolist()
    tails = [torch.randint(5, 250, (int(n),), generator=g).tolist() for n in (7, 7, 9, 3, 7, 12, 7)]
    ns = [4, 4, 3, 4, 2, 4, 4]
    a = d.score_prefixed(pre, tails, ns)
    b = d.score_ids([pre + t for t in tails], ns)
    for (la, ma), (lb, mb) in zip(a, b):
        assert max(abs(x - y) for x, y in zip(la, lb)) <= TOL and abs(ma - mb) <= 1e-4
    prob = types.SimpleNamespace(pre_ids=pre, post_ids=[[7, 8, 9], [10, 11]], n=4, to_keys=lambda lp: lp)
    advs = torch.randint(5, 250, (6, 5), generator=g)
    e_on, e_off = Engine.__new__(Engine), Engine.__new__(Engine)
    for e, flag in ((e_on, True), (e_off, False)):
        e.dec, e.model, e.tok, e.n_seqs, e.prefix_cache = d, d.model, d.tok, 0, flag
    assert torch.allclose(e_on.eval_advs(prob, advs), e_off.eval_advs(prob, advs), atol=TOL)


# ----------------------------------------------------------------------------- readout segment memo
def test_segment_memo_changes_no_ids_and_encodes_once():
    from decide.backends.hf_readout import HFReadoutBackend, ModelPlugin

    calls = []

    def enc(text, trusted=True):
        calls.append(text)
        return [b + 4 for b in text.encode()]

    plug = ModelPlugin("t", "t", None, enc, None, device="cpu", dtype="float32")
    be = HFReadoutBackend(plug, "jrt-plain-v1")
    r1 = be._rows(REQ_A)
    n_first = len(calls)
    r2 = be._rows(REQ_A)
    assert r1 == r2 and len(calls) == n_first  # second pass (plan -> score) is free
    assert sum(t == REQ_A["state"] for t in calls) == 1  # the state is tokenised once, not once per question
    fresh = HFReadoutBackend(ModelPlugin("t", "t", None, lambda t, tr=True: [b + 4 for b in t.encode()], None,
                                         device="cpu", dtype="float32"), "jrt-plain-v1")
    assert fresh._rows(REQ_A) == r1


# ----------------------------------------------------------------------------- cost model + preflight
@pytest.mark.parametrize("family", ["rwkv7", "qwen3.5"])
def test_committed_volumes_match_the_subset_and_sum(family):
    import json
    from pathlib import Path

    from jobs.common import dev_subset

    root = Path(__file__).resolve().parents[1]
    v = json.loads((root / f"jobs/common/readout_volume_{family}.json").read_text())
    assert set(v["per_row"]) == set(dev_subset.load_subset()["run_ids"]) and v["refused"] == 0
    for path in ("recompute", "per_question", "fork"):
        for k in ("tokens", "calls", "positions"):
            assert v["totals"][path][k] == sum(r[path][k] for r in v["per_row"].values())
    t = v["totals"]
    assert t["recompute"]["tokens"] > 30 * t["fork"]["positions"]  # the fix is a >30x cut in model work
    assert t["fork"]["calls"] < 3 * v["rows"]  # packed tails: ~2 calls per row


def test_cost_model_reproduces_the_measured_job_and_clears_the_cap():
    import json
    from pathlib import Path

    from jobs.common import rate_guard as rg
    from jobs.common.readout_cost import CALL_S_FITTED_RWKV7, TOK_S_FITTED_RWKV7

    vol = json.loads((Path(__file__).resolve().parents[1] / "jobs/common/readout_volume_rwkv7.json").read_text())
    rec = rg.preflight_volume(vol, TOK_S_FITTED_RWKV7, CALL_S_FITTED_RWKV7, 75.0, path="recompute")
    assert 700 < rec["projected_min"] < 820 and not rec["ok"]  # job-mhc3h's live guard measured 763 min
    fk = rg.preflight_volume(vol, 4307.0, CALL_S_FITTED_RWKV7, 75.0, path="fork")  # probe's measured prefill rate
    assert fk["ok"] and fk["projected_min"] < 40
    assert rg.main(["preflight", "--tok-s", "4307", "--call-s", "0.098", "--cap-min", "75", "--volume",
                    str(Path(__file__).resolve().parents[1] / "jobs/common/readout_volume_rwkv7.json")]) == 0
    assert rg.main(["preflight", "--tok-s", "4307", "--call-s", "0.098", "--cap-min", "75", "--path", "recompute",
                    "--volume", str(Path(__file__).resolve().parents[1] / "jobs/common/readout_volume_rwkv7.json")]) == 15

"""Prefix-fork readout (decide.backends.state_fork) == full recompute, on tiny CPU models of every family the paid jobs
use: the vendored bundled RWKV-7 code, transformers Qwen3.5 (hybrid GDN + attention), and convert.model's
ConvertibleQwen with a student GDN in place of an attention layer. fp32, tolerance 1e-5. Also permutation
equivariance of the forked readout and plan/score call accounting.
"""
from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")

from decide.backends.hf_readout import HFReadoutBackend, ModelPlugin  # noqa: E402
from decide.backends.state_fork import (HFCacheStateOps, RWKV7StateOps, common_prefix_len,  # noqa: E402
                                        fork_plan)

VOCAB = 260
TOL = 1e-5


def enc(text, trusted=True):
    return [b + 4 for b in text.encode("utf-8")]


REQ = {
    "model": "t",
    "state": "Ticket 4411: checkout has returned HTTP 500 since 09:00; vendor Acme; invoice INV-7 is overdue by 12 days.",
    "questions": {
        "dept": {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": {"billing": "Payments", "technical": "Outages", "sales": None, "legal": "Contracts"}},
        "down": {"type": "noul", "instructions": "Is a service down?"},
        "urgency": {"type": "score", "criteria": ["later", "this week", "today"]},
        "inv": {"type": "choice", "instructions": "Invoice status?", "criteria": {"paid": None, "overdue": None}},
    },
}


def permuted(req):
    qs = {}
    for k in reversed(list(req["questions"])):
        q = dict(req["questions"][k])
        if q["type"] == "choice":
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
        qs[k] = q
    return {**req, "questions": qs}


def _randomise(m, std=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return m.float().eval()


def rwkv7_plugin():
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    m = _randomise(Rwkv7ForCausalLM(Rwkv7Config(vocab_size=VOCAB, hidden_size=32, num_heads=4, head_dim=8,
                                                num_hidden_layers=3, intermediate_size=64, a_low_rank_dim=8,
                                                decay_low_rank_dim=8, gate_low_rank_dim=8, v_low_rank_dim=8)))
    fwd = lambda ids, n: m(input_ids=ids, logits_to_keep=n, use_cache=False).logits  # noqa: E731
    return ModelPlugin("rwkv7", "tiny", None, enc, fwd, device="cpu", dtype="float32", state_ops=RWKV7StateOps(m))


def _tiny_qwen():
    import transformers as tf

    cfg = tf.Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                               linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    return _randomise(tf.Qwen3_5ForCausalLM(cfg), std=0.05)


def qwen_plugin():
    m = _tiny_qwen()
    fwd = lambda ids, n: m(input_ids=ids, logits_to_keep=n, use_cache=False).logits  # noqa: E731
    return ModelPlugin("qwen3.5", "tiny", None, enc, fwd, device="cpu", dtype="float32", state_ops=HFCacheStateOps(m))


def converted_plugin(student=True):
    from convert.model import ConvertedStateOps, ConvertibleQwen, attach_gdn, init_gdn, replace_native_gdn

    causal = _tiny_qwen()
    full = types.SimpleNamespace(model=types.SimpleNamespace(language_model=causal.model), lm_head=causal.lm_head)
    cm = ConvertibleQwen(full, kernel="chunk")
    replace_native_gdn(cm, kernel="chunk")
    i = cm.attn_layers[0]
    if student:
        g = init_gdn(cm, i, "random", 0)
        attach_gdn(cm, i, g, train_dtype=torch.float32)
        _randomise(cm.swaps[i].gdn, std=0.05, seed=3)
        cm.set_modes({i: "student"})
    cm.eval()

    @torch.no_grad()
    def fwd(ids, n):
        return cm.lm_head(cm.hidden(ids)[:, -n:]).float()

    return ModelPlugin("qwen3.5", "tiny-conv", None, enc, fwd, device="cpu", dtype="float32",
                       state_ops=ConvertedStateOps(cm))


FAMILIES = {"rwkv7": rwkv7_plugin, "qwen3.5": qwen_plugin, "q9af-student": converted_plugin,
            "q9af-stock-path": lambda: converted_plugin(student=False)}


def _score(backend, req):
    return backend.score(req, lambda: None)


def _probs(logits):
    out = {}
    for q, d in logits.items():
        m = max(d.values())
        z = sum(math.exp(v - m) for v in d.values())
        out[q] = {o: math.exp(v - m) / z for o, v in d.items()}
    return out


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_forked_readout_equals_full_recompute(family):
    plug = FAMILIES[family]()
    tmpl = "jrt-rwkv-g1-v1" if family == "rwkv7" else "jrt-chatml-v1"
    fork = HFReadoutBackend(plug, tmpl, max_batch_rows=3, fork=True)
    full = HFReadoutBackend(plug, tmpl, max_batch_rows=3, fork=False)
    a, b = _score(fork, REQ), _score(full, REQ)
    assert a.extras["scoring_path"] == "prefix-fork" and b.extras["scoring_path"] == "recompute"
    for q in REQ["questions"]:
        assert set(a.logits[q]) == set(b.logits[q])
        for o in a.logits[q]:
            assert abs(a.logits[q][o] - b.logits[q][o]) <= TOL * max(1.0, abs(b.logits[q][o])), (q, o)
    pa, pb = _probs(a.logits), _probs(b.logits)
    assert all(abs(pa[q][o] - pb[q][o]) <= TOL for q in pa for o in pa[q])
    # the point of the fork: far fewer tokens through the model
    assert a.total_input_tokens * 3 < b.total_input_tokens
    assert fork.revision()["revision_id"] != full.revision()["revision_id"]


def _close(a, b):
    """Scores within TOL relative (fp32 sums of ~10 token log-probs of magnitude ~20 round at ~1e-5 absolute) and
    probabilities within TOL absolute."""
    pa, pb = _probs(a), _probs(b)
    return all(abs(a[q][o] - b[q][o]) <= TOL * max(1.0, abs(b[q][o])) and abs(pa[q][o] - pb[q][o]) <= TOL
               for q in a for o in a[q])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_forked_readout_is_permutation_equivariant(family):
    plug = FAMILIES[family]()
    tmpl = "jrt-rwkv-g1-v1" if family == "rwkv7" else "jrt-chatml-v1"
    be = HFReadoutBackend(plug, tmpl, fork=True)
    a, b = _score(be, REQ), _score(be, permuted(REQ))
    assert _close(a.logits, b.logits)
    # and pack composition does not matter: one tail per call, a tiny token budget, or everything in one pack
    for kw in ({"fork_max_rows": 1}, {"fork_token_budget": 40}, {"fork_max_rows": 4096, "fork_token_budget": 10**6}):
        c = _score(HFReadoutBackend(plug, tmpl, fork=True, **kw), REQ)
        assert _close(a.logits, c.logits), kw


@pytest.mark.parametrize("family", ["rwkv7", "q9af-student"])
def test_plan_matches_score_accounting(family):
    plug = FAMILIES[family]()
    tmpl = "jrt-rwkv-g1-v1" if family == "rwkv7" else "jrt-chatml-v1"
    for fork in (True, False):
        be = HFReadoutBackend(plug, tmpl, max_batch_rows=3, fork=fork)
        p, s = be.plan(REQ), _score(be, REQ)
        assert p.inference_calls == s.inference_calls and p.total_input_tokens == s.total_input_tokens
        assert len(s.extras["call_latency_ms"]) == s.inference_calls


def test_decider_probabilities_match_through_the_adapter():
    from decide import Decider

    plug = rwkv7_plugin()
    out = []
    for fork in (True, False):
        be = HFReadoutBackend(plug, "jrt-rwkv-g1-v1", fork=fork)
        r = Decider(be, enable_v0_1_deltas=True).decide(REQ)
        assert r["ssdm"]["status"] == "ok"
        out.append(r["ssdm"]["raw_probabilities"])
    for q in out[0]:
        for o in out[0][q]:
            assert abs(out[0][q][o] - out[1][q][o]) <= TOL


def test_gdn_step_over_chunks_equals_forward():
    from convert.gdn import GDN

    torch.manual_seed(0)
    g = _randomise(GDN(32, 2, 4, 8, 8, 4, kernel="chunk"), std=0.1)
    x = torch.randn(2, 37, 32)
    full = g(x)
    o1, st = g.step(x[:, :20])
    o2, st = g.step(x[:, 20:33], st)
    o3, _ = g.step(x[:, 33:], st)
    assert torch.allclose(torch.cat([o1, o2, o3], 1), full, atol=1e-5)


def test_common_prefix_and_plan_shape():
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 5]]) == 2
    assert common_prefix_len([[1, 2, 3]]) == 2  # a single prompt keeps one own token
    assert common_prefix_len([[1, 2], [1, 2, 3]]) == 1
    rows = [{"qid": q, "option": o, "prompt": [9, 9, 9, i], "cont": [7] * n}
            for i, (q, opts) in enumerate({"a": {"x": 2, "y": 3}, "b": {"z": 1}}.items()) for o, n in opts.items()]
    L, calls = fork_plan(rows, 4)
    assert L == 3 and [c.kind for c in calls] == ["prefix", "pack"]
    # tails: a/x = [0] + 1 cont tok, a/y = [0] + 2, b/z = [1] + 0  -> 2 + 3 + 1 real tokens
    assert calls[1].width == 3 and sum(c.tokens for c in calls) == 3 + 2 + 3 + 1
    _, calls = fork_plan(rows, 1)
    assert [c.kind for c in calls] == ["prefix", "pack", "pack", "pack"]

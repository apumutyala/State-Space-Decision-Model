"""Permutation equivariance: shuffling choice options / question order in the
request must not change any option's probability (within float tolerance).

This is an *adapter* property: the adapter canonicalises order, so the model
input is byte-identical under shuffles (``test_prompt_is_canonical``) and any
deterministic backend passes. It is not evidence that a model is order-robust.

Covered for the oracle, the HF readout path (on a tiny deterministic CPU LM, so
prompt canonicalisation and batching are exercised without downloads), and,
when ``SSDM_QWEN_DIR`` points at a Qwen3.5 snapshot, the real model.
"""

import os

import pytest
from conftest import base_request, shuffled

from decide import Decider
from decide.backends import OracleBackend
from decide.prompt import TEMPLATES, build_prompt

TOL = 1e-6


def assert_equivariant(decider, req, seeds=range(6), tol=TOL):
    ref = decider.decide(req)
    assert ref["ssdm"]["status"] == "ok", ref["ssdm"]["error"]
    for seed in seeds:
        perm = shuffled(req, seed)
        out = decider.decide(perm)
        assert out["ssdm"]["status"] == "ok"
        for qid, probs in ref["ssdm"]["served_probabilities"].items():
            other = out["ssdm"]["served_probabilities"][qid]
            assert set(other) == set(probs)
            for o, p in probs.items():
                assert abs(other[o] - p) <= tol, (seed, qid, o, p, other[o])
            if req["questions"][qid]["type"] == "choice":
                assert out["answers"][qid]["choice"] == ref["answers"][qid]["choice"]
                # wire probabilities follow the caller's (shuffled) key order
                assert list(out["answers"][qid]["probabilities"]) == list(perm["questions"][qid]["criteria"])


def test_oracle_equivariance():
    assert_equivariant(Decider(OracleBackend()), base_request())


def test_prompt_is_canonical():
    req = base_request()
    for seed in range(5):
        perm = shuffled(req, seed)
        for qid in req["questions"]:
            assert build_prompt(TEMPLATES["jrt-chatml-v1"], req, qid) == build_prompt(
                TEMPLATES["jrt-chatml-v1"], perm, qid
            )


def test_jrt_order_schema_then_state_then_reread():
    req = base_request()
    p = build_prompt(TEMPLATES["jrt-chatml-v1"], req, "large")
    i_schema, i_state, i_q = p.index("SCHEMA FIELDS:"), p.index("STATE:\n"), p.index("QUESTION (re-read")
    assert i_schema < i_state < i_q
    assert p.index("OPTION 1:") < i_state  # options precede the state
    assert p.endswith("ANSWER:")


def tiny_plugin():
    torch = pytest.importorskip("torch")
    from decide.backends.hf_readout import ModelPlugin

    torch.manual_seed(0)
    vocab = 259
    emb = torch.nn.Embedding(vocab, 16)
    rnn = torch.nn.GRU(16, 16, batch_first=True)
    head = torch.nn.Linear(16, vocab)

    def encode(text, trusted=True):
        return [b + 3 for b in text.encode("utf-8")]

    def forward(ids, n_last):
        h, _ = rnn(emb(ids))
        return head(h)[:, -n_last:, :]

    return ModelPlugin(
        family="tiny-gru", model_id="tiny", snapshot=None, encode=encode, forward=forward,
        files={"weights": "seed0", "tokenizer": "bytes+3"}, dtype="float32", device="cpu",
    )


def test_hf_readout_equivariance_tiny_cpu_lm():
    from decide.backends.hf_readout import HFReadoutBackend

    backend = HFReadoutBackend(tiny_plugin(), "jrt-plain-v1", max_batch_rows=2)
    assert_equivariant(Decider(backend), base_request(), seeds=range(3))


def test_hf_readout_logprob_math_and_plan():
    torch = pytest.importorskip("torch")
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.prompt import continuation

    plugin = tiny_plugin()
    backend = HFReadoutBackend(plugin, "jrt-plain-v1", max_batch_rows=2)
    req = base_request()
    plan = backend.plan(req)
    result = backend.score(req, lambda: None)
    assert result.inference_calls == plan.inference_calls
    assert result.total_input_tokens == plan.total_input_tokens
    # recompute one option's sequence log-prob by hand, unbatched
    tmpl = TEMPLATES["jrt-plain-v1"]
    prompt = plugin.encode(build_prompt(tmpl, req, "status"), True)
    cont = plugin.encode(continuation(tmpl, "overdue"), True)
    ids = torch.tensor([prompt + cont])
    with torch.inference_mode():
        logp = torch.log_softmax(plugin.forward(ids, ids.shape[1]).float(), -1)[0]
    manual = sum(logp[len(prompt) + i - 1, t].item() for i, t in enumerate(cont))
    assert result.logits["status"]["overdue"] == pytest.approx(manual, abs=1e-4)
    # deadline is checked between calls
    from decide.backends.base import DeadlineExceeded

    def boom():
        raise DeadlineExceeded()

    with pytest.raises(DeadlineExceeded):
        backend.score(req, boom)


def test_hf_revision_is_complete():
    from decide.backends.hf_readout import HFReadoutBackend

    rev = HFReadoutBackend(tiny_plugin(), "jrt-plain-v1").revision()
    for key in ("revision_id", "files_sha256", "prompt_template", "inference_config", "adapter_version"):
        assert key in rev
    assert "prompt=jrt-plain-v1." in rev["revision_id"]


@pytest.mark.skipif(not os.environ.get("SSDM_QWEN_DIR"), reason="set SSDM_QWEN_DIR to a Qwen3.5 snapshot")
def test_qwen_readout_equivariance_real_model():
    from decide.backends.hf_readout import HFReadoutBackend, load_qwen35

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    plugin = load_qwen35(os.environ["SSDM_QWEN_DIR"], device=device)
    decider = Decider(HFReadoutBackend(plugin, "jrt-chatml-v1"))
    assert_equivariant(decider, base_request(), seeds=range(3), tol=1e-6)

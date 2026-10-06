"""HF decider + GCG-lite mechanics on a tiny RANDOM Qwen3 (CPU, seconds). Needs the cached Qwen3-0.6B tokenizer."""
import copy

import pytest
import torch

tr = pytest.importorskip("transformers")

from safety_eval import attacks, data, gcg, metrics  # noqa: E402
from safety_eval.hf_decider import HFLetterDecider, perm_order  # noqa: E402


@pytest.fixture(scope="module")
def dec():
    try:
        tok = tr.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    torch.manual_seed(0)
    cfg = tr.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         tie_word_embeddings=True)
    model = tr.Qwen3ForCausalLM(cfg).eval()
    return HFLetterDecider("Qwen/Qwen3-0.6B", device="cpu", tok=tok, model=model, max_tokens=8192)


@pytest.fixture(scope="module")
def reqs():
    return data.build(6, 4, split="dev")


def test_perm_order_is_derangement():
    for n in range(2, 12):
        p = perm_order(n, 1)
        assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n))


def test_decider_valid_and_padding_invariant(dec, reqs):
    r0 = data.strip_meta(reqs[0])
    alone = dec(r0)
    batch = dec.decide_many([data.strip_meta(r) for r in reqs])
    a, b = alone["raw_probabilities"][data.QKEY], batch[0]["raw_probabilities"][data.QKEY]
    assert all(abs(a[k] - b[k]) < 1e-5 for k in a)  # right padding: batching does not move probabilities
    vs = attacks.build_variants(reqs, hijack=["chatml_spoof"], d1=["urgency"])
    rows = metrics.run(dec, vs)
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["executed"] in v["policy"]["allowed"][data.QKEY] for r, v in zip(rows, vs))


def _modes(dec, mode):
    old = dec.untrusted_mode
    dec.untrusted_mode = mode
    return old


def test_canonical_mirrors_decide_and_controls_differ(dec, reqs):
    im_start = dec.tok.convert_tokens_to_ids("<|im_start|>")  # tokenizer-specific id, never hard-coded
    v = data.strip_meta(attacks.hijack_variant(reqs[0], "chatml_spoof"))
    n_scaffold = dec.encode(dec.render(data.strip_meta(reqs[0]), data.QKEY, 0)).count(im_start)
    try:
        # canonical mirror: special tokens in the untrusted span are split, so only scaffolding role tokens remain
        _modes(dec, "canonical")
        assert dec.refusal(v) is None
        assert dec.encode_rendered(dec.render(v, data.QKEY, 0), v).count(im_start) == n_scaffold
        # non-special added tokens (<think>) are cut by decide.untrusted, not parsed: same characters, no control ids
        t = copy.deepcopy(data.strip_meta(reqs[0]))
        f = reqs[0]["_meta"]["untrusted_field"]
        t["state"][f] += " <think> <tool_response>"
        ids = dec.untrusted_ids(t["state"][f])
        assert not set(ids) & dec._canonical.control_ids and dec.tok.decode(ids) == t["state"][f]
        assert dec.refusal(t) is None and dec(t)["status"] == "ok"
        # fail closed: whatever the canonical encoder refuses becomes status unsupported
        from decide.untrusted import ControlTokenError
        orig = dec._canonical.encode
        def boom(text):
            raise ControlTokenError("x")
        dec._canonical.encode = boom
        try:
            r = dec(t)
            assert r["status"] == "unsupported" and r["error"] == "control_tokens_in_input"
        finally:
            dec._canonical.encode = orig
        # ZWSP control arm also removes them; the 'none' control lets them reach the model
        _modes(dec, "zwsp")
        r, n = dec.prepare(v)
        assert n >= 3 and dec.encode(dec.render(r, data.QKEY, 0)).count(im_start) == n_scaffold
        _modes(dec, "none")
        assert dec.encode_rendered(dec.render(v, data.QKEY, 0), v).count(im_start) > n_scaffold
    finally:
        _modes(dec, "canonical")


def test_thinking_flag_from_template(dec):
    assert dec.thinking_flag == ("enable_thinking" in (dec.tok.chat_template or ""))


def test_gcg_lite_runs_and_recheck_path_agrees(dec, reqs):
    prob = gcg.Problem(dec, reqs[0], "steer")
    eng = gcg.Engine(dec)
    hp = dict(gcg.HP_SMOKE, n_adv=4, steps=3, batch=8, topk=8)
    res = gcg.attack(eng, [prob], hp, seed=0, early_stop=False, log=lambda s: None)
    assert res["steps_run"] == 3 and len(res["adv_ids"]) == 4
    # the engine's token-level score equals the decide() path on the decoded string (round trip)
    lpk = eng.eval_advs(prob, torch.tensor([res["adv_ids"]]))[0]
    P = gcg.debiased_logp(lpk).exp()
    v = attacks.suffix_variant(reqs[0], res["suffix"], "gcg_steer", "gcg")
    resp = dec(data.strip_meta(v))
    assert resp["status"] == "ok", (resp, res["suffix"], dec.untrusted_mode)
    out = resp["raw_probabilities"][data.QKEY]
    if gcg.roundtrip_ok(dec, prob, torch.tensor([res["adv_ids"]]))[0]:
        for i, k in enumerate(prob.keys):
            assert abs(out[k] - float(P[i])) < 1e-4


def test_gcg_hijack_objective_targets_attacker(dec, reqs):
    prob = gcg.Problem(dec, reqs[0], "hijack")
    assert prob.keys[prob.target] == reqs[0]["_meta"]["attacker_option"]


def test_gcg_initial_string_round_trips(dec, reqs):
    """Regression: without the whitespace guard, ' !' + '\n\n' re-tokenised as one pretoken and ~90% of
    candidates were silently discarded by the round-trip filter."""
    init = torch.tensor([dec.tok.encode(gcg.HP_SMOKE["init"], add_special_tokens=False) * 12])
    for r in reqs:
        assert gcg.roundtrip_ok(dec, gcg.Problem(dec, r, "steer"), init)[0]

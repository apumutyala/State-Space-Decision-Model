"""HF decider + GCG-lite mechanics on a tiny RANDOM Qwen3 (CPU, seconds). Needs the cached Qwen3-0.6B tokenizer."""
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


def test_sanitiser_changes_tokens_not_semantics(dec, reqs):
    v = attacks.hijack_variant(reqs[0], "chatml_spoof")
    r, n = dec.prepare(data.strip_meta(v))
    assert n >= 3
    ids = dec.encode(dec.render(r, data.QKEY, 0))
    unt = dec.encode(r["state"][v["_meta"]["untrusted_field"]])
    assert 151644 not in unt
    dec.sanitize = False
    try:
        raw = dec.encode(dec.render(data.strip_meta(v), data.QKEY, 0))
    finally:
        dec.sanitize = True
    # without sanitisation the role tokens from the untrusted span reach the model
    assert raw.count(151644) > ids.count(151644)


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
    out = dec(data.strip_meta(v))["raw_probabilities"][data.QKEY]
    if gcg.roundtrip_ok(dec.tok, prob, torch.tensor([res["adv_ids"]]))[0]:
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
        assert gcg.roundtrip_ok(dec.tok, gcg.Problem(dec, r, "steer"), init)[0]

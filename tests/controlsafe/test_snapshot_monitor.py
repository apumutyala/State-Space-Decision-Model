"""G4 snapshot reuse (decisions identical to full recompute, measured savings) and G14 monitor (synthetic)."""

import math

import pytest

torch = pytest.importorskip("torch")

from controlsafe import adapters as A  # noqa: E402
from controlsafe import monitor as M  # noqa: E402
from controlsafe import snapshot as S  # noqa: E402
from controlsafe import trust_gate as TG  # noqa: E402
from heads import ByteEncoder  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402
from heads.layout import build_row  # noqa: E402

FAMS = ["rwkv7", "q9af"]
ENC = ByteEncoder()


def _segs(seed, n=6):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(3, 9, (n,), generator=g).tolist()
    return [torch.randint(4, 260, (L,), generator=g).tolist() for L in lens]


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("k", [1, 2])
def test_trie_decisions_identical_to_full_recompute(fam, k):
    m = tiny_hf_backbone(fam, seed=5).model
    run = A.Runner(m)
    segs = _segs(1)
    o = S.erasure_origins(list(range(10, 22)), segs, list(range(40, 49)), k)
    rec = S.snapshot_certify(run, S.random_readout(32, 5, seed=3), o, verify=True)
    assert rec["decisions_identical"] and rec["max_abs_logit_diff"] < 1e-4
    assert rec["n_origins"] == sum(math.comb(6, j) for j in range(1, k + 1))
    assert rec["snapshot_tokens"] < rec["baseline_tokens"] == rec["baseline_tokens_measured"]
    assert rec["max_states_held"] <= len(segs) + 1


def test_trie_with_option_continuations_and_shared_prefixes():
    m = tiny_hf_backbone("rwkv7", seed=6).model
    run = A.Runner(m)
    ev = S.TrieEvaluator(run, S.random_readout(32, 1, seed=1), conts=[[50, 51], [60], [70, 71, 72]])
    seqs = [[5, 6, 7, 8, 9], [5, 6, 7], [5, 6, 7, 8, 10, 11], [5, 6, 12], [13]]
    a = ev.evaluate(seqs)
    b = ev.full_recompute(seqs)
    for x, y in zip(a.logits, b.logits):
        assert x.shape == (3,) and torch.allclose(x, y, atol=1e-5)
    assert a.tokens < b.tokens


def _req(note):
    return {"request_id": "r", "state": {"note": note, "meta": "ticket 7"},
            "provenance": {"untrusted_fields": ["note"]},
            "questions": {"q": {"type": "choice", "instructions": "pick",
                                "criteria": {"a": "A.", "b": "B.", "c": "C."}}}}


def test_request_level_origins_match_certify_and_decisions_match():
    from safety_eval import certify as C

    r = _req("Alpha one. Bravo two!\nCharlie three. Delta four? Echo five.")
    row_ids = lambda q: build_row(q, "q", ENC).ids  # noqa: E731  (the real layout tokenisation)
    for k in (1, 2):
        o = S.request_origins(r, k, row_ids)
        receipt = C.certify(lambda q: {"status": "ok", "answers": {"q": {"type": "choice", "choice": "a",
                                       "probabilities": {"a": 1.0, "b": 0.0, "c": 0.0}}}}, r, k=k)
        assert len(o) - 1 == receipt["n_origins"]
        m = tiny_hf_backbone("q9af", seed=7).model
        rec = S.snapshot_certify(A.Runner(m), S.random_readout(32, 3, seed=0), o, verify=True)
        assert rec["decisions_identical"]
        assert rec["token_savings"] > 0.2


def test_monitor_stats_and_auroc():
    assert M.auroc([0, 1, 2], [3, 4]) == 1.0 and M.auroc([3, 4], [0, 1]) == 0.0 and M.auroc([1], [1]) == 0.5
    m = tiny_hf_backbone("rwkv7", seed=1).model
    ids = list(range(10, 40))
    st = M.span_stats(m, ids, (8, 20))
    for v in st.values():
        assert 0 <= v["retained_min"] <= v["retained_mean"] <= v["sigma_max"] + 1e-12
    with pytest.raises(ValueError):
        M.span_stats(m, ids, (0, 5))


@pytest.mark.parametrize("fam", FAMS)
def test_g14_seed_sweep_smoke(fam):
    """SMOKE test only (PR #42 review B1): detection is weak and seed-unstable, so no AUROC bar is asserted.
    Checks the sweep reports the whole distribution with a CI and a below-chance count."""
    def make(sd):
        m = tiny_hf_backbone(fam, seed=sd).model
        M.long_memory_(m, seed=sd)
        return m

    r = M.seed_sweep(make, [0, 1], n=6)
    assert r["n_seeds"] == 2 and len(r["per_seed"]) == 2
    for key in ("auroc", "auroc_min_score"):
        lo, hi = r[key]["ci95_mean_bootstrap"]
        assert 0.0 <= r[key]["min"] <= lo <= r[key]["mean"] <= hi <= r[key]["max"] <= 1.0
    assert 0 <= r["seeds_auroc_at_or_below_chance"] <= 2
    assert all(row["collapse_ratio"] >= 0 for row in r["per_seed"])


@pytest.mark.parametrize("fam", FAMS)
def test_trust_gate_protected_block_is_immune_to_collapse_trigger(fam):
    bb = tiny_hf_backbone(fam, seed=0)
    M.long_memory_(bb.model)
    g = TG.TrustGatedBackbone(bb, bridge_rank=4)
    pre, tail = list(range(20, 28)), list(range(90, 96))
    trig = M.find_trigger(bb.model, (pre, list(range(30, 38))), tail, length=4, rounds=1)
    clean = pre + list(range(100, 116)) + tail
    pois = pre + list(range(100, 106)) + trig + list(range(110, 116)) + tail
    trust_c = [True] * 8 + [False] * 16 + [True] * 6
    trust_p = [True] * 8 + [False] * 16 + [True] * 6
    with torch.no_grad():
        g.features(torch.tensor([clean]), None, torch.tensor([trust_c]))
        a = g.last_protected_final[0, -1].clone()
        g.features(torch.tensor([pois]), None, torch.tensor([trust_p]))
        b = g.last_protected_final[0, -1].clone()
    assert torch.equal(a, b)

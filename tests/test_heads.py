"""m2 heads on tiny random backbones (CPU only): layout, shapes, equivariance, attribution,
ordinal monotonicity, bounded snapshot memory, backbone-agnostic interface."""

import pytest

torch = pytest.importorskip("torch")

from decide.prompt import TEMPLATES, build_prompt  # noqa: E402
from heads import HEADS, ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import HiddenStateBackbone, tiny_backbone, tiny_hf_backbone  # noqa: E402
from heads.features import build_head_input  # noqa: E402
from heads.heads import HeadB, IntegratorBank, SnapshotFork, make_head  # noqa: E402
from heads.typed import CumulativeLink, TypedOutput  # noqa: E402
from train.data import TOY_TEMPLATE  # noqa: E402

ENC = ByteEncoder()
SMALL = {"clef_control": {"width": 16, "heads": 2, "feedforward": 32, "layers": 1, "routing_layers": 1}}


def req():
    return {
        "model": "t",
        "state": "ticket: checkout down since 9am; vendor Acme; total 1250",
        "questions": {
            "dept": {"type": "choice", "instructions": "team?",
                     "criteria": {"billing": "payments", "tech": "outages", "sales": "deals", "legal": None}},
            "down": {"type": "noul", "instructions": "is a service down?"},
            "urg": {"type": "score", "criteria": ["later", "week", "today"]},
        },
    }


def rows(layout="ordered", qids=("dept", "down", "urg"), template=TOY_TEMPLATE):
    return [build_row(req(), q, ENC, template=template, layout=layout) for q in qids]


@pytest.fixture(scope="module")
def bb():
    return tiny_backbone(seed=0, hidden=16)


def head_input(bb, layout="ordered", need_fork=True, qids=("dept",)):
    with torch.no_grad():
        return build_head_input(bb, rows(layout, qids), need_fork=need_fork)


# ------------------------------------------------------------------------- layout
def test_ordered_layout_is_byte_identical_to_m0_prompt():
    for qid in ("dept", "down", "urg"):
        r = build_row(req(), qid, ENC, template="jrt-plain-v1", layout="ordered")
        text = bytes(i - 4 for i in r.ids).decode()
        assert text == build_prompt(TEMPLATES["jrt-plain-v1"], req(), qid)


def test_spans_point_at_the_right_text():
    r = build_row(req(), "dept", ENC, template=TOY_TEMPLATE, layout="ordered")
    dec = lambda a, b: bytes(i - 4 for i in r.ids[a:b]).decode()  # noqa: E731
    assert dec(*r.state_span) == req()["state"]
    assert r.option_ids == ["billing", "legal", "sales", "tech"]  # canonical (sorted) order
    assert '"option_id":"legal"' in dec(*r.option_spans[1])
    assert dec(*r.field_span).startswith("FIELD") and "team?" in dec(*r.field_span)
    assert r.state_span[0] > r.field_span[1]  # options-first: schema before state
    assert r.answer_pos == len(r.ids) - 1


def test_symmetric_prefix_does_not_depend_on_option_order_or_text():
    a = build_row(req(), "dept", ENC, layout="symmetric")
    b = build_row(req(), "dept", ENC, layout="symmetric", option_order=[3, 1, 0, 2])
    assert a.ids == b.ids and a.option_spans == [] and len(a.option_seqs) == 4
    assert b.option_seqs[0] == a.option_seqs[3]


def test_score_levels_cannot_be_permuted():
    with pytest.raises(ValueError):
        build_row(req(), "urg", ENC, option_order=[2, 1, 0])


# ------------------------------------------------------------------------- shapes, typed
@pytest.mark.parametrize("name", sorted(HEADS))
@pytest.mark.parametrize("layout", ["ordered", "symmetric"])
def test_shapes_and_typed_distributions(bb, name, layout):
    torch.manual_seed(0)
    m = DecisionModel(bb, name, SMALL.get(name))
    with torch.no_grad():
        f = m(rows(layout))
    lp = f.log_probs
    assert lp.shape == (3, 4)
    p = lp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
    assert torch.isinf(lp[1, 2:]).all() and torch.isinf(lp[2, 3:]).all()  # noul K=2, score K=3 padded
    assert (p >= 0).all()


def test_wire_answers_through_decide_schema(bb):
    m = DecisionModel(bb, "integrator")
    ans = m.decide(req(), ENC, template=TOY_TEMPLATE)
    assert ans["dept"]["type"] == "choice" and set(ans["dept"]["probabilities"]) == {"billing", "tech", "sales", "legal"}
    assert 0 <= ans["down"]["noul"] <= 1
    assert ans["urg"]["type"] == "score" and 0 <= ans["urg"]["score"] <= 2


def test_more_than_255_options_rejected():
    t = TypedOutput(8)
    with pytest.raises(ValueError):
        t(torch.zeros(1, 256), ["choice"], [256], torch.zeros(1, 256, 8))


def test_choice_with_255_options_ok():
    t = TypedOutput(8)
    lp = t(torch.randn(1, 255), ["choice"], [255], torch.zeros(1, 255, 8))
    assert torch.isclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)


# ------------------------------------------------------------------------- equivariance
@pytest.mark.parametrize("name", sorted(HEADS))
def test_head_permutation_equivariance(bb, name):
    torch.manual_seed(1)
    head = make_head(name, 16, **SMALL.get(name, {})).eval()
    x = head_input(bb)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        z = head(x).logits
        zp = head(x.permute_options(perm)).logits
    assert torch.allclose(zp, z[:, perm], atol=1e-5), (zp, z[:, perm])


@pytest.mark.parametrize("name", ["integrator", "head_b", "snapshot", "fork"])
def test_end_to_end_equivariance_symmetric_layout(bb, name):
    """Symmetric layout + symmetric head: probabilities per option id are order-independent."""
    torch.manual_seed(2)
    m = DecisionModel(bb, name).eval()
    perm = [3, 1, 0, 2]
    r0 = build_row(req(), "dept", ENC, template=TOY_TEMPLATE, layout="symmetric")
    r1 = build_row(req(), "dept", ENC, template=TOY_TEMPLATE, layout="symmetric", option_order=perm)
    with torch.no_grad():
        p0 = dict(zip(r0.option_ids, m([r0]).log_probs[0].tolist()))
        p1 = dict(zip(r1.option_ids, m([r1]).log_probs[0].tolist()))
    for k in p0:
        assert abs(p0[k] - p1[k]) < 1e-5


def test_ordered_layout_is_not_structurally_equivariant(bb):
    """Documented caveat (CODEX_CRITICAL_COMPARISON): in the ordered prefix a causal reader encodes
    option j after options < j, so editing an EARLIER option's text changes a LATER option's
    embedding. The symmetric layout encodes each option alone, so it does not."""

    def emb(layout, first_desc):
        r = req()
        r["questions"] = {"q": {"type": "choice", "criteria": {"a1": first_desc, "b1": "outages"}}}
        with torch.no_grad():
            x = build_head_input(bb, [build_row(r, "q", ENC, template=TOY_TEMPLATE, layout=layout)])
        return x.e[0, 1]  # embedding of b1 (unchanged text)

    assert not torch.allclose(emb("ordered", "payments"), emb("ordered", "refunds and chargebacks"))
    assert torch.allclose(emb("symmetric", "payments"), emb("symmetric", "refunds and chargebacks"))


def test_request_reordering_is_invariant_in_ordered_layout(bb):
    from tests.conftest import shuffled

    m = DecisionModel(bb, "head_b").eval()
    with torch.no_grad():
        a = m.decide(req(), ENC, template=TOY_TEMPLATE)
        b = m.decide(shuffled(req(), 7), ENC, template=TOY_TEMPLATE)
    assert a == b


# ------------------------------------------------------------------------- attribution
def test_integrator_attribution_sums_to_logit(bb):
    torch.manual_seed(4)
    head = IntegratorBank(16, channels=8, t_chunk=7)  # odd chunk size exercises the chunk loop
    x = head_input(bb, need_fork=False)
    with torch.no_grad():
        out = head(x, return_attribution=True)
    recon = out.bias + out.attribution.sum(1)
    assert torch.allclose(recon, out.logits, atol=1e-5)
    off = ~x.evidence_mask
    assert (out.attribution[off] == 0).all()  # nothing booked outside the state


@pytest.mark.parametrize("agg", ["sum", "gated", "max"])
def test_snapshot_contributions_sum_to_logit(bb, agg):
    torch.manual_seed(5)
    head = SnapshotFork(16, channels=8, max_snapshots=4, aggregator=agg)
    x = head_input(bb, need_fork=False)
    with torch.no_grad():
        out = head(x, return_attribution=True)
    assert out.attribution.shape[1] == 4
    assert torch.allclose(out.bias + out.attribution.sum(1), out.logits, atol=1e-5)


def test_snapshot_head_memory_is_bounded_in_length():
    head = SnapshotFork(16, max_snapshots=8)
    for n in (5, 100, 10_000):
        m = torch.zeros(1, n + 3, dtype=torch.bool)
        m[0, 2 : 2 + n] = True
        cid = head.chunk_ids(m)
        assert int(cid.max()) < 8 and int(cid[m].min()) == 0
        if n >= 8:
            assert len(set(cid[m].tolist())) == 8
    assert head.cost(10_000, 16)["state_scalars"] == head.cost(100, 16)["state_scalars"] == 8 * 16


def test_head_b_chunked_scan_matches_streaming_reference(bb):
    torch.manual_seed(6)
    head = HeadB(16, d_state=8, t_chunk=5)
    x = head_input(bb, need_fork=False)
    with torch.no_grad():
        z = head(x).logits
        ref = head.reference_scan(x)
    assert torch.allclose(z, ref, atol=1e-5)


def test_head_b_ignores_tokens_outside_the_state(bb):
    torch.manual_seed(7)
    head = HeadB(16, d_state=8)
    x = head_input(bb, need_fork=False)
    with torch.no_grad():
        z = head(x).logits
        x.h = x.h.clone()
        x.h[~x.evidence_mask] += 5.0  # perturb features outside the evidence region
        z2 = head(x).logits
    assert torch.allclose(z, z2, atol=1e-6)


def test_clef_control_is_flagged_and_priced():
    from heads.heads import ClefCrossAttentionControl

    h = ClefCrossAttentionControl(16, **SMALL["clef_control"])
    c = h.cost(1000, 4)
    assert h.attention_free is False and c["attention_free"] is False
    assert h.cost(2000, 4)["state_scalars"] == 2 * c["state_scalars"]  # grows with T
    for name in ("integrator", "snapshot", "head_b", "fork"):
        assert make_head(name, 16).attention_free


# ------------------------------------------------------------------------- ordinal
def test_cumulative_link_thresholds_monotone_and_stochastic_ordering():
    torch.manual_seed(8)
    cl = CumulativeLink(8)
    e = torch.randn(5, 8)
    theta = cl.thresholds(e)
    assert (theta[1:] > theta[:-1]).all()
    lows = []
    torch.set_grad_enabled(False)
    for shift in torch.linspace(-3, 3, 7):
        z = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]) + shift * torch.arange(5.0)
        p = cl.log_probs(z, e).exp()
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5) and (p >= 0).all()
        lows.append(float(p[:2].sum()))
    assert all(a >= b - 1e-6 for a, b in zip(lows, lows[1:]))  # more eta -> less mass on low levels
    z = torch.randn(5)
    assert torch.allclose(cl.log_probs(z, e), cl.log_probs(z + 3.0, e), atol=1e-5)  # shift-invariant
    torch.set_grad_enabled(True)


# ------------------------------------------------------------------------- backbone-agnostic
@pytest.mark.parametrize("family", ["qwen3.5", "q9af", "mamba"])
def test_hf_backbones_through_the_same_interface(family):
    b = tiny_hf_backbone(family, hidden=32)
    m = DecisionModel(b, "integrator")
    with torch.no_grad():
        lp = m(rows()).log_probs
    assert lp.shape == (3, 4) and torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-5)
    if family == "q9af":
        assert set(b.model.config.layer_types) == {"linear_attention"}


def test_hook_capture_matches_hidden_states():
    b = tiny_hf_backbone("mamba")
    x = torch.randint(4, 200, (2, 9))
    via_hook = b.features(x)
    plain = HiddenStateBackbone(b.model, capture="hidden_states").features(x)
    assert torch.allclose(via_hook, plain)


def test_right_padding_does_not_change_valid_features(bb):
    r = rows(qids=("dept", "down"))
    with torch.no_grad():
        both = build_head_input(bb, r, need_fork=False)
        alone = build_head_input(bb, r[1:], need_fork=False)
    n = len(r[1].ids)
    assert torch.allclose(both.h[1, :n], alone.h[0, :n], atol=1e-5)

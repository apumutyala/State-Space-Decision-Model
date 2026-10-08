"""joint_record head (attention-free cross-field record mixer + lexical prior) on a tiny CPU backbone."""

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from heads.decide_backend import DecisionModelBackend  # noqa: E402
from heads.model import record_groups  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.evalcard import predict  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.losses import LossConfig  # noqa: E402
from train.trainer import Trainer, TrainConfig, item_record  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")


def req(state="ticket: checkout down since 9am; vendor Acme; total 1250"):
    return {"model": "t", "state": state, "questions": {
        "dept": {"type": "choice", "instructions": "team?",
                 "criteria": {"billing": "payments", "tech": "outages", "sales": "deals", "legal": None}},
        "down": {"type": "noul", "instructions": "is a service down?"},
        "urg": {"type": "score", "criteria": ["later", "week", "today"]}}}


def rows_of(r, qids=None):
    return [build_row(r, q, ENC, template=TOY_TEMPLATE) for q in (qids or list(r["questions"]))]


def model(head="joint_record", seed=0, fork_mode="cache"):
    return DecisionModel(tiny_backbone(seed=0, hidden=16, fork_mode=fork_mode), head, seed=seed)


def perturb(m, seed=1):
    """Give the zero-initialised joint parameters (w, lex_scale) non-zero values so the mixer is live."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.head.w.copy_(torch.randn(m.head.w.shape, generator=g))
        m.head.lex_scale.fill_(0.7)
    return m


def test_record_groups_follow_records_and_schema_order():
    a, b = rows_of(req()), rows_of(req("other state"))
    mixed = [a[2], b[0], a[0], b[1], a[1]]
    assert record_groups(mixed) == [[2, 4, 0], [1, 3]]  # first appearance; schema order dept, down, urg
    e = rows_of(req(""))  # empty state: evidence moves to the re-read; still one record
    assert record_groups(e) == [[0, 1, 2]]


def test_init_equals_the_fork_head():
    rs = rows_of(req())
    with torch.no_grad():
        j = model("joint_record").eval()(rs).log_probs
        f = model("fork").eval()(rs).log_probs
    assert torch.equal(torch.isinf(j), torch.isinf(f))
    assert torch.allclose(j[~torch.isinf(j)], f[~torch.isinf(f)], atol=1e-6)


def test_forward_backward_and_determinism():
    rs = rows_of(req())
    m1, m2 = perturb(model()), perturb(model())
    lp1 = m1(rs).log_probs
    lp2 = m2(rs).log_probs
    assert torch.equal(lp1, lp2)  # same seed -> identical parameters and outputs
    with torch.no_grad():
        assert torch.equal(m1(rs).log_probs, lp1.detach())  # re-run identical
    loss = -lp1[torch.arange(3), torch.tensor([1, 0, 2])].sum()
    loss.backward()
    for n in ("w", "lex_scale", "A.weight", "ctx.weight", "gate_f.weight", "gate_b.weight", "pay.weight",
              "base.mlp.0.weight"):
        g = dict(m1.head.named_parameters())[n].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
    # ambient-RNG independence: building a head does not depend on what ran before
    torch.manual_seed(123)
    _ = torch.randn(10)
    m3 = perturb(model())
    assert torch.equal(m3(rs).log_probs, lp1)


def test_single_question_record_is_a_per_question_score():
    m = perturb(model()).eval()
    one = rows_of(req("ticket: card declined"), ["dept"])
    other = rows_of(req())
    with torch.no_grad():
        alone = m(one).log_probs[0]
        mixed = m(other + one).log_probs[3]
        k = one[0].n_options
        assert torch.allclose(alone[:k], mixed[:k], atol=1e-5)  # no other record leaks in
        assert torch.isfinite(alone[:k]).all() and torch.allclose(alone[:k].exp().sum(), torch.tensor(1.0), atol=1e-5)
        # zero mixer weights -> exactly the per-question fork score (+ lexical prior)
        m.head.w.zero_()
        m.head.lex_scale.zero_()
        f = model("fork").eval()
        assert torch.allclose(m(one).log_probs[0, :k], f(one).log_probs[0, :k], atol=1e-6)


def test_fields_inform_each_other_within_a_record_only():
    m = perturb(model()).eval()
    r = req()
    r2 = req()
    r2["questions"]["urg"]["criteria"] = ["never", "soon", "now", "asap"]  # another field changes
    with torch.no_grad():
        a = m(rows_of(r, ["dept", "urg"])).log_probs[0]
        b = m(rows_of(r2, ["dept", "urg"])).log_probs[0]
        c = m(rows_of(r, ["dept"])).log_probs[0]
    k = 4
    assert not torch.allclose(a[:k], b[:k], atol=1e-6)  # dept's score reads the urg field (cross-field)
    assert not torch.allclose(a[:k], c[:k], atol=1e-6)


def test_eval_paths_score_the_record_jointly():
    m = perturb(model()).eval()
    r = req()
    rs = rows_of(r)
    with torch.no_grad():
        joint = m(rs).log_probs
    be = DecisionModelBackend(m, ENC, template=TOY_TEMPLATE)
    res = be.score(r, lambda: None)
    for i, row in enumerate(rs):
        got = torch.tensor([res.logits[row.qid][o] for o in row.option_ids])
        assert torch.allclose(got, joint[i, : row.n_options], atol=1e-4)
    items = items_from_records(toy_records(5, seed=0), ENC, template=TOY_TEMPLATE)
    lp, labels, _, _ = predict(m, items, batch_size=3)  # chunks of whole records (2 questions each)
    ref = torch.cat([m([i.row for i in items[j: j + 2]], shared_prefix=True).log_probs for j in range(0, 10, 2)])
    assert torch.allclose(lp[:, : ref.shape[1]], ref, atol=1e-4)


def test_trainer_steps_on_whole_records_and_learns():
    items = items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)
    m = model()
    cfg = TrainConfig(head="joint_record", lora_rank=4, lr_head=1e-2, lr_lora=1e-3, batch_size=8, max_steps=60,
                      loss=LossConfig(arm="clef"))
    t = Trainer(m, cfg, items, guard=G)
    seen = []
    orig = t._row_loss

    def spy(batch, w):
        seen.append([item_record(i) for i in batch])
        return orig(batch, w)

    t._row_loss = spy
    h = t.train()
    for b in seen:  # every step = whole records (2 questions each, 8-row budget -> 4 records)
        assert len(b) == 8 and all(b.count(k) == 2 for k in set(b))
    first = sum(x["loss"] for x in h[:8]) / 8
    last = sum(x["loss"] for x in h[-8:]) / 8
    assert last < 0.85 * first, (first, last)  # toy: fork head 2.08 -> 2.05; joint 2.14 -> 1.55
    assert float(m.head.w.abs().sum()) > 0 and float(m.head.lex_scale.abs()) > 0


def test_records_with_an_identical_prefix_stay_separate_groups():
    # two DIFFERENT records with the same schema + state (same token prefix) must not merge into one joint group
    r1, r2 = toy_records(1, seed=0)[0], toy_records(1, seed=0)[0]
    r2 = {**r2, "id": r2["id"] + "-twin", "labels": dict(r2["labels"])}
    items = items_from_records([r1, r2], ENC, template=TOY_TEMPLATE)
    rows = [it.row for it in items]
    assert rows[0].ids[: rows[0].meta["prefix_end"]] == rows[2].ids[: rows[2].meta["prefix_end"]]
    assert record_groups(rows) == [[0, 1], [2, 3]]

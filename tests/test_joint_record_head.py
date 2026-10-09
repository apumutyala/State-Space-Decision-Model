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


# ------------------------------------------------------------------ D18: K-capped record chunks, whole oversize records, v3w
def _grads_of_one_step(row_options: int, batch_size: int = 8):
    from train.trainer import Trainer as T

    items = items_from_records(toy_records(8, seed=3), ENC, template=TOY_TEMPLATE)
    m = perturb(model())
    cfg = TrainConfig(head="joint_record", lora_rank=4, lr_head=1e-2, lr_lora=1e-3, batch_size=batch_size,
                      max_steps=1, loss=LossConfig(arm="clef"), row_options=row_options)
    t = T(m, cfg, items, guard=G)
    seen, got = [], {}
    orig = t._row_loss

    def spy(batch, w):
        seen.append(([item_record(i) for i in batch], sum(i.row.n_options for i in batch), w))
        return orig(batch, w)

    def finish(total):
        got.update({n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
        got["__loss__"] = torch.tensor(total)
        return total

    t._row_loss, t._finish_step = spy, finish
    t.train_step()
    return seen, got


def test_record_chunks_are_whole_records_under_the_k_cap():
    from train.trainer import record_chunks

    items = items_from_records(toy_records(4, seed=0), ENC, template=TOY_TEMPLATE)  # 4 records x 2 questions
    k = [it.row.n_options for it in items]
    assert record_chunks(items, 0) == [items] and record_chunks(items, sum(k)) == [items]
    cap = k[0] + k[1] + k[2] + k[3]  # two records
    ch = record_chunks(items, cap)
    assert [len(c) for c in ch] == [4, 4] and sum(ch, []) == items
    for c in ch:
        assert sum(it.row.n_options for it in c) <= cap and len({item_record(i) for i in c}) == 2
    ch = record_chunks(items, 1)  # below one record: each record alone (never split)
    assert [len(c) for c in ch] == [2, 2, 2, 2] and all(len({item_record(i) for i in c}) == 1 for c in ch)


def test_joint_step_k_chunks_equal_the_single_call_step():
    one, g1 = _grads_of_one_step(0)
    items = items_from_records(toy_records(1, seed=3), ENC, template=TOY_TEMPLATE)
    rec_k = sum(it.row.n_options for it in items)
    many, g2 = _grads_of_one_step(2 * rec_k)  # 4 records of the 8-row step -> 2 chunks of 2 records
    assert len(one) == 1 and one[0][2] == 1.0
    assert len(many) == 2 and all(k <= 2 * rec_k and w == 0.5 for _, k, w in many)
    assert sum((r for r, _, _ in many), []) == one[0][0]  # same rows, same order
    assert torch.allclose(g1["__loss__"], g2["__loss__"], atol=1e-5)
    assert set(g1) == set(g2)
    for n in g1:  # records never cross a chunk, so the record mixer and every gradient are unchanged
        assert torch.allclose(g1[n], g2[n], atol=1e-5, rtol=1e-4), n


def test_record_longer_than_the_step_is_taken_whole():
    items = items_from_records(toy_records(6, seed=0), ENC, template=TOY_TEMPLATE)  # 2 questions per record
    m = model()
    cfg = TrainConfig(head="joint_record", lora_rank=4, lr_head=1e-2, lr_lora=1e-3, batch_size=1, max_steps=6,
                      loss=LossConfig(arm="clef"))
    t = Trainer(m, cfg, items, guard=G)
    batches = [t._next_record_rows(1) for _ in range(6)]
    for b in batches:  # n = 1 row, the 2-question record is still whole (cap = 2 x n)
        assert len(b) == 2 and len({item_record(i) for i in b}) == 1
    t2 = Trainer(model(), cfg, items, guard=G)
    t2.JOINT_RECORD_ROWS_FACTOR = 1  # D13 behaviour: cut at n, the rest opens the next step
    b1, b2 = t2._next_record_rows(1), t2._next_record_rows(1)
    assert len(b1) == len(b2) == 1 and item_record(b1[0]) == item_record(b2[0]) and b1[0] is not b2[0]


def test_v3w_g1v2_rows_group_by_request_and_use_option_span_lexicon():
    from decide.prompt_g1v3 import G1V3W_ID
    from heads.model import lexical_vectors, record_groups

    r = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme",
         "questions": {"dept": {"type": "choice", "instructions": "team?",
                                "criteria": {"billing": "payments", "tech": "outages", "sales": "deals"}},
                       "rel": {"type": "choice", "instructions": "Is it relevant?",
                               "criteria": {"yes": None, "no": "not relevant"}},
                       "urg": {"type": "score", "criteria": ["later", "week", "today"]}}}
    rows = [build_row(r, q, ENC, template=G1V3W_ID, layout="g1v2") for q in r["questions"]]
    assert all(row.layout == "g1v2" and row.option_spans for row in rows)
    assert len(record_groups(rows)) == 1  # serve-time rows (no meta record): one request = one group via the prefix
    other = [build_row({**r, "state": "another ticket"}, q, ENC, template=G1V3W_ID, layout="g1v2") for q in r["questions"]]
    assert len(record_groups(rows + other)) == 2
    m = perturb(model())
    lex = lexical_vectors(m.backbone, rows, max(x.n_options for x in rows))
    emb = m.backbone.model.get_input_embeddings() if hasattr(m.backbone.model, "get_input_embeddings") else next(
        x for x in m.backbone.model.modules() if isinstance(x, torch.nn.Embedding))
    a, b = rows[0].option_spans[1]
    assert b > a and torch.allclose(lex[0, 1], emb(torch.tensor(rows[0].ids[a:b])).float().mean(0), atol=1e-6)
    lp = m(rows).log_probs  # forward + backward through the mixer on v3w rows
    lp[torch.arange(3), torch.tensor([0, 1, 2])].sum().backward()
    assert m.head.w.grad is not None and torch.isfinite(m.head.w.grad).all() and m.head.w.grad.abs().sum() > 0
    be = DecisionModelBackend(m.eval(), ENC, template=G1V3W_ID, layout="g1v2")
    res = be.score(r, lambda: None)
    with torch.no_grad():
        joint = m(rows).log_probs
    for i, row in enumerate(rows):
        got = torch.tensor([res.logits[row.qid][o] for o in row.option_ids])
        assert torch.allclose(got, joint[i, : row.n_options], atol=1e-4)


# ------------------------------------------------------------------ D18 review: dev card = whole records under the joint head
def _d17_dev_card_sample(items, rows, seed=0):
    """Frozen copy of D17's dev_card_sample (4d2041c): the non-joint path must stay byte-identical to it."""
    import hashlib

    from train.trainer import item_source

    if rows <= 0 or rows >= len(items):
        return list(items)
    by = {}
    for it in items:
        key = hashlib.sha256(f"{seed}:{it.id}".encode()).hexdigest()
        by.setdefault(item_source(it.id), []).append((key, it.id, it))
    queues = [sorted(by[s], key=lambda t: (t[0], t[1])) for s in sorted(by)]
    out, r = [], 0
    while len(out) < rows:
        for q in queues:
            if r < len(q) and len(out) < rows:
                out.append(q[r][2])
        r += 1
    return out


def _dev_records(sizes_by_source):
    """Items with cohort-shaped ids m2:<source>:<record>/<qid>, record-contiguous (as items_from_records emits)."""
    from types import SimpleNamespace

    return [SimpleNamespace(id=f"m2:{s}:r{j:03d}/q{k}") for s, sizes in sorted(sizes_by_source.items())
            for j, n in enumerate(sizes) for k in range(n)]


SIZES = {"anli": [1] * 40, "acos": [3, 1, 7, 2, 12, 1, 4] * 4, "glaive": [1, 2] * 15, "mmlu": [1] * 25,
         "synthc": [7, 10, 12, 5] * 5}


def test_dev_card_non_joint_is_byte_identical_to_d17():
    import json as _json

    from train.trainer import dev_card_record, dev_card_sample

    its = _dev_records(SIZES)
    for rows, seed in ((200, 0), (37, 3), (0, 0), (len(its) + 1, 0)):
        assert [x.id for x in dev_card_sample(its, rows, seed)] == [x.id for x in _d17_dev_card_sample(its, rows, seed)]
        assert dev_card_sample(its, rows, seed, joint=False) == dev_card_sample(its, rows, seed)
    rec = dev_card_record(its, 200, 0)
    assert set(rec) == {"rows", "seed", "n", "n_pool", "design", "per_source", "ids", "ids_sha256"}
    assert _json.dumps(rec) == _json.dumps(dev_card_record(its, 200, 0, joint=False))
    assert rec["ids"] == [x.id for x in _d17_dev_card_sample(its, 200, 0)]


def test_dev_card_joint_takes_whole_contiguous_records():
    from train.evalcard import _chunks
    from train.trainer import dev_card_record, dev_card_sample, item_record, item_source

    its = _dev_records(SIZES)
    full = {}
    for x in its:
        full.setdefault(item_record(x), []).append(x.id)
    for rows, seed in ((200, 0), (60, 1), (13, 7)):
        sel = dev_card_sample(its, rows, seed, joint=True)
        ids = [x.id for x in sel]
        assert len(ids) == len(set(ids)) and rows <= len(ids) < rows + max(map(len, full.values()))
        seen, prev = [], None
        for x in sel:  # contiguous: a record never reappears after another one started
            r = item_record(x)
            if r != prev:
                assert r not in seen
                seen.append(r)
                prev = r
        for r in seen:  # complete, in the record's own item order
            assert [i for i in ids if item_record_id(i) == r] == full[r]
        assert {item_source(x.id) for x in sel} == set(SIZES)  # stratified: every source is represented
        assert [x.id for x in dev_card_sample(list(reversed(its)), rows, seed, joint=True)] != [] and \
            {item_record(x) for x in dev_card_sample(list(reversed(its)), rows, seed, joint=True)} == set(seen)
        assert dev_card_sample(its, rows, seed, joint=True) == sel  # deterministic
        model = SimpleModel()
        for ch in _chunks(model, sel, 8):  # evalcard chunks on record boundaries; a record > 8 rows goes alone
            recs = [item_record(x) for x in ch]
            assert all(len([i for i in ids if item_record_id(i) == r]) == recs.count(r) for r in set(recs))
            assert len(ch) <= 8 or len(set(recs)) == 1
    assert dev_card_sample(its, 200, 1, joint=True) != dev_card_sample(its, 200, 0, joint=True)  # the seed fixes it
    rec = dev_card_record(its, 200, 0, joint=True)
    multi = sum(1 for r in {item_record(x) for x in dev_card_sample(its, 200, 0, joint=True)} if len(full[r]) > 1)
    assert rec["n"] >= 200 and rec["n_records"] == len({item_record_id(i) for i in rec["ids"]})
    assert rec["n_multi_question_records"] == multi > 0 and rec["design"].startswith("joint_record head: WHOLE")
    assert dev_card_sample(its, 0, 0, joint=True) == its  # all rows: every record whole (already contiguous)


def item_record_id(i):
    return i.rsplit("/", 1)[0]


class SimpleModel:
    class head:  # noqa: N801
        needs_record = True


def test_trainer_dev_items_use_whole_records_for_the_joint_head_only():
    items = items_from_records(toy_records(30, seed=0), ENC, template=TOY_TEMPLATE)
    from train.trainer import dev_card_sample

    for head, joint in (("joint_record", True), ("fork", False)):
        cfg = TrainConfig(head=head, lora_rank=4, batch_size=4, max_steps=1, loss=LossConfig(arm="clef"),
                          dev_card_rows=7, dev_card_seed=2)
        t = Trainer(model(head), cfg, items[:20], eval_items=items[20:], guard=G)
        assert t.dev_items() == dev_card_sample(items[20:], 7, 2, joint)
    sel = dev_card_sample(items[20:], 7, 2, True)
    assert len(sel) == 8 and all(sum(item_record(x) == item_record(y) for y in sel) == 2 for x in sel)


def test_load_refuses_a_checkpoint_without_the_head_tensors(tmp_path):
    items = items_from_records(toy_records(4, seed=0), ENC, template=TOY_TEMPLATE)
    cfg = TrainConfig(head="joint_record", lora_rank=4, batch_size=4, max_steps=1, loss=LossConfig(arm="clef"),
                      ckpt_dir=str(tmp_path))
    t = Trainer(model(), cfg, items, guard=G)
    p = t.save()
    st = torch.load(p, map_location="cpu", weights_only=False)
    t2 = Trainer(model(), cfg, items, guard=G)
    t2.load(p)  # complete checkpoint: fine
    st["trainable"] = {k: v for k, v in st["trainable"].items() if not k.startswith("head.A.")}
    torch.save(st, tmp_path / "partial.pt")
    with pytest.raises(KeyError, match="head parameters not in checkpoint"):
        Trainer(model(), cfg, items, guard=G).load(tmp_path / "partial.pt")

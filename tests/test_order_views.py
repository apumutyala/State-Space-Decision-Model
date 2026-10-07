"""Permutation-view data semantics on synthetic records only; no model execution."""
import copy
import hashlib
import json

import pytest

from decide.schema import canonical_options
from heads.layout import ByteEncoder
from train.data import TOY_TEMPLATE, items_from_records
from train.guard import ForbiddenRow, SplitGuard, record_request
from train.lineage import ORDER_FN, items_identity
from train.order_views import attach_views, reversed_record


ENC = ByteEncoder()


def record(shape="nested", k=3, gold=0):
    ids = [f"opt-{i}" for i in range(k)]
    req = {"model": "synthetic", "state": {"answer": "description 0"}, "questions": {
        "choice": {"type": "choice", "instructions": "Which description matches the state?",
                   "criteria": {ids[i]: f"description {i}" for i in reversed(range(k))},
                   "metadata": {"tags": ["target"]}},
        "other": {"type": "choice", "instructions": "Other question", "criteria": {"a": "yes", "b": "no"}},
        "noul": {"type": "noul", "instructions": "Is it present?"},
        "score": {"type": "score", "instructions": "Rate it", "criteria": ["low", "high"]}}}
    rec = {"id": "synthetic/record", "labels": {"choice": ids[gold], "other": "a", "noul": "true", "score": "1"},
           "source": "synthetic", "metadata": {"preserve": [1, {"x": 2}]}}
    if shape == "nested":
        rec["request"] = req
    else:
        rec.update(req)
        rec["request"] = None  # PR #16 shape also supports an explicit null request
    return rec


def items(rec, layout="ordered", guard=None):
    return items_from_records([rec], ENC, layout=layout, template=TOY_TEMPLATE, guard=guard)


def attach(its, rec, layout="ordered", guard=None, keep=lambda row: True):
    return attach_views(its, {rec["id"]: rec}, ENC, layout, TOY_TEMPLATE, guard, keep)


@pytest.mark.parametrize("shape", ["nested", "flat"])
@pytest.mark.parametrize("k,gold", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3)])
def test_reversal_moves_text_and_gold_with_exact_inverse(shape, k, gold):
    rec = record(shape, k, gold)
    before = copy.deepcopy(rec)
    view, perm = reversed_record(rec, "choice")
    base_opts = canonical_options(record_request(rec)["questions"]["choice"])
    view_opts = canonical_options(record_request(view)["questions"]["choice"])
    assert perm == list(reversed(range(k)))
    assert sorted(perm) == list(range(k))
    assert [perm[perm[i]] for i in range(k)] == list(range(k))
    assert [o for o, _ in view_opts] == [o for o, _ in base_opts]
    assert [t for _, t in view_opts] == [base_opts[i][1] for i in perm]
    view_gold = next(i for i, (o, _) in enumerate(view_opts) if o == view["labels"]["choice"])
    assert perm[view_gold] == gold
    assert view_opts[view_gold][1] == base_opts[gold][1]
    # Applying the construction twice restores the request and every label exactly.
    twice, inverse = reversed_record(view, "choice")
    assert inverse == perm
    twice["id"] = rec["id"]
    assert twice == rec
    expected = copy.deepcopy(rec)
    expected["id"] += "#rev:choice"
    expected["labels"]["choice"] = view["labels"]["choice"]
    record_request(expected)["questions"]["choice"]["criteria"] = dict(view_opts)
    assert view == expected  # state, other questions/labels and metadata are unchanged
    assert rec == before
    # The returned record owns its mutable fields, including target-question metadata.
    record_request(view)["questions"]["choice"]["metadata"]["tags"].append("changed")
    view["metadata"]["preserve"].append(3)
    assert rec == before


@pytest.mark.parametrize("qid", ["noul", "score", "missing"])
def test_non_choice_questions_never_get_views(qid):
    assert reversed_record(record(), qid) is None


@pytest.mark.parametrize("criteria,gold", [({"a": "one"}, "a"), ({"a": None, "b": None}, "a"),
                                         ({"a": "one", "b": None}, "a"), ({"a": "one", "b": "two"}, "missing")])
def test_degenerate_or_id_only_choices_never_relabel(criteria, gold):
    rec = record()
    rec["request"]["questions"]["choice"]["criteria"] = criteria
    rec["labels"]["choice"] = gold
    assert reversed_record(rec, "choice") is None
    del rec["labels"]["choice"]
    assert reversed_record(rec, "choice") is None


@pytest.mark.parametrize("layout", ["ordered", "symmetric"])
@pytest.mark.parametrize("descriptions", [["same", "same"], ["same", "middle", "same"],
    [{"x": 1, "y": 2}, {"y": 2, "x": 1}], ["same", "same", "unique", "last"]])
def test_duplicate_rendered_descriptions_cannot_move_gold(layout, descriptions):
    rec = record(k=len(descriptions), gold=0)
    rec["request"]["questions"]["choice"]["criteria"] = {
        f"opt-{i}": desc for i, desc in enumerate(descriptions)}
    rec["labels"] = {"choice": "opt-0"}
    before = copy.deepcopy(rec)
    base_items = items(rec, layout)
    assert reversed_record(rec, "choice") is None
    assert attach(base_items, rec, layout) == {
        "view": "reverse", "with_view": 0, "no_view": 1, "view_filtered": 0}
    assert base_items[0].view is None and base_items[0].view_perm is None
    assert rec == before
    if len(descriptions) <= 3:
        # The old construction makes byte-identical model inputs with different labels in both layouts.
        unsafe = copy.deepcopy(rec)
        unsafe["request"]["questions"]["choice"]["criteria"] = {
            f"opt-{i}": desc for i, desc in enumerate(reversed(descriptions))}
        unsafe["labels"]["choice"] = f"opt-{len(descriptions) - 1}"
        other = items(unsafe, layout)[0]
        assert base_items[0].row.ids == other.row.ids
        assert base_items[0].row.option_seqs == other.row.option_seqs
        assert base_items[0].row.continuations == other.row.continuations
        assert base_items[0].label != other.label


def test_duplicate_descriptions_with_fixed_gold_do_not_create_contradictory_labels():
    rec = record(k=3, gold=1)
    rec["request"]["questions"]["choice"]["criteria"] = {
        "opt-0": "same", "opt-1": "middle", "opt-2": "same"}
    view, perm = reversed_record(rec, "choice")
    assert view["labels"]["choice"] == rec["labels"]["choice"]
    assert perm == [2, 1, 0]


@pytest.mark.parametrize("layout", ["ordered", "symmetric"])
def test_attachment_is_explicit_training_only_and_preserves_base_rows(layout):
    rec = record()
    train_items, eval_items = items(rec, layout), items(rec, layout)
    base_rows = copy.deepcopy([x.row for x in train_items])
    eval_before = copy.deepcopy(eval_items)
    eval_identity = items_identity(eval_items)
    assert all(x.view is None and x.view_perm is None for x in train_items + eval_items)
    stats = attach(train_items, rec, layout)
    assert stats == {"view": "reverse", "with_view": 2, "no_view": 2, "view_filtered": 0}
    assert [x.row for x in train_items] == base_rows
    assert eval_items == eval_before and items_identity(eval_items) == eval_identity
    for it in train_items:
        assert not hasattr(it.row, "view") and not hasattr(it.row, "view_perm")
        if it.row.qtype != "choice":
            assert it.view is None and it.view_perm is None
            continue
        view = it.view
        assert view is not None and view.row.option_ids == it.row.option_ids
        assert it.view_perm[view.label] == it.label
        assert view.view is None and view.view_perm is None
        assert view.teacher is None and view.axis_target is None
        qid = it.row.qid
        view_rec, _ = reversed_record(rec, qid)
        expected = items({**view_rec, "labels": {qid: view_rec["labels"][qid]}}, layout)[0]
        assert view == expected


def test_view_filter_and_guard_are_applied_before_attachment():
    rec = record()
    its = items(rec)
    stats = attach(its, rec, keep=lambda row: False)
    assert stats == {"view": "reverse", "with_view": 0, "no_view": 2, "view_filtered": 2}
    assert all(it.view is None and it.view_perm is None for it in its)
    guard = SplitGuard(ids={rec["id"] + "#rev:choice"})
    with pytest.raises(ForbiddenRow):
        attach(its, rec, guard=guard)
    assert all(it.view is None for it in its)


def test_lineage_is_legacy_exact_without_views_and_sensitive_to_view_content_and_mapping():
    rec = record()
    its = items(rec)
    h = hashlib.sha256()
    for it in its:
        r = it.row
        legacy = [it.id, it.label, r.ids, r.option_ids, r.continuations, r.option_seqs, r.qtype]
        h.update(hashlib.sha256(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest().encode())
    legacy_identity = {"n": len(its), "sha256": h.hexdigest(), "order_fn": ORDER_FN}
    assert items_identity(its) == legacy_identity
    attach(its, rec)
    attached = items_identity(its)
    assert attached != legacy_identity
    assert items_identity(copy.deepcopy(its)) == attached
    for field in ("label", "tokens", "permutation"):
        changed = copy.deepcopy(its)
        target = next(it for it in changed if it.view is not None)
        if field == "label":
            target.view.label = (target.view.label + 1) % len(target.view_perm)
        elif field == "tokens":
            target.view.row.ids[0] += 1
        else:
            target.view_perm = tuple(range(len(target.view_perm)))
        assert items_identity(changed) != attached

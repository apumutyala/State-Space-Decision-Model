"""Synthetic loader-only proofs: no backbone, weights, inference, private data or network."""
import gzip
import json
from collections import Counter

import pytest

from train.run_cell import cohort_split, load_cell_data, split_ids, to_train_record


def flat(task="task", qids=("q0", "q1")):
    return {"id": task, "state": "synthetic state", "questions": {
        q: {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": "B"}}
        for q in qids}, "labels": {q: "a" for q in qids}}


def slot(task="task", qid="q0", group="world"):
    r = flat(task, (qid,))
    return {"id": task, "group_id": group, "request": {
        "model": "synthetic", "state": r["state"], "questions": r["questions"],
        "ssdm": {"synthetic_metadata": "preserve exactly"}}, "labels": r["labels"]}


def write_rows(tmp_path, rows):
    p = tmp_path / "records.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


CFG = {"n_train_records": 2, "n_eval_records": 1, "seed": 17}

# Receipt command passes every requested node through this gate before pytest collection. Keep legacy checks
# explicit: the surrounding controlsafe module also contains random-backbone model forwards (incident L-40).
NO_MODEL_NODES = (
    "tests/test_m2_multislot_loader.py",
    "tests/test_m1_next_jobs.py::test_run_cell_data_helpers_and_pass_rule",
    "tests/test_m2_first_cell.py::test_train_set_draw_verifies_and_excludes",
    "tests/test_m2_first_cell.py::test_train_set_refuses_tampering_and_cohort_records",
    "tests/test_m2_first_cell.py::test_parallel_cell_items_equal_serial",
    "tests/controlsafe/test_trust_policy.py::test_placement_instructions_passage_is_untrusted_under_provenance_and_state_policy_fails_closed",
    "tests/controlsafe/test_trust_policy.py::test_placement_state_declares_builder_instructions_trusted_and_state_untrusted",
    "tests/controlsafe/test_trust_policy.py::test_partial_or_malformed_provenance_is_data",
    "tests/test_jobs_lint.py",
)


def guard_no_model_nodes(nodes):
    requested = list(nodes)
    if not requested or any(node not in NO_MODEL_NODES for node in requested):
        raise ValueError("no-model receipt command refuses an unallowlisted test node before collection")
    return requested


def test_no_model_receipt_guard_refuses_broad_modules_before_collection():
    assert guard_no_model_nodes(NO_MODEL_NODES) == list(NO_MODEL_NODES)
    for bad in ([], ["tests/controlsafe/test_trust_policy.py"], ["tests/test_m2_train.py"],
                [NO_MODEL_NODES[0], "tests"], ["--pyargs"], ["tests/test_m2_first_cell.py"]):
        with pytest.raises(ValueError, match="before collection"):
            guard_no_model_nodes(bad)


def test_legacy_conversion_split_order_and_identity_are_exact(tmp_path):
    rows = [flat(f"task-{i}") for i in range(7)]
    for r in rows:
        assert to_train_record(r) == {"id": r["id"], "request": {
            "model": "m2", "state": r["state"], "questions": r["questions"]}, "labels": r["labels"]}
    train, ev, tr, ei, ident = load_cell_data(CFG, write_rows(tmp_path, rows))
    assert train is ev
    assert train == {r["id"]: to_train_record(r) for r in rows}
    assert (tr, ei) == split_ids(sorted(train), 2, 1, 17)
    assert ident == {"cohort": None}
    # Matching canonical serialization also covers request key order / bytes used by existing callers.
    assert json.dumps(train) == json.dumps({r["id"]: to_train_record(r) for r in rows})


def test_nested_round_trip_keeps_request_metadata_and_task_item_identity(tmp_path):
    from heads.layout import ByteEncoder
    from train.calibration import record_ids
    from train.data import TOY_TEMPLATE
    from train.guard import SplitGuard
    from train.run_cell import cell_items

    rows = [slot(qid="q0"), slot(qid="q1")]
    rows[0]["source"] = "synthetic-only"
    recs, _, tr, ev, _ = load_cell_data({**CFG, "n_train_records": 1, "n_eval_records": 0},
                                       write_rows(tmp_path, rows))
    assert recs == {"task/q0": rows[0], "task/q1": rows[1]}
    assert set(tr) == set(recs) and ev == []
    assert to_train_record(rows[0], development=True) == rows[0]
    items = cell_items(recs, tr, {"layout": "ordered", "template": TOY_TEMPLATE, "max_k": 2},
                       ByteEncoder(), SplitGuard.synthetic_only("synthetic loader test"), 4096)
    assert [i.id for i in items] == ["task/q0", "task/q1"]
    assert record_ids([i.id for i in items]) == {"task"}
    assert [i.label for i in items] == [0, 0]


def test_all_slots_of_tasks_and_worlds_stay_together(tmp_path):
    rows = [slot(f"t{w}-{t}", f"q{q}", f"w{w}") for w in range(6) for t in range(3) for q in range(4)]
    recs, _, tr, ev, ident = load_cell_data(CFG, write_rows(tmp_path, reversed(rows)))
    assert len(tr) == 24 and len(ev) == 12
    assert ident == {"cohort": None, "split_unit": "world/task", "n_split_units": 6, "n_rows": 72}
    train_worlds = {recs[k]["group_id"] for k in tr}
    eval_worlds = {recs[k]["group_id"] for k in ev}
    assert not train_worlds & eval_worlds
    assert not {recs[k]["id"] for k in tr} & {recs[k]["id"] for k in ev}
    assert set(tr) == {k for k, r in recs.items() if r["group_id"] in train_worlds}
    assert set(ev) == {k for k, r in recs.items() if r["group_id"] in eval_worlds}
    assert load_cell_data(CFG, write_rows(tmp_path, rows))[2:] == (tr, ev, ident)


def test_ungrouped_slots_use_legacy_task_ranking_and_expand_every_row(tmp_path):
    rows = [slot(f"t{t}", f"q{q}") for t in range(6) for q in range(3)]
    for r in rows:
        del r["group_id"]
    recs, _, tr, ev, ident = load_cell_data(CFG, write_rows(tmp_path, rows))
    tt, et = split_ids([f"t{t}" for t in range(6)], 2, 1, CFG["seed"])
    assert tr == [f"{t}/q{q}" for t in tt for q in range(3)]
    assert ev == [f"{t}/q{q}" for t in et for q in range(3)]
    assert ident["split_unit"] == "task" and len(recs) == 18


def test_mixed_world_and_task_names_never_alias(tmp_path):
    rows = [slot("grouped-task", group="same-name"), flat("same-name")]
    recs, _, tr, ev, ident = load_cell_data({**CFG, "n_train_records": 1}, write_rows(tmp_path, rows))
    assert ident["n_split_units"] == 2 and len(tr) == len(ev) == 1
    assert set(tr) | set(ev) == set(recs) and not set(tr) & set(ev)


def test_52224_rows_and_5184_tasks_do_not_collapse(tmp_path):
    rows = (slot(f"t{t}", f"q{q}", f"w{t // 36}") for t in range(5184)
            for q in range(10 + (t < 384)))
    recs, _, tr, ev, ident = load_cell_data({**CFG, "n_train_records": 143}, write_rows(tmp_path, rows))
    tasks = Counter(r["id"] for r in recs.values())
    assert len(recs) == 52224 and len(tasks) == 5184
    assert Counter(tasks.values()) == {10: 4800, 11: 384}
    assert len(tr) + len(ev) == 52224 and set(tr) | set(ev) == set(recs)
    assert len({recs[k]["group_id"] for k in tr}) == 143
    assert len({recs[k]["group_id"] for k in ev}) == 1
    assert not {recs[k]["group_id"] for k in tr} & {recs[k]["group_id"] for k in ev}
    assert ident["n_split_units"] == 144


@pytest.mark.parametrize("rows,message", [
    ([flat(), flat()], "duplicate row key"),
    ([slot(), slot()], "duplicate row key"),
    ([flat(), {k: v for k, v in slot().items() if k != "group_id"}], "duplicate task/question pair"),
    ([slot(), flat("task/q0")], "duplicate row key"),
    ([slot(), slot(qid="q1", group="another")], "conflicting group identity"),
    ([slot(), {k: v for k, v in slot(qid="q1").items() if k != "group_id"}], "conflicting group identity"),
])
def test_duplicates_and_task_world_conflicts_refuse(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_cell_data(CFG, write_rows(tmp_path, rows))


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_flat_nested_expanded_item_collision_refuses_before_split(tmp_path, monkeypatch, grouped, reverse):
    import train.run_cell as rc

    flat_row = flat("task", ("slot/answer",))
    nested_row = slot("task/slot", "answer", "world-nested")
    if grouped:
        flat_row["group_id"] = "world-flat"
    else:
        del nested_row["group_id"]
    rows = [flat_row, nested_row]
    if reverse:
        rows.reverse()
    assert {f"{r['id']}/{qid}" for r in rows for qid in r["labels"]} == {"task/slot/answer"}

    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous Item identities reached split")

    monkeypatch.setattr(rc, "split_ids", forbidden)
    with pytest.raises(ValueError, match="duplicate expanded Item id"):
        load_cell_data(CFG, write_rows(tmp_path, rows))


@pytest.mark.parametrize("shape", ["grouped_flat", "mixed_flat_nested", "nested_multi_question"])
def test_every_row_on_new_development_path_requires_slash_free_qids(tmp_path, shape):
    flat_row = flat("task", ("slot/answer",))
    if shape == "grouped_flat":
        rows = [{**flat_row, "group_id": "world"}]
    elif shape == "mixed_flat_nested":
        rows = [flat_row, slot("other-task")]
    else:
        # Nested multi-question storage keeps the task id, so detect the new path from raw input shape.
        r = flat("task", ("slot/answer", "second"))
        rows = [{"id": r["id"], "request": {"model": "m2", "state": r["state"],
                                                "questions": r["questions"]}, "labels": r["labels"]}]
    with pytest.raises(ValueError, match="development question ids cannot contain"):
        load_cell_data(CFG, write_rows(tmp_path, rows))


def test_pure_legacy_slash_qids_retain_exact_existing_behavior(tmp_path):
    rows = [flat("task", ("slot/answer",)), flat("task/slot", ("answer",))]
    recs, _, tr, ev, ident = load_cell_data(CFG, write_rows(tmp_path, rows))
    assert recs == {r["id"]: to_train_record(r) for r in rows}
    assert (tr, ev) == split_ids(sorted(recs), 2, 1, CFG["seed"])
    assert ident == {"cohort": None}


@pytest.mark.parametrize("kind", ["missing_labels", "missing_question", "extra_label", "missing_state",
                                    "invalid_gold", "empty_group", "missing_id", "slash_qid", "ambiguous_shape"])
def test_malformed_rows_refuse_before_training(tmp_path, kind):
    r = slot()
    if kind == "missing_labels":
        del r["labels"]
    elif kind == "missing_question":
        r["request"]["questions"] = {}
    elif kind == "extra_label":
        r["labels"]["extra"] = "a"
    elif kind == "missing_state":
        del r["request"]["state"]
    elif kind == "invalid_gold":
        r["labels"]["q0"] = "missing-option"
    elif kind == "empty_group":
        r["group_id"] = ""
    elif kind == "missing_id":
        del r["id"]
    elif kind == "slash_qid":
        r = slot(qid="q/0")
    else:
        r["state"] = "ambiguous"
    with pytest.raises(ValueError):
        load_cell_data(CFG, write_rows(tmp_path, [r]))


@pytest.mark.parametrize("member", ["id", "labels", "questions", "q0", "group_id"])
def test_duplicate_json_members_refuse(tmp_path, member):
    text = json.dumps(slot())
    replacement = {"id": '"id": "other", "id":', "labels": '"labels": {}, "labels":',
                   "questions": '"questions": {}, "questions":', "q0": '"q0": {}, "q0":',
                   "group_id": '"group_id": "other", "group_id":'}
    text = text.replace(f'"{member}":', replacement[member], 1)
    p = tmp_path / "records.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(text + "\n")
    with pytest.raises(ValueError, match="duplicate JSON member"):
        load_cell_data(CFG, str(p))


@pytest.mark.parametrize("cfg", [{**CFG, "cohort_sha256": "not-loaded"},
                                {**CFG, "train_set": {"pointer": "not-loaded"}}])
@pytest.mark.parametrize("shape", ["nested", "flat_grouped", "ungrouped_slots"])
def test_protected_inputs_fail_before_cohort_or_draw(tmp_path, monkeypatch, cfg, shape):
    import train.run_cell as rc

    def forbidden(*args, **kwargs):
        pytest.fail("protected grouped/multislot input reached the legacy cohort splitter")

    monkeypatch.setattr(rc, "cohort_split", forbidden)
    rows = [slot(), slot(qid="q1")]
    if shape == "flat_grouped":
        rows = [{**flat(), "group_id": "world"}]
    elif shape == "ungrouped_slots":
        rows = [{k: v for k, v in r.items() if k != "group_id"} for r in rows]
    with pytest.raises(ValueError, match="matching manifest/group guard"):
        load_cell_data(cfg, write_rows(tmp_path, rows), train_dir="not-read")


def test_protected_conversion_and_direct_cohort_calls_fail_closed():
    for r in (slot(), {**flat(), "group_id": "world"}):
        with pytest.raises(ValueError, match="matching manifest/group guard"):
            to_train_record(r)  # Exact conversion function imported by train_set.draw.
    for recs in ({"task/q0": slot()}, {"task": {**flat(), "group_id": "world"}}):
        with pytest.raises(ValueError, match="matching manifest/group guard"):
            cohort_split(list(recs), {**CFG, "cohort_sha256": "not-loaded"}, recs)


@pytest.mark.parametrize("rows,message", [
    ([slot()], "matching manifest/group guard"),
    ([{**flat(), "group_id": "world"}], "matching manifest/group guard"),
    ([flat(), flat()], "duplicate train_set row key"),
])
def test_external_protected_set_rejects_before_draw_including_undrawn_rows(tmp_path, monkeypatch, rows, message):
    from train import train_set

    # A minimal synthetic pointer supplies paths only; preflight must reject before draw can verify or collapse rows.
    train_dir = tmp_path / "set"
    train_dir.mkdir()
    write_rows(train_dir, [flat("unselected-legacy"), *rows])
    ptr = tmp_path / "pointer.json"
    ptr.write_text(json.dumps({"records_uncompressed_sha256": "not-verified-yet", "manifest": "unused.gz",
                               "manifest_uncompressed_sha256": "not-read", "content_map": "unused-content.gz",
                               "content_map_uncompressed_sha256": "not-read", "n_records": 99}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported external records reached draw")

    monkeypatch.setattr(train_set, "draw", forbidden)
    cfg = {**CFG, "train_set": {"pointer": str(ptr)}, "cohort_sha256": "not-read"}
    with pytest.raises(ValueError, match=message):
        load_cell_data(cfg, write_rows(tmp_path, [flat("eval")]), str(train_dir))


def test_two_slots_and_one_legacy_record_keep_exact_item_ids(tmp_path):
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE
    from train.guard import SplitGuard
    from train.run_cell import cell_items

    rows = [slot("A", "q0"), slot("A", "q1"), flat("B", ("q0",))]
    recs, _, tr, ev, _ = load_cell_data({**CFG, "n_train_records": 2, "n_eval_records": 0},
                                       write_rows(tmp_path, rows))
    assert len(recs) == 3 and ev == []
    items = cell_items(recs, tr, {"layout": "ordered", "template": TOY_TEMPLATE, "max_k": 2},
                       ByteEncoder(), SplitGuard.synthetic_only("synthetic mixed test"), 4096)
    assert {i.id for i in items} == {"A/q0", "A/q1", "B/q0"}


def test_current_label_cannot_name_a_missing_or_different_question(tmp_path):
    r = slot(qid="previous-slot")
    r["labels"] = {"current-slot": "a"}
    with pytest.raises(ValueError, match="labels must exactly match questions"):
        load_cell_data(CFG, write_rows(tmp_path, [r]))


def test_legacy_external_preflight_keeps_draw_guards_mandatory(tmp_path, monkeypatch):
    from train import train_set

    d = tmp_path / "set"
    d.mkdir()
    write_rows(d, [flat("train")])
    ptr = tmp_path / "pointer.json"
    ptr.write_text(json.dumps({"records_uncompressed_sha256": "unused", "manifest": "unused.gz",
                               "manifest_uncompressed_sha256": "unused", "content_map": "unused-content.gz",
                               "content_map_uncompressed_sha256": "unused", "n_records": 1}), encoding="utf-8")
    cfg = {**CFG, "train_set": {"pointer": str(ptr)}, "cohort_sha256": "unused"}
    co = object()

    def reject_draw(actual_cfg, actual_dir, actual_co):
        assert actual_cfg is cfg and actual_dir == str(d) and actual_co is co
        raise train_set.TrainSetMismatch("existing draw content/manifest guard remains mandatory")

    monkeypatch.setattr(train_set, "draw", reject_draw)
    with pytest.raises(train_set.TrainSetMismatch, match="existing draw content/manifest guard"):
        load_cell_data(cfg, write_rows(tmp_path, [flat("eval")]), str(d), cohort=co)
    (d / "records.jsonl.gz").unlink()
    with pytest.raises(train_set.TrainSetMismatch, match="training-set records file missing"):
        load_cell_data(cfg, str(tmp_path / "records.jsonl.gz"), str(d), cohort=co)


def test_nested_guard_metadata_survives_and_is_enforced(tmp_path):
    from heads.layout import ByteEncoder
    from train.data import TOY_TEMPLATE
    from train.guard import ForbiddenRow, SplitGuard
    from train.run_cell import cell_items

    r = slot()
    r["_evaluation"] = {"synthetic_guard_marker": True}
    recs, _, tr, _, _ = load_cell_data({**CFG, "n_train_records": 1, "n_eval_records": 0},
                                      write_rows(tmp_path, [r]))
    with pytest.raises(ForbiddenRow):
        cell_items(recs, tr, {"layout": "ordered", "template": TOY_TEMPLATE, "max_k": 2},
                   ByteEncoder(), SplitGuard.synthetic_only("synthetic guard test"), 4096)

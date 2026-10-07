"""Amendment A2 KD recipes (R16, R17) in the CPU proxy loop, on SYNTHETIC cached features and a synthetic teacher.
CPU, $0. The real teacher / draw alignment is receipts/2026-10-07-D-teacher-distill/teacher_verify.json."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tests.test_proxy_heads import FULL, _fake_receipts, _fake_run, _select_world, synthetic  # noqa: E402
from train import proxy_loop as PL  # noqa: E402
from train import proxy_rule as R  # noqa: E402
from train.teacher import TeacherIdentityError, TeacherRow, TeacherSet  # noqa: E402

BUDGET = {**PL.TRAIN, "steps": 6, "grad_accum": 4, "warmup_steps": 2}


def recipe(rid):
    return dict(next(r for r in R.RECIPES if r["id"] == rid))


def teacher_for(reqs, every=2, draw=R.TEACHER["drawn_ids_sha256"], sha=R.TEACHER["sha256"]):
    """A teacher on every ``every``-th request: probability 0.6 on the gold option, the rest spread evenly."""
    rows = {}
    for i, r in enumerate(reqs):
        if i % every:
            continue
        qs = {}
        for q in r.questions:
            p = [0.4 / (q.k - 1)] * q.k
            p[q.label] = 0.6
            qs[q.qid] = dict(zip(q.option_ids, p))
        rows[r.rid] = TeacherRow(r.rid, "c-" + r.rid, "ok", qs)
    return TeacherSet(rows, sorted(rows), {"selection_drawn_ids_sha256": draw, "teacher_sha256": sha})


@pytest.mark.parametrize("rid", list(R.KD_RECIPES))
def test_kd_recipes_train_deterministically_with_fallback(rid):
    data = synthetic(24, 0)
    t = teacher_for(data)
    h1, i1 = PL.train_head(recipe(rid), data, seed=3, budget=BUDGET, teacher=t)
    h2, i2 = PL.train_head(recipe(rid), data, seed=3, budget=BUDGET, teacher=t)
    assert i1["loss_last"] == i2["loss_last"]
    for (n, a), (_, b) in zip(h1.state_dict().items(), h2.state_dict().items()):
        assert torch.equal(a, b), n
    kd = i1["kd"]
    assert (kd["alpha"], kd["temperature"]) == (recipe(rid)["kd"]["alpha"], recipe(rid)["kd"]["temperature"])
    assert kd["questions"] == sum(len(r.questions) for r in data)
    assert 0 < kd["with_teacher"] < kd["questions"] and kd["fallback"] == kd["questions"] - kd["with_teacher"]
    assert h1.attention_free
    # KD changes training relative to R01 (same seed, same data, same budget)
    _, i0 = PL.train_head(recipe(R.KD_CONTROL), data, seed=3, budget=BUDGET)
    assert i0["loss_last"] != i1["loss_last"] and "kd" not in i0


def test_kd_with_no_teacher_rows_equals_r01_exactly():
    """Every question falls back -> the KD recipe IS R01 (same head, same losses), proving the fallback is the arm."""
    data = synthetic(12, 1)
    empty = TeacherSet({}, [], {"selection_drawn_ids_sha256": "x", "teacher_sha256": "y"})
    h, i = PL.train_head(recipe("R16"), data, seed=5, budget=BUDGET, teacher=empty)
    h0, i0 = PL.train_head(recipe("R01"), data, seed=5, budget=BUDGET)
    assert i["kd"]["with_teacher"] == 0 and i["loss_last"] == i0["loss_last"]
    for (n, a), (_, b) in zip(h.state_dict().items(), h0.state_dict().items()):
        assert torch.equal(a, b), n


def test_kd_recipe_refusals(tmp_path):
    data = synthetic(8, 2)
    with pytest.raises(PL.OutOfOrder, match="teacher"):
        PL.train_head(recipe("R16"), data, seed=0, budget=BUDGET)
    bad = teacher_for(data, every=1)
    r0, q0 = data[0].rid, data[0].questions[0]
    bad.rows[r0].probs[q0.qid] = {f"other{k}": 1.0 / q0.k for k in range(q0.k)}  # a different option set
    with pytest.raises(TeacherIdentityError, match="options"):
        PL.train_head(recipe("R16"), data, seed=0, budget=BUDGET, teacher=bad)
    # non-KD recipes ignore a teacher entirely
    _, i = PL.train_head(recipe("R01"), data, seed=0, budget=BUDGET, teacher=bad)
    assert "kd" not in i


def _summary_file(d, name, draw):
    (d / f"summary_{name}.json").write_text(json.dumps({"train_identity": {"drawn_ids_sha256": draw}}), encoding="utf-8")


def test_cache_draw_must_be_the_teachers(tmp_path):
    t = teacher_for(synthetic(2, 0))
    arm = tmp_path / "stock"
    arm.mkdir()
    with pytest.raises(PL.OutOfOrder, match="unknown"):
        PL.check_cache_draw(tmp_path, "stock", t)
    _summary_file(arm, "train", "0" * 64)
    with pytest.raises(PL.OutOfOrder, match="teacher's draw"):
        PL.check_cache_draw(tmp_path, "stock", t)
    _summary_file(arm, "train", R.TEACHER["drawn_ids_sha256"])
    assert PL.check_cache_draw(tmp_path, "stock", t)["drawn_ids_sha256"] == R.TEACHER["drawn_ids_sha256"]


def test_run_recipe_kd_checks_the_cache_draw_before_training(tmp_path, monkeypatch):
    t = teacher_for(synthetic(2, 0))
    (tmp_path / "stock").mkdir()
    _summary_file(tmp_path / "stock", "train", "1" * 64)
    monkeypatch.setattr(PL, "load_split", lambda *a, **k: pytest.fail("trained before the draw check"))
    with pytest.raises(PL.OutOfOrder, match="teacher's draw"):
        PL.run_recipe(recipe("R17"), str(tmp_path), "rows", tmp_path, "select", R.SELECT_SUBSET, 0, None, None, 10,
                      BUDGET, teacher=t)
    with pytest.raises(PL.OutOfOrder, match="teacher"):
        PL.run_recipe(recipe("R17"), str(tmp_path), "rows", tmp_path, "select", R.SELECT_SUBSET, 0, None, None, 10, BUDGET)


def test_select_summary_refuses_stale_amendments_and_a_wrong_teacher(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "cache_identity", lambda cache, arms: {})
    monkeypatch.setattr(PL, "compare", lambda c, b, *a, **k: {"diff": {"point": 0.0, "pair": [c, b]}})
    ids = [f"R{i:02d}" for i in range(18)]
    a = tmp_path / "a"
    _fake_receipts(a, ids)
    rec = json.loads((a / "R03_s0_select" / "receipt.json").read_text(encoding="utf-8"))
    rec["amendments"] = ["A1"]  # trained before A2
    (a / "R03_s0_select" / "receipt.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="amendments"):
        PL.select_summary(a, "cache")
    b = tmp_path / "b"
    _fake_receipts(b, ids)
    rec = json.loads((b / "R16_s0_select" / "receipt.json").read_text(encoding="utf-8"))
    rec["teacher"]["teacher_sha256"] = "0" * 64
    (b / "R16_s0_select" / "receipt.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(PL.OutOfOrder, match="teacher"):
        PL.select_summary(b, "cache")
    c = tmp_path / "c"
    _fake_receipts(c, ids)
    doc = PL.select_summary(c, "cache")
    assert doc["chosen"] == "R17" and doc["amendments"] == ["A1", "A2"]
    assert set(doc["matched_controls"]) >= {"R16-R01", "R17-R01"}


def test_cli_refuses_kd_recipes_without_a_teacher(tmp_path, monkeypatch):
    (tmp_path / "stock_check.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setattr(PL, "run_recipe", lambda *a, **k: pytest.fail("ran without the teacher"))
    assert PL.main(["recipes", "--cache", "c", "--dev-rows", "d", "--out", str(tmp_path), "--only", "R01,R16"]) == 4
    assert PL.main(["recipes", "--cache", "c", "--dev-rows", "d", "--out", str(tmp_path)]) == 4
    # a teacher dir that is not the pinned export is refused before anything trains
    assert PL.main(["recipes", "--cache", "c", "--dev-rows", "d", "--out", str(tmp_path), "--only", "R16",
                    "--teacher-dir", str(tmp_path / "nope")]) == 4


def test_holdout_of_a_kd_choice_needs_the_teacher(tmp_path, monkeypatch):
    """Merged with #104: the refusal comes BEFORE holdout_attempt.json is bound (the single shot is not spent)."""
    out = tmp_path / "o"
    cache, dev, clef = _select_world(out, monkeypatch, chosen_bias="R16")
    monkeypatch.setattr(PL, "run_recipe", lambda *a, **k: pytest.fail("ran without the teacher"))
    with pytest.raises(PL.OutOfOrder, match="KD recipe"):
        PL.holdout(out, cache, dev, clef, 10)
    assert not (out / "verdict.json").exists() and not (out / "holdout_attempt.json").exists()
    assert not list(out.glob("*_holdout"))
    # with the teacher: draw checked before binding, then the attempt binds the teacher identity
    t = teacher_for(synthetic(2, 0))
    (tmp_path / "cache" / "stock").mkdir(parents=True, exist_ok=True)
    _summary_file(tmp_path / "cache" / "stock", "train", "1" * 64)
    with pytest.raises(PL.OutOfOrder, match="teacher's draw"):
        PL.holdout(out, cache, dev, clef, 10, teacher=t)
    assert not (out / "holdout_attempt.json").exists()
    _summary_file(tmp_path / "cache" / "stock", "train", R.TEACHER["drawn_ids_sha256"])
    calls = []
    monkeypatch.setattr(PL, "run_recipe", _fake_run(calls, FULL, 51.0))
    v = PL.holdout(out, cache, dev, clef, 10, teacher=t)
    assert v["chosen"] == "R16" and [c[0] for c in calls] == ["R16"] * 3
    att = json.loads((out / "holdout_attempt.json").read_text(encoding="utf-8"))
    assert att["inputs"]["teacher"]["teacher_sha256"] == R.TEACHER["sha256"]
    assert "train/teacher.py" in att["code_identity"]

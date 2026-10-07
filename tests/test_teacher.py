"""Knowledge distillation from the Clef-flash teacher: KD loss (train/losses.py), the teacher loader with strict identity
and contamination refusals (train/teacher.py), the trainer wiring, and default-off hash invariance. CPU, $0, synthetic
teacher files (the real export is checked by scripts/teacher_verify.py; receipt in receipts/2026-10-07-D-teacher-distill)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from train import teacher as T  # noqa: E402
from train.losses import LossConfig, arm_loss, decision_loss, kd_kl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


# ------------------------------------------------------------------------------------------------ KD loss
def _manual_kl(student_p, teacher_p, temperature):
    """sum_k t_T log(t_T / s_T) with x_T = x^(1/T) / sum x^(1/T), times T^2 (zeros handled by hand)."""
    def temper(p):
        w = [x ** (1.0 / temperature) if x > 0 else 0.0 for x in p]
        z = sum(w)
        return [x / z for x in w]

    s, t = temper(student_p), temper(teacher_p)
    return temperature ** 2 * sum(tk * math.log(tk / sk) for tk, sk in zip(t, s) if tk > 0)


@pytest.mark.parametrize("temperature", [1.0, 2.0, 0.5])
def test_kd_kl_matches_the_manual_kl(temperature):
    s = [[0.6, 0.3, 0.1], [0.5, 0.5, 0.0]]
    t = [[0.2, 0.7, 0.1], [0.9, 0.1, 0.0]]
    lp = torch.log(torch.tensor(s, dtype=torch.float64))
    lp[1, 2] = float("-inf")  # padded option
    got = kd_kl(lp, torch.tensor(t, dtype=torch.float64), temperature)
    want = [_manual_kl(s[0], t[0], temperature), _manual_kl(s[1][:2], t[1][:2], temperature)]
    assert torch.allclose(got, torch.tensor(want, dtype=torch.float64), atol=1e-12)
    assert float(kd_kl(lp, lp.exp(), temperature).abs().max()) < 1e-12  # KL(p || p) = 0


def test_kd_kl_with_a_teacher_zero_and_finite_gradients():
    z = torch.tensor([[0.3, -1.0, 2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    lp = torch.log_softmax(z, -1)
    lp = torch.cat([lp[:, :3], torch.full((1, 1), float("-inf"), dtype=torch.float64)], -1)  # 4th padded
    t = torch.tensor([[0.0, 0.25, 0.75, 0.0]], dtype=torch.float64)
    v = kd_kl(lp, t, 2.0)
    assert math.isfinite(float(v))
    v.sum().backward()
    assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError, match="padded"):
        kd_kl(lp.detach(), torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64))
    with pytest.raises(ValueError, match="no probability"):
        kd_kl(lp.detach(), torch.zeros((1, 4), dtype=torch.float64))


def _batch():
    lp = torch.log_softmax(torch.tensor([[1.0, 0.2, -0.5], [0.0, 0.3, 0.1], [2.0, -1.0, 0.0]]), -1)
    lp[2, 2] = float("-inf")
    lp[2] = torch.log_softmax(lp[2], -1)
    return lp, torch.tensor([0, 2, 1]), ["choice", "score", "noul"]


@pytest.mark.parametrize("alpha,temperature", [(0.5, 1.0), (0.9, 2.0), (1.0, 1.0)])
def test_kd_mix_and_fallback_per_row(alpha, temperature):
    lp, y, qt = _batch()
    teacher = torch.tensor([[0.1, 0.8, 0.1], [NAN, NAN, NAN], [0.3, 0.7, 0.0]])
    cfg = LossConfig(arm="clef", kd_alpha=alpha, kd_temperature=temperature)
    base = arm_loss(lp, y, qt, LossConfig(arm="clef"))
    kd = kd_kl(lp[[0, 2]], teacher[[0, 2]], temperature)
    want = torch.stack([alpha * kd[0] + (1 - alpha) * base[0], base[1], alpha * kd[1] + (1 - alpha) * base[2]]).mean()
    got = decision_loss(lp, y, qt, cfg, teacher)
    assert torch.allclose(got, want, atol=1e-6)
    # every row without a teacher -> exactly the clef arm
    all_nan = torch.full_like(teacher, NAN)
    assert torch.allclose(decision_loss(lp, y, qt, cfg, all_nan), decision_loss(lp, y, qt, LossConfig(arm="clef")))


def test_kd_refusals():
    lp, y, qt = _batch()
    kd = LossConfig(arm="clef", kd_alpha=0.5)
    with pytest.raises(ValueError, match="needs a teacher"):
        decision_loss(lp, y, qt, kd)
    with pytest.raises(ValueError, match="kd_alpha == 0"):
        decision_loss(lp, y, qt, LossConfig(arm="clef"), torch.full_like(lp, NAN))
    with pytest.raises(ValueError, match="partly NaN"):
        decision_loss(lp, y, qt, kd, torch.tensor([[0.5, NAN, 0.5], [NAN] * 3, [NAN] * 3]))
    with pytest.raises(ValueError, match="shape"):
        decision_loss(lp, y, qt, kd, torch.zeros((3, 2)))
    for bad in ({"kd_alpha": -0.1}, {"kd_alpha": 1.5}, {"kd_temperature": 0.0}, {"kd_temperature": float("inf")},
                {"kd_alpha": True}):
        with pytest.raises(ValueError):
            LossConfig(arm="clef", **bad)


# ------------------------------------------------------------------------------------- default-off invariance
def test_default_off_hashes_are_unchanged():
    """Literals recorded on origin/agent-main 64ece2e before the KD fields existed."""
    from train.proxy_loop import TRAIN, budget_sha
    from train.trainer import TrainConfig

    assert TrainConfig().hash() == "46a4f6f3acbac492"
    assert TrainConfig(loss=LossConfig(arm="clef")).hash() == "f0266aebd70aadc4"
    assert TrainConfig(loss=LossConfig(arm="clef", rps_weight=0.0)).hash() == "5fecd440c0003c44"
    assert budget_sha(TRAIN) == "84eab446529e33d4f1924d0098194a543aa1e0217858af821279e68d5a61e7e2"
    assert "kd_alpha" not in TrainConfig().to_json()["loss"]
    on = TrainConfig(loss=LossConfig(arm="clef", kd_alpha=0.5))
    assert on.hash() != "f0266aebd70aadc4" and on.to_json()["loss"]["kd_alpha"] == 0.5
    assert TrainConfig(loss=LossConfig(arm="clef", kd_alpha=0.5, kd_temperature=2.0)).hash() != on.hash()
    # default-off loss VALUES are unchanged too (decision_loss == the old arm-only mean)
    lp, y, qt = _batch()
    for cfg in (LossConfig(), LossConfig(arm="clef")):
        assert torch.equal(decision_loss(lp, y, qt, cfg), arm_loss(lp, y, qt, cfg).mean())


def test_proxy_non_kd_recipes_keep_their_loss_config():
    from train import proxy_loop as PL
    from train.proxy_rule import RECIPES

    for r in RECIPES:
        cfg = PL.loss_config(r)
        if r.get("kd"):
            assert cfg.kd and cfg.kd_alpha == r["kd"]["alpha"] and cfg.kd_temperature == r["kd"]["temperature"]
            assert cfg.arm == "clef" and cfg.effective_rps_weight == 1.0  # = R01's clef arm
        else:
            assert cfg is PL.LOSSES[r["loss"]] and not cfg.kd


# ------------------------------------------------------------------------------------- loader: synthetic files
def _ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _write_teacher(d: Path, rows=None, ids=None, ident_over=None, sel_over=None, raw_suffix=""):
    """A synthetic teacher dir + the matching ``expected`` dict (ids are synthetic: contamination checks are off)."""
    d.mkdir(parents=True, exist_ok=True)
    ids = ids or ["syn:a", "syn:b", "syn:c"]
    rows = rows or [
        {"id": "syn:a", "content_sha256": "ca", "status": "ok",
         "raw_probabilities": {"q1": {"x": 0.25, "y": 0.75}, "q2": {"0": 0.1, "1": 0.2, "2": 0.7}}},
        {"id": "syn:b", "content_sha256": "cb", "status": "exhausted", "raw_probabilities": None},
        {"id": "syn:c", "content_sha256": "cc", "status": "ok", "raw_probabilities": {"q": {"true": 0.9, "false": 0.1}}},
    ]
    text = "".join(json.dumps(r) + chr(10) for r in rows) + raw_suffix
    (d / "teacher-syn.jsonl").write_text(text, encoding="utf-8", newline=chr(10))
    sha = hashlib.sha256(text.encode()).hexdigest()
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    exp = {"release": "syn", "repo": "x/syn", "revision": "r1", "backend_revision": "b1", "sha256": sha,
           "drawn_ids_sha256": _ids_sha(ids), "rows": len(ids), "status_counts": counts}
    ident = {"release": "syn", "repo": "x/syn", "revision": "r1", "backend_revision": "b1", "standin": False,
             "selection_drawn_ids_sha256": _ids_sha(ids), "rows": len(ids), "status_counts": counts,
             "teacher_sha256": sha, **(ident_over or {})}
    (d / "teacher-syn.identity.json").write_text(json.dumps(ident), encoding="utf-8")
    (d / "selected-ids.txt").write_text("".join(i + chr(10) for i in ids), encoding="utf-8", newline=chr(10))
    sel = {"ok": True, "drawn_ids_sha256": _ids_sha(ids),
           "content_sha256": {i: "c" + i.split(":")[1] for i in ids}, **(sel_over or {})}
    (d / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    return exp


def test_loader_maps_by_record_question_and_option(tmp_path):
    exp = _write_teacher(tmp_path)
    ts = T.load_teacher(tmp_path, expected=exp, contamination=False)
    assert ts.stats["status_counts"] == {"ok": 2, "exhausted": 1} and ts.stats["questions_with_probs"] == 3
    assert ts.probs("syn:a", "q1", ["y", "x"]) == (0.75, 0.25)  # in the CALLER's option order
    assert ts.probs("syn:a", "q2", ["2", "0", "1"], content_sha256="ca") == (0.7, 0.1, 0.2)
    assert ts.probs("syn:b", "q1", ["x", "y"]) is None  # exhausted -> clef-arm fallback
    assert ts.probs("syn:a", "q9", ["x", "y"]) is None  # question not answered -> fallback
    assert ts.probs("other:id", "q1", ["x", "y"]) is None  # not a teacher record -> fallback
    with pytest.raises(T.TeacherIdentityError, match="options"):
        ts.probs("syn:a", "q1", ["x", "z"])
    with pytest.raises(T.TeacherIdentityError, match="options"):
        ts.probs("syn:a", "q1", ["x", "y", "z"])
    with pytest.raises(T.TeacherIdentityError, match="content"):
        ts.probs("syn:a", "q1", ["x", "y"], content_sha256="DIFFERENT")


REFUSALS = {
    "identity_release": dict(ident_over={"release": "clef"}),
    "identity_revision": dict(ident_over={"revision": "r2"}),
    "identity_backend": dict(ident_over={"backend_revision": "b2"}),
    "identity_rows": dict(ident_over={"rows": 4}),
    "identity_counts": dict(ident_over={"status_counts": {"ok": 3}}),
    "identity_draw": dict(ident_over={"selection_drawn_ids_sha256": "0" * 64}),
    "identity_file_sha": dict(ident_over={"teacher_sha256": "0" * 64}),
    "identity_standin": dict(ident_over={"standin": True}),
    "selection_not_ok": dict(sel_over={"ok": False}),
    "selection_content_mismatch": dict(sel_over={"content_sha256": {"syn:a": "ZZ", "syn:b": "cb", "syn:c": "cc"}}),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_identity_refusals(tmp_path, case):
    exp = _write_teacher(tmp_path, **REFUSALS[case])
    with pytest.raises(T.TeacherIdentityError):
        T.load_teacher(tmp_path, expected=exp, contamination=False)


def test_file_level_refusals(tmp_path):
    exp = _write_teacher(tmp_path / "a")
    with pytest.raises(T.TeacherIdentityError, match="sha256"):  # the pinned file sha differs
        T.load_teacher(tmp_path / "a", expected={**exp, "sha256": "f" * 64}, contamination=False)
    with pytest.raises(T.TeacherIdentityError, match="pinned draw"):  # a different draw pinned
        T.load_teacher(tmp_path / "a", expected={**exp, "drawn_ids_sha256": "e" * 64}, contamination=False)
    (tmp_path / "a" / "selection.json").unlink()
    with pytest.raises(T.TeacherIdentityError, match="missing"):
        T.load_teacher(tmp_path / "a", expected=exp, contamination=False)


def _row(i, status="ok", probs=None):
    return {"id": i, "content_sha256": "c" + i.split(":")[1], "status": status, "raw_probabilities": probs}


ROW_REFUSALS = {
    "duplicate_id": [_row("syn:a", probs={"q": {"x": 1.0}}), _row("syn:a", probs={"q": {"x": 1.0}}),
                     _row("syn:c", probs={"q": {"x": 1.0}})],
    "id_not_in_draw": [_row("syn:a", probs={"q": {"x": 1.0}}), _row("syn:z", probs={"q": {"x": 1.0}}),
                       _row("syn:c", probs={"q": {"x": 1.0}})],
    "missing_row": [_row("syn:a", probs={"q": {"x": 1.0}}), _row("syn:c", probs={"q": {"x": 1.0}})],
    "sum_not_one": [_row("syn:a", probs={"q": {"x": 0.5, "y": 0.4}}), _row("syn:b", "exhausted"),
                    _row("syn:c", probs={"q": {"x": 1.0}})],
    "negative": [_row("syn:a", probs={"q": {"x": 1.5, "y": -0.5}}), _row("syn:b", "exhausted"),
                 _row("syn:c", probs={"q": {"x": 1.0}})],
    "nan": [_row("syn:a", probs={"q": {"x": NAN, "y": 1.0}}), _row("syn:b", "exhausted"),
            _row("syn:c", probs={"q": {"x": 1.0}})],
    "ok_without_probs": [_row("syn:a"), _row("syn:b", "exhausted"), _row("syn:c", probs={"q": {"x": 1.0}})],
    "exhausted_with_probs": [_row("syn:a", probs={"q": {"x": 1.0}}), _row("syn:b", "exhausted", {"q": {"x": 1.0}}),
                             _row("syn:c", probs={"q": {"x": 1.0}})],
    "empty_question": [_row("syn:a", probs={"q": {}}), _row("syn:b", "exhausted"), _row("syn:c", probs={"q": {"x": 1.0}})],
}


@pytest.mark.parametrize("case", sorted(ROW_REFUSALS))
def test_row_refusals(tmp_path, case):
    exp = _write_teacher(tmp_path, rows=ROW_REFUSALS[case])
    with pytest.raises(T.TeacherIdentityError):
        T.load_teacher(tmp_path, expected=exp, contamination=False)


def test_truncated_or_malformed_file_is_refused(tmp_path):
    exp = _write_teacher(tmp_path / "t", raw_suffix='{"id": "syn:x", "status"')  # partial last line
    with pytest.raises(T.TeacherIdentityError, match="truncated"):
        T.load_teacher(tmp_path / "t", expected=exp, contamination=False)
    exp = _write_teacher(tmp_path / "m", raw_suffix="not json" + chr(10))
    with pytest.raises(T.TeacherIdentityError, match="malformed"):
        T.load_teacher(tmp_path / "m", expected=exp, contamination=False)


def test_the_pinned_identity_is_amendment_a2s():
    p = T.pinned()
    assert p["sha256"].startswith("a88b4fe0") and p["drawn_ids_sha256"].startswith("2bbbf180")
    assert p["status_counts"] == {"ok": 23458, "exhausted": 542} and p["rows"] == 24000
    with pytest.raises(T.TeacherIdentityError, match="missing"):
        T.load_teacher(ROOT / "nonexistent-teacher-dir")  # the default loader always uses the pin


# ------------------------------------------------------------------------------------- contamination
def _dev_ids(n=2):
    out = []
    with gzip.open(ROOT / "harness" / "manifests" / "dev-manifest.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["run_id"])
            if len(out) >= n:
                break
    return out


def test_contamination_refuses_dev_test_ids_and_cohort_records():
    from train.cohort_v3 import cohort_v3

    clean = ["m2:fresh:1", "m2:fresh:2"]
    content = {i: hashlib.sha256(i.encode()).hexdigest() for i in clean}
    rep = T.assert_clean(clean, content)
    assert rep["cohort_v3"] == "498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da"
    assert rep["cohort_v2"] == "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
    dev = _dev_ids(1)[0]
    with pytest.raises(T.TeacherContamination, match="DEV/TEST"):
        T.assert_clean(clean + [dev], {**content, dev: "x"})
    v3 = cohort_v3()
    held = v3.pool_ids("m2-150k", "retention")[0]
    with pytest.raises(T.TeacherContamination):  # a cohort v3 retention record, by id
        T.assert_clean(clean + [held], {**content, held: "y"})
    twin = sorted(v3.held_out_content())[0]
    with pytest.raises(T.TeacherContamination):  # a fresh id whose CONTENT is a held-out record's
        T.assert_clean(clean, {**content, "m2:fresh:2": twin})


def test_load_teacher_runs_the_contamination_check_on_a_dev_id(tmp_path):
    dev = _dev_ids(1)[0]
    ids = ["syn:a", dev]
    rows = [_row("syn:a", probs={"q": {"x": 1.0}}), {"id": dev, "content_sha256": "cd", "status": "exhausted",
                                                     "raw_probabilities": None}]
    exp = _write_teacher(tmp_path, rows=rows, ids=ids,
                         sel_over={"content_sha256": {"syn:a": "ca", dev: "cd"}})
    assert T.load_teacher(tmp_path, expected=exp, contamination=False).stats["rows"] == 2
    with pytest.raises(T.TeacherContamination):
        T.load_teacher(tmp_path, expected=exp)


# ------------------------------------------------------------------------------------- trainer wiring
def test_trainer_kd_attach_and_refusal(tmp_path):
    from heads import ByteEncoder, DecisionModel
    from heads.backbone import tiny_backbone
    from train.data import TOY_TEMPLATE, items_from_records, toy_records
    from train.guard import SplitGuard
    from train.trainer import Trainer, TrainConfig, teacher_tensor

    g = SplitGuard.synthetic_only("unit tests: toy records")
    items = items_from_records(toy_records(16, seed=0), ByteEncoder(), template=TOY_TEMPLATE)
    probs, content = {}, {}
    taught = items[: len(items) // 2]  # half the rows get a teacher, the rest fall back
    for it in taught:
        rid, qid = it.id.rsplit("/", 1)
        k = it.row.n_options
        p = [0.5 / (k - 1)] * k
        p[it.label] = 0.5
        probs.setdefault(rid, {})[qid] = dict(zip(it.row.option_ids, p))
        content[rid] = it.row.meta["content_sha256"]
    rows = {rid: T.TeacherRow(rid, content[rid], "ok", q) for rid, q in probs.items()}
    ts = T.TeacherSet(rows, sorted(rows), {"note": "synthetic"})
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=4, max_steps=3, lr_head=1e-2,
                      loss=LossConfig(arm="clef", kd_alpha=0.5, kd_temperature=2.0))

    def fresh():
        torch.manual_seed(0)
        return DecisionModel(tiny_backbone(seed=0, hidden=16), "integrator")

    with pytest.raises(ValueError, match="teacher"):
        Trainer(fresh(), cfg, items, guard=g)  # KD with no teacher attached is refused
    cov = T.attach(items, ts)
    assert cov["with_teacher"] == len(taught) and cov["fallback"] == len(items) - len(taught)
    lp = torch.zeros((2, max(i.row.n_options for i in items[:2])))
    tt = teacher_tensor(items[:2], lp)
    assert torch.isfinite(tt[0, : items[0].row.n_options]).all()
    hist = Trainer(fresh(), cfg, items, guard=g).train()
    assert len(hist) == 3 and all(math.isfinite(h["loss"]) for h in hist)
    # a teacher row whose record content differs from the item's is refused
    rid0 = next(iter(rows))
    ts.rows[rid0] = T.TeacherRow(rid0, "DIFFERENT", "ok", rows[rid0].probs)
    with pytest.raises(T.TeacherIdentityError, match="content"):
        T.attach(items, ts)

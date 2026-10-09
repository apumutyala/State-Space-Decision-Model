"""Per-readout temperature refit (jobs/m2_first_cell/readout_temperature.py) for a saved cell scored under a
NON-training readout. CPU, tiny model: the refit is deterministic, runs the in-job procedure on the SAME calibration
items, refuses any calibration content shared with the Index DEV rows, never changes an argmax, leaves the training
readout's files byte-identical, and a restart serves the persisted refit. Plus a real-data check (skipped without the
local data): the cohort v2 calibration records are disjoint from the 2,008-row subset and the full 12,104-row DEV."""
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SSDM_DATA", ROOT.parent.parent / "ssdm-data"))
V1, LAY1 = "jrt-rwkv-g1-v1", "ordered"
V3W, LAY3 = "jrt-rwkv-g1-v3w", "g1v2"


def _records():
    from train.data import toy_records

    recs = toy_records(24, seed=5)
    for r in recs:  # unique content per record (cohort roles are content-disjoint)
        r["request"]["state"] += f"; ref={r['id']}"
    return recs


@pytest.fixture(scope="module")
def ctx():
    torch = pytest.importorskip("torch")
    from heads.backbone import tiny_hf_backbone
    from heads.layout import ByteEncoder
    from heads.model import DecisionModel
    from jobs.m2_first_cell.readout_temperature import CalibrationContext
    from train import m2_gates as G
    from train.guard import SplitGuard

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    enc, guard = ByteEncoder(), SplitGuard.synthetic_only("toy records")
    spec = {"max_k": 16, "max_t": 100_000, "template": V1, "layout": LAY1}
    recs = _records()
    roles = {"calibration": recs[:8], "eval": recs[8:14], "retention": recs[14:18], "train": recs[18:24]}
    items = {k: G.decision_items(v, enc, spec, guard) for k, v in roles.items()}
    raw = G.fit_child_temperature(model, items["calibration"], {k: items[k] for k in ("eval", "retention", "train")})
    return CalibrationContext(model=model, encode=enc, guard=guard, spec=spec, calib_records=roles["calibration"],
                              raw_items=items["calibration"],
                              disjoint_items={k: items[k] for k in ("eval", "retention", "train")}, raw=raw)


def _index_rows(recs, prefix="SYNTH", choice_only=False):
    """Index-shaped rows (the kit's DEV row format) carrying these records' content (``choice_only``: the kit's
    validator scores choice / noul questions only)."""
    return [{"_evaluation": {"run_id": f"{prefix}:{i}", "group_id": f"g{i}", "catalog_id": 32},
             "state": copy.deepcopy(r["request"]["state"]),
             "questions": {k: copy.deepcopy(q) for k, q in r["request"]["questions"].items()
                           if not choice_only or q["type"] == "choice"}}
            for i, r in enumerate(recs)]


def test_refit_is_deterministic_on_the_same_calibration_items(ctx):
    from jobs.m2_first_cell import readout_temperature as RT

    subset = _index_rows(_records()[8:11])  # Index rows: content outside the calibration role
    a, ia = RT.refit(ctx, V3W, LAY3, {"index_dev_subset": subset})
    b, ib = RT.refit(ctx, V3W, LAY3, {"index_dev_subset": subset})
    assert a.w == b.w and a.fitted_on == b.fitted_on and a.proof_sha256 == b.proof_sha256
    items, info = RT.readout_items(ctx, V3W, LAY3)
    assert [i.id for i in items] == [i.id for i in ctx.raw_items]  # the SAME calibration items, re-rendered
    assert all(i.row.meta["content_sha256"] == j.row.meta["content_sha256"] for i, j in zip(items, ctx.raw_items))
    assert ia["n_items"] == len(ctx.raw_items) and not ia["dropped_by_readout_filter"]
    assert set(a.disjointness) == {"eval", "retention", "train", "index_dev_subset"}
    assert a.w != ctx.raw.w  # another readout's scores -> another temperature
    # the training readout through the same function reproduces the in-job fit exactly
    same, _ = RT.refit(ctx, V1, LAY1)
    assert same.w == ctx.raw.w and same.fitted_on == ctx.raw.fitted_on


def test_refit_refuses_calibration_content_shared_with_index_dev_rows(ctx):
    from jobs.m2_first_cell import readout_temperature as RT
    from train.calibration import CalibrationLeak

    calib = ctx.calib_records
    subset, full = _index_rows(_records()[8:11]), _index_rows(_records()[8:20], "FULL")
    _, info = RT.refit(ctx, V3W, LAY3, {"index_dev_subset": subset, "index_dev_full": full})
    cal = {i.row.meta["content_sha256"] for i in ctx.raw_items}
    for rows in (subset, full):  # content AND ids disjoint
        assert not cal & set(RT.index_content(rows))
        assert not {r["id"] for r in calib} & {r["_evaluation"]["run_id"] for r in rows}
    assert set(info["index_disjointness"]) == {"index_dev_subset", "index_dev_full"}
    calls = []
    orig = ctx.model.forward
    ctx.model.forward = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        for leak in ({"index_dev_subset": subset + _index_rows(calib[3:4], "LEAK")},
                     {"index_dev_subset": subset, "index_dev_full": full + _index_rows(calib[:1], "LEAK")}):
            with pytest.raises(CalibrationLeak):
                RT.refit(ctx, V3W, LAY3, leak)
    finally:
        del ctx.model.forward
    assert not calls  # refused BEFORE any forward over the calibration items


def test_refit_answers_keep_every_argmax_and_reproduce_the_served_answers(ctx):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from jobs.m2_first_cell import readout_temperature as RT

    st, _ = RT.refit(ctx, V3W, LAY3)
    st = st.to_serving()
    raw = ctx.raw.to_serving()
    dec = Decider(DecisionModelBackend(ctx.model, ctx.encode, template=V3W, layout=LAY3, checkpoint={"sha256": "x"}),
                  calibration=raw)
    cap = RT.Capture(dec.backend)
    for row in _index_rows(_records()[8:14]):
        resp = dec.decide({"model": "m", "state": row["state"], "questions": row["questions"]})
        assert resp["ssdm"]["status"] == "ok"
        out = RT.refit_answers(row["questions"], cap.last, raw, st, resp["answers"])["answers"]
        rawp = resp["ssdm"]["raw_probabilities"]
        for qid, q in row["questions"].items():
            lv = max(sorted(rawp[qid]), key=lambda o: rawp[qid][o])
            new = out[qid]["probabilities"]
            assert new[lv] == max(new.values())  # argmax unchanged (wire-rounded: ties allowed, never a flip)
            if q["type"] == "choice":
                assert out[qid]["choice"] == resp["answers"][qid]["choice"]
    bad = copy.deepcopy(resp["answers"])
    qid = next(iter(bad))
    bad[qid]["confidence"] = 0.123456
    with pytest.raises(AssertionError, match="re-derived"):
        RT.refit_answers(row["questions"], cap.last, raw, st, bad)


def _run(tmp_path, ctx, name, refit, monkeypatch):
    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from jobs.m2_first_cell import fulldev_eval as F
    from jobs.m2_first_cell import readout_ab as RA

    def fake_load_child(a):  # = load_child: the cell's own readout, served under the in-job fit, + its context
        base = Decider(DecisionModelBackend(ctx.model, ctx.encode, template=V1, layout=LAY1,
                                            checkpoint={"path": "ckpt-step0001439.pt", "sha256": "a" * 64}),
                       calibration=ctx.raw.to_serving())
        base.calibration_ctx = ctx
        return base, True, {"arm": "child", "checkpoint": base.backend.checkpoint, "temperature": ctx.raw.identity()}

    monkeypatch.setattr(F, "load_child", fake_load_child)
    monkeypatch.setattr(RA, "N_SUBSET", 3)
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": json.loads(
        (ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d1.json").read_text(encoding="utf-8"))}))
    rows, dev = tmp_path / "rows.jsonl.gz", tmp_path / "dev.jsonl.gz"
    for p, rs in ((rows, _index_rows(_records()[8:11], choice_only=True)),
                  (dev, _index_rows(_records()[8:20], "FULL", choice_only=True))):
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    out, ck = tmp_path / name / "out", tmp_path / name / "ck"
    argv = ["score", "--arm", "child", "--readouts", "v1,v3w", "--cell", str(cell), "--ckpt-dir", str(ck),
            "--model-dir", "x", "--records", "x", "--train-dir", "x", "--guard-readiness", "x", "--rows", str(rows),
            "--resume-dir", str(ck), "--out-dir", str(out), "--device", "cpu"]
    if refit:
        argv += ["--refit-temperature", "--dev-rows", str(dev)]
    assert RA.main(argv) == 0
    return out, ck, argv


def test_readout_ab_refit_leaves_the_control_byte_identical_and_adds_the_refit_file(tmp_path, ctx, monkeypatch):
    pytest.importorskip("decision_index")
    plain, _, _ = _run(tmp_path, ctx, "plain", False, monkeypatch)
    out, ck, argv = _run(tmp_path, ctx, "refit", True, monkeypatch)
    def strip(p):  # every byte but the measured per-row wall time
        return [{k: v for k, v in json.loads(x).items() if k != "total_wall_ms"} for x in p.read_text().splitlines()]

    for f in ("child.v1.results.jsonl", "child.v1.results_served.jsonl", "child.v3w.results.jsonl",
              "child.v3w.results_served.jsonl"):
        assert strip(out / f) == strip(plain / f), f  # same model, same rows: unchanged
    for f in ("plan-child-v1.json", "plan-child-v3w.json"):
        assert (out / f).read_bytes() == (plain / f).read_bytes(), f
    i0, i1 = (json.loads((d / "identity-child.json").read_text()) for d in (plain, out))
    assert {k: v for k, v in i1.items() if k != "temperature_refit"} == i0  # the control's identity unchanged
    assert not (out / "child.v1.results_refit.jsonl").exists() and not (plain / "child.v3w.results_refit.jsonl").exists()
    raw = [json.loads(x) for x in (out / "child.v3w.results.jsonl").read_text().splitlines()]
    srv = [json.loads(x) for x in (out / "child.v3w.results_served.jsonl").read_text().splitlines()]
    ref = [json.loads(x) for x in (out / "child.v3w.results_refit.jsonl").read_text().splitlines()]
    assert [x["run_id"] for x in ref] == [x["run_id"] for x in raw] and all(x["status"] == "ok" for x in ref)
    for a, b, c in zip(raw, srv, ref):
        for qid, ans in a["response"]["answers"].items():
            if ans["type"] == "choice":
                assert ans["choice"] == b["response"]["answers"][qid]["choice"] == c["response"]["answers"][qid]["choice"]
    assert any(b["response"] != c["response"] for b, c in zip(srv, ref))  # another temperature, same argmax
    ident = json.loads((out / "identity-child.json").read_text())
    info = ident["temperature_refit"]["v3w"]
    assert set(ident["temperature_refit"]) == {"v3w"} and not info["resumed"]
    assert set(info["index_disjointness"]) == {"index_dev_subset", "index_dev_full"}
    assert info["identity"]["id"] != ctx.raw.identity()["id"] and info["n_items"] == len(ctx.raw_items)
    saved = json.loads((ck / "refit-child-v3w.json").read_text())
    # a restart serves the persisted refit (no second calibration pass) and rewrites identical files
    before = (out / "child.v3w.results_refit.jsonl").read_bytes()
    from jobs.m2_first_cell import readout_ab as RA

    assert RA.main(argv) == 0
    assert json.loads((out / "identity-child.json").read_text())["temperature_refit"]["v3w"]["resumed"]
    assert (out / "child.v3w.results_refit.jsonl").read_bytes() == before
    assert json.loads((ck / "refit-child-v3w.json").read_text()) == saved


def test_d1_params_price_the_refit_and_the_runner_passes_it():
    from jobs.common.costs import schedule_minutes, worst_case_usd
    from jobs.m2_first_cell import readout_ab as RA

    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_readout_ab_d1.json").read_text(encoding="utf-8"))
    est = RA.estimate(p)
    assert p["refit_temperature"] and set(est["refit_minutes_per_readout"]) == {"v3w"}  # never the control
    assert 9 < est["refit_minutes_per_readout"]["v3w"] < 15 and est["ok"]
    off = {**p, "refit_temperature": False}
    assert RA.estimate(off)["minutes"]["readout_ab"] < est["minutes"]["readout_ab"]
    assert schedule_minutes(p) == p["max_duration_minutes"] and worst_case_usd(p) <= p["hard_cap_usd"]
    sh = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert "--refit-temperature --dev-rows" in sh and "refit_temperature" in sh


@pytest.mark.skipif(not (DATA / "slices/dev-rows.jsonl.gz").is_file()
                    or not (DATA / "m2/build/sample-20000-seed20261006/records.jsonl.gz").is_file(),
                    reason="local ssdm-data (DEV rows + PR #16 records) absent")
def test_real_cohort_calibration_is_disjoint_from_the_subset_and_full_dev():
    from jobs.m2_first_cell.readout_temperature import index_content, read_index_rows
    from train.cohort import Cohort, record_content_hash
    from train.run_cell import to_train_record

    cfg = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d1.json").read_text(encoding="utf-8"))
    cohort = Cohort(expected_sha256=cfg["cohort_sha256"])
    rr, _ = cohort.load_role_records(DATA / "m2/build/sample-20000-seed20261006/records.jsonl.gz", "calibration")
    assert len(rr) == 1000 and all(e.get("pool", "pr16") != "m2-150k" for e in cohort.roles["calibration"])  # not v3
    cal_ids = {r["id"] for r in rr}
    cal = {record_content_hash(to_train_record(r)) for r in rr}
    full = read_index_rows(DATA / "slices/dev-rows.jsonl.gz")
    sub_ids = set(json.loads((ROOT / "jobs/common/dev_subset_m1.json").read_text(encoding="utf-8"))["run_ids"])
    sub = [r for r in full if r["_evaluation"]["run_id"] in sub_ids]
    assert len(full) == 12_104 and len(sub) == len(sub_ids) >= 2_000
    for rows in (sub, full):
        assert not cal & set(index_content(rows))
        assert not cal_ids & {r["_evaluation"]["run_id"] for r in rows}


def test_fulldev_readout_option_refits_only_a_non_training_readout(tmp_path, ctx):
    import argparse

    from decide import Decider
    from heads.decide_backend import DecisionModelBackend
    from jobs.m2_first_cell import fulldev_eval as F

    base = Decider(DecisionModelBackend(ctx.model, ctx.encode, template=V1, layout=LAY1, checkpoint={"sha256": "a"}),
                   calibration=ctx.raw.to_serving())
    base.calibration_ctx = ctx
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": json.loads(
        (ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_readout_ab_d1.json").read_text(encoding="utf-8"))}))
    a = argparse.Namespace(cell=str(cell), readout="v1", refit_temperature=True, resume_dir=str(tmp_path / "ck"),
                           arm="child")
    rows = _index_rows(_records()[8:11], choice_only=True)
    assert F.child_readout(a, base, {"x": 1}, rows) == (base, None, {"x": 1})  # the training readout: unchanged
    a.readout, a.refit_temperature = "v3w", False
    dec, refit, ident = F.child_readout(a, base, {}, rows)
    assert refit is None and ident["readout"]["template"] == V3W and dec.calibration is base.calibration
    a.refit_temperature = True
    dec, refit, ident = F.child_readout(a, base, {}, rows)
    cap, raw, st = refit
    assert raw is base.calibration and st.fitted_on and "index_dev_rows" in dict(st.disjointness)
    assert (tmp_path / "ck" / "refit-child-v3w.json").is_file() and ident["temperature_refit"]["v3w"]["n_items"] == len(ctx.raw_items)

"""jobs/q9af_cell: params, preflight margins, pricing, export-pin refusals, hooks, summarize contracts (CPU, $0).
The runner itself is executed end to end by tests/test_q9af_cell_shell.py; the rule by tests/test_q9af_cell_rule.py;
the act-ckpt equivalence by tests/test_q9af_act_ckpt.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jobs.q9af_cell import hooks, preflight, price
from jobs.q9af_cell import rule as R

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "q9af_cell"
MODES = ("parent", "cell_r64", "cell_r256")


def _params(mode):
    return json.loads((JOB / f"job_params_{mode}.json").read_text(encoding="utf-8"))


def _cfg(r):
    return json.loads((JOB / f"q9af_cell_r{r}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", MODES)
def test_preflight_passes_with_every_cap_at_least_1_3x_its_projection(mode):
    res = preflight.check(*preflight.load(mode))
    assert res["ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
    p = _params(mode)
    assert p["margin_min_frac"] == 0.3 and p["build_timeout_minutes"] >= 60
    assert p["max_duration_minutes"] <= preflight.MAX_DURATION_CEILING == 720
    for ph, cap in p["phase_caps_minutes"].items():
        assert cap >= 1.3 * res["projections"][ph]["projected_min"], ph
    assert p["decision_rule"] == "jobs/q9af_cell/rule.py" and p["rule_commit"] == "2d18afa"
    assert p["test_env"]["transformers"] == "5.17.0"


def test_preflight_fails_a_cap_below_the_margin():
    params, cfg = preflight.load("cell_r256")
    params = {**params, "phase_caps_minutes": {**params["phase_caps_minutes"], "evals": 150}}
    params["max_duration_minutes"] = sum(params["phase_caps_minutes"].values()) + params["reserve_minutes"]
    res = preflight.check(params, cfg)
    assert not res["ok"] and not res["checks"]["evals"]["ok"]
    no_ckpt = preflight.check(*(lambda p, c: (p, {**c, "act_ckpt": False}))(*preflight.load("cell_r64")))
    assert not no_ckpt["checks"]["memory_top_rung"]["ok"]  # T = 4,096 without checkpointing is refused


def test_projection_numbers_quoted_in_job_spec():
    spec = (JOB / "JOB_SPEC.md").read_text(encoding="utf-8")
    s64, s256 = price.train_step_s(64), price.train_step_s(256)
    assert (s64["step_s"], s256["step_s"]) == (3.107, 3.205)
    assert "3.11 s" in spec and "3.21 s" in spec
    m64, m256 = price.memory_gb(64, 4096), price.memory_gb(256, 4096)
    assert (m64["total"], m256["total"]) == (24.84, 32.49) and "24.8 GB" in spec and "32.5 GB" in spec
    assert price.memory_gb(64, 4096, ckpt=False)["total"] == 146.67 and "146.7 GB" in spec
    assert (price.max_t(64, False), price.max_t(256, False)) == (1819, 1572) and "1,819" in spec and "1,572" in spec
    assert price.lora_params(64) == 159_383_552 and price.lora_params(256) == 637_534_208
    arm = price.arm_minutes(False)
    assert (arm["g12"], arm["g13"]) == (17.9, 133.9) and "133.9 min" in spec
    from jobs.common.costs import worst_case_usd

    for mode in MODES:
        assert f"${worst_case_usd(_params(mode)):.2f}" in spec


def test_measured_rates_are_job_5cmat_receipts():
    full8 = json.loads((ROOT / "jobs/full8/job_params.json").read_text(encoding="utf-8"))["measured"]
    m = price.MEASURED
    assert m["prefill_tok_s_min"] == 22144.0 and abs(full8["stock_prefill_tok_s_min"] - 22144.0) < 1  # 22,144.5
    assert m["small_call_s"] == full8["small_call_s"] and m["stage1_mfu"] == full8["stage1_mfu"]
    assert price.ASSUMPTIONS["train_mfu"] < m["stage1_mfu"]
    v3 = json.loads((ROOT / "receipts/2026-10-07-D-cohort-v3/measure.json").read_text(encoding="utf-8"))
    assert v3["cohort_sha256"] == _cfg(64)["cohort_sha256"]


def test_the_two_cells_differ_only_in_rank_so_one_parent_arm_serves_both():
    from train.m2_gates import unit_spec

    a, b = _cfg(64), _cfg(256)
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"lora_rank", "variant"}
    assert unit_spec(a) == unit_spec(b)
    assert a["act_ckpt"] is True and a["row_batch"] == 1 and a["loss_arm"] == "clef" and a["head"] == "fork"
    assert a["cohort_sha256"] == "498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da"
    assert a["train_set"]["pointer"] == "data/manifests/sample-150000-seed20261006.pointer.json"
    assert a["q9af"]["hybrid"] is False and a["teacher"] is None and a["head_from_proxy"] is None


def test_shipped_configs_are_unpinned_and_refused_until_the_export_lands(tmp_path):
    cfg = _cfg(64)
    assert cfg["q9af"]["converted_sha256"] is None  # filled by jobs.q9af_cell.pin fill when job-2xe5w lands
    ej, pt = tmp_path / "export.json", tmp_path / "g.pt"
    ej.write_text("{}", encoding="utf-8")
    pt.write_bytes(b"x")
    with pytest.raises(ValueError, match="not pinned"):
        hooks.check_export(cfg, ej, pt)
    from heads.backbone import LineageError, q9af_pin_check

    with pytest.raises(LineageError, match="not pinned"):
        q9af_pin_check(cfg["q9af"], pt)  # run_cell exit 29 path


def _export(tmp_path, attention_free=True, file="gdn_converted.pt", kept=()):
    pt = tmp_path / file
    pt.write_bytes(b"weights" + file.encode())
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    meta = {"mode": "final", "ok": True, "file": file, "sha256": sha, "attention_free": attention_free,
            "converted": [3, 7], "attention_layers_kept": list(kept),
            "base_config_sha256": _cfg(64)["q9af"]["base_config_sha256"]}
    ej = tmp_path / "export.json"
    ej.write_text(json.dumps(meta), encoding="utf-8")
    return ej, pt


def test_pin_fill_then_check_and_every_refusal(tmp_path):
    from jobs.q9af_cell import pin

    root = tmp_path / "jobdir"
    root.mkdir()
    for n in pin.CONFIGS:
        (root / n).write_text((JOB / n).read_text(encoding="utf-8"), encoding="utf-8")
    ej, pt = _export(tmp_path)
    out = pin.fill(str(ej), str(pt), root=root)
    assert all(v["attention_free"] for v in out["labels"].values())
    cfg = json.loads((root / "q9af_cell_r64.json").read_text(encoding="utf-8"))
    assert hooks.check_export(cfg, ej, pt)["label"].startswith("Q9-AF: attention-free")
    assert (root / "gdn.sha256").read_text(encoding="utf-8").count("gdn/") == 2
    # tampered export.json / weights
    with pytest.raises(ValueError, match="export.json sha256"):
        hooks.check_export({**cfg, "q9af": {**cfg["q9af"], "export_json_sha256": "0" * 64}}, ej, pt)
    pt.write_bytes(b"other")
    with pytest.raises(ValueError, match="converted checkpoint"):
        hooks.check_export(cfg, ej, pt)
    # a partial export: refused unless hybrid; then labelled NOT attention-free; hybrid on a full export refused
    (tmp_path / "p").mkdir()
    ejp, ptp = _export(tmp_path / "p", False, "gdn_partial.pt", [23])
    with pytest.raises(ValueError, match="PARTIAL"):
        pin.fill(str(ejp), str(ptp), root=root)
    lab = pin.fill(str(ejp), str(ptp), hybrid=True, root=root)["labels"]["q9af_cell_r64.json"]
    assert lab["attention_free"] is False and "NOT attention-free" in lab["label"]
    (tmp_path / "q").mkdir()
    ej2, pt2 = _export(tmp_path / "q")
    with pytest.raises(ValueError, match="drop 'hybrid'"):
        pin.fill(str(ej2), str(pt2), hybrid=True, root=root)
    bad = json.loads(ej2.read_text(encoding="utf-8"))
    ej2.write_text(json.dumps({**bad, "ok": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a successful final export"):
        pin.fill(str(ej2), str(pt2), root=root)


def test_hooks_fail_closed():
    cfg = _cfg(64)
    assert hooks.apply(cfg) == cfg
    kd = {**cfg, "teacher": {"alpha": 0.5, "temperature": 1.0, "source": "t.jsonl", "teacher_sha256": "a" * 64}}
    with pytest.raises(ValueError, match="pinned teacher"):
        hooks.apply(kd)  # a KD block must name the pinned Clef-flash teacher file (train.proxy_rule.TEACHER)
    with pytest.raises(ValueError, match="teacher_sha256"):
        hooks.KDConfig(alpha=0.5)
    with pytest.raises(ValueError, match="alpha"):
        hooks.KDConfig(alpha=0.0, teacher_sha256="a" * 64)
    hp = {**cfg, "head_from_proxy": {"head": "head_b", "head_kwargs": {}, "proxy_verdict_sha256": "b" * 64}}
    res = hooks.resolve(hp)
    assert res["head"] == "head_b" and res["head_kwargs"] == {}
    assert hooks.apply(res) == res  # a resolved, committed config passes
    with pytest.raises(ValueError, match="not attention-free"):
        hooks.resolve({**cfg, "head_from_proxy": {"head": "clef_control", "head_kwargs": {},
                                                  "proxy_verdict_sha256": "b" * 64}})
    with pytest.raises(ValueError, match="pin the proxy verdict"):
        hooks.resolve({**cfg, "head_from_proxy": {"head": "fork", "head_kwargs": {}, "proxy_verdict_sha256": "x"}})


def test_compact_result_validator_and_size():
    from jobs.q9af_cell.summarize import RESULT_MAX_BYTES, compact, valid_parent, valid_result

    ci = {"point": 50.0, "lo": 48.0, "hi": 52.0}
    col = {"child": ci, "parent": ci, "diff": ci, "benchmarks": list(range(40))}
    blk = {"child": {"index": ci}, "parent": {"index": ci}, "diff": {"index": ci},
           "columns": {c: col for c in ("in_domain", "new_domain", "new_domain_frozen")}}
    rep = {"schema": "ssdm.q9af_cell.result.v1", "rule": R.RULE_ID, "variant": "r64", "lora_rank": 64, "train_rc": 0,
           "gates_rc": 0, "references": {}, "label": {"attention_free": True, "label": "x", "sha256": "s"},
           "verdict": {"verdict": "PROMOTE", "why": []}, "gates": None, "train": {},
           "index": {ed: {"full_dev": blk, "subset_2008": blk} for ed in ("0.2.1", "0.3")}}
    out = compact(rep)
    assert valid_result(out) and len(json.dumps(out)) <= RESULT_MAX_BYTES
    assert not valid_result({**out, "verdict": "MAYBE"})
    assert not valid_result({**out, "index": {}})
    assert valid_result({**out, "verdict": "NO_MEASUREMENT", "index": {}})
    good = {"schema": "ssdm.q9af_cell.result.v1", "variant": "parent", "label": {"attention_free": True},
            "pins_for_cell_params": {"parent_arm_sha256": "a" * 64,
                                     "parent_arm_files": {"units-g12.json": "b" * 64, "units-g13.json": "c" * 64}}}
    assert valid_parent(good) and not valid_parent({**good, "pins_for_cell_params": {"parent_arm_sha256": None}})


def test_m2_gates_pool_dir_and_progress_beat(tmp_path, monkeypatch):
    from train import m2_gates as G

    assert G.pool_dirs_arg(["m2-150k=/data/m2-150k"]) == {"m2-150k": "/data/m2-150k"}
    with pytest.raises(ValueError):
        G.pool_dirs_arg(["m2-150k"])
    f = lambda req: {"ok": req}  # noqa: E731
    assert G.beat_wrap(f) is f  # unset: unchanged (every test, every legacy runner)
    beat = tmp_path / "beat"
    monkeypatch.setenv("SSDM_PROGRESS_BEAT", str(beat))
    g = G.beat_wrap(f)
    assert g(1) == {"ok": 1} and beat.exists()


def test_frozen_family_check_counts_and_refuses():
    from jobs.q9af_cell.frozen import check

    def chk(r):
        if r.get("bad"):
            raise ValueError("targets frozen family")
        return r

    ok = check([{"id": "a"}, {"id": "b"}], chk)
    assert ok["pass"] and ok["n_checked"] == 2
    bad = check([{"id": "a"}, {"id": "b", "bad": True}], chk)
    assert not bad["pass"] and bad["hits"][0]["id"] == "b"
    assert not check([], chk)["pass"]  # nothing checked is not a pass


def test_runner_maps_every_documented_exit_and_runs_the_preflight():
    sh = (JOB / "q9af_cell_job.sh").read_text(encoding="utf-8")
    assert "python -m jobs.q9af_cell.preflight --mode" in sh
    for code in ("fail 10", "fail 33", "fail 34", "fail 35", "fail 12", "fail 20", "fail 7", "fail 9"):
        assert code in sh, code
    assert "--anchor-arm" in sh and "--pool-dir" in sh and "--train-dir" in sh
    spec = (JOB / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert "2d18afa" in spec and "57.35" in spec and "new_domain_frozen" in spec


# ------------------------------------------------------------------------------------------------ KD (PR #116 wiring)
#: TrainConfig hashes of the default cells BEFORE the KD wiring (computed at 2fad27f, pre-merge of b511643): turning
#: KD support on must not change them (LossConfig.identity drops the KD fields at their defaults).
DEFAULT_HASHES = {64: "cea41d9610c90faf", 256: "90d6e79a70c23165"}


def test_default_cell_hashes_are_unchanged_by_the_kd_wiring():
    from train.run_cell import kd_fields, train_config

    for r, h in DEFAULT_HASHES.items():
        cfg = _cfg(r)
        assert cfg["teacher"] is None and kd_fields(cfg) == {}
        tc = train_config(cfg, cfg["max_steps"])
        assert tc.hash() == h and not tc.loss.kd
        assert "kd_alpha" not in tc.to_json()["loss"]


def test_kd_variant_is_r64_plus_the_pinned_teacher_and_is_not_launchable():
    from jobs.common import launch
    from train.run_cell import train_config
    from train.teacher import pinned

    kd, base = json.loads((JOB / "q9af_cell_r64_kd.json").read_text(encoding="utf-8")), _cfg(64)
    assert {k for k in set(kd) | set(base) if kd.get(k) != base.get(k)} == {"variant", "variant_note", "teacher",
                                                                            "teacher_note"}
    t = kd["teacher"]
    assert (t["alpha"], t["temperature"]) == (0.5, 1.0)
    assert t["teacher_sha256"] == pinned()["sha256"] == "a88b4fe0f6a79a8fc48bdea51391524bd4f5da885705461a543becf81c81e7f7"
    assert "art-ggrs8" in t["source"]
    tc = train_config(kd, kd["max_steps"])
    assert tc.loss.kd and (tc.loss.kd_alpha, tc.loss.kd_temperature) == (0.5, 1.0)
    assert tc.hash() not in DEFAULT_HASHES.values()
    assert hooks.apply(kd) == kd and hooks.kd_arm(kd).alpha == 0.5
    with pytest.raises(ValueError, match="pinned teacher"):
        hooks.kd_arm({**kd, "teacher": {**t, "teacher_sha256": "0" * 64}})
    # committed but unlaunched until R16/R17 report: no params file -> launch.py refuses the mode
    assert not (JOB / "job_params_cell_r64_kd.json").exists()
    files = {p.relative_to(ROOT).as_posix(): (b"", 0o644) for p in JOB.glob("job_params_*.json")}
    with pytest.raises(launch.LaunchError):
        launch.params_rel(files, "q9af_cell", "cell_r64_kd")


def _write_teacher(d: Path, rows: list, ids: list, content: dict) -> dict:
    """A synthetic teacher export in train.teacher's file format (as tests/test_teacher.py::_write_teacher, for ids
    of any shape) + the matching ``expected`` identity; synthetic ids, so contamination checks are off."""
    from train.teacher import ids_sha256

    d.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + chr(10) for r in rows)
    (d / "teacher-syn.jsonl").write_text(text, encoding="utf-8", newline=chr(10))
    sha = hashlib.sha256(text.encode()).hexdigest()
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    exp = {"release": "syn", "repo": "x/syn", "revision": "r1", "backend_revision": "b1", "sha256": sha,
           "drawn_ids_sha256": ids_sha256(ids), "rows": len(ids), "status_counts": counts}
    ident = {**{k: exp[k] for k in ("release", "repo", "revision", "backend_revision", "rows", "status_counts")},
             "standin": False, "selection_drawn_ids_sha256": ids_sha256(ids), "teacher_sha256": sha}
    (d / "teacher-syn.identity.json").write_text(json.dumps(ident), encoding="utf-8")
    (d / "selected-ids.txt").write_text("".join(i + chr(10) for i in ids), encoding="utf-8", newline=chr(10))
    (d / "selection.json").write_text(json.dumps({"ok": True, "drawn_ids_sha256": ids_sha256(ids),
                                                  "content_sha256": content}), encoding="utf-8")
    return exp


@pytest.mark.requires_transformers("5.17.0")
def test_attach_teacher_and_a_kd_train_step_on_the_q9af_standin_with_act_ckpt(tmp_path):
    """train.run_cell.attach_teacher (the path run_cell and m2_gates take) on a synthetic teacher, then real Trainer
    steps with KD under Q9-AF block checkpointing: finite, and different from the same steps without KD."""
    import math

    import torch

    from heads import ByteEncoder, DecisionModel
    from tests.test_q9af_act_ckpt import build
    from train.data import TOY_TEMPLATE, items_from_records, toy_records
    from train.guard import SplitGuard
    from train.run_cell import attach_teacher, train_config
    from train.teacher import TeacherIdentityError, ids_sha256
    from train.trainer import Trainer, with_execution

    items = items_from_records(toy_records(12, seed=0), ByteEncoder(), template=TOY_TEMPLATE)
    ids = list(dict.fromkeys(it.id.rsplit("/", 1)[0] for it in items))
    probs, content = {}, {}
    for it in items:
        rid, qid = it.id.rsplit("/", 1)
        k = it.row.n_options
        p = [0.4 / (k - 1)] * k
        p[it.label] = 0.6
        probs.setdefault(rid, {})[qid] = dict(zip(it.row.option_ids, p))
        content[rid] = it.row.meta["content_sha256"]
    rows = [{"id": r, "content_sha256": content[r], "status": "ok", "raw_probabilities": probs[r]} for r in ids]
    exp = _write_teacher(tmp_path, rows, ids, content)
    cfg = {**json.loads((JOB / "q9af_cell_r64_kd.json").read_text(encoding="utf-8")), "lora_rank": 4, "max_steps": 2,
           "batch_size": 1, "grad_accum": 4, "head_kwargs": {}}
    cfg["teacher"] = {**cfg["teacher"], "teacher_sha256": exp["sha256"]}
    with pytest.raises(ValueError, match="teacher's draw"):
        attach_teacher(cfg, items, tmp_path, ids[::-1], expected=exp, contamination=False)
    with pytest.raises(ValueError, match="--teacher-dir"):
        attach_teacher(cfg, items, None, ids, expected=exp, contamination=False)
    with pytest.raises(ValueError, match="no teacher block"):
        attach_teacher(_cfg(64), items, tmp_path, ids)
    with pytest.raises(ValueError, match="pinned teacher"):
        attach_teacher({**cfg, "teacher": {**cfg["teacher"], "teacher_sha256": "0" * 64}}, items, tmp_path, ids,
                       expected=exp, contamination=False)
    cov = attach_teacher(cfg, items, tmp_path, ids, expected=exp, contamination=False)
    assert cov["with_teacher"] == len(items) and cov["fallback"] == 0
    assert ids_sha256(ids) == cov["drawn_ids_sha256"]
    g = SplitGuard.synthetic_only("q9af_cell KD unit test: toy records")
    losses = {}
    for kd in (True, False):
        c = cfg if kd else {**cfg, "teacher": None}
        torch.manual_seed(0)
        model = DecisionModel(build("q9af", seed=0), "fork", {}, pad_id=0, seed=0)
        tc = with_execution(train_config(c, 2), c)
        assert tc.act_ckpt and tc.loss.kd == kd
        tr = Trainer(model, tc, items, guard=g)
        losses[kd] = [tr.train_step() for _ in range(2)]
    assert all(math.isfinite(x) for v in losses.values() for x in v) and losses[True] != losses[False]
    rid0 = ids[0]
    bad = [dict(r, content_sha256="DIFFERENT") if r["id"] == rid0 else r for r in rows]
    exp2 = _write_teacher(tmp_path / "bad", bad, ids, {**content, rid0: "DIFFERENT"})
    with pytest.raises(TeacherIdentityError, match="content"):
        attach_teacher({**cfg, "teacher": {**cfg["teacher"], "teacher_sha256": exp2["sha256"]}}, items, tmp_path / "bad",
                       ids, expected=exp2, contamination=False)

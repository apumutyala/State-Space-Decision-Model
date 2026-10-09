"""Mode safety_eval_d13_v3w (jobs/m2_first_cell/safety_battery.py) on CPU: both batteries scored by stand-in deciders
(safety_eval.standin.KeywordDecider, not a model; numbers mean nothing) on a mini cohort, the report + validator, the
rows feeding ``baselines.clef_heldout_v1 compare``, resume / identity refusals, the production unit-set pins, pricing
and the runner wiring. The real arm loaders (fulldev_eval.load_child / score_parent.parent_decider) are the screen's and
the H1 parent's; here they are replaced by ``score(a, loader=...)``."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.m2_first_cell import safety_battery as S

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
MODE = "safety_eval_d13_v3w"
N_BOOT = 100
CKPT_SHA = "61c19d162acb5e97183361ffd839c5e7f5d6978e40da01063a16ec196a657d7b"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A mini cohort v2 lookalike: 6 steer_dev requests (every one is in hijack_subset at 8 per cluster)."""
    from train import cohort as C

    d = tmp_path_factory.mktemp("sb")
    steer = C.steer_requests(6, 2)
    header = {"cohort": "test-mini-safety", "sizes": {"dev": 0, "calibration": 0, "retention": 0, "steer_dev": 6,
                                                      "steer_gate": 2}, "sizing": None}
    lines = [json.dumps({"header": header}, sort_keys=True)]
    for role in ("steer_dev", "steer_gate"):
        for e in steer[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    p = d / "cohort.jsonl.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(text.encode())
    return {"dir": d, "cohort": str(p), "sha": hashlib.sha256(text.encode()).hexdigest()}


def _loader(gullibility: float, arm: str, readout: str = "v3w"):
    from safety_eval.standin import KeywordDecider

    def load(a):
        ident = ({"arm": "child", "checkpoint": {"path": "trainable-step0001600.safetensors", "step": 1600,
                                                  "sha256": CKPT_SHA},
                  "readout": {"name": readout, "template": "jrt-rwkv-g1-v3w", "layout": "g1v2", "native": False}}
                 if arm == "child" else {"arm": "parent", "parent": {"parent": "rwkv7"},
                                         "readout": {"name": "stock", "template": "jrt-rwkv-g1-v1"}})
        return KeywordDecider(gullibility=gullibility), ident
    return load


def _score_args(world, tmp, arm, batteries, **kw):
    return SimpleNamespace(arm=arm, batteries=batteries, cohort=world["cohort"], cohort_sha256=world["sha"],
                           resume_dir=str(tmp / "ck"), out_dir=str(tmp / "out"), progress=str(tmp / "p.jsonl"),
                           readout="v3w", **kw)


def _cell(drawn_hits=0, templates=None):
    from train import attack_guard as AG

    cfg = json.loads((M / "m2_first_cell_rwkv7_safety_eval_d13_v3w.json").read_text(encoding="utf-8"))
    return {"config": cfg, "data": {"train_set": {
        "n_drawn": 24000, "drawn_ids_sha256": "8884d1a8dbb3be9366e7e07447f3ad726a5296f746a21aeeca4e7d02c82d5075",
        "attack_guard": {"rule": "attack_guard/v1", "templates_sha256": templates or AG.eval_templates_sha256(),
                         "n_records": 150000, "n_excluded": 8049, "per_set": {"d1a_steer": 8049, "d1b_steer": 3},
                         "drawn_hits": drawn_hits}}}}


@pytest.fixture(scope="module")
def scored(world, tmp_path_factory):
    t = tmp_path_factory.mktemp("scored")
    assert S.score(_score_args(world, t, "parent", "official"), loader=_loader(6.0, "parent")) == 0
    assert S.score(_score_args(world, t, "child", "official,heldout_v1"), loader=_loader(0.0, "child")) == 0
    (t / "cell.json").write_text(json.dumps(_cell()), encoding="utf-8")
    return t


def _report(world, t, cell="cell.json", **kw):
    a = SimpleNamespace(out_dir=str(t / "out"), cell=str(t / cell), params=str(M / f"job_params_{MODE}.json"),
                        elapsed_min=12.0, n_boot=N_BOOT, cohort=world["cohort"], cohort_sha256=world["sha"], **kw)
    return S.report(a)


def test_both_arms_score_every_unit_of_their_batteries(world, scored):
    from train.cohort import Cohort

    c = Cohort(world["cohort"], expected_sha256=world["sha"])
    ov, _ = S.battery_variants("official", c, production=False)
    hv, _ = S.battery_variants("heldout_v1", c, production=False)
    out = scored / "out"
    for name, v in (("units-parent-official.jsonl", ov), ("units-child-official.jsonl", ov),
                    ("units-child-heldout_v1.jsonl", hv)):
        rows = S._rows(out / name)
        S.check_rows(rows, v, name)  # exactly one row per unit, same unit_sha, nothing extra
        assert [r["request_id"] for r in rows] == [x["request_id"] for x in v]
    assert not (out / "units-parent-heldout_v1.jsonl").exists()
    idc = json.loads((out / "identity-child.json").read_text(encoding="utf-8"))
    assert set(idc["batteries"]) == {"official", "heldout_v1"} and idc["checkpoint"]["sha256"] == CKPT_SHA
    assert idc["units_this_attempt"] == len(ov) + len(hv)


def test_report_is_valid_and_official_is_reportable_for_an_attack_clean_draw(world, scored):
    res = _report(world, scored)
    assert S.valid_result(res), json.dumps(res, default=str)[:2000]
    assert res["official"]["reportable"] is True and res["training_exclusion"]["official_reportable"] is True
    o = res["official"]
    # stand-ins: the robust child is hijacked less than the gullible parent (paired child - parent)
    assert o["hijack"]["hijack_rate"]["diff"]["point"] < 0 and o["hijack"]["n_units"] > 0
    assert {"hijack", "steer_d1a", "units"} <= set(o) and o["units"]["by_family"]["hijack"] > 0
    assert res["heldout_v1"]["complete"] and set(res["heldout_v1"]["child"]["families"]) == {
        "heldout_hijack", "heldout_steer", "heldout_abstain"}
    ca = res["clean_agreement"]
    assert ca["n_shared"] == ca["n_equal"] > 0 and ca["max_abs_prob_diff"] == 0.0
    assert res["readout"]["name"] == res["readout_requested"] == "v3w"
    assert "baselines.clef_heldout_v1 compare --champion" in res["compare_command"]


@pytest.mark.parametrize("cell_kw", [{"drawn_hits": 3}, {"templates": "0" * 64}])
def test_official_is_not_reportable_unless_the_draw_excluded_every_eval_template(world, scored, cell_kw):
    (scored / "cell-dirty.json").write_text(json.dumps(_cell(**cell_kw)), encoding="utf-8")
    res = _report(world, scored, cell="cell-dirty.json")
    assert res["official"]["reportable"] is False and res["training_exclusion"]["official_reportable"] is False
    assert S.valid_result(res)  # still a valid measurement; the flag is what the landing must respect


def test_validator_refuses_another_checkpoint_or_readout(world, scored):
    res = _report(world, scored)
    bad = json.loads(json.dumps(res, default=float))
    bad["checkpoint"]["loaded"]["sha256"] = "0" * 64
    assert not S.valid_result(bad)
    bad = json.loads(json.dumps(res, default=float))
    bad["readout"]["name"] = "native"
    assert not S.valid_result(bad)
    bad = json.loads(json.dumps(res, default=float))
    bad["official"]["complete"] = False
    assert not S.valid_result(bad)


def test_child_rows_feed_the_clef_heldout_v1_compare(world, scored):
    """What the landing runs: compare --champion <child heldout rows> --champion-official <child official rows>."""
    from baselines import clef_heldout_v1 as CH
    from safety_eval import metrics as Mx
    from safety_eval.standin import KeywordDecider
    from train.cohort import Cohort

    c = Cohort(world["cohort"], expected_sha256=world["sha"])
    hv, _ = S.battery_variants("heldout_v1", c, production=False)
    units = {"clef-flash": Mx.run(KeywordDecider(gullibility=6.0), hv), "clef": Mx.run(KeywordDecider(gullibility=2.0), hv)}
    out = CH.compare(CH.read_rows(scored / "out/units-child-heldout_v1.jsonl"), units, N_BOOT, world["cohort"],
                     world["sha"], json.loads((scored / "out/identity-child.json").read_text(encoding="utf-8")),
                     CH.read_rows(scored / "out/units-child-official.jsonl"))
    assert out["complete"] and len(out["primary"]) == 4
    assert out["champion"]["checkpoint"]["sha256"] == CKPT_SHA
    assert set(out["generalisation_gap_champion"]) == {"hijack_rate", "steer_rate_vs_d1A"}


def test_resume_scores_only_the_missing_units_and_matches_a_clean_run(world, scored, tmp_path):
    from safety_eval.standin import KeywordDecider

    a = _score_args(world, tmp_path, "child", "official,heldout_v1")
    full = S._rows(scored / "out/units-child-official.jsonl")
    (tmp_path / "ck").mkdir()
    keep = full[:7]
    with open(tmp_path / "ck/child-official.partial.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
        f.write('{"request_id": "torn')  # a torn tail (killed mid-write) is dropped, not an error
    calls = []

    def load(a_):
        dec, ident = _loader(0.0, "child")(a_)
        return (lambda req: calls.append(req) or dec(req)), ident
    assert S.score(a, loader=load) == 0
    idc = json.loads((scored / "out/identity-child.json").read_text(encoding="utf-8"))
    assert len(calls) == idc["units_this_attempt"] - len(keep)
    for name in ("units-child-official.jsonl", "units-child-heldout_v1.jsonl"):
        assert (tmp_path / "out" / name).read_bytes() == (scored / "out" / name).read_bytes()
    # another checkpoint's resume log is refused before any scoring
    other = _loader(0.0, "child")

    def load_other(a_):
        dec, ident = other(a_)
        return dec, {**ident, "checkpoint": {**ident["checkpoint"], "sha256": "1" * 64}}
    assert S.score(a, loader=load_other) == 10
    _ = KeywordDecider


def test_unit_set_drift_and_bad_batteries_fail_before_any_model_load(world, tmp_path):
    loaded = []

    def load(a_):
        loaded.append(1)
        raise AssertionError("must not load")
    assert S.score(_score_args(world, tmp_path, "child", "official,official"), loader=load) == 2
    assert S.score(_score_args(world, tmp_path, "child", "d1b"), loader=load) == 2
    bad = _score_args(world, tmp_path, "child", "official")
    bad.cohort_sha256 = "0" * 64
    assert S.score(bad, loader=load) == 10
    assert not loaded


def test_production_batteries_match_their_pins():
    ov, oid = S.battery_variants("official")
    assert {k: oid[k] for k in S.OFFICIAL_PIN} == S.OFFICIAL_PIN and len(ov) == 3600
    assert oid["templates"] == sorted(["authority_override", "chatml_spoof", "completion_fake", "header_spoof",
                                       "ignore_previous", "authority", "fake_correction", "system_note", "urgency"])
    hv, hid = S.battery_variants("heldout_v1")
    assert hid["n_units"] == 7920 and len(hv) == 7920
    # the 360 clean units are the same units in both batteries
    from safety_eval.metrics import unit_fingerprint

    co = {x["request_id"]: unit_fingerprint(x) for x in ov if x["_meta"]["attack_family"] == "clean"}
    ch = {x["request_id"]: unit_fingerprint(x) for x in hv if x["_meta"]["attack_family"] == "clean"}
    assert co == ch and len(co) == 360


def test_params_price_under_the_caps_and_pin_d13s_best_checkpoint():
    from jobs.common import costs

    p = json.loads((M / f"job_params_{MODE}.json").read_text(encoding="utf-8"))
    v3w = json.loads((M / "job_params_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    assert p["mode"] == MODE and p["eval_only"] == "safety" and p["readout"] == "v3w"
    assert p["checkpoint"] == v3w["checkpoint"] and p["checkpoint"]["ckpt_sha256"] == CKPT_SHA
    assert (M / "m2_first_cell_rwkv7_safety_eval_d13_v3w.json").read_bytes() == \
        (M / "m2_first_cell_rwkv7_d13_joint_safe_nofla.json").read_bytes()
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 254
    assert costs.worst_case_usd(p) == 28.12 <= p["hard_cap_usd"] and costs.quoted_max_cost_usd(p) == 16.0
    est = S.estimate(p)
    assert est["ok"] and est["phases"]["safety"]["units"] == 11520 and est["phases"]["parent_safety"]["units"] == 3600
    assert est["phases"]["safety"]["expected_min"] == 148.2 and est["phases"]["parent_safety"]["expected_min"] == 19.7
    slow = json.loads(json.dumps(p))
    slow["safety"]["child_s_per_unit"] = 1.0
    assert not S.estimate(slow)["ok"]
    pins = [d for d in p["context"]["data"] if "pin_json" in d]
    assert pins == [{"src": "m2/checkpoints/job-hjwpq/cell.json", "dest": "checkpoint/cell.json",
                     "pin_json": [f"jobs/m2_first_cell/job_params_{MODE}.json", "checkpoint", "cell_json_sha256"]}]
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {MODE}") and "<" in p["submit"]["env"]["CKPT_URL"]
    assert MODE in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert "$28.12" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")


def test_runner_wiring():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(M / "first_cell_job.sh")]).returncode == 0
    assert 'if ev not in ("none", "fulldev", "safety"):' in sh
    assert "VALIDATOR=jobs.m2_first_cell.safety_battery:valid_result DECISION=$WORK/out/safety_eval_result.json" in sh
    assert "python -X utf8 -m baselines.clef_heldout_v1 precheck" in sh
    assert "python -X utf8 -m jobs.m2_first_cell.safety_battery estimate" in sh
    assert 'for ARMNAME in parent child; do' in sh and "PH=parent_safety; BATS=$SB_PARENT" in sh
    assert 'jobs.m2_first_cell.safety_battery score --arm "$ARMNAME"' in sh and '--readout "$SB_READOUT"' in sh
    assert '[ "$EVAL_ONLY" = safety ] && sb_export_partial' in sh
    # the safety branch exits before any training phase
    i, j = sh.index('if [ "$EVAL_ONLY" = safety ]; then  # ----'), sh.index("# ------------------------------------------------------------------ held-out D1-B")
    assert i < j and "RC=0\n  exit 0\nfi" in sh[i:j]

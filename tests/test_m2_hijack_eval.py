"""BOARD H1: train.m2_gates stage ``hijack`` + jobs/m2_first_cell/hijack_eval.py (the saved-checkpoint hijack / steer
add-on, mode hijack_eval), on a mini cohort with stand-in deciders. Plumbing, identities and fail-closed behaviour only."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A mini cohort: no decision records (the hijack stage scores none) + real safety_eval dev requests."""
    from train import cohort as C

    d = tmp_path_factory.mktemp("hj")
    steer = C.steer_requests(6, 2)
    header = {"cohort": "test-mini-hijack", "sizes": {"dev": 0, "calibration": 0, "retention": 0, "steer_dev": 6,
                                                      "steer_gate": 2}, "sizing": None}
    lines = [json.dumps({"header": header}, sort_keys=True)]
    for role in ("steer_dev", "steer_gate"):
        for e in steer[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    p = d / "cohort.jsonl.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(text.encode())
    return {"dir": d, "cohort": C.Cohort(p, expected_sha256=hashlib.sha256(text.encode()).hexdigest())}


SPEC = {"max_k": 16, "max_t": 100000, "template": "jrt-plain-v1", "layout": "ordered"}


def _parent(world, out, steer):
    from jobs.m2_parent_arm.score_parent import score_parent_arm

    return score_parent_arm(stages=["hijack"], cohort=world["cohort"], spec=SPEC, decide=None, steer=steer,
                            items_by_role={}, records_by_id={}, out_dir=out,
                            parent_identity={"parent": {"parent": "tiny"}, "revision": "r0"})


def _child(world, cache, steer, out):
    from train import m2_gates as G

    return G.run_gates(stages=["hijack"], cohort=world["cohort"], spec=SPEC, child_decide=None, child_steer=steer,
                       child_identity={"revision_id": "child"}, items_by_role={}, records_by_id={}, parent_arm=cache,
                       n_boot=100, out=out,
                       parent_arm_sha256=hashlib.sha256((Path(cache) / "arm.json").read_bytes()).hexdigest())


def test_hijack_stage_variants_use_the_official_builder_and_never_d1b(world):
    from safety_eval import attacks
    from train import m2_gates as G

    v = G.stage_variants(world["cohort"], "hijack")
    fam = Counter(x["_meta"]["attack_family"] for x in v)
    n = fam["clean"]
    assert n == len(G.hijack_subset(world["cohort"].steer_requests("steer_dev")))
    assert fam == {"clean": n, "hijack": n * len(attacks.HIJACK_TEMPLATES), "d1": n * len(attacks.D1_TEMPLATES_A)}
    assert {x["_meta"]["template"] for x in v if x["_meta"]["attack_family"] == "d1"} == set(attacks.D1_TEMPLATES_A)
    assert not {x["_meta"]["template"] for x in v} & set(attacks.D1_TEMPLATES_B)  # held-out B never built here
    assert G.STAGE_ROLES["hijack"] == () and "hijack" in G.MEASURE_STAGES and "hijack" not in G.STAGES
    assert v == G.stage_variants(world["cohort"], "hijack")  # deterministic


def test_hijack_subset_is_cluster_stratified_and_deterministic():
    from safety_eval.data import build
    from train import m2_gates as G

    reqs = build(n_route=60, n_pay=40, split="dev")
    sub = G.hijack_subset(reqs, per_cluster=2)
    per = Counter(r["_meta"]["cluster"] for r in sub)
    allc = Counter(r["_meta"]["cluster"] for r in reqs)
    assert set(per) == set(allc) and all(per[c] == min(2, allc[c]) for c in allc)
    assert [r["request_id"] for r in sub] == [r["request_id"] for r in G.hijack_subset(list(reversed(reqs)), 2)]
    assert G.HIJACK_PER_CLUSTER == 8
    with pytest.raises(ValueError):
        G.hijack_subset(reqs, 0)


def test_hijack_volume_matches_the_stage_size():
    from jobs.m2_parent_arm import score_parent as SP
    from safety_eval import attacks
    from train import m2_gates as G

    assert SP.HIJACK_UNITS == 45 * G.HIJACK_PER_CLUSTER * (1 + len(attacks.HIJACK_TEMPLATES) + len(attacks.D1_TEMPLATES_A))
    e = SP.estimate(3757, ["hijack"], 0, "child", 0.117, child_extras=False)
    assert set(e["tokens"]) == {"hijack"} and e["total_calls"] == 3600
    assert set(SP.estimate(3757, ["hijack"], 0, "child", 0.117)["tokens"]) == {"hijack", "calibration_fit", "dev_card"}


def test_measured_exit_code_and_promotion_never_set_by_hijack():
    from train import m2_gates as G

    m = {"accept": None, "measurement": True}
    assert G.exit_code({"hijack": m}, pending=False) == G.EXIT["measured"] == 46
    assert G.exit_code({"hijack": m, "g12": {"accept": True, "failed": []}}, pending=False) == 44
    assert G.promotion_verdict({"hijack": m})["stages"] == []


@pytest.fixture(scope="module")
def runs(world):
    from safety_eval.standin import KeywordDecider

    cache = world["dir"] / "parent_arm_hijack"
    _parent(world, cache, KeywordDecider(gullibility=6.0))  # follows injected imperatives: hijackable
    out = world["dir"] / "child" / "report.json"
    code, rep = _child(world, cache, KeywordDecider(gullibility=0.0), out)
    return {"cache": cache, "out": out, "code": code, "rep": rep}


def test_child_vs_parent_hijack_is_measured_and_complete(runs):
    r = runs
    assert r["code"] == 46 and r["rep"]["exit_code"] == 46
    v = r["rep"]["verdicts"]["hijack"]
    assert v["measurement"] is True and v["complete"] is True
    h = v["families"]["hijack"]["two_way"]
    assert h["n_units"] > 0 and h["eligibility"] == "reference_y" and h["complete"]
    assert h["hijack_rate"]["y"] > h["hijack_rate"]["x"]  # the gullible parent follows the hijacks, the child not
    assert v["families"]["d1_A"]["fixed_template"]["bootstrap"] == "fixed_template"
    assert r["out"].with_name("units-child-hijack.json").exists()
    assert not r["out"].with_name("child-arm").exists()  # a measurement never becomes a comparator


def _params(tmp_path, ck_sha):
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_hijack_eval.json").read_text(encoding="utf-8"))
    p["checkpoint"] = {**p["checkpoint"], "ckpt_sha256": ck_sha}
    f = tmp_path / "params.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    return f


def test_hijack_eval_result_is_valid_compact_and_pins_the_loaded_checkpoint(world, runs, tmp_path, monkeypatch):
    from jobs.m2_first_cell import hijack_eval as H

    rep = json.loads(runs["out"].read_text(encoding="utf-8"))
    rep["identities"]["checkpoint"] = {"path": "ckpt-step0000006.pt", "step": 6, "sha256": "a" * 64}
    runs["out"].write_text(json.dumps(rep), encoding="utf-8")
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": {"cohort_sha256": world["cohort"].sha256, "parent": "rwkv7"},
                                "train": {"steps": 6}}), encoding="utf-8")
    from train import cohort as C

    monkeypatch.setattr(C, "Cohort", lambda expected_sha256: world["cohort"])
    a = SimpleNamespace(cell=str(cell), report=str(runs["out"]), parent_arm=str(runs["cache"]),
                        params=str(_params(tmp_path, "a" * 64)), elapsed_min=60.0, n_boot=100)
    res = H.summarize(a)
    assert H.valid_result(res), res
    assert res["units"]["by_family"]["hijack"] == 5 * res["units"]["n_requests"]
    assert res["hijack"]["hijack_rate"]["parent"] > res["hijack"]["hijack_rate"]["child"]
    assert len(json.dumps(res, default=float)) < 24_000  # the platform result.json limit is 32 KiB (LESSONS)
    a.params = str(_params(tmp_path, "b" * 64))  # the loaded checkpoint is not the pinned one
    assert not H.valid_result(H.summarize(a))
    bad = dict(res, complete=False)
    assert not H.valid_result(bad) and not H.valid_result(dict(res, m2_gates_exit=44))


def test_mode_files_and_runner_wiring():
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_hijack_eval.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json").read_text(encoding="utf-8"))
    mode = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_hijack_eval.json").read_text(encoding="utf-8"))
    assert mode == base and p["mode"] == "hijack_eval" and p["decision_rule"] is None
    assert p["eval_only"] == "hijack"  # fulldev_eval uses eval_only "fulldev"; only "hijack" takes this branch
    assert p["submit"]["command"].endswith("first_cell_job.sh hijack_eval") and "CKPT_URL" in p["submit"]["env"]
    assert set(p["phase_caps_minutes"]) == {"setup", "guard", "checkpoint", "parent_hijack", "evals", "report"}
    assert any(d["dest"] == "checkpoint/cell.json" for d in p["context"]["data"])
    assert not any(d["dest"].startswith("parent_arm_seed/") for d in p["context"]["data"])
    sh = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    for s in ("jobs.m2_first_cell.checkpoint_loader install", "--stage hijack", "jobs.m2_first_cell.hijack_eval",
              "case $r in 46) ;;", 'p.get("eval_only") == "hijack"', "VALIDATOR=jobs.m2_first_cell.hijack_eval:valid_result"):
        assert s in sh, s

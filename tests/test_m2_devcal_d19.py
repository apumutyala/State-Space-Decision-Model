"""Mode devcal_d19_v3w (B-PREREG-D19-TEMP addendum 1): D19's per-question probabilities on its cohort-dev pool and the
CPU temperature fit. CPU only: no model, no kit."""
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.m2_first_cell import checkpoint_loader as L
from jobs.m2_first_cell import devcal_eval as D

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
MODE = "devcal_d19_v3w"


def _p(mode):
    return json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))


def test_params_pin_d19_best_and_its_lineage():
    p, fd = _p(MODE), _p("fulldev_eval_d19_v3w_s1of2")
    assert L.checkpoint_block(p) == L.checkpoint_block(fd)  # the same pinned best-step-2800 export bytes
    assert p["devcal_expect_step"] == 2800 and p["submit"]["env"]["CKPT_URL"] == "<CKPT_URL>"
    assert (ROOT / p["cell_config"]).read_bytes() == (M / "m2_first_cell_rwkv7_d19_v3w_v3b_joint.json").read_bytes()
    assert p["context"]["data"][:-1] == fd["context"]["data"][:-1]
    assert p["context"]["data"][-1]["pin_json"] == [f"jobs/m2_first_cell/job_params_{MODE}.json", "checkpoint",
                                                    "cell_json_sha256"]
    assert p["train_dir"] == fd["train_dir"] and p["mode"] == MODE
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {MODE}")
    assert "jobs/m2_first_cell/devcal_eval.py" in p["context"]["manifest_files"]


def test_params_fit_the_hold():
    from jobs.common.costs import quoted_max_cost_usd, schedule_minutes, worst_case_usd

    p = _p(MODE)
    assert schedule_minutes(p) <= p["max_duration_minutes"]
    assert quoted_max_cost_usd(p) <= 8.0 and worst_case_usd(p) <= p["hard_cap_usd"]
    assert p["phase_caps_minutes"]["devcal"] >= 1.5 * p["expected_minutes"]["devcal"]
    keys = {_p(m)["submit"]["key"] for m in (MODE, "fulldev_eval_d19_v3w_s1of2", "fulldev_eval_d19_v3w_s2of2",
                                             "d19_v3w_v3b_joint")}
    assert len(keys) == 4


def test_runner_knows_the_mode():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert '"devcal"' in sh and "jobs.m2_first_cell.devcal_eval score" in sh and "$(cap devcal)m" in sh


def _synthetic(n=3000, k=4, t0=1.8, seed=0):
    r = random.Random(seed)
    probs, labels = [], []
    for _ in range(n):
        z = [r.gauss(0, 2.5) for _ in range(k)]
        true = [math.exp(x / t0) for x in z]
        s = sum(true)
        labels.append(r.choices(range(k), weights=[x / s for x in true])[0])
        e = [math.exp(x) for x in z]
        probs.append([x / sum(e) for x in e])
    return probs, labels


def test_fit_recovers_a_known_temperature():
    probs, labels = _synthetic()
    t = D.fit_t(probs, labels)
    assert t == pytest.approx(1.8, abs=0.2)
    assert D.mean_nll(probs, labels, t) <= D.mean_nll(probs, labels, 1.0)
    assert D._softmax_t([0.7, 0.2, 0.1], 1.0) == pytest.approx([0.7, 0.2, 0.1])


def _items(ids):
    return [SimpleNamespace(id=i) for i in ids]


def test_check_pool_refuses_a_different_pool_and_training_overlap():
    from train.trainer import dev_card_record

    ids = [f"m2:{s}:{n}/q1" for s in ("anli", "boolq", "nq") for n in range(10)]
    items = _items(ids)
    cell = {"dev_card_sample": dev_card_record(items, 6, 0, True)}
    got = D.check_pool(cell, items, set(), True)
    assert got["n_pool"] == 30 and got["dev_card_n"] == 6 and got["train_overlap"] == 0
    with pytest.raises(ValueError, match="n_pool"):
        D.check_pool(cell, items[:-1], set(), True)
    with pytest.raises(ValueError, match="training ids"):
        D.check_pool(cell, items, {"m2:nq:3"}, True)
    sampled = set(cell["dev_card_sample"]["ids"])
    other = _items([i if i not in sampled or i != sorted(sampled)[0] else "m2:anli:99/q1" for i in ids])
    with pytest.raises(ValueError, match="dev-card sample"):
        D.check_pool(cell, other, set(), True)


def test_fit_end_to_end_with_the_devcard_gate(tmp_path):
    from train.metrics import decision_metrics

    probs, labels = _synthetic(n=400)
    ids = [f"m2:x:{i}/q1" for i in range(400)]
    dev = list(range(0, 400, 4))
    for name, ix in (("pool", range(400)), ("devcard", dev)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "units-step0002800.json").write_text(json.dumps(
            {"item_ids": [ids[i] for i in ix], "labels": [labels[i] for i in ix],
             "decision_probs": [probs[i] for i in ix]}), encoding="utf-8")
    raw = decision_metrics(probs, labels)
    ref = decision_metrics([probs[i] for i in dev], [labels[i] for i in dev])
    sc = {"checkpoint": {"step": 2800, "sha256": "a" * 64},
          "identity": {"pool_ids_sha256": D.ids_sha256(ids), "n_pool": 400, "dev_card_n": 100, "train_overlap": 0},
          "pool": {"n": 400, "raw": raw, "units": "pool/units-step0002800.json"},
          "devcard": {"n": 100, "raw": ref, "units": "devcard/units-step0002800.json"},
          "training_card": {"step": 2800, "nll": ref["nll"] + 0.005, "accuracy": ref["accuracy"]}}
    (tmp_path / "devcal-score.json").write_text(json.dumps(sc), encoding="utf-8")
    rep = D.fit(SimpleNamespace(score_dir=str(tmp_path)))
    assert rep["devcard_gate"]["pass"] and rep["usable"] and rep["n"] == 400
    assert rep["sets"]["devcard"]["n"] == 100 and rep["sets"]["rest"]["n"] == 300
    assert rep["sets"]["pool"]["after"]["nll"] <= rep["sets"]["pool"]["before"]["nll"]
    sc["training_card"]["nll"] = ref["nll"] + 0.05  # the rescored dev card does not reproduce the run's card
    (tmp_path / "devcal-score.json").write_text(json.dumps(sc), encoding="utf-8")
    assert not D.fit(SimpleNamespace(score_dir=str(tmp_path)))["usable"]


# ---- job-s6kz7: the devcal params lacked "eval_only": "devcal", so the runner took the TRAINING setup branch and
# priced parent_g12 (no cap in these params -> KeyError, reported as exit 15). Run the runner's own mode parse on the
# real params, then the setup preflight's stage selection, on CPU.
def _mode_env_script() -> str:
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    start = sh.index("MODE_ENV=$(python -I - ")
    body = sh[sh.index("<<'EOF'\n", start) + len("<<'EOF'\n"):]
    return body[:body.index("\nEOF\n")]


def _mode_env(params: dict, tmp_path):
    import subprocess
    import sys

    f = tmp_path / f"job_params_{MODE}.json"
    f.write_text(json.dumps(params), encoding="utf-8")
    return subprocess.run([sys.executable, "-I", "-", MODE, params["cell_config"], str(f), ""], cwd=ROOT,
                          input=_mode_env_script(), capture_output=True, text=True, timeout=120)


def _priced_stages(params: dict, eval_only: str, stage: str) -> list[str]:
    """The phase caps first_cell_job.sh's setup preflight looks up (the EVAL_PRICES loop), per branch."""
    if eval_only in ("fulldev", "devcal", "ragtruth_rephrase"):
        return []  # fulldev: fulldev_eval estimate; devcal / ragtruth_rephrase: own preflights; no parent arm priced
    priced = ["parent_g12", "parent_index", "evals"]
    return priced[:2] if stage == "screen" else priced


def test_devcal_mode_parses_as_eval_only_and_prices_no_parent_stage(tmp_path):
    p = _p(MODE)
    r = _mode_env(p, tmp_path)
    assert r.returncode == 0, r.stderr
    train_sub, teacher, ev, stage = r.stdout.split()
    assert (train_sub, ev, stage) == (p["train_dir"], "devcal", "gates")
    priced = _priced_stages(p, ev, stage)
    assert priced == [] and not any(s.startswith("parent") for s in priced)
    caps = p["phase_caps_minutes"]
    assert all(s in caps for s in ("setup", "guard", "checkpoint", "devcal")) and "parent_g12" not in caps


def test_devcal_params_without_eval_only_are_refused_at_mode_parse(tmp_path):
    p = dict(_p(MODE))
    del p["eval_only"]  # the job-s6kz7 params
    r = _mode_env(p, tmp_path)
    assert r.returncode != 0 and "devcal_expect_step" in (r.stdout + r.stderr)


def test_runner_devcal_branch_never_schedules_parent_stages():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    # setup: the devcal branch sits before the training preflight's else (where EVAL_PRICES / parent_g12 live)
    pre = sh.index('elif [ "$EVAL_ONLY" = devcal ]')
    els = sh.index("\nelse\n", pre)
    assert sh.index('EVAL_PRICES=("g12 parent parent_g12') > els and "parent" not in sh[pre:els]
    # phases: the devcal block exits 0 before the fulldev / heldout / parent / cell phases
    blk = sh.index('if [ "$EVAL_ONLY" = devcal ]; then  # ----- eval-only')
    end = sh.index("  exit 0\nfi\n", blk)
    assert "parent" not in sh[blk:end] and sh.index("cap parent_g12") > end
    # a missing cap is a params error (exit 5), never a KeyError swallowed into exit 15
    assert 'CAPM=$(cap "$3" 2> /dev/null) || fail 5' in sh and '--cap-min "$CAPM"' in sh


def test_every_first_cell_mode_caps_the_stages_its_preflight_prices(tmp_path):
    for f in sorted(M.glob("job_params*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        if not p.get("runner", "").endswith("first_cell_job.sh") or "phase_caps_minutes" not in p:
            continue
        if str(p.get("status", "")).startswith("draft"):  # unsubmitted drafts (hijack_eval: phases the runner lacks)
            continue
        if "devcal_expect_step" in p:
            assert p.get("eval_only") == "devcal", f.name
        missing = [s for s in _priced_stages(p, p.get("eval_only", "none"), p.get("stage", "gates"))
                   if s not in p["phase_caps_minutes"]]
        assert not missing, (f.name, missing)

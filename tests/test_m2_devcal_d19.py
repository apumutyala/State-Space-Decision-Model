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

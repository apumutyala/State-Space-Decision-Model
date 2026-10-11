"""A-PREREG-JEV2-4B (mode jev2_q35_4b) on CPU, stdlib-light: the package invariants, the scalar T fit, the re-tempering,
the balanced draw builder (tiny synthetic sources) and r2's frozen jevkit block (no new keys unless set)."""
import gzip
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "jobs/m2_first_cell"
CELL = json.loads((M / "m2_first_cell_rwkv7_jev2_q35_4b.json").read_text(encoding="utf-8"))
PARAMS = json.loads((M / "job_params_jev2_q35_4b.json").read_text(encoding="utf-8"))
R2 = json.loads((M / "job_params_jev1_q35_4b.json").read_text(encoding="utf-8"))


def test_one_pass_fixed_budget():
    assert CELL["layout"] == "sym-fork"
    assert CELL["max_steps"] == CELL["n_train_records"] // (CELL["batch_size"] * CELL["grad_accum"])
    assert CELL["min_steps"] == 600 <= CELL["max_steps"]  # step-10 projection continues down to this floor
    assert "lr_horizon_steps" not in CELL and abs(CELL["lr_horizon_frac"] - 753 / 1121) < 1e-3
    assert CELL["max_minutes"] >= math.floor(CELL["max_steps"] * 10.92 * 0.985 * 1.3 / 60)
    assert PARAMS["phase_caps_minutes"]["cell"] == CELL["max_minutes"] + 35


def test_4k_and_label_smoothing():
    assert max(CELL["t_ladder"]) == CELL["eval_max_t"] == CELL["require_t_max"] == 4096
    assert CELL["ladder_probe_longest"] is True and CELL["label_smoothing"] <= 0.05
    assert CELL["label_smoothing_noul"] == 0.0


def test_noul_rows_unsmoothed_choice_rows_smoothed():
    torch = pytest.importorskip("torch")
    from train.losses import LossConfig, arm_loss, label_smoothed_ce, brier

    lp = torch.log_softmax(torch.tensor([[2.0, 0.0], [1.0, 0.5]]), -1)
    y = torch.tensor([0, 1])
    cfg = LossConfig(arm="clef", label_smoothing=0.05, label_smoothing_noul=0.0, rps_weight=0.0)
    per = arm_loss(lp, y, ["noul", "choice"], cfg)
    want0 = label_smoothed_ce(lp[:1], y[:1], 0.0) + brier(lp[:1], y[:1])
    want1 = label_smoothed_ce(lp[1:], y[1:], 0.05) + brier(lp[1:], y[1:])
    assert torch.allclose(per, torch.cat([want0, want1]))
    assert "label_smoothing_noul" not in LossConfig(arm="clef").identity()  # old config hashes unchanged


def test_t_constrained_by_noul_band():
    import math

    from jobs.m2_first_cell.screen import constrain_t_by_band, noul_band_rate
    noul = [([math.log(0.85), math.log(0.15)], 0)] * 10  # p(yes) 0.85 at T = 1: outside the band
    r = constrain_t_by_band({"T": 3.0}, noul)  # T 3 would flatten them into the band
    assert r["constrained"] and r["band_T"] <= r["band_T1"] == 0.0 and r["T"] < 3.0
    assert noul_band_rate([0.1, 0.5, 0.9])["in_band"] == 1


def test_pools_jev2a_wired_placeholders_declared():
    pools = CELL["train_set"]["pools"]
    assert pools["jev2a"]["status"] == "gated" and len(pools["jev2a"]["pointer_sha256"]) == 64
    assert {pools["jev2b"]["status"], pools["jev2c"]["status"]} == {"PLACEHOLDER"}
    assert len({f for p in pools.values() for f in p["families"]}) == 18
    assert "--exclude long_policy=bschema-draw-jev1" in CELL["train_set"]["note"]
    assert (ROOT / pools["jev2a"]["pointer"]).is_file()


def test_calib_slice_pinned_and_disjoint_role():
    cs = CELL["calib_slice"]
    assert cs["per_family"] >= 200 and cs["records"] == PARAMS["jevkit"]["tfit"]["records"]
    assert cs["records"].startswith("m2-jev2mix-calib/") and not PARAMS["train_dir"].endswith("calib")
    from jobs.m2_first_cell.screen import slice_ids_sha256
    import hashlib
    assert slice_ids_sha256(["b", "a"]) == hashlib.sha256(b"a\nb").hexdigest()


def test_schedule_and_cost():
    caps = sum(PARAMS["phase_caps_minutes"].values()) + PARAMS["reserve_minutes"]
    assert PARAMS["max_duration_minutes"] == caps <= 899  # one job, no chain
    worst = (60 + 2 * caps) * 0.063
    assert worst <= PARAMS["hard_cap_usd"] <= 80.0  # operator hard budget (2026-10-10, D20 links 2-3 parked)


def test_jevkit_additions_and_r2_untouched():
    jk = PARAMS["jevkit"]
    assert jk["perturb_ops"] == ["option_reverse", "option_shuffle", "key_relabel"]
    assert jk["tfit"]["records"].startswith("m2-jev12-calib/")
    for k in ("price", "shim_calibration", "stock_shim", "paired_primary"):
        assert jk[k] == R2["jevkit"][k]
    assert not {"tfit", "perturb_ops", "covered_families"} & set(R2["jevkit"])
    from jobs.m2_first_cell.screen import jevkit_block
    blk = jevkit_block(R2, "jrt-h2o-shim:qwen3.5-4b")
    assert not {"tfit", "perturb_ops", "covered_families"} & set(blk)  # r2's eval path: same block as before


def test_fit_temperature_recovers_t():
    import random

    from jobs.m2_first_cell.screen import fit_temperature
    rng = random.Random(0)
    logps, labels = [], []
    for _ in range(3000):  # true probs = softmax(z); the "model" reports z * 2 (overconfident) -> T = 2
        z = [rng.gauss(0, 1) for _ in range(4)]
        m = max(z)
        p = [math.exp(v - m) for v in z]
        s = sum(p)
        u, acc, y = rng.random(), 0.0, 3
        for k, v in enumerate(p):
            acc += v / s
            if u <= acc:
                y = k
                break
        logps.append([2 * v for v in z])
        labels.append(y)
    r = fit_temperature(logps, labels)
    assert 1.8 < r["T"] < 2.2 and r["nll_T"] < r["nll_1"] and not r["at_bound"]


def test_retemper_answer():
    from jobs.m2_first_cell.screen import retemper_answer
    a = retemper_answer({"type": "choice", "criteria": {"x": "", "y": ""}},
                        {"type": "choice", "probabilities": {"x": 0.8, "y": 0.2}}, 2.0)
    assert a["choice"] == "x" and abs(a["probabilities"]["x"] - 2 / 3) < 1e-9
    n = retemper_answer({"type": "noul"}, {"type": "noul", "noul": 0.9}, 1.0)
    assert abs(n["noul"] - 0.9) < 1e-9


def _src(tmp: Path, name: str, fams: dict) -> Path:
    d = tmp / name
    d.mkdir()
    with gzip.open(d / "records.jsonl.gz", "wt", encoding="utf-8") as f:
        for fam, n in fams.items():
            for i in range(n):
                f.write(json.dumps({"id": f"{name}:{fam}:{i}", "family": f"{name}_{fam}", "source": "t",
                                    "state": {"k": f"{name}{fam}{i}"},
                                    "questions": {"q": {"type": "noul", "instructions": f"q{i}"}},
                                    "labels": {"q": "true"}, "aug": {name: {"family8": fam}}}) + "\n")
    return d


def test_balanced_draw(tmp_path):
    from scripts.jev12_balanced_draw import main
    a = _src(tmp_path, "jev1", {"intent": 30, "policy": 25})
    b = _src(tmp_path, "jev2", {"multi_hop": 22})
    assert main(["--src", str(a), "--src", str(b), "--train-per-family", "15", "--calib-per-family", "5",
                 "--name", "bal", "--out", str(tmp_path / "out")]) == 0
    rep = json.loads((tmp_path / "out/bal/draw-report.json").read_text(encoding="utf-8"))
    assert rep["train_rows"] == {"intent": 15, "policy": 15, "multi_hop": 15} or set(rep["train_rows"].values()) == {15}
    assert set(rep["calib_rows"].values()) == {5} and rep["content_disjoint"]
    with pytest.raises(SystemExit):
        main(["--src", str(a), "--train-per-family", "40", "--calib-per-family", "5", "--name", "bal2",
              "--out", str(tmp_path / "out")])

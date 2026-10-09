"""job-4ha7t (fulldev_eval_champ, self_terminated): the full-DEV plan pass (12,104 rows of tokenisation, ~0.21 s/row
under v3w = ~42 min, job-39gnn's plan timings) wrote nothing, the runner's watchdog (STALL_S 1200) withheld the
heartbeat and the platform killed the job after two missed 600 s polls. The plan pass must keep the progress beat
alive, and the plan result must be unchanged by it."""
import json
from pathlib import Path

from jobs.m2_first_cell import fulldev_eval as F
from tests.test_m2_fulldev_eval import _rows, _TokBackend

ROOT = Path(__file__).resolve().parents[1]


def test_plan_pass_touches_the_progress_beat_and_its_result_is_unchanged(tmp_path, monkeypatch):
    from decide import Decider

    beat = tmp_path / "beat"
    dec = Decider(_TokBackend())
    rows = _rows()
    monkeypatch.delenv("SSDM_PROGRESS_BEAT", raising=False)
    quiet = F.plan_rows(dec, rows, F.DEFAULT_MAX_INPUT_TOKENS)
    assert not beat.exists()
    monkeypatch.setenv("SSDM_PROGRESS_BEAT", str(beat))
    loud = F.plan_rows(dec, rows, F.DEFAULT_MAX_INPUT_TOKENS)
    assert beat.exists() and loud == quiet


def test_beat_is_throttled_and_keeps_a_long_loop_inside_the_watchdog_window(tmp_path, monkeypatch):
    beat = tmp_path / "beat"
    monkeypatch.setenv("SSDM_PROGRESS_BEAT", str(beat))
    clock = [1000.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    b = F.progress_beat(30.0)
    touches = []
    real_touch = Path.touch
    monkeypatch.setattr(Path, "touch", lambda self, *a, **k: (touches.append(clock[0]), real_touch(self, *a, **k)))
    for _ in range(12_104):  # one call per DEV row, 0.21 s apart (the measured v3w plan rate)
        b()
        clock[0] += 0.21
    gaps = [y - x for x, y in zip(touches, touches[1:])]
    assert touches and max(gaps) < 31 and len(touches) < 12_104 / 100  # every ~30 s, never per row
    assert max(gaps) < 1200  # inside STALL_S (jobs/m2_first_cell/first_cell_job.sh)


def test_fulldev_fixed_minutes_price_the_plan_pass():
    """The setup preflight must price the plan pass (job-39gnn: ~7 min per 2,008 rows per readout)."""
    for f in ("job_params_fulldev_eval_champ.json",):
        p = json.loads((ROOT / "jobs/m2_first_cell" / f).read_text(encoding="utf-8"))
        assert p["evals_fixed_minutes"]["child"] >= 30 + 12_104 * 7 / 2_008
        e = F.estimate(p)
        assert e["ok"], e

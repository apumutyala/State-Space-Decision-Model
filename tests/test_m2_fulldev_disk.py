"""job-afpc2 (fulldev_eval_champ, failed exit 9 at +214 min, no artifact): the worker's SHARED disk (checkpoint dir,
output dir, the platform spool) filled mid-scoring: the child died rc 1, the phase tgz could not be written and the
platform refused the artifact. No checkpoint slot was ever declared, so neither the three preempted attempts nor the
failure resumed anything. The child must keep the slot declared, stop cleanly on a low disk (declare + yield, exit 16)
with every scored row kept, and a rerun must resume from the per-row log."""
import json
import sys
import types
from pathlib import Path

import pytest

from jobs.m2_first_cell import checkpoint_loader as L
from jobs.m2_first_cell import fulldev_eval as F
from tests.test_m2_checkpoint_loader import PT, _artifact, _install

ROOT = Path(__file__).resolve().parents[1]


def test_loader_removes_its_downloaded_tar_after_installing_both_pinned_members(tmp_path):
    tar, cell, cfg, block = _artifact(tmp_path)
    rec = _install(tmp_path, tar, cell, cfg, block)
    assert rec["reused"] is False and L.sha256_file(tmp_path / "ck" / PT) == block["ckpt_sha256"]
    assert not (tmp_path / "work/artifact.tar").exists()  # the 3.5 GB copy is gone; the source is untouched
    assert tar.exists()


def test_declarer_is_throttled_to_the_platform_floor_and_force_always_posts(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    posts = []
    d = F.checkpoint_declarer(post=lambda: posts.append(clock[0]) or True)
    assert d(force=True) and posts == [0.0]
    for _ in range(12_104):  # one call per DEV row, 1.8 s apart (D1's measured child rate)
        clock[0] += 1.8
        d()
    gaps = [y - x for x, y in zip(posts, posts[1:])]
    assert min(gaps) >= 300 and max(gaps) < F.CHECKPOINT_EVERY_S + 2  # >= the 300 s floor, every ~10 min
    n = len(posts)
    assert d(force=True) and len(posts) == n + 1


def test_disk_watch_reports_only_low_paths_and_is_throttled(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    free = {"a": 50 << 30, "b": 50 << 30}
    w = F.disk_watch(["a", "b"], usage=lambda p: free[p])
    assert w() is None
    free["b"] = 200 << 20
    assert w() is None  # throttled: no new reading inside 30 s
    clock[0] += 31
    assert w() == {"b": 200 << 20}
    assert F.disk_watch([tmp_path], low_bytes=0)() is None  # the real shutil path
    assert list(F.disk_watch([tmp_path], low_bytes=1 << 62)()) == [str(tmp_path)]


def test_stop_for_disk_declares_yields_at_most_twice_and_exits_16(tmp_path):
    decl, ylds = [], []
    declare = lambda force=False: decl.append(force) or True  # noqa: E731
    for _ in range(3):
        assert F.stop_for_disk(tmp_path, declare, "test", yield_post=lambda: ylds.append(1) or True) == F.DISK_LOW_RC
    assert decl == [True, True, True] and len(ylds) == F.MAX_YIELDS
    assert json.loads((tmp_path / "disk_yields.json").read_text())["yields"] == F.MAX_YIELDS


def _score_env(monkeypatch, tmp_path, n_rows=7):
    if not sys.flags.utf8_mode:  # the harness package refuses to import otherwise (the launcher runs -X utf8)
        pytest.skip("needs python -X utf8")
    from train import m2_gates as G

    rows = [{"_evaluation": {"run_id": f"r{i}"}, "state": "s", "questions": []} for i in range(n_rows)]
    monkeypatch.setattr(F, "N_DEV", n_rows)
    monkeypatch.setattr(F, "selected_ids", lambda s: None)
    # harness.suite needs the Decision Index kit (in the job image, not every checkout); score() only calls read_rows
    monkeypatch.setitem(sys.modules, "harness.suite", types.SimpleNamespace(read_rows=lambda p: rows))
    dec = types.SimpleNamespace(decide=lambda req: {"ok": True})
    monkeypatch.setattr(F, "load_child", lambda a: (dec, False, {"arm": "child"}))
    monkeypatch.setattr(F, "plan_rows", lambda d, r, m: {"n_rows": len(r), "n_refused": 0, "n_over_16384": 0,
                                                         "max_row_tokens": 1})
    scored = []

    class _Prog:
        def __init__(self, p):
            pass

        def plan(self, w):
            pass

        def wrap(self, fn):
            return fn

    monkeypatch.setattr(G, "Progress", _Prog)
    monkeypatch.setattr(G, "beat_wrap", lambda fn: fn)
    monkeypatch.setattr(G, "unit_weight", lambda r: 1)

    def predict(decide, rs, untempered_from_raw):
        r = rs[0]
        scored.append(r["_evaluation"]["run_id"])
        return {r["_evaluation"]["run_id"]: {"run_id": r["_evaluation"]["run_id"], "status": "ok",
                                             "response": {"x": 1}, "total_wall_ms": 5}}

    monkeypatch.setattr(G, "index_predict", predict)
    a = types.SimpleNamespace(rows="rows", rows_select="full", arm="child", readout=None, max_input_tokens=65536,
                              out_dir=str(tmp_path / "out"), resume_dir=str(tmp_path / "ck"),
                              progress=str(tmp_path / "p.jsonl"))
    return a, scored


def test_low_disk_mid_scoring_stops_with_16_keeps_rows_and_the_rerun_resumes(monkeypatch, tmp_path):
    a, scored = _score_env(monkeypatch, tmp_path)
    decl, ylds = [], []
    monkeypatch.setattr(F, "_meta_post", lambda door: (decl if door == "checkpoint" else ylds).append(door) or True)
    calls = [0]

    def watch(paths, **k):
        def low():
            calls[0] += 1
            return {"ck": 1 << 20} if calls[0] > 3 else None  # the disk fills before the 4th row
        return low

    monkeypatch.setattr(F, "disk_watch", watch)
    assert F.score(a) == F.DISK_LOW_RC
    part = (tmp_path / "ck/child.partial.jsonl").read_text().splitlines()
    assert [json.loads(x)["run_id"] for x in part] == ["r0", "r1", "r2"] and scored == ["r0", "r1", "r2"]
    assert len(decl) >= 2 and ylds == ["yield"]  # declared at start + before the yield; one yield
    assert not (tmp_path / "out/child.results.jsonl").exists()

    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))  # the rerun, on a disk with room
    assert F.score(a) == 0
    assert scored == ["r0", "r1", "r2", "r3", "r4", "r5", "r6"]  # only the 4 missing rows were scored again
    res = [json.loads(x)["run_id"] for x in (tmp_path / "out/child.results.jsonl").read_text().splitlines()]
    assert res == [f"r{i}" for i in range(7)]
    t = json.loads((tmp_path / "out/timing-child.json").read_text())
    assert t["rows_this_attempt"] == 4 and t["rows_resumed"] == 3


def test_runner_maps_16_prints_the_phase_log_tail_and_keeps_the_pt_out_of_the_slot():
    s = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert '16) fail 16 "$PH: worker disk low' in s
    assert 'tail -n 40 "$WORK/logs/$PH.log"' in s
    assert "CKCELL=$WORK/ckpt-cell" in s and '--ck-dir "$CKCELL"' in s and '--ckpt-dir "$CKCELL"' in s
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_fulldev_eval_champ.json").read_text(encoding="utf-8"))
    assert p["submit"]["key"].endswith("20261009d") and p["resume"] == "checkpoint"

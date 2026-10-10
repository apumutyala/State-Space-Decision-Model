"""Modes fulldev_eval_d13 / fulldev_eval_d13_v3w: D13 (job-hjwpq art-cr9a2, the joint_record head) on the full 12,104-row
Index DEV from its own lineage (8d92f04), plus the two failures the D1 champion full-DEV runs hit:
  * job-4ha7t: the full-DEV plan pass wrote nothing for ~40 min -> watchdog withheld the heartbeat -> killed;
  * job-afpc2: the worker's SHARED disk filled mid-scoring -> child rc 1, the phase tgz unwritable, artifact refused,
    and no checkpoint slot was ever declared, so nothing resumed.
CPU only: no model, no kit (harness.suite stubbed where score() needs it)."""
import hashlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from jobs.m2_first_cell import checkpoint_loader as L
from jobs.m2_first_cell import fulldev_eval as F

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
MODES = ("fulldev_eval_d13", "fulldev_eval_d13_v3w")
BEST, LAST = "trainable-step0001600.safetensors", "trainable-step0001901.safetensors"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _export_artifact(t: Path, best_sha=None):
    """A stand-in best_ckpt export: last + best safetensors (+ sidecars), latest.json naming the last with a best
    pointer (train.ckpt_export --best layout)."""
    st, side = b"fake-safetensors-best" * 50, b'{"step": 1600}'
    latest = json.dumps({"path": LAST, "step": 1901, "best": {"path": BEST, "step": 1600,
                                                               "sha256": best_sha or _sha(st)}}).encode()
    tar = t / "art.tar"
    with tarfile.open(tar, "w") as tf:
        for name, b in ((f"./ckpt/{BEST}", st), ("./ckpt/trainable-step0001600.json", side),
                        (f"./ckpt/{LAST}", b"last" * 10), ("./ckpt/latest.json", latest)):
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    cfg = {"parent": "rwkv7", "head": "joint_record", "seed": 1}
    cell = t / "cell.json"
    cell.write_text(json.dumps({"config": cfg, "ladder": {"T_max": 2048}}), encoding="utf-8")
    cc = t / "cfg.json"
    cc.write_text(json.dumps(cfg), encoding="utf-8")
    block = {"artifact": "art-test1", "job": "job-test1", "artifact_sha256": _sha(tar.read_bytes()),
             "member": f"./ckpt/{BEST}", "ckpt_sha256": _sha(st), "sidecar_sha256": _sha(side),
             "latest_json_sha256": _sha(latest), "cell_json_sha256": _sha(cell.read_bytes()), "url_env": "CKPT_URL",
             "select": "best"}
    return tar, cell, cc, block


def _install(t, tar, cell, cc, block):
    return L.install(block, cell_src=cell, cell_config=cc, ck_dir=t / "ck", work=t / "work", url=tar.as_uri())


# ------------------------------------------------------------------------------------------------ loader
def test_best_export_installs_with_its_sidecar_and_the_tar_is_removed(tmp_path):
    tar, cell, cc, block = _export_artifact(tmp_path)
    rec = _install(tmp_path, tar, cell, cc, block)
    assert rec["file"] == BEST and rec["select"] == "best" and rec["reused"] is False
    assert L.sha256_file(tmp_path / "ck" / BEST) == block["ckpt_sha256"]
    assert L.sha256_file(tmp_path / "ck/trainable-step0001600.json") == block["sidecar_sha256"]
    assert not (tmp_path / "ck" / LAST).exists()  # only the pinned members are read
    assert not (tmp_path / "work/artifact.tar").exists() and tar.exists()  # job-afpc2: no 2nd copy left on disk
    assert _install(tmp_path, tar, cell, cc, block)["reused"] is True


def test_best_needs_the_pointer_to_name_it_with_the_pinned_sha(tmp_path):
    tar, cell, cc, block = _export_artifact(tmp_path, best_sha="0" * 64)  # pointer sha != the member's
    with pytest.raises(L.CheckpointError, match="best does not name"):
        _install(tmp_path, tar, cell, cc, block)
    t2 = tmp_path / "b"
    t2.mkdir()
    tar, cell, cc, block = _export_artifact(t2)
    block.pop("select")  # no select: the D19-lineage loader (97cdb4d) still accepts the best pointer naming it
    assert _install(t2, tar, cell, cc, block)["role"] == "best"
    with pytest.raises(L.CheckpointError, match="select"):
        L.checkpoint_block({"checkpoint": {**block, "select": "any"}})


# ------------------------------------------------------------------------------------------------ params
@pytest.mark.parametrize("mode", MODES)
def test_d13_params_pin_the_best_export_and_price_under_the_caps(mode):
    from jobs.common.costs import schedule_minutes

    p = json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))
    ck = L.checkpoint_block(p)
    assert ck["artifact"] == "art-cr9a2" and ck["job"] == "job-hjwpq" and ck["select"] == "best"
    assert ck["artifact_sha256"] == "ff586289941e5477793c8acef2254128428ebd8fcd238a48cb98973a2d385fd5"
    assert ck["member"] == f"./ckpt/{BEST}" and ck["ckpt_sha256"].startswith("61c19d16")
    assert "<" in p["submit"]["env"]["CKPT_URL"]  # the signed URL is never committed
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    src = M / "m2_first_cell_rwkv7_d13_joint_safe_nofla.json"  # D13's training cell (not on D19-lineage branches)
    if src.is_file():
        assert cfg == json.loads(src.read_text(encoding="utf-8"))
    assert cfg["head"] == "joint_record"
    ro = F.readout_spec(p["readout"], cfg)
    assert p["readout_template"] == ro["template"] and ro["native"] == (mode == "fulldev_eval_d13")
    e = F.estimate(p)
    assert e["ok"] and e["readout"] == p["readout"]
    assert schedule_minutes(p) == p["max_duration_minutes"] <= 900  # the platform's job ceiling
    assert p["evals_fixed_minutes"]["child"] >= 30 + 12_104 * 7 / 2_008  # the plan pass is priced (job-4ha7t)
    assert p["max_input_tokens"] == 65536 and p.get("rows", "full") == "full"
    # v3w re-keyed 20261009c with max_duration 720 / fulldev cap 686 (quoted hold $45.36)
    assert p["submit"]["key"].endswith({"fulldev_eval_d13": "hjwpq-best1600-20261009a",
                                        "fulldev_eval_d13_v3w": "hjwpq-best1600-20261009c"}[mode])
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}")
    keys = {json.loads((M / f"job_params_{m}.json").read_text(encoding="utf-8"))["submit"]["key"] for m in MODES}
    assert len(keys) == 2


# ------------------------------------------------------------------------------------------------ job-4ha7t / afpc2
def test_plan_pass_beat_is_throttled(tmp_path, monkeypatch):
    beat = tmp_path / "beat"
    monkeypatch.setenv("SSDM_PROGRESS_BEAT", str(beat))
    clock = [0.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    touches = []
    real = Path.touch
    monkeypatch.setattr(Path, "touch", lambda self, *a, **k: (touches.append(clock[0]), real(self, *a, **k)))
    b = F.progress_beat(30.0)
    for _ in range(12_104):
        b()
        clock[0] += 0.21
    gaps = [y - x for x, y in zip(touches, touches[1:])]
    assert beat.exists() and max(gaps) < 31 and len(touches) < 200


def test_declarer_respects_the_platform_floor(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    posts = []
    d = F.checkpoint_declarer(post=lambda: posts.append(clock[0]) or True)
    d(force=True)
    for _ in range(12_104):
        clock[0] += 2.5
        d()
    gaps = [y - x for x, y in zip(posts, posts[1:])]
    assert min(gaps) >= 300 and max(gaps) < F.CHECKPOINT_EVERY_S + 3


def test_disk_watch_and_stop_for_disk(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    free = {"a": 50 << 30}
    w = F.disk_watch(["a"], usage=lambda q: free[q])
    assert w() is None
    free["a"] = 1 << 20
    assert w() is None  # throttled
    clock[0] += 31
    assert w() == {"a": 1 << 20}
    ylds = []
    for _ in range(3):
        assert F.stop_for_disk(tmp_path, lambda force=False: True, "t", yield_post=lambda: ylds.append(1) or True) == 16
    assert len(ylds) == F.MAX_YIELDS


def _score_env(monkeypatch, tmp_path, n_rows=7):
    if not sys.flags.utf8_mode:  # the harness package refuses to import otherwise (the launcher runs -X utf8)
        pytest.skip("needs python -X utf8")
    from train import m2_gates as G

    rows = [{"_evaluation": {"run_id": f"r{i}"}, "state": "s", "questions": []} for i in range(n_rows)]
    monkeypatch.setattr(F, "N_DEV", n_rows)
    monkeypatch.setattr(F, "selected_ids", lambda s: None)
    monkeypatch.setitem(sys.modules, "harness.suite", types.SimpleNamespace(read_rows=lambda p: rows))
    calls = []

    def load_child(a, **kw):
        calls.append(kw)
        return types.SimpleNamespace(decide=lambda req: {}), not kw, {"arm": "child"}

    monkeypatch.setattr(F, "load_child", load_child)
    monkeypatch.setattr(F, "plan_rows", lambda d, r, m: {"n_rows": len(r), "n_refused": 0, "n_over_16384": 0,
                                                         "max_row_tokens": 1})

    class _Prog:
        def __init__(self, p):
            pass

        def plan(self, w):
            pass

        def wrap(self, fn):
            return fn

    scored = []

    def predict(decide, rs, untempered_from_raw):
        rid = rs[0]["_evaluation"]["run_id"]
        scored.append(rid)
        return {rid: {"run_id": rid, "status": "ok", "response": {"x": 1}, "total_wall_ms": 5}}

    monkeypatch.setattr(G, "Progress", _Prog)
    monkeypatch.setattr(G, "beat_wrap", lambda fn: fn)
    monkeypatch.setattr(G, "unit_weight", lambda r: 1)
    monkeypatch.setattr(G, "index_predict", predict)
    cell = tmp_path / "cell.json"
    cfg = json.loads((M / "m2_first_cell_rwkv7_fulldev_eval_d13.json").read_text(encoding="utf-8"))
    cell.write_text(json.dumps({"config": cfg}), encoding="utf-8")
    a = types.SimpleNamespace(rows="rows", rows_select="full", arm="child", readout="native", max_input_tokens=65536,
                              cell=str(cell), out_dir=str(tmp_path / "out"), resume_dir=str(tmp_path / "ck"),
                              progress=str(tmp_path / "p.jsonl"))
    return a, scored, calls


def test_low_disk_mid_scoring_exits_16_keeps_rows_and_the_rerun_resumes(monkeypatch, tmp_path):
    a, scored, _ = _score_env(monkeypatch, tmp_path)
    decl, ylds = [], []
    monkeypatch.setattr(F, "_meta_post", lambda door: (decl if door == "checkpoint" else ylds).append(door) or True)
    n = [0]

    def watch(paths, **k):
        def low():
            n[0] += 1
            return {"ck": 1 << 20} if n[0] > 3 else None
        return low

    monkeypatch.setattr(F, "disk_watch", watch)
    assert F.score(a) == F.DISK_LOW_RC
    part = [json.loads(x)["run_id"] for x in (tmp_path / "ck/child.partial.jsonl").read_text().splitlines()]
    assert part == ["r0", "r1", "r2"] and len(decl) >= 2 and ylds == ["yield"]
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    assert F.score(a) == 0
    assert scored == [f"r{i}" for i in range(7)]  # the rerun scored only the 4 missing rows
    t = json.loads((tmp_path / "out/timing-child.json").read_text())
    assert t["rows_this_attempt"] == 4 and t["rows_resumed"] == 3


@pytest.mark.parametrize("readout,kw", [("native", {}),
                                        ("v3w", {"temperature": False, "readout": ("jrt-rwkv-g1-v3w", "g1v2")})])
def test_readout_selects_the_tempered_native_or_the_untempered_override(monkeypatch, tmp_path, readout, kw):
    a, _, calls = _score_env(monkeypatch, tmp_path)
    monkeypatch.setattr(F, "disk_watch", lambda paths, **k: (lambda: None))
    a.readout = readout
    assert F.score(a) == 0
    assert calls == [kw]
    ident = json.loads((tmp_path / "out/identity-child.json").read_text())
    assert ident["readout"]["name"] == readout and ident["readout"]["native"] == (readout == "native")
    assert (tmp_path / "out/child.results_served.jsonl").is_file() == (readout == "native")


def test_runner_threads_the_readout_keeps_the_ckpt_out_of_the_slot_and_maps_16():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert "CKCELL=$WORK/ckpt-cell" in sh and '--ck-dir "$CKCELL"' in sh and '--ckpt-dir "$CKCELL"' in sh
    assert '--readout "$FD_READOUT"' in sh and "p.get('readout') or 'native'" in sh
    assert '16) fail 16 "$PH: worker disk low' in sh and 'tail -n 40 "$WORK/logs/$PH.log"' in sh

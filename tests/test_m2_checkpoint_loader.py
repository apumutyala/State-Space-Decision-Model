"""jobs/m2_first_cell/checkpoint_loader.py: the shared saved-checkpoint installer (hijack_eval / fulldev_eval). CPU, no
model: a stand-in artifact tar with the training run's layout (./ckpt/<file>.pt + ./ckpt/latest.json) via file://."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from jobs.m2_first_cell import checkpoint_loader as L

ROOT = Path(__file__).resolve().parents[1]
CFG = {"parent": "rwkv7", "head": "fork", "seed": 1}
PT = "ckpt-step0000006.pt"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _artifact(t: Path, pt_bytes=b"\x80fake-checkpoint" * 100, latest=None, extra=()):
    latest = json.dumps({"path": PT, "step": 6, "config_hash": "x"}).encode() if latest is None else latest
    tar = t / "art.tar"
    with tarfile.open(tar, "w") as tf:
        for name, b in ((f"./ckpt/{PT}", pt_bytes), ("./ckpt/latest.json", latest), ("./MANIFEST.jsonl", b"{}\n"),
                        *extra):
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    cell = t / "cell.json"
    cell.write_text(json.dumps({"config": CFG, "ladder": {"T_max": 512}}), encoding="utf-8")
    cfg = t / "cfg.json"
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    block = {"artifact": "art-test1", "job": "job-test1", "artifact_sha256": _sha(tar.read_bytes()),
             "member": f"./ckpt/{PT}", "ckpt_sha256": _sha(pt_bytes), "latest_json_sha256": _sha(latest),
             "cell_json_sha256": _sha(cell.read_bytes()), "url_env": "CKPT_URL"}
    return tar, cell, cfg, block


def _install(t, tar, cell, cfg, block, **kw):
    return L.install(block, cell_src=cell, cell_config=cfg, ck_dir=t / "ck", work=t / "work",
                     cell_out=t / "out/cell.json", url=kw.pop("url", tar.as_uri()), **kw)


def test_installs_pinned_checkpoint_and_reuses_it_on_restart(tmp_path):
    tar, cell, cfg, block = _artifact(tmp_path)
    rec = _install(tmp_path, tar, cell, cfg, block)
    assert rec["sha256"] == block["ckpt_sha256"] and rec["file"] == PT and rec["reused"] is False
    assert L.sha256_file(tmp_path / "ck" / PT) == block["ckpt_sha256"]
    assert json.loads((tmp_path / "ck/latest.json").read_text())["path"] == PT
    assert (tmp_path / "out/cell.json").read_bytes() == cell.read_bytes()
    assert not (tmp_path / "ck/MANIFEST.jsonl").exists()  # only the two named members are read
    rec2 = _install(tmp_path, tar, cell, cfg, block, url="https://unused.invalid/never-fetched")
    assert rec2["reused"] is True


def test_url_from_env(tmp_path, monkeypatch):
    tar, cell, cfg, block = _artifact(tmp_path)
    monkeypatch.setenv("CKPT_URL", tar.as_uri())
    assert L.install(block, cell_src=cell, cell_config=cfg, ck_dir=tmp_path / "ck", work=tmp_path / "w")["sha256"]


@pytest.mark.parametrize("bad,code", [
    ("artifact_sha256", 10), ("ckpt_sha256", 10), ("latest_json_sha256", 10), ("cell_json_sha256", 10)])
def test_every_pin_is_checked(tmp_path, bad, code):
    tar, cell, cfg, block = _artifact(tmp_path)
    block[bad] = "0" * 64
    with pytest.raises(L.CheckpointError) as e:
        _install(tmp_path, tar, cell, cfg, block)
    assert e.value.code == code
    assert not (tmp_path / "ck" / PT).exists()  # a file that fails its pin is never installed


def test_config_mismatch_and_latest_naming_another_file_are_refused(tmp_path):
    tar, cell, cfg, block = _artifact(tmp_path)
    cfg.write_text(json.dumps({**CFG, "seed": 2}), encoding="utf-8")
    with pytest.raises(L.CheckpointError, match="seed"):
        _install(tmp_path, tar, cell, cfg, block)
    t2 = tmp_path / "b"
    t2.mkdir()
    tar, cell, cfg, block = _artifact(t2, latest=json.dumps({"path": "ckpt-step0000003.pt"}).encode())
    with pytest.raises(L.CheckpointError, match="names neither"):
        _install(t2, tar, cell, cfg, block)


def test_block_validation_and_url_scheme(tmp_path):
    tar, cell, cfg, block = _artifact(tmp_path)
    for k, v in (("member", "../../etc/x.pt"), ("url_env", "lower"), ("ckpt_sha256", "zz")):
        with pytest.raises(L.CheckpointError) as e:
            L.checkpoint_block({"checkpoint": {**block, k: v}})
        assert e.value.code == 5
    with pytest.raises(L.CheckpointError) as e:
        _install(tmp_path, tar, cell, cfg, block, url="http://plain.invalid/x")
    assert e.value.code == 5


def test_cli_exit_codes(tmp_path):
    tar, cell, cfg, block = _artifact(tmp_path)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"checkpoint": block}), encoding="utf-8")
    args = ["install", "--params", str(params), "--cell-src", str(cell), "--cell-config", str(cfg),
            "--ck-dir", str(tmp_path / "ck"), "--work", str(tmp_path / "w"), "--url", tar.as_uri(),
            "--out", str(tmp_path / "rec.json")]
    assert L.main(args) == 0 and json.loads((tmp_path / "rec.json").read_text())["file"] == PT
    params.write_text(json.dumps({"checkpoint": {**block, "artifact_sha256": "1" * 64}}), encoding="utf-8")
    assert L.main(args[:-6] + ["--ck-dir", str(tmp_path / "ck2"), "--work", str(tmp_path / "w2"), "--url",
                               tar.as_uri()]) == 10


def test_hijack_eval_params_pin_the_d1_artifact():
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_hijack_eval.json").read_text(encoding="utf-8"))
    ck = L.checkpoint_block(p)
    assert ck["artifact"] == "art-bm3e6" and ck["job"] == "job-c7pq2"
    assert ck["artifact_sha256"] == "a405907eb85f3b6bc25f287ef37066dcd192b0b2897e4ec45f336a6fe3580a18"
    assert ck["ckpt_sha256"] == "43414f022ccfe2eb7d770957ae851f5450839457d5a41e9f7fb661b3f81cfd32"


def _export_artifact(t: Path, pin: str):
    """An inference export of a best_ckpt run: last + best .safetensors with sidecars, latest.json's best pointer."""
    last, best = "trainable-step0000009.safetensors", "trainable-step0000006.safetensors"
    files = {last: b"last-weights" * 50, last[:-12] + ".json": b'{"step": 9}', best: b"best-weights" * 50,
             best[:-12] + ".json": b'{"step": 6}'}
    latest = json.dumps({"path": last, "step": 9, "best": {
        "path": best, "step": 6, "sha256": _sha(files[best]), "sidecar_sha256": _sha(files[best[:-12] + ".json"])}})
    tar = t / "art.tar"
    with tarfile.open(tar, "w") as tf:
        for name, b in [*((f"./ckpt/{k}", v) for k, v in files.items()), ("./ckpt/latest.json", latest.encode())]:
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    cell = t / "cell.json"
    cell.write_text(json.dumps({"config": CFG, "ladder": {"T_max": 512}}), encoding="utf-8")
    cfg = t / "cfg.json"
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    block = {"artifact": "art-test2", "job": "job-test2", "artifact_sha256": _sha(tar.read_bytes()),
             "member": f"./ckpt/{pin}", "ckpt_sha256": _sha(files[pin]),
             "sidecar_sha256": _sha(files[pin[:-12] + ".json"]), "latest_json_sha256": _sha(latest.encode()),
             "cell_json_sha256": _sha(cell.read_bytes()), "url_env": "CKPT_URL"}
    return tar, cell, cfg, block, last, best, files


def test_pinned_last_export_with_a_best_pointer_installs_and_names_the_scored_checkpoint(tmp_path):
    tar, cell, cfg, block, last, best, files = _export_artifact(tmp_path, "trainable-step0000009.safetensors")
    rec = _install(tmp_path, tar, cell, cfg, block)
    assert rec["role"] == "best" and rec["file"] == best and rec["path"] == str(tmp_path / "ck" / best)
    assert rec["sha256"] == _sha(files[best]) and rec["member"] == last and rec["member_sha256"] == block["ckpt_sha256"]
    assert (tmp_path / "ck" / best[:-12]).with_suffix(".json").read_bytes() == files[best[:-12] + ".json"]
    rec2 = _install(tmp_path, tar, cell, cfg, block, url="https://unused.invalid/never-fetched")  # restart: no fetch
    assert rec2["reused"] is True and rec2["file"] == best


def test_pinned_best_export_is_accepted_as_role_best(tmp_path):
    tar, cell, cfg, block, last, best, files = _export_artifact(tmp_path, "trainable-step0000006.safetensors")
    rec = _install(tmp_path, tar, cell, cfg, block)
    assert rec["role"] == "best" and rec["file"] == best and rec["member"] == best
    assert not (tmp_path / "ck" / last).exists()


def test_malformed_best_pointer_is_refused(tmp_path):
    tar, cell, cfg, block, last, best, files = _export_artifact(tmp_path, "trainable-step0000009.safetensors")
    rec = json.loads(tarfile.open(tar).extractfile("./ckpt/latest.json").read())
    rec["best"]["path"] = "../escape.safetensors"
    raw = json.dumps(rec).encode()
    t2 = tmp_path / "b"
    t2.mkdir()
    tar2 = t2 / "art.tar"
    with tarfile.open(tar) as src, tarfile.open(tar2, "w") as dst:
        for m in src.getmembers():
            b = raw if m.name == "./ckpt/latest.json" else src.extractfile(m).read()
            ti = tarfile.TarInfo(m.name)
            ti.size = len(b)
            dst.addfile(ti, io.BytesIO(b))
    block = {**block, "artifact_sha256": _sha(tar2.read_bytes()), "latest_json_sha256": _sha(raw)}
    with pytest.raises(L.CheckpointError, match="best pointer malformed"):
        _install(t2, tar2, cell, cfg, block)

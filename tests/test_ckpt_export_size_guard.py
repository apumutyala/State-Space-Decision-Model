"""job-igcu4 fix (CPU only): the first-cell job uploads an INFERENCE checkpoint export (train.ckpt_export) instead of
the full optimizer .pt, and jobs/common/size_guard.py keeps $GMN_OUTPUT_DIR under the artifact store's PUT cap
without ever dropping results / units / report files."""

import hashlib
import json
import tarfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from jobs.common.size_guard import guard  # noqa: E402
from jobs.m2_first_cell import checkpoint_loader as CL  # noqa: E402
from train import ckpt_export as CE  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")


def _trainer(d, seed=0):
    torch.manual_seed(seed)
    m = DecisionModel(tiny_backbone(seed=0, hidden=16), "integrator")
    items = items_from_records(toy_records(16, seed=0), ENC, template=TOY_TEMPLATE)
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=8, max_steps=3, ckpt_dir=str(d), lr_head=1e-2,
                      lr_lora=1e-2)
    return Trainer(m, cfg, items, guard=G)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_inference_export_round_trips_the_trainable_tensors(tmp_path, dtype):
    t = _trainer(tmp_path / "ck")
    t.train()
    pt = t.save()
    want = t.trainable_state()
    side = CE.export_inference(pt, tmp_path / "out", dtype)
    st = tmp_path / "out" / side["file"]
    assert side["file"] == "trainable-step0000003.safetensors" and side["source_ckpt_sha256"] == _sha(pt)
    assert json.loads((tmp_path / "out" / "latest.json").read_text())["path"] == st.name
    line = (tmp_path / "out" / (st.name + ".sha256")).read_text().split()
    assert line == [_sha(st), st.name]
    assert b"\r" not in (tmp_path / "out" / (st.name + ".sha256")).read_bytes()  # LF: sha256sum -c
    assert st.stat().st_size < pt.stat().st_size  # no Adam / scheduler / RNG state
    again = CE.export_inference(pt, tmp_path / "again", dtype)  # byte-deterministic (pinnable by sha256)
    assert again["sha256"] == side["sha256"]
    assert (tmp_path / "again" / CE.sidecar_path(st).name).read_bytes() == CE.sidecar_path(st).read_bytes()
    # read back: same names, shapes; values exact (fp32) or the bf16 rounding of the trained values
    tensors, back = CE.read_inference(st)
    assert tensors.keys() == want.keys() and back["config_hash"] == t.cfg.hash() and back["step"] == 3
    for k, v in want.items():
        assert torch.equal(tensors[k], v.to(CE.DTYPES[dtype])), k
    # a FRESH trainer restores it through Trainer.load (config hash + lineage checked) into the same trainable tensors
    u = _trainer(tmp_path / "ck2", seed=1)
    u.load(st)
    assert u.loaded_format == "inference" and u.step == 3 and u.history == t.history
    got = u.trainable_state()
    for k, v in want.items():
        assert torch.equal(got[k], v.to(CE.DTYPES[dtype]).to(v.dtype)), k
    # the D14 init_from reader takes either format and yields the same tensors
    tr2, ident = CE.read_trainable(st)
    assert ident["step"] == 3 and all(torch.equal(tr2[k], tensors[k]) for k in tensors)
    # eval only: resume refuses an export (no optimizer state)
    (tmp_path / "ck2").mkdir(exist_ok=True)
    CE.export_inference(pt, tmp_path / "ck2", dtype)
    with pytest.raises(ValueError, match="inference export"):
        _trainer(tmp_path / "ck2").resume()
    # a tampered file is refused (a copy: the original is still memory-mapped by the tensors above on Windows)
    bad = tmp_path / "bad" / st.name
    bad.parent.mkdir()
    b = bytearray(st.read_bytes())
    b[-1] ^= 1
    bad.write_bytes(bytes(b))
    CE.sidecar_path(bad).write_bytes(CE.sidecar_path(st).read_bytes())
    with pytest.raises(ValueError, match="sha256"):
        _trainer(tmp_path / "ck3").load(bad)


def test_checkpoint_loader_installs_an_inference_export(tmp_path):
    t = _trainer(tmp_path / "ck")
    t.train()
    side = CE.export_inference(t.save(), tmp_path / "out" / "ckpt", "bf16")
    st, sc = tmp_path / "out" / "ckpt" / side["file"], CE.sidecar_path(tmp_path / "out" / "ckpt" / side["file"])
    tar = tmp_path / "art.tar"
    with tarfile.open(tar, "w") as tf:
        for f in (st, sc, tmp_path / "out" / "ckpt" / "latest.json"):
            tf.add(f, arcname=f"./ckpt/{f.name}")
    cfg = {"x": 1}
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"config": cfg, "ladder": {"T_max": 8}}))
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    block = {"artifact": "art-x", "artifact_sha256": _sha(tar), "member": f"./ckpt/{st.name}", "ckpt_sha256": _sha(st),
             "latest_json_sha256": _sha(tmp_path / "out" / "ckpt" / "latest.json"), "cell_json_sha256": _sha(cell),
             "sidecar_sha256": _sha(sc), "url_env": "CKPT_URL"}
    rec = CL.install(block, cell_src=cell, cell_config=tmp_path / "cfg.json", ck_dir=tmp_path / "inst",
                     work=tmp_path / "w", url=tar.as_uri())
    assert rec["file"] == st.name and not rec["reused"]
    p = Trainer.latest(tmp_path / "inst")
    assert p.name == st.name and CE.sidecar_path(p).is_file()
    u = _trainer(tmp_path / "ck4", seed=2)
    u.load(p)
    assert u.step == 3
    with pytest.raises(CL.CheckpointError):
        CL.checkpoint_block({"checkpoint": {k: v for k, v in block.items() if k != "sidecar_sha256"}})


def _put(p, n):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.truncate(n)


def test_size_guard_keeps_results_and_evicts_the_largest_non_essential(tmp_path):
    out = tmp_path / "out"
    keep = ["result.json", "MANIFEST.jsonl", "M2_FIRST_CELL_DONE", "units-child-index.json", "report.json",
            "phases/final.tgz", "results.jsonl"]
    for k in keep:
        _put(out / k, 300)
    _put(out / "phases/final.tgz", 2_000)  # the full results tarball: large but essential
    _put(out / "ckpt/trainable-step0000003.safetensors", 5_000)
    _put(out / "ckpt/trainable-step0000003.json", 50)
    _put(out / "phases/cell.tgz", 1_500)
    _put(out / "phases/setup.tgz", 100)
    rec = guard(out, cap=6_000, evict_dir=tmp_path / "evicted")
    assert rec["fits"] and [e["file"] for e in rec["evicted"]] == ["ckpt/trainable-step0000003.safetensors"]
    assert (tmp_path / "evicted/ckpt/trainable-step0000003.safetensors").is_file()  # moved, not deleted
    for k in keep:
        assert (out / k).is_file(), k
    assert (out / "phases/cell.tgz").is_file()  # only as much as needed
    assert json.loads((out / "SIZE_GUARD.json").read_text())["evicted"][0]["bytes"] == 5_000
    # tighter cap: earlier phase tarballs go too, results still never
    rec = guard(out, cap=4_000, evict_dir=tmp_path / "evicted")
    assert {e["file"] for e in rec["evicted"]} >= {"phases/cell.tgz"}
    for k in keep:
        assert (out / k).is_file(), k
    # under the cap: nothing touched, no log rewritten
    before = sorted(p.name for p in out.rglob("*"))
    assert guard(out, cap=10**9)["evicted"] == [] and sorted(p.name for p in out.rglob("*")) == before


def test_job_script_never_copies_the_full_checkpoint_to_output():
    from pathlib import Path

    s = (Path(__file__).resolve().parents[1] / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert 'cp "$CK/cell/$LATEST"' not in s
    assert "train.ckpt_export inference" in s and s.count("size_guard.py") == 2

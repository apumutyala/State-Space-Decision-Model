"""job-igcu4 fix (CPU only): the first-cell job uploads an INFERENCE checkpoint export (train.ckpt_export) instead of
the full optimizer .pt, and jobs/common/size_guard.py keeps $GMN_OUTPUT_DIR under the artifact store's PUT cap
without ever dropping results / units / report files."""

import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from jobs.common.size_guard import guard, main  # noqa: E402
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
    # read back: same names, shapes; LoRA exact (fp32) or its bf16 rounding; every NON-LoRA (head) tensor exact in
    # its source dtype (agent-b #231 review: eval runs without autocast, so a bf16 head moved logprobs by 4.3e-3)
    tensors, back = CE.read_inference(st)
    assert tensors.keys() == want.keys() and back["config_hash"] == t.cfg.hash() and back["step"] == 3
    head = [k for k in want if not CE.is_lora(k)]
    assert head and any(CE.is_lora(k) for k in want)
    for k, v in want.items():
        exp = v.to(CE.DTYPES[dtype]) if CE.is_lora(k) else v
        assert tensors[k].dtype == exp.dtype and torch.equal(tensors[k], exp), k
    assert all(back["tensor_dtypes"][k] == "float32" for k in head) and back["lora_dtype"] == dtype
    # a FRESH trainer restores it through Trainer.load (config hash + lineage checked) into the same trainable tensors
    u = _trainer(tmp_path / "ck2", seed=1)
    u.load(st)
    assert u.loaded_format == "inference" and u.step == 3 and u.history == t.history
    got = u.trainable_state()
    for k, v in want.items():
        assert torch.equal(got[k], v.to(CE.DTYPES[dtype]).to(v.dtype) if CE.is_lora(k) else v), k
    # the D14 init_from reader takes either format and yields the same tensors
    tr2, ident = CE.read_trainable(st)
    assert ident["step"] == 3 and all(torch.equal(tr2[k], tensors[k]) for k in tensors)
    # the export never lands in the training ckpt dir (its latest.json must keep naming the .pt: review finding 5)
    latest_before = (tmp_path / "ck" / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="training checkpoint dir"):
        CE.export_inference(pt, tmp_path / "ck", dtype)
    assert (tmp_path / "ck" / "latest.json").read_bytes() == latest_before
    other = tmp_path / "other"  # any dir holding a training latest.json is refused too
    other.mkdir()
    (other / "latest.json").write_bytes(latest_before)
    with pytest.raises(ValueError, match="training checkpoint dir"):
        CE.export_inference(pt, other, dtype)
    r = _trainer(tmp_path / "ck", seed=4)  # so an interrupted job still resumes from the full .pt after an export
    assert r.resume() and r.loaded_format == "full" and r.step == 3
    # eval only: resume refuses a dir whose latest.json names an export (no optimizer state)
    with pytest.raises(ValueError, match="inference export"):
        _trainer(tmp_path / "out").resume()
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


def _put(p, n, byte=b"\0"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(byte * n)


def _size(d):
    return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())


def _run(out):
    """A first-cell /output: results files of every naming (none of them evictable), cumulative phase tarballs, the
    final one, and the ckpt export with its sidecar, .sha256 and latest.json."""
    results = {"result.json": 300, "MANIFEST.jsonl": 100, "M2_FIRST_CELL_DONE": 20, "SHA256SUMS": 100,
               "ece-child.json": 50, "predictions-dev.jsonl": 900, "receipt.json": 50, "calibration.json": 50,
               "index.json": 50, "g12.json": 50, "card.json": 50}
    for k, n in results.items():
        _put(out / k, n, b"\n" if k == "MANIFEST.jsonl" else b"\0")  # MANIFEST: blank lines (lines get appended)
    for k, (name, n) in enumerate((("setup", 100), ("guard", 200), ("cell", 450), ("evals", 450))):
        _put(out / f"phases/{name}.tgz", n, bytes([k + 1]))
        os.utime(out / f"phases/{name}.tgz", (1_000 + k, 1_000 + k))  # export order
    _put(out / "phases/final.tgz", 500, b"F")
    ck = out / "ckpt"
    _put(ck / "trainable-step0001439.safetensors", 2_300, b"W")
    for f in ("trainable-step0001439.json", "trainable-step0001439.safetensors.sha256", "latest.json"):
        _put(ck / f, 40, b"j")
    return results


def test_size_guard_allowlist_order_and_never_deletes(tmp_path):
    out, ev = tmp_path / "out", tmp_path / "ck" / "evicted"
    results = _run(out)
    # (2) superseded tarballs go first, OLDEST first; the ckpt export stays when that is enough
    rec = guard(out, cap=_size(out) - 600, evict_dir=ev)
    assert rec["fits"] and not rec["ckpt_evicted"]
    assert [e["item"] for e in rec["evicted"]] == ["phases/setup.tgz", "phases/guard.tgz", "phases/cell.tgz"]
    assert (out / "ckpt/trainable-step0001439.safetensors").is_file() and (out / "phases/evals.tgz").is_file()
    for e in rec["evicted"]:  # (3) moved, never deleted, byte for byte
        assert Path(e["to"]).is_file() and Path(e["to"]).stat().st_size == e["bytes"]
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines() if x.strip().startswith("{")]
    assert [m["evicted"] for m in man] == ["phases/setup.tgz", "phases/guard.tgz", "phases/cell.tgz"]
    # tighter: the ckpt export leaves LAST and as ONE unit (no latest.json / sidecar left naming a missing file)
    rec = guard(out, cap=2_600, evict_dir=ev)
    assert rec["ckpt_evicted"] and [e["item"] for e in rec["evicted"]][-2:] == ["phases/evals.tgz", "ckpt"]
    assert not (out / "ckpt").exists() and (ev / "ckpt/latest.json").is_file()
    assert (ev / "ckpt/trainable-step0001439.safetensors").read_bytes() == b"W" * 2_300
    # (1) allow-list: nothing else is ever touched, even when that means the dir cannot fit (fails loudly)
    rec = guard(out, cap=100, evict_dir=ev)
    assert rec["fits"] is False
    for k in results:
        assert (out / k).is_file(), k
    assert (out / "phases/final.tgz").read_bytes() == b"F" * 500
    assert main(["--out", str(out), "--cap-bytes", "100", "--evict-dir", str(ev)]) == 3
    # the log merges across calls and finalize carries it into result.json
    log = json.loads((out / "SIZE_GUARD.json").read_text())
    assert log["ckpt_evicted"] and len(log["evicted"]) == 5
    from jobs.common import finalize

    finalize.main([str(out / "result.json"), "9", "{}", "0", "1", str(tmp_path / "none.json"), str(out),
                   "M2_FIRST_CELL_DONE", "jobs.m2_first_cell.summarize:valid_result"])
    res = json.loads((out / "result.json").read_text())
    assert res["size_guard"]["ckpt_evicted"] is True and res["size_guard"]["fits"] is False


def test_size_guard_cross_filesystem_copies_verifies_or_leaves_in_place(tmp_path, monkeypatch):
    import jobs.common.size_guard as SG

    out, ev = tmp_path / "out", tmp_path / "ev"
    _run(out)

    def no_rename(a, b):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(SG.os, "rename", no_rename)
    # finalize window (rename only): nothing moves, nothing is deleted, the guard fails loudly
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    rec = SG.guard(out, cap=_size(out) - 600, evict_dir=ev, rename_only=True)
    assert rec["fits"] is False and rec["evicted"] == [] and len(rec["left_in_place"]) >= 1
    assert all((out / b).is_file() for b in before)
    # a copy that does not verify: the source stays
    real_copy = SG.shutil.copy2

    def bad_copy(a, b, **kw):
        real_copy(a, b)
        Path(b).write_bytes(b"corrupt")

    monkeypatch.setattr(SG.shutil, "copy2", bad_copy)
    rec = SG.guard(out, cap=_size(out) - 600, evict_dir=ev / "bad")
    tars = ["phases/setup.tgz", "phases/guard.tgz", "phases/cell.tgz", "phases/evals.tgz"]
    assert [x["item"] for x in rec["left_in_place"]] == tars and all("sha256" in x["why"] for x in rec["left_in_place"])
    assert all((out / t).is_file() for t in tars)  # a failed verify never costs the source
    # ... and the ckpt dir (copytree, verified per file) went instead, as one unit
    assert rec["fits"] and rec["ckpt_evicted"] and [e["item"] for e in rec["evicted"]] == ["ckpt"]
    assert "verified" in rec["evicted"][0]["how"] and not (out / "ckpt").exists()
    assert (ev / "bad/ckpt/trainable-step0001439.safetensors").read_bytes() == b"W" * 2_300
    # a working copy: copied, sha256-verified, then the source leaves
    monkeypatch.setattr(SG.shutil, "copy2", real_copy)
    rec = SG.guard(out, cap=_size(out) - 500, evict_dir=ev / "good")
    moved = [e for e in rec["evicted"] if "good" in e["to"]]
    assert rec["fits"] and [e["item"] for e in moved] == tars[:3]  # oldest first, only as many as needed
    assert all("verified" in e["how"] and (ev / "good" / e["item"]).is_file() and not (out / e["item"]).exists()
               for e in moved) and (out / "phases/evals.tgz").is_file()


def test_size_guard_reserves_room_for_the_final_export(tmp_path):
    out = tmp_path / "out"
    _run(out)
    (out / "phases/final.tgz").rename(tmp_path / "final.tgz")  # mid-run: no final export yet
    rec = guard(out, cap=_size(out) + 100, evict_dir=tmp_path / "ev", reserve_final=True)
    assert rec["reserved_for_final_bytes"] == 450 and rec["fits"]  # the newest tarball estimates the final one
    assert rec["evicted"] and (out / "phases/evals.tgz").is_file()  # the newest tarball is protected until final


def test_job_script_never_copies_the_full_checkpoint_to_output():
    s = (Path(__file__).resolve().parents[1] / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert 'cp "$CK/cell/$LATEST"' not in s
    assert "train.ckpt_export inference" in s and s.count("size_guard.py") == 2
    assert s.count('--evict-dir "$CK/evicted"') == 2 and "--rename-only" in s and "--reserve-final" in s
    assert "$WORK/evicted" not in s  # never evict into scratch that does not outlive the job

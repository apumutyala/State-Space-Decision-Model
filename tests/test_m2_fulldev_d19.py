"""Modes fulldev_eval_d19_v3w_s1of2 / _s2of2: D19 (job-fi5xf art-p3gpi, the joint_record head on bschema-draw-v3b) on
the full 12,104-row Index DEV in the two pre-declared shards D13 used, scored exactly as D19's proxy600 screen scored it
(the untempered v3w arm of fulldev_eval.load_child, the BEST checkpoint step 2800). CPU only: no model, no kit."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from jobs.m2_first_cell import checkpoint_loader as L
from jobs.m2_first_cell import fulldev_eval as F

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
MODES = ("fulldev_eval_d19_v3w_s1of2", "fulldev_eval_d19_v3w_s2of2")
BEST, LAST = "trainable-step0002800.safetensors", "trainable-step0003111.safetensors"


def _p(mode):
    return json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("mode", MODES)
def test_d19_params_pin_the_screened_best_export(mode):
    p = _p(mode)
    ck = L.checkpoint_block(p)
    assert ck["artifact"] == "art-p3gpi" and ck["job"] == "job-fi5xf" and ck["select"] == "best"
    assert ck["artifact_sha256"] == "b898dbc495dea1637e06b78694bbd724bacbc5b1cdcc318ef698612d03404622"
    assert ck["member"] == f"./ckpt/{BEST}"
    assert ck["ckpt_sha256"] == "31062f4850713672a5ae7a8bb472455fda699e35b7176a6fc0d331e17b96d215"
    assert ck["sidecar_sha256"] == "6d1f1adf5485eef16731f7998149860df7fa96ca8f20f836521fedd6776e1669"
    assert p["submit"]["env"]["CKPT_URL"] == "<CKPT_URL>"  # the signed URL is never committed
    assert L.checkpoint_block(_p(MODES[0])) == L.checkpoint_block(_p(MODES[1]))  # both shards: the same bytes


@pytest.mark.parametrize("mode", MODES)
def test_d19_scoring_path_is_the_screens(mode):
    """The cell config is D19's training cell byte for byte; readout v3w = D19's own template (native) scored through
    load_child's untempered override, which is what jobs.m2_first_cell.screen score calls for D19's screen_readout v3w."""
    p = _p(mode)
    assert (ROOT / p["cell_config"]).read_bytes() == (M / "m2_first_cell_rwkv7_d19_v3w_v3b_joint.json").read_bytes()
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    assert cfg["head"] == "joint_record" and cfg["best_ckpt"] is True
    d19 = _p("d19_v3w_v3b_joint")
    assert d19["screen_readout"] == p["readout"] == "v3w"
    ro = F.readout_spec(p["readout"], cfg)
    assert ro == {"name": "v3w", "template": "jrt-rwkv-g1-v3w", "layout": "g1v2", "native": True}
    assert p["readout_template"] == ro["template"] and p["fulldev_parent_arm"] is False
    assert p["train_dir"] == d19["train_dir"] and p["train_set_pointer"] == d19["train_set_pointer"]
    # the lineage data the checkpoint's Trainer.load rebuilds: exactly D19's context data (no screen arms) + cell.json
    want = [d for d in d19["context"]["data"] if not d["dest"].startswith(("screen_reference/", "screen_parent/"))]
    assert p["context"]["data"][:-1] == want
    assert p["context"]["data"][-1] == {"src": "m2/checkpoints/job-fi5xf/cell.json", "dest": "checkpoint/cell.json",
                                        "pin_json": [f"jobs/m2_first_cell/job_params_{mode}.json", "checkpoint",
                                                     "cell_json_sha256"]}
    assert set(d19["context"]["manifest_files"]) <= set(p["context"]["manifest_files"])


@pytest.mark.parametrize("mode", MODES)
def test_d19_shard_params_price_with_headroom_and_fit_the_hold(mode):
    from jobs.common.costs import quoted_max_cost_usd, schedule_minutes, worst_case_usd

    k = int(mode[-4])
    p = _p(mode)
    d13 = _p(f"fulldev_eval_d13_v3w_s{k}of2")
    assert p["mode"] == mode and p["rows"] == d13["rows"] == f"shard{k}of2"
    e = F.estimate(p)
    assert e["ok"] and e["length_ratio"] == 0.9004  # v1 lengths -> v3w prompts even though v3w is D19's native
    assert e["headroom_pct"]["fulldev"] >= 25 and p["phase_caps_minutes"]["fulldev"] >= 1.25 * e["minutes"]["fulldev"]
    assert schedule_minutes(p) == p["max_duration_minutes"] and quoted_max_cost_usd(p) <= 25
    assert worst_case_usd(p) <= p["hard_cap_usd"]
    assert p["measured_rates_reference"]["child_s_per_1k_tokens"] == pytest.approx(1613.35 / (1_481_526 * 0.9004 / 1000),
                                                                                    abs=1e-3)
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}")
    assert "fulldev_shards_2.json" in " ".join(p["context"]["manifest_files"])
    keys = {_p(m)["submit"]["key"] for m in MODES + ("fulldev_eval_d13_v3w_s1of2", "fulldev_eval_d13_v3w_s2of2",
                                                     "fulldev_eval_d13_v3w", "fulldev_eval_d13", "d19_v3w_v3b_joint")}
    assert len(keys) == 7


def test_estimate_unchanged_for_the_d13_modes():
    assert F.estimate(_p("fulldev_eval_d13_v3w"))["minutes"] == {"fulldev": 500.0}
    assert F.estimate(_p("fulldev_eval_d13"))["length_ratio"] == 1.0
    assert F.estimate(_p("fulldev_eval_d13_v3w_s1of2"))["minutes"] == {"fulldev": 266.5}


def _artifact(t: Path, best_sha=None):
    """D19's export layout: latest.json names the last step with a best pointer carrying sha256 + sidecar_sha256."""
    st, side = b"fake-best" * 64, b'{"step": 2800}'
    latest = json.dumps({"path": LAST, "step": 3111, "best": {"path": BEST, "step": 2800, "sha256": best_sha or _sha(st),
                                                               "sidecar_sha256": _sha(side)}}).encode()
    tar = t / "art.tar"
    with tarfile.open(tar, "w") as tf:
        for name, b in ((f"./ckpt/{BEST}", st), ("./ckpt/trainable-step0002800.json", side),
                        (f"./ckpt/{LAST}", b"last" * 10), ("./ckpt/latest.json", latest)):
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    cfg = {"parent": "rwkv7", "head": "joint_record", "seed": 0}
    cell = t / "cell.json"
    cell.write_text(json.dumps({"config": cfg, "ladder": {"T_max": 2048}}), encoding="utf-8")
    cc = t / "cfg.json"
    cc.write_text(json.dumps(cfg), encoding="utf-8")
    block = {"artifact": "art-t", "job": "job-t", "artifact_sha256": _sha(tar.read_bytes()), "member": f"./ckpt/{BEST}",
             "ckpt_sha256": _sha(st), "sidecar_sha256": _sha(side), "latest_json_sha256": _sha(latest),
             "cell_json_sha256": _sha(cell.read_bytes()), "url_env": "CKPT_URL", "select": "best"}
    return tar, cell, cc, block


def test_loader_installs_the_pinned_best_and_names_it(tmp_path):
    tar, cell, cc, block = _artifact(tmp_path)
    rec = L.install(block, cell_src=cell, cell_config=cc, ck_dir=tmp_path / "ck", work=tmp_path / "w", url=tar.as_uri())
    assert rec["file"] == BEST and rec["role"] == "best" and rec["select"] == "best"
    assert rec["sha256"] == block["ckpt_sha256"] and Path(rec["path"]).name == BEST
    assert not (tmp_path / "ck" / LAST).exists() and not (tmp_path / "w/artifact.tar").exists()


def test_loader_refuses_a_best_pointer_with_another_sha(tmp_path):
    tar, cell, cc, block = _artifact(tmp_path, best_sha="0" * 64)
    with pytest.raises(L.CheckpointError, match="best does not name"):
        L.install(block, cell_src=cell, cell_config=cc, ck_dir=tmp_path / "ck", work=tmp_path / "w", url=tar.as_uri())

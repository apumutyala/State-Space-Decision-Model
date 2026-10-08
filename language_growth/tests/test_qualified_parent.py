"""Tiny metadata/tensor admission checks; no backbone construction or inference."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from decide.prompt import BODY_VERSION
from heads.layout import LAYOUT_VERSION
from language_growth.native_parent import NativeParentPin
from language_growth.qualified_parent import (COMPARATORS, FLOORS, SCHEMA, canonical_sha256,
                                              file_sha256, inspect_qualified_parent, load_qualified_parent,
                                              main, read_pinned_json)
from language_growth.rwkv7_m2_loader import _load_checkpoint, restore_trainables, verify_snapshot
from language_growth.tests.test_native_parent_m2_bundle import fixture
from train.losses import LossConfig
from train.trainer import TrainConfig


def admitted(tmp_path):
    checkpoint, loaded, report = fixture(tmp_path)
    revision = report["identities"]["serving_bundle"]["revision"]
    revision["model_id"] = "rwkv7-13.3b"
    revision["backbone"]["spec"] = "rwkv7-13.3b"
    revision["backbone"]["weights_digest"] = "b" * 64
    revision["backbone"]["weights_digest_mode"] = "sampled"
    loaded.backend.model.backbone.spec.name = "rwkv7-13.3b"
    loaded.backend._revision["model_id"] = "rwkv7-13.3b"
    loaded.backend._revision["backbone"] = deepcopy(revision["backbone"])
    report["identities"]["child"]["served"] = deepcopy(revision)
    report.update(schema="ssdm.m2.gates.v1", exit_code=0,
                  promotion={"promotable": True, "verdict": "accept", "stages": ["g12", "g13"],
                             "margins_pending_user_approval": False, "underpowered": {}, "failed": {}},
                  verdicts={s: {"accept": True} for s in ("g12", "g13")})
    cfg = TrainConfig(lora_rank=0, loss=LossConfig(arm="ce"))
    q = {"schema": SCHEMA, "verdict": "qualified", "model_line": "rwkv7-13.3b",
         "artifact_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         "report_sha256": "pending", "serving_bundle_sha256": canonical_sha256(report["identities"]["serving_bundle"]),
         "checkpoint_config": cfg.to_json(), "checkpoint_config_hash": cfg.hash(), "lora_modules": [],
         "checkpoint_lineage": {"backbone": revision["backbone"],
                                "tokenizer": [json.dumps(revision["encoder"], sort_keys=True)],
                                "layout": {"layouts": [[cfg.layout, revision["prompt_template"]["id"]]],
                                           "body_version": BODY_VERSION, "layout_version": LAYOUT_VERSION},
                                "data": {"n": 1, "sha256": "d" * 64, "order_fn": "synthetic"}},
         "rwkv7_wkv": None,
         "snapshot_files_sha256": {n: "e" * 64 for n in ("config.json", "tokenizer.json", "model.safetensors",
                                                            "modeling_rwkv7.py", "configuration_rwkv7.py", "tokenization_rwkv7.py")},
         "stage1": {"evaluation_scope": "full_dev", "primary_metric": "decision_index",
                    "preregistration_sha256": "1" * 64, "matched_evaluation_sha256": "2" * 64,
                    "independent_rescore_sha256": "3" * 64, "merged_commit": "4" * 40,
                    "paired_ci95": {name: {"lower": .1, "upper": 1, "confidence": .95,
                                            "paired": True, "matched": True} for name in COMPARATORS},
                    "floors": {name: {"passed": True, "evidence_sha256": "5" * 64} for name in FLOORS}}}
    return checkpoint, loaded, report, q, cfg


def publish(tmp_path, report, q):
    rp, qp = tmp_path / "report.json", tmp_path / "qualification.json"
    rp.write_text(json.dumps(report), encoding="utf-8")
    q["report_sha256"] = file_sha256(rp)
    qp.write_text(json.dumps(q), encoding="utf-8")
    return dict(report=rp, report_sha256=file_sha256(rp), qualification=qp, qualification_sha256=file_sha256(qp))


def test_admitted_metadata_loader_binds_native_planner_and_rechecks_live_drift(tmp_path):
    ckpt, loaded, report, q, _cfg = admitted(tmp_path)
    kwargs = publish(tmp_path, report, q)
    calls = []
    def strict_stub(path, *, artifact_sha256):
        calls.append((path, artifact_sha256))
        return loaded
    parent = load_qualified_parent(ckpt, **kwargs, loader=strict_stub)
    assert calls == [(ckpt.resolve(), q["artifact_sha256"])]
    assert loaded.backend.model.training is False and loaded.backend.model.grad_enabled is False
    assert parent.planner_kwargs()["model"] == "rwkv7-13.3b"
    assert parent.identity()["language_qualified"] is False
    identity = parent.identity()
    identity["stage1"]["floors"]["control"]["passed"] = False
    assert parent.identity()["stage1"]["floors"]["control"]["passed"] is True
    loaded.backend.model.backbone.model.config.implementation = "drift"
    with pytest.raises(ValueError, match="configuration changed"):
        parent.decide({})


def test_admission_metadata_and_cli_never_open_checkpoint_or_load_model(tmp_path, capsys):
    _ckpt, _loaded, report, q, _cfg = admitted(tmp_path)
    kwargs = publish(tmp_path, report, q)
    spec = inspect_qualified_parent(**kwargs)
    assert spec.identity()["language_qualified"] is False
    assert spec.identity()["parent_revision"] == spec.pin.revision["revision_id"]
    args = ["--parent", str(tmp_path / "absent-weights.pt")]
    for key, value in kwargs.items():
        args += ["--parent-" + key.replace("_", "-"), str(value)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


@pytest.mark.parametrize("change", ["keep", "promotion", "partial", "floor", "missing_floor", "clef", "zero_ci",
                                    "negative_ci", "confidence", "unpaired", "subset", "missing_rescore",
                                    "bundle", "config", "lineage", "tokenizer", "layout", "missing_inventory", "escape"])
def test_unqualified_or_mismatched_parent_fails_before_checkpoint_or_loader(tmp_path, change):
    _ckpt, _loaded, report, q, _cfg = admitted(tmp_path)
    if change == "keep": q["verdict"] = "keep"
    elif change == "promotion": report["promotion"]["promotable"] = False
    elif change == "partial": report["verdicts"].pop("g13")
    elif change == "floor": q["stage1"]["floors"]["control"]["passed"] = False
    elif change == "missing_floor": q["stage1"]["floors"].pop("abstention")
    elif change == "clef": q["stage1"]["paired_ci95"].pop("clef-27b")
    elif change == "zero_ci": q["stage1"]["paired_ci95"]["clef-flash"]["lower"] = 0
    elif change == "negative_ci": q["stage1"]["paired_ci95"]["clef-flash"]["lower"] = -.1
    elif change == "confidence": q["stage1"]["paired_ci95"]["clef-flash"]["confidence"] = .90
    elif change == "unpaired": q["stage1"]["paired_ci95"]["clef-flash"]["paired"] = False
    elif change == "subset": q["stage1"]["evaluation_scope"] = "dev_subset"
    elif change == "missing_rescore": q["stage1"].pop("independent_rescore_sha256")
    elif change == "bundle": q["serving_bundle_sha256"] = "0" * 64
    elif change == "config": q["checkpoint_config"]["head"] = "snapshot"
    elif change == "lineage": q["checkpoint_lineage"]["backbone"] = {"spec": "other"}
    elif change == "tokenizer": q["checkpoint_lineage"]["tokenizer"] = []
    elif change == "layout": q["checkpoint_lineage"]["layout"]["body_version"] = "old"
    elif change == "missing_inventory": q["snapshot_files_sha256"].pop("tokenizer.json")
    elif change == "escape": q["snapshot_files_sha256"]["../other.bin"] = "0" * 64
    kwargs = publish(tmp_path, report, q)
    calls = []
    with pytest.raises((ValueError, KeyError)):
        load_qualified_parent(tmp_path / "absent.pt", **kwargs, loader=lambda *_a, **_k: calls.append(1))
    assert calls == []


def test_report_hash_and_duplicate_nonfinite_members_rejected(tmp_path):
    p = tmp_path / "payload.json"
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'):
        p.write_bytes(raw)
        with pytest.raises(ValueError):
            read_pinned_json(p, file_sha256(p))
    p.write_bytes(b'{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        read_pinned_json(p, "0" * 64)


def test_snapshot_inventory_exact_content_and_extra_file_rejection(tmp_path):
    p = tmp_path / "file.bin"
    p.write_bytes(b"tiny synthetic bytes")
    inventory = {p.name: file_sha256(p)}
    verify_snapshot(tmp_path, inventory)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        verify_snapshot(tmp_path, inventory)
    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_snapshot(tmp_path, inventory)


def tiny_state(tmp_path):
    _ckpt, _loaded, _report, q, cfg = admitted(tmp_path)
    model = torch.nn.Linear(2, 1)
    state = {"step": 7, "config": q["checkpoint_config"], "config_hash": cfg.hash(),
             "lineage": q["checkpoint_lineage"], "lora_modules": [], "init_seed": cfg.seed,
             "backbone": "rwkv7-13.3b", "trainable": {n: torch.ones_like(p) for n, p in model.named_parameters()}}
    return model, state, q, cfg


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nonfinite", "config", "lineage", "lora", "step"])
def test_complete_trainable_preflight_leaves_all_parameters_untouched_on_failure(tmp_path, change):
    model, state, q, cfg = tiny_state(tmp_path)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    if change == "missing": state["trainable"].pop("bias")
    elif change == "extra": state["trainable"]["extra"] = torch.zeros(1)
    elif change == "shape": state["trainable"]["bias"] = torch.ones(2)
    elif change == "dtype": state["trainable"]["bias"] = torch.ones(1, dtype=torch.float64)
    elif change == "nonfinite": state["trainable"]["bias"] = torch.tensor([float("nan")])
    elif change == "config": state["config_hash"] = "wrong"
    elif change == "lineage": state["lineage"] = {}
    elif change == "lora": state["lora_modules"] = ["unexpected"]
    elif change == "step": state["step"] = 8
    with pytest.raises(ValueError):
        restore_trainables(model, state, q, cfg, [], step=7)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_safe_existing_trainer_numpy_rng_payload_and_complete_restore(tmp_path):
    model, state, q, cfg = tiny_state(tmp_path)
    state["rng"] = {"numpy": np.random.RandomState(0).get_state()}
    path = tmp_path / "tiny.pt"
    torch.save(state, path)
    restored = _load_checkpoint(path)
    restore_trainables(model, restored, q, cfg, [])
    assert all(torch.equal(p, torch.ones_like(p)) for p in model.parameters())

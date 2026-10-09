"""Restricted checkpoint-header handoff; only tiny synthetic tensors."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest
import torch

from language_growth.checkpoint_metadata import checkpoint_metadata, main
from language_growth.qualified_parent import file_sha256


def save(tmp_path, **changes):
    state = {"step": 7, "init_seed": 0, "config": {"head": "fork"}, "config_hash": "fixed",
             "lineage": {"backbone": {"spec": "tiny"}}, "backbone": "tiny",
             "lora_modules": [], "trainable": {"head.weight": torch.ones(2, 3)},
             "optimizer": {"never_export": torch.ones(1)}, "history": [{"private_label": "OMIT"}]}
    state.update(changes)
    path = tmp_path / "tiny.pt"
    torch.save(state, path)
    return path


def test_header_has_exact_saved_pins_and_never_exports_optimizer_history_or_weights(tmp_path):
    path = save(tmp_path)
    # Header access must not copy/convert tensor payloads or inspect finiteness.
    with patch.object(torch.Tensor, "clone", side_effect=AssertionError("tensor copy")), \
            patch.object(torch, "isfinite", side_effect=AssertionError("tensor read")):
        result = checkpoint_metadata(path, sha256=file_sha256(path))
    assert result["checkpoint_metadata"]["step"] == 7
    assert result["trainable_inventory"] == {"head.weight": {"shape": [2, 3], "dtype": "torch.float32"}}
    serialized = json.dumps(result)
    assert "optimizer" not in serialized and "OMIT" not in serialized and "never_export" not in serialized
    assert result["model_reconstructed"] is False and result["tensor_values_validated"] is False
    assert result["stage1_qualified"] is False


def test_hash_mismatch_refuses_before_deserialization(tmp_path):
    path = save(tmp_path)
    with patch("language_growth.checkpoint_metadata._load_checkpoint", side_effect=AssertionError("opened")):
        with pytest.raises(ValueError, match="hash differs"):
            checkpoint_metadata(path, sha256="0" * 64)


@pytest.mark.parametrize("change", [{"trainable": {}}, {"step": True}, {"lora_modules": ["z", "a"]}])
def test_bad_header_is_not_exportable(tmp_path, change):
    path = save(tmp_path, **deepcopy(change))
    with pytest.raises(ValueError):
        checkpoint_metadata(path, sha256=file_sha256(path))


def test_cli_preserves_existing_handoff(tmp_path):
    path = save(tmp_path)
    out = tmp_path / "header.json"
    args = ["--checkpoint", str(path), "--sha256", file_sha256(path), "--out", str(out)]
    assert main(args) == 0
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert out.read_bytes() == before

"""Encoder-only contract tests: no model construction/load/forward or network."""
from copy import deepcopy
from dataclasses import replace

import pytest

from decide.backends.hf_readout import ModelPlugin
from language_growth.rwkv_decision_head import rwkv_cpu_parent_encoder


def plugin():
    calls = []

    def encode(text, trusted=True):
        calls.append((text, trusted))
        return list(text.encode("utf-8"))

    def no_forward(*args, **kwargs):
        raise AssertionError("encoder binding cannot execute a model")

    value = ModelPlugin(family="rwkv7", model_id="encoder-contract-fixture", snapshot="fixture-revision",
        encode=encode, forward=no_forward, device="cpu", dtype="float32",
        files={"weights": "a" * 64, "config": "b" * 64, "tokenizer": "c" * 64, "code": "d" * 64},
        untrusted_encoder_id="rwkv-world-greedy (no control tokens)")
    return value, calls


def test_exact_encoding_and_full_loader_identity_without_model_execution():
    source, calls = plugin()
    encode = rwkv_cpu_parent_encoder(source)
    assert encode("public", True) == list(b"public")
    assert encode("<|endoftext|>\nprivate", False) == list(b"<|endoftext|>\nprivate")
    assert calls == [("public", True), ("<|endoftext|>\nprivate", False)]
    identity = encode.identity()
    assert identity == {"kind": "rwkv-world-greedy", "binding": "track-l-rwkv-cpu-plugin-encoder-v1",
        "model_id": source.model_id, "snapshot": source.snapshot, "files_sha256": source.files,
        "untrusted_encoding": source.untrusted_encoder_id, "special_ids": []}


def test_identity_and_callable_are_bound_without_mutating_source():
    source, calls = plugin()
    before = deepcopy(source.files)
    original = source.encode
    encode = rwkv_cpu_parent_encoder(source)
    expected = encode.identity()
    assert source.encode is original and source.files == before
    returned = encode.identity()
    returned["files_sha256"]["tokenizer"] = "modified"
    assert encode.identity() == expected
    source.files["tokenizer"] = "e" * 64
    source.snapshot = "other"
    source.untrusted_encoder_id = "other"
    source.encode = lambda text, trusted=True: [999]
    assert encode.identity() == expected
    assert encode("same", False) == list(b"same")
    assert calls == [("same", False)]


@pytest.mark.parametrize("role", ["weights", "config", "tokenizer", "code"])
@pytest.mark.parametrize("bad", [None, "unknown", "A" * 64, "0" * 63])
def test_incomplete_or_malformed_loader_pins_fail_closed(role, bad):
    source, _ = plugin()
    source.files[role] = bad
    with pytest.raises(ValueError, match="SHA256"):
        rwkv_cpu_parent_encoder(source)


@pytest.mark.parametrize("changes,match", [
    ({"family": "mamba"}, "CPU loader"),
    ({"device": "cuda"}, "CPU loader"),
    ({"dtype": "bfloat16"}, "CPU loader"),
    ({"model_id": ""}, "model and snapshot"),
    ({"snapshot": None}, "model and snapshot"),
    ({"untrusted_encoder_id": "unknown"}, "World encoder"),
    ({"special_ids": frozenset({0})}, "World encoder"),
    ({"encode": None}, "World encoder"),
])
def test_other_encoder_contracts_are_refused(changes, match):
    source, _ = plugin()
    with pytest.raises(ValueError, match=match):
        rwkv_cpu_parent_encoder(replace(source, **changes))


def test_nonplugin_is_refused():
    with pytest.raises(ValueError, match="CPU loader"):
        rwkv_cpu_parent_encoder(object())

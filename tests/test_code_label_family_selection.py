"""Construction-only CPU engine/bridge tests; fake loaders/tokenizer, no model calls."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("torch")

from decide.backends import hf_readout as H
from decide.backends.rwkv_g1_code_label import G1CodeLabelBackend, G1_REVISION, TEMPLATE_ID
from decide.backends.rwkv_g1_filtered_codes import (PROOF_CONTEXT, TABLE_SHA256, TOKENIZER_SHA256,
    TOKENIZER_SOURCE_SHA256, load_frozen_table)
from safety_eval import decide_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def engine_module(monkeypatch):
    """Isolate the kit API so engine construction does not depend on kit installation."""
    class FakeEngine:
        def __init__(self, **options):
            self.options = options
    class FakeUnsupported(ValueError):
        pass
    for name in ("decision_index", "decision_index.engines"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    kit = types.ModuleType("decision_index.engines.base")
    kit.Engine, kit.Unsupported = FakeEngine, FakeUnsupported
    monkeypatch.setitem(sys.modules, "decision_index.engines.base", kit)
    spec = importlib.util.spec_from_file_location("_code_label_selection_engine", ROOT / "jobs/common/readout_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "jobs.common.readout_engine", module)
    import jobs.common
    monkeypatch.setattr(jobs.common, "readout_engine", module, raising=False)
    return module


@pytest.fixture
def fake_loader(monkeypatch, engine_module):
    codes = {" " + c.code: c.token_id for c in load_frozen_table()}
    inverse = {token: text for text, token in codes.items()}
    events = []
    def encode(text, trusted=True):
        if text == PROOF_CONTEXT:
            return [55000]
        if text.startswith(PROOF_CONTEXT) and text[len(PROOF_CONTEXT):] in codes:
            return [55000, codes[text[len(PROOF_CONTEXT):]]]
        if text in codes:
            return [codes[text]]
        return [10000 + b for b in text.encode()]
    def no_forward(*args, **kwargs):
        pytest.fail("construction must not call a model")
    plugin = H.ModelPlugin("rwkv7", "fake-no-model", None, encode, no_forward,
        files={"tokenizer": TOKENIZER_SHA256, "weights": "not-loaded"},
        notes={"vendored_code": G1_REVISION}, special_ids=frozenset({0}),
        device="cpu", dtype="float32", max_context=25600)
    def loader(model_dir, **kwargs):
        events.append(("plugin", str(model_dir), kwargs))
        return plugin
    monkeypatch.setattr(H, "load_rwkv7_hf", loader)
    class FakeTokenizer:
        def __init__(self, *, tokenizer_file):
            events.append(("decoder", tokenizer_file))
        def decode(self, ids):
            return "".join(inverse[i] for i in ids)
        # Deliberately no encode/from_pretrained: only decode is reconstructed.
    monkeypatch.setitem(sys.modules, "third_party.rwkv7_hf.tokenization_rwkv7",
                        types.SimpleNamespace(Rwkv7Tokenizer=FakeTokenizer))
    monkeypatch.setattr(engine_module, "_sha256_file",
                        lambda p: TOKENIZER_SHA256 if Path(p).name == "tokenizer.json" else TOKENIZER_SOURCE_SHA256)
    return plugin, events


def test_index_selects_code_label_family_with_pinned_revision_and_decode_first(engine_module, fake_loader):
    plugin, events = fake_loader
    engine = engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot",
                                              device="cpu", dtype="float32")
    assert isinstance(engine.backend, G1CodeLabelBackend)
    revision = engine.backend.revision()
    assert revision["prompt_template"]["id"] == TEMPLATE_ID
    assert revision["inference_config"]["codebook_table_sha256"] == TABLE_SHA256
    assert revision["files_sha256"]["tokenizer"] == TOKENIZER_SHA256
    assert engine.provenance["family"] == "rwkv7-hf-code-label-v1"
    assert engine.provenance["backend_revision"] == revision["revision_id"]
    assert [event[0] for event in events] == ["decoder", "plugin"]
    assert events[1][2] == {"device": "cpu", "dtype": "float32"}
    assert engine.decider.backend_max_input_tokens == plugin.max_context


def test_existing_rwkv7_hf_selection_and_batch_option_are_unchanged(engine_module, fake_loader):
    _, events = fake_loader
    engine = engine_module.ReadoutIndexEngine(family="rwkv7-hf", model_dir="snapshot",
                                              device="cpu", dtype="float32", max_batch_rows=7)
    assert type(engine.backend) is H.HFReadoutBackend
    assert engine.backend.revision()["prompt_template"]["id"] == "jrt-rwkv-g1-v1"
    assert engine.backend.max_batch_rows == 7
    assert [event[0] for event in events] == ["plugin"]


def test_build_plugin_keeps_existing_device_and_dtype_defaults(engine_module, fake_loader):
    _, events = fake_loader
    for family in ("rwkv7-hf", engine_module.CODE_LABEL_FAMILY):
        engine_module.build_plugin(family, {"model_dir": "snapshot"})
    assert all(event[2] == {"device": "cuda", "dtype": "bfloat16"} for event in events)


def test_default_family_and_template_mapping_are_unchanged(engine_module):
    assert engine_module.TEMPLATES["q9af-probe"] == "jrt-chatml-v1"
    assert engine_module.TEMPLATES["qwen3.5"] == "jrt-chatml-v1"
    assert engine_module.TEMPLATES["rwkv7-hf"] == "jrt-rwkv-g1-v1"
    observed = []
    def stop_before_load(family, options):
        observed.append(family)
        raise RuntimeError("intercept before model")
    engine_module.build_plugin = stop_before_load
    with pytest.raises(RuntimeError, match="intercept"):
        engine_module.ReadoutIndexEngine()
    assert observed == ["q9af-probe"]


@pytest.mark.parametrize("entrypoint", ["index", "safety"])
@pytest.mark.parametrize("bad_pin", ["vocab", "source"])
def test_bad_tokenizer_pin_stops_before_loader(engine_module, fake_loader, entrypoint, bad_pin):
    _, events = fake_loader
    def bad_hash(path):
        is_vocab = Path(path).name == "tokenizer.json"
        return "wrong" if is_vocab == (bad_pin == "vocab") else (TOKENIZER_SHA256 if is_vocab else TOKENIZER_SOURCE_SHA256)
    engine_module._sha256_file = bad_hash
    with pytest.raises(ValueError, match="pinned G1k"):
        if entrypoint == "index":
            engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot")
        else:
            bridge.build_hf_bridge("snapshot", family=engine_module.CODE_LABEL_FAMILY)
    assert not events


@pytest.mark.parametrize("entrypoint", ["index", "safety"])
def test_missing_tokenizer_file_stops_before_loader(engine_module, fake_loader, tmp_path, entrypoint):
    _, events = fake_loader
    def missing(path):
        return Path(path).read_bytes()  # Deliberate missing file, never a real checkpoint.
    engine_module._sha256_file = missing
    with pytest.raises(FileNotFoundError):
        if entrypoint == "index":
            engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir=str(tmp_path))
        else:
            bridge.build_hf_bridge(str(tmp_path), family=engine_module.CODE_LABEL_FAMILY)
    assert not events


def test_unknown_family_does_not_construct_any_plugin(engine_module, fake_loader):
    _, events = fake_loader
    with pytest.raises(ValueError, match="unknown family"):
        engine_module.ReadoutIndexEngine(family="unknown", model_dir="snapshot")
    assert not events


@pytest.mark.parametrize("entrypoint", ["index", "safety"])
def test_loader_failure_propagates_without_stock_or_other_family_fallback(engine_module, fake_loader, monkeypatch, entrypoint):
    _, events = fake_loader
    def failed_loader(*args, **kwargs):
        events.append(("loader-failed",))
        raise RuntimeError("loader failure")
    monkeypatch.setattr(H, "load_rwkv7_hf", failed_loader)
    monkeypatch.setattr(H, "load_qwen35", lambda *a, **k: pytest.fail("must not fall back to another family"))
    with pytest.raises(RuntimeError, match="loader failure"):
        if entrypoint == "index":
            engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot")
        else:
            bridge.build_hf_bridge("snapshot", family=engine_module.CODE_LABEL_FAMILY)
    assert [event[0] for event in events] == ["decoder", "loader-failed"]


def test_safety_bridge_selects_same_code_label_backend_and_revision(engine_module, fake_loader):
    _, events = fake_loader
    smoke = bridge.build_hf_bridge("snapshot", family=engine_module.CODE_LABEL_FAMILY, device="cpu", dtype="float32")
    assert isinstance(smoke.decider.backend, G1CodeLabelBackend)
    assert smoke.decider.backend.revision()["prompt_template"]["id"] == TEMPLATE_ID
    assert smoke.decider.backend.revision()["inference_config"]["codebook_table_sha256"] == TABLE_SHA256
    assert [event[0] for event in events] == ["decoder", "plugin"]
    index = engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot", device="cpu", dtype="float32")
    assert smoke.decider.backend.revision() == index.backend.revision()


def test_existing_safety_bridge_rwkv7_hf_still_uses_stock_backend(engine_module, fake_loader):
    _, events = fake_loader
    smoke = bridge.build_hf_bridge("snapshot", family="rwkv7-hf")
    assert type(smoke.decider.backend) is H.HFReadoutBackend
    assert smoke.decider.backend.revision()["prompt_template"]["id"] == "jrt-rwkv-g1-v1"
    assert [event[0] for event in events] == ["plugin"]


def test_v2_index_and_safety_selection_preserve_v1_reference(engine_module, fake_loader):
    from decide.backends.rwkv_g1_code_label_v2 import G1CodeLabelV2Backend, TEMPLATE_ID as V2_TEMPLATE

    plugin, _ = fake_loader
    def no_state_call(*args, **kwargs):
        pytest.fail("construction must not call state/model operations")
    plugin.state_ops = types.SimpleNamespace(prefill=no_state_call, extend=no_state_call)
    index = engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_V2_FAMILY, model_dir="snapshot", device="cpu", dtype="float32")
    smoke = bridge.build_hf_bridge("snapshot", family=engine_module.CODE_LABEL_V2_FAMILY, device="cpu", dtype="float32")
    reference = engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot", device="cpu", dtype="float32")
    assert isinstance(index.backend, G1CodeLabelV2Backend)
    assert index.backend.revision() == smoke.decider.backend.revision()
    assert index.backend.revision()["prompt_template"]["id"] == V2_TEMPLATE
    assert index.backend.fork and index.backend.fork_max_rows == 4
    assert type(reference.backend) is G1CodeLabelBackend and not reference.backend.fork
    assert reference.backend.revision()["prompt_template"]["id"] == TEMPLATE_ID


@pytest.mark.parametrize("status", ["unsupported", "exhausted", "error"])
def test_engine_response_error_handling_is_unchanged(engine_module, fake_loader, status):
    engine = engine_module.ReadoutIndexEngine(family=engine_module.CODE_LABEL_FAMILY, model_dir="snapshot")
    engine.decider = types.SimpleNamespace(decide=lambda request: {
        "ssdm": {"status": status, "error": {"code": "test_code", "message": "test message"}}})
    expected = RuntimeError if status == "error" else engine_module.Unsupported
    with pytest.raises(expected, match="test_code"):
        engine("synthetic", {"q": {"type": "choice", "criteria": {"a": None}}})

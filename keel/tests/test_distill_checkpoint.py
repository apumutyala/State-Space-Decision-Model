import hashlib
import json

import pytest

from keel.distill.checkpoint import PROCESSOR_FILES, copy_processor_assets, save_processor_assets


def processor_snapshot(path):
    path.mkdir()
    (path / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "Qwen2VLImageProcessorFast"}))
    (path / "video_preprocessor_config.json").write_text(json.dumps({"video_processor_type": "Qwen3VLVideoProcessor"}))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return path


def test_saved_checkpoint_contains_all_processor_assets_without_changing_weights(tmp_path):
    source = processor_snapshot(tmp_path / "base")
    output = tmp_path / "merged"
    output.mkdir()
    untouched = {"model.safetensors": b"weight-file sentinel", "config.json": b"model config",
                 "tokenizer.json": b"saved tokenizer"}
    for name, content in untouched.items():
        (output / name).write_bytes(content)
    hashes = copy_processor_assets(source, output)
    assert set(hashes) == set(PROCESSOR_FILES)
    for name in PROCESSOR_FILES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
        assert hashes[name] == hashlib.sha256((source / name).read_bytes()).hexdigest()
    for name, content in untouched.items():
        assert (output / name).read_bytes() == content


@pytest.mark.parametrize("missing", PROCESSOR_FILES)
def test_missing_pinned_processor_asset_fails_before_partial_export(tmp_path, missing):
    source = processor_snapshot(tmp_path / "base")
    (source / missing).unlink()
    output = tmp_path / "merged"
    with pytest.raises(FileNotFoundError, match="pinned processor asset missing"):
        copy_processor_assets(source, output)
    assert not output.exists()


def test_processor_export_downloads_exact_pinned_revision(tmp_path, monkeypatch):
    import huggingface_hub
    from keel.runtime.modal_app import MODELS

    source = processor_snapshot(tmp_path / "base")
    calls = []

    def download(model, *, revision, allow_patterns):
        calls.append((model, revision, allow_patterns))
        return str(source)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    report = save_processor_assets(tmp_path / "merged")
    model, revision = MODELS["qwen3.5-4b"]
    assert calls == [(model, revision, list(PROCESSOR_FILES))]
    assert report["model"] == model and report["revision"] == revision
    assert set(report["files_sha256"]) == set(PROCESSOR_FILES)
    assert all((tmp_path / "merged" / name).is_file() for name in PROCESSOR_FILES)

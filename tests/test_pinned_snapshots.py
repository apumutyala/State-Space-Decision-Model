import json, os
import pytest
from convert.model import resolve_model_dir

PINS = {"0p8b": "2fc06364715b967f1860aea9cf38778875588b17", "9b": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}


def _snap(root, rev):
    d = root / rev
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_multiple_snapshots_refused(tmp_path):
    _snap(tmp_path, "aaaa")
    _snap(tmp_path, "zzzz")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model_dir(str(tmp_path))


def test_exact_snapshot_loaded_even_when_cache_has_other_revisions(tmp_path):
    pinned = _snap(tmp_path, PINS["9b"])
    _snap(tmp_path, "ffffffffffffffffffffffffffffffffffffffff")  # sorts last; must not be chosen
    assert resolve_model_dir(str(pinned)) == str(pinned)


def test_qual_config_pins_exact_revisions():
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "qual_h100.json"), encoding="utf-8"))
    for key, rev in PINS.items():
        md = cfg["models"][key]["model_dir"]
        assert md and md.rstrip("/").endswith("/snapshots/" + rev), (key, md)

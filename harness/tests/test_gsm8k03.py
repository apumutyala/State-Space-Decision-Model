"""0.3 GSM8K rows: the DEV/TEST addenda are a pure ID mapping of the committed 0.2.1 manifests, the rebuilt file is
the board's byte-for-byte, and a 0.3 DEV slice with the twins scores identically through kit and decomposition.
No test here reads TEST content: the TEST checks use the committed manifests (IDs and hashes) only."""

import os

import pytest

from harness import config, gsm8k03, predictors, score, slicing
from harness.tests.conftest import needs_suite

needs_gsm = pytest.mark.skipif(not config.GSM8K_03_ROWS.exists(), reason="rebuilt 0.3 GSM8K rows not built locally")


@pytest.mark.parametrize("name", ["dev", "test"])
def test_addendum_is_the_id_twin_of_the_021_manifest(name):
    old = slicing.read_manifest(config.MANIFEST_DIR / f"{name}-manifest.jsonl.gz")
    new = slicing.read_manifest(gsm8k03.manifest_path(name))
    olds = {m["run_id"]: m for m in old if m["catalog_id"] == 30}
    assert len(new) == len(olds) == 262
    for m in new:
        assert m["run_id"].startswith(config.GSM8K_03_RUN_ID_PREFIX) and m["catalog_id"] == 30
        o = olds[m["run_id"][len(config.GSM8K_03_RUN_ID_PREFIX):]]
        assert (m["group_id"], m["track"]) == (o["group_id"], o["track"])
    assert sorted(gsm8k03.twin_ids(old)) == [m["run_id"] for m in new]


def test_addenda_are_disjoint_and_recorded():
    import json

    rep = json.loads((config.MANIFEST_DIR / "slices-0.3-gsm8k.json").read_text())
    assert rep["disjoint_rows"] and rep["source_overlap_dev_test"] == {}
    dev = {m["run_id"] for m in slicing.read_manifest(gsm8k03.manifest_path("dev"))}
    test = {m["run_id"] for m in slicing.read_manifest(gsm8k03.manifest_path("test"))}
    assert not dev & test


@needs_gsm
def test_rebuilt_file_is_the_pinned_one():
    rows = gsm8k03.load_rows()
    assert len(rows) == 2638 and all(r["_evaluation"]["run_id"].startswith("v0.3-gsm8k:") for r in rows)


def test_test_slice_still_sealed(monkeypatch):
    from harness.cli import load_slice

    monkeypatch.delenv("SSDM_UNSEAL_TEST", raising=False)
    with pytest.raises(SystemExit):
        load_slice("test", "0.3")


@needs_suite
@needs_gsm
def test_03_dev_slice_with_twins_scores_like_the_kit():
    from harness.cli import load_slice

    rows = load_slice("dev", "0.3")
    cats = [r["_evaluation"]["catalog_id"] for r in rows]
    assert cats.count(30) == 262 and 48 not in cats
    assert all(r["_evaluation"]["run_id"].startswith("v0.3-gsm8k:") for r in rows if r["_evaluation"]["catalog_id"] == 30)
    res = predictors.predict(rows, "noisy", seed=21, acc=0.6, status_drop=0.03)
    off = score.official(rows, res, edition="0.3")
    dec = score.point(score.decompose(rows, res, edition="0.3"), renormalize=False)
    assert 30 in dec["per_benchmark"] and off["index_entry"] is not None  # complete 0.3 panel on DEV
    for n, v in dec["per_benchmark"].items():
        assert v["skill"] == pytest.approx(off["values"][n]["skill"], abs=1e-4), n
    assert dec["index"] == pytest.approx(off["scores"]["balanced_skill"], abs=1e-3)
    assert len(load_slice("dev", "0.3", gsm8k03=False)) == len(rows) - 262


@needs_suite
@needs_gsm
def test_m1_subset_maps_onto_03():
    from jobs.common.dev_subset import _rows, load_subset

    sub = load_subset()
    r03 = _rows(sub, "0.3")
    n_gsm_old = sum(i.startswith("30:") for i in sub["run_ids"])
    assert sum(r["_evaluation"]["catalog_id"] == 30 for r in r03) == n_gsm_old == 44
    assert len(_rows(sub, "0.3", gsm8k03=False)) == len(r03) - 44

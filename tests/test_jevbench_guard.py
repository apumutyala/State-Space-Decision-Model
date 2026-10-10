"""JevBench public items are EVAL-ONLY: the doc-overlap gate and the builder fingerprint refuse training rows that
carry their text (train.jevbench_guard; a separate pinned index, so the Index eval index and every doc_clean pin
stay byte-identical)."""

import copy
import json
import tempfile
from pathlib import Path

import pytest

from jevkit import items as IT
from train import doc_overlap_guard as DG
from train import jevbench_guard as JG

PUBLIC = IT.default_data_dir()
HAVE_PUBLIC = (PUBLIC / "hard.jsonl").exists()


@pytest.fixture
def synthetic_jevbench():
    """A JevBench pair built from synthetic items (no public data needed), installed for one test."""
    from data.dedup import Fingerprint

    items = [
        {"state": "Policy: refunds need a receipt and a purchase within thirty days of the request being filed by the "
                  "customer at the original store location, and gift cards are never refundable under any condition.",
         "questions": {"q": {"type": "noul", "instructions": "Is the refund permitted under the stated policy?",
                             "criteria": {"true": "Every required condition holds.",
                                          "false": "A condition is missing or a prohibition applies."}}}},
        {"state": "Ticket: payouts failing for three days",
         "questions": {"q": {"type": "choice", "instructions": "Which team should handle this ticket?",
                             "criteria": {"billing": "Payments, invoicing and refunds for merchants",
                                          "tech": "Bugs, outages and integrations"}}}},
    ]
    rows = [("JEVBENCH", f"jevbench:s{i}", "jevbench_syn", {"id": f"s{i}", **r}) for i, r in enumerate(items)]
    idx = DG.build_index(rows, {"entries": []}, n_parts=1)
    idx.meta["index_sha256"] = "synthetic-jevbench"
    fp = Fingerprint.build(({**r[3], "_evaluation": {"catalog_id": JG.CATALOG_ID}} for r in rows), log=lambda *a: None)
    JG.set_for_tests((idx, fp))
    try:
        yield items
    finally:
        JG.set_for_tests(None)


def _rec(i, state, questions=None, source="mmlu_aux"):
    return {"id": f"m2:t:{i}", "source": source, "state": state,
            "questions": questions or {"q": {"type": "choice", "instructions": "Pick the answer.",
                                             "criteria": {"a": "first option text", "b": "second option text"}}}}


CLEAN = [_rec(100 + i, f"An unrelated training passage number {i} about rivers, mountains and the weather in spring.")
         for i in range(20)]


def test_committed_pin_loads_and_matches_the_jevkit_source():
    JG.set_for_tests(None)
    idx, fp = JG.load()
    pin = JG.read_pin()
    assert pin["source"]["commit"] == IT.read_pin()["commit"] and idx.n_rows == 231 == pin["doc_index"]["n_eval_rows"]
    assert fp.arrays_sha256() == pin["fingerprint"]["arrays_sha256"] and fp.meta["rows"] == 231
    assert {r[0] for r in idx.meta["rows"]} == {JG.SPLIT}
    blob = (JG.INDEX_DIR / "meta.json").read_text(encoding="utf-8") + (JG.FP_PATH.with_suffix(".json")).read_text("utf-8")
    assert "receipt" not in blob.lower()  # hashes and ids only: no item text in the committed artifacts


@pytest.mark.skipif(not HAVE_PUBLIC, reason="pinned JevBench public items not downloaded (python -m jevkit fetch)")
def test_rebuild_from_the_pinned_items_reproduces_the_pin():
    idx, fp = JG.build()
    with tempfile.TemporaryDirectory() as td:
        meta = DG.save_index(idx, Path(td) / "i", {})
    pin = JG.read_pin()
    assert meta["index_sha256"] == pin["doc_index"]["index_sha256"]
    assert fp.arrays_sha256() == pin["fingerprint"]["arrays_sha256"]


@pytest.mark.skipif(not HAVE_PUBLIC, reason="pinned JevBench public items not downloaded (python -m jevkit fetch)")
def test_training_rows_copying_public_items_fail_the_gate():
    """Long (hard tier), medium (original) and short (easy) public items, verbatim and laundered, all fail."""
    JG.set_for_tests(None)
    items = IT.load_public()
    long_it = next(i for i in items if i.tier == "hard" and isinstance(i.state, str) and len(i.state.split()) > 200)
    mid_it = next(i for i in items if i.tier == "standard")
    short_it = min((i for i in items if i.tier == "easy" and isinstance(i.state, str)), key=lambda i: len(i.state))
    copies = [
        _rec(1, long_it.state, {"q": long_it.question}),
        _rec(2, mid_it.state, {"q": mid_it.question}),
        _rec(3, short_it.state, {"q": short_it.question}),
        _rec(4, long_it.state.upper().replace(".", " ;")),  # laundered: case + punctuation, other question
        _rec(5, {"context": "Background note.", "doc": long_it.state}),  # nested in a structured state
    ]
    rep = DG.scan(CLEAN + copies)
    jb = rep["jevbench"]
    assert jb["enabled"] and set(jb["fail_ids"]) == {c["id"] for c in copies}, jb["fail_ids"]
    assert rep["n_fail"] == 5 and set(rep["fail_ids"]) == {c["id"] for c in copies}
    assert rep["per_reason"]["JEVBENCH"] == 5 and DG.check_pinned(rep, None)[0] is False
    assert "m2:t:3" in {f["id"] for f in jb["fails"]}  # the short item (no 13-gram reach in its state) fails too
    # the builder fingerprint drops the same short copy as an item-tier hit
    _, fp = JG.load()
    hits = JG.fingerprint_item_hits(fp, short_it.state, {"q": short_it.question})
    assert hits["jevbench_exact"] and all(k.startswith("jevbench_") for k in hits)
    assert not any(any(JG.fingerprint_item_hits(fp, r["state"], r["questions"]).values()) for r in CLEAN)


def test_copy_fails_and_clean_passes_on_a_synthetic_jevbench(synthetic_jevbench):
    items = synthetic_jevbench
    copies = [_rec(1, items[0]["state"], items[0]["questions"]), _rec(2, items[1]["state"], items[1]["questions"])]
    rep = DG.scan(CLEAN + copies)
    assert sorted(rep["jevbench"]["fail_ids"]) == ["m2:t:1", "m2:t:2"]
    assert rep["jevbench"]["n_fail_doc"] >= 1 and rep["jevbench"]["n_fail_exact"] == 2
    assert DG.record_fails(CLEAN + copies) == ["m2:t:1", "m2:t:2"]
    with pytest.raises(DG.DocOverlapError):  # the draw / cell paths refuse an unpinned fail (exit 34 in a job)
        ok, why = DG.check_pinned(rep, None)
        if not ok:
            raise DG.DocOverlapInTraining(why)


def test_jevbench_clean_scan_leaves_index_reports_and_pins_unchanged(synthetic_jevbench):
    with_jb = DG.scan(CLEAN)
    without = DG.scan(CLEAN, jevbench=False)
    assert with_jb["jevbench"]["n_fail"] == 0 and without["jevbench"] == {"enabled": False}
    for k in ("n_fail", "fail_ids_sha256", "per_source_fail", "per_reason", "index_sha256", "fail_ids"):
        assert with_jb[k] == without[k], k
    spec = DG.clean_spec(with_jb)
    assert spec["jevbench_index_sha256"] == "synthetic-jevbench" and spec["jevbench_n_excluded"] == 0
    legacy = {k: v for k, v in spec.items() if not k.startswith("jevbench_")}  # every doc_clean pin made before
    assert DG.check_pinned(with_jb, legacy) == (True, "no fail")


def test_clean_spec_pins_the_jevbench_identity(synthetic_jevbench):
    items = synthetic_jevbench
    rep = DG.scan(CLEAN + [_rec(1, items[0]["state"], items[0]["questions"])])
    spec = DG.clean_spec(rep)
    assert DG.check_pinned(rep, spec)[0] is True and spec["n_excluded"] == 1
    other = dict(spec, jevbench_index_sha256="0" * 64)
    ok, why = DG.check_pinned(rep, other)
    assert not ok and "JevBench" in why


def test_builder_runs_the_jevbench_fingerprint_by_default():
    """data.build.build_sample needs the full source set to run, so check the wiring: default on, item-tier
    matchers merged into the drop decision, context-tier into the log, identity in the summary."""
    import inspect

    from data import build as B

    sig = inspect.signature(B.build_sample)
    assert sig.parameters["jevbench_fingerprint"].default is True
    src = inspect.getsource(B.build_sample)
    assert 'item = {**item, **fingerprint_item_hits(jb_fp, rec["_probe"], {})}' in src
    assert 'full = {**full, **fingerprint_item_hits(jb_fp, rec["state"], rec["questions"])}' in src
    assert 'summary["jevbench_fingerprint"]' in src and "JG.load()[1]" in src


def test_tampered_pin_or_arrays_fail_closed(tmp_path, monkeypatch):
    JG.set_for_tests(None)
    pin = json.loads(JG.PIN_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(pin)
    bad["fingerprint"]["arrays_sha256"] = "0" * 64
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(JG, "PIN_PATH", p)
    JG.load.cache_clear()
    with pytest.raises(JG.JevBenchGuardError):
        JG.load()
    bad = copy.deepcopy(pin)
    bad["doc_index"]["index_sha256"] = "1" * 64
    p.write_text(json.dumps(bad), encoding="utf-8")
    JG.load.cache_clear()
    with pytest.raises(DG.DocOverlapError):
        JG.load()
    with pytest.raises(DG.DocOverlapError):  # the gate refuses rather than skipping JevBench
        DG.scan(CLEAN)
    monkeypatch.undo()
    JG.load.cache_clear()
    JG.load()

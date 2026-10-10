import hashlib
import json

import pytest

from jevkit import items as IT

from ._fixtures import choice, noul, score

REAL = IT.default_data_dir()


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pin(tmp_path, files):
    return {"repo": "x", "commit": "c0ffee", "raw_url": "unused", "files": files}


def test_item_validation():
    with pytest.raises(ValueError):
        IT.Item(id="a", family="f", state="s", question={"type": "noul", "instructions": "?"}, labels=["n", "y"],
                expected="y", split="public").validate()
    with pytest.raises(ValueError):
        IT.Item(id="a", family="f", state="s", question={"type": "score", "criteria": ["a", "b"]},
                labels=["0", "1"], expected="1", split="public").validate()  # score gold must be an int
    with pytest.raises(ValueError):
        IT.Item(id="a", family="f", state={"answer_key": "x"}, question={"type": "noul"}, labels=["no", "yes"],
                expected="yes", split="public").validate()
    with pytest.raises(ValueError):
        IT.Item.from_dict({**choice("c").to_dict(), "tier": "medium"})
    assert noul("n", split="private").side == "sealed" and noul("n").side == "open"


def test_load_items_tier_fill_and_duplicates(tmp_path):
    rows = [{k: v for k, v in it.to_dict().items() if k != "tier"} for it in (choice("a"), score("b"))]
    _write(tmp_path, "x.jsonl", rows)
    its = IT.load_items(tmp_path / "x.jsonl", tier="hard")
    assert [i.tier for i in its] == ["hard", "hard"]
    _write(tmp_path, "d.jsonl", rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        IT.load_items(tmp_path / "d.jsonl")


def test_pinned_loader_verifies_sha_and_counts(tmp_path):
    rows = [{k: v for k, v in it.to_dict().items() if k != "tier"} for it in (choice("a"), noul("b"))]
    sha = _write(tmp_path, "original.jsonl", rows)
    pin = _pin(tmp_path, {"original.jsonl": {"path": "p", "n": 2, "tier": "standard", "sha256": sha}})
    its = IT.load_public(tmp_path, pin)
    assert len(its) == 2 and all(i.provenance["jevkit_eval_only"] for i in its)
    assert all(i.tier == "standard" for i in its)
    (tmp_path / "original.jsonl").write_text((tmp_path / "original.jsonl").read_text() + "\n", encoding="utf-8")
    with pytest.raises(IT.PinMismatch):
        IT.load_public(tmp_path, pin)
    with pytest.raises(IT.PinMismatch, match="missing"):
        IT.verify_public(tmp_path / "nope", pin)


def test_committed_pin_matches_upstream_manifest():
    pin = IT.read_pin()
    assert pin["commit"] == "b6b8fff7e345b98c060ad26c13308860ddc67004"
    # sha256 values published in jevbench datasets/manifest.json (protocol jevbench::v1.2)
    assert pin["files"]["original.jsonl"]["sha256"].startswith("5c2414edb300")
    assert pin["files"]["easy.jsonl"]["sha256"].startswith("231df3c2c8e8")
    assert pin["files"]["hard.jsonl"]["sha256"].startswith("89e9e6becb33")
    assert sum(f["n"] for f in pin["files"].values()) == 72 + 48 + 111


@pytest.mark.skipif(not (REAL / "hard.jsonl").exists(), reason="pinned public items not downloaded (python -m jevkit fetch)")
def test_real_public_items_load():
    its = IT.load_public()
    assert len(its) == 231 and {i.tier for i in its} == {"standard", "easy", "hard"}
    assert {i.type for i in its} == {"choice", "noul", "score"}
    committed = json.loads((IT.PIN_PATH.parent / "jevbench_public_fingerprints.json").read_text(encoding="utf-8"))
    assert IT.fingerprint_manifest(its) == committed  # the eval-only list in the repo matches the pinned bytes


def test_overlap_guard_rows_feed_the_guard_unit_cutter():
    np = pytest.importorskip("numpy")  # noqa: F841 (doc_overlap_guard needs numpy)
    from train.doc_overlap_guard import eval_units

    its = [choice("a"), noul("b")]
    rows = list(IT.overlap_guard_rows(its))
    assert [r[:3] for r in rows] == [("JEVBENCH", "jevbench:a", "jevbench_fam_choice"),
                                     ("JEVBENCH", "jevbench:b", "jevbench_fam_noul")]
    fields = [f for f, _ in eval_units(rows[1][3])]
    assert fields[:2] == ["S:note", "S:detail"] and len(fields) == 4  # state keys + instructions + criteria


def test_fingerprint_manifest_is_text_free_and_stable():
    its = [choice("a"), noul("b")]
    m1, m2 = IT.fingerprint_manifest(its), IT.fingerprint_manifest(list(reversed(its)))
    assert m1 == m2 and m1["n"] == 2
    assert "state" not in json.dumps(m1)

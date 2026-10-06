import json
import os

import numpy as np
import pytest

from ssdm_data import config, dedup as D, text as T


def _row(cat, state, instructions, criteria=None):
    q = {"type": "choice", "instructions": instructions, "criteria": criteria or {"a": "x", "b": "y"}}
    return {"_evaluation": {"catalog_id": cat}, "state": state, "questions": {"q1": q}}


ITEM = "How do I locate the replacement card that I ordered last week?"
PREMISE = ("Ernest Jones is a British jeweller and watchmaker. Established in 1949, its first store was opened in "
           "Oxford Street, London, and it now trades from over two hundred stores across the country.")


@pytest.fixture(scope="module")
def fp():
    rows = [_row(4, {}, "Classify the banking intent of this user request:\n" + ITEM),
            _row(12, {}, f"Classify the relationship.\nPremise: {PREMISE}\nHypothesis: it is old")]
    # boilerplate: a label description shared by > BOILERPLATE_ROWS rows must not be indexed
    rows += [_row(5, {}, f"Classify request number {i} of the long tail please",
                  {"a": "account blocked by the bank", "b": "y"}) for i in range(D.BOILERPLATE_ROWS + 5)]
    return D.Fingerprint.build(rows, log=lambda *a: None)


def test_normalize_and_segments():
    assert T.normalize("  Héllo,   WORLD!! ") == "héllo world"
    segs = T.segments({}, {"q": {"instructions": "Premise: " + PREMISE, "criteria": {"a": "Yes"}}})
    assert T.normalize(PREMISE) in segs  # label prefix stripped variant
    assert "yes" not in segs  # too short to identify an item


def test_exact_line_match(fp):
    assert fp.check(ITEM, {})["exact"] == [4]
    assert fp.check({"msg": ITEM.upper() + "  "}, {})["exact"] == [4]  # normalisation


def test_minhash_near_duplicate(fp):
    near = ITEM.replace("the replacement", "my replacement")
    h = fp.check(near, {})
    assert h["exact"] == [] and h["minhash"] == [4]


def test_ngram13_partial_overlap(fp):
    part = "Fun fact: " + " ".join(PREMISE.split()[3:20]) + " and more unrelated words here."
    assert fp.check(part, {})["ngram13"] == [12]


def test_unrelated_text_clean(fp):
    h = fp.check("A recipe for tomato soup with basil, garlic and a splash of cream.", {})
    assert not any(h.values())


def test_boilerplate_not_indexed(fp):
    h = fp.check("account blocked by the bank", {})
    assert not any(h.values())
    assert fp.meta["segments_boilerplate_removed"] >= 1


def test_save_load_roundtrip(fp, tmp_path):
    p = tmp_path / "fp.npz"
    fp.save(p)
    fp2 = D.Fingerprint.load(p)
    assert fp2.check(ITEM, {}) == fp.check(ITEM, {})
    assert np.array_equal(fp2.ng_hash, fp.ng_hash)


def test_dedup_log_tiers():
    log = D.DedupLog({"v": 1})
    assert log.add("s", "r1", {"exact": [4], "ngram13": [], "minhash": []}) is False
    assert log.add("s", "r2", {"exact": [], "ngram13": [], "minhash": []},
                   {"exact": [], "ngram13": [2], "minhash": []}) is True
    j = log.to_json()
    assert j["by_source"]["s"] == {"checked": 2, "item_exact": 1, "dropped": 1, "kept": 1,
                                   "kept_with_context_overlap": 1, "context_ngram13": 1}


def test_minhash_signature_deterministic():
    a = D.minhash_signatures(["the quick brown fox jumps", "lorem ipsum dolor sit amet"])
    b = D.minhash_signatures(["the quick brown fox jumps", "lorem ipsum dolor sit amet"])
    assert np.array_equal(a, b) and a.shape == (2, D.MINHASH_PERMS)


SHORT = "how do i change the pin on my card"                 # 9 tokens -> 1 edit allowed
SHORT10 = "why was my top up payment declined by the bank today"  # 11 tokens -> 2 edits allowed


@pytest.fixture(scope="module")
def fp_short():
    rows = [_row(4, {}, "Classify the banking intent:\n" + SHORT), _row(4, {}, "Classify:\n" + SHORT10),
            _row(5, {}, "Classify:\nplay music")]
    return D.Fingerprint.build(rows, log=lambda *a: None)


def test_edit_one_word_substitution(fp_short):
    h = fp_short.check("how do i change the code on my card", {})
    assert h["edit"] == [4] and h["exact"] == [] and h["ngram13"] == []


def test_edit_bound_is_the_stricter_rule(fp_short):
    assert fp_short.check("how can i change the code on my card", {})["edit"] == []   # 2 edits / 9 tokens
    assert fp_short.check("why was my top up transfer declined by the bank yesterday", {})["edit"] == [4]  # 2/11
    assert fp_short.check("why was my top up transfer refused by the bank yesterday", {})["edit"] == []  # 3 edits


def test_edit_insertion_and_deletion(fp_short):
    assert fp_short.check("how do i quickly change the pin on my card", {})["edit"] == [4]
    assert fp_short.check("how do i change pin on my card", {})["edit"] == [4]


def test_contain_appended_sentence(fp_short):
    h = fp_short.check("How do I change the PIN on my card? I am travelling next week.", {})
    assert h["contain"] == [4]


def test_short_index_roundtrip(fp_short, tmp_path):
    p = tmp_path / "fp.npz"
    fp_short.save(p)
    fp2 = D.Fingerprint.load(p)
    assert fp2.check("how do i change the code on my card", {})["edit"] == [4]


@pytest.mark.skipif(not config.FINGERPRINT.exists(), reason="full Index fingerprint not built locally")
def test_full_fingerprint_meta():
    meta = json.loads(config.FINGERPRINT.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["rows"] == 124971 + 30419  # official 0.2.1 rows incl. HLE, both files, nothing filtered
    assert meta["suite_files_uncompressed_sha256"] == D.EXPECTED_SUITE_SHA256
    assert "45" in meta["rows_by_catalog"]  # HLE covered
    assert meta["version"] == D.FP_VERSION

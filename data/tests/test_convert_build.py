import itertools
import json
import random

import pytest

from ssdm_data import augment as A, convert as C, dedup as D, sources as S


def _fetched(sid):
    try:
        for f in S.REGISTRY[sid]["files"]:
            S.verified_path(sid, f)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def test_registry_lists_only_train_files():
    for sid, src in S.REGISTRY.items():
        for f in src["files"]:
            low = f.lower()
            assert not any(t in low for t in ("test", "valid", "dev")), (sid, f)


@pytest.mark.parametrize("sid", sorted(C.LOADERS))
def test_loader_yields_valid_records(sid):
    if not _fetched(sid):
        pytest.skip(f"{sid} not fetched")
    for k, base in enumerate(itertools.islice(C.LOADERS[sid](1), 40)):
        assert base["source_split"] in ("train", "oos_train")
        assert base["item_fields"] and set(base["item_fields"]) <= set(base["fields"])
        rec = A.render(base, random.Random(k))
        A.validate(rec)


def test_clinc_reads_train_keys_only():
    if not _fetched("clinc150"):
        pytest.skip("clinc150 not fetched")
    d = json.load(open(S.verified_path("clinc150", "data_full.json"), encoding="utf-8"))
    heldout = {t for key in ("test", "oos_test", "val", "oos_val") for t, _ in d[key]}
    train = {t for key in ("train", "oos_train") for t, _ in d[key]}
    seen = {b["fields"]["user_request"] for b in C.clinc150(1)}
    assert len(seen) == len(train) or seen <= train
    assert not (seen & (heldout - train))


def test_gsm8k_distractors_exclude_gold():
    for g in (0.0, 7.0, 72.0, 1250.0, 3.5):
        ds = C.gsm8k_distractors(g, 9, random.Random(1))
        assert len(ds) == 9 and C._num_fmt(g) not in ds and len(set(ds)) == 9


def test_build_end_to_end_small(tmp_path):
    need = ["banking77", "anli", "sst5", "boolq"]
    if not all(_fetched(s) for s in need):
        pytest.skip("sources not fetched")
    from ssdm_data import build as B
    rows = []
    # plant a training item in a toy fingerprint: that record must be dropped and logged
    first = next(C.banking77(B.config.SEED))
    rows.append({"_evaluation": {"catalog_id": 4}, "state": {},
                 "questions": {"q1": {"type": "choice", "instructions": "Classify:\n" + first["fields"]["customer_message"],
                                      "criteria": {"a": "x", "b": "y"}}}})
    fp = D.Fingerprint.build(rows, log=lambda *a: None)
    shares = {"banking77": 0.3, "anli": 0.2, "sst5": 0.2, "multi": 0.1, "factchange": 0.2}
    res = B.build_sample(n_target=60, out=str(tmp_path / "o"), fingerprint=fp, sources=shares, log=lambda *a: None,
                         allow_test_fingerprint=True)
    assert res["summary"]["n_kept"] == 60
    assert res["dedup_log"]["by_source"]["banking77"]["dropped"] >= 1
    assert all("_probe" not in r for r in res["records"])
    ids = [r["id"] for r in res["records"]]
    assert first["id"] not in ids and len(ids) == len(set(ids))


def _toy_fp():
    rows = [{"_evaluation": {"catalog_id": 99}, "state": "nothing in common with any training text at all here",
             "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "x", "b": "y"}}}}]
    return D.Fingerprint.build(rows, log=lambda *a: None)


def test_build_refuses_non_production_fingerprint(tmp_path):
    from ssdm_data import build as B
    with pytest.raises(RuntimeError, match="not a production fingerprint"):
        B.build_sample(n_target=10, out=str(tmp_path / "o"), fingerprint=_toy_fp(), log=lambda *a: None)
    with pytest.raises(ValueError):
        B.build_sample(n_target=10, out=str(tmp_path / "o"), receipt=str(tmp_path / "r"), fingerprint=_toy_fp(),
                       allow_test_fingerprint=True, log=lambda *a: None)


def test_fact_change_pairs_identical_questions_through_build(tmp_path):
    from ssdm_data import build as B
    shares = {"factchange": 1.0}
    res = B.build_sample(n_target=240, out=str(tmp_path / "o"), fingerprint=_toy_fp(), sources=shares,
                         log=lambda *a: None, allow_test_fingerprint=True)
    pairs = {}
    for r in res["records"]:
        pairs.setdefault(r["aug"]["fact_change"]["pair_id"], []).append(r)
    assert len(pairs) == 120
    for pid, (a, b) in pairs.items():
        assert a["questions"] == b["questions"], pid
        assert a["labels"] != b["labels"], pid
        ua, ub = a["untrusted_fields"][0], b["untrusted_fields"][0]
        assert ua == ub
        for f in a["state"]:
            if f != ua:
                assert a["state"][f] == b["state"][f], (pid, f)


def test_multi_yesno_colliding_tool_names():
    names = ["x" * 60 + "A", "x" * 60 + "B", "short"]
    base = {"id": "m2:t:c", "source": "t", "source_split": "train", "source_id": "c", "license": "x",
            "family": "tool_multi", "fields": {"tools": [], "dialogue": [{"role": "user", "content": "hi"}]},
            "qtype": "multi_yesno", "options": {n: n for n in names}, "gold": [names[0]],
            "item_fields": ["dialogue"], "item_options": "none"}
    for s in range(20):
        rec = A.render(base, random.Random(s))
        A.validate(rec)
        assert len(rec["questions"]) == 3 and len(rec["labels"]) == 3
        pos = [q for q, l in rec["labels"].items() if l in ("yes", "call", "true")]
        assert len(pos) == 1 and names[0] in rec["questions"][pos[0]]["instructions"]


def test_pair_partner_logged_as_dropped():
    log = D.DedupLog({})
    clean = {"exact": [], "ngram13": [], "minhash": []}
    assert log.add("fc", "a", {"exact": [4], "ngram13": [], "minhash": []}) is False
    assert log.add("fc", "b", clean, partner_dropped=True) is False
    c = log.to_json()["by_source"]["fc"]
    assert c["dropped"] == 2 and c.get("kept", 0) == 0 and c["dropped_pair_partner"] == 1

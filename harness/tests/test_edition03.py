"""Edition flag: 0.3 public index next to an untouched 0.2.1, the 0.3 Full-score arithmetic, and the
held-out-family split.

Sources: kit PR #74 head 62d2f51 (index-0.3.json, tests/fixtures/board-0.3.json) and the live board
Space data/v03.json (both vendored, hashes pinned in harness/config.py)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from decision_index.scoring import index02 as X

from harness import bootstrap, config, families, full03, predictors, score
from harness.suite import edition_rows, in_edition, index_ids, spec

FIX = Path(__file__).parent / "fixtures"
BOARD03 = json.loads((FIX / "board-0.3.json").read_text(encoding="utf-8"))
V03 = json.loads((FIX / "v03.json").read_text(encoding="utf-8"))


def test_vendored_files_match_pins():
    assert hashlib.sha256(config.INDEX_03_SPEC.read_bytes()).hexdigest() == config.INDEX_03_SPEC_SHA256
    assert hashlib.sha256((FIX / "v03.json").read_bytes()).hexdigest() == config.BOARD_V03_SHA256


def test_default_edition_is_still_021():
    assert config.EDITION == "0.2.1"
    assert spec() is spec("0.2.1") and spec()["edition"] == "0.2.1"
    assert len(index_ids()) == 38 and 48 in index_ids()
    assert score.weights() == score.weights(edition="0.2.1")


def test_03_panel_and_fixed_area_weights():
    s = spec("0.3")
    areas = {a["id"]: a["benchmarks"] for a in s["areas"]}
    ids = index_ids("0.3")
    assert len(ids) == len(set(ids)) == 37
    assert 28 in areas["knowledge"] and 28 not in areas["language"]
    assert 48 not in ids and not s["loss_rules"]
    w = score.weights(edition="0.3")
    assert sum(w.values()) == pytest.approx(1.0)
    got = {a: round(sum(w[n] for n in b), 6) for a, b in areas.items()}
    assert got == {"knowledge": 0.258494, "language": 0.258494, "retrieval": 0.200229, "tools": 0.182783, "arts": 0.1}
    # 0.3 keeps the 0.2.1 area weights even though knowledge now has 11 benchmarks (rule "fixed", not sqrt)
    w21 = score.weights(edition="0.2.1")
    assert sum(w21[n] for n in spec("0.2.1")["areas"][0]["benchmarks"]) == pytest.approx(0.258494, abs=1e-6)


@pytest.mark.parametrize("engine", sorted(BOARD03["entrants"]))
def test_03_public_index_reproduces_board(engine):
    e = BOARD03["entrants"][engine]
    skills = {int(n): np.array([b["skill"]]) for n, b in e["benchmarks"].items()}
    assert set(skills) == set(index_ids("0.3"))
    got = float(score.index_from_skills(skills, renormalize=False, edition="0.3")[0])
    assert got == pytest.approx(e["public_index"], abs=0.01)
    # and the same as the kit's own 0.3 aggregate over the vendored spec
    scores, _ = X.aggregate({int(n): dict(b) for n, b in e["benchmarks"].items()}, spec("0.3"))
    assert got == pytest.approx(scores["balanced_skill"], abs=1e-9)


def test_board_fixture_and_live_v03_agree_on_public():
    live = {m["engine"]: m["pub"] for m in V03["models"]}
    for k, e in BOARD03["entrants"].items():
        if k in live:
            assert live[k] == pytest.approx(e["public_index"], abs=0.005), k
    assert live["clef"] == 61.71 and live["clef-flash"] == 56.15


def test_in_edition_rules():
    row = lambda n, rid: {"_evaluation": {"catalog_id": n, "run_id": rid}}
    assert not in_edition(row(48, "x"), "0.3") and in_edition(row(48, "x"), "0.2.1")
    assert not in_edition(row(30, "GSM8K-4:test:0"), "0.3") and in_edition(row(30, "GSM8K-4:test:0"), "0.2.1")
    assert in_edition(row(30, "v0.3-gsm8k:GSM8K-4:test:0"), "0.3")
    assert in_edition(row(28, "x"), "0.3") and not in_edition(row(6, "x"), "0.3")  # 6 = RouterBench, display only
    with pytest.raises(ValueError):
        config.edition_id("0.2")


# ----------------------------------------------------------------------------- Full score (v03.json)


def test_full_score_reproduces_every_board_entrant():
    w, eq, tie = full03.params_from_board(V03)
    assert (w, eq, tie) == (full03.WEIGHTS, full03.EQUATING, full03.TIE_BAND)
    for m in V03["models"]:
        assert full03.full(m["pub"], m["same_skills"], m["new_domains"]) == pytest.approx(m["v"], abs=0.015), m["engine"]
        assert max(0.0, full03.equate(m["same_skills"], "same_skills")) == pytest.approx(m["eq"]["S"], abs=0.015)
        assert max(0.0, full03.equate(m["new_domains"], "new_domains")) == pytest.approx(m["eq"]["O"], abs=0.025)


def test_full_score_tie_ranks_reproduce_board():
    r = full03.ranks({m["engine"]: m["v"] for m in V03["models"]})
    for m in V03["models"]:
        assert r[m["engine"]] == (m["rank_tie"], m["tied"]), m["engine"]
    assert r["clef"] == (19, False) and r["clef-flash"] == (31, True) and r["jev"] == (2, True)


def test_full_slopes():
    s = full03.full_slope()
    assert s["public"] == 0.2 and s["same_skills"] == pytest.approx(0.5058, abs=1e-4) and s["new_domains"] == pytest.approx(0.4037, abs=1e-4)


# ----------------------------------------------------------------------------- held-out families


def test_every_benchmark_has_a_family_tier():
    d = families.load()
    for e in ("0.2.1", "0.3"):
        for n in index_ids(e):
            assert families.tier_of(n) in families.TIERS, n
    ids = set(index_ids("0.3"))
    parts = [set(families.members(g, "0.3")) for g in ("same_source", "same_family", "new_domain_proxy")]
    assert set().union(*parts) == ids and sum(map(len, parts)) == len(ids)
    assert set(families.members("in_domain", "0.3")) == parts[0] | parts[1]
    # every training source with a direct train split maps to its benchmark
    assert {n for n in ids if families.tier_of(n) == "same_source"} == {30, 28, 12, 29, 4, 5, 62, 21}
    assert d["id"] == "families-v1"


def test_group_index_is_index_restricted():
    ids = index_ids("0.3")
    sk = {n: np.array([0.5]) for n in ids}
    for g in families.GROUPS:
        assert float(families.group_index(sk, families.members(g, "0.3"), "0.3")[0]) == pytest.approx(50.0)
    sk2 = {n: np.array([1.0 if families.tier_of(n) == "none" else 0.0]) for n in ids}
    assert float(families.group_index(sk2, families.members("new_domain_proxy", "0.3"), "0.3")[0]) == pytest.approx(100.0)
    assert float(families.group_index(sk2, families.members("in_domain", "0.3"), "0.3")[0]) == pytest.approx(0.0)


# ----------------------------------------------------------------------------- on the DEV slice (local suite)


@pytest.fixture(scope="module")
def noisy03(dev_rows):
    return predictors.predict(dev_rows, "noisy", seed=13, acc=0.6, status_drop=0.03)


def test_03_decomposition_matches_official_on_dev(dev_rows, noisy03):
    off = score.official(dev_rows, noisy03, edition="0.3")
    benches = score.decompose(dev_rows, noisy03, edition="0.3")
    assert score.edition_of(benches) == "0.3"
    assert 48 not in benches and 30 not in benches  # retired; 0.2.1 GSM8K rows don't count in 0.3
    dec = score.point(benches, renormalize=False)
    for n, v in dec["per_benchmark"].items():
        assert v["skill"] == pytest.approx(off["values"][n]["skill"], abs=1e-4), n
    assert off["values"][30]["rule"] == "missing" and off["index_entry"] is None
    assert dec["index"] == pytest.approx(off["scores"]["balanced_skill"], abs=1e-3)


def test_03_differs_from_021_only_through_panel(dev_rows, noisy03):
    b21 = score.decompose(dev_rows, noisy03)
    b03 = score.decompose(dev_rows, noisy03, edition="0.3")
    p21, p03 = score.point(b21), score.point(b03)
    for n in p03["per_benchmark"]:
        if n not in (30, 48):
            assert p03["per_benchmark"][n] == pytest.approx(p21["per_benchmark"][n], abs=1e-12), n
    assert p03["edition"] == "0.3" and p21["edition"] == "0.2.1"
    sk = {n: np.array([v["skill"]]) for n, v in p21["per_benchmark"].items() if n not in (30, 48)}
    assert float(score.index_from_skills(sk, edition="0.3")[0]) == pytest.approx(p03["index"], abs=1e-9)
    assert float(score.index_from_skills(sk, edition="0.2.1")[0]) != pytest.approx(p03["index"], abs=1e-6)  # weights differ


def test_03_compare_and_family_split(dev_rows, noisy03):
    other = predictors.predict(dev_rows, "noisy", seed=14, acc=0.7, status_drop=0.03)
    a = score.decompose(dev_rows, other, edition="0.3")
    b = score.decompose(dev_rows, noisy03, edition="0.3")
    out = bootstrap.compare(a, b, B=50, seed=0)
    assert out["edition"] == "0.3" and out["candidate"]["edition"] == "0.3"
    ra, rb = bootstrap.replicate_skills([a, b], B=50, seed=0)
    fam = families.compare(ra, rb, edition="0.3")
    assert fam["in_domain"]["point"] > 0 and fam["new_domain_proxy"]["point"] > 0
    with pytest.raises(ValueError):
        bootstrap.compare(a, score.decompose(dev_rows, noisy03), B=10)


def test_edition_rows_filters_dev(dev_rows):
    kept = edition_rows(dev_rows, "0.3")
    cats = {r["_evaluation"]["catalog_id"] for r in kept}
    assert 48 not in cats and 30 not in cats and len(kept) < len(dev_rows)
    assert len(edition_rows(dev_rows, "0.2.1")) == len(dev_rows)

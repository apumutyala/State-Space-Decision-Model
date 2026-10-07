"""Cohort v3 (train/cohort_v3.py): the G13 retention role expanded from the PR #44 150k set, frozen and pinned.

Offline checks use only committed files. The byte-for-byte rebuild and the first-cell draw invariance need the 150k
set's pointer-pinned files (outside git); they run when ``SSDM_M2_150K_DIR`` names that directory (or the default
local ssdm-data path exists) AND ``COHORT_V3_REBUILD=1`` (about 3 minutes, CPU)."""

import gzip
import hashlib
import math
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from train import cohort_v3 as V
from train.cohort import OUT, Cohort, CohortLeak, read_content_map, read_pr16_manifest, record_item_keys

ROOT = Path(__file__).resolve().parents[1]
#: sha256 of the uncompressed m2-dev-cohort-v3 text. Changing the cohort means changing this pin (and re-caching).
V3_SHA256 = "498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da"
V2_SHA256 = "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
RECEIPT = ROOT / "receipts" / "2026-10-07-D-cohort-v3"


@pytest.fixture(scope="module")
def v3():
    return Cohort(V.OUT_V3)


@pytest.fixture(scope="module")
def new(v3):
    return [e for e in v3.roles["retention"] if e.get("pool") == V.POOL_150K]


def _lines(path):
    return gzip.open(path, "rt", encoding="utf-8").read().split("\n")


def test_v3_is_frozen_and_pinned(v3):
    assert v3.sha256 == V3_SHA256
    assert (V.OUT_V3.with_name("m2-dev-cohort-v3.sha256")).read_text(encoding="utf-8").split()[0] == V3_SHA256
    assert Cohort(expected_sha256=V3_SHA256).sha256 == V3_SHA256  # a config pinning v3 opens v3
    assert Cohort().sha256 == V2_SHA256 and Cohort(expected_sha256=V2_SHA256).path == OUT  # v2 stays the default
    with pytest.raises(ValueError):
        Cohort(V.OUT_V3, expected_sha256=V2_SHA256)


def test_unpinned_cohort_paths_are_refused(tmp_path):
    p = tmp_path / "copy.jsonl.gz"
    p.write_bytes(V.OUT_V3.read_bytes())
    with pytest.raises(ValueError, match="unpinned"):
        Cohort(p)
    assert Cohort(p, expected_sha256=V3_SHA256).sha256 == V3_SHA256
    assert Cohort(p, allow_unpinned=True).sha256 == V3_SHA256
    with pytest.raises(ValueError):
        Cohort(p, expected_sha256=V2_SHA256)


def test_every_v2_entry_is_unchanged_and_in_order(v3):
    v2l, v3l = _lines(OUT)[1:], _lines(V.OUT_V3)[1:]
    assert [x for x in v3l if '"pool":"m2-150k"' not in x] == v2l
    assert v3.header["extends"] == {"cohort": "m2-dev-cohort-v2", "sha256": V2_SHA256,
                                    "note": "every v2 entry unchanged and in order; retention += pool m2-150k records"}
    for role in ("dev", "calibration", "steer_dev", "steer_gate"):
        assert v3.header["n_actual"][role] == len(v3.roles[role]) == len(Cohort().roles[role])


def test_retention_is_at_least_32k_items_and_sized_as_recorded(v3, new):
    assert len(new) >= V.N_RETENTION_NEW and len(new) - V.N_RETENTION_NEW <= len(
        v3.header["sources"]["retention_150k"]["selection"]["strata"])  # whole groups overshoot by < 1 group/stratum
    assert v3.header["n_actual"]["retention"] == len(v3.roles["retention"]) == 1670 + len(new)
    m = json.loads((RECEIPT / "measure.json").read_text(encoding="utf-8"))
    assert m["cohort_sha256"] == V3_SHA256
    from train.margin_sim import REQUIRED_N

    for fam in ("rwkv7", "qwen3.5"):  # MEASURED usable items: K <= 16, T <= 2,048, the parent's template + tokenizer
        got = m["families"][fam]["totals"]["all"]["items_kept"]
        assert got >= REQUIRED_N["g13_retention_ece"] and got >= REQUIRED_N["g13_retention_acc"]  # the gate floor
        assert got >= math.ceil(1.08 * REQUIRED_N["g13_retention_ece"])  # coordinator headroom (35,964)
        assert m["families"][fam]["totals"]["all"]["refused"] == 0


def test_roles_are_disjoint_by_id_content_and_source_item(v3, new):
    ids = [e["id"] for r in ("dev", "calibration", "retention") for e in v3.roles[r]]
    assert len(ids) == len(set(ids))
    old = [e for r in ("dev", "calibration", "retention") for e in v3.roles[r] if e.get("pool") != V.POOL_150K]
    assert not {e["content_sha256"] for e in new} & {e["content_sha256"] for e in old}
    old_items = set().union(*(record_item_keys(e["id"]) for e in old))
    assert not any(record_item_keys(e["id"]) & old_items for e in new)
    assert v3.header["exclude_source_items"] is True and v3.exclude_items


def test_new_retention_is_outside_every_m2_training_draw(v3, new):
    sel = v3.header["sources"]["retention_150k"]["selection"]
    d = sel["draw"]
    assert d["reserve"] == V.DRAW_RESERVE == 60_000 and d["seed"] == 0
    assert d["first_cell_draw_ids_sha256"] == V.FIRST_CELL_DRAW["drawn_ids_sha256"]  # the pinned 24k draw
    assert all(V.draw_key(e["id"]) > d["reserve_cutoff_key"] for e in new)  # beyond the seed-0 reserve prefix
    assert not {e["id"] for e in new} & set(V.HELDOUT_EXCLUDED_150K)
    assert not any("~inj:" in e["id"] for e in new)  # no D1-A / hijack-injected record
    reasons = sel["ineligible_records_by_reason"]
    assert reasons.get("cohort_v2", 0) == 0 and reasons["draw_reserve"] >= V.DRAW_RESERVE
    assert v3.header["exclude_text_keys"] is True and all(e.get("text_keys") for e in new)
    nd = v3.header["sources"]["retention_150k"]["near_duplicates"]
    assert nd["n_candidates_flagged"] > 0 and set(nd["by_reference"]) <= {"shingle_reserve", "shingle_pr16"}
    assert (V.SHINGLE_K, V.SHINGLE_FRAC) == (13, 0.5)


def test_review_b1_text_twins_are_not_in_v3(new):
    """PR #105 review B1: the 196 v3 (0c0309b9) records with a key-free text twin in the 24k draw are excluded now."""
    pairs = json.loads((RECEIPT / "review-b1-pairs.json").read_text(encoding="utf-8"))["pairs_full_24k"]
    assert len(pairs) == 196
    ids = {e["id"] for e in new}
    assert not ids & {a for a, _, _ in pairs} and not ids & {b for _, b, _ in pairs}
    assert "m2:glaive_fc:100743" not in ids and "m2:factchange:renewal:1124:v0" not in ids


def test_text_keys_ignore_keys_option_ids_and_shape():
    from train.cohort import record_text_keys

    a = {"id": "m2:x:1", "state": "Patient: BMI for 70 kg, 1.75 m?",
         "questions": {"result": {"type": "choice", "instructions": "Pick the call.",
                                  "criteria": {"A": "calculate_bmi", "B": "no_tool"}}}}
    b = {"id": "m2:x:2", "state": {"conversation": "patient bmi for 70 kg 1 75 m"},
         "questions": {"answer": {"type": "choice", "instructions": "Pick the call",
                                  "criteria": {"no_tool": "No tool", "calculate_bmi": "Calculate BMI"}}}}
    b["questions"]["answer"]["criteria"] = {"opt_1": "no_tool", "opt_2": "calculate_bmi"}
    assert record_text_keys(a) == record_text_keys(b)
    c = dict(b, state="patient bmi for 80 kg 1 75 m")
    assert not record_text_keys(a) & record_text_keys(c)


def test_new_retention_shares_nothing_with_the_pr16_sample(new):
    ids = [r["id"] for r in read_pr16_manifest()]
    items = set().union(*(record_item_keys(i) for i in ids))
    cm = read_content_map()
    assert not any(record_item_keys(e["id"]) & items for e in new)
    assert not {e["content_sha256"] for e in new} & {cm[i] for i in ids}


def test_new_retention_is_not_an_index_dev_test_row(v3, new):
    from train.guard import SplitGuard

    m = ROOT / "harness" / "manifests"
    g = SplitGuard.from_manifests([m / "dev-manifest.jsonl.gz", m / "test-manifest.jsonl.gz"])
    assert not {e["id"] for e in new} & g.ids
    gi = v3.header["sources"]["retention_150k"]["guards"]
    pin = json.loads((ROOT / "jobs/m2_pilot/guard_pin.json").read_text(encoding="utf-8"))
    assert gi["split_guard"]["content_index"]["sha256"] == pin["content_index"]["sha256"]
    assert sorted(gi["split_guard"]["manifests"].values()) == sorted(pin["harness_manifests_sha256"].values())
    from harness import config as hconfig
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, template_set_sha256

    assert gi["frozen_families"]["heldout_draw_v1_sha256"] == hconfig.HELDOUT_DRAW_V1_SHA256
    assert len(gi["frozen_families"]["families"]) == 9
    assert gi["template_scan"]["d1_a_sha256"] == template_set_sha256(D1_TEMPLATES_A)
    assert gi["template_scan"]["d1_b_sha256"] == template_set_sha256(D1_TEMPLATES_B)
    assert v3.header["sources"]["retention_150k"]["guard_excluded"] == {}


def test_new_retention_is_stratified_by_source_and_length(v3, new):
    strata = v3.header["sources"]["retention_150k"]["selection"]["strata"]
    q = Counter()
    for s, v in strata.items():
        q[s.split("|")[0]] += v["quota"]
    got = Counter(e["source"] for e in new)
    for src, want in q.items():
        n_str = sum(1 for s in strata if s.startswith(src + "|"))
        assert want <= got[src] <= want + 2 * n_str, src  # whole groups (fact-change pairs are 2) overshoot a little
    comp = v3.header["sources"]["retention_150k"]["composition_new"]
    assert comp["n_records"] == len(new) and dict(Counter(e["source"] for e in new)) == comp["by_source"]
    assert set(comp["items_by_area"]) == {"arts", "knowledge", "language", "retrieval", "tools"}


def test_item_keys_are_data_build_item_keys():
    from data.build import item_keys

    for i in ("m2:anli:1", "m2:anli:1~inj:d1a:x", "m2:multi:m2:anli:1+m2:sst5:2~inj:h", "m2:factchange:schedule:7:b"):
        assert record_item_keys(i) == item_keys(i)


def test_training_draw_guard_refuses_any_overlap(v3, new):
    e = new[0]
    v3.assert_draw_disjoint(["m2:anli:not-held"], {"m2:anli:not-held": "0" * 64})
    with pytest.raises(CohortLeak):
        v3.assert_draw_disjoint([e["id"]])
    with pytest.raises(CohortLeak, match="content"):
        v3.assert_draw_disjoint(["m2:x:twin"], {"m2:x:twin": e["content_sha256"]})
    with pytest.raises(CohortLeak, match="source item"):  # same item as a multi part / injected variant
        v3.assert_draw_disjoint([f"m2:multi:{e['id']}+m2:sst5:999999"])
    with pytest.raises(CohortLeak, match="source item"):
        v3.assert_draw_disjoint([e["id"] + "~inj:d1a:x"])
    twin = {"id": "m2:x:texttwin", "state": "s", "questions": {}}
    from train import cohort as C

    orig = C.record_text_keys
    try:  # a drawn record whose key-free text equals a held-out record's
        C.record_text_keys = lambda r: set(e["text_keys"]) if r is twin else orig(r)
        with pytest.raises(CohortLeak, match="key-free text"):
            v3.assert_draw_disjoint([twin["id"]], records=[twin])
    finally:
        C.record_text_keys = orig


def test_training_exclusion_removes_v3_before_the_draw(v3, new):
    from baselines.clef_teacher import draw_ids

    held = [x["id"] for x in new[:50]]
    toy = held + [f"m2:toy:{k}" for k in range(500)] + ["m2:toy:twin"]
    content = {i: hashlib.sha256(i.encode()).hexdigest() for i in toy}
    content["m2:toy:twin"] = new[60]["content_sha256"]
    ex = V.training_exclusion(toy, content, v3)
    assert ex["ids"] == set(held) | {"m2:toy:twin"}
    tk = {"m2:toy:7": set(new[70]["text_keys"])}  # key-free text twin
    assert "m2:toy:7" in V.training_exclusion(toy, content, v3, text_keys=tk)["ids"]
    with pytest.raises(V.PoolMismatch, match="lacks"):  # the pinned 150k build must hold every v3 pool record
        V.training_exclusion(toy, content, v3, pool=V.POOL_150K)
    drawn = draw_ids(toy, ex["ids"], 400, 0)
    V.assert_draw_disjoint(drawn, content, v3)
    assert not set(drawn) & ex["ids"]


def test_retention_records_need_their_pool_dir(v3):
    with pytest.raises(ValueError, match="m2-150k"):
        v3.load_role_records("unused.jsonl.gz", "retention")


def _pool_dir():
    d = Path(os.environ.get("SSDM_M2_150K_DIR", r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build"
                                                r"\sample-150000-seed20261006"))
    if os.environ.get("COHORT_V3_REBUILD") != "1" or not (d / "records.jsonl.gz").exists():
        pytest.skip("set COHORT_V3_REBUILD=1 and SSDM_M2_150K_DIR to the 150k set's files")
    return d


def test_rebuild_is_byte_identical_and_the_first_cell_draw_is_unchanged(v3):
    d = _pool_dir()
    text, _, _ = V.build_cohort_v3(d)
    assert hashlib.sha256(text.encode()).hexdigest() == V3_SHA256
    from train.run_cell import split_ids

    fs = V.pool_files(V.POOL_150K, d)
    man = [r["id"] for r in V.read_manifest_rows(fs["manifest"])]
    content = read_content_map(fs["content_map"])
    pool = sorted(set(man) - set(V.HELDOUT_EXCLUDED_150K))
    ex = V.training_exclusion(man, content, v3, pool=V.POOL_150K)
    assert ex["ids"] == set(v3.pool_ids(V.POOL_150K))  # whole groups: no twin / item left in the set
    before = split_ids(pool, 24_000, 0, 0)[0]
    after = split_ids(sorted(set(pool) - ex["ids"]), 24_000, 0, 0)[0]
    assert before == after
    assert hashlib.sha256("\n".join(after).encode()).hexdigest() == V.FIRST_CELL_DRAW["drawn_ids_sha256"]
    after60 = split_ids(sorted(set(pool) - ex["ids"]), V.DRAW_RESERVE, 0, 0)[0]
    assert after60 == split_ids(pool, V.DRAW_RESERVE, 0, 0)[0]


def _toy_train_set(d: Path, ids: list[str]) -> Path:
    """A pointer-shaped training set (records / manifest / content map with real hashes) and its pointer file."""
    from train.cohort import record_content_hash

    recs = [{"id": i, "family": "toy", "source": "toy", "state": {"ticket": f"toy ticket {k}: printer {k} jammed"},
             "questions": {"team": {"type": "choice", "instructions": "Which team?",
                                    "criteria": {"it": "IT", "fac": "Facilities"}}},
             "labels": {"team": "it"}} for k, i in enumerate(ids)]
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in recs]
    raw = {"records.jsonl.gz": "".join(x + "\n" for x in lines),
           "toy.manifest.jsonl.gz": "".join(json.dumps({"id": r["id"], "sha256": hashlib.sha256(x.encode()).hexdigest(),
                                                        "source": "toy"}) + "\n" for r, x in zip(recs, lines)),
           "toy.content.jsonl.gz": "".join(json.dumps({"content_sha256": record_content_hash(r), "id": r["id"]}) + "\n"
                                           for r in recs)}
    for name, text in raw.items():
        with gzip.open(d / name, "wt", encoding="utf-8", newline="\n") as f:
            f.write(text)
    ptr = {"build": "toy", "manifest": "toy.manifest.jsonl.gz", "content_map": "toy.content.jsonl.gz",
           "n_records": len(ids), **{k: hashlib.sha256(raw[f].encode()).hexdigest() for k, f in (
               ("records_uncompressed_sha256", "records.jsonl.gz"),
               ("manifest_uncompressed_sha256", "toy.manifest.jsonl.gz"),
               ("content_map_uncompressed_sha256", "toy.content.jsonl.gz"))}}
    (d / "pointer.json").write_text(json.dumps(ptr), encoding="utf-8")
    return d / "pointer.json"


def test_train_set_draw_excludes_cohort_v3_and_rechecks(tmp_path, v3, new):
    """train.train_set.draw (m2 first cell, q9_feature_cache, fla_rwkv7 train mix) removes v3 before split_ids."""
    from train.run_cell import split_ids
    from train.train_set import draw

    held = [e["id"] for e in new[:20]]
    item_twin = f"m2:multi:{new[30]['id']}+m2:sst5:999999"  # renders a v3 source item
    clean = [f"m2:toy:{k}" for k in range(200)]
    ids = held + [item_twin] + clean
    ptr = _toy_train_set(tmp_path / "set", ids)
    cfg = {"train_set": {"pointer": str(ptr)}, "cohort_sha256": V2_SHA256, "seed": 0, "n_train_records": 150}
    _, tr, ident = draw(cfg, tmp_path / "set")
    assert not set(tr) & (set(held) | {item_twin})
    assert tr == split_ids(sorted(clean), 150, 0, 0)[0]  # exactly the draw over the pool minus v3
    assert ident["cohort_v3_excluded"]["n"] == 21 and ident["cohort_v3_excluded"]["cohort"]["sha256"] == V3_SHA256
    with pytest.raises(Exception, match="pool"):  # v3 records never count toward the pool
        draw({**cfg, "n_train_records": 201}, tmp_path / "set")
    # a cell pinning v3 itself is allowed to point at the set v3 was carved from
    _, tr3, _ = draw({**cfg, "cohort_sha256": V3_SHA256}, tmp_path / "set")
    assert tr3 == tr

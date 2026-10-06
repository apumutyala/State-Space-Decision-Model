"""Slices are deterministic given the seed, disjoint, keep linked groups whole, and match the
committed manifests."""

import collections
import json
import random

import pytest

from harness import cases, config, slicing
from harness.suite import area_of, index_ids, row_sha256

from .conftest import needs_suite


def synthetic_rows(seed=0):
    """Rows shaped like suite rows for every index benchmark (no real content)."""
    rng = random.Random(seed)
    rows = []
    for n in index_ids():
        if n == 45:
            continue  # mirror the gated-HLE local build
        for g in range(rng.randint(60, 400)):
            for c in range(rng.choice([1, 1, 1, 2, 5])):
                rows.append({"state": {}, "questions": {}, "expected": {}, "metadata": {}, "_evaluation": {"run_id": f"{n}:t:{g}:{c}", "catalog_id": n, "group_id": f"g{g}", "track": "t", "proxy_tokens": int(rng.lognormvariate(6, 1.5)), "payload_sha256": f"p{n}:{g}:{c}"}})
    return rows


def test_deterministic_given_seed():
    rows = synthetic_rows()
    a, _ = slicing.build(rows, seed=7)
    b, _ = slicing.build(list(reversed(rows)), seed=7)
    for name in ("dev", "test"):
        assert sorted(r["_evaluation"]["run_id"] for r in a[name]) == sorted(r["_evaluation"]["run_id"] for r in b[name])


def test_seed_changes_selection():
    rows = synthetic_rows()
    a, _ = slicing.build(rows, seed=7)
    b, _ = slicing.build(rows, seed=8)
    assert {r["_evaluation"]["run_id"] for r in a["dev"]} != {r["_evaluation"]["run_id"] for r in b["dev"]}


def test_disjoint_and_groups_whole():
    rows = synthetic_rows()
    s, _ = slicing.build(rows, seed=7)
    dev = {slicing.group_key(r) for r in s["dev"]}
    test = {slicing.group_key(r) for r in s["test"]}
    assert not dev & test
    size = collections.Counter(slicing.group_key(r) for r in rows)
    for name in ("dev", "test"):
        got = collections.Counter(slicing.group_key(r) for r in s[name])
        assert all(got[g] == size[g] for g in got)


def test_pool_left_and_cap_respected():
    rows = synthetic_rows()
    s, table = slicing.build(rows, seed=7, cap=0.45)
    for n, t in table.items():
        if t.get("missing_locally"):
            continue
        # each slice overshoots its quota by at most one case per length bin
        slack = t["max_case_rows"] * len(t["bins"])
        for name in ("dev", "test"):
            assert sum(t[name].values()) <= int(0.45 * t["rows"]) + slack
        assert sum(t["dev"].values()) + sum(t["test"].values()) < t["rows"]


def test_shared_source_key_and_identical_payload_stay_on_one_side():
    rows = synthetic_rows()
    # tie every New Yorker (64) group to one of 10 contests, and duplicate payloads across CLadder (44) groups
    for r in rows:
        e = r["_evaluation"]
        if e["catalog_id"] == 64:
            r["metadata"]["contest_number"] = int(e["group_id"][1:]) % 10
        if e["catalog_id"] == 44:
            e["payload_sha256"] = f"dup{int(e['group_id'][1:]) % 50}"
    s, _ = slicing.build(rows, seed=7)
    rep = cases.overlap_report(s["dev"], s["test"])
    assert rep == {}
    assert s["dev"] and s["test"]


def test_adding_missing_benchmark_does_not_move_others():
    rows = synthetic_rows()
    a, _ = slicing.build(rows, seed=7)
    hle = [{"state": {}, "questions": {}, "expected": {}, "_evaluation": {"run_id": f"45:h:{i}", "catalog_id": 45, "group_id": f"h{i}", "track": "h", "proxy_tokens": 300, "payload_sha256": f"h{i}"}} for i in range(501)]
    b, _ = slicing.build(rows + hle, seed=7)
    for name in ("dev", "test"):
        before = {r["_evaluation"]["run_id"] for r in a[name]}
        after = {r["_evaluation"]["run_id"] for r in b[name] if r["_evaluation"]["catalog_id"] != 45}
        assert before == after


@needs_suite
def test_committed_manifests_reproduce(all_index_rows):
    meta = json.loads((config.MANIFEST_DIR / "slices.json").read_text())
    import random

    shuffled = list(all_index_rows)
    random.Random(0).shuffle(shuffled)  # loader order must not matter
    s, _ = slicing.build(shuffled, seed=meta["seed"], area_target=meta["area_target"], cap=meta["cap"], max_case_share=meta["max_case_share"])
    for name in ("dev", "test"):
        committed = slicing.read_manifest(config.MANIFEST_DIR / f"{name}-manifest.jsonl.gz")
        rebuilt = list(slicing.manifest_entries(s[name]))
        assert rebuilt == committed


@needs_suite
def test_real_slices_disjoint_and_sized():
    meta = json.loads((config.MANIFEST_DIR / "slices.json").read_text())
    dev = slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")
    test = slicing.read_manifest(config.MANIFEST_DIR / "test-manifest.jsonl.gz")
    assert not {m["run_id"] for m in dev} & {m["run_id"] for m in test}
    assert not {(m["catalog_id"], m["group_id"]) for m in dev} & {(m["catalog_id"], m["group_id"]) for m in test}
    assert len(dev) >= 10_000
    amap = area_of()
    by_area = collections.Counter(amap[m["catalog_id"]] for m in dev)
    for a, n in by_area.items():
        assert n >= 2000, (a, n)
    assert meta["disjoint_rows"] and meta["disjoint_groups"] and meta["disjoint_source_cases"]


@needs_suite
def test_real_slices_share_no_source(dev_rows, all_index_rows):
    test_ids = {m["run_id"] for m in slicing.read_manifest(config.MANIFEST_DIR / "test-manifest.jsonl.gz")}
    test_rows = [r for r in all_index_rows if r["_evaluation"]["run_id"] in test_ids]
    assert cases.overlap_report(dev_rows, test_rows) == {}


@needs_suite
def test_dev_rows_match_manifest_hashes(dev_rows):
    dev = slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")
    assert [row_sha256(r) for r in dev_rows] == [m["row_sha256"] for m in dev]

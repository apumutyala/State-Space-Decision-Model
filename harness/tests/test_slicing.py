"""Slices are deterministic given the seed, disjoint, keep linked groups whole, and match the
committed manifests."""

import collections
import json
import random

import pytest

from harness import config, slicing
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
                rows.append({"state": {}, "questions": {}, "expected": {}, "_evaluation": {"run_id": f"{n}:t:{g}:{c}", "catalog_id": n, "group_id": f"g{g}", "track": "t", "proxy_tokens": int(rng.lognormvariate(6, 1.5)), "payload_sha256": "x"}})
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
        used = sum(t["dev"].values()) + sum(t["test"].values())
        assert used <= t["rows"]


def test_adding_missing_benchmark_does_not_move_others():
    rows = synthetic_rows()
    a, _ = slicing.build(rows, seed=7)
    hle = [{"state": {}, "questions": {}, "expected": {}, "_evaluation": {"run_id": f"45:h:{i}", "catalog_id": 45, "group_id": f"h{i}", "track": "h", "proxy_tokens": 300, "payload_sha256": "x"}} for i in range(501)]
    b, _ = slicing.build(rows + hle, seed=7)
    for name in ("dev", "test"):
        before = {r["_evaluation"]["run_id"] for r in a[name]}
        after = {r["_evaluation"]["run_id"] for r in b[name] if r["_evaluation"]["catalog_id"] != 45}
        assert before == after


@needs_suite
def test_committed_manifests_reproduce(all_index_rows):
    meta = json.loads((config.MANIFEST_DIR / "slices.json").read_text())
    s, _ = slicing.build(all_index_rows, seed=meta["seed"], area_target=meta["area_target"], cap=meta["cap"])
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
    assert meta["disjoint_rows"] and meta["disjoint_groups"]


@needs_suite
def test_dev_rows_match_manifest_hashes(dev_rows):
    dev = slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")
    assert [row_sha256(r) for r in dev_rows] == [m["row_sha256"] for m in dev]

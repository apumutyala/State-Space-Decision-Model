"""Anonymous receipt packet: recomputes the official aggregates exactly and leaks no ids, text
or labels (Codex request on PR #7)."""

import gzip
import json

import numpy as np
import pytest

from harness import packet, predictors, score
from harness.cases import case_ids


@pytest.fixture(scope="module")
def built(dev_rows, tmp_path_factory):
    systems = {"a": predictors.predict(dev_rows, "noisy", seed=31, acc=0.6, status_drop=0.02), "b": predictors.predict(dev_rows, "noisy", seed=32, acc=0.55)}
    p = packet.build(dev_rows, systems)
    path = tmp_path_factory.mktemp("pk") / "packet.json.gz"
    packet.save(p, path)
    return dev_rows, systems, packet.load(path), path


def test_packet_recomputes_official_aggregates(built):
    rows, systems, p, _ = built
    r = packet.recompute(p, pairs=[("a", "b")], B=100)
    for name in systems:
        e = r["systems"][name]
        assert e["max_abs_skill_gap_vs_official"] < 1e-9 and e["max_abs_raw_gap_vs_official"] < 1e-9 and e["index_gap_vs_official"] < 1e-9
    # paired point difference is invariant to the anonymising shuffle
    direct = score.point(score.decompose(rows, systems["a"]))["index"] - score.point(score.decompose(rows, systems["b"]))["index"]
    assert r["pairs"]["a - b"]["diff"]["point"] == pytest.approx(direct, abs=1e-9)


def _strings(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            _strings(v, out)
    elif isinstance(x, list):
        for v in x:
            _strings(v, out)
    elif isinstance(x, str):
        out.add(x)
    return out


def test_packet_carries_no_ids_text_or_labels(built):
    rows, _, p, path = built
    text = gzip.open(path, "rt", encoding="utf-8").read()
    strings = _strings(p, set())
    # only structural keys, spec vocab, system names and catalog ids may appear as strings
    allowed = {"format", "edition", "kit_commit", "slice", "weights", "benchmarks", "spec", "n_clusters", "width", "systems", "shape", "rows", "cols", "values",
               "official_reference", "balanced_skill", "per_benchmark", "raw", "skill", "kind", "parts", "type", "K", "base", "metric", "L", "chance", "f1",
               "track", "display", "added", "pergroup", "cf1", "macro", "macro_f1", "ratio", "brier_vs_0.25", "dev", packet.FORMAT, "a", "b"}
    allowed |= {str(n) for n in range(100)} | {p["edition"], p["kit_commit"]}
    assert strings <= allowed, strings - allowed
    for r in rows[:: max(1, len(rows) // 500)]:
        e = r["_evaluation"]
        for ident in (e["run_id"], str(e["group_id"]), e["payload_sha256"]):
            if len(ident) >= 8:
                assert ident not in text
        for q in r["questions"].values():
            for v in list(q.get("criteria", {}).values())[:3]:
                if isinstance(v, str) and len(v) >= 12:
                    assert v not in text
    assert not ({"seed", "permutation", "order"} & strings)


def test_packet_shuffles_clusters(built):
    rows, systems, p, _ = built
    dec = score.decompose(rows, systems["a"])
    shuffled = 0
    for n, b in dec.items():
        if b.S.shape[0] < 20:
            continue
        S_pk = packet._dense(p["benchmarks"][str(n)]["systems"]["a"])
        assert np.allclose(S_pk.sum(axis=0).sum(), b.S.sum(axis=0).sum())
        shuffled += not np.array_equal(S_pk, b.S)
    assert shuffled >= 30

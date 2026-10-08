"""The pre-declared proxy600 screen (jobs/common/dev_proxy.py): pinned ids, construction rule, and its wiring into
fulldev_eval (rows "proxy600") and m2_gates (--index-proxy: comparator units restricted, identity untouched)."""
import json

import pytest

from jobs.common import dev_proxy as P
from jobs.common import dev_subset as D


def test_proxy600_is_pinned_and_a_subset_of_the_m1_subset():
    d = P.load_proxy("proxy600")
    base = D.load_subset()
    assert d["ids_sha256"] == P.PROXIES["proxy600"][1] == D.ids_sha256(d["run_ids"])
    assert set(d["run_ids"]) <= set(base["run_ids"]) and d["base_subset_ids_sha256"] == base["ids_sha256"]
    assert d["design"] == "weight" and d["seed"] == P.SEED and d["n_target"] == 600
    assert d["summary"]["benchmarks"] == base["summary"]["benchmarks"] == 38  # every Index benchmark kept
    assert len(d["run_ids"]) == len(set(d["run_ids"])) == 644


def test_tampered_proxy_is_refused(tmp_path):
    d = json.loads(P.PROXY600.read_text(encoding="utf-8"))
    d["run_ids"] = d["run_ids"][1:]
    f = tmp_path / "p.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit):
        P.load_proxy(str(f))  # ids no longer match ids_sha256
    d["ids_sha256"] = D.ids_sha256(d["run_ids"])
    d["run_ids"].append("99:not:a:subset:row")
    d["ids_sha256"] = D.ids_sha256(d["run_ids"])
    f.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit):
        P.load_proxy(str(f))  # not a subset of the m1 subset


def test_bench_quotas_floor_cap_and_total():
    rows = {1: 100, 2: 12, 3: 5, 4: 300}
    w = {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.1}
    for design in P.DESIGNS:
        q = P.bench_quotas(rows, w, 60, design)
        assert sum(q.values()) == 60
        assert all(min(P.min_per_bench(60), rows[n]) <= q[n] <= rows[n] for n in rows)
    q = P.bench_quotas(rows, w, 60, "weight")
    assert q[3] == 5 and q[2] == 12  # capped at the benchmark's rows; the excess re-spread
    with pytest.raises(ValueError):
        P.bench_quotas(rows, w, 60, "fit")


def test_build_is_whole_cases_and_seeded():
    cases = {(1, f"c{i}"): {"rows": [f"1:r{i}a", f"1:r{i}b"], "bin": "<128"} for i in range(20)}
    cases |= {(2, f"d{i}"): {"rows": [f"2:r{i}"], "bin": "128-511" if i % 2 else "<128"} for i in range(30)}
    w = {1: 0.5, 2: 0.5}
    a = P.build_proxy(cases, w, 20, "weight", "s1")
    assert a == P.build_proxy(cases, w, 20, "weight", "s1") != P.build_proxy(cases, w, 20, "weight", "s2")
    for v in cases.values():
        assert all(r in a for r in v["rows"]) or not any(r in a for r in v["rows"])


def test_fulldev_eval_rows_mode_proxy600():
    from jobs.m2_first_cell import fulldev_eval as F

    assert "proxy600" in F.ROW_MODES and F.selected_ids("proxy600") == set(P.load_proxy("proxy600")["run_ids"])


def test_m2_gates_restricts_comparator_units_to_the_proxy():
    from train import m2_gates as G

    od = {"rows": 3, "questions": {f"r{i}/q": {"gold": "a", "k": 2,
                                              "forward": {"choice": "a", "correct": True, "chose_presented_first": True},
                                              "reversed": {"choice": "b", "correct": False,
                                                           "chose_presented_first": True}} for i in range(3)}}
    units = {"index_results": {f"r{i}": {"status": "ok"} for i in range(3)}, "order_diag": od, "decision": {"x": 1}}
    ids = frozenset({"r0", "r2"})
    r = G.restrict_units(units, ids)
    assert set(r["index_results"]) == ids and set(r["order_diag"]["questions"]) == {"r0/q", "r2/q"}
    assert r["order_diag"]["rows"] == 2 and r["order_diag"]["summary"]["n_questions"] == 2 and r["decision"] == {"x": 1}
    assert set(units["index_results"]) == {"r0", "r1", "r2"}  # the cached comparator itself is untouched
    assert G.restrict_units(units, None) is units
    rows = [{"_evaluation": {"run_id": f"r{i}"}} for i in range(3)]
    assert [x["_evaluation"]["run_id"] for x in G.proxy_rows(rows, ids)] == ["r0", "r2"]
    assert G.proxy_rows(rows, None) is rows

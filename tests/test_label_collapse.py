"""jobs.common.label_collapse: the pre-declared non-gating collapse rule on synthetic rows (no kit needed)."""
from jobs.common.label_collapse import detect


def _rows(n, cid, qtype, golds):
    crit = {"true": "x", "false": "y"} if qtype == "noul" else {"A": "a", "B": "b"}
    return [{"_evaluation": {"run_id": f"{cid}:{i}", "catalog_id": cid},
             "questions": {"q": {"type": qtype, "criteria": crit}}, "expected": {"q": golds[i % len(golds)]}}
            for i in range(n)]


def _res(rows, fn, status="ok"):
    return {r["_evaluation"]["run_id"]: {"run_id": r["_evaluation"]["run_id"], "status": status,
                                         "response": {"answers": {"q": fn(i)}}} for i, r in enumerate(rows)}


def test_noul_collapse_flags_both_labels():
    rows = _rows(40, 59, "noul", [True, False])
    w = detect(rows, _res(rows, lambda i: {"noul": 0.1}))
    assert w["label_collapse_count"] == 2 and w["label_collapse_benchmarks"] == [59] and w["gating"] is False
    t = next(e for e in w["label_collapse"] if e["label"] == "true")
    assert t["gold_share"] == 0.5 and t["pred_share"] == 0.0 and t["n"] == 40 and t["direction"] == "never_predicted"


def test_near_collapse_and_healthy():
    rows = _rows(100, 12, "choice", ["A", "B"])
    w = detect(rows, _res(rows, lambda i: {"choice": "A" if i else "B"}))  # B predicted 1%
    assert w["label_collapse_count"] == 0 and w["label_near_collapse_count"] == 2
    w = detect(rows, _res(rows, lambda i: {"choice": "AB"[i % 2]}))
    assert w["label_collapse_count"] == 0 and w["label_near_collapse_count"] == 0 and w["groups_checked"] == 1


def test_min_n_gold_band_and_refusals():
    rows = _rows(19, 40, "choice", ["A", "B"])
    assert detect(rows, _res(rows, lambda i: {"choice": "A"}))["label_collapse_count"] == 0  # < 20 questions
    rows = _rows(100, 40, "choice", ["A"] * 19 + ["B"])  # gold B = 5%: outside [10%, 90%]
    assert detect(rows, _res(rows, lambda i: {"choice": "A"}))["label_collapse_count"] == 0
    rows = _rows(40, 40, "choice", ["A", "B"])
    assert detect(rows, _res(rows, lambda i: {"choice": "A"}, status="refused"))["groups_checked"] == 0


def test_fulldev_report_wiring(tmp_path, monkeypatch):
    import json

    from harness import cli
    from jobs.m2_first_cell import fulldev_eval as F

    rows = _rows(30, 59, "noul", [True, False])
    for arm, v in (("child", 0.1), ("parent", 0.9)):
        (tmp_path / f"{arm}.results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _res(rows, lambda i: {"noul": v}).values()) + "\n\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_slice", lambda *a, **k: rows)
    w = F.collapse_warnings(tmp_path, ["child", "parent"], None)
    assert w["label_collapse_count"] == 2 and w["parent"]["label_collapse_count"] == 2
    assert {e["direction"] for e in w["label_collapse"] if e["label"] == "true"} == {"never_predicted"}

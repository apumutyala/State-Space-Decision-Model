import json
from pathlib import Path

import pytest

from jevkit import items as IT
from jevkit import scoring as SC
from jevkit.cli import main

from ._fixtures import good_answers, mixed_set

FIX = Path(__file__).parent / "fixtures" / "board_v161_excerpt.json"


def _setup(tmp_path):
    its = mixed_set("public") + mixed_set("private")
    ip = tmp_path / "items.jsonl"
    IT.write_items(its, ip)
    ans = good_answers(its)
    pp = tmp_path / "pred.jsonl"
    pp.write_text("".join(json.dumps({"id": i.id, "answer": ans[i.id], "status": "ok", "latency_s": 0.05 + 0.001 * n,
                                      "usage": {"input_tokens": 300 + n, "output_tokens": 1}}) + "\n"
                          for n, i in enumerate(its)), encoding="utf-8")
    return its, ip, pp


def test_score_report_all_axes(tmp_path):
    its, ip, pp = _setup(tmp_path)
    out = tmp_path / "rep.json"
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"ids": [its[0].id]}), encoding="utf-8")
    assert main(["score", "--items", str(ip), "--pred", str(pp), "--out", str(out), "--B", "20",
                 "--price", "kind=size_class_estimate,in=0.1,out=0.1,source=test size class,as_of=2026-10-09",
                 "--exclude-from-cost", str(ex)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    for k in ("intelligence", "calibration", "speed", "cost"):
        assert rep["axes"][k] is not None, k
    assert set(rep["scores"]) == {"A", "B", "C"} and rep["capability"] is not None
    assert rep["ci95"]["B"] == 20 and rep["ci95"]["composite"]["A"][0] <= rep["ci95"]["composite"]["A"][1]
    assert rep["speed"]["source"].startswith("pred latency_s on open standard+judge")
    assert rep["cost"]["n_excluded"] == 1 and rep["cost"]["price"]["kind"] == "size_class_estimate"
    assert rep["inputs"]["n_items_without_pred"] == 0 and rep["intelligence"]["basis"] == "open+sealed"
    a = rep["axes"]
    assert rep["scores"]["A"] == pytest.approx(SC.composite(a["intelligence"], a["calibration"], a["speed"], a["cost"]))


def test_score_without_price_has_null_cost_and_composite(tmp_path):
    _, ip, pp = _setup(tmp_path)
    out = tmp_path / "rep.json"
    assert main(["score", "--items", str(ip), "--pred", str(pp), "--out", str(out), "--B", "0", "--api"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["axes"]["cost"] is None and rep["scores"]["A"] is None and rep["capability"] is not None
    assert rep["speed"]["adjustment"] == "none (API)"
    with pytest.raises(SystemExit):
        main(["score", "--items", str(ip), "--pred", str(pp), "--usd-per-1000", "0.01"])  # price source required


def test_board_command_exit_code(tmp_path):
    out = tmp_path / "b.json"
    assert main(["board", "--api", str(FIX), "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["checks"]["composite_A"]["exact"] == rep["checks"]["composite_A"]["rows"] == 11
    bad = json.loads(FIX.read_text(encoding="utf-8"))
    bad["systems"][0]["scores"]["A"] += 0.01
    bp = tmp_path / "bad.json"
    bp.write_text(json.dumps(bad), encoding="utf-8")
    assert main(["board", "--api", str(bp)]) == 1


def test_guarantees_conformal_cli(tmp_path):
    _, ip, pp = _setup(tmp_path)
    out = tmp_path / "g.json"
    assert main(["guarantees", "conformal", "--items", str(ip), "--pred", str(pp), "--out", str(out),
                 "--alpha", "0.2", "--epsilon", "0.3", "--delta", "0.2"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["split"]["n_cal"] + rep["split"]["n_test"] == 72 and "coverage" in rep["sets"]["all"]

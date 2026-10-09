"""jobs/clef_baseline JOB=V (JOB_SPEC 14): baselines.clef_heldout_v1 on CPU. The per-arm summary and the paired Holm
compare run on the real 7,920 pinned units scored by the keyword stand-in decider (safety_eval.standin, not a model;
numbers mean nothing); the Clef scoring path runs on a mini cohort with the mocked Clef when torch is present.
Plumbing, pins, metric definitions (same as the D1-A battery: metrics.compare_arms / summarise) and fail-closed exits."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from baselines import clef_heldout_v1 as C
from safety_eval import heldout_v1 as HV
from safety_eval import metrics as M
from safety_eval.standin import KeywordDecider

N_BOOT = 200


@pytest.fixture(scope="module")
def variants():
    return C.expected_variants()


@pytest.fixture(scope="module")
def arms(variants):
    """Three stand-in arms of different gullibility: champion (robust), 'clef-flash' (gullible), 'clef' (between)."""
    return {name: M.run(KeywordDecider(gullibility=g), variants)
            for name, g in (("champion", 0.0), ("clef-flash", 6.0), ("clef", 2.0))}


def test_summary_per_family_with_template_ids(arms, variants):
    s = C.summarize_arm(arms["clef-flash"], variants, N_BOOT)
    assert s["complete"] and set(s["families"]) == set(HV.FAMILIES)
    t = HV.load_templates()
    for f, goal in (("heldout_hijack", "hijack"), ("heldout_steer", "steer"), ("heldout_abstain", "steer_abstain")):
        names = sorted(n for n, x in t.items() if x["goal"] == goal)
        assert s["families"][f]["templates"] == names and sorted(s["per_template"][f]) == names
        for k in ("hijack_rate", "steer_rate", "attacked_acc"):
            ci = s["families"][f][k]
            assert {"point", "lo", "hi"} <= set(ci) and ci["n_boot"] == N_BOOT
    # the metric definitions are metrics.summarise's: recompute the pooled hijack rate by hand
    rows = [r for r in arms["clef-flash"] if r["attack_family"] == "heldout_hijack"]
    assert abs(s["families"]["heldout_hijack"]["hijack_rate"]["point"] - np.mean([bool(r["hijack"]) for r in rows])) < 1e-12
    assert set(s["per_axis"]["heldout_hijack"]) == {"style", "position", "transform"}


def test_compare_holm_and_claim_rule(arms):
    out = C.compare(arms["champion"], {"clef-flash": arms["clef-flash"], "clef": arms["clef"]}, N_BOOT)
    assert out["complete"] and out["units"]["n_units"] == 7920
    prim = out["primary"]
    assert len(prim) == 4 and sorted(c["holm_rank"] for c in prim) == [1, 2, 3, 4]
    assert {(c["release"], c["family"], c["metric"]) for c in prim} == {
        (r, f, k) for r in ("clef-flash", "clef") for f, k in C.PRIMARY}
    by_rank = sorted(prim, key=lambda c: c["holm_rank"])
    assert [round(c["holm_level"], 6) for c in by_rank] == [round(1 - 0.05 / k, 6) for k in (4, 3, 2, 1)]
    for a, b in zip(by_rank, by_rank[1:]):  # step-down: a later rejection needs every earlier one
        assert a["p_boot"] <= b["p_boot"] or np.isnan(b["p_boot"])
        assert b["p_holm"] >= a["p_holm"]
        assert not (b["rejects"] and not a["rejects"])
    for rel in ("clef-flash", "clef"):
        mine = [c for c in prim if c["release"] == rel]
        assert out["claims"][rel]["champion_more_robust"] == all(c["rejects"] and c["diff_point"] > 0 for c in mine)
        # the primary points are compare_arms' (both | two_way) points
        for c in mine:
            ref = out["paired_vs_champion"][rel][c["family"]]["both|two_way"][c["metric"]]["diff"]["point"]
            assert c["diff_point"] == pytest.approx(ref, abs=1e-12)
    # S07 is reported, never pooled into the primary steer contrast
    assert "heldout_abstain" in out["paired_vs_champion"]["clef"] and all(c["family"] != "heldout_abstain" for c in prim)
    # the gullible stand-in is hijacked more than the robust one
    assert out["headline"]["clef-flash"]["heldout_hijack"]["hijack_rate"]["diff"]["point"] > 0


def test_holm_on_known_draws():
    rng = np.random.default_rng(0)
    cs = [{"draws": rng.normal(mu, 0.01, 2000)} for mu in (0.10, 0.0, 0.05, -0.2)]
    out = C.holm(cs)
    assert [c["rejects"] for c in out] == [True, False, True, True]
    assert all("draws" not in c and c["n_draws"] == 2000 for c in out)
    # step-down: once the zero-mean contrast fails, nothing after it (in p order) may reject
    assert sorted((c["holm_rank"], c["rejects"]) for c in out)[-1] == (4, False)


def test_unit_set_fails_closed(arms, variants):
    rows = [dict(r) for r in arms["clef"]]
    rows[5]["unit_sha"] = "0" * 64
    with pytest.raises(C.UnitSetMismatch, match="unit_sha"):
        C.check_unit_set(rows, variants, "clef")
    with pytest.raises(C.UnitSetMismatch, match="missing"):
        C.check_unit_set(arms["clef"][:-1], variants, "clef")
    bad = [dict(v) for v in variants]
    bad[3] = dict(bad[3], request_id=bad[3]["request_id"] + "x")
    with pytest.raises(C.UnitSetMismatch):
        C.check_production_variants(bad)


def _write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")


def test_cli_summarize_and_compare_exits(tmp_path, arms):
    for rel in ("clef-flash", "clef"):
        _write_jsonl(tmp_path / f"units-{rel}.jsonl", arms[rel])
    (tmp_path / "champ.json").write_text(json.dumps({"steer_rows": arms["champion"]}, default=str), encoding="utf-8")
    assert C.main(["summarize", "--units", str(tmp_path / "units-clef.jsonl"), "--out", str(tmp_path / "s.json"),
                   "--n-boot", "50"]) == 0
    s = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert s["schema"] == "ssdm.safety.heldout_v1.arm.v1" and s["units"]["units_sha256"] == C.UNITS_SHA256
    assert C.main(["compare", "--champion", str(tmp_path / "champ.json"), "--units",
                   str(tmp_path / "units-clef-flash.jsonl"), str(tmp_path / "units-clef.jsonl"),
                   "--out", str(tmp_path / "c.json"), "--n-boot", "50"]) == 0
    assert set(json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["claims"]) == {"clef-flash", "clef"}
    # a dropped champion row: exit 21, no output written
    (tmp_path / "champ2.json").write_text(json.dumps(arms["champion"][1:], default=str), encoding="utf-8")
    assert C.main(["compare", "--champion", str(tmp_path / "champ2.json"), "--units",
                   str(tmp_path / "units-clef.jsonl"), "--out", str(tmp_path / "c2.json"), "--n-boot", "50"]) == 21
    assert not (tmp_path / "c2.json").exists()
    assert C.main(["compare", "--champion", str(tmp_path / "champ.json"), "--units",
                   str(tmp_path / "units-other.jsonl"), "--out", str(tmp_path / "c3.json")]) == 2


def test_precheck_writes_receipt(tmp_path):
    assert C.main(["precheck", "--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "heldout-precheck.json").read_text(encoding="utf-8"))
    assert r["template_set_sha256"] == HV.TEMPLATE_SET_SHA256 and r["n_units"] == 7920 and r["disjointness"]["ok"]


def test_params_and_registration():
    from jobs.common import costs, launch

    for rel in ("clef_flash", "clef"):
        rel_path = f"jobs/clef_baseline/job_params_heldout_v1_{rel}.json"
        p = json.loads((Path(__file__).resolve().parents[2] / rel_path).read_text(encoding="utf-8"))
        assert p["mode"] == f"heldout_v1_{rel}" and p["env"] == {"RELEASE": rel.replace("_", "-"), "JOB": "V"}
        assert p["submit"]["env"] == p["env"] and p["heldout"]["template_set_sha256"] == HV.TEMPLATE_SET_SHA256
        assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
        assert costs.worst_case_usd(p) <= p["hard_cap_usd"] and rel_path in costs.PARAMS.values()
        assert ("clef_baseline", f"heldout_v1_{rel}") in launch.CONTEXT_RECIPES


# ----------------------------------------------------------------------------- Clef scoring path (mocked Clef)
@pytest.fixture(scope="module")
def mini_cohort(tmp_path_factory):
    from train import cohort as Co

    d = tmp_path_factory.mktemp("hv")
    steer = Co.steer_requests(6, 2)
    header = {"cohort": "test-mini-heldout", "sizes": {"dev": 0, "calibration": 0, "retention": 0, "steer_dev": 6,
                                                       "steer_gate": 2}, "sizing": None}
    lines = [json.dumps({"header": header}, sort_keys=True)]
    for role in ("steer_dev", "steer_gate"):
        for e in steer[role]:
            lines.append(json.dumps({"role": role, **{k: e[k] for k in sorted(e) if k != "family"}}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    p = d / "cohort.jsonl.gz"
    with gzip.GzipFile(p, "wb", mtime=0) as f:
        f.write(text.encode())
    return str(p), hashlib.sha256(text.encode()).hexdigest()


def test_score_on_mocked_clef(tmp_path, monkeypatch, mini_cohort):
    pytest.importorskip("torch")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baselines import clef_backend as CB
    from clef_mocks import FakeTok, MockProcessor, build_mock_model

    jsm = CB.load_jsm()
    model = build_mock_model(jsm, seed=4)

    def fake_load(model_dir, release="clef-flash", *, encoding="ssdm", device="cpu", dtype="float32", **kw):
        be = CB.ClefBackend(model, FakeTok(), release=release, encoding=encoding, jsm=jsm, device="cpu",
                            dtype="float32", files_sha256={"note": "mock"})
        return be, model, MockProcessor(FakeTok())

    monkeypatch.setattr(CB, "load_clef", fake_load)
    monkeypatch.setenv("CLEF_DRY_RUN", "1")
    ident = C.score("unused", "clef", tmp_path, device="cpu", dtype="float32", verify=False,
                    cohort_path=mini_cohort[0], cohort_sha256=mini_cohort[1], log=lambda m: None)
    rows = C.read_rows(tmp_path / "units-clef.jsonl")
    v = C.expected_variants(*mini_cohort)
    C.check_unit_set(rows, v, "clef")
    assert ident["variants"]["n_units"] == len(v) == len(rows) and ident["gdn_path"] == "fast"
    assert {r["attack_family"] for r in rows} == {"clean", *HV.FAMILIES}

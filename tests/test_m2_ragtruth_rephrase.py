"""B-PREREG-RAGTRUTH-REPHRASE: the RT1 readout rewrite (decide.ragtruth_rephrase) and the train-split mode
(jobs.m2_first_cell.ragtruth_rephrase). CPU only, no model."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from decide import ragtruth_rephrase as RR
from jobs.m2_first_cell import ragtruth_polarity as RP
from jobs.m2_first_cell import ragtruth_rephrase as M

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "jobs/m2_first_cell/job_params_ragtruth_rephrase.json"


def test_rule_constants_pinned():
    assert M.NULL_AUROC == 0.5
    assert M.CANDIDATE == "rt1"
    assert M.WORDINGS == ("original", "rt1")
    assert M.ARM_NAMES == ("d13", "d19")
    assert M.INDRAW_GUARD_ARMS == ("d19",)
    assert M.SLOW_BOX == 1.25
    assert M.SCHEMA == "ssdm.m2.ragtruth_rephrase.v1"
    assert M.INDRAW_FILE == "jobs/m2_first_cell/ragtruth_rephrase_d19_indraw.json"
    assert "0683fa7c" in M.PREREG
    assert "lower bound" in M.DECISION_RULE and "outside its training draw" in M.DECISION_RULE
    assert (M.N_BOOT, M.BOOT_SEED) == (2000, 20261009)


def test_original_is_the_index_question():
    assert RR.ORIGINAL == RP.QUESTION["q"]


def test_rt1_wording_is_the_registered_text():
    q = RR.WORDINGS["rt1"]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert q["instructions"].startswith("Hallucination check. The prompt contains the source material")
    assert q["criteria"]["true"].endswith("(hallucinated).") and q["criteria"]["false"].endswith("(faithful).")
    assert "not supported" not in json.dumps(q)
    assert RR.wording_sha256("rt1") == RR.wording_sha256("rt1") != RR.wording_sha256("original")


def _req(q=None, extra=None):
    qs = {"q": json.loads(json.dumps(q if q is not None else RP.QUESTION["q"]))}
    if extra:
        qs.update(extra)
    return {"model": "m", "state": {"prompt": "P", "response": "R"}, "questions": qs}


def test_rewrite_only_exact_matches():
    other = {"type": "noul", "instructions": "Is it sunny?", "criteria": {"true": "yes", "false": "no"}}
    req = _req(extra={"z": other})
    new, hit = RR.rephrase_request(req, "rt1")
    assert hit == ["q"]
    assert new["questions"]["q"] == RR.WORDINGS["rt1"]
    assert new["questions"]["z"] == other and new["state"] == req["state"]
    assert req["questions"]["q"] == RP.QUESTION["q"]  # input untouched
    near = dict(RP.QUESTION["q"], instructions=RP.QUESTION["q"]["instructions"] + " ")
    assert RR.rephrase_request(_req(near), "rt1")[1] == []
    assert RR.rephrase_request(_req(dict(RP.QUESTION["q"], type="choice")), "rt1")[1] == []
    same, none = RR.rephrase_request(req, "original")
    assert same is req and none == []
    with pytest.raises(ValueError):
        RR.rephrase_request(req, "rt2")


def test_wrapper_passes_answers_through_and_records():
    seen = {}

    def decide(r):
        seen["q"] = r["questions"]["q"]
        return {"answers": {"q": {"type": "noul", "noul": 0.7}}, "ssdm": {"status": "ok"}}

    out = RR.rephrased(decide, "rt1")(_req())
    assert seen["q"] == RR.WORDINGS["rt1"]
    assert out["answers"]["q"]["noul"] == 0.7
    assert out["ssdm"]["ragtruth_rephrase"]["questions"] == ["q"]
    assert RR.rephrased(decide, "original") is decide


def test_decide_arm():
    ok = {"lo": 0.51}
    assert M.decide_arm("d13", ok, None).startswith("pass")
    assert M.decide_arm("d13", {"lo": 0.5}, None).startswith("fail")
    assert M.decide_arm("d13", {"lo": float("nan")}, None).startswith("fail")
    assert M.decide_arm("d19", ok, 0.55).startswith("pass")
    assert "in-draw" in M.decide_arm("d19", ok, 0.5)
    assert "in-draw" in M.decide_arm("d19", ok, None)


def test_paired_bootstrap_identity_and_sign():
    lab = [i % 2 == 0 for i in range(40)]
    a = [0.5 + (0.1 if y else -0.1) * (i % 3) for i, y in enumerate(lab)]
    b = [1 - x for x in a]
    z = M.paired_bootstrap(a, a, lab, b=200, seed=1)
    assert z["point"] == 0 and z["lo"] == 0 and z["hi"] == 0
    d = M.paired_bootstrap(b, a, lab, b=200, seed=1)
    assert d["point"] > 0 and d["lo"] > 0
    assert math.isclose(d["point"], RP.auroc(a, lab) - RP.auroc(b, lab))


def test_params_schedule_and_estimate():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 79
    assert round(p["max_duration_minutes"] * 0.063, 2) <= 5.0  # the platform hold
    assert p["eval_only"] == "ragtruth_rephrase" and p["mode"] == "ragtruth_rephrase"
    assert p["arms"]["d13"]["wordings"] == ["rt1"] and p["arms"]["d19"]["wordings"] == ["original", "rt1"]
    est = M.estimate(p)
    assert est["ok"], est


def test_d19_params_schedule_and_estimate():
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_ragtruth_rephrase_d19.json").read_text(encoding="utf-8"))
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 62
    assert list(p["arms"]) == ["d19"] and p["arms"]["d19"]["block"] == "checkpoint"
    assert p["checkpoint"]["ckpt_sha256"].startswith("31062f48") and p["mode"] == "ragtruth_rephrase_d19"
    assert (ROOT / p["cell_config"]).read_bytes() == (
        ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_fulldev_eval_d19_v3w_s1of2.json").read_bytes()
    assert M.estimate(p)["ok"]


def test_indraw_file_identity():
    rec = json.loads((ROOT / M.INDRAW_FILE).read_text(encoding="utf-8"))
    assert rec["n"] == len(rec["ids"]) == 175 and rec["n_items"] == 300
    assert RP.ids_sha256(rec["ids"]) == rec["ids_sha256"]
    sel = json.loads((ROOT / "jobs/m2_first_cell/ragtruth_polarity_selection.json").read_text(encoding="utf-8"))
    assert rec["items_ids_sha256"] == sel["ids_sha256"] and rec["items_sha256"] == sel["items_sha256"]
    assert set(rec["ids"]) <= set(sel["ids"])

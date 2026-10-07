"""jobs/q9_feature_cache params, one file per PART (one job = one <= 4.5 GiB output artifact; review B1): caps >= 1.3x
measured-rate projections of the committed exact volume, schedule, worst case, output-size bound, pins, registration.
CPU, no data, < 1 s."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jobs.common import costs
from jobs.q9_feature_cache import preflight as PF

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "jobs" / "q9_feature_cache"
PARTS = ("stock_dev", "stock_train", "conv1_dev", "conv1_train", "e5_dev", "e5_train")
#: the platform's max_artifact_bytes (list_limits; "cannot be raised past it today"); one job's output is ONE tar
MAX_ARTIFACT_GIB = 5.0


def _p(part):
    return json.loads((J / f"job_params_{part}.json").read_text(encoding="utf-8"))


def _vol():
    return json.loads((J / "volume_plan.json").read_text(encoding="utf-8"))


def test_parts_are_exactly_these_files():
    assert sorted(p.name for p in J.glob("job_params_*.json")) == sorted(f"job_params_{p}.json" for p in PARTS)


@pytest.mark.parametrize("part", PARTS)
def test_preflight_passes_on_committed_params_and_volume(part):
    res = PF.check(_p(part), _vol())
    assert res["ok"], res["checks"]
    for arm in _p(part)["arms"]:
        assert res["checks"][f"arm_{arm}"]["margin_frac"] >= 0.3
        assert res["checks"][f"output_{arm}"]["ok"]


@pytest.mark.parametrize("part", PARTS)
def test_output_of_every_part_fits_one_artifact(part):
    p, v = _p(part), _vol()
    assert len(p["arms"]) == 1 and p["max_output_gib"] == 4.5 < MAX_ARTIFACT_GIB
    gib = PF.part_gib(v["arms"][p["arms"][0]]["est_gib"], p["splits"])
    assert gib <= p["max_output_gib"], (part, gib)
    assert abs(gib - p["est_output_gib"]) < 1e-3
    cfg = json.loads((J / "cache_config.json").read_text(encoding="utf-8"))
    assert cfg["max_output_gib"] == 4.5


def test_parts_cover_every_arm_and_split_once():
    cover = sorted((_p(x)["arms"][0], s) for x in PARTS for s in _p(x)["splits"])
    want = sorted((a, s) for a in ("stock", "conv1", "e5") for s in ("select", "holdout", "train"))
    assert cover == want


@pytest.mark.parametrize("part", PARTS)
def test_schedule_worst_case_and_spec(part):
    p = _p(part)
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert p["build_timeout_minutes"] >= 60 and p["max_restarts"] == 0
    w = costs.worst_case_usd(p)
    assert w * 1.2 <= p["hard_cap_usd"], (w, p["hard_cap_usd"])  # >= 20% margin (review: extra's was 1.03x)
    if p["arms"][0] in ("conv1", "e5"):
        assert p["hard_cap_usd"] >= 30
    assert f"${w:.2f}" in (J / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert costs.PARAMS[f"q9_feature_cache_{part}"] == f"jobs/q9_feature_cache/job_params_{part}.json"
    assert f"jobs/q9_feature_cache/job_params_{part}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert p["measured"]["stock_prefill_tok_s_min"] == 22144 and p["measured"]["call_s_fitted_5cmat"] == 0.0595
    fit = (p["measured"]["readout_I_stock_wall_s"] - p["measured"]["readout_fork_positions"]
           / p["measured"]["stock_prefill_tok_s_min"]) / p["measured"]["readout_fork_calls"]
    assert abs(fit - p["measured"]["call_s_fitted_5cmat"]) < 5e-4
    assert set(p["phase_caps_minutes"]) == {"setup", "guard", "heldout", "report", f"arm_{p['arms'][0]}"}
    assert p["submit"]["env"] == {"Q9C_MODE": part}
    pins = {x["dest"]: x["pin"] for x in p["context"]["data"]}
    assert pins["slices/dev-rows.jsonl.gz"] == "jobs/q9_feature_cache/data.sha256"
    assert ("m2-150k/records.jsonl.gz" in pins) == ("train" in p["splits"])
    assert ("gdn/gdn_converted.pt" in pins) == (p["arms"][0] == "conv1")


def test_pins_and_runner_hygiene():
    dev = (J / "data.sha256").read_text(encoding="utf-8").split()
    assert dev[0] == (ROOT / "jobs/clef_baseline/data.sha256").read_text(encoding="utf-8").split()[0]
    assert dev[1] == "slices/dev-rows.jsonl.gz" and len(dev) == 2
    tr = (J / "data_train.sha256").read_text(encoding="utf-8").split()
    assert tr[1::2] == ["m2-150k/records.jsonl.gz", "m2-150k/sample-150000-seed20261006.manifest.jsonl.gz",
                        "m2-150k/sample-150000-seed20261006.content.jsonl.gz", "suite-0.2/selected-rows.jsonl.gz",
                        "suite-0.2/added-rows.jsonl.gz", "suite-0.2/excluded-questions.json"]
    gdn = (J / "gdn_converted.sha256").read_text(encoding="utf-8").split()[0]
    cfg = json.loads((J / "cache_config.json").read_text(encoding="utf-8"))
    assert gdn == cfg["arms"]["conv1"]["gdn_sha256"] == "8a50c36ca010e54000d246f76f4ba93c4c0948fd2f4e3d0d35f4fcb06f0f0724"
    assert cfg["arms"]["conv1"]["layers"] == [23]
    for f in ("data.sha256", "data_train.sha256", "gdn_converted.sha256", "cache_job.sh", "build_context.sh"):
        assert b"\r" not in (J / f).read_bytes(), f
    bad = re.compile(r"[\\]n[ \t]+--")  # == jobs/common/lint.py BACKSLASH_N (L-10)
    assert bad.search("x " + chr(92) + "n  --out") and not bad.search("x --out")
    for f in ("cache_job.sh", "build_context.sh"):
        assert not bad.search((J / f).read_text(encoding="utf-8")), f
    sh = (J / "cache_job.sh").read_text(encoding="utf-8")
    assert "jobs.common.rate_guard run-weighted" in sh and "jobs.q9_feature_cache.preflight" in sh
    assert "--max-gib" in sh and "--splits" in sh
    assert cfg["cohort_sha256"] == "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
    assert cfg["train_set"]["pointer"] == "data/manifests/sample-150000-seed20261006.pointer.json"
    assert cfg["max_k"] == 16 and cfg["max_t"] == 2048 and cfg["template"] == "jrt-chatml-v1"


def test_volume_plan_matches_the_rule_splits():
    v = _vol()
    cfg = json.loads((J / "cache_config.json").read_text(encoding="utf-8"))
    for arm, x in v["arms"].items():
        assert x["select"]["requests"] == 2008 and x["holdout"]["requests"] == 2014
        assert x["train"]["questions"] == cfg["arms"][arm]["train_rows"]
        assert x["frozen_family_check"].startswith("harness.heldout.check_record")

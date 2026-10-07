"""jobs/q9_feature_cache params: caps >= 1.3x measured-rate projections of the committed exact volume, schedule, worst
case, pins, registration. CPU, no data, < 1 s."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jobs.common import costs
from jobs.q9_feature_cache import preflight as PF

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "jobs" / "q9_feature_cache"
MODES = ("stock", "extra")


def _p(mode):
    return json.loads((J / f"job_params_{mode}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", MODES)
def test_preflight_passes_on_committed_params_and_volume(mode):
    res = PF.check(_p(mode), json.loads((J / "volume_plan.json").read_text(encoding="utf-8")))
    assert res["ok"], res["checks"]
    for arm in _p(mode)["arms"]:
        assert res["checks"][f"arm_{arm}"]["margin_frac"] >= 0.3


@pytest.mark.parametrize("mode", MODES)
def test_schedule_worst_case_and_spec(mode):
    p = _p(mode)
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert p["build_timeout_minutes"] >= 60 and p["max_restarts"] == 0
    w = costs.worst_case_usd(p)
    assert w <= p["hard_cap_usd"]
    assert f"${w:.2f}" in (J / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert costs.PARAMS[f"q9_feature_cache_{mode}"] == f"jobs/q9_feature_cache/job_params_{mode}.json"
    assert f"jobs/q9_feature_cache/job_params_{mode}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert p["measured"]["stock_prefill_tok_s_min"] == 22144 and p["measured"]["call_s_fitted_5cmat"] == 0.0595
    fit = (p["measured"]["readout_I_stock_wall_s"] - p["measured"]["readout_fork_positions"]
           / p["measured"]["stock_prefill_tok_s_min"]) / p["measured"]["readout_fork_calls"]
    assert abs(fit - p["measured"]["call_s_fitted_5cmat"]) < 5e-4
    assert set(p["arms"]) <= set(json.loads((J / "cache_config.json").read_text(encoding="utf-8"))["arms"])
    assert set(p["phase_caps_minutes"]) == {"setup", "guard", "heldout", "report"} | {f"arm_{a}" for a in p["arms"]}


def test_pins_and_runner_hygiene():
    dev = (J / "data.sha256").read_text(encoding="utf-8").split()[0]
    assert dev == (ROOT / "jobs/clef_baseline/data.sha256").read_text(encoding="utf-8").split()[0]
    gdn = (J / "gdn_converted.sha256").read_text(encoding="utf-8").split()[0]
    cfg = json.loads((J / "cache_config.json").read_text(encoding="utf-8"))
    assert gdn == cfg["arms"]["conv1"]["gdn_sha256"] == "8a50c36ca010e54000d246f76f4ba93c4c0948fd2f4e3d0d35f4fcb06f0f0724"
    assert cfg["arms"]["conv1"]["layers"] == [23]
    for f in ("data.sha256", "gdn_converted.sha256", "cache_job.sh", "build_context.sh"):
        assert b"\r" not in (J / f).read_bytes(), f
    bad = re.compile(chr(92) + r"n[ \t]+--")
    assert bad.search("x " + chr(92) + "n  --out") and not bad.search("x --out")
    for f in ("cache_job.sh", "build_context.sh"):
        assert not bad.search((J / f).read_text(encoding="utf-8")), f
    sh = (J / "cache_job.sh").read_text(encoding="utf-8")
    assert "jobs.common.rate_guard run-weighted" in sh and "jobs.q9_feature_cache.preflight" in sh
    # the train draw is the first cell's (cohort, pointer, pinned exclusions)
    assert cfg["cohort_sha256"] == "be6efe2bada29f436e9c06b9a92cf48ffffd3f7f392fd05cb6dab96ed9fdacf2"
    assert cfg["train_set"]["pointer"] == "data/manifests/sample-150000-seed20261006.pointer.json"
    assert cfg["max_k"] == 16 and cfg["max_t"] == 2048 and cfg["template"] == "jrt-chatml-v1"


def test_volume_plan_matches_the_rule_splits():
    v = json.loads((J / "volume_plan.json").read_text(encoding="utf-8"))
    for arm, x in v["arms"].items():
        assert x["select"]["requests"] == 2008 and x["holdout"]["requests"] == 2014
        assert x["train"]["questions"] == json.loads((J / "cache_config.json").read_text())["arms"][arm]["train_rows"]

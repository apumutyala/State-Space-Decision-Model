"""Mode ragtruth_polarity (jobs/m2_first_cell/ragtruth_polarity.py): the RAGTruth TRAIN item builder (Index catalog-59
format), the stratified one-per-source selection with DEV exclusions, the AUROC / bootstrap metric, the pre-registered
polarity rule, and the job params / selection / cell-config pins. CPU only, no model. The real-data checks (DEV format
reproduction, selection rebuild) run when the local ssdm-data files exist and skip otherwise."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from jobs.m2_first_cell import checkpoint_loader as L
from jobs.m2_first_cell import ragtruth_polarity as R

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
PARAMS = M / "job_params_ragtruth_polarity.json"
SEL = M / "ragtruth_polarity_selection.json"
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
SRC = DATA / "m2/sources/dl-2026-10-08/ragtruth"


# ------------------------------------------------------------------------------------------------ fixtures
def _src(sid, task="QA"):
    return {"source_id": str(sid), "task_type": task, "source": "x", "source_info": "doc",
            "prompt": f"Instruction:\nAnswer from passage {sid}.\nOutput:"}


def _resp(rid, sid, hall, split="train"):
    labels = [{"label_type": "Evident Conflict"}, {"label_type": "Evident Baseless Info"},
              {"label_type": "Evident Conflict"}] if hall else []
    return {"id": str(rid), "source_id": str(sid), "model": "m", "temperature": 0.7, "labels": labels,
            "split": split, "quality": "good", "response": f"response {rid}"}


def _corpus():
    tasks = ("QA", "Summary", "Data2txt")
    srcs = {str(s): _src(s, tasks[s % 3]) for s in range(60)}
    resp = {}
    rid = 0
    for s in range(60):
        for k in range(4):  # 4 responses per source, mixed labels
            resp[str(rid)] = _resp(rid, s, hall=(rid % 3 == 0), split="test" if s >= 50 else "train")
            rid += 1
    return resp, srcs


# ------------------------------------------------------------------------------------------------ item builder
def test_build_row_is_the_index_catalog59_shape():
    row = R.build_row(_resp(7, 3, True), _src(3))
    assert row["state"] == {"prompt": _src(3)["prompt"], "response": "response 7"}
    assert row["questions"] == R.QUESTION and row["questions"] is not R.QUESTION
    assert row["questions"]["q"]["type"] == "noul" and set(row["questions"]["q"]["criteria"]) == {"true", "false"}
    assert row["expected"] == {"q": True}
    assert row["id"] == "59:7" and row["benchmark"] == row["family"] == R.DATASET and row["split"] == "evaluation"
    assert row["metadata"] == {"source_id": "7", "group_id": "59:7", "track": "QA", "source_task_id": "3",
                               "responder_model": "m", "quality": "good",
                               "label_types": ["Evident Baseless Info", "Evident Conflict"]}
    assert row["scoring"] == {"type": "candidate_decisions", "primary_field": "q"}
    e = row["_evaluation"]
    assert e["run_id"] == "ragtruth-train:59:7" and e["catalog_id"] == R.CATALOG_ID == 59
    assert R.build_row(_resp(8, 3, False), _src(3))["expected"] == {"q": False}
    with pytest.raises(ValueError):
        R.build_row(_resp(7, 4, True), _src(3))


def test_dev_format_check_requires_byte_equal_rebuild():
    resp, srcs = _corpus()
    dev = [R.build_row(resp[i], srcs[resp[i]["source_id"]]) for i in ("200", "201")]
    for r in dev:
        r["_evaluation"]["run_id"] = "candidates-v3:" + r["id"]  # the kit's run ids / payload differ by design
        r["_evaluation"]["payload_sha256"] = "0" * 64
    other = {"_evaluation": {"catalog_id": 12, "run_id": "x"}}
    assert R.check_dev_format(dev + [other], resp, srcs) == {"dev_rows_checked": 2, "fields": list(R.FORMAT_KEYS),
                                                            "equal": True}
    dev[1]["state"]["prompt"] += " "
    with pytest.raises(ValueError, match="1 of 2"):
        R.check_dev_format(dev, resp, srcs)
    with pytest.raises(ValueError):
        R.check_dev_format([other], resp, srcs)


def test_dev_exclusions_by_source_and_guard_and_never_a_dev_response():
    resp, srcs = _corpus()
    resp["900"] = _resp(900, 50, True, split="train")  # a TRAIN response on a source that a DEV row uses
    dev = [R.build_row(resp["200"], srcs["50"])]
    ex = R.dev_exclusions(resp, dev, guard_fail_ids=["3", "999", "200"])
    assert ex["source_id_in_dev"] == ["900"] and ex["doc_overlap_guard_fail"] == ["3"]
    assert ex["all"] == ["3", "900"] and ex["n_excluded"] == 2 and ex["n_dev_sources"] == 1
    resp["200"]["split"] = "train"
    with pytest.raises(ValueError, match="DEV RAGTruth rows"):
        R.dev_exclusions(resp, dev, [])


def test_select_is_stratified_one_per_source_excluding_and_deterministic():
    resp, srcs = _corpus()
    s1 = R.select(resp, srcs, exclude={"0", "1"}, n=30, seed="t")
    s2 = R.select(resp, srcs, exclude={"0", "1"}, n=30, seed="t")
    assert s1 == s2 and s1["n"] == 30 and s1["ids_sha256"] == R.ids_sha256(s1["ids"])
    picked = [resp[i] for i in s1["ids"]]
    assert all(r["split"] == "train" for r in picked) and not {"0", "1"} & set(s1["ids"])
    assert len({r["source_id"] for r in picked}) == len(picked)  # one response per source document
    assert sum(v["picked"] for v in s1["strata"].values()) == 30
    for v in s1["strata"].values():  # proportional allocation over the one-per-source pool
        assert abs(v["picked"] - 30 * v["pool"] / s1["n_sources_eligible"]) < 1
    assert R.select(resp, srcs, set(), n=30, seed="other")["ids"] != s1["ids"]


def test_allocate_largest_remainder_and_caps():
    assert R.allocate({("a", 0): 10, ("b", 1): 10, ("c", 0): 10}, 10) == {("a", 0): 4, ("b", 1): 3, ("c", 0): 3}
    assert sum(R.allocate({("a", 0): 1, ("b", 0): 99}, 50).values()) == 50
    assert R.allocate({("a", 0): 2}, 5) == {("a", 0): 2}


# ------------------------------------------------------------------------------------------------ metric + rule
def test_auroc_known_values():
    y = [True, True, False, False]
    assert R.auroc([0.9, 0.8, 0.2, 0.1], y) == 1.0
    assert R.auroc([0.1, 0.2, 0.8, 0.9], y) == 0.0
    assert R.auroc([0.5] * 4, y) == 0.5
    assert R.auroc([0.9, 0.3, 0.5, 0.1], y) == 0.75  # pairs: (.9>.5,.9>.1,.3<.5,.3>.1) = 3/4
    assert R.auroc([0.7, 0.4, 0.4, 0.1], y) == 0.875  # one tie counts 1/2
    assert math.isnan(R.auroc([0.1, 0.2], [True, True]))


def test_bootstrap_ci_deterministic_and_flip_symmetric():
    import random

    rng = random.Random(1)
    y = [rng.random() < 0.45 for _ in range(300)]
    s = [min(1.0, max(0.0, (0.25 if t else 0.75) + rng.gauss(0, 0.25))) for t in y]  # inverted scorer
    a = R.bootstrap_auroc(s, y, b=400)
    assert a == R.bootstrap_auroc(s, y, b=400) and a["lo"] <= a["point"] <= a["hi"] and a["hi"] < 0.5
    f = R.bootstrap_auroc([1 - x for x in s], y, b=400)
    assert abs(f["point"] - (1 - a["point"])) < 1e-12 and f["lo"] > 0.5
    assert R.decide_polarity(a).startswith("inverted")
    assert R.decide_polarity(f) == "no change"


def test_rule_constants_pinned():
    assert R.NULL_AUROC == 0.5 and R.CI_LEVEL == 0.95 and R.N_BOOT == 2000 and R.BOOT_SEED == 20261009
    assert R.N_TARGET == 300 and R.SEED == "ragtruth-polarity-v1" and R.READOUT == "v3w" and R.CATALOG_ID == 59
    assert R.ARM_NAMES == ("d13", "d1")
    assert "upper bound" in R.DECISION_RULE and "< 0.5" in R.DECISION_RULE
    assert R.decide_polarity({"hi": 0.4999}) == "inverted: propose a statement rephrase"
    assert R.decide_polarity({"hi": 0.5}) == "no change"  # the CI touching 0.5 is "includes 0.5"
    assert R.decide_polarity({"hi": float("nan")}) == "no change"


def test_arm_metrics_reads_noul_p_true_and_counts_failures():
    resp, srcs = _corpus()
    items = [R.build_row(resp[str(i)], srcs[resp[str(i)]["source_id"]]) for i in range(12)]
    res = {}
    for it in items[:-1]:
        p = 0.2 if it["expected"]["q"] else 0.8  # perfectly inverted
        res[it["_evaluation"]["run_id"]] = {"run_id": it["_evaluation"]["run_id"], "status": "ok",
                                            "response": {"answers": {"q": {"type": "noul", "noul": p}}}}
    m = R.arm_metrics(items, res)
    assert m["status"] == {"ok": 11, "missing": 1} and m["n_scored"] == 11
    assert m["auroc"]["point"] == 0.0 and m["auroc_flipped"]["point"] == 1.0
    assert m["decision"].startswith("inverted") and m["per_item"][0]["p_hallucinated"] in (0.2, 0.8)
    assert m["per_item"][-1]["p_hallucinated"] is None


def test_valid_result_requires_both_arms_and_pinned_checkpoints():
    ok_arm = lambda sha: {"status": {"ok": 2}, "n_scored": 2, "checkpoint_loaded": {"sha256": sha},  # noqa: E731
                          "auroc": {"point": 0.2, "lo": 0.1, "hi": 0.3},
                          "auroc_flipped": {"point": 0.8, "lo": 0.7, "hi": 0.9}}
    d = {"schema": R.SCHEMA, "verdict": "measured", "n_items": 2, "arms": {"d13": ok_arm("a"), "d1": ok_arm("b")},
         "checkpoints": {"checkpoint": {"ckpt_sha256": "a"}, "checkpoint_d1": {"ckpt_sha256": "b"}}}
    assert R.valid_result(d)
    assert not R.valid_result({**d, "arms": {"d13": ok_arm("a")}})
    assert not R.valid_result({**d, "checkpoints": {"checkpoint": {"ckpt_sha256": "a"},
                                                    "checkpoint_d1": {"ckpt_sha256": "z"}}})


# ------------------------------------------------------------------------------------------------ params / pins
def test_params_pins_selection_cell_configs_and_checkpoint_blocks():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    s = json.loads(SEL.read_text(encoding="utf-8"))
    assert p["mode"] == "ragtruth_polarity" and p["eval_only"] == "ragtruth" and p["readout"] == R.READOUT
    assert p["ragtruth"]["items_sha256"] == s["items_sha256"] and p["ragtruth"]["ids_sha256"] == s["ids_sha256"]
    assert s["ids_sha256"] == R.ids_sha256(s["ids"]) and s["n"] == len(s["ids"]) == R.N_TARGET
    assert s["format_check"]["equal"] and s["format_check"]["dev_rows_checked"] == 270
    assert s["exclusions"]["n_source_id_in_dev"] == 0 and s["exclusions"]["n_excluded"] == 24
    # D13 arm = the fulldev_eval_d13_v3w cell config byte for byte + its checkpoint block; D1 = fulldev_eval's
    assert (M / "m2_first_cell_rwkv7_ragtruth_polarity.json").read_bytes() == \
        (M / "m2_first_cell_rwkv7_fulldev_eval_d13_v3w.json").read_bytes()
    d13 = json.loads((M / "job_params_fulldev_eval_d13_v3w.json").read_text(encoding="utf-8"))
    d1 = json.loads((M / "job_params_fulldev_eval.json").read_text(encoding="utf-8"))
    assert p["checkpoint"] == d13["checkpoint"] and p["checkpoint"]["url_env"] == "CKPT_URL"
    assert {k: v for k, v in p["checkpoint_d1"].items() if k not in ("url_env", "_note")} == \
        {k: v for k, v in d1["checkpoint"].items() if k not in ("url_env", "_note")}
    assert p["checkpoint_d1"]["url_env"] == "CKPT_URL_D1"
    assert L.checkpoint_block(p)["artifact"] == "art-cr9a2"
    assert L.checkpoint_block(p, "checkpoint_d1")["artifact"] == "art-bm3e6"
    with pytest.raises(L.CheckpointError):
        L.checkpoint_block(p, "checkpoint_d2")
    assert p["cell_config_d1"] == "jobs/m2_first_cell/m2_first_cell_rwkv7_fulldev_eval.json"
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert "ACELL=jobs/m2_first_cell/m2_first_cell_rwkv7_fulldev_eval.json" in sh
    assert p["submit"]["env"]["CKPT_URL"].startswith("<") and p["submit"]["env"]["CKPT_URL_D1"].startswith("<")
    dests = {e["dest"]: e for e in p["context"]["data"]}
    assert dests["ragtruth/items.jsonl.gz"]["pin_json"][1:] == ["ragtruth", "items_sha256"]
    assert dests["checkpoint_d1/cell.json"]["pin_json"][1:] == ["checkpoint_d1", "cell_json_sha256"]
    assert dests["checkpoint/cell.json"]["pin_json"][0] == "jobs/m2_first_cell/job_params_ragtruth_polarity.json"
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    est = R.estimate(p, s)
    assert est["ok"] and set(est["minutes"]) == {"score_d13", "score_d1"}


def test_readme_preregisters_the_rule():
    t = (M / "RAGTRUTH_POLARITY.md").read_text(encoding="utf-8")
    assert "Pre-registered decision rule" in t and "upper" in t and "below 0.5" in t and "No change" in t
    s = json.loads(SEL.read_text(encoding="utf-8"))
    assert s["ids_sha256"] in t and s["items_sha256"] in t


# ------------------------------------------------------------------------------------------------ real data (local)
needs_data = pytest.mark.skipif(not (SRC / "response.jsonl").is_file() or not (DATA / "slices/dev-rows.jsonl.gz").is_file()
                                or not (DATA / "m2/ragtruth-polarity/guard-scan.json").is_file(),
                                reason="local ssdm-data RAGTruth sources / DEV rows / guard scan not present")


@needs_data
def test_real_dev_rows_rebuild_and_selection_reproduces():
    resp, srcs = R.load_source(SRC)
    dev = R.read_jsonl(DATA / "slices/dev-rows.jsonl.gz")
    assert R.check_dev_format(dev, resp, srcs)["dev_rows_checked"] == 270
    for r in R.dev_ragtruth_rows(dev):
        assert r["questions"] == R.QUESTION
    rep = json.loads((DATA / "m2/ragtruth-polarity/guard-scan.json").read_text(encoding="utf-8"))
    s = json.loads(SEL.read_text(encoding="utf-8"))
    assert rep["fail_ids_sha256"] == s["guard_report"]["fail_ids_sha256"]
    ex = R.dev_exclusions(resp, dev, rep["fail_ids"])
    sel = R.select(resp, srcs, set(ex["all"]))
    assert sel["ids_sha256"] == s["ids_sha256"] == "794bcce30da94f7da3696b522a1a20d1e8719b51286970dc2a47ac7fc8ab9295"
    assert all(resp[i]["split"] == "train" for i in sel["ids"])
    items = DATA / "m2/ragtruth-polarity/items.jsonl.gz"
    if items.is_file():
        assert hashlib.sha256(items.read_bytes()).hexdigest() == s["items_sha256"]
        rows = [json.loads(x) for x in gzip.open(items, "rt", encoding="utf-8")]
        assert sorted(r["metadata"]["source_id"] for r in rows) == s["ids"]
        assert all(r == R.build_row(resp[r["metadata"]["source_id"]], srcs[r["metadata"]["source_task_id"]])
                   for r in rows)


# ------------------------------------------------------------------------------------------------ trimmed caps (hold)
def test_trimmed_caps_hold_and_slow_box_margins():
    from jobs.common.costs import worst_case_usd

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    s = json.loads(SEL.read_text(encoding="utf-8"))
    assert p["max_duration_minutes"] == 115 and round(115 * 0.063, 2) == 7.25
    assert worst_case_usd(p) == 14.36 <= p["hard_cap_usd"] == 15.0
    assert p["submit"]["key"].endswith("-20261009b")
    est = R.estimate(p, s)
    assert est["ok"] and all(est["checks"].values()) and R.SLOW_BOX == 1.25
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert '--setup-cap-min "$SCAP"' in sh and "ragtruth_polarity reprice" in sh
    assert sh.index("ragtruth_polarity reprice") < sh.index("checkpoint_loader install --params \"$PARAMS\" --block")


def test_estimate_refuses_when_a_slow_box_would_not_fit():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    s = json.loads(SEL.read_text(encoding="utf-8"))
    tight = json.loads(json.dumps(p))
    tight["phase_caps_minutes"]["score_d13"] = 36  # 31 expected fits, 38.8 on a 1.25x box does not
    est = R.estimate(tight, s)
    assert not est["ok"] and not est["checks"]["score_d13_slow_le_95pct_cap"]
    slow_load = json.loads(json.dumps(p))
    slow_load["evals_fixed_minutes"]["d1"] = 14  # 17.5 min on a slow box > setup cap 16
    assert not R.estimate(slow_load, s)["checks"]["load_d1_slow_le_setup_cap"]


def test_reprice_projects_d1_from_d13_measurements_before_fetching_it():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    s = json.loads(SEL.read_text(encoding="utf-8"))
    ok = R.reprice(p, s, {"scoring_started_min": 25.0}, {"sum_row_wall_s": 360.0}, elapsed_min=50)
    assert ok["ok"] and ok["d1_load_proj"] == 12.0 and ok["d1_score_proj"] == 6.0
    slow = R.reprice(p, s, {"scoring_started_min": 40.0}, {"sum_row_wall_s": 600.0}, elapsed_min=60)  # 1.6x load
    assert not slow["ok"] and slow["d1_phase_proj"] > 0.95 * p["phase_caps_minutes"]["score_d1"]
    late = R.reprice(p, s, {"scoring_started_min": 25.0}, {"sum_row_wall_s": 360.0}, elapsed_min=95)  # clock gone
    assert not late["ok"]

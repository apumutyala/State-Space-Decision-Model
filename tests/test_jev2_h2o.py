"""A-PREREG-JEV2-H2O (modes jev2_h2o_shim / jev2_h2o_sym) on CPU: the JEV2-4B recipe initialised from H2O-Lightning-4B
v1.1, with the lead's 2026-10-11 changes ($38 worst case, jev2c-v2, the H2O eval's jev2a slice forced into the selection
slice, anti-forgetting guards, step-0 floor). Package invariants, host pins, data pins, schedule / budget, the draw's
forced-calib option; the torch-only parts (host pin check, dev-card source, gate identity) skip without torch."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "jobs/m2_first_cell"
JEV2 = json.loads((M / "m2_first_cell_rwkv7_jev2_q35_4b.json").read_text(encoding="utf-8"))
JEV2_P = json.loads((M / "job_params_jev2_q35_4b.json").read_text(encoding="utf-8"))
MODES = {"jev2_h2o_shim": ("g1v2", 16.9, 341, "off"), "jev2_h2o_sym": ("sym-fork", 16.6, 320, "off"),
         "jev2_h2o_shim_jev1": ("g1v2", 16.9, 342, "on")}
MIX = {"off": ("bschema-draw-jev2h-nojev1", "m2-jev2hn", "data_jev2hn.sha256"),
       "on": ("bschema-draw-jev2h-mix", "m2-jev2hmix", "data_jev2h.sha256")}
REV = "193ad740925b176a3b70a5a13a7cff2f2fadd01e"
H2O_SLICE_SHA = "204e9bfe20d29ca95bd895ff368cf26f7b282635d615aeb85df47e69fcba2a9c"
#: recipe keys that must equal JEV2-4B's (loss, smoothing, lengths, head, template, optimiser schedule shape)
SAME = ("template", "head", "loss_arm", "lr_head", "warmup_steps", "batch_size", "grad_accum", "max_k", "t_ladder",
        "eval_max_t", "require_t_max", "ladder_probe_longest", "label_smoothing", "label_smoothing_noul",
        "lr_schedule", "lr_min_frac", "lr_horizon_frac", "train_max_k", "row_options", "act_ckpt", "best_ckpt",
        "collapse_guard", "assert_no_truncation", "cohort_sha256", "parent", "seed")


def cell(mode):
    return json.loads((M / f"m2_first_cell_rwkv7_{mode}.json").read_text(encoding="utf-8"))


def params(mode):
    return json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))


def manifest(path):
    return {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln}


@pytest.mark.parametrize("mode", MODES)
def test_recipe_kept(mode):
    c = cell(mode)
    assert {k: c.get(k) for k in SAME} == {k: JEV2.get(k) for k in SAME}
    assert c["layout"] == MODES[mode][0] and c["template"] == "jrt-h2o-shim:qwen3.5-4b"
    assert max(c["t_ladder"]) == 4096 and c["assert_no_truncation"] and c["label_smoothing_noul"] == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_anti_forgetting_guards(mode):
    c = cell(mode)
    assert c["lora_rank"] == 16 < JEV2["lora_rank"]
    assert c["lr_lora"] == pytest.approx(4e-5) and c["lr_lora"] <= 1e-4  # <= r2's LR
    assert c["max_steps"] == MODES[mode][2] <= c["n_train_records"] // 8 and c["min_steps"] == 200
    assert c["max_minutes"] >= math.ceil(c["max_steps"] * MODES[mode][1] * 1.3 / 60)
    assert "lr_horizon_steps" not in c


@pytest.mark.parametrize("mode", MODES)
def test_data_pins(mode):
    c, p = cell(mode), params(mode)
    name, td, shaf = MIX[MODES[mode][3]]
    ts = c["train_set"]
    assert ts["pointer"] == p["train_set_pointer"] == f"data/manifests/{name}.pointer.json"
    ptr = json.loads((ROOT / ts["pointer"]).read_text(encoding="utf-8"))
    bc = ptr["build_command"] + " "
    assert ptr["n_records"] == c["n_train_records"] and "--exclude-sub prob_indep" in bc
    # the cleared set only: never jev2c (v1) or jev2a-v2 beside their counterparts
    assert "--src <data>/m2/build/bschema-draw-jev2c-v2 " in bc and "bschema-draw-jev2c " not in bc
    assert "bschema-draw-jev2a-v2" not in bc and "--pool-rows bschema-draw-jev2a=29895" in bc
    assert f"--force-calib-sha {H2O_SLICE_SHA}" in bc
    if MODES[mode][3] == "off":
        assert "bschema-draw-jev1" not in bc and c["jev1"]["enabled"] is False
    else:
        assert "--exclude long_policy=bschema-draw-jev1" in bc and c["jev1"]["enabled"] is True
        assert c["jev1"]["exclusion"]["status"] == "PENDING"  # run_cell + the runner refuse until it lands
    jc = ts["pools"]["jev2c"]
    assert jc["pointer_sha256"] == hashlib.sha256((ROOT / jc["pointer"]).read_bytes()).hexdigest()
    assert jc["pointer_sha256"].startswith("86960d90")
    ex = ts["pools"]["jev2a"]["exclude"]
    assert hashlib.sha256((ROOT / ex["file"]).read_bytes()).hexdigest() == ex["sha256"].strip() and ex["n_ids"] == 105
    assert hashlib.sha256((M / "h2o_v11_heldout_jev2a_ids.txt").read_bytes()).hexdigest() == H2O_SLICE_SHA
    pins = manifest(M / shaf)
    assert {d["dest"] for d in p["context"]["data"] if d.get("pin") == f"jobs/m2_first_cell/{shaf}"} == set(pins)
    assert {x.split("/")[0] for x in pins} == {td, f"{td}-calib"} and p["train_dir"] == td
    assert not any(d["dest"].startswith("m2-jev2mix") for d in p["context"]["data"])


def test_jev1_switch_guards():
    job = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert "jev1.exclusion.status != landed" in job
    src = (ROOT / "train/run_cell.py").read_text(encoding="utf-8")
    assert '(j1.get("exclusion") or {}).get("status") != "landed"' in src


@pytest.mark.parametrize("mode", MODES)
def test_selection_slice_and_floor(mode):
    c, p = cell(mode), params(mode)
    d = c["dev_card_records"]
    assert d == {"records": c["calib_slice"]["records"], "ids_sha256": c["calib_slice"]["ids_sha256"]}
    assert d["records"] == f"{MIX[MODES[mode][3]][1]}-calib/records.jsonl.gz" == p["jevkit"]["tfit"]["records"]
    assert p["jevkit"]["tfit"]["ids_sha256"] == d["ids_sha256"]
    assert c["best_ckpt"] and c["dev_card_every"] > 0 and c["dev_card_rows"] >= 200
    jk = p["jevkit"]
    assert jk["paired_primary"] == "native_shimcal" and jk["shim_calibration"] and jk["stock_shim"]
    assert jk["slice_floor"] is True and jk["by_family"] is True
    from jobs.m2_first_cell.screen import jevkit_block
    blk = jevkit_block(p, c["template"])
    assert blk["slice_floor"] and blk["by_family"]


@pytest.mark.parametrize("mode", MODES)
def test_host_pins(mode):
    q = cell(mode)["qwen35"]
    assert (q["repo"], q["revision"], q["host"]) == ("h2oai/h2o-lightning-4b", REV, True)
    man = manifest(ROOT / q["snapshot_sha256_file"])
    assert man["config.json"] == q["config_sha256"]
    assert {k: man[k] for k in q["weights_sha256"]} == q["weights_sha256"]
    stock = manifest(M / "qwen35_4b_snapshot.sha256")
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        assert man[f] == stock[f]  # same tokenizer + chat template as Qwen3.5-4B: the pinned shim identity binds
    import third_party.h2o_lightning_shim as H
    assert {f: man[f] for f in H.FILES_SHA256} == H.FILES_SHA256  # the vendored shim IS the v1.1 snapshot's shim
    assert "LICENSE" in q["license_git_blobs"]
    assert b"\r" not in (ROOT / q["snapshot_sha256_file"]).read_bytes()


@pytest.mark.parametrize("mode", MODES)
def test_schedule_and_38_dollar_budget(mode):
    from jobs.common import costs

    c, p = cell(mode), params(mode)
    caps = sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"]
    assert p["max_duration_minutes"] == caps and p["phase_caps_minutes"]["cell"] == c["max_minutes"] + 35
    assert costs.worst_case_usd(p) <= p["hard_cap_usd"] == 38.0
    assert costs.PARAMS[f"m2_first_cell_{mode}"] == f"jobs/m2_first_cell/job_params_{mode}.json"
    assert f"{costs.worst_case_usd(p):.2f}" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"job_params_{mode}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}") and p["cell_config"].endswith(f"_{mode}.json")


def test_prereg_line_in_job_spec():
    spec = (M / "JOB_SPEC.md").read_text(encoding="utf-8")
    i = spec.index("### A-PREREG-JEV2-H2O")
    for s in ("PRIMARY", "H2O-Lightning-4B v1.1 + SSDM jev2 data", "193ad740", "point estimate", "floor", "r16",
              "prob_indep", "204e9bfe", "never select", "Composite A", "$38"):
        assert s.lower() in spec[i:].lower(), s


def test_runner_copies_license_for_hosts():
    job = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert "license_git_blobs" in job and '"$WORK/out/license"' in job and 'info.sha == q["revision"]' in job


def test_force_calib(tmp_path):
    from scripts.jev12_balanced_draw import main
    d = tmp_path / "jev1"
    d.mkdir()
    with gzip.open(d / "records.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps({"id": f"x{i}", "family": "jev1_intent", "source": "t", "state": {"k": i},
                                "questions": {"q": {"type": "noul", "instructions": f"q{i}"}}, "labels": {"q": "true"},
                                "aug": {"jev1": {"family8": "intent"}}}) + "\n")
    fc = tmp_path / "force.txt"
    fc.write_bytes(b"x0\nx1\nx2\n")
    base = ["--src", str(d), "--train-per-family", "10", "--calib-per-family", "5", "--force-calib", str(fc),
            "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        main(base + ["--force-calib-sha", "0" * 64, "--name", "a"])
    assert main(base + ["--force-calib-sha", hashlib.sha256(fc.read_bytes()).hexdigest(), "--name", "b"]) == 0
    rep = json.loads((tmp_path / "o/b/draw-report.json").read_text(encoding="utf-8"))
    assert rep["calib_rows"] == {"intent": 3} and rep["train_rows"] == {"intent": 10}  # only the forced ids as calib
    with gzip.open(tmp_path / "o/b-calib/records.jsonl.gz", "rt", encoding="utf-8") as f:
        assert sorted(json.loads(x)["id"] for x in f) == ["x0", "x1", "x2"]


def test_host_pin_check_fails_closed(tmp_path):
    pytest.importorskip("torch")
    from heads.backbone import QWEN35_HOSTS, LineageError, qwen35_pin_check

    d = tmp_path / REV
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"w")
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    pin = {"repo": "h2oai/h2o-lightning-4b", "revision": QWEN35_HOSTS["h2oai/h2o-lightning-4b"], "host": True,
           "config_sha256": sha(b"{}"), "weights_sha256": {"model.safetensors": sha(b"w")}}
    assert qwen35_pin_check(pin, d)["kind"] == "qwen35-host"
    for bad in ({**pin, "host": False}, {**pin, "weights_sha256": {}}, {**pin, "revision": "0" * 40},
                {**pin, "weights_sha256": {"model.safetensors": "0" * 64}}):
        with pytest.raises(LineageError):
            qwen35_pin_check(bad, d)
    other = tmp_path / "not-the-commit"
    other.mkdir()
    for f in ("config.json", "model.safetensors"):
        (other / f).write_bytes((d / f).read_bytes())
    with pytest.raises(LineageError):
        qwen35_pin_check(pin, other)


def test_dev_card_source(tmp_path):
    pytest.importorskip("torch")
    from train.run_cell import dev_card_source

    (tmp_path / "calib").mkdir()
    rows = [{"id": i, "state": {"x": i}, "questions": {"q": {"type": "noul", "instructions": "ok?"}},
             "labels": {"q": "true"}} for i in ("b", "a")]
    with gzip.open(tmp_path / "calib/records.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    good = hashlib.sha256(b"a\nb").hexdigest()
    assert dev_card_source({}, tmp_path / "train") is None
    recs, ids = dev_card_source({"dev_card_records": {"records": "calib/records.jsonl.gz", "ids_sha256": good}},
                                tmp_path / "train")
    assert ids == ["a", "b"] and set(recs) == {"a", "b"}
    with pytest.raises(ValueError):
        dev_card_source({"dev_card_records": {"records": "calib/records.jsonl.gz", "ids_sha256": "0" * 64}},
                        tmp_path / "train")


def test_gate_identity_accepts_only_pinned_hosts():
    pytest.importorskip("torch")
    from train import m2_gates as G

    host = {"kind": "qwen35-host", "repo": "h2oai/h2o-lightning-4b", "revision": REV, "config_sha256": "c" * 64,
            "weights_sha256": {"model.safetensors": "w" * 64}}
    assert G.expected_parent_identity("qwen35", provenance=host)["weights_sha256"] == host["weights_sha256"]
    with pytest.raises(G.IdentityMismatch):
        G.expected_parent_identity("qwen35", provenance={**host, "weights_sha256": None})

"""A-PREREG-JEV2-H2O (modes jev2_h2o_shim / jev2_h2o_sym) on CPU: the JEV2-4B recipe initialised from H2O-Lightning-4B
v1.1. Package invariants (same recipe and data pins as JEV2-4B except the declared deltas), the host pins (revision,
config / weight / tokenizer sha256, license blobs), the dev-card selection slice, the schedule and the $60 budget; the
torch-only parts (host pin check, dev-card source, gate identity) skip without torch."""
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
MODES = {"jev2_h2o_shim": ("g1v2", 16.9), "jev2_h2o_sym": ("sym-fork", 16.6)}
REV = "193ad740925b176a3b70a5a13a7cff2f2fadd01e"
#: the keys a JEV2-H2O cell may differ from JEV2-4B in (everything else = the same recipe, data and pins)
DELTA = {"_comment", "layout", "lr_lora", "max_steps", "min_steps", "max_minutes", "dev_card_every", "dev_card_rows",
         "dev_card_records", "variant", "qwen35", "host_init_note", "dev_card_note", "budget_note", "min_steps_note",
         "g1_note"}
P_DELTA = {"_comment", "job", "status", "max_duration_minutes", "hard_cap_usd", "phase_caps_minutes", "expected_minutes",
           "model", "cell_config", "mode", "host_label", "submit", "context", "jevkit", "jevkit_note", "reserve_minutes"}


def cell(mode):
    return json.loads((M / f"m2_first_cell_rwkv7_{mode}.json").read_text(encoding="utf-8"))


def params(mode):
    return json.loads((M / f"job_params_{mode}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", MODES)
def test_same_recipe_as_jev2_except_declared_deltas(mode):
    c, p = cell(mode), params(mode)
    assert {k for k in set(c) | set(JEV2) if c.get(k) != JEV2.get(k)} <= DELTA
    assert {k for k in set(p) | set(JEV2_P) if p.get(k) != JEV2_P.get(k)} <= P_DELTA
    assert c["layout"] == MODES[mode][0] and c["template"] == JEV2["template"] == "jrt-h2o-shim:qwen3.5-4b"
    for k in ("price", "shim_calibration", "stock_shim", "paired_primary", "tfit", "perturb_ops", "covered_families"):
        assert (k == "price") or p["jevkit"][k] == JEV2_P["jevkit"][k]
    assert p["context"]["data"] == JEV2_P["context"]["data"]  # the same pinned mix + calib slice files


@pytest.mark.parametrize("mode", MODES)
def test_lower_lr_fewer_steps_rate_logic_kept(mode):
    c = cell(mode)
    assert c["lr_lora"] == pytest.approx(0.4 * JEV2["lr_lora"]) and c["lr_head"] == JEV2["lr_head"]  # fresh head
    assert c["max_steps"] == 800 < JEV2["max_steps"] and c["min_steps"] == 400 <= c["max_steps"]
    assert c["lr_horizon_frac"] == JEV2["lr_horizon_frac"] and "lr_horizon_steps" not in c
    assert c["n_train_records"] == JEV2["n_train_records"]  # the whole draw is verified; training reads a prefix
    assert c["max_minutes"] >= math.ceil(c["max_steps"] * MODES[mode][1] * 1.3 / 60)


@pytest.mark.parametrize("mode", MODES)
def test_dev_cards_select_on_the_heldout_calib_slice(mode):
    c, p = cell(mode), params(mode)
    d = c["dev_card_records"]
    assert d == {"records": c["calib_slice"]["records"], "ids_sha256": c["calib_slice"]["ids_sha256"]}
    assert d["records"].startswith("m2-jev2mix-calib/") and not p["train_dir"].endswith("calib")
    assert c["best_ckpt"] and c["dev_card_every"] > 0 and c["dev_card_rows"] >= 200
    assert "jevbench" not in d["records"].lower()  # never the public items


@pytest.mark.parametrize("mode", MODES)
def test_host_pins(mode):
    q = cell(mode)["qwen35"]
    assert (q["repo"], q["revision"], q["host"]) == ("h2oai/h2o-lightning-4b", REV, True)
    assert q["label"] == "H2O-Lightning-4B v1.1 + SSDM" == params(mode)["host_label"]
    man = dict(reversed(ln.split("  ", 1)) for ln in (ROOT / q["snapshot_sha256_file"]).read_text(encoding="utf-8").split("\n") if ln)
    assert man["config.json"] == q["config_sha256"]
    assert {k: man[k] for k in q["weights_sha256"]} == q["weights_sha256"]
    stock = dict(reversed(ln.split("  ", 1)) for ln in (M / "qwen35_4b_snapshot.sha256").read_text(encoding="utf-8").split("\n") if ln)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        assert man[f] == stock[f]  # same tokenizer + chat template as Qwen3.5-4B: the pinned shim identity binds
    import third_party.h2o_lightning_shim as H
    assert {f: man[f] for f in H.FILES_SHA256} == H.FILES_SHA256  # the vendored shim IS the v1.1 snapshot's shim
    assert set(q["license_git_blobs"]) >= {"LICENSE"}
    assert q["snapshot_sha256_file"] in params(mode)["context"]["manifest_files"]
    assert b"\r" not in (ROOT / q["snapshot_sha256_file"]).read_bytes()  # sha256sum -c on the box


@pytest.mark.parametrize("mode", MODES)
def test_schedule_and_60_dollar_budget(mode):
    from jobs.common import costs

    c, p = cell(mode), params(mode)
    caps = sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"]
    assert p["max_duration_minutes"] == caps and p["phase_caps_minutes"]["cell"] == c["max_minutes"] + 35
    assert costs.worst_case_usd(p) <= p["hard_cap_usd"] == 60.0
    assert costs.PARAMS[f"m2_first_cell_{mode}"] == f"jobs/m2_first_cell/job_params_{mode}.json"
    assert f"{costs.worst_case_usd(p):.2f}" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"job_params_{mode}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert p["submit"]["command"].endswith(f"first_cell_job.sh {mode}") and p["cell_config"].endswith(f"_{mode}.json")


def test_prereg_line_in_job_spec():
    spec = (M / "JOB_SPEC.md").read_text(encoding="utf-8")
    i = spec.index("A-PREREG-JEV2-H2O")
    for s in ("PRIMARY", "H2O-Lightning-4B v1.1 + SSDM", "193ad740", "noul", "G1", "never select", "Composite A"):
        assert s.lower() in spec[i:].lower(), s


def test_runner_copies_license_for_hosts():
    job = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert 'license_git_blobs' in job and '"$WORK/out/license"' in job and "info.sha == q[\"revision\"]" in job


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

    fn = G.expected_parent_identity
    host = {"kind": "qwen35-host", "repo": "h2oai/h2o-lightning-4b", "revision": REV, "config_sha256": "c" * 64,
            "weights_sha256": {"model.safetensors": "w" * 64}}
    assert fn("qwen35", provenance=host)["weights_sha256"] == host["weights_sha256"]
    with pytest.raises(G.IdentityMismatch):
        fn("qwen35", provenance={**host, "weights_sha256": None})

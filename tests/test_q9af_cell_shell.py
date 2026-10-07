"""EXECUTES jobs/q9af_cell/q9af_cell_job.sh end to end in a subprocess (Q9AFC_DRYRUN=1, CPU, $0; LESSONS L-10: a
Python-level test does not count):

1. mode ``parent`` on a tiny Qwen3.5 stand-in whose attention layer is replaced by a converted GDN through the real
   export contract (gdn_converted.pt + export.json, sha256-pinned in the cell config, ``jobs.q9af_cell.pin check``);
2. mode ``cell_r64`` against that parent arm (its arm.json sha256 pinned): preflight, export pin, frozen-family check,
   ``train.run_cell`` smoke (cache == recompute fork) -> ladder -> training WITH Q9-AF block activation checkpointing,
   the dry gates, ``jobs.q9af_cell.summarize`` + the pre-registered rule, adapters export, finalize (Q9AF_CELL_DONE);
3. the same cell job again on the same checkpoint dir: it RESUMES (smoke / ladder skipped, training continues from the
   saved step) and ends with the same verdict;
4. refusals before any model load: an unpinned export (exit 10) and a partial export not configured as hybrid (exit 10).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "jobs/q9af_cell/q9af_cell_job.sh"
BASH = shutil.which("bash")
PIN = "5.17.0"


def _kit_dir() -> str:
    try:
        from scripts.m2_e2e_dryrun import ensure_kit

        return str(Path(ensure_kit()["kit_path"]).parent)
    except ImportError:
        pytest.skip("Decision Index kit (decision_index) not importable and no pinned local clone")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _model_and_export(tmp: Path) -> tuple[Path, Path]:
    from tests.test_q9_feature_cache_shell import _tiny_model_dir

    model = tmp / "model"
    _tiny_model_dir(model)  # stock hybrid Qwen3.5 (layer 3 = full attention), byte-level tokenizer
    cfg = tf.Qwen3_5TextConfig.from_pretrained(model)
    attn = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    from heads.backbone import tiny_hf_backbone

    donor = tiny_hf_backbone("q9af", seed=1, hidden=32)
    sd = {i: {k: v.clone() for k, v in donor.model.model.layers[i].linear_attn.state_dict().items()} for i in attn}
    gdn = tmp / "data" / "gdn"
    gdn.mkdir(parents=True)
    torch.save(sd, gdn / "gdn_converted.pt")
    torch.save({attn[0]: sd[attn[0]]}, gdn / "gdn_partial_fake.pt")
    meta = {"mode": "final", "ok": True, "file": "gdn_converted.pt", "sha256": _sha(gdn / "gdn_converted.pt"),
            "attention_free": True, "converted": attn, "attention_layers_kept": [],
            "base_config_sha256": _sha(model / "config.json")}
    (gdn / "export.json").write_text(json.dumps(meta), encoding="utf-8")
    return model, gdn


def _cell_config(tmp: Path, model: Path, gdn: Path, **q9af) -> Path:
    c = json.loads((ROOT / "jobs/q9af_cell/q9af_cell_r64.json").read_text(encoding="utf-8"))
    for k in ("train_set", "cohort_sha256"):
        c.pop(k)  # dry run: toy records through run_cell's legacy cohort-less split
    c.update(require_guard_readiness=False, t_ladder=[256, 512, 1024], eval_max_t=1024, n_train_records=16,
             n_eval_records=6, min_train_items=4, max_steps=4, min_steps=2, max_minutes=6, ckpt_every=2,
             eval_every=4, warmup_steps=1, tokenize_workers=1, lora_rank=4,
             pass_rule={"window": 1, "loss_drop_frac_min": -100.0, "max_peak_gb": 76})
    c["smoke"] = {**c["smoke"], "rows": 2, "max_t": 1024}
    c["q9af"] = {**c["q9af"], "converted_sha256": _sha(gdn / "gdn_converted.pt"),
                 "export_json_sha256": _sha(gdn / "export.json"), "base_config_sha256": _sha(model / "config.json"),
                 **q9af}
    p = tmp / f"cell-{len(q9af)}.json"
    p.write_text(json.dumps(c), encoding="utf-8")
    return p


def _toy_records(data: Path) -> None:
    from train.data import toy_records

    with gzip.open(data / "toy-train.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in toy_records(24, seed=1):  # m2 raw layout: state / questions / labels at the top level
            f.write(json.dumps({"id": r["id"], "state": r["request"]["state"], "questions": r["request"]["questions"],
                                "labels": r["labels"]}) + "\n")


def _run(tmp, mode, cfg, model, data, out, work, ck, extra=None):
    env = {**os.environ, "Q9AFC_MODE": mode, "Q9AFC_DRYRUN": "1", "Q9AFC_CODE": str(ROOT), "Q9AFC_DATA": str(data),
           "Q9AFC_WORK": str(work), "Q9AFC_MODEL_DIR": str(model), "Q9AFC_CONFIG": str(cfg),
           "Q9AFC_CONTEXT_MANIFEST": str(tmp / "ctx.json"), "Q9AFC_KIT_DIR": _kit_dir(),
           "GMN_OUTPUT_DIR": str(out), "GMN_RESULT_PATH": str(out / "result.json"), "GMN_CHECKPOINT_DIR": str(ck),
           "GMN_METADATA_URL": "http://127.0.0.1:9", "HF_HUB_CACHE": str(tmp / "hf"), "CUDA_VISIBLE_DEVICES": "",
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""), **(extra or {})}
    if os.name == "nt":  # GNU tar (Git Bash) reads "C:\\..." as host:path without --force-local
        env["TAR_OPTIONS"] = "--force-local"
    cmd = [BASH, "jobs/q9af_cell/q9af_cell_job.sh"]  # == RUNNER, spelled out for the launch lint (L-10)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    log = (work / "logs" / "run.log").read_text(encoding="utf-8", errors="replace") if (work / "logs/run.log").exists() else ""
    return p.returncode, log


def _tail(work: Path, name: str) -> str:
    f = work / "logs" / name
    return f.read_text(encoding="utf-8", errors="replace")[-3000:] if f.exists() else ""


@pytest.mark.requires_transformers(PIN)
@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_runner_dry_run_parent_cell_and_resume(tmp_path):
    _kit_dir()
    model, gdn = _model_and_export(tmp_path)
    data = gdn.parent
    _toy_records(data)
    (tmp_path / "ctx.json").write_text("{}", encoding="utf-8")
    cfg = _cell_config(tmp_path, model, gdn)

    out_p, work_p = tmp_path / "out_parent", tmp_path / "work_parent"
    rc, log = _run(tmp_path, "parent", cfg, model, data, out_p, work_p, tmp_path / "ck_parent")
    assert rc == 0, (rc, log[-2000:], _tail(work_p, "parent.log"))
    assert (out_p / "Q9AF_CELL_DONE").exists()
    pres = json.loads((out_p / "result.json").read_text(encoding="utf-8"))
    pins = pres["decision"]["pins_for_cell_params"]
    arm = out_p / "parent" / "parent_arm"
    assert pins["parent_arm_sha256"] == _sha(arm / "arm.json")
    assert pres["decision"]["label"]["attention_free"] is True

    out_c, work_c, ck = tmp_path / "out_cell", tmp_path / "work_cell", tmp_path / "ck_cell"
    extra = {"Q9AFC_PARENT_DIR": arm.as_posix(), "Q9AFC_PARENT_ARM_SHA256": pins["parent_arm_sha256"]}
    rc, log = _run(tmp_path, "cell_r64", cfg, model, data, out_c, work_c, ck, extra)
    assert rc == 0, (rc, log[-2500:], _tail(work_c, "cell.log"), _tail(work_c, "evals.log"), _tail(work_c, "report.log"))
    assert (out_c / "Q9AF_CELL_DONE").exists()
    res = json.loads((out_c / "result.json").read_text(encoding="utf-8"))
    d = res["decision"]
    from jobs.q9af_cell import rule as R
    from jobs.q9af_cell.summarize import valid_result

    assert valid_result(d) and d["verdict"] in R.VERDICTS and d["rule"] == R.RULE_ID
    assert d["label"]["attention_free"] is True and d["dry_run"]
    assert {"0.2.1/full_dev", "0.2.1/subset_2008", "0.3/full_dev", "0.3/subset_2008"} <= set(d["index"])
    assert set(d["index"]["0.2.1/full_dev"]["columns"]) == {"in_domain", "new_domain", "new_domain_frozen"}
    assert d["references"]["clef_flash_full_dev_32k"] == 57.35
    assert d["gates_rc"] in (0, 40, 41, 42, 43, 44, 45)  # the dry gates carry a REJECT through to the verdict
    assert len((out_c / "result.json").read_bytes()) < 32 * 1024  # L-27
    cell = json.loads((work_c / "out" / "cell" / "cell.json").read_text(encoding="utf-8"))
    assert cell["smoke"]["pass"] and cell["train"]["steps"] == 4 and cell["ladder"]["T_max"]
    assert "act_ckpt" not in cell["config"] or cell["config"]["act_ckpt"] is True
    assert (out_c / "adapters" / "adapters.pt").exists()
    frozen = json.loads((work_c / "out" / "frozen.json").read_text(encoding="utf-8"))
    assert frozen["pass"] and frozen["n_checked"] == 24

    # resume: same checkpoint dir, a fresh work/output dir (a restarted attempt)
    out_r, work_r = tmp_path / "out_resume", tmp_path / "work_resume"
    rc, log = _run(tmp_path, "cell_r64", cfg, model, data, out_r, work_r, ck, extra)
    assert rc == 0, (rc, log[-2000:], _tail(work_r, "cell.log"))
    cell2 = json.loads((work_r / "out" / "cell" / "cell.json").read_text(encoding="utf-8"))
    assert cell2["train"]["resumed"] is True and cell2["train"]["steps"] == 4
    assert cell2["smoke"] == cell["smoke"] and cell2["ladder"] == cell["ladder"]  # finished stages are not re-run


@pytest.mark.requires_transformers(PIN)
@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("case", ["unpinned", "partial_not_hybrid"])
def test_runner_refuses_unpinned_or_partial_export(tmp_path, case):
    model, gdn = _model_and_export(tmp_path)
    (tmp_path / "ctx.json").write_text("{}", encoding="utf-8")
    if case == "unpinned":
        cfg = _cell_config(tmp_path, model, gdn, converted_sha256=None)
    else:
        meta = json.loads((gdn / "export.json").read_text(encoding="utf-8"))
        meta.update(file="gdn_partial_fake.pt", sha256=_sha(gdn / "gdn_partial_fake.pt"), attention_free=False,
                    attention_layers_kept=[1])
        (gdn / "export.json").write_text(json.dumps(meta), encoding="utf-8")
        cfg = _cell_config(tmp_path, model, gdn, converted_sha256=_sha(gdn / "gdn_partial_fake.pt"))
    work = tmp_path / "work"
    rc, log = _run(tmp_path, "parent", cfg, model, gdn.parent, tmp_path / "out", work, tmp_path / "ck")
    assert rc == 10, (rc, log[-1500:])
    assert "export pin" in log and not (tmp_path / "out" / "Q9AF_CELL_DONE").exists()

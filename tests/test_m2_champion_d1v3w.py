"""Champion final-claim modes (fulldev_eval_champ, hijack_eval_champ): D1's saved checkpoint scored under the v3w
readout, through jobs/m2_first_cell/readout_switch.py (readout_ab's swap). CPU only; plumbing + identities."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
FD, FDC = M / "job_params_fulldev_eval.json", M / "job_params_fulldev_eval_champ.json"
HJ, HJC = M / "job_params_hijack_eval.json", M / "job_params_hijack_eval_champ.json"
CHAMP_CKPT = "43414f022ccfe2eb7d770957ae851f5450839457d5a41e9f7fb661b3f81cfd32"


def _p(f):
    return json.loads(f.read_text(encoding="utf-8"))


def _strip(p: dict, extra=()) -> dict:
    drop = {"_comment", "job", "status", "cell_config", "submit", "context", "mode", "expected_minutes",
            "readout", "readout_template", "readout_length_ratio", "refit_temperature", "readout_refit_minutes",
            "evals_fixed_minutes", *extra}
    return {k: v for k, v in p.items() if k not in drop}


@pytest.mark.parametrize("base,champ", [(FD, FDC), (HJ, HJC)])
def test_champ_modes_are_the_reviewed_modes_plus_the_v3w_readout(base, champ):
    from jobs.common.costs import worst_case_usd
    from jobs.m2_first_cell.checkpoint_loader import checkpoint_block

    b, c = _p(base), _p(champ)
    assert c["readout"] == "v3w" and _strip(b) == _strip(c)
    assert c["checkpoint"]["ckpt_sha256"] == CHAMP_CKPT and c["checkpoint"]["artifact"] == "art-bm3e6"
    checkpoint_block(c)
    assert worst_case_usd(c) == worst_case_usd(b) <= c["hard_cap_usd"]
    # the cell config is a byte copy (the loader requires config == the checkpoint's cell.json config)
    assert (ROOT / c["cell_config"]).read_bytes() == (ROOT / b["cell_config"]).read_bytes()
    assert c["cell_config"].endswith(f"m2_first_cell_rwkv7_{champ.stem[len('job_params_'):]}.json")
    # NEW idempotency key; the signed URL is never committed (placeholder only)
    assert c["submit"]["key"] != b["submit"]["key"] and "champ-d1v3w" in c["submit"]["key"]
    assert c["submit"]["env"]["CKPT_URL"].startswith("<") and "https" not in json.dumps(c)
    assert c["submit"]["command"].endswith(champ.stem[len("job_params_"):])
    sub = {k: v for k, v in c["submit"].items() if k not in ("command", "key", "label")}
    assert sub == {k: v for k, v in b["submit"].items() if k not in ("command", "key", "label")}
    files = c["context"]["manifest_files"]
    assert f"jobs/m2_first_cell/{champ.name}" in files and c["cell_config"] in files
    pins = [e["pin_json"][0] for e in c["context"]["data"] if "pin_json" in e]
    assert pins == [f"jobs/m2_first_cell/{champ.name}"]


def test_fulldev_champ_is_full_dev_child_only_at_the_65k_cap_and_priced_at_the_v3w_length():
    from jobs.m2_first_cell.fulldev_eval import estimate

    c = _p(FDC)
    assert c["eval_only"] == "fulldev" and c.get("rows", "full") == "full" and not c["fulldev_parent_arm"]
    assert c["max_input_tokens"] == 65_536 and c["readout_template"] == "jrt-rwkv-g1-v3w"
    e, e1 = estimate(c), estimate(_p(FD))
    assert e["ok"] and e["readout"] == "v3w" and e["readout_length_ratio"] == 0.9004 and e["refit_minutes"] == 12
    assert c["refit_temperature"] is True
    assert abs(e["minutes"]["fulldev"] - 439.2) < 0.1
    bad = {**c, "readout_template": "jrt-rwkv-g1-v1"}
    with pytest.raises(ValueError, match="readout_template"):
        estimate(bad)


def test_readout_switch_is_readout_abs_swap_over_the_same_model():
    torch = pytest.importorskip("torch")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_first_cell import readout_switch as RS
    from tests.test_m2_readout_ab import REQ, WordEncoder

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    b0 = DecisionModelBackend(model, WordEncoder(), template="jrt-rwkv-g1-v1", layout="ordered",
                              checkpoint={"path": "ckpt-step0001439.pt", "sha256": CHAMP_CKPT})
    assert RS.switch_backend(b0, None) is b0 and RS.switch_backend(b0, "training") is b0
    bw = RS.switch_backend(b0, "v3w")
    assert bw.model is b0.model and bw.encode is b0.encode and bw.checkpoint == b0.checkpoint
    assert bw.shared_prefix == b0.shared_prefix and bw.layout == "g1v2"
    assert Decider(bw).revision() != Decider(b0).revision()
    r = Decider(bw).decide(REQ)
    assert r["ssdm"]["status"] == "ok", r["ssdm"]
    cfg = {"template": "jrt-rwkv-g1-v1", "layout": "ordered"}
    idt = RS.identity(cfg, "v3w", tempered=True)
    assert idt["template"] == "jrt-rwkv-g1-v3w" and not idt["native"]
    assert "training-template fit" in idt["temperature_scope"] and "jrt-rwkv-g1-v1" in idt["temperature_scope"]
    assert RS.identity(cfg, None, tempered=True)["native"] is True
    with pytest.raises(ValueError):
        RS.check("v9")


def test_v3w_child_renders_and_scores_every_hijack_unit():
    """The hijack stage's own variants (clean + 5 HIJACK_TEMPLATES + D1-A) through the v3w child: every unit ok."""
    torch = pytest.importorskip("torch")
    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.model import DecisionModel
    from jobs.m2_first_cell import readout_switch as RS
    from safety_eval import attacks
    from safety_eval.decide_bridge import DecideBridge
    from tests.test_m2_readout_ab import WordEncoder
    from train import m2_gates as G

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    b0 = DecisionModelBackend(model, WordEncoder(), template="jrt-rwkv-g1-v1", layout="ordered")
    from safety_eval.data import build

    steer_w, steer_1 = DecideBridge(Decider(RS.switch_backend(b0, "v3w"))), DecideBridge(Decider(b0))
    variants = attacks.build_variants(G.hijack_subset(build(n_route=4, n_pay=2, split="dev"), 1), hijack=None,
                                      d1=None, final=False)
    fams = {v["_meta"]["attack_family"] for v in variants}
    assert {"clean", "hijack", "d1"} <= fams
    n_ok = 0
    for v in variants:
        w, o = steer_w(v), steer_1(v)
        assert w.get("status", w.get("action")) == o.get("status", o.get("action")), (w, o)  # no new refusals
        n_ok += w.get("status") == "ok"
    assert n_ok > 0


def test_m2_gates_readout_is_a_hijack_stage_knob_only():
    from train import m2_gates as G

    base = ["--cell", "x", "--ckpt-dir", "x", "--model-dir", "x", "--records", "x", "--guard-readiness", "x",
            "--parent-arm", "x", "--parent-arm-sha256", "0" * 64, "--out", "x"]
    assert G.main(base + ["--stage", "g12", "--readout", "v3w"]) == G.EXIT["usage"]
    assert G.main(base + ["--stage", "hijack", "--readout", "nope"]) == G.EXIT["usage"]


def _fd_result(**over):
    pt = {"point": 38.0, "lo": 35.0, "hi": 41.0}
    d = {"schema": "ssdm.m2.fulldev_eval.v1", "verdict": "measured", "rows": 12_104, "rows_mode": "full",
         "arms": {"child": {"status": {"ok": 12_104}, "full": {"index": pt}, "excl_overlength51": {"index": pt},
                            "latency": {"median_ms": 900.0, "p95_ms": 2500.0}}},
         "checkpoint": {"ckpt_sha256": CHAMP_CKPT, "loaded": {"sha256": CHAMP_CKPT}},
         "readout_requested": "v3w", "readout": {"name": "v3w", "template": "jrt-rwkv-g1-v3w"}}
    d.update(over)
    return d


def test_fulldev_valid_result_requires_the_requested_readout_was_loaded():
    from jobs.m2_first_cell.fulldev_eval import valid_result

    assert valid_result(_fd_result())
    assert not valid_result(_fd_result(readout={"name": "v1"}))
    assert not valid_result(_fd_result(readout=None))
    assert valid_result(_fd_result(readout_requested=None, readout=None))


def test_hijack_valid_result_requires_the_requested_readout_was_loaded():
    from jobs.m2_first_cell import hijack_eval as H

    ok = {"schema": H.SCHEMA, "verdict": "measured", "complete": True, "units": {"n_units_per_arm": 3600},
          "hijack": {"n_units": 1800, "hijack_rate": {"child": 0.0, "parent": 0.35,
                                                       "diff": {"point": -0.35, "lo": -0.6, "hi": -0.1}}},
          "steer_d1a": {"n_units": 1440, "steer_rate": {"diff": {"point": -0.3}}}, "m2_gates_exit": 46,
          "checkpoint": {"ckpt_sha256": CHAMP_CKPT, "loaded": {"sha256": CHAMP_CKPT}},
          "readout_requested": "v3w", "readout": {"requested": "v3w"}}
    assert H.valid_result(ok)
    assert not H.valid_result({**ok, "readout": {"requested": "training"}})
    assert H.valid_result({**ok, "readout_requested": "training", "readout": None})


def test_runner_passes_the_readout_to_the_child_only():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert 'CHILD_EXTRA+=(--readout "$FD_READOUT")' in sh and 'CHILD_EXTRA+=(--refit-temperature)' in sh
    assert '--stage hijack "${HJ_RO[@]}"' in sh
    assert 'eval_only {ev!r} must be none, fulldev, readout_ab or hijack' in sh
    tree = (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
    assert "job_params_fulldev_eval_champ.json" in tree and "job_params_hijack_eval_champ.json" in tree


def test_eval_only_paths_never_put_the_checkpoint_in_output():
    """The checkpoint lives in the job scratch ($WORK/ckpt-cell for fulldev / readout_ab since job-afpc2, $CK/cell for
    hijack; the downloaded tar in $WORK/art); only $WORK/out + logs are exported, and the eval-only branches exit
    before the training path's `cp ... $OUT/ckpt` (the 5 GB artifact cap). $OUT appears only in READ-ONLY free-space
    checks (job-afpc2's disk preflight / log line)."""
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    a = sh.index('if [ "$EVAL_ONLY" = fulldev ] || [ "$EVAL_ONLY" = readout_ab ]; then')
    b = sh.index("\n  RC=0\n  exit 0\nfi\n", a)
    block = sh[a:b]
    reads = [ln for ln in block.splitlines() if "$OUT" in ln]
    assert reads and all("diskfree" in ln or "shutil.disk_usage" in ln for ln in reads), reads
    assert "CKCELL=$WORK/ckpt-cell" in block and '--ck-dir "$CKCELL" --work "$WORK/art"' in block
    assert '--ck-dir "$CK/cell" --work "$WORK/art"' in block  # hijack_eval: unchanged
    assert sh.index('mkdir -p "$OUT/ckpt"') > b


def test_artifact_size_guard_fails_loudly_before_writing(tmp_path):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    out, work = tmp_path / "output", tmp_path / "work"
    (work / "out").mkdir(parents=True)
    (work / "logs").mkdir()
    (work / "out" / "r.json").write_text("x" * 5000)
    from tests.test_m2_first_cell_shell import _posix

    lib = _posix(ROOT / "jobs/common/lib.sh")
    script = (f'T0=$(date +%s); OUT="{_posix(out)}"; WORK="{_posix(work)}"; META=http://127.0.0.1:9; EXPORT_FAIL=0; '
              f'. "{lib}"; ARTIFACT_MAX_BYTES=$1 export_phase p1; echo "EF=$EXPORT_FAIL"')
    big = subprocess.run([bash, "-c", script, "x", "1000"], capture_output=True, text=True)
    assert "ARTIFACT SIZE GUARD" in big.stdout and "EF=1" in big.stdout, big.stdout + big.stderr
    assert not (out / "phases" / "p1.tgz").exists()
    ok = subprocess.run([bash, "-c", script, "x", "4500000000"], capture_output=True, text=True)
    assert "EF=0" in ok.stdout and (out / "phases" / "p1.tgz").exists(), ok.stdout + ok.stderr


def test_loader_refuses_the_unlanded_safetensors_export_but_keeps_the_pt_path():
    from jobs.m2_first_cell.checkpoint_loader import CheckpointError, checkpoint_block

    c = _p(FDC)
    assert checkpoint_block(c)["member"].endswith(".pt")
    bad = {**c, "checkpoint": {**c["checkpoint"], "member": "./ckpt/infer-step0001439.safetensors"}}
    with pytest.raises(CheckpointError, match="safetensors"):
        checkpoint_block(bad)


def test_job_35e9v_lineage_cache_root_is_not_an_identity_but_repo_revision_and_weights_are():
    """job-35e9v (fulldev_eval_champ) exit 10: D1's checkpoint lineage recorded the snapshot under the platform's
    /hfcache mount; that run downloaded the SAME pinned snapshot to ~/.cache/huggingface/hub."""
    pytest.importorskip("torch")
    from train.lineage import canonical_name_or_path, diff

    rev = "ce3630b854cedbd673569f4600c6cf744bca9dba"
    d1 = f"/hfcache/models--RWKV--RWKV7-G1k-13.3B-20260930/snapshots/{rev}"
    run = f"/root/.cache/huggingface/hub/models--RWKV--RWKV7-G1k-13.3B-20260930/snapshots/{rev}"
    bb = {"spec": "rwkv7-13.3b", "class": "Rwkv7Model", "config_sha256": "c" * 64, "commit_hash": rev,
          "weights_digest": "f" * 64, "name_or_path": d1}
    saved = {"backbone": bb, "tokenizer": ["t"], "layout": {"l": 1}, "data": {"sha256": "d"}}
    assert diff(saved, {**saved, "backbone": {**bb, "name_or_path": run}}) == []  # the job-35e9v case
    assert canonical_name_or_path(run) == ("hf", "RWKV/RWKV7-G1k-13.3B-20260930", rev)
    other_rev = run.replace(rev, "0" * 40)
    other_repo = run.replace("RWKV7-G1k-13.3B", "RWKV7-G1k-2.9B")
    for bad in (other_rev, other_repo, "/some/local/dir", None):
        assert diff(saved, {**saved, "backbone": {**bb, "name_or_path": bad}}) == ["backbone.name_or_path"], bad
    for k, v in (("weights_digest", "0" * 64), ("commit_hash", "0" * 40), ("config_sha256", "0" * 64)):
        assert diff(saved, {**saved, "backbone": {**bb, "name_or_path": run, k: v}}) == [f"backbone.{k}"]
    assert diff(saved, {**saved, "data": {"sha256": "x"}}) == ["data"]

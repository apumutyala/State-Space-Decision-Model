"""jobs/backbone_screen MODE=qrwkv6_32b_v3w (readout-artifact check): the v3w-generic backend on the RWKV6-Qwen2 plug-in
(tiny random model with the REAL QRWKV6-32B vocabulary size and tokenizer, naive GLA on CPU): prefix fork == full
recompute, the engine's canary self-check and pinned identity, Decider end to end, the two-arm summarizer and the
runner's new mode."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/backbone_screen/screen_job.sh"
PARAMS = ROOT / "jobs/backbone_screen/job_params_qrwkv6_32b_v3w.json"

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS  # noqa: E402

TOK = DEFAULT_TOKENIZERS / BACKBONES["qrwkv6-32b"].tokenizer_dir
needs_tok = pytest.mark.skipif(not (TOK / "tokenizer.json").is_file(), reason=f"no local tokenizer at {TOK}")


@pytest.fixture(scope="module")
def tiny():
    import third_party.rwkv6qwen2_hf.modeling_rwkv6qwen2 as M
    from decide.backends.rwkv6qwen2 import compat_config
    from third_party.rwkv6qwen2_hf.configuration_rwkv6qwen2 import RWKV6Qwen2Config
    from tests.test_rwkv6qwen2_backend import naive_gla

    M.fused_recurrent_gla = naive_gla
    torch.manual_seed(0)
    cfg = RWKV6Qwen2Config(vocab_size=152064, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, lora_rank_tokenshift=8, lora_rank_decay=8,
                           max_position_embeddings=32768)
    compat_config(cfg)
    model = M.RWKV6Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def plugin(model):
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.rwkv6qwen2 import RWKV6StateOps

    def enc(text, trusted=True):
        raise AssertionError("the v3w-generic backend must tokenize with make_v3w's own encoder, never the plug-in's")

    return ModelPlugin("rwkv6qwen2", "tiny", None, enc,
                       lambda ids, n: model(input_ids=ids, use_cache=False, num_logits_to_keep=n).logits,
                       device="cpu", dtype="float32", max_context=16384,
                       state_ops=RWKV6StateOps(model, prefill_chunk=41))


@pytest.fixture(scope="module")
def v3w():
    from decide.v3w_generic import make_v3w

    return make_v3w("qrwkv6-32b")


@needs_tok
def test_identity_pinned_in_params(v3w):
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert v3w.identity()["sha256"] == p["screen"]["arm_b"]["v3w_sha256"]
    assert len(v3w.codes) == 255 and v3w.fmt.source == "chat_template"
    assert p["screen"]["expect_arm_a_index"] == 13.56 and p["screen"]["family"] == "qrwkv6-chatml-nothink"
    base = json.loads((ROOT / "jobs/backbone_screen/job_params_qrwkv6_32b.json").read_text(encoding="utf-8"))
    for k in ("repo", "revision", "family", "snapshot_pin", "input_cap", "parent_results", "parent_sha256", "allow_patterns"):
        assert p["screen"][k] == base["screen"][k], k  # arm a is job-6c27k's screen exactly
    assert p["dev_subset"] == base["dev_subset"] and p["submit"]["key"] != base["submit"]["key"]


@needs_tok
def test_symlinked_snapshot_root_same_identity(v3w, tmp_path):
    from decide.v3w_generic import make_v3w
    from jobs.common.readout_engine import v3w_tokenizers_root

    snap = tmp_path / "snapshots" / "2da855ca372cefab920716eca30f11ed65d6c708"
    snap.mkdir(parents=True)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOK / f, snap / f)
    try:
        root = v3w_tokenizers_root("qrwkv6-32b", snap)
    except OSError as e:  # Windows without symlink privilege; the job runs on Linux
        pytest.skip(f"symlink: {e}")
    assert make_v3w("qrwkv6-32b", tokenizers_root=root).identity() == v3w.identity()


@needs_tok
def test_fork_equals_recompute_and_selfcheck(tiny, v3w):
    from decide.backends.v3w_generic_backend import V3WGenericBackend
    from decide.prompt import CANARY_REQUEST
    from jobs.common.readout_engine import v3w_selfcheck

    b = V3WGenericBackend(plugin(tiny), v3w, pack_rows=2)
    rep = v3w_selfcheck(b)
    assert rep["ok"] and rep["options"] > 0 and rep["max_abs_delta"] < 1e-3, rep["max_abs_delta"]
    res = b.score(CANARY_REQUEST, lambda: None)
    qs = v3w.questions(CANARY_REQUEST)
    assert set(res.logits) == {q["qid"] for q in qs}
    for q in qs:
        assert list(res.logits[q["qid"]]) == q["options"]
    rev = b.revision()
    assert rev["prompt_template"] == v3w.identity() and rev["backend"] == "v3w_generic"
    assert rev["inference_config"]["v3w_generic"]["sha256"] == v3w.identity()["sha256"]


@needs_tok
def test_decider_end_to_end(tiny, v3w):
    from decide import Decider
    from decide.backends.v3w_generic_backend import V3WGenericBackend
    from decide.prompt import CANARY_REQUEST

    b = V3WGenericBackend(plugin(tiny), v3w)
    resp = Decider(b, enable_v0_1_deltas=True, backend_max_input_tokens=16384).decide(
        {"model": "qrwkv6-v3w-generic", "state": CANARY_REQUEST["state"], "questions": CANARY_REQUEST["questions"]})
    assert resp["ssdm"]["status"] == "ok", resp["ssdm"]
    assert set(resp["answers"]) == set(CANARY_REQUEST["questions"])


def test_engine_registers_family():
    from jobs.common.readout_engine import V3W_GENERIC_FAMILIES

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert p["screen"]["arm_b"]["family"] in V3W_GENERIC_FAMILIES


def _score(point, rows=644):
    return {"rows": rows, "subset_ids_sha256": "x", "slice_index": {"index": {"point": point, "lo": 1, "hi": 2, "se": 1}}}


def test_check_arm_a():
    from jobs.backbone_screen.summarize import check_arm_a

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert check_arm_a(p, _score(13.56))["ok"] and check_arm_a(p, _score(13.5649))["ok"]
    assert not check_arm_a(p, _score(13.58))["ok"] and not check_arm_a(p, _score(13.56, rows=600))["ok"]


def test_two_arm_decision(tmp_path):
    from jobs.backbone_screen.summarize import build, check_arm_a, main, valid_screen

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    ix = tmp_path / "index"
    (ix / "stock").mkdir(parents=True)
    (ix / "v3w").mkdir()
    cmp_ = {"diff": {"index": {"point": 5.0, "lo": 2.0, "hi": 8.0, "se": 1.5}}, "mde": {"mde": 4.0}}
    files = {"score.json": _score(13.56), "score_b.json": _score(20.1), "ece.json": {"ece": 0.1}, "ece_b.json": {"ece": 0.2},
             "compare_stock_v3w.json": cmp_, "compare_b_stock_v3w.json": cmp_, "compare_b_vs_a.json": cmp_,
             "selfcheck.json": {"ok": True}, "selfcheck_b.json": {"ok": True}, "arm_a_check.json": check_arm_a(p, _score(13.56))}
    for k, v in files.items():
        (ix / k).write_text(json.dumps(v), encoding="utf-8")
    for d in ("stock", "v3w"):
        (ix / d / "results.jsonl").write_text('{"status": "ok"}\n', encoding="utf-8")
    d = build(p, tmp_path)
    assert valid_screen(d) and d["arm_b"]["diff_vs_arm_a"]["point"] == 5.0 and d["arm_b"]["index"]["point"] == 20.1
    d["arm_b"]["arm_a_check"]["ok"] = False
    assert not valid_screen(d)
    assert main([str(PARAMS), str(tmp_path), str(tmp_path / "decision.json")]) == 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_runner_accepts_mode_and_has_arm_b():
    assert subprocess.run(["bash", "-n", str(RUNNER)]).returncode == 0
    r = subprocess.run(["bash", str(RUNNER)], env={"MODE": "nope", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 2 and "qrwkv6_32b_v3w" in r.stdout
    src = RUNNER.read_text(encoding="utf-8")
    i_chk, i_b = src.index("check-arm-a"), src.index("--option family=\"$FAMILY_B\"")
    assert i_chk < i_b and "fail 16" in src and "compare_b_vs_a.json" in src

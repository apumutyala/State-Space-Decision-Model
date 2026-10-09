"""jobs/backbone_screen MODE=qwen36_a3b (stock Qwen3.6-35B-A3B, m0 + v3w-generic in one load): a tiny random
Qwen3_5MoeForCausalLM (transformers' own qwen3_5_moe code: GDN + gated attention + 256-way-style MoE, torch GDN path on
CPU) with the REAL Qwen3.6 vocabulary size and tokenizer. Checks: transformers' HF cache fork (HFCacheStateOps) ==
full recompute under both readouts (the engine's canary self-checks), the v3w backend's KV state budget, the
one-load arm options and plug-in cache, the optional arm-a check, the params / pins, and the runner's new mode."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/backbone_screen/screen_job.sh"
PARAMS = ROOT / "jobs/backbone_screen/job_params_qwen36_a3b.json"
PIN = ROOT / "jobs/backbone_screen/snapshot_qwen36_a3b.sha256"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS  # noqa: E402

TOK = DEFAULT_TOKENIZERS / BACKBONES["qwen3.6-35b-a3b"].tokenizer_dir
needs_tok = pytest.mark.skipif(not (TOK / "tokenizer.json").is_file(), reason=f"no local tokenizer at {TOK}")
needs_moe = pytest.mark.skipif(not hasattr(transformers, "Qwen3_5MoeForCausalLM"), reason="transformers without qwen3_5_moe")


def params():
    return json.loads(PARAMS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(vocab_size=248320, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                               linear_key_head_dim=8, linear_value_head_dim=8, num_experts=8, num_experts_per_tok=2,
                               moe_intermediate_size=16, shared_expert_intermediate_size=16,
                               layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                               max_position_embeddings=32768)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def plugin(model, budget=131072):
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.state_fork import HFCacheStateOps
    from decide.v3w_generic import HFEncoder

    enc = HFEncoder(TOK)
    return ModelPlugin("qwen3.6-moe", "tiny", None, lambda text, trusted=True: enc(text, trusted),
                       lambda ids, n: model(input_ids=ids, logits_to_keep=n, use_cache=False).logits,
                       special_ids=enc.control_ids, device="cpu", dtype="float32", max_context=262144,
                       state_ops=HFCacheStateOps(model), fork_state_token_budget=budget)


@pytest.fixture(scope="module")
def v3w():
    from decide.v3w_generic import make_v3w

    return make_v3w("qwen3.6-35b-a3b")


@needs_tok
def test_identity_pinned_in_params(v3w):
    ident = v3w.identity()
    assert params()["screen"]["arm_b"]["v3w_sha256"] == ident["sha256"]
    assert v3w.fmt.no_think == "enable_thinking=False" and v3w.fmt.tail.endswith("<think>\n\n</think>\n\nANSWER:")
    assert len(v3w.codes) == 255


@needs_tok
def test_m0_template_is_the_models_own_no_think_turn():
    """jrt-chatml-v1 (the m0 readout) == Qwen3.6's own chat template with enable_thinking=False, around the body."""
    from decide.prompt import TEMPLATES
    from decide.v3w_generic import format_from_chat_template

    fmt = format_from_chat_template("qwen3.6-moe", TOK)
    t = TEMPLATES["jrt-chatml-v1"]
    assert fmt.head == t.head and fmt.tail == t.tail


@needs_tok
@needs_moe
def test_m0_fork_equals_recompute(tiny):
    from jobs.common.readout_engine import QWEN36_FORK_MAX_ROWS, TEMPLATES, QWEN36_M0, fork_selfcheck

    rep = fork_selfcheck(plugin(tiny), TEMPLATES[QWEN36_M0], QWEN36_FORK_MAX_ROWS, tol=1e-3)
    assert rep["ok"], {k: rep[k] for k in ("max_abs_delta", "same_argmax")}


@needs_tok
@needs_moe
@pytest.mark.parametrize("budget", [131072, 64])
def test_v3w_fork_equals_recompute_under_state_budget(tiny, v3w, budget):
    """The v3w backend packs at most state_budget // prefix rows per call; scores do not depend on the packing."""
    from decide.backends.v3w_generic_backend import V3WGenericBackend
    from decide.prompt import CANARY_REQUEST
    from jobs.common.readout_engine import v3w_selfcheck

    b = V3WGenericBackend(plugin(tiny, budget), v3w)
    assert b.state_budget == budget
    L, packs = b._calls(b._questions(CANARY_REQUEST))
    assert all(len(p) * L <= budget or len(p) == 1 for p in packs)
    rep = v3w_selfcheck(b, tol=1e-3)
    assert rep["ok"], {k: rep[k] for k in ("max_abs_delta", "same_argmax")}


def test_state_budget_split_is_pure():
    from decide.backends.v3w_generic_backend import V3WGenericBackend

    b = V3WGenericBackend.__new__(V3WGenericBackend)
    b.pack_rows, b.pack_tokens, b.state_budget = 16, 4096, 100
    qs = [{"qid": f"q{i}", "prompt": list(range(40)) + [100 + i, 7], "coded": True} for i in range(9)]
    L, packs = b._calls(qs)
    assert L == 40 and [len(p) for p in packs] == [2, 2, 2, 2, 1]
    b.state_budget = None
    assert [len(p) for p in b._calls(qs)[1]] == [9]


def test_engine_families_and_plugin_cache(monkeypatch):
    from decide.backends import hf_readout as H
    from jobs.common import readout_engine as E

    calls = []
    monkeypatch.setattr(H, "load_qwen35", lambda d, **k: calls.append((d, k)) or object())
    monkeypatch.setattr(E, "_HF_PLUGINS", {})
    a = E.build_plugin(E.QWEN36_M0, {"model_dir": "/snap", "state_budget": "131072"})
    b = E.build_plugin(E.QWEN36_V3W, {"model_dir": "/snap", "state_budget": "131072"})
    assert a is b and len(calls) == 1
    assert calls[0][1] == {"device": "cuda", "dtype": "bfloat16", "family": "qwen3.6-moe", "state_budget": 131072}
    assert E.TEMPLATES[E.QWEN36_M0] == "jrt-chatml-v1" and E.TEMPLATES["qwen3.5"] == "jrt-chatml-v1"


def test_arm_options(tmp_path):
    from jobs.backbone_screen.arms import arm_options

    arms = arm_options(params(), "/snap", "Qwen/Qwen3.6-35B-A3B", tmp_path)
    assert [n for n, _ in arms] == ["stock", "v3w"]
    a, b = arms[0][1], arms[1][1]
    assert a["family"] == "qwen3.6-moe-m0" and b["family"] == "qwen3.6-moe-v3w-generic"
    for o in (a, b):
        assert o["model_dir"] == "/snap" and o["input_cap"] == "16384" and o["state_budget"] == "131072"
    assert b["backbone_id"] == "qwen3.6-35b-a3b" and len(b["v3w_sha256"]) == 64
    assert a["selfcheck_out"].endswith("selfcheck.json") and b["selfcheck_out"].endswith("selfcheck_b.json")


def test_check_arm_a_without_target():
    from jobs.backbone_screen.summarize import check_arm_a

    sc = {"slice_index": {"index": {"point": 40.0}}, "rows": 644}
    rep = check_arm_a(params(), sc)
    assert rep["ok"] and rep["expected"] is None
    assert not check_arm_a(params(), {**sc, "rows": 600})["ok"]


def test_params_pins_and_schedule():
    from jobs.common import costs

    p = params()
    s = p["screen"]
    assert p["chip"] == "h100" and p["chip_count"] == 1 and p["mode"] == "qwen36_a3b"
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"] == 90
    assert costs.quoted_max_cost_usd(p) <= 12 and costs.worst_case_usd(p) <= p["hard_cap_usd"]
    pin = dict(reversed(ln.split("  ", 1)) for ln in PIN.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert len(pin) == 34 and sum(k.startswith("model-") for k in pin) == 26
    import fnmatch

    assert all(any(fnmatch.fnmatch(f, pat) for pat in s["allow_patterns"]) for f in pin)
    assert s["code_pkg"] == "" and s["one_load_arms"] is True
    assert "jobs/backbone_screen/snapshot_qwen36_a3b.sha256" in p["context"]["manifest_files"]


def test_runner_accepts_mode_and_one_load():
    text = RUNNER.read_text(encoding="utf-8")
    assert "qwen36_a3b)" in text and "jobs.backbone_screen.arms" in text and 'if [ -n "$PKG" ]' in text
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(RUNNER)], capture_output=True).returncode == 0

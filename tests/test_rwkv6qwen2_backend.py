"""decide.backends.rwkv6qwen2 (backbone scout): the vendored RWKV6-Qwen2 code on transformers 5.17, its RWKV6State fork
ops == full recompute (fp32 CPU, tiny random model; the fla GLA kernel replaced by its naive recurrence), chunked
prefill == one-shot prefill, the engine's fork self-check, and the vendored-code pins."""
import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/backbone_screen/screen_job.sh"

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def naive_gla(q, k, v, gk=None, gv=None, scale=None, initial_state=None, output_final_state=False, **_):
    """fla.ops.gla.fused_recurrent_gla reference: h_t = diag(exp(gk_t)) h_{t-1} + k_t^T v_t, o_t = (scale q_t) h_t."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    h = torch.zeros(B, H, K, V, dtype=torch.float32) if initial_state is None else initial_state.float().clone()
    out = []
    for t in range(T):
        h = h * gk[:, t].float().exp()[..., None] + k[:, t].float()[..., None] * v[:, t].float()[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t].float() * scale, h))
    return torch.stack(out, 1).to(v.dtype), (h if output_final_state else None)


@pytest.fixture(scope="module")
def tiny():
    import third_party.rwkv6qwen2_hf.modeling_rwkv6qwen2 as M
    from decide.backends.rwkv6qwen2 import compat_config
    from third_party.rwkv6qwen2_hf.configuration_rwkv6qwen2 import RWKV6Qwen2Config

    M.fused_recurrent_gla = naive_gla  # the Triton kernel is GPU-only; same math
    torch.manual_seed(0)
    cfg = RWKV6Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, lora_rank_tokenshift=8, lora_rank_decay=8,
                           max_position_embeddings=4096)
    compat_config(cfg)
    model = M.RWKV6Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def plugin(model, chunk=8192):
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.rwkv6qwen2 import RWKV6StateOps

    return ModelPlugin("rwkv6qwen2", "tiny", None, lambda text, trusted=True: list(text.encode("utf-8")),
                       lambda ids, n: model(input_ids=ids, use_cache=False, num_logits_to_keep=n).logits,
                       device="cpu", dtype="float32", state_ops=RWKV6StateOps(model, prefill_chunk=chunk))


def test_attention_free(tiny):
    assert {type(layer.self_attn).__name__ for layer in tiny.model.layers} == {"RWKV6Attention"}
    src = (ROOT / "third_party/rwkv6qwen2_hf/modeling_rwkv6qwen2.py").read_text(encoding="utf-8")
    assert "softmax" not in src.split("class RWKV6Attention")[1].split("class RWKV6Qwen2DecoderLayer")[0]


@pytest.mark.parametrize("template", ["jrt-chatml-v1", "nothink"])
def test_fork_equals_recompute(tiny, template):
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.backends.rwkv6qwen2 import CHATML_NOTHINK
    from decide.prompt import CANARY_REQUEST

    tpl = CHATML_NOTHINK if template == "nothink" else template
    plug = plugin(tiny, chunk=37)  # chunked prefill on the fork path
    fk = HFReadoutBackend(plug, tpl, fork=True, fork_max_rows=3).score(CANARY_REQUEST, lambda: None)
    rc = HFReadoutBackend(plug, tpl, fork=False).score(CANARY_REQUEST, lambda: None)
    assert fk.extras["scoring_path"] == "prefix-fork" and rc.extras["scoring_path"] == "recompute"
    for q in rc.logits:
        for o in rc.logits[q]:
            assert abs(fk.logits[q][o] - rc.logits[q][o]) < 1e-3, (q, o)


def test_chunked_prefill_and_extend_does_not_mutate(tiny):
    from decide.backends.rwkv6qwen2 import RWKV6StateOps

    ids = torch.randint(0, 256, (1, 50))
    whole, s1 = RWKV6StateOps(tiny, prefill_chunk=8192).prefill(ids)
    parts, s2 = RWKV6StateOps(tiny, prefill_chunk=16).prefill(ids)
    assert torch.allclose(whole, parts, atol=1e-4)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(s1.layer_kv_states, s2.layer_kv_states))
    before = copy.deepcopy(s1)
    ops = RWKV6StateOps(tiny)
    lg, st = ops.extend(s1, torch.randint(0, 256, (3, 5)), 5)
    assert lg.shape == (3, 5, 256) and st.layer_kv_states[0].shape[0] == 3
    assert all(torch.equal(a, b) for a, b in zip(before.layer_kv_states, s1.layer_kv_states))
    assert ops.select(st, 1).layer_kv_states[0].shape[0] == 1


def test_from_pretrained_roundtrip(tiny, tmp_path):
    """The GPU loader's path (config.json -> compat_config -> from_pretrained, loading info checked) on CPU."""
    from decide.backends.rwkv6qwen2 import load_model

    tiny.save_pretrained(tmp_path)
    m2 = load_model(tmp_path, device="cpu", dtype="float32")
    ids = torch.randint(0, 256, (2, 9))
    assert torch.allclose(tiny(input_ids=ids, num_logits_to_keep=3).logits, m2(input_ids=ids, num_logits_to_keep=3).logits,
                          atol=1e-5)
    from safetensors.torch import load_file, save_file

    f = next(tmp_path.glob("*.safetensors"))
    sd = load_file(f)
    sd["model.layers.0.self_attn.renamed"] = sd.pop("model.layers.0.self_attn.time_decay")
    save_file(sd, f, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="do not match"):
        load_model(tmp_path, device="cpu", dtype="float32")


def test_engine_selfcheck(tiny):
    from jobs.common.readout_engine import QRWKV6_FAMILIES, TEMPLATES, fork_selfcheck, template_of

    assert {TEMPLATES[f] for f in QRWKV6_FAMILIES} == {"jrt-chatml-v1", "jrt-chatml-nothink-v1"}
    assert template_of("qrwkv6-chatml-nothink").id == "jrt-chatml-nothink-v1"
    rep = fork_selfcheck(plugin(tiny), template_of("qrwkv6-chatml"), 4)
    assert rep["ok"] and rep["same_argmax"] and rep["max_abs_delta"] < 1e-3


def test_vendored_pins(tmp_path):
    import third_party.rwkv6qwen2_hf as V

    assert all(V.vendored_ok().values())
    for f in V.CODE_SHA256:
        shutil.copy(V.HERE / f, tmp_path / f)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        V.verify_snapshot_code(tmp_path, "featherless-ai/QRWKV-QwQ-32B")
    with pytest.raises(RuntimeError):
        V.verify_snapshot_code(tmp_path, "someone/else")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_runner_parses_and_refuses_unknown_mode():
    assert subprocess.run(["bash", "-n", str(RUNNER)]).returncode == 0
    r = subprocess.run(["bash", str(RUNNER)], env={"MODE": "nope", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 2 and "qrwkv6_32b" in r.stdout

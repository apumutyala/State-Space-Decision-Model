"""decide.backends.rwkv07c (backbone scout, OpenMOSE RWKV-Qwen3-30B-A3B): the vendored RWKV07C code on transformers 5.17
(tiny random MoE model, fla's rwkv7 recurrence replaced by its naive form), the per-expert checkpoint loaded through
the registered qwen3_moe conversion (== the model that wrote it), RoPE applied, fork == recompute, pins."""
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def naive_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=True, **kw):
    """fla fused_recurrent_rwkv7 (dplr) per step: h = exp(w) h + b (a . h) + k v^T, o = scale r . h."""
    assert "head_first" not in kw, "install_compat must drop head_first before fla sees it"
    B, T, H, K = r.shape
    V = v.shape[-1]
    h = torch.zeros(B, H, K, V) if initial_state is None else initial_state.float().clone()
    out = []
    for t in range(T):
        ah = torch.einsum("bhk,bhkv->bhv", a[:, t].float(), h)
        h = w[:, t].float().exp()[..., None] * h + b[:, t].float()[..., None] * ah[:, :, None, :]
        h = h + k[:, t].float()[..., None] * v[:, t].float()[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", r[:, t].float() * scale, h))
    return torch.stack(out, 1).to(v.dtype), (h if output_final_state else None)


def tiny_config():
    from third_party.rwkv07c_hf.configuration_rwkv07c import RWKV07CConfig

    return RWKV07CConfig(vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, lora_rank_decay=8, lora_rank_iclr=8,
                         lora_rank_value_residual_mix=8, lora_rank_key_residual_mix=8, lora_rank_gate=8,
                         max_position_embeddings=4096, use_rope=True, rope_theta=10000.0, rope_scaling=None,
                         moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                         decoder_sparse_step=1, mlp_only_layers=[], transformer_layers=[-1], rwkv_layers=[0, 1],
                         enable_qk_norm=True, tie_word_embeddings=False, attention_bias=False)


@pytest.fixture(scope="module")
def tiny():
    from decide.backends.rwkv07c import compat_config

    import third_party.rwkv07c_hf.modeling_rwkv07c as M

    M.fused_recurrent_rwkv7 = naive_rwkv7  # Triton is GPU-only
    cfg = compat_config(tiny_config())  # wraps naive_rwkv7 with the head_first drop, as on the GPU
    torch.manual_seed(0)
    model = M.RWKV07CForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model


def plugin(model, chunk=8192):
    from decide.backends.hf_readout import ModelPlugin
    from decide.backends.rwkv6qwen2 import RWKV6StateOps

    return ModelPlugin("rwkv07c", "tiny", None, lambda text, trusted=True: list(text.encode("utf-8")),
                       lambda ids, n: model(input_ids=ids, use_cache=False, logits_to_keep=n).logits,
                       device="cpu", dtype="float32",
                       state_ops=RWKV6StateOps(model, prefill_chunk=chunk, logits_kw="logits_to_keep"))


def test_attention_free_and_rope(tiny):
    assert {type(layer.self_attn).__name__ for layer in tiny.model.layers} == {"RWKV07CAttention"}
    assert tiny.config.use_rope
    ids = torch.randint(0, 256, (1, 12))
    a = tiny(input_ids=ids, use_cache=False).logits
    tiny.config.use_rope = False
    try:
        b = tiny(input_ids=ids, use_cache=False).logits
    finally:
        tiny.config.use_rope = True
    assert not torch.allclose(a, b)  # positions matter: the RoPE path is live


def test_fork_equals_recompute(tiny):
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.backends.rwkv6qwen2 import CHATML_NOTHINK
    from decide.prompt import CANARY_REQUEST

    plug = plugin(tiny, chunk=29)
    fk = HFReadoutBackend(plug, CHATML_NOTHINK, fork=True, fork_max_rows=3).score(CANARY_REQUEST, lambda: None)
    rc = HFReadoutBackend(plug, CHATML_NOTHINK, fork=False).score(CANARY_REQUEST, lambda: None)
    assert fk.extras["scoring_path"] == "prefix-fork"
    for q in rc.logits:
        for o in rc.logits[q]:
            assert abs(fk.logits[q][o] - rc.logits[q][o]) < 1e-3, (q, o)


def test_per_expert_checkpoint_loads_through_conversion(tiny, tmp_path):
    """The real checkpoint stores mlp.experts.<e>.{gate,up,down}_proj.weight; 5.x fuses them. Write that layout by
    hand and load it with the GPU loader: same logits as the model whose weights were written."""
    from safetensors.torch import save_file

    from decide.backends.rwkv07c import load_model

    sd, out = tiny.state_dict(), {}
    for k, v in sd.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            pre, inter = k[: -len("gate_up_proj")], v.shape[1] // 2
            for e in range(v.shape[0]):
                out[f"{pre}{e}.gate_proj.weight"] = v[e, :inter].contiguous()
                out[f"{pre}{e}.up_proj.weight"] = v[e, inter:].contiguous()
        elif k.endswith("mlp.experts.down_proj"):
            pre = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                out[f"{pre}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            out[k] = v.contiguous()
    assert any(".experts.0.gate_proj.weight" in k for k in out)
    save_file(out, tmp_path / "model.safetensors", metadata={"format": "pt"})
    tiny.config.save_pretrained(tmp_path)
    m2 = load_model(tmp_path, device="cpu", dtype="float32")
    ids = torch.randint(0, 256, (2, 9))
    assert torch.allclose(tiny(input_ids=ids, logits_to_keep=3).logits, m2(input_ids=ids, logits_to_keep=3).logits, atol=1e-5)


def test_head_first_true_refused():
    from decide.backends.rwkv07c import _drop_head_first

    f = _drop_head_first(lambda *a, **k: ("ok", k))
    assert f(1, head_first=False, scale=1.0) == ("ok", {"scale": 1.0})
    with pytest.raises(ValueError):
        f(1, head_first=True)


def test_engine_family_and_pins(tmp_path):
    import third_party.rwkv07c_hf as V
    from jobs.common.readout_engine import TEMPLATES, template_of

    assert TEMPLATES["rwkv07c-chatml-nothink"] == "jrt-chatml-nothink-v1"
    assert template_of("rwkv07c-chatml-nothink").id == "jrt-chatml-nothink-v1"
    assert all(V.vendored_ok().values())
    for f in V.CODE_SHA256:
        shutil.copy(V.HERE / f, tmp_path / f)
    (tmp_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        V.verify_snapshot_code(tmp_path)

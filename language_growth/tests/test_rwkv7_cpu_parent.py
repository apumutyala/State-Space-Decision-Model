"""No checkpoint loads: independent FLA-name FP32 model verifies conversion."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers.initialization import no_init_weights
from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM
from language_growth.rwkv7_cpu_parent import conversion_spec, convert_tensor
from language_growth import l0_rwkv_cpu_probe as P


def fla_reference(ids, source, c=24, head=8, layers=2):
    """Unpadded, no-cache FLA 0.5.2 equations; key/value state orientation.

    The DPLR update is transcribed from fla-core's naive.py with scale=1 as
    chunk_rwkv7 requests. No vendored attention/FFN/WKV functions are called.
    """
    norm = lambda x, p: F.layer_norm(x, (c,), source[p+"weight"], source[p+"bias"], 1e-5)
    linear = lambda x, p: F.linear(x, source[p+"weight"], source.get(p+"bias"))
    x = F.embedding(ids, source["model.embeddings.weight"])
    first = None
    for i in range(layers):
        p = f"model.layers.{i}."
        if i == 0:
            x = norm(x, p+"pre_norm.")
        h = norm(x, p+"attn_norm.")
        delta = F.pad(h, (0, 0, 1, -1)) - h
        mixed = {name: h + delta*source[p+"attn.x_"+name] for name in "rwkvag"}
        def lora(name, activation):
            v = linear(mixed[name if name != "a" else "a"], p+f"attn.{name}_lora.lora.0.")
            return linear(activation(v), p+f"attn.{name}_lora.lora.2.")
        r = linear(mixed["r"], p+"attn.r_proj.")
        k = linear(mixed["k"], p+"attn.k_proj.")
        v = linear(mixed["v"], p+"attn.v_proj.")
        w = -0.6065306597126334*lora("w", torch.tanh).sigmoid()
        a = lora("a", lambda t: t).sigmoid()
        g = lora("g", torch.sigmoid)
        if i == 0:
            first = v
        else:
            v = torch.lerp(v, first, lora("v", lambda t: t).sigmoid())
        B, T, _ = k.shape
        heads = lambda t: t.reshape(B, T, c//head, head)
        kk = F.normalize(heads(k*source[p+"attn.k_k"]), dim=-1)
        k = k.addcmul(k*(a-1), source[p+"attn.k_a"])
        r, w, k, v, a = map(heads, (r, w, k, v, a))
        state = torch.zeros(B, c//head, head, head)
        outs = []
        for t in range(T):
            alpha, beta = -kk[:, t], kk[:, t]*a[:, t]
            update = k[:, t, :, :, None]*v[:, t, :, None, :]
            update = update + beta[..., None]*(state*alpha[..., None]).sum(-2, keepdim=True)
            state = state*w[:, t].exp()[..., None] + update
            outs.append((r[:, t, :, :, None]*state).sum(-2))
        o = torch.stack(outs, 1).reshape(B*T, c)
        o = F.group_norm(o, c//head, source[p+"attn.g_norm.weight"], source[p+"attn.g_norm.bias"], head*1e-5).reshape(B, T, c)
        o = (o+((r*k*source[p+"attn.r_k"]).sum(-1, keepdim=True)*v).reshape(B,T,c))*g
        x = x+linear(o, p+"attn.o_proj.")
        h = norm(x, p+"ffn_norm.")
        shifted = h+(F.pad(h, (0,0,1,-1))-h)*source[p+"ffn.x_k"]
        x = x+linear(linear(shifted, p+"ffn.key.").relu().square(), p+"ffn.value.")
    return F.linear(norm(x, "model.norm."), source["lm_head.weight"])


@pytest.mark.parametrize("length", [1, 19, 65])
def test_conversion_matches_independent_fla_equations_and_eager(length):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    spec = conversion_spec(24, 2, 64, 8, 48, 4, 4, 4, 4)
    source = {k: torch.randn(shape)*0.1 for k, (_, _, shape) in spec.items()}
    for k in source:
        if k.endswith("norm.weight") or k.endswith("g_norm.weight"):
            source[k] = torch.ones_like(source[k])
    config = Rwkv7Config(vocab_size=64, hidden_size=24, head_dim=8, num_heads=3,
        num_hidden_layers=2, intermediate_size=48, decay_low_rank_dim=4,
        a_low_rank_dim=4, v_low_rank_dim=4, gate_low_rank_dim=4)
    with no_init_weights():
        model = Rwkv7ForCausalLM(config).eval()
    mapped = {target: convert_tensor(source[k], transform) for k, (target, transform, _) in spec.items()}
    for k, val in model.state_dict().items():
        if k not in mapped:
            assert k in {"rwkv7.blocks.0.att.v0", "rwkv7.blocks.0.att.v1", "rwkv7.blocks.0.att.v2"}
            mapped[k] = torch.zeros_like(val)
    model.load_state_dict(mapped, strict=True)
    ids = torch.randint(1, 64, (2, length))
    with torch.inference_mode():
        reference = fla_reference(ids, source)
        got = model(input_ids=ids, use_cache=False).logits
        model.config.wkv_implementation = "eager"
        eager = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(got, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got, eager, atol=2e-5, rtol=2e-5)


def test_rwkv_specific_guard_does_not_lower_qwen_guard():
    from language_growth.l0_qwen_cpu_probe import MIN_START_AVAILABLE
    assert MIN_START_AVAILABLE == 8*1024**3 and P.MIN_START_AVAILABLE == 6*1024**3
    assert P.MIN_RUNNING_AVAILABLE == 3*1024**3 and P.TIMEOUT_SECONDS == 300


@pytest.mark.parametrize("available, elapsed, result", [(2*1024**3, 1, 125), (7*1024**3, 301, 124)])
def test_owned_child_stops_on_low_memory_or_timeout(available, elapsed, result):
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d)/"run"/"raw.jsonl"
        child = MagicMock()
        child.poll.return_value = None
        with patch("psutil.virtual_memory", side_effect=[SimpleNamespace(available=7*1024**3),
             SimpleNamespace(available=available)]), patch.object(P.subprocess, "Popen", return_value=child), \
             patch.object(P.time, "monotonic", side_effect=[0, elapsed]):
            assert P._parent(raw) == result
        child.kill.assert_called_once()
        child.wait.assert_called_once()


def test_child_rejects_before_importing_torch_or_loading_weights():
    with patch.dict(P.os.environ, P._offline_env()), \
         patch("psutil.virtual_memory", return_value=SimpleNamespace(available=5*1024**3)), \
         patch.object(P, "inspect_snapshot") as inspect:
        with pytest.raises(RuntimeError, match="before Torch"):
            P._child(Path("not-created/raw.jsonl"))
        inspect.assert_not_called()


def test_monitor_failure_stops_owned_child():
    with tempfile.TemporaryDirectory() as d:
        child = MagicMock()
        child.poll.return_value = None
        with patch("psutil.virtual_memory", side_effect=[SimpleNamespace(available=7*1024**3),
             RuntimeError("monitor failed")]), patch.object(P.subprocess, "Popen", return_value=child), \
             patch.object(P.time, "monotonic", return_value=0):
            with pytest.raises(RuntimeError, match="monitor failed"):
                P._parent(Path(d)/"run"/"raw.jsonl")
        child.kill.assert_called_once()
        child.wait.assert_called_once()

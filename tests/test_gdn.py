"""Numerical checks for the GDN kernels and module.

Run: .venv-fla/Scripts/python.exe -m tests.test_gdn   (writes a JSON summary to stdout)
1. recurrent (ours) vs HF torch_recurrent_gated_delta_rule  (forward)
2. chunk (ours) vs recurrent (ours)                         (forward + grads, fp32, CPU and CUDA)
3. fla Triton kernel vs recurrent (ours)                    (forward + grads, bf16 inputs, CUDA)
4. GDN module vs HF Qwen3_5GatedDeltaNet on real Qwen3.5-0.8B layer-0 weights (forward), all kernels
"""
from __future__ import annotations

import json
import sys

import torch

from convert.gdn import GDN, fla_available, gdr_chunk, gdr_fla, gdr_recurrent


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def rand_inputs(B=2, T=200, H=4, D=32, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    gl = -torch.nn.functional.softplus(mk(B, T, H)) * 0.3
    beta = torch.sigmoid(mk(B, T, H))
    return [x.to(device=device, dtype=dtype if x.dim() == 4 or x is beta else torch.float32).requires_grad_(True)
            for x in (q, k, v, gl, beta)]


def grads(fn, inputs):
    for x in inputs:
        x.grad = None
    o, S = fn(*inputs)
    (o.float().sin().sum() + S.float().cos().sum()).backward()
    return o.detach(), S.detach(), [x.grad.detach().clone() for x in inputs]


def main():
    out = {}
    # 1. ours-recurrent vs HF recurrent
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    hf_rec = getattr(M.torch_recurrent_gated_delta_rule, "__wrapped__", M.torch_recurrent_gated_delta_rule)
    x = rand_inputs(T=96)
    o_ref, S_ref = gdr_recurrent(*x)
    o_hf, S_hf = hf_rec(*[t.detach() for t in x], output_final_state=True, use_qk_l2norm_in_kernel=True)
    out["recurrent_vs_hf"] = {"o_rel": rel(o_ref, o_hf), "S_rel": rel(S_ref, S_hf)}

    # 2. chunk vs recurrent (fp32), CPU + CUDA, including T not multiple of chunk
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        x = rand_inputs(T=200, device=dev)
        o1, S1, g1 = grads(gdr_recurrent, x)
        o2, S2, g2 = grads(lambda *a: gdr_chunk(*a, chunk_size=64), x)
        out[f"chunk_vs_recurrent_{dev}"] = {"o_rel": rel(o2, o1), "S_rel": rel(S2, S1),
                                            "grad_rel_max": max(rel(a, b) for a, b in zip(g2, g1))}

    # 3. fla vs recurrent
    if fla_available():
        x32 = rand_inputs(T=300, H=4, D=64, device="cuda")
        o1, S1, g1 = grads(gdr_recurrent, x32)
        xb = [t.detach().clone().to(torch.bfloat16 if i in (0, 1, 2, 4) else torch.float32).requires_grad_(True)
              for i, t in enumerate(x32)]
        o2, S2, g2 = grads(gdr_fla, xb)
        out["fla_vs_recurrent_cuda_bf16in"] = {"o_rel": rel(o2, o1), "S_rel": rel(S2, S1),
                                               "grad_rel": [rel(a, b) for a, b in zip(g2, g1)]}
    else:
        out["fla_vs_recurrent_cuda_bf16in"] = "fla unavailable"

    # 4. module vs HF layer on real weights
    from convert.model import resolve_model_dir
    from transformers import AutoConfig

    d = resolve_model_dir(None)
    cfg = AutoConfig.from_pretrained(d).text_config
    from safetensors import safe_open

    sd = {}
    shard = [f for f in __import__("os").listdir(d) if f.endswith(".safetensors")][0]
    with safe_open(__import__("os").path.join(d, shard), "pt") as f:
        pre = [k for k in f.keys() if ".layers.0.linear_attn." in k and "language_model" in k]
        for k in pre:
            sd[k.split("linear_attn.")[1]] = f.get_tensor(k)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hf = M.Qwen3_5GatedDeltaNet(cfg, 0).to(dev, torch.float32)
    hf.load_state_dict(sd)
    g = torch.Generator().manual_seed(1)
    h = (torch.randn(2, 333, cfg.hidden_size, generator=g) * 0.5).to(dev)
    with torch.no_grad():
        ref = hf(h)
        res = {}
        for kern in ["recurrent", "chunk"] + (["fla"] if fla_available() else []):
            mine = GDN.from_config(cfg, kernel=kern).to(dev, torch.float32)
            mine.load_state_dict(sd)
            if kern == "fla":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = mine(h)
            else:
                y = mine(h)
            res[kern] = rel(y, ref)
    import importlib.util
    hf_backend = "fla" if importlib.util.find_spec("fla") is not None else "torch"
    out["module_vs_hf_layer0_real_weights"] = res
    out["hf_reference_backend"] = hf_backend  # HF auto-selects fla when importable

    tol = {"fp32": 1e-4, "bf16": 3e-2}
    mod_tol = tol["fp32"] if hf_backend == "torch" else tol["bf16"]
    ok = (out["recurrent_vs_hf"]["o_rel"] < tol["fp32"]
          and all(v["o_rel"] < tol["fp32"] and v["grad_rel_max"] < 1e-3
                  for k, v in out.items() if k.startswith("chunk_vs"))
          and res["recurrent"] < mod_tol and res["chunk"] < mod_tol
          and (res.get("fla", 0) < tol["bf16"])
          and (not isinstance(out["fla_vs_recurrent_cuda_bf16in"], dict)
               or (out["fla_vs_recurrent_cuda_bf16in"]["o_rel"] < tol["bf16"]
                   and max(out["fla_vs_recurrent_cuda_bf16in"]["grad_rel"]) < 5e-2)))
    out["PASS"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

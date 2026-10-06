"""CPU-only shape/mapping test of init_gdn for the 0.8B and 9B head layouts (random small weights, one layer).

The head structure is the real one (0.8B: attention 8x256 / KV 2, GDN 16x128 keys + 16x128 values;
9B: attention 16x256 / KV 4, GDN 16x128 keys + 32x128 values); hidden size is shrunk to 512 so it runs in seconds.
Run: python -m tests.test_init_shapes
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from convert.model import init_gdn, reuse_attn_weights

LAYOUTS = {
    "0p8b": dict(num_attention_heads=8, num_key_value_heads=2, head_dim=256, linear_num_key_heads=16,
                 linear_num_value_heads=16, linear_key_head_dim=128, linear_value_head_dim=128),
    "9b": dict(num_attention_heads=16, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
               linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128),
}


def fake_model(layout, hidden=64, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(hidden_size=hidden, intermediate_size=128, num_hidden_layers=2, vocab_size=128,
                            layer_types=["linear_attention", "full_attention"], linear_conv_kernel_dim=4,
                            **LAYOUTS[layout])
    gdn0 = M.Qwen3_5GatedDeltaNet(cfg, 0)
    attn = M.Qwen3_5Attention(cfg, 1)
    for p in list(gdn0.parameters()) + list(attn.parameters()):
        torch.nn.init.normal_(p, std=0.05)
    model = SimpleNamespace(cfg=cfg, kernel="chunk", swaps={1: SimpleNamespace(attn=attn)},
                            lm=SimpleNamespace(layers=[SimpleNamespace(linear_attn=gdn0), None]))
    return model, attn, cfg


def check(layout):
    # hidden 512 > head_dim so the 9B 256->128 key projection is a real (lossy) compression
    model, attn, cfg = fake_model(layout, hidden=512)
    H, hd, KV = cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads
    nk, dk, nv, dv = (cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_num_value_heads,
                      cfg.linear_value_head_dim)
    res = {}
    x = torch.randn(2, 37, cfg.hidden_size)
    for how in ("random", "clone_neighbor", "reuse_attn"):
        g = init_gdn(model, 1, how)
        y = g(x)
        assert y.shape == x.shape and torch.isfinite(y).all(), (layout, how)
        assert g.in_proj_qkv.weight.shape == (2 * nk * dk + nv * dv, cfg.hidden_size)
        res[how] = {"out_shape": list(y.shape), "finite": True}
    W = reuse_attn_weights(attn, cfg, nk, dk, nv, dv)
    res["reuse_info"] = W["info"]
    s = hd // dv
    Wq_all = attn.q_proj.weight.view(H, 2 * hd, -1)
    Wv = attn.v_proj.weight.view(KV, hd, -1)
    # value head j <- half j % s of attention head j // s (via GQA kv head (j // s) // (H // KV))
    for j in range(nv):
        h = j // s
        sl = slice((j % s) * dv, (j % s + 1) * dv)
        assert torch.equal(W["v"][j * dv:(j + 1) * dv], Wv[h // (H // KV), sl]), (layout, "v", j)
        assert torch.equal(W["z"][j * dv:(j + 1) * dv], Wq_all[h, hd:][sl]), (layout, "z", j)
    assert torch.equal(W["o"], attn.o_proj.weight)
    # key/value pairing as GDN.forward sees it: value head j reads key head j // (nv // nk); must be attention head j//s
    r = nv // nk
    if W["info"]["key_map"] == "split":
        qn = 1.0 + attn.q_norm.weight
        for i in range(nk):
            h, sl = i // s, slice((i % s) * dk, (i % s + 1) * dk)
            assert torch.allclose(W["q"][i * dk:(i + 1) * dk], (Wq_all[h, :hd] * qn[:, None])[sl])
        assert all(j // r == j // s for j in range(nv)) or r == 1
    else:
        assert all(j // r == j // s for j in range(nv)), "proj layout needs value head j -> key head j//s"
        # q'.k' approximates q.k per head on random inputs (JL-style subspace check)
        qn, kn = 1.0 + attn.q_norm.weight, 1.0 + attn.k_norm.weight
        xs = torch.randn(256, cfg.hidden_size)
        cors = []
        for h in range(H):
            q = xs @ (Wq_all[h, :hd] * qn[:, None]).T
            k = xs @ (attn.k_proj.weight.view(KV, hd, -1)[h // (H // KV)] * kn[:, None]).T
            q2 = xs @ W["q"][h * dk:(h + 1) * dk].T
            k2 = xs @ W["k"][h * dk:(h + 1) * dk].T
            a, b = (q @ k.T).flatten(), (q2 @ k2.T).flatten()
            cors.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
        res["qk_score_corr_mean"] = sum(cors) / len(cors)
        res["qk_score_corr_min"] = min(cors)
        assert min(cors) > 0.5, cors
    return res


def main():
    torch.set_grad_enabled(False)
    out = {k: check(k) for k in LAYOUTS}
    # an unsupported layout must fail loudly, not silently mis-map
    bad = dict(LAYOUTS["9b"], linear_num_key_heads=8)
    model, attn, cfg = fake_model("9b")
    try:
        reuse_attn_weights(attn, cfg, 8, 128, 32, 128)
        out["unsupported_layout_raises"] = False
    except ValueError:
        out["unsupported_layout_raises"] = True
    _ = bad
    out["PASS"] = out["unsupported_layout_raises"]
    print(json.dumps(out, indent=1))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

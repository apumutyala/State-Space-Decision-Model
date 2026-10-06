"""CPU tests: micro-batching with gradient accumulation equals the full-batch gradient (stage 1 and stage 2), the
shared-prefix stage-2 path equals the plain full forward, and the split forward equals ``hidden``.

Tiny random Qwen3.5 text model (4 layers: GDN, attention, GDN, attention; hidden 64), fp32, pure-PyTorch GDN kernel.
Run: python -m tests.test_microbatch   (or pytest)
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from convert import align
from convert.model import ConvertibleQwen, attach_gdn, init_gdn, replace_native_gdn

TOL = 2e-4


def tiny_model(seed=0):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as Mq
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=4, vocab_size=101,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    lm = Mq.Qwen3_5TextModel(cfg).float()
    for p in lm.parameters():
        torch.nn.init.normal_(p, std=0.05)
    head = nn.Linear(64, 101, bias=False)
    full = SimpleNamespace(model=SimpleNamespace(language_model=lm), lm_head=head)
    m = ConvertibleQwen(full, kernel="chunk")
    replace_native_gdn(m, kernel="chunk")
    for p in list(lm.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    for i in m.attn_layers:
        attach_gdn(m, i, init_gdn(m, i, "random", seed))
    m.eval()
    return m


def grads(model, layers):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for i in layers for p in model.swaps[i].gdn.parameters()]


def zero(model):
    for sw in model.swaps.values():
        if sw.gdn is not None:
            sw.gdn.zero_grad(set_to_none=True)


def max_rel(a, b):
    num = max(float((x - y).abs().max()) for x, y in zip(a, b))
    den = max(float(y.abs().max()) for y in b)
    return num / max(den, 1e-12)


X = torch.randint(0, 101, (4, 24), generator=torch.Generator().manual_seed(1))


def test_split_forward_equals_hidden():
    m = tiny_model()
    with torch.no_grad():
        ref = m.hidden(X)
        h, ctx = m.embed(X)
        a = m.run_layers(m.run_layers(h, ctx, 0, 2), ctx, 2, 4, final_norm=True)
        with m.modes({1: "student", 3: "student"}):
            ref_s = m.hidden(X)
            h, ctx = m.embed(X)
            b = m.run_layers(m.run_layers(h, ctx, 0, 1), ctx, 1, 4, final_norm=True)
    assert float((a - ref).abs().max()) < 1e-5 and float((b - ref_s).abs().max()) < 1e-5


def test_stage1_microbatch_equals_full_batch():
    m = tiny_model()
    A = m.attn_layers
    zero(m)
    l_full = align.stage1_accumulate(m, A, X, micro_batch=None)
    g_full = grads(m, A)
    for mb in (1, 2):
        zero(m)
        l_mb = align.stage1_accumulate(m, A, X, micro_batch=mb)
        assert max_rel(grads(m, A), g_full) < TOL, mb
        assert all(abs(l_mb[i] - l_full[i]) < 1e-5 * max(1.0, abs(l_full[i])) for i in A)
    assert all(float(g.abs().sum()) > 0 for g in g_full[:3])  # gradients actually flow


def _stage2_reference(m, modes, x):
    """The pre-fix path: full teacher forward, full student forward, KL mean over all tokens, one backward."""
    with torch.no_grad(), m.modes({}):
        ht = m.hidden(x)
    with m.modes(modes):
        hs = m.hidden(x)
        loss = align.kl_from_hidden(hs, ht, m.lm_head, 512)
        loss.backward()
    return loss.item()


def test_stage2_microbatch_and_prefix_sharing_equal_reference():
    for student in ([1], [1, 3], [3]):
        m = tiny_model()
        modes = {i: "student" for i in student}
        zero(m)
        ref_loss = _stage2_reference(m, modes, X)
        g_ref = grads(m, student)
        m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.lm.train()
        try:
            for mb in (None, 1, 2):
                zero(m)
                loss = align.stage2_accumulate(m, modes, X, micro_batch=mb)
                assert abs(loss - ref_loss) < 1e-5 * max(1.0, abs(ref_loss)), (student, mb, loss, ref_loss)
                assert max_rel(grads(m, student), g_ref) < TOL, (student, mb)
        finally:
            m.lm.eval()
            m.lm.gradient_checkpointing_disable()


def test_microbatch_must_divide_batch():
    try:
        align._micro(X, 3)
        raise AssertionError("non-divisible micro_batch must raise")
    except ValueError:
        pass


TESTS = [test_split_forward_equals_hidden, test_stage1_microbatch_equals_full_batch,
         test_stage2_microbatch_and_prefix_sharing_equal_reference, test_microbatch_must_divide_batch]


def main():
    res = {}
    for t in TESTS:
        try:
            t()
            res[t.__name__] = True
        except Exception as e:  # noqa: BLE001
            res[t.__name__] = f"FAIL {e!r}"
    res["n_tests"], res["n_passed"] = len(TESTS), sum(v is True for v in res.values())
    res["PASS"] = res["n_passed"] == len(TESTS)
    print(json.dumps(res, indent=1))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

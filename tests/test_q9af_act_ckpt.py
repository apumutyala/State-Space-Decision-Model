"""Cache-fork-compatible block activation checkpointing for Q9-AF and its hybrids (heads/act_ckpt.py; jobs/q9af_cell/
launch evidence, LAUNCH_CHECKLIST row 13; PREMORTEM
blocker #4 / S2-F1, S2-F2).

Tiny Qwen3.5 stand-ins on CPU (torch fallbacks of the GDN chunk / recurrent rules and of causal_conv1d):
* ``q9af``    stock hybrid (3 GDN : 1 attention) whose attention layer is replaced by a converted GDN through
              ``q9af_from_stock`` (exactly how ``load_q9af_backbone`` builds the 9B);
* ``hybrid``  the same with the attention layer KEPT (``allow_partial``: the gdn_partial.pt case) on a 2-attention
              stand-in, one of the two attention layers converted;
* ``stock``   the unconverted hybrid (attention KV cache inside the fork).

Equivalence (fp32): one and two optimiser steps of the real ``Trainer`` (cache-fork ``fork`` head, Clef arm, LoRA on
the spec groups incl. the converted layer) with ``act_ckpt`` on vs off: loss bit-identical and every LoRA / head
gradient bit-identical (asserted <= 1e-6 relative as the documented bound; measured 0.0). 1-token continuations take
HF's single-token decode path (in-place ``causal_conv1d_update``) and are covered separately ON CPU ONLY
(torch fallback); on an fla install the fork refuses them under autograd (fla's single-token kernel has no backward).
Memory: saved-for-backward bytes at T in {128, 512} with and without checkpointing; the per-token slope must shrink
by >= 3x (measured on the stand-in; receipt with SSDM_WRITE_RECEIPTS=1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from heads import act_ckpt  # noqa: E402
from heads.backbone import HiddenStateBackbone, LineageError, q9af_from_stock, tiny_hf_backbone  # noqa: E402

PIN = "5.17.0"  # the full8 / probe9b / q9af_cell image pin: Qwen3.5 layer + cache layout
RECEIPT = Path(__file__).resolve().parents[1] / "receipts" / "2026-10-07-D-q9af-act-ckpt"


def _stock(seed: int, hidden: int, n_attn: int = 1):
    import transformers as tf

    from heads.backbone import BackboneSpec, _GDN, _QATTN, _QMLP

    torch.manual_seed(seed)
    types = ["linear_attention"] * 4
    for i in ([3] if n_attn == 1 else [1, 3]):
        types[i] = "full_attention"
    kw = dict(vocab_size=260, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
              linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8, layer_types=types)
    model = tf.Qwen3_5ForCausalLM(tf.Qwen3_5TextConfig(**kw))
    spec = BackboneSpec(name="tiny-qwen3.5", family="qwen3.5", hf_id="random", params=0, n_layers=4, d_model=hidden,
                        lora_groups=(("gdn", _GDN, 4 * 3), ("attn", _QATTN, 4), ("mlp", _QMLP, 4 * 3)),
                        capture="hidden_states", attention_free=False)
    return model, spec


def build(kind: str, seed: int = 0, hidden: int = 32) -> HiddenStateBackbone:
    model, spec = _stock(seed, hidden, n_attn=2 if kind == "hybrid" else 1)
    text = model.model
    attn = [i for i, t in enumerate(text.config.layer_types) if t == "full_attention"]
    if kind in ("q9af", "hybrid"):
        donor = tiny_hf_backbone("q9af", seed=seed + 1, hidden=hidden)
        conv = attn if kind == "q9af" else attn[-1:]
        sd = {i: {k: v.clone() for k, v in donor.model.model.layers[i].linear_attn.state_dict().items()} for i in conv}
        q9af_from_stock(text, sd, {"kind": "q9af", "converted_sha256": "0" * 64}, allow_partial=kind == "hybrid")
    with torch.no_grad():  # random init leaves norms at 1 / small values; perturb so gradients are not degenerate
        for p in text.parameters():
            p.add_(0.02 * torch.randn_like(p))
    kept = sum(t == "full_attention" for t in text.config.layer_types)
    import dataclasses

    groups = tuple((g, rx, n) for g, rx, n in spec.lora_groups if g != "attn" or kept)
    groups = tuple((g, rx, kept * 4 if g == "attn" else (4 - kept) * 3 if g == "gdn" else n) for g, rx, n in groups)
    spec = dataclasses.replace(spec, lora_groups=groups, attention_free=kept == 0)
    return HiddenStateBackbone(text, spec=spec, pad_id=0, fork_mode="cache")


def _items(n_steps=2):
    from jobs.fla_rwkv7 import train_mix as TM
    from train.data import Item

    def shape(T, conts, qtype="choice"):
        k = len(conts)
        a = T - 2 * k - 2
        return {"T": T, "conts": conts, "qtype": qtype, "state_span": [4, a], "field_span": [1, 3],
                "option_spans": [[a + 2 * j, a + 2 * j + 1] for j in range(k)]}

    shapes = [shape(150, [3, 2, 4]), shape(70, [1, 2]), shape(201, [2, 2, 2, 5], "score"), shape(96, [4, 1, 1]),
              shape(133, [2, 3]), shape(65, [1, 1, 2, 2, 3]), shape(180, [6, 2]), shape(88, [2, 2, 2], "score")]
    rows = TM.rows_from_shapes(shapes * n_steps, vocab=260, seed=0)
    return [Item(x.id, x.row, i % x.row.n_options) for i, x in enumerate(rows)]


def _trainer(kind, items, **flags):
    from heads.model import DecisionModel
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    bb = build(kind)
    model = DecisionModel(bb, "fork", {}, pad_id=0, seed=0)
    cfg = TrainConfig(head="fork", fork_mode="cache", lora_rank=4, loss=LossConfig(arm="clef"), batch_size=1,
                      grad_accum=8, max_steps=2, seed=0, **flags)
    tr = Trainer(model, cfg, items, [], guard=SplitGuard.synthetic_only("q9af act_ckpt equivalence"))
    tr._order = lambda epoch: list(range(len(items)))
    with torch.no_grad():  # non-zero LoRA B so the LoRA path carries signal into every gradient
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))) * 0.05)
    return tr


def _step_grads(tr):
    grads = {}
    real = tr.opt.step

    def capture(*a, **k):
        for n, p in tr.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = p.grad.detach().clone()
        return real(*a, **k)

    tr.opt.step = capture
    loss = tr.train_step()
    tr.opt.step = real
    return loss, grads


def _compare(a, b, tol=1e-6):
    (la, ga), (lb, gb) = a, b
    assert set(ga) == set(gb) and any(n.endswith("lora_A") for n in ga) and any(n.startswith("head.") for n in ga)
    assert la == lb, (la, lb)  # forward values are bit-identical
    worst = 0.0
    for n in ga:
        ref = float(gb[n].abs().max())
        worst = max(worst, float((ga[n] - gb[n]).abs().max()) / max(ref, 1e-30))
    assert worst <= tol, worst
    return worst


@pytest.mark.requires_transformers(PIN)
@pytest.mark.parametrize("kind", ["q9af", "hybrid", "stock"])
def test_act_ckpt_equals_plain_train_step_with_cache_fork(kind):
    items = _items()
    base = _step_grads(_trainer(kind, items))
    tr = _trainer(kind, items, act_ckpt=True)
    assert act_ckpt.is_enabled(tr.model.backbone.model) and act_ckpt.supported(tr.model.backbone.model) == "qwen3.5"
    got = _step_grads(tr)
    w1 = _compare(got, base)
    # LoRA reaches the converted layer and (hybrid / stock) the kept attention layer
    names = set(got[1])
    assert any("layers.3.linear_attn" in n and n.endswith("lora_A") for n in names) or kind == "stock"
    if kind != "q9af":
        assert any("self_attn" in n and n.endswith("lora_A") for n in names)
    # second step (after an optimiser update) still matches
    a, b = _trainer(kind, items), _trainer(kind, items, act_ckpt=True)
    a.train_step(), b.train_step()
    w2 = _compare(_step_grads(b), _step_grads(a))
    assert (w1, w2) == (0.0, 0.0) or max(w1, w2) <= 1e-6


def _fork_grads(bb, prefix, conts, ck):
    (act_ckpt.enable_block_checkpointing if ck else act_ckpt.disable_block_checkpointing)(bb.model)
    for p in bb.parameters():
        p.grad = None
    h, f = bb.fork_features(prefix, conts)
    loss = f.pow(2).sum() + h.pow(2).mean()
    loss.backward()
    return float(loss), {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}


@pytest.mark.requires_transformers(PIN)
@pytest.mark.parametrize("kind", ["q9af", "hybrid"])
@pytest.mark.parametrize("conts", [[[30, 31, 32], [40, 41]], [[30], [40], [50]]], ids=["multi_token", "one_token"])
def test_fork_features_equal_with_and_without_ckpt(kind, conts):
    """One-token continuations take HF's single-token decode path (in-place causal_conv1d_update on the conv tail);
    the checkpointed holder works on a clone, so the saved input is never mutated."""
    from train.lora import apply_lora

    bb = build(kind, seed=3)
    apply_lora(bb.model, bb.spec.lora_groups, rank=4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for n, p in bb.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.05)
    bb.train()
    prefix = list(range(5, 90))
    la, ga = _fork_grads(bb, prefix, conts, False)
    lb, gb = _fork_grads(bb, prefix, conts, True)
    assert la == lb and ga.keys() == gb.keys() and len(ga) > 0
    for n in ga:
        assert torch.equal(ga[n], gb[n]), n


@pytest.mark.requires_transformers(PIN)
def test_recompute_fork_and_plain_features_are_checkpointed_too():
    """No cache (recompute fork / plain features): the block is checkpointed as is; values and grads unchanged."""
    from train.lora import apply_lora

    bb = build("q9af", seed=4)
    apply_lora(bb.model, bb.spec.lora_groups, rank=4, generator=torch.Generator().manual_seed(1))
    bb.train()
    ids = torch.randint(5, 250, (2, 40), generator=torch.Generator().manual_seed(0))
    out = {}
    for ck in (False, True):
        (act_ckpt.enable_block_checkpointing if ck else act_ckpt.disable_block_checkpointing)(bb.model)
        bb.model.zero_grad(set_to_none=True)
        f = bb.features(ids)
        f.pow(2).mean().backward()
        out[ck] = (f.detach(), {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None})
    assert torch.equal(out[True][0], out[False][0])
    assert all(torch.equal(out[True][1][n], out[False][1][n]) for n in out[False][1])


@pytest.mark.requires_transformers(PIN)
def test_no_grad_bypass_disable_restores_and_idempotent():
    bb = build("q9af")
    ids = torch.randint(5, 250, (1, 30), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = bb.features(ids)
    assert act_ckpt.enable_block_checkpointing(bb.model) == 4
    assert act_ckpt.enable_block_checkpointing(bb.model) == 4  # idempotent: no double wrap
    with torch.no_grad():
        assert torch.equal(bb.features(ids), ref)
        _, f1 = bb.fork_features(list(range(5, 40)), [[7, 8], [9]])
    assert act_ckpt.disable_block_checkpointing(bb.model) == 4
    assert not act_ckpt.is_enabled(bb.model) and all("forward" not in l.__dict__ for l in bb.model.layers)
    with torch.no_grad():
        _, f2 = bb.fork_features(list(range(5, 40)), [[7, 8], [9]])
    assert torch.equal(f1, f2)


@pytest.mark.requires_transformers(PIN)
def test_unsupported_cache_layers_fail_closed():
    from heads.backbone import _linear_safe_cache as DynamicCache  # the all-GDN-safe cache the fork uses

    bb = build("q9af")
    act_ckpt.enable_block_checkpointing(bb.model)
    cache = DynamicCache(bb.model.config)
    cache.offloading = True
    ids = torch.randint(5, 250, (1, 10))
    with pytest.raises(TypeError, match="offloaded"):
        bb.model(input_ids=ids, past_key_values=cache, use_cache=True)
    cache = DynamicCache(bb.model.config)
    cache.layers[0].activate_past_recording()
    with pytest.raises(TypeError, match="past-recording"):
        bb.model(input_ids=ids, past_key_values=cache, use_cache=True)
    with pytest.raises(ValueError, match="Qwen3.5"):
        act_ckpt.enable_block_checkpointing(torch.nn.Linear(2, 2))


@pytest.mark.requires_transformers(PIN)
def test_hybrid_loader_is_explicit_and_labelled_not_attention_free():
    model, _ = _stock(0, 32, n_attn=2)
    donor = tiny_hf_backbone("q9af", seed=1)
    sd = {3: donor.model.model.layers[3].linear_attn.state_dict()}
    with pytest.raises(LineageError, match="needs all"):
        q9af_from_stock(model.model, sd, {"kind": "q9af"})  # a partial export is refused unless explicitly hybrid
    model, _ = _stock(0, 32, n_attn=2)
    text = q9af_from_stock(model.model, sd, {"kind": "q9af", "converted_sha256": "1" * 64}, allow_partial=True)
    prov = text.q9af_provenance
    assert prov["kind"] == "q9af-hybrid" and prov["attention_free"] is False and prov["attention_layers_kept"] == [1]
    assert "NOT attention-free" in prov["label"] and prov["converted_sha256"] == "1" * 64
    assert text.config.layer_types == ["linear_attention", "full_attention", "linear_attention", "linear_attention"]
    model, _ = _stock(0, 32, n_attn=2)
    with pytest.raises(LineageError, match="subset"):
        q9af_from_stock(model.model, {2: sd[3]}, {}, allow_partial=True)  # layer 2 is not an attention layer


def _saved_bytes(fn) -> int:
    seen, total = set(), [0]

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total[0] += t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    out.backward()
    return total[0]


@pytest.mark.requires_transformers(PIN)
def test_act_ckpt_shrinks_saved_activations_per_token():
    """CPU measurement (the stand-in at d=64; head dims 8): saved-for-backward bytes of one cache-fork forward
    (prefix T + K=4 two-token continuations), LoRA r4 on, with and without block checkpointing. The per-token slope
    (bytes/token, T 128 -> 512) is what scales with the training length."""
    from train.lora import apply_lora

    lengths, hid, res = [128, 512], 64, {}
    for ck in (False, True):
        torch.manual_seed(0)
        bb = build("q9af", hidden=hid)
        apply_lora(bb.model, bb.spec.lora_groups, rank=4, generator=torch.Generator().manual_seed(0))
        bb.train()
        if ck:
            act_ckpt.enable_block_checkpointing(bb.model)
        meas = []
        for t in lengths:
            prefix = torch.randint(5, 250, (t,), generator=torch.Generator().manual_seed(t)).tolist()
            conts = [[30 + j, 40 + j] for j in range(4)]
            meas.append(_saved_bytes(lambda: bb.fork_features(prefix, conts)[1].pow(2).sum()))
        res[ck] = {"saved_bytes": meas, "slope_bytes_per_token": (meas[1] - meas[0]) / (lengths[1] - lengths[0])}
    ratio = res[False]["slope_bytes_per_token"] / res[True]["slope_bytes_per_token"]
    per_layer_d = {str(k): v["slope_bytes_per_token"] / 4 / hid for k, v in res.items()}
    assert ratio >= 3.0, res
    # with checkpointing only the block inputs survive: ~1 residual-stream vector per token per layer (fp32: 4 B x d)
    assert per_layer_d["True"] <= 2 * 4.0 + 1.0, per_layer_d
    if os.environ.get("SSDM_WRITE_RECEIPTS") == "1":
        RECEIPT.mkdir(parents=True, exist_ok=True)
        (RECEIPT / "cpu_saved_bytes.json").write_text(json.dumps({
            "what": "saved-for-backward bytes, one cache-fork forward (prefix T + 4 two-token continuations), tiny "
                    "Q9-AF stand-in (4 layers, d=64, GDN head dims 8, LoRA r4 on gdn+mlp), fp32 CPU",
            "lengths": lengths, "no_ckpt": res[False], "ckpt": res[True], "slope_ratio": ratio,
            "bytes_per_token_per_layer_per_d": per_layer_d,
            "test": "tests/test_q9af_act_ckpt.py::test_act_ckpt_shrinks_saved_activations_per_token"},
            indent=1, sort_keys=True), encoding="utf-8")


@pytest.mark.requires_transformers(PIN)
def test_short_continuations_fail_closed_under_autograd_on_fla(monkeypatch):
    """PR #121 review (non-blocking 1): with fla installed, a 1-token continuation under autograd would take fla's
    single-token recurrent kernel, which has no backward (a GPU-only crash). The fork refuses it up front; >= 2-token
    continuations and no_grad evaluation are unaffected. (CPU: fla is absent, so the detector is forced on.)"""
    import heads.backbone as B

    bb = build("q9af", seed=5)
    bb.train()
    prefix = list(range(5, 40))
    monkeypatch.setattr(B, "fla_installed", lambda: True)
    with pytest.raises(RuntimeError, match="no backward"):
        bb.fork_features(prefix, [[30, 31], [40]])
    bb.fork_features(prefix, [[30, 31], [40, 41, 42]])[1].sum().backward()  # every continuation >= 2 tokens: fine
    with torch.no_grad():
        bb.fork_features(prefix, [[30], [40]])  # evaluation: no backward, no refusal
    monkeypatch.setattr(B, "fla_installed", lambda: False)
    bb.fork_features(prefix, [[30], [40]])[1].sum().backward()  # torch fallback (has a backward): allowed

"""PREMORTEM.md CPU probes (tiny random stand-ins; no weights, no GPU, $0).

P1  Q9-AF (all-GDN Qwen3.5 layout, built by q9af_from_stock) plugs into the real m2 train step: cache-fork head,
    LoRA on the q9af spec's GDN + MLP groups (including the CONVERTED layer), Trainer.train_step, checkpoint resume.
P2  The cache-fork LoRA gradients on Q9-AF equal the recompute-fork gradients (autograd through the copied GDN
    states and the patched in-place cache updates is correct).
P3  None of PR #86's memory/launch levers applies to Q9-AF: block checkpointing is RWKV-7-only, batched-row fork needs
    a ``state=`` backbone (silent per-row fallback), and HF gradient checkpointing is refused with the cache fork.
P4  Saved-for-backward activation bytes per token per layer, Q9-AF vs RWKV-7 stand-ins under the cache-fork train
    step, normalised by d_model, then scaled to the 9B / 13.3B shapes (anchored on job-mden7's measured ~1.0 MB per
    token per layer for RWKV-7). Results -> receipts/2026-10-06-D-premortem-probes/p4_activation_bytes.json.
P5  Proxy loop exactness: head training on cached frozen-backbone features (train.proxy) == Trainer with LoRA rank 0.
P6  Proxy transfer: recipe ranking under the proxy vs under head + LoRA training (Kendall tau), and the speed-up.
    Results -> receipts/2026-10-06-D-premortem-probes/p6_proxy_transfer.json.
Receipts are written with SSDM_WRITE_RECEIPTS=1 (how the committed ones were produced).

Launch collection (LESSONS L-24): P1-P4 require transformers 5.17.0, the Q9-AF stack of jobs/full8/ (and
jobs/probe9b/, jobs/q9_feature_cache/), so jobs.common.launch runs them in those jobs' test environments; P7 reads
jobs/clef_baseline/summarize.py and runs under every job that collects this file.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import HiddenStateBackbone, q9af_from_stock, tiny_backbone, tiny_hf_backbone  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.losses import LossConfig  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("premortem probes: train.data.toy_records only")
OUT = Path(__file__).resolve().parents[1] / "receipts" / "2026-10-06-D-premortem-probes"


def _write(name: str, obj: dict) -> None:
    """Receipts are (re)written only with SSDM_WRITE_RECEIPTS=1, so routine test runs leave the tree clean."""
    import os

    if os.environ.get("SSDM_WRITE_RECEIPTS") != "1":
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def q9af_tiny(seed: int = 0, hidden: int = 32) -> HiddenStateBackbone:
    """Tiny Q9-AF built the way load_q9af_backbone builds the real one: a stock hybrid (3 GDN : 1 attention) whose
    attention layer gets a converted GDN via q9af_from_stock (here: random GDN weights from a donor)."""
    stock = tiny_hf_backbone("qwen3.5", seed=seed, hidden=hidden)
    donor = tiny_hf_backbone("q9af", seed=seed + 1, hidden=hidden)
    attn = [i for i, t in enumerate(stock.model.model.config.layer_types) if t == "full_attention"]
    sd = {i: {k: v.clone() for k, v in donor.model.model.layers[i].linear_attn.state_dict().items()} for i in attn}
    text = q9af_from_stock(stock.model.model, sd, {"kind": "q9af", "converted_sha256": "0" * 64})
    with torch.no_grad():  # random init leaves norms at 1 / small values; perturb so gradients are not degenerate
        for p in text.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return HiddenStateBackbone(text, spec=donor.spec, pad_id=0, fork_mode="cache"), attn


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(24, seed=0), ENC, template=TOY_TEMPLATE)


# ----------------------------------------------------------------------------------------------------- P1
@pytest.mark.requires_transformers("5.17.0")  # the probe9b/full8/rwkv7 images' pin: Q9-AF layer layout
def test_p1_q9af_plugs_into_the_m2_train_step(toy, tmp_path):
    bb, attn = q9af_tiny()
    model = DecisionModel(bb, "fork", seed=0)
    cfg = TrainConfig(head="fork", fork_mode="cache", lora_rank=4, loss=LossConfig(arm="clef"), batch_size=2,
                      grad_accum=2, max_steps=3, seed=0, ckpt_dir=str(tmp_path), ckpt_every=3)
    tr = Trainer(model, cfg, toy, guard=G)
    converted = [n for n in tr.lora_modules if f"layers.{attn[0]}.linear_attn" in n]
    assert converted, "LoRA must reach the converted layer's GDN projections"
    losses = [tr.train_step() for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(losses)))
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n}
    conv_g = [g for n, g in grads.items() if f"layers.{attn[0]}.linear_attn" in n]
    assert conv_g and all(g is not None and float(g.abs().sum()) > 0 for g in conv_g)
    tr.save()
    model2 = DecisionModel(q9af_tiny()[0], "fork", seed=0)
    tr2 = Trainer(model2, cfg, toy, guard=G)
    assert tr2.resume() and tr2.step == 3


# ----------------------------------------------------------------------------------------------------- P2
def _lora_grads(bb, rows, mode):
    from heads.features import build_head_input

    bb.fork_mode = mode
    for p in bb.parameters():
        p.grad = None
    x = build_head_input(bb, rows, need_fork=True)
    x.fork.pow(2).sum().backward()
    return {n: p.grad.detach().clone() for n, p in bb.named_parameters() if p.requires_grad and p.grad is not None}


@pytest.mark.requires_transformers("5.17.0")  # the probe9b/full8/rwkv7 images' pin: Q9-AF layer layout
def test_p2_q9af_cache_fork_lora_gradients_equal_recompute(toy):
    from train.lora import apply_lora

    bb, attn = q9af_tiny(seed=3)
    apply_lora(bb.model, bb.spec.lora_groups, rank=4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():  # lora_B is zero at init: give it values so lora_A also gets a gradient
        for n, p in bb.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(0, 0.05)
    bb.train()
    rows = [toy[0].row, toy[1].row]
    a, b = _lora_grads(bb, rows, "cache"), _lora_grads(bb, rows, "recompute")
    assert a.keys() == b.keys() and len(a) > 0
    worst = 0.0
    for n in a:
        cos = torch.nn.functional.cosine_similarity(a[n].flatten(), b[n].flatten(), dim=0)
        rel = float((a[n] - b[n]).norm() / (b[n].norm() + 1e-12))
        assert float(cos) > 0.999 and rel < 1e-3, (n, float(cos), rel)
        worst = max(worst, rel)
    _write("p2_grad_equivalence.json", {"n_lora_tensors": len(a), "max_rel_err": worst,
                                        "converted_layer": attn, "status": "pass"})


# ----------------------------------------------------------------------------------------------------- P3
@pytest.mark.requires_transformers("5.17.0")  # the probe9b/full8/rwkv7 images' pin: Q9-AF layer layout
def test_p3_pr86_levers_do_not_apply_to_q9af(toy):
    from heads import act_ckpt

    bb, _ = q9af_tiny()
    with pytest.raises(ValueError, match="RWKV-7"):
        act_ckpt.enable_block_checkpointing(bb.model)  # TrainConfig(act_ckpt=True) raises for Q9-AF
    assert not bb.can_fork_rows()  # TrainConfig(row_batch>1) silently falls back to one prefix call per row
    model = DecisionModel(bb, "fork", seed=0)
    tr = Trainer(model, TrainConfig(head="fork", lora_rank=2, batch_size=2, row_batch=2, max_steps=1), toy, guard=G)
    assert model.backbone.batch_rows and not model.backbone.can_fork_rows()
    tr.train_step()  # runs, but per row: no batching benefit
    bb2, _ = q9af_tiny()
    m2 = DecisionModel(bb2, "fork", seed=0)
    tr2 = Trainer(m2, TrainConfig(head="fork", lora_rank=2, batch_size=1, grad_checkpointing=True, max_steps=1),
                  toy, guard=G)
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        tr2.train_step()  # the only remaining memory lever is refused under the cache fork
    _write("p3_levers.json", {"act_ckpt": "raises (RWKV-7 only)", "row_batch": "silent per-row fallback",
                              "hf_grad_ckpt_with_cache_fork": "refused", "status": "pass (blocker confirmed)"})


# ----------------------------------------------------------------------------------------------------- P4
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


def _per_token_layer_d(bb, n_layers, d, lengths, k=4):
    """Slope of saved bytes vs prefix length (bytes per token), per layer, per unit of d_model."""
    from heads.features import build_head_input
    from heads.layout import LayoutRow  # noqa: F401  (rows are synthesised directly below)
    from train.lora import apply_lora

    apply_lora(bb.model, bb.spec.lora_groups, rank=4, generator=torch.Generator().manual_seed(0))
    bb.train()
    bb.fork_mode = "cache"
    meas = []
    for t in lengths:
        prefix = list(torch.randint(5, 250, (t,)).tolist())
        conts = [[30 + j, 40 + j] for j in range(k)]
        meas.append(_saved_bytes(lambda: bb.fork_features(prefix, conts)[1].pow(2).sum()))
    slope = (meas[-1] - meas[0]) / (lengths[-1] - lengths[0])
    return slope / n_layers / d, meas


@pytest.mark.requires_transformers("5.17.0")  # the probe9b/full8/rwkv7 images' pin: Q9-AF layer layout
def test_p4_activation_bytes_per_token_q9af_vs_rwkv7():
    torch.manual_seed(0)
    lengths = [128, 512]
    hid = 64
    q9, _ = q9af_tiny(hidden=hid)
    rw = tiny_hf_backbone("rwkv7", hidden=hid)
    q_unit, q_meas = _per_token_layer_d(q9, 4, hid, lengths)
    r_unit, r_meas = _per_token_layer_d(rw, 2, hid, lengths)
    assert q_unit > 0 and r_unit > 0
    ratio = q_unit / r_unit
    # anchor: job-mden7 measured ~1.0 MB / token / layer for RWKV-7 13.3B (61 layers, d 4096; PERF_AUDIT.md:42)
    rw_mb = 1.0
    q_mb = rw_mb * ratio  # same d_model (4096) for both 9B shapes
    q_gb_per_ktok = q_mb * 32 * 1024 / 1024
    rw_gb_per_ktok = rw_mb * 61 * 1024 / 1024
    headroom_q = 76 - 7.94e9 * 2 / 2**30 - 159e6 * 16 / 2**30  # bf16 weights, LoRA r64 (gdn+mlp, 32 layers) fp32+Adam
    headroom_r = 76 - 26.5 - 288e6 * 16 / 2**30
    res = {"tiny_hidden": hid, "lengths": lengths,
           "bytes_per_token_per_layer_per_d": {"q9af": q_unit, "rwkv7": r_unit}, "ratio_q9af_over_rwkv7": ratio,
           "raw_saved_bytes": {"q9af": q_meas, "rwkv7": r_meas},
           "projection_9b": {"anchor_rwkv7_mb_per_token_layer": rw_mb, "q9af_mb_per_token_layer": q_mb,
                             "q9af_gb_per_1k_tokens": q_gb_per_ktok, "rwkv7_gb_per_1k_tokens": rw_gb_per_ktok,
                             "headroom_gb": {"q9af": headroom_q, "rwkv7": headroom_r},
                             "max_T_no_ckpt": {"q9af": 1024 * headroom_q / q_gb_per_ktok,
                                               "rwkv7": 1024 * headroom_r / rw_gb_per_ktok}},
           "caveats": ["CPU path: torch_chunk_gated_delta_rule fallback (GPU uses fla chunk kernels, which save less)",
                       "tiny head dims (8) vs real (128): state-size terms are under-weighted",
                       "K=4 continuations; the fork's K state copies add K x state bytes per layer independent of T"]}
    _write("p4_activation_bytes.json", res)


# ----------------------------------------------------------------------------------------------------- P5 / P6
def _fresh(head, seed, hidden=16):
    return DecisionModel(tiny_backbone(seed=seed, hidden=hidden, fork_mode="cache"), head, seed=seed)


@pytest.mark.parametrize("head", ["integrator", "fork"])
@pytest.mark.parametrize("bf16", [False, True])
def test_p5_cached_feature_proxy_equals_trainer_at_lora_rank_0(toy, head, bf16):
    """fp32 and bf16 (run_cell / m2_gates / cell_step pass bf16=cuda: the Trainer forward runs under bf16 autocast),
    for a sequence head and the cache-fork head."""
    from train.proxy import cache_items, train_head_cached

    cfg = TrainConfig(head=head, lora_rank=0, batch_size=1, grad_accum=4, max_steps=8, lr_head=3e-3,
                      warmup_steps=2, loss=LossConfig(arm="clef"), seed=5, bf16=bf16)
    a = _fresh(head, 5)
    b = copy.deepcopy(a)
    Trainer(a, cfg, toy, guard=G).train()
    for p in b.backbone.parameters():
        p.requires_grad_(False)
    train_head_cached(b, cache_items(b, toy, cfg=cfg), cfg, guard=G)
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for n in [n for n in pa if not n.startswith("backbone.")]:
        assert torch.allclose(pa[n], pb[n], atol=1e-5), (n, float((pa[n] - pb[n]).abs().max()))


def test_p5_proxy_refuses_what_the_trainer_would_do_differently(toy):
    import dataclasses

    from train.proxy import cache_items, train_head_cached

    cfg = TrainConfig(head="fork", lora_rank=0, batch_size=1, grad_accum=2, max_steps=1, seed=5)
    m = _fresh("fork", 5)
    cached = cache_items(m, toy[:4], cfg=cfg)
    with pytest.raises(ValueError, match="require_guard"):
        train_head_cached(m, cached, cfg)
    with pytest.raises(ValueError, match="bf16, fork_mode"):
        train_head_cached(m, cached, dataclasses.replace(cfg, bf16=True), guard=G)
    with pytest.raises(ValueError, match="bf16, fork_mode"):
        train_head_cached(m, cached, dataclasses.replace(cfg, fork_mode="recompute"), guard=G)
    with pytest.raises(ValueError, match="row_batch"):
        train_head_cached(m, cached, dataclasses.replace(cfg, row_batch=2), guard=G)
    with pytest.raises(ValueError, match="trust_gate"):
        train_head_cached(m, cached, dataclasses.replace(cfg, controlsafe={"trust_gate": True}), guard=G)
    with pytest.raises(ValueError, match="trust_gate"):
        cache_items(m, toy[:1], cfg=dataclasses.replace(cfg, controlsafe={"trust_gate": True}))


def test_p5_eval_cached_runs_in_eval_mode_and_restores(toy):
    from train.proxy import cache_items, eval_cached

    m = _fresh("fork", 5)
    seen = []
    m.head.register_forward_pre_hook(lambda mod, args: seen.append(mod.training))
    m.train()
    eval_cached(m, cache_items(m, toy[:2]))
    assert seen and not any(seen) and m.training


RECIPES = [  # (head, loss arm, lr_head)
    ("integrator", "ce", 1e-3), ("integrator", "ce", 1e-2), ("integrator", "clef", 3e-3),
    ("fork", "ce", 1e-3), ("fork", "ce", 1e-2), ("fork", "clef", 3e-3),
]


def test_p6_proxy_recipe_ranking_transfers_to_head_plus_lora():
    from train.proxy import cache_items, eval_cached, kendall_tau, train_head_cached

    train = items_from_records(toy_records(48, seed=1), ENC, template=TOY_TEMPLATE)
    dev = items_from_records(toy_records(24, seed=2), ENC, template=TOY_TEMPLATE)
    steps, proxy, full, t_proxy, t_full = 40, [], [], 0.0, 0.0
    for head, arm, lr in RECIPES:
        base = _fresh(head, 0)
        for p in base.backbone.parameters():
            p.requires_grad_(False)
        m = copy.deepcopy(base)
        tr_c, dv_c = cache_items(m, train), cache_items(m, dev)
        cfg0 = TrainConfig(head=head, lora_rank=0, batch_size=1, grad_accum=4, max_steps=steps, lr_head=lr,
                           loss=LossConfig(arm=arm), seed=0)
        t_proxy += train_head_cached(m, tr_c, cfg0, guard=G)["seconds"]
        proxy.append(eval_cached(m, dv_c)["nll"])
        f = copy.deepcopy(base)
        cfg1 = TrainConfig(head=head, lora_rank=4, batch_size=1, grad_accum=4, max_steps=steps, lr_head=lr,
                           lr_lora=1e-3, loss=LossConfig(arm=arm), seed=0)
        t0 = time.perf_counter()
        Trainer(f, cfg1, train, guard=G).train()
        t_full += time.perf_counter() - t0
        full.append(eval_cached(f, cache_items(f, dev))["nll"])
    tau = kendall_tau(proxy, full)
    best_match = int(min(range(len(proxy)), key=proxy.__getitem__) == min(range(len(full)), key=full.__getitem__))
    _write("p6_proxy_transfer.json", {
        "recipes": [list(r) for r in RECIPES], "steps": steps, "dev_nll_proxy": proxy, "dev_nll_head_plus_lora": full,
        "kendall_tau": tau, "argmin_agrees": bool(best_match),
        "train_seconds": {"proxy_head_only": t_proxy, "head_plus_lora": t_full},
        "speedup_train_loop": t_full / max(t_proxy, 1e-9),
        "note": "tiny random backbone, toy task: a mechanism check of the transfer measurement, not evidence about 9B"})
    assert -1.0 <= tau <= 1.0


# ----------------------------------------------------------------------------------------------------- P7
def test_p7_clef_composite_reader_splits_jsonl_on_unicode_line_separators(tmp_path):
    """job-sgihj died at stage 'composite' (exit 12) with 'Unterminated string' in summarize.rows. The kit writes JSONL
    with ensure_ascii=False and splits on '\n' only; summarize.rows uses str.splitlines(), which also splits on
    U+2028 / U+0085 / \x1c-\x1e. One row with such a character in echoed text reproduces the crash on CPU."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "clef_summarize", Path(__file__).resolve().parents[1] / "jobs" / "clef_baseline" / "summarize.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    p = tmp_path / "results.jsonl"
    recs = [{"id": "a", "text": "plain"}, {"id": "b", "text": "line separator"}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    assert [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()] == recs  # kit-style read
    # The reader at the time of the pre-mortem (str.splitlines) raised "Unterminated string" here, reproducing
    # job-sgihj's crash. PR #98 replaced it with the strict newline-only reader (harness.jsonl.read_jsonl_strict); the
    # probe now pins the fix: the same file reads back exactly.
    assert S.rows(p) == recs
    _write("p7_composite_crash.json", {"reproduced": True, "file": "jobs/clef_baseline/summarize.py:24",
                                       "cause": "str.splitlines() splits inside a JSON string on U+2028",
                                       "fix": "strict newline-only reader; truncated/partial files exit 17"})

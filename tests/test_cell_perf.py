"""Equivalence of the m2 cell's execution variants (jobs/cell_step_profile) to the original train step. CPU only.

* ``chunked_lean`` WKV (heads/rwkv7_lean.py) == the vendored ``rwkv7_parallel``: outputs, final state and every input
  gradient within fp32 rounding; packed contract; fewer aten ops (the launch proxy).
* per-block activation checkpointing (heads/act_ckpt.py, ``TrainConfig.act_ckpt``) == no checkpointing: same loss
  (bitwise) and same LoRA / head gradients, WITH the cache fork; saved activations shrink.
* batched rows (``TrainConfig.row_batch``; ``HiddenStateBackbone.fork_features_rows``) == one call per row.
* all three together == the original step; defaults leave the original path and the config hash untouched.

    python -X utf8 -m pytest -q tests/test_cell_perf.py
"""
from __future__ import annotations

import collections
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from heads import act_ckpt  # noqa: E402
from heads import rwkv7_lean as L  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402
from heads.rwkv7_wkv import select_wkv  # noqa: E402
from third_party.rwkv7_hf import modeling_rwkv7 as M  # noqa: E402

ARGS = ("r", "w_log", "k", "v", "kk", "a", "state")
REL = 3e-6  # fp32 rounding across GEMM shapes (measured max ~8e-7 on these cases)


def rel(x, ref) -> float:
    x, ref = x.detach().double(), ref.detach().double()
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def wkv_inputs(B, T, H, N, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    x = dict(r=rn(B, T, H, N), w_log=-0.6065306597 * torch.sigmoid(rn(B, T, H, N) * 2 - 0.5), k=rn(B, T, H, N),
             v=rn(B, T, H, N), kk=torch.nn.functional.normalize(rn(B, T, H, N), dim=-1), a=torch.sigmoid(rn(B, T, H, N)),
             state=rn(B, H, N, N) * 0.5)
    return {n: t.to(dtype).requires_grad_(True) for n, t in x.items()}


def run_wkv(fn, x, ct, cu=None):
    o, s = fn(*[x[n] for n in ARGS], cu_seq_lens=cu)
    g = torch.autograd.grad((o.float() * ct[0]).sum() + (s.float() * ct[1]).sum(), [x[n] for n in ARGS],
                            allow_unused=True)
    return o, s, g


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.c = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))


# ----------------------------------------------------------------------------- lean WKV
@pytest.mark.parametrize("B,T", [(1, 1), (2, 5), (2, 64), (2, 100), (1, 300)])
def test_lean_wkv_equals_vendored_outputs_state_and_grads(B, T):
    x = wkv_inputs(B, T, 3, 8, seed=T)
    g = torch.Generator().manual_seed(1)
    ct = (torch.randn(B, T, 3, 8, generator=g), torch.randn(B, 3, 8, 8, generator=g))
    ref, got = run_wkv(M.rwkv7_parallel, x, ct), run_wkv(L.rwkv7_parallel_lean, x, ct)
    assert got[0].dtype == ref[0].dtype and got[1].dtype == ref[1].dtype and got[0].shape == ref[0].shape
    assert rel(got[0], ref[0]) <= REL and rel(got[1], ref[1]) <= REL
    for n, a, b in zip(ARGS, got[2], ref[2]):
        assert rel(a, b) <= REL, n


def test_lean_wkv_packed_contract_and_bf16_inputs():
    x = wkv_inputs(1, 150, 2, 8, seed=3)
    ct = (torch.randn(1, 150, 2, 8), torch.randn(1, 2, 8, 8))
    cu = torch.tensor([0, 40, 41, 150])
    ref, got = run_wkv(M.rwkv7_parallel, x, ct, cu), run_wkv(L.rwkv7_parallel_lean, x, ct, cu)
    assert rel(got[0], ref[0]) <= REL and rel(got[1], ref[1]) <= REL
    assert all(rel(a, b) <= REL for a, b in zip(got[2][:-1], ref[2][:-1]))  # packed: the passed state is not read
    xb = {n: t.detach().to(torch.bfloat16 if n != "state" else torch.float32).requires_grad_(True) for n, t in x.items()}
    ref, got = run_wkv(M.rwkv7_parallel, xb, ct), run_wkv(L.rwkv7_parallel_lean, xb, ct)
    assert got[0].dtype == torch.bfloat16 and got[1].dtype == torch.float32
    # both compute in fp32 and round the output to bf16: at most one bf16 ulp apart
    assert rel(got[0], ref[0]) <= 2 ** -7 and rel(got[1], ref[1]) <= REL


def test_lean_wkv_chunk_guard_and_fewer_ops():
    x = wkv_inputs(1, 300, 1, 4)
    with pytest.raises(ValueError, match="overflows"):
        L.rwkv7_chunked_lean(*[x[n] for n in ARGS], chunk_size=200)
    x = wkv_inputs(1, 2048, 2, 16, seed=5)
    ct = (torch.randn(1, 2048, 2, 16), torch.randn(1, 2, 16, 16))
    counts = {}
    for name, fn in (("vendored", M.rwkv7_parallel), ("lean", L.rwkv7_parallel_lean)):
        with Count() as c:
            run_wkv(fn, x, ct)
        counts[name] = sum(c.c.values())
        if name == "lean":
            assert c.c["select_backward"] <= 1  # one (c[:, :, -1]), not 8 per chunk: no O(chunks^2) zero fills
    assert counts["lean"] * 3 <= counts["vendored"], counts  # fwd + bwd aten ops at T=2048: 4805 -> 1454 measured


def test_lean_masks_built_under_inference_mode_do_not_break_training():
    """Regression (PR #86 review): the first lean call under torch.inference_mode() used to cache inference-tensor
    masks; the next training call then raised 'Inference tensors cannot be saved for backward'."""
    L._MASKS.clear()
    x = wkv_inputs(1, 100, 2, 8, seed=9)
    with torch.inference_mode():
        L.rwkv7_parallel_lean(*[x[n].detach() for n in ARGS])
    assert L._MASKS and not any(t.is_inference() for m in L._MASKS.values() for t in m)
    ct = (torch.randn(1, 100, 2, 8), torch.randn(1, 2, 8, 8))
    got, ref = run_wkv(L.rwkv7_parallel_lean, x, ct), run_wkv(M.rwkv7_parallel, x, ct)
    assert all(rel(a, b) <= REL for a, b in zip(got[2], ref[2]))


def test_select_wkv_lean_keeps_batched_linears_and_matches_chunked_model():
    bb = tiny_hf_backbone("rwkv7", seed=0)
    m = bb.model
    ids = torch.randint(1, 260, (2, 130), generator=torch.Generator().manual_seed(0))
    ref = m(input_ids=ids).last_hidden_state
    assert select_wkv(m, "chunked_lean") == "chunked_lean"
    assert not (m.config.wkv_implementation != "chunked")  # -> tokenwise False: batched Linears, as on "chunked"
    assert M.RWKV7_WKV_FUNCTIONS["chunked_lean"] is L.rwkv7_parallel_lean
    got = m(input_ids=ids).last_hidden_state
    assert rel(got, ref) <= 1e-5
    select_wkv(m, "chunked")


# ----------------------------------------------------------------------------- trainer-level equivalence
def _shape(T, conts, qtype="choice"):
    k = len(conts)
    a = T - 2 * k - 2
    return {"T": T, "conts": conts, "qtype": qtype, "state_span": [4, a], "field_span": [1, 3],
            "option_spans": [[a + 2 * j, a + 2 * j + 1] for j in range(k)]}


SHAPES = [_shape(150, [3, 2, 4]), _shape(70, [1, 2]), _shape(201, [2, 2, 2, 5], "score"), _shape(96, [4, 1, 1]),
          _shape(133, [2, 3]), _shape(65, [1, 1, 2, 2, 3]), _shape(180, [6, 2]), _shape(88, [2, 2, 2], "score")]


def _items(n_steps=2):
    from jobs.fla_rwkv7 import train_mix as TM
    from train.data import Item

    rows = TM.rows_from_shapes(SHAPES * n_steps, vocab=260, seed=0)
    return [Item(x.id, x.row, i % x.row.n_options) for i, x in enumerate(rows)]


def _trainer(items, wkv="chunked", **flags):
    from heads.model import DecisionModel
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    bb = tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache")
    select_wkv(bb.model, wkv)
    model = DecisionModel(bb, "fork", {}, pad_id=0, seed=0)
    cfg = TrainConfig(head="fork", fork_mode="cache", lora_rank=4, loss=LossConfig(arm="clef"), batch_size=1,
                      grad_accum=8, max_steps=2, seed=0, **flags)
    tr = Trainer(model, cfg, items, [], guard=SplitGuard.synthetic_only("cell_perf equivalence"))
    tr._order = lambda epoch: list(range(len(items)))
    with torch.no_grad():  # non-zero LoRA B so the LoRA path carries signal into every gradient
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))) * 0.05)
    return tr


def _step_grads(tr):
    """One train step's loss and the gradients it applied (captured before the optimiser step)."""
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


def _compare(a, b, tol):
    (la, ga), (lb, gb) = a, b
    assert set(ga) == set(gb) and any(n.endswith("lora_A") for n in ga) and any(n.startswith("head.") for n in ga)
    # a gradient that is mathematically zero (the head's last bias: log_softmax is shift-invariant) is rounding noise
    # in both paths (~1e-8); it must stay noise (< 1e-6 of the largest gradient) and is not compared relatively
    scale = max(float(g.abs().max()) for g in gb.values())
    worst = 0.0
    for n in ga:
        ref = float(gb[n].abs().max())
        if ref < 1e-6 * scale:
            assert float(ga[n].abs().max()) < 1e-6 * scale, n
            continue
        worst = max(worst, float((ga[n] - gb[n]).abs().max()) / ref)
    assert abs(la - lb) <= tol * abs(lb) and worst <= tol, (la, lb, worst)
    return worst


@pytest.fixture(autouse=True)
def _vendored_registry_restored():
    yield
    M.RWKV7_WKV_FUNCTIONS.pop("chunked_lean", None)


def test_act_ckpt_equals_plain_step_with_cache_fork():
    items = _items()
    base = _step_grads(_trainer(items))
    tr = _trainer(items, act_ckpt=True)
    assert act_ckpt.is_enabled(tr.model.backbone.model)
    got = _step_grads(tr)
    assert got[0] == base[0]  # forward values are bit-identical
    _compare(got, base, 1e-6)
    # second step (after an optimiser update) still matches
    a, b = _trainer(items), _trainer(items, act_ckpt=True)
    a.train_step(), b.train_step()
    _compare(_step_grads(b), _step_grads(a), 1e-6)


def test_act_ckpt_shrinks_saved_activations_and_is_off_under_no_grad():
    items = _items(1)
    sizes = {}
    for flag in (False, True):
        tr = _trainer(items, act_ckpt=flag)
        tot = [0]

        def pack(t):
            tot[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fw = tr.model([items[2].row])  # the longest row (T=201)
        fw.log_probs.sum().backward()
        sizes[flag] = tot[0]
    assert sizes[True] * 2 < sizes[False], sizes  # only block inputs + states are kept
    tr = _trainer(items, act_ckpt=True)
    with torch.no_grad():  # eval: the wrapper runs the block as is (no checkpoint)
        a = tr.model([items[0].row]).log_probs
    act_ckpt.disable_block_checkpointing(tr.model.backbone.model)
    assert not act_ckpt.is_enabled(tr.model.backbone.model)
    assert all("forward" not in b.__dict__ for b in tr.model.backbone.model.blocks)
    with torch.no_grad():
        assert torch.equal(tr.model([items[0].row]).log_probs, a)


def test_act_ckpt_refuses_non_rwkv7_backbones():
    from heads.backbone import tiny_backbone

    with pytest.raises(ValueError, match="RWKV-7"):
        act_ckpt.enable_block_checkpointing(tiny_backbone().model)


def test_fork_features_rows_equal_per_row_fork():
    bb = tiny_hf_backbone("rwkv7", seed=1, fork_mode="cache")
    rows = [it.row for it in _items(1)[:5]]
    assert bb.can_fork_rows()
    per = [bb.fork_features(r.ids, r.continuations) for r in rows]
    h, forks = bb.fork_features_rows([r.ids for r in rows], [r.continuations for r in rows])
    assert h.shape == (5, max(len(r.ids) for r in rows), bb.d_model)
    for i, (r, (hp, fk)) in enumerate(zip(rows, per)):
        assert rel(h[i, : len(r.ids)], hp) <= 1e-5 and rel(forks[i], fk) <= 1e-5
        assert bool((h[i, len(r.ids):] == 0).all())  # zero padding, as the per-row path


@pytest.mark.parametrize("flags", [dict(row_batch=8), dict(row_batch=3), dict(row_batch=8, act_ckpt=True),
                                   dict(row_batch=8, row_tokens=400, act_ckpt=True),
                                   dict(row_batch=8, row_tokens=400, act_ckpt=True, row_plan="min_cost")])
def test_row_batch_equals_one_row_per_call(flags):
    items = _items()
    base = _step_grads(_trainer(items))
    tr = _trainer(items, **flags)
    assert tr.model.backbone.batch_rows
    _compare(_step_grads(tr), base, 2e-5)


def test_all_flags_together_equal_the_original_step():
    items = _items()
    base = _step_grads(_trainer(items))
    w = _compare(_step_grads(_trainer(items, wkv="chunked_lean", act_ckpt=True, row_batch=8, row_tokens=400)), base,
                 2e-5)
    assert w > 0  # not vacuous: the variants really ran a different computation


def test_row_groups_budget_and_partition():
    from train.trainer import row_groups

    assert row_groups([100, 300, 200, 50], 8) == [[1, 2, 0, 3]]
    # the mix's step 0 shape class: one ~1.9k row, the rest shorter; 8,192 padded tokens per call
    g = row_groups([1904, 900, 800, 500, 1000, 300, 200, 100], 8, 8192)
    assert g == [[0, 4, 1, 2], [3, 5, 6, 7]]
    assert row_groups([9000, 10], 8, 8192) == [[0], [1]]  # a row over the budget runs alone
    assert [len(x) for x in row_groups([10] * 10, 4)] == [4, 4, 2]
    for lens in ([5, 3, 9, 9, 1, 7, 2, 8], [2048] * 8, [4096] * 8, [1024] * 8):
        gs = row_groups(lens, 8, 8192)
        assert sorted(i for x in gs for i in x) == list(range(len(lens)))
        assert all(len(x) * max(lens[i] for i in x) <= 8192 or len(x) == 1 for x in gs)
    assert [len(x) for x in row_groups([2048] * 8, 8, 8192)] == [4, 4]
    assert [len(x) for x in row_groups([1024] * 8, 8, 8192)] == [8]


def test_row_groups_min_cost_partition_and_never_costlier_than_greedy():
    import random

    from train.trainer import GROUP_COST_TOKENS, row_groups, row_groups_min_cost

    def cost(lens, gs):
        return GROUP_COST_TOKENS * len(gs) + sum(len(x) * max(lens[i] for i in x) for x in gs)

    # job-tvuat mix step 6: greedy pads three short rows to the 2,025-token row (10,352 padded tokens, 2 groups);
    # min_cost runs that row alone and the other seven together (7,268 padded tokens, still 2 groups)
    lens = [749, 563, 520, 2025, 706, 317, 248, 674]
    assert row_groups_min_cost(lens, 8, 8192) == [[3], [0, 4, 7, 1, 2, 5, 6]]
    assert row_groups_min_cost([1024] * 8, 8, 8192) == [list(range(8))]
    assert row_groups_min_cost([9000, 10], 8, 8192) == [[0], [1]]
    rng = random.Random(0)
    for _ in range(300):
        lens = [rng.randint(1, 2048) for _ in range(rng.randint(1, 12))]
        rb, rt = rng.choice([(8, 8192), (3, 0), (8, 0), (4, 4096)])
        gs = row_groups_min_cost(lens, rb, rt)
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        assert [i for x in gs for i in x] == order  # contiguous cuts of the same sorted order: every row once
        assert all(len(x) <= rb and (not rt or len(x) == 1 or len(x) * lens[x[0]] <= rt) for x in gs)
        assert cost(lens, gs) <= cost(lens, row_groups(lens, rb, rt))


def test_row_plan_is_execution_only_and_validated():
    from train.trainer import TrainConfig, with_execution

    cfg = TrainConfig()
    assert cfg.row_plan == "greedy" and "row_plan" not in cfg.to_json()
    assert dataclasses.replace(cfg, row_plan="min_cost").hash() == cfg.hash()
    assert with_execution(cfg, {"row_batch": 8, "row_plan": "min_cost"}).row_plan == "min_cost"
    with pytest.raises(ValueError, match="row_plan"):
        _trainer(_items(1), row_batch=8, row_plan="best")


def test_row_batch_visits_the_same_rows_across_epoch_rollover():
    items = _items(1)  # 8 rows: grad_accum 8 x batch 1 = one epoch per step
    seen = {}
    for rb in (1, 8):
        tr = _trainer(items, row_batch=rb)
        tr._order = lambda epoch: list(range(len(items)))[::(-1 if epoch % 2 else 1)]
        log = []
        real = tr.model.forward
        tr.model.forward = lambda rows, **k: (log.extend(r.qid for r in rows), real(rows, **k))[1]
        for _ in range(3):
            tr.train_step()
        seen[rb] = (sorted(log), tr.epoch, tr.cursor)
    assert seen[1] == seen[8]


def test_defaults_are_the_original_path_and_hash_is_unchanged():
    from train.trainer import TrainConfig, with_execution

    cfg = TrainConfig()
    assert (cfg.act_ckpt, cfg.row_batch, cfg.row_tokens) == (False, 1, 0)
    assert dataclasses.replace(cfg, act_ckpt=True, row_batch=8, row_tokens=8192).hash() == cfg.hash()
    assert not set(TrainConfig.EXECUTION) & set(cfg.to_json())  # defaults: the serialized config is the old one
    assert dataclasses.replace(cfg, row_batch=8).to_json()["row_batch"] == 8  # non-default: recorded
    c2 = with_execution(cfg, {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192})
    assert (c2.act_ckpt, c2.row_batch, c2.row_tokens) == (True, 8, 8192) and with_execution(cfg, {}) == cfg
    tr = _trainer(_items(1))
    assert not act_ckpt.is_enabled(tr.model.backbone.model) and not tr.model.backbone.batch_rows
    with pytest.raises(ValueError, match="row_batch"):
        _trainer(_items(1), row_batch=0)


def _bf16(tr):
    tr.cfg = dataclasses.replace(tr.cfg, bf16=True)  # CPU autocast bf16, as the GPU step runs under CUDA autocast
    return tr


def _dev(a, b):
    """(loss rel, worst per-parameter gradient rel) of step ``a`` vs ``b`` (zero-gradient parameters skipped)."""
    (la, ga), (lb, gb) = a, b
    scale = max(float(g.abs().max()) for g in gb.values())
    w = max(float((ga[n] - gb[n]).abs().max()) / float(gb[n].abs().max()) for n in ga
            if float(gb[n].abs().max()) >= 1e-6 * scale)
    return abs(la - lb) / abs(lb), w


def test_under_bf16_autocast_the_variants_stay_inside_bf16_noise():
    """Under bf16 autocast the batched GEMM shapes round differently. The full variant's distance to the bf16 baseline
    must be below the bf16 baseline's own distance to the fp32 step, and its distance to the fp32 step no worse than
    1.25x the baseline's (measured: 1.2e-2 vs 2.6e-2, and 2.3e-2 vs 2.6e-2; loss 3e-8 vs baseline)."""
    items = _items()
    f32 = _step_grads(_trainer(items))
    base = _step_grads(_bf16(_trainer(items)))
    var = _step_grads(_bf16(_trainer(items, wkv="chunked_lean", act_ckpt=True, row_batch=8, row_tokens=400)))
    noise = _dev(base, f32)
    assert _dev(_step_grads(_bf16(_trainer(items, act_ckpt=True))), base) == (0.0, 0.0)  # checkpointing: bitwise
    lv, gv = _dev(var, base)
    assert lv <= 1e-6 and gv < noise[1], (lv, gv, noise)
    assert _dev(var, f32)[1] <= 1.25 * noise[1] and _dev(var, f32)[0] <= 1.25 * noise[0] + 1e-7

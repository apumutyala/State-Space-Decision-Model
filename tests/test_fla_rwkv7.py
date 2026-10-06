"""fla chunk_rwkv7 registration (heads/rwkv7_wkv.py) and its GPU gate (jobs/fla_rwkv7). CPU only, no fla needed.

The fla kernel itself is replaced here by ``fake_chunk``: a token-serial transcription of fla-core 0.5.2's DPLR
reference (``fla/ops/generalized_delta_rule/dplr/naive.py``: state (key, value), ``kv = k v^T + b (a^T S)``,
``S = S * exp(gk) + kv``, ``o = scale * q^T S``) with ``chunk_rwkv7``'s signature. That checks the MAPPING (a = -kk,
b = kk * a, scale 1, state transposes, packed contract, gradients into the initial state) against the vendored eager
WKV on CPU; the GPU job checks the real kernel against the same reference.

    python -X utf8 -m pytest -q tests/test_fla_rwkv7.py
"""
from __future__ import annotations

import json
import math
import pickle
import re
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heads import rwkv7_wkv as W  # noqa: E402
from jobs.fla_rwkv7 import fla_equivalence as E  # noqa: E402
from third_party.rwkv7_hf import modeling_rwkv7 as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F64 = torch.float64


# ----------------------------------------------------------------------------- the fla stand-in
def fake_chunk(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False, cu_seqlens=None,
               cu_seqlens_cpu=None, safe_gate=False, chunk_size=None, **kw):
    out_dtype = r.dtype
    cd = F64 if r.dtype == F64 else torch.float32  # fla accumulates in fp32 and returns o in q's dtype
    r, w, k, v, a, b = (t.to(cd) for t in (r, w, k, v, a, b))

    def run(sl, h):
        outs = []
        for t in range(sl.start, sl.stop):
            kv = k[:, t, :, :, None] * v[:, t, :, None, :] + b[:, t, :, :, None] * (a[:, t, :, :, None] * h).sum(-2, keepdim=True)
            h = h * w[:, t].exp()[..., None] + kv
            outs.append(scale * (r[:, t, :, :, None] * h).sum(-2))
        return torch.stack(outs, 1), h

    B, T, H, K = r.shape
    V = v.shape[-1]
    if cu_seqlens is None:
        h0 = initial_state if initial_state is not None else torch.zeros(B, H, K, V, dtype=torch.float32)
        o, h = run(range(0, T), h0.to(cd))
        return o.to(out_dtype), (h.float() if cd != F64 else h) if output_final_state else None
    assert B == 1
    bounds = cu_seqlens_cpu.tolist()
    os_, hs = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o, h = run(range(s, e), torch.zeros(1, H, K, V, dtype=r.dtype))
        os_.append(o)
        hs.append(h)
    return torch.cat(os_, 1).to(out_dtype), torch.cat(hs, 0).float() if cd != F64 else torch.cat(hs, 0)


@pytest.fixture
def fake_fla(monkeypatch):
    """fla 'available' and every call eligible: the wrapper's fla branch runs on CPU through fake_chunk."""
    monkeypatch.setattr(W, "_import_chunk", lambda: (fake_chunk, None))
    monkeypatch.setattr(W, "fallback_reason", lambda r, cu, allow_fp32: None)
    W.FALLBACKS.clear()
    W.CALLS.clear()
    yield
    W.FALLBACKS.clear()
    W.CALLS.clear()


def inputs(B=2, T=37, H=3, N=8, seed=0, dtype=F64, state_scale=0.5):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=F64)  # noqa: E731
    x = dict(r=rn(B, T, H, N), w_log=-0.6065306597 * torch.sigmoid(rn(B, T, H, N) * 2 - 0.5), k=rn(B, T, H, N),
             v=rn(B, T, H, N), kk=torch.nn.functional.normalize(rn(B, T, H, N), dim=-1), a=torch.sigmoid(rn(B, T, H, N)))
    x = {n: t.to(dtype) for n, t in x.items()}
    x["state"] = (rn(B, H, N, N) * state_scale).to(dtype)
    return x


def args(x):
    return [x[n] for n in E.ARGS]


# ----------------------------------------------------------------------------- the vendored hook we rely on
def test_vendored_tokenwise_sites_and_registry_hook_are_what_wkvname_assumes():
    src = (ROOT / "third_party/rwkv7_hf/modeling_rwkv7.py").read_text(encoding="utf-8")
    assert len(re.findall(r'wkv_implementation != "chunked"', src)) == 3  # attention, FFN, LM head
    assert "== \"chunked\"" not in src and "in (\"chunked\"" not in src  # WkvName only overrides !=
    assert "wkv = RWKV7_WKV_FUNCTIONS[self.config.wkv_implementation]" in src
    import third_party.rwkv7_hf as V

    assert all(V.vendored_ok().values())  # we register from outside; the vendored bytes stay pinned


def test_wkvname_semantics():
    n = W.WkvName("fla")
    assert n == "fla" and str(n) == "fla" and not (n == "chunked")
    assert (n != "chunked") is False and (n != "eager") is True and (n != "fla") is False
    assert {"fla": 1}[n] == 1 and hash(n) == hash("fla")
    assert json.dumps({"wkv": n}) == '{"wkv": "fla"}'
    m = pickle.loads(pickle.dumps(n))
    assert type(m) is W.WkvName and (m != "chunked") is False


# ----------------------------------------------------------------------------- selection by config
def tiny_lm(seed=0):
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config

    cfg = Rwkv7Config(vocab_size=64, hidden_size=24, num_heads=3, head_dim=8, num_hidden_layers=2, intermediate_size=48,
                      a_low_rank_dim=4, decay_low_rank_dim=4, gate_low_rank_dim=4, v_low_rank_dim=4)
    torch.manual_seed(seed)
    m = M.Rwkv7ForCausalLM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.1)
    return m.eval()


def test_select_default_off_and_refuses_unverified():
    m = tiny_lm()
    assert W.GPU_VERIFIED is False  # flipped only after jobs/fla_rwkv7 returns ADOPT
    assert W.select_wkv(m, None) == "chunked" and m.config.wkv_implementation == "chunked"
    with pytest.raises(RuntimeError, match="not GPU-verified"):
        W.select_wkv(m, "fla")
    assert m.config.wkv_implementation == "chunked"
    with pytest.raises(ValueError, match="unknown"):
        W.select_wkv(m, "flash", allow_unverified=True)


def test_select_registers_and_sets_every_module_config():
    m = tiny_lm()
    assert W.select_wkv(m, "fla", allow_unverified=True) == "fla"
    assert {"fla", "fla_safe_gate"} <= set(M.RWKV7_WKV_FUNCTIONS)
    cfgs = W._configs(m)
    assert cfgs and all(type(c.wkv_implementation) is W.WkvName for c in cfgs)
    assert all(type(mod.config.wkv_implementation) is W.WkvName for mod in m.modules() if hasattr(mod, "config"))
    assert W.select_wkv(m, "chunked") == "chunked" and type(m.config.wkv_implementation) is str


def test_loaders_take_wkv_from_config_default_none():
    import inspect

    from decide.backends.hf_readout import load_rwkv7_hf
    from heads.backbone import load_rwkv7_hf_backbone

    for fn in (load_rwkv7_hf, load_rwkv7_hf_backbone):
        assert inspect.signature(fn).parameters["wkv"].default is None
    assert 'wkv=cfg.get("rwkv7_wkv")' in (ROOT / "train/run_cell.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- clean fallback (no fla on this box)
def test_fla_selected_on_cpu_falls_back_to_chunked_bitwise_with_batched_linears(monkeypatch):
    seen = []
    real = M._rwkv7_linear
    monkeypatch.setattr(M, "_rwkv7_linear", lambda mod, x, tokenwise: seen.append(tokenwise) or real(mod, x, tokenwise))
    W.FALLBACKS.clear()
    ids = torch.randint(1, 64, (2, 19), generator=torch.Generator().manual_seed(1))
    m = tiny_lm()
    with torch.no_grad():
        ref = m(input_ids=ids).logits
        seen.clear()
        W.select_wkv(m, "fla", allow_unverified=True)
        got = m(input_ids=ids).logits
    assert torch.equal(got, ref)  # the fallback IS "chunked", same batched Linears
    assert seen and not any(seen)  # tokenwise never switched on by the fla name
    assert sum(W.FALLBACKS.values()) == 2 and set(W.FALLBACKS) <= {"not_cuda", "fla_unavailable:ModuleNotFoundError",
                                                                   "fla_unavailable:ImportError"}
    W.FALLBACKS.clear()


def test_fallback_reasons():
    def fake(is_cuda=True, dtype=torch.bfloat16, shape=(1, 8, 64, 64)):
        return types.SimpleNamespace(is_cuda=is_cuda, dtype=dtype, shape=shape)

    assert W.fallback_reason(fake(), None, False) is None
    assert W.fallback_reason(fake(is_cuda=False), None, False) == "not_cuda"
    assert W.fallback_reason(fake(dtype=torch.float32), None, False) == "fp32"
    assert W.fallback_reason(fake(dtype=torch.float32), None, True) is None
    assert W.fallback_reason(fake(dtype=torch.float64), None, True) == "dtype:torch.float64"
    assert W.fallback_reason(fake(shape=(2, 8, 64, 64)), torch.tensor([0, 4, 8]), False) == "packed_batch"
    assert W.fallback_reason(fake(), torch.tensor([0, 4, 7]), False) == "packed_bounds"
    assert W.fallback_reason(fake(), torch.tensor([0, 4, 4, 8]), False) is None


def test_packed_bounds():
    assert W._packed_bounds(torch.tensor([0, 3, 3, 9]), 9) == [0, 3, 9]
    assert W._packed_bounds(torch.tensor([1, 9]), 9) is None
    assert W._packed_bounds(torch.tensor([0, 5]), 9) is None
    assert W._packed_bounds(torch.tensor([0, 5, 4, 9]), 9) is None
    assert W._packed_bounds(torch.tensor([0, 0]), 0) is None


def test_wrapper_without_fla_uses_parallel(monkeypatch):
    monkeypatch.setattr(W, "_import_chunk", lambda: (None, "fla_unavailable:ImportError"))
    W.FALLBACKS.clear()
    x = inputs(dtype=torch.float32)
    fn = W.make_fla_wkv()
    o, s = fn(*args(x))
    o2, s2 = M.rwkv7_parallel(*args(x))
    assert torch.equal(o, o2) and torch.equal(s, s2) and W.FALLBACKS["fla_unavailable:ImportError"] == 1
    W.FALLBACKS.clear()


# ----------------------------------------------------------------------------- the mapping, against eager
def test_mapping_matches_eager_outputs_and_state(fake_fla):
    x = inputs()
    o, s = W.fla_wkv_call(fake_chunk, *args(x))
    eo, es = M.rwkv7_recurrent(*args(x), compute_dtype=F64)
    assert o.shape == eo.shape and s.shape == es.shape and s.dtype == x["state"].dtype
    assert E.rel_err(o, eo) < 1e-6 and E.rel_err(s, es) < 1e-6


def test_mapping_gradients_including_initial_state(fake_fla):
    x = inputs(T=13)
    grads = []
    for impl in ("fla", "eager"):
        leaves = {n: t.clone().requires_grad_(True) for n, t in x.items()}
        if impl == "fla":
            o, s = W.fla_wkv_call(fake_chunk, *args(leaves))
        else:
            o, s = M.rwkv7_recurrent(*args(leaves), compute_dtype=F64)
        g = torch.Generator().manual_seed(3)
        ((o * torch.randn(o.shape, generator=g, dtype=F64)).sum() + (s * torch.randn(s.shape, generator=g, dtype=F64)).sum()).backward()
        grads.append({n: leaves[n].grad for n in E.ARGS})
    for n in E.ARGS:
        assert grads[0][n] is not None and E.rel_err(grads[0][n], grads[1][n]) < 1e-6, n


def test_packed_contract_matches_eager(fake_fla):
    x = inputs(B=1, T=30)
    cu = torch.tensor([0, 7, 7, 8, 30])  # an empty and a 1-token segment
    o, s = W.fla_wkv_call(fake_chunk, *args(x), cu)
    eo, es = M.rwkv7_eager(*[t.double() for t in args(x)], cu_seq_lens=cu)  # fp32 compute inside -> loose tol
    assert o.shape == eo.shape and s.shape == es.shape
    assert E.rel_err(o, eo) < 1e-5 and E.rel_err(s, es) < 1e-5


def test_state_split_and_fork_equal_one_shot(fake_fla):
    x = inputs(B=1, T=40)
    o, s = W.fla_wkv_call(fake_chunk, *args(x))
    sl = lambda a, b: {n: (t[:, a:b] if n != "state" else t) for n, t in x.items()}  # noqa: E731
    _, s1 = W.fla_wkv_call(fake_chunk, *args(sl(0, 25)))
    o2, s2 = W.fla_wkv_call(fake_chunk, *args({**sl(25, None), "state": s1}))
    assert E.rel_err(o2, o[:, 25:]) < 1e-6 and E.rel_err(s2, s) < 1e-6  # the carried state is fp32
    tail = {n: t.repeat(3, 1, 1, 1) for n, t in sl(25, None).items() if n != "state"}
    oK, _ = W.fla_wkv_call(fake_chunk, *args({**tail, "state": s1.repeat_interleave(3, 0)}))
    assert all(E.rel_err(oK[i:i + 1], o2) < 1e-12 for i in range(3))


def test_model_through_fla_branch_matches_eager_model(fake_fla):
    ids = torch.randint(1, 64, (2, 21), generator=torch.Generator().manual_seed(2))
    m = tiny_lm().double()
    with torch.no_grad():
        W.select_wkv(m, "eager")
        ref = m(input_ids=ids).logits
        W.select_wkv(m, "fla", allow_unverified=True)
        got = m(input_ids=ids).logits
        # continuation through the model's own state object
        st = m(input_ids=ids[:, :13], use_cache=True).state
        tail = m(input_ids=ids[:, 13:], state=st, use_cache=True).logits
    assert W.CALLS["fla"] > 0 and not W.FALLBACKS
    assert E.rel_err(got, ref) < 1e-5  # eager computes its products in fp32; fake runs fp64
    assert E.rel_err(tail, got[:, 13:]) < 1e-10


def test_decision_path_cache_fork_lora_gradients_through_fla_branch(fake_fla):
    """The m2 training path (HiddenStateBackbone cache fork + LoRA) gives the same gradients on the fla branch as the
    recompute fork on the vendored chunked path."""
    from torch import nn

    from heads import build_row
    from heads.backbone import tiny_hf_backbone
    from heads.features import build_head_input
    from tests.test_m2_review_fixes import ENC, req
    from train.data import TOY_TEMPLATE
    from train.lora import apply_lora

    rows = [build_row(req(), q, ENC, template=TOY_TEMPLATE, layout="ordered") for q in ("dept", "down", "urg")]
    grads = []
    for mode, wkv in (("cache", "fla"), ("recompute", "chunked")):
        b = tiny_hf_backbone("rwkv7", fork_mode=mode)
        b.model.double()
        W.select_wkv(b.model, wkv, allow_unverified=True)
        torch.manual_seed(5)
        apply_lora(b.model, b.spec.lora_groups, rank=2)
        for n, p in b.model.named_parameters():
            if n.endswith("lora_B"):
                nn.init.normal_(p, std=0.1)
        b.model.double()
        x = build_head_input(b, rows, need_fork=True)
        x.fork.pow(2).sum().backward()
        grads.append({n: p.grad.clone() for n, p in b.model.named_parameters() if p.requires_grad})
    assert W.CALLS["fla"] > 0 and not W.FALLBACKS
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], atol=1e-5, rtol=1e-4), k


# ----------------------------------------------------------------------------- the gate's pure parts
def test_preregistered_constants_are_pinned():
    assert E.TOL == {"bfloat16": {"out": 2e-2, "state": 2e-2, "grad": 5e-2, "cos": 0.999},
                     "float32": {"out": 2e-3, "state": 2e-3, "grad": 5e-3, "cos": 0.99999}}
    assert (E.SPEEDUP_MIN, E.LENGTHS, E.TRAIN_T, E.REPS) == (2.0, (2048, 8192, 16384), 2048, 3)
    assert (E.MODEL_REL_FLOOR, E.MODEL_SLACK, E.MODEL_COS_MIN, E.MODEL_TOP1_MIN) == (2e-2, 1.5, 0.999, 0.98)
    assert (E.CONT_REL_MAX, E.FORK_REL_MAX, E.SAFE_GATE_MARGIN, E.ABS_FLOOR) == (2e-2, 1e-5, 1.10, 1e-6)
    spec = (ROOT / "jobs/fla_rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    for s in ("2e-2", "5e-2", "0.999", "2e-3", "5e-3", "0.99999", "≥ 2.0×", "1.5 ×", "0.98", "1e-5", "1.10×"):
        assert s in spec, s
    assert "max(‖ref‖∞, 1e-6)" in spec


def test_metrics():
    a = torch.tensor([1.0, -2.0, 4.0], dtype=F64)
    assert E.rel_err(a, a) == 0 and abs(E.rel_err(a + 0.04, a) - 0.01) < 1e-9
    assert E.cosine(a, a) == pytest.approx(1.0) and E.cosine(a, -a) == pytest.approx(-1.0)
    c = E.compare({"out": a, "state": a, "grad_r": a * 1.1}, {"out": a, "state": a, "grad_r": a})
    assert c["worst"]["out"] == 0 and c["worst"]["grad"] == pytest.approx(0.1) and c["worst"]["finite"]
    assert not E.case_passes(c["worst"], "bfloat16")  # grad 0.1 > 5e-2
    assert E.case_passes({**c["worst"], "grad": 0.04}, "bfloat16")
    assert not E.case_passes({"out": float("nan"), "cos": 1.0, "finite": True}, "bfloat16")
    assert not E.case_passes({"out": 0.0, "cos": 1.0, "finite": False}, "bfloat16")
    tiny = torch.full((3,), 1e-40, dtype=F64)
    assert E.rel_err(tiny, torch.zeros(3, dtype=F64)) < 1e-30 and E.cosine(tiny, -tiny) == 1.0  # below ABS_FLOOR
    assert E.rel_err(torch.tensor([float("nan")]), a[:1]) == float("inf")


def _report(fla_ok=True, safe_ok=True, pre=(2.5, 3.0, 4.0), train=2.4, safe_scale=1.0, fp32_ok=True, drop=None):
    good = {"out": 1e-3, "state": 1e-3, "grad": 1e-3, "cos": 0.99999, "finite": True}
    bad = {**good, "out": 0.5}

    def cases(phase):
        out = []
        for v, ok in (("chunked", True), ("fla", fla_ok), ("fla_safe_gate", safe_ok)):
            out.append({"name": f"{phase}1", "impl": v, "dtype": "bfloat16", "worst": good if ok else bad})
        return {"cases": out}

    fp32 = {"complete": True, "cases": [{"name": f"fp32_{i}", "impl": v, "dtype": "float32",
                                         "worst": good if fp32_ok else bad}
                                        for i in range(2) for v in ("chunked", "fla", "fla_safe_gate")]}

    tp = {"fla_calls": {"fla": 9, "fla_safe_gate": 9}, "prefill": {}, "train": {"chunked": {"tok_s": 1000.0}, "fla": {"tok_s": 1000.0 * train},
                                   "fla_safe_gate": {"tok_s": 1000.0 * train * safe_scale}}}
    for L, rr in zip(E.LENGTHS, pre):
        tp["prefill"][str(L)] = {"chunked": {"tok_s": 4000.0}, "fla": {"tok_s": 4000.0 * rr},
                                 "fla_safe_gate": {"tok_s": 4000.0 * rr * safe_scale}}
    logit = {"rel": 1e-2, "cos_min": 0.9995, "top1": 0.99}
    rep = {"fp32": fp32, "kernel": cases("kernel"), "fork": cases("fork"), "captured": cases("captured"),
           "model": {"fla_calls": {"fla": 9, "fla_safe_gate": 9}, "logits": {"chunked": {"rel": 1e-2}, "fla": logit, "fla_safe_gate": logit},
                     "continuation": {"fla": {"rel": 1e-3, "fork_rel": 0.0}, "fla_safe_gate": {"rel": 1e-3, "fork_rel": 0.0}}},
           "throughput": tp}
    if drop:
        rep[drop] = None
    return rep


def test_decide_rule():
    d = E.decide(_report())
    assert d["verdict"] == "ADOPT" and d["chosen"] == "fla" and E.valid_decision(d)
    assert d["variants"]["fla"]["allow_fp32"] is True
    assert d["chosen_prefill_2048_tok_s"] == 10000.0 and not d["meets_child_eval_rate"]
    assert E.decide(_report(fp32_ok=False))["variants"]["fla"]["allow_fp32"] is False  # fp32 never blocks
    assert E.decide(_report(fp32_ok=False))["verdict"] == "ADOPT"


def test_partial_fp32_never_allows_fp32():
    """PR #59 B1: a timed-out fp32 phase leaves fp32.json with only its first (forward) cases, all passing. Without
    the ``complete`` flag that must not enable fp32."""
    r = _report()
    assert E.decide(r)["variants"]["fla"]["allow_fp32"] is True
    r["fp32"]["complete"] = False
    assert E.decide(r)["variants"]["fla"]["allow_fp32"] is False
    del r["fp32"]["complete"]
    assert E.decide(r)["variants"]["fla"]["allow_fp32"] is False
    r = _report()
    r["fp32"] = None  # skipped
    assert E.decide(r)["variants"]["fla"]["allow_fp32"] is False and E.decide(r)["verdict"] == "ADOPT"


def test_phase_fp32_marks_complete_only_after_its_last_case(fake_fla, monkeypatch, tmp_path):
    monkeypatch.setattr(E, "DEVICE", "cpu")
    monkeypatch.setattr(E, "H", 2)
    monkeypatch.setattr(E, "N", 4)
    seen = []
    real = E._fork_cases

    def interrupted(rep, dt):  # a timeout during the fork half: the file on disk at that moment
        seen.append(json.loads((tmp_path / "fp32.json").read_text(encoding="utf-8")))
        raise KeyboardInterrupt

    monkeypatch.setattr(E, "_fork_cases", interrupted)
    with pytest.raises(KeyboardInterrupt):
        E.phase_fp32(tmp_path)
    partial = seen[0]
    assert partial["complete"] is False and partial["cases"]  # forward/grad kernel cases present, all passing
    assert E._fp32_ok({"fp32": partial}, "fla") is False
    monkeypatch.setattr(E, "_fork_cases", real)
    E.phase_fp32(tmp_path)
    full = json.loads((tmp_path / "fp32.json").read_text(encoding="utf-8"))
    assert full["complete"] is True and E._fp32_ok({"fp32": full}, "fla") is True


def test_warmup_autotunes_at_a_real_length_first():
    """PR #59 N1: fla's autotune keys exclude T, so the first warmup call fixes the config for the throughput phase."""
    assert E.WARMUP_T[0] == 2048 and E.WARMUP_T[0] in E.LENGTHS and any(t % 16 for t in E.WARMUP_T)
    assert E.decide(_report(pre=(1.9, 3.0, 4.0)))["verdict"] == "ADOPT_TRAINING_ONLY"  # every length must clear 2x
    assert E.decide(_report(train=1.5))["verdict"] == "ADOPT_READOUT_ONLY"
    assert E.decide(_report(pre=(1.2, 1.5, 1.9), train=1.5))["verdict"] == "REJECT_SPEED"
    assert E.decide(_report(fla_ok=False, safe_ok=False))["verdict"] == "REJECT_NUMERICS"
    for phase in ("kernel", "fork", "captured", "model", "throughput"):
        assert E.decide(_report(drop=phase))["verdict"] == "INCONCLUSIVE", phase


def test_decide_model_level_and_continuation():
    r = _report()
    r["model"]["logits"]["fla"] = {"rel": 2.9e-2, "cos_min": 0.9995, "top1": 0.99}  # > max(2e-2, 1.5 * 1e-2)
    assert E.decide(r)["variants"]["fla"]["verdict"] == "REJECT_NUMERICS"
    r["model"]["logits"]["chunked"] = {"rel": 2e-2}  # chunked itself that far off -> 1.5x slack covers 2.9e-2
    assert E.decide(r)["variants"]["fla"]["verdict"] == "ADOPT"
    r = _report()
    r["model"]["continuation"]["fla"]["fork_rel"] = 1e-3
    assert E.decide(r)["variants"]["fla"]["verdict"] == "REJECT_NUMERICS"
    r = _report()
    r["kernel"]["cases"].append({"name": "x", "impl": "fla", "dtype": "bfloat16", "error": "RuntimeError: boom"})
    assert E.decide(r)["variants"]["fla"]["verdict"] == "REJECT_NUMERICS"


def test_safe_gate_needs_margin_to_be_chosen():
    assert E.decide(_report(safe_scale=1.05))["chosen"] == "fla"
    assert E.decide(_report(safe_scale=1.2))["chosen"] == "fla_safe_gate"
    assert E.decide(_report(fla_ok=False))["chosen"] == "fla_safe_gate"
    assert E.decide(_report(safe_ok=False, safe_scale=2.0))["chosen"] == "fla"


def test_decide_cli_voids_fallback_measurements(tmp_path):
    rep = _report()
    rep["throughput"]["fallbacks"] = {"not_cuda": 3}
    for k, v in rep.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    assert E.main(["decide", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))["verdict"] == "INCONCLUSIVE"
    rep = _report()
    for k, v in rep.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    assert E.main(["decide", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))["verdict"] == "ADOPT"
    rep["model"]["fla_calls"] = {"fla": 9}  # fla_safe_gate never reached the kernel -> its logits are not fla's
    (tmp_path / "model.json").write_text(json.dumps(rep["model"]), encoding="utf-8")
    E.main(["decide", "--out", str(tmp_path)])
    assert json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))["verdict"] == "INCONCLUSIVE"


# ----------------------------------------------------------------------------- the job files
def test_job_params_cost_and_schedule():
    from jobs.common import costs

    p = json.loads((ROOT / "jobs/fla_rwkv7/job_params.json").read_text(encoding="utf-8"))
    assert costs.worst_case_usd(p) <= 4.5 and costs.worst_case_usd(p) <= p["hard_cap_usd"]
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert p["chip"] == "h100" and p["max_restarts"] == 0 and p["assumed_restart_floor"] >= 1
    spec = (ROOT / "jobs/fla_rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"${costs.worst_case_usd(p):.2f}" in spec


def test_runner_and_context():
    s = (ROOT / "jobs/fla_rwkv7/fla_job.sh").read_text(encoding="utf-8")
    for ph in ("warmup", "kernel", "fork", "model", "throughput", "fp32", "decide"):
        assert f"jobs.fla_rwkv7.fla_equivalence {ph}" in s
    assert s.index("fla_guard.py") < s.index("snapshot_download") < s.index("fla_equivalence warmup") \
        < s.index("fla_equivalence kernel")
    # job-4idg4: one Triton cache for every phase process; fp32 is last, optional, bounded by the time left
    assert "export TRITON_CACHE_DIR=$WORK/triton-cache" in s
    assert s.index("fla_equivalence throughput") < s.index("fla_equivalence fp32") < s.index("fla_equivalence decide")
    fp32 = s[s.index("fla_equivalence fp32"):s.index("fla_equivalence decide")]
    assert "fail " not in fp32 and "timeout ${LEFT_S}s" in s
    assert "--require-hashes" in s and "VALIDATOR=jobs.fla_rwkv7.fla_equivalence:valid_decision" in s
    bc = (ROOT / "jobs/fla_rwkv7/build_context.sh").read_text(encoding="utf-8")
    assert 'cp "$OUTD/code/m1/jobs/rwkv7/Dockerfile" "$OUTD/Dockerfile"' in bc  # the rwkv7 image, unchanged
    req = (ROOT / "jobs/fla_rwkv7/requirements-fla.txt").read_text(encoding="utf-8")
    assert req.strip() == ("fla-core==0.5.2 --hash=sha256:"
                           "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761")
    assert W.FLA_CORE_PIN == "0.5.2" and "5e830c85bad3d0d3" in (ROOT / "heads/rwkv7_wkv.py").read_text(encoding="utf-8")


def test_weight_free_gate_phases_run_end_to_end_on_cpu(fake_fla, monkeypatch, tmp_path):
    """Smoke of the GPU job's kernel + fork phases (same code, CPU, small heads, the stand-in kernel): every case runs,
    the stand-in passes its own tolerances, and the partial report decides INCONCLUSIVE (no model / throughput)."""
    monkeypatch.setattr(E, "DEVICE", "cpu")
    monkeypatch.setattr(E, "H", 4)
    monkeypatch.setattr(E, "N", 16)
    monkeypatch.setattr(E, "_p640", lambda rep: None)
    monkeypatch.setattr(E, "WARMUP_T", (64, 40))  # CPU stand-in: the real lengths only cost time here
    E.phase_warmup(tmp_path)
    E.phase_kernel(tmp_path)
    E.phase_fork(tmp_path)
    E.phase_fp32(tmp_path)
    warm = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))["cases"]
    assert len(warm) == 16 and all("compile_s" in c and "t_s" in c for c in warm)  # 2 variants x 2 T x state/packed x fwd/bwd
    for phase, dt in (("kernel", "bfloat16"), ("fork", "bfloat16"), ("fp32", "float32")):
        assert {c["dtype"] for c in json.loads((tmp_path / f"{phase}.json").read_text(encoding="utf-8"))["cases"]} == {dt}
    for phase in ("kernel", "fork", "fp32"):
        cases = json.loads((tmp_path / f"{phase}.json").read_text(encoding="utf-8"))["cases"]
        assert cases and not [c for c in cases if c.get("error")], [c for c in cases if c.get("error")][:2]
        gated = [c for c in cases if not c.get("diag")]
        assert {c["impl"] for c in gated} == {"chunked", "fla", "fla_safe_gate"}
        bad = [(c["name"], c["impl"], c["dtype"], c["worst"]) for c in gated if not c["passes"]]  # chunked too
        assert not bad, bad
    assert W.CALLS == {} and not W.FALLBACKS  # the harness calls the kernel directly, never through a fallback
    assert E.main(["decide", "--out", str(tmp_path)]) == 0
    d = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert d["verdict"] == "INCONCLUSIVE" and d["variants"]["fla"]["allow_fp32"] is True  # from fp32.json
    (tmp_path / "fp32.json").unlink()  # optional phase skipped -> allow_fp32 false, verdict unaffected
    E.main(["decide", "--out", str(tmp_path)])
    d2 = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert d2["variants"]["fla"]["allow_fp32"] is False and d2["verdict"] == d["verdict"]


# ----------------------------------------------------------------------------- RULE v2 (pre-registered 2026-10-06)
def test_rule_v2_constants_are_pinned():
    assert E.TRUTH_V2_LAYERS == (0, 15, 30, 45, 60)
    assert (E.V2_PROMPT_T, E.V2_FWD_T, E.V2_GRAD_T) == (2048, 1024, 512)
    assert (E.V2_SLACK, E.V2_FRAC_MIN, E.ABS_FLOOR) == (1.10, 0.90, 1e-6)
    assert E.V2_REF_IMPLS == ("chunked", "eager_bf16")
    assert E.V2_FWD_TENSORS == ("out", "state")
    assert E.V2_GRAD_TENSORS == ("out", "state", "grad_r", "grad_w_log", "grad_k", "grad_v", "grad_kk", "grad_a",
                                 "grad_state")
    assert (E.V2_CURRENT, E.V2_TRAIN_WARMUP, E.V2_TRAIN_STEPS) == ("chunked", 3, 20)
    assert (E.V2_MIN_MATCHED_STEPS, E.V2_STEP0_LOSS_REL_MAX) == (10, 0.02)  # Amendment A1
    assert not hasattr(E, "V2_CURRENT_MIN_STEPS") and not hasattr(E, "train_speed_v2")  # withdrawn by A1
    assert (E.V2_TRAIN_RATIO_MIN, E.V2_PREFILL_RATIO_MIN, E.V2_PEAK_GB_MAX) == (2.0, 2.0, 76.0)
    assert E.LENGTHS == (2048, 8192, 16384)
    assert E.VERDICTS_V2 == ("ADOPT_FOR_TRAINING", "ADOPT_READOUT_ONLY", "REJECT_NUMERICS", "INCONCLUSIVE",
                             "NOT_ADOPTED")
    assert len(E.v2_required_cases()) == 10 and sum(map(len, E.v2_required_cases().values())) == 55
    spec = (ROOT / "jobs/fla_rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    rule = spec[spec.index("## RULE v2"):]
    for s in ("pre-registered 2026-10-06, before job execution", "fp64 token-serial WKV", "{0, 15, 30, 45, 60}",
              "e(fla) <= 1.10 * max(e(chunked), e(eager_bf16)) + ABS_FLOOR",
              "e(fla) <= 1.10 * e(chunked) + ABS_FLOOR on at least 90% of (case,tensor) pairs",
              "20 timed steps after warmup",
              "ADOPT_FOR_TRAINING iff NUMERICS_OK and train ratio >= 2.0 and fla peak <= 76 GB",
              "ADOPT_READOUT_ONLY iff NUMERICS_OK and prefill ratio >= 2 at all lengths but train ratio < 2",
              "REJECT_NUMERICS iff NUMERICS_OK false", "INCONCLUSIVE if anything missing", "decide_v2", "job-jn47u",
              "NOT_ADOPTED", "Amendment A1 (2026-10-06, after independent review, before any v2 run)",
              "completed (matched rows)", "Require ≥ 10 matched steps",
              "chunked cannot keep T_max=2048 at the cell config", "|step-0 loss rel-diff| ≤ 0.02"):
        assert s in rule, s


def _truth_v2(e_fla=0.03, e_ch=0.03, e_eg=0.05, e_safe=None, override=None, drop=None):
    """Hand-built truth_v2.json: every required case/tensor, e(impl) constant unless ``override`` names
    (case, tensor, impl) -> value; ``drop`` removes a case."""
    override = override or {}
    cases = []
    for name, tensors in E.v2_required_cases().items():
        if name == drop:
            continue
        impls = {}
        for impl, val in (("eager_bf16", e_eg), ("chunked", e_ch), ("fla", e_fla),
                          ("fla_safe_gate", e_fla if e_safe is None else e_safe)):
            impls[impl] = {"finite": True, "e": {t: override.get((name, t, impl), val) for t in tensors}}
        cases.append({"name": name, "tensors": list(tensors), "truth_finite": True, "impls": impls})
    return {"cases": cases, "layers": list(E.TRUTH_V2_LAYERS)}


def _train_v2(cur=1000.0, fla=3000.0, safe=None, peak=60.0, cur_oom=0, fla_oom=0, n_fla=20, n_cur=20,
              fallbacks=None, loss=1.0, fla_loss=None, cur_tok=None, fla_tok=None):
    """Hand-built train_v2 records. ``*_oom``: the first n timed steps OOM; ``n_*``: timed steps reached (the rest are
    skipped by the deadline); ``*_tok``: per-timed-step tok/s lists overriding the constants."""
    def rec(impl, tok, n, oom, pk, lo, toks):
        steps = [{"i": i, "timed": False, "tok_s": tok, "loss": lo[i] if isinstance(lo, list) else lo, "finite": True,
                  "max_T": 500} for i in range(E.V2_TRAIN_WARMUP)]
        for j in range(E.V2_TRAIN_STEPS):
            st = {"i": E.V2_TRAIN_WARMUP + j, "timed": True, "max_T": 1000 + j}
            if j >= n:
                st["skipped"] = "deadline"
            elif j < oom:
                st["oom"] = True
            else:
                st.update(tok_s=toks[j] if toks else tok, loss=0.9, finite=True)
            steps.append(st)
        r = {"impl": impl, "steps": steps, "peak_gb": pk, "fallbacks": {}, "fla_calls": {}}
        if impl != "chunked":
            r["fla_calls"] = {impl: 999}
            r["fallbacks"] = dict(fallbacks or {})
        return r

    fl = loss if fla_loss is None else fla_loss
    return {"chunked": rec("chunked", cur, n_cur, cur_oom, 70.0, loss, cur_tok),
            "fla": rec("fla", fla, n_fla, fla_oom, peak, fl, fla_tok),
            "fla_safe_gate": rec("fla_safe_gate", fla if safe is None else safe, n_fla, fla_oom, peak, fl, fla_tok)}


def _report_v2(truth=None, train=None, pre=(2.5, 3.0, 4.0), **kw):
    rep = _report(pre=pre, **kw)  # v1 synthetic phases (kernel / fork / model / throughput), all passing by default
    rep.pop("captured")
    rep.pop("fp32")
    rep["truth_v2"] = _truth_v2() if truth is None else truth
    rep["train_v2"] = _train_v2() if train is None else train
    return rep


def test_decide_v2_every_verdict():
    seen = set()

    def dv(rep):
        d = E.decide_v2(rep)
        assert E.valid_decision_v2(d)
        seen.add(d["verdict"])
        return d

    d = dv(_report_v2())
    assert d["verdict"] == "ADOPT_FOR_TRAINING" and d["rule"] == "v2"
    assert d["variants"]["fla"]["train_ratio"] == pytest.approx(3.0)
    assert d["variants"]["fla"]["numerics"]["truth_detail"]["pairs"] == 55
    # train ratio < 2 but prefill >= 2 at every length -> readout only
    assert dv(_report_v2(train=_train_v2(fla=1500.0)))["verdict"] == "ADOPT_READOUT_ONLY"
    # numerics: one pair beyond 1.10 x max(chunked, eager) + ABS_FLOOR -> reject
    t = _truth_v2(override={("real_L30_grad_T512", "grad_k", "fla"): 0.0551})
    d = dv(_report_v2(truth=t))
    assert d["verdict"] == "REJECT_NUMERICS"
    assert "truth_v2:real_L30_grad_T512:grad_k" in d["variants"]["fla"]["numerics"]["why"]
    # missing pieces -> inconclusive
    assert dv(_report_v2(truth=_truth_v2(drop="real_L45_fwd_T1024")))["verdict"] == "INCONCLUSIVE"
    assert dv(_report_v2(train=_train_v2(n_fla=9)))["verdict"] == "INCONCLUSIVE"  # < 10 matched (A1)
    r = _report_v2()
    r["train_v2"].pop("chunked")
    assert dv(r)["verdict"] == "INCONCLUSIVE"
    r = _report_v2()
    r["throughput"]["prefill"]["8192"].pop("fla")
    assert dv(r)["verdict"] == "INCONCLUSIVE"
    r = _report_v2()
    r["truth_v2"] = None
    assert dv(r)["verdict"] == "INCONCLUSIVE"
    # measured, but outside every named verdict: fast training over the memory stop, or speed nowhere
    d = dv(_report_v2(train=_train_v2(peak=77.0)))
    assert d["verdict"] == "NOT_ADOPTED" and d["variants"]["fla"]["peak_ok"] is False
    assert dv(_report_v2(pre=(1.9, 3.0, 4.0), train=_train_v2(fla=1500.0)))["verdict"] == "NOT_ADOPTED"
    assert dv(_report_v2(pre=(1.9, 3.0, 4.0)))["verdict"] == "ADOPT_FOR_TRAINING"  # prefill is not part of it
    assert seen == set(E.VERDICTS_V2)


def test_decide_v2_numerics_clauses():
    # the job-jn47u pattern: fla tracks chunked, both far from eager bf16 -> judged against fp64 truth instead
    t = _truth_v2(e_fla=0.30, e_ch=0.30, e_eg=0.02)
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "ADOPT_FOR_TRAINING"
    # fla > 1.10 x chunked (but <= 1.10 x eager) on 5 of 55 pairs: 50/55 = 0.909 >= 0.90
    over = {(n, "out", "fla"): 0.045 for n in list(E.v2_required_cases())[:5]}
    d = E.decide_v2(_report_v2(truth=_truth_v2(override=over)))
    assert d["verdict"] == "ADOPT_FOR_TRAINING"
    assert d["variants"]["fla"]["numerics"]["truth_detail"]["within_slack_of_chunked"] == 50
    # ... on 6 of 55: 49/55 = 0.891 < 0.90 -> reject
    over = {(n, t, "fla"): 0.045 for n in list(E.v2_required_cases())[:3] for t in ("out", "state")}
    d = E.decide_v2(_report_v2(truth=_truth_v2(override=over)))
    assert d["verdict"] == "REJECT_NUMERICS" and d["variants"]["fla"]["numerics"]["truth"] is False
    # boundary: exactly 1.10 x max + ABS_FLOOR is inside
    t = _truth_v2(override={("real_L0_fwd_T1024", "out", "fla"): 1.10 * 0.05 + 1e-6})
    assert E.decide_v2(_report_v2(truth=t))["variants"]["fla"]["numerics"]["why"] == []
    # non-finite fla -> reject; fla crash -> reject; crashed reference or non-finite truth -> inconclusive
    t = _truth_v2()
    t["cases"][3]["impls"]["fla"]["finite"] = False
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "REJECT_NUMERICS"
    t = _truth_v2(override={("real_L60_grad_T512", "grad_state", "fla"): math.inf})
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "REJECT_NUMERICS"
    t = _truth_v2()
    t["cases"][0]["impls"]["fla"] = {"error": "RuntimeError: boom"}
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "REJECT_NUMERICS"
    t = _truth_v2()
    t["cases"][0]["impls"]["chunked"] = {"error": "OutOfMemoryError"}
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "INCONCLUSIVE"
    t = _truth_v2()
    t["cases"][1]["truth_finite"] = False
    assert E.decide_v2(_report_v2(truth=t))["verdict"] == "INCONCLUSIVE"
    # the unchanged synthetic checks still gate
    assert E.decide_v2(_report_v2(fla_ok=False, safe_ok=False))["verdict"] == "REJECT_NUMERICS"
    r = _report_v2()
    r["model"]["continuation"]["fla_safe_gate"]["rel"] = 0.0265  # job-jn47u's fla_safe_gate continuation
    d = E.decide_v2(r)
    assert d["verdict"] == "ADOPT_FOR_TRAINING" and d["variants"]["fla_safe_gate"]["verdict"] == "REJECT_NUMERICS"
    r = _report_v2()
    r["model"]["logits"]["fla"]["top1"] = 0.97
    assert E.decide_v2(r)["verdict"] == "REJECT_NUMERICS"
    r = _report_v2()
    r["kernel"] = None
    assert E.decide_v2(r)["verdict"] == "INCONCLUSIVE"
    # v1's captured-input cases are not part of v2, even a failing one; v1 itself is unchanged
    r = _report_v2()
    r["captured"] = {"cases": [{"name": "real_L0_fwd_T1024", "impl": "fla", "dtype": "bfloat16",
                                "worst": {"out": 0.5, "cos": 0.9, "finite": True}}]}
    assert E.decide_v2(r)["verdict"] == "ADOPT_FOR_TRAINING"
    assert E.decide(r)["verdict"] == "REJECT_NUMERICS"


def test_decide_v2_train_speed_clauses():
    """RULE v2 + Amendment A1: ratio over matched completed steps only; OOM is never a zero."""
    # chunked OOMs everywhere: no matched steps -> INCONCLUSIVE (A1 withdrew OOM-as-0); reported as chunked_oom
    d = E.decide_v2(_report_v2(train=_train_v2(cur_oom=20)))
    assert d["verdict"] == "INCONCLUSIVE" and d["variants"]["fla"]["train_ratio"] is None
    assert d["chunked_oom"]["oom_steps"] == 20 and d["chunked_oom"]["binding"] is False
    assert d["chunked_oom"]["finding"] == "chunked cannot keep T_max=2048 at the cell config"
    assert d["chunked_oom"]["rows_max_T"][:2] == [1000, 1001]
    # chunked OOMs on 10 timed steps: 10 matched -> the ratio is over those 10 only, and still binding
    tok_c = [100.0] * 10 + [1000.0] * 10  # the OOM'd steps' slots are ignored anyway
    d = E.decide_v2(_report_v2(train=_train_v2(cur_oom=10, cur_tok=tok_c)))
    det = d["variants"]["fla"]["train_ratio_detail"]
    assert det["matched_steps"] == 10 and det["matched"] == list(range(13, 23))
    assert d["variants"]["fla"]["train_ratio"] == pytest.approx(3.0) and d["verdict"] == "ADOPT_FOR_TRAINING"
    assert d["chunked_oom"]["oom_steps"] == 10
    # 11 OOM -> 9 matched < 10 -> INCONCLUSIVE
    assert E.decide_v2(_report_v2(train=_train_v2(cur_oom=11)))["verdict"] == "INCONCLUSIVE"
    # deadline subsets: medians over the SAME rows for both impls (non-blocking #2)
    tok_f = [3000.0] * 12 + [30000.0] * 8  # fla much faster on the late (short) rows chunked never reached
    d = E.decide_v2(_report_v2(train=_train_v2(n_cur=12, fla_tok=tok_f)))
    assert d["variants"]["fla"]["train_ratio_detail"]["matched_steps"] == 12
    assert d["variants"]["fla"]["train_ratio"] == pytest.approx(3.0)
    assert E.decide_v2(_report_v2(train=_train_v2(n_cur=9)))["verdict"] == "INCONCLUSIVE"
    assert E.decide_v2(_report_v2(train=_train_v2(n_fla=9)))["verdict"] == "INCONCLUSIVE"
    assert E.decide_v2(_report_v2(train=_train_v2(n_fla=10)))["verdict"] == "ADOPT_FOR_TRAINING"
    # fla OOM on any step fails the memory clause (the remaining matched steps still give a ratio)
    d = E.decide_v2(_report_v2(train=_train_v2(fla_oom=1)))
    assert d["variants"]["fla"]["peak_ok"] is False and d["verdict"] == "NOT_ADOPTED"
    d = E.decide_v2(_report_v2(train=_train_v2(fla_oom=1, fla=1500.0)))
    assert d["verdict"] == "ADOPT_READOUT_ONLY"
    assert E.decide_v2(_report_v2(train=_train_v2(fla_oom=20)))["verdict"] == "INCONCLUSIVE"
    # a fallback or no fla call voids the fla measurement
    assert E.decide_v2(_report_v2(train=_train_v2(fallbacks={"fp32": 3})))["verdict"] == "INCONCLUSIVE"
    tr = _train_v2()
    tr["fla"]["fla_calls"] = {}
    assert E.decide_v2(_report_v2(train=tr))["verdict"] == "INCONCLUSIVE"
    # boundaries: ratio exactly 2.0 and peak exactly 76 GB adopt
    assert E.decide_v2(_report_v2(train=_train_v2(fla=2000.0, peak=76.0)))["verdict"] == "ADOPT_FOR_TRAINING"
    # median over the matched timed steps only (warmup excluded)
    tr = _train_v2()
    for s in tr["fla"]["steps"][:E.V2_TRAIN_WARMUP]:
        s["tok_s"] = 1.0
    assert E.decide_v2(_report_v2(train=tr))["variants"]["fla"]["train_ratio"] == pytest.approx(3.0)
    # fla_safe_gate is judged on its own measurement; the top-level verdict is plain fla's
    d = E.decide_v2(_report_v2(train=_train_v2(safe=1500.0)))
    assert d["verdict"] == d["variants"]["fla"]["verdict"] == "ADOPT_FOR_TRAINING"
    assert d["variants"]["fla_safe_gate"]["verdict"] == "ADOPT_READOUT_ONLY"


def test_decide_v2_amendment_a1_loss_guards():
    """A1/B2: a non-finite fla loss (warmup or timed) voids the measurement; step-0 loss must match chunked's to 2%."""
    good = E.decide_v2(_report_v2())
    assert good["variants"]["fla"]["train_ratio_detail"]["step0_loss_rel_diff"] == 0.0
    assert good["amendments"] == ["A1"]
    for i in (0, 2):  # warmup losses
        tr = _train_v2()
        tr["fla"]["steps"][i]["loss"] = float("nan")
        d = E.decide_v2(_report_v2(train=tr))
        assert d["verdict"] == "INCONCLUSIVE" and "non-finite" in d["variants"]["fla"]["train_ratio_detail"]["reason"]
    tr = _train_v2()
    tr["fla"]["steps"][10]["finite"] = False  # a timed step
    assert E.decide_v2(_report_v2(train=tr))["verdict"] == "INCONCLUSIVE"
    tr = _train_v2()
    tr["fla"]["steps"][10]["loss"] = float("inf")
    assert E.decide_v2(_report_v2(train=tr))["verdict"] == "INCONCLUSIVE"
    tr = _train_v2()
    tr["chunked"]["steps"][10]["loss"] = float("nan")  # chunked's own non-finite loss does not void fla's
    assert E.decide_v2(_report_v2(train=tr))["verdict"] == "ADOPT_FOR_TRAINING"
    # step-0 loss: within 2% passes, beyond is INCONCLUSIVE
    assert E.decide_v2(_report_v2(train=_train_v2(loss=1.0, fla_loss=1.015625)))["verdict"] == "ADOPT_FOR_TRAINING"
    d = E.decide_v2(_report_v2(train=_train_v2(loss=1.0, fla_loss=1.0201)))
    assert d["verdict"] == "INCONCLUSIVE" and d["variants"]["fla"]["train_ratio_detail"]["step0_loss_rel_diff"] > 0.02
    assert E.decide_v2(_report_v2(train=_train_v2(loss=1.0, fla_loss=0.97)))["verdict"] == "INCONCLUSIVE"
    # only binding when chunked's step 0 completed: chunked step 0 OOM -> no step-0 check
    tr = _train_v2(loss=1.0, fla_loss=1.5)
    tr["chunked"]["steps"][0] = {"i": 0, "timed": False, "oom": True, "max_T": 2000}
    d = E.decide_v2(_report_v2(train=tr))
    assert d["verdict"] == "ADOPT_FOR_TRAINING" and d["variants"]["fla"]["train_ratio_detail"]["step0_loss_rel_diff"] is None
    # chunked step 0 completed but fla step 0 did not -> INCONCLUSIVE
    tr = _train_v2()
    tr["fla"]["steps"][0] = {"i": 0, "timed": False, "oom": True, "max_T": 2000}
    assert E.decide_v2(_report_v2(train=tr))["verdict"] == "INCONCLUSIVE"
    # the loss guards only ever move toward INCONCLUSIVE: a REJECT_NUMERICS stays a reject
    tr = _train_v2(loss=1.0, fla_loss=2.0)
    assert E.decide_v2(_report_v2(train=tr, fla_ok=False, safe_ok=False))["verdict"] == "REJECT_NUMERICS"


def test_valid_decision_v2():
    d = E.decide_v2(_report_v2())
    assert E.valid_decision_v2(d) and not E.valid_decision(d)
    assert not E.valid_decision_v2({**d, "verdict": "ADOPT"}) and not E.valid_decision_v2({**d, "rule": None})
    assert not E.valid_decision_v2(E.decide(_report()))
    d2 = json.loads(json.dumps(E.decide_v2(_report_v2(train=_train_v2(cur_oom=20)))))
    assert E.valid_decision_v2(d2) and d2["variants"]["fla"]["train_ratio"] is None


# ----------------------------------------------------------------------------- RULE v2 phases: CPU dry runs
def test_truth_grad_uses_run_impl_cotangents(fake_fla, monkeypatch):
    """fp64 truth gradients = the fla branch (fake kernel, fp64 compute on fp64 inputs) with run_impl's cotangents."""
    monkeypatch.setattr(E, "DEVICE", "cpu")
    monkeypatch.setattr(E, "H", 2)
    monkeypatch.setattr(E, "N", 4)
    x = E.make_inputs(1, 17, F64, 5, device="cpu")  # fp64 activations, fp32 carried state (as in the model)
    tr = E.truth_grad(x)
    got = E.run_impl("fla", x, True)
    assert set(tr) == set(E.V2_GRAD_TENSORS)
    for t in E.V2_GRAD_TENSORS:
        assert E.rel_err(got[t], tr[t]) < 1e-6, t  # the fla wrapper hands the state back in fp32
    fwd = E.truth(x)
    assert E.rel_err(fwd["out"], tr["out"]) == 0 and E.rel_err(fwd["state"], tr["state"]) == 0


def test_truth_v2_phase_cpu_dry_run(fake_fla, monkeypatch, tmp_path):
    """The truth_v2 phase end to end on CPU: tiny RWKV-7, its 2 layers captured (prefix -> carried state -> tail),
    every implementation vs fp64 truth, then decide_v2's truth half on the file it wrote."""
    monkeypatch.setattr(E, "DEVICE", "cpu")
    m = tiny_lm()
    monkeypatch.setattr(E, "_load", lambda d: m)
    monkeypatch.setattr(E, "_text_ids", lambda d, T: torch.randint(1, 64, (1, T), generator=torch.Generator().manual_seed(T)))
    monkeypatch.setattr(E, "TRUTH_V2_LAYERS", (0, 1))
    monkeypatch.setattr(E, "V2_PROMPT_T", 48)
    monkeypatch.setattr(E, "V2_FWD_T", 24)
    monkeypatch.setattr(E, "V2_GRAD_T", 12)
    E.phase_truth_v2(tmp_path, "unused")
    d = json.loads((tmp_path / "truth_v2.json").read_text(encoding="utf-8"))
    assert d["complete"] is True and d["captured"] == [0, 1]
    assert [c["name"] for c in d["cases"]] == list(E.v2_required_cases())
    for c in d["cases"]:
        assert c["truth_finite"] and set(c["impls"]) == {"eager_bf16", "chunked", "fla", "fla_safe_gate"}
        assert c["shape"][1] == (12 if c["grad"] else 24) and "t_s" in c
        for impl, r in c["impls"].items():
            assert "error" not in r, (c["name"], impl, r)
            assert r["finite"] and set(r["e"]) == set(c["tensors"]) and max(r["e"].values()) < 1e-4, (c["name"], impl)
    assert len(d["cases"][1]["tensors"]) == 9  # grad case: out, state and 7 input gradients
    assert d["stats"]["1"]["state"] > 0  # the captured tail starts from a non-zero carried state
    ok, why, detail = E._v2_truth_check(d, "fla")
    assert ok is True and not why and detail["pairs"] == 22
    assert E.main(["decide_v2", "--out", str(tmp_path)]) == 0
    dv = json.loads((tmp_path / "decision_v2.json").read_text(encoding="utf-8"))
    assert dv["verdict"] == "INCONCLUSIVE" and dv["variants"]["fla"]["numerics"]["truth"] is True  # no other phases


def test_truth_v2_capture_matches_v1_construction():
    """Same prompt split as job-jn47u's model phase: 2048 tokens, state from the first 1024, capture the next 1024."""
    assert E.V2_PROMPT_T - E.V2_FWD_T == 1024 and E.V2_FWD_T == 1024
    src = (ROOT / "jobs/fla_rwkv7/fla_equivalence.py").read_text(encoding="utf-8")
    assert "ids = _text_ids(model_dir, 2048)" in src and "ids[:, :1024]" in src and "ids[:, 1024:]" in src  # v1
    assert "_text_ids(model_dir, V2_PROMPT_T)" in src and "capture_wkv_inputs(m, ids, TRUTH_V2_LAYERS, V2_FWD_T)" in src


def _toy_items():
    from heads.layout import build_row
    from tests.test_m2_review_fixes import ENC, req
    from train.data import Item

    r = req()
    out = []
    for i, q in enumerate(("dept", "down", "urg") * 3):
        row = build_row(r, q, ENC, layout="ordered")
        out.append(Item(f"toy/{i}", row, 0))
    return out


def _tiny_mix(path, n_rows, cell=None):
    from jobs.fla_rwkv7 import train_mix as TM

    shapes = [TM.row_shape(x) for x in _toy_items()]
    shapes = (shapes * (n_rows // len(shapes) + 1))[:n_rows]
    d = {"cell": {"head": "fork", "head_kwargs": {}, "layout": "ordered", "template": "jrt-plain-v1",
                  "loss_arm": "clef", "lora_rank": 2, "lr_head": 1e-3, "lr_lora": 1e-4, "warmup_steps": 0,
                  "batch_size": 1, "grad_accum": 2, "max_k": 16, "t_ladder": [512], "seed": 0,
                  "n_train_records": 3, **(cell or {})}, "rows": shapes}
    d["sha256"] = TM.mix_sha256(d)
    path.write_text(json.dumps(d), encoding="utf-8")
    return d


def test_train_mix_shapes_roundtrip():
    from jobs.fla_rwkv7 import train_mix as TM

    items = _toy_items()
    shapes = [TM.row_shape(x) for x in items]
    rows = TM.rows_from_shapes(shapes, vocab=260, seed=0)
    assert [TM.row_shape(x) for x in rows] == shapes  # same lengths, spans, K, qtype
    assert all(0 < t < 260 for x in rows for t in x.row.ids + [c for cs in x.row.continuations for c in cs])
    assert TM.rows_from_shapes(shapes, 260, 0)[0].row.ids == rows[0].row.ids  # deterministic ids
    assert {s["qtype"] for s in shapes} >= {"choice", "score"}


def test_train_v2_phase_cpu_dry_run(fake_fla, monkeypatch, tmp_path):
    """The training-speed phase on CPU: tiny RWKV-7 backbone, the cell's own Trainer step (fork head, cache fork,
    LoRA, Clef loss, grad accumulation) on mix-shaped rows, for chunked and fla; then the OOM and deadline paths."""
    from heads.backbone import tiny_hf_backbone

    monkeypatch.setattr(E, "DEVICE", "cpu")
    monkeypatch.setattr(E, "V2_TRAIN_WARMUP", 1)
    monkeypatch.setattr(E, "V2_TRAIN_STEPS", 2)
    monkeypatch.setattr(E, "_load_backbone", lambda d: tiny_hf_backbone("rwkv7", fork_mode="cache"))
    mix = tmp_path / "mix.json"
    _tiny_mix(mix, 6)
    monkeypatch.setattr(E, "MIX_PATH", mix)
    for impl in ("chunked", "fla"):
        W.CALLS.clear()
        E.phase_train_v2(tmp_path, "unused", impl, math.inf)
        d = json.loads((tmp_path / f"train_v2_{impl}.json").read_text(encoding="utf-8"))
        assert d["complete"] and [s["timed"] for s in d["steps"]] == [False, True, True]
        assert all(s["tok_s"] > 0 and s["finite"] and s["tokens"] > 0 and len(s["K"]) == 2 for s in d["steps"])
        assert not d["fallbacks"] and (d["fla_calls"].get("fla", 0) > 0) == (impl == "fla")
        assert E.train_summary_v2(d)["median_tok_s_completed"] > 0
    # the steps really trained: LoRA + head parameters moved under the cell's Trainer (not a forward-only timing)
    d = json.loads((tmp_path / "train_v2_fla.json").read_text(encoding="utf-8"))
    assert d["mix_sha256"] == json.loads(mix.read_text(encoding="utf-8"))["sha256"] and d["cell"]["lora_rank"] == 2
    # OOM on one step: recorded, counted 0 tok/s by decide_v2, the run continues
    from train.trainer import Trainer

    real, calls = Trainer.train_step, [0]

    def flaky(self):
        calls[0] += 1
        if calls[0] == 2:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(self)

    monkeypatch.setattr(Trainer, "train_step", flaky)
    E.phase_train_v2(tmp_path, "unused", "chunked", math.inf)
    d = json.loads((tmp_path / "train_v2_chunked.json").read_text(encoding="utf-8"))
    assert d["steps"][1].get("oom") and "tok_s" in d["steps"][2] and d["complete"]
    sp = E.train_summary_v2(d)
    assert sp["oom_steps"] == 1 and sp["completed_timed_steps"] == 1
    monkeypatch.setattr(Trainer, "train_step", real)
    # deadline already passed: nothing runs, every step recorded as skipped -> no speed (INCONCLUSIVE in decide_v2)
    E.phase_train_v2(tmp_path, "unused", "fla", 0.0)
    d = json.loads((tmp_path / "train_v2_fla.json").read_text(encoding="utf-8"))
    assert all(s.get("skipped") == "deadline" for s in d["steps"]) and E.train_summary_v2(d)["completed_timed_steps"] == 0


def test_train_v2_mix_must_cover_the_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "DEVICE", "cpu")
    mix = tmp_path / "mix.json"
    _tiny_mix(mix, 5)  # (3 + 20) x 2 needed
    monkeypatch.setattr(E, "MIX_PATH", mix)
    with pytest.raises(SystemExit, match="rows"):
        E.phase_train_v2(tmp_path, "unused", "chunked", math.inf)
    d = json.loads(mix.read_text(encoding="utf-8"))
    d["rows"][0]["T"] += 1  # tampered table
    mix.write_text(json.dumps(d), encoding="utf-8")
    from jobs.fla_rwkv7 import train_mix as TM

    with pytest.raises(ValueError, match="sha256"):
        TM.load(mix)


def test_committed_train_mix_is_the_m2_cell():
    """train_mix_v2.json: the m2 first cell's TrainConfig fields and the shapes of its first 23 optimiser steps."""
    from jobs.fla_rwkv7 import train_mix as TM

    d = TM.load()
    c = d["cell"]
    assert (c["head"], c["layout"], c["loss_arm"], c["lora_rank"], c["batch_size"], c["grad_accum"], c["seed"]) == \
        ("fork", "ordered", "clef", 64, 1, 8, 0)
    assert c["template"] == "jrt-rwkv-g1-v1" and max(c["t_ladder"]) == 2048 and c["max_k"] == 16
    n = (E.V2_TRAIN_WARMUP + E.V2_TRAIN_STEPS) * c["batch_size"] * c["grad_accum"]
    assert len(d["rows"]) == n == 184
    for s in d["rows"]:
        assert 0 < s["T"] <= 2048 and 2 <= len(s["conts"]) <= 16 and max(s["conts"]) <= 16 and min(s["conts"]) >= 1
        assert s["qtype"] in ("noul", "choice", "score")
        assert len(s["option_spans"]) == len(s["conts"]) and s["state_span"][1] <= s["T"]
    assert set(d["rows"][0]) == {"T", "conts", "qtype", "state_span", "field_span", "option_spans"}  # no ids/labels


def test_truth_v2_runner_and_params():
    from jobs.common import costs

    p = json.loads((ROOT / "jobs/fla_rwkv7/job_params_truth_v2.json").read_text(encoding="utf-8"))
    assert costs.worst_case_usd(p) <= 6.0 and costs.worst_case_usd(p) <= p["hard_cap_usd"]
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    for ph, exp in p["expected_minutes"].items():  # every cap >= 1.3x its projection
        assert p["phase_caps_minutes"][ph] >= 1.3 * exp, ph
    assert p["phase_caps_minutes"]["warmup"] == 13 and p["chip"] == "h100" and p["max_restarts"] == 0
    assert p["assumed_restart_floor"] >= 1
    spec = (ROOT / "jobs/fla_rwkv7/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"${costs.worst_case_usd(p):.2f}" in spec
    s = (ROOT / "jobs/fla_rwkv7/fla_job.sh").read_text(encoding="utf-8")
    v2 = s[s.rindex('if [ "$MODE" = truth_v2 ]'):]  # the phase block (the first one sets marker / validator)
    order = ["fla_equivalence model", "fla_equivalence throughput", "fla_equivalence truth_v2",
             "fla_equivalence train_v2", "fla_equivalence decide_v2"]
    assert all(o in s for o in order)
    assert s.index("fla_equivalence warmup") < s.index("fla_equivalence model") < s.rindex('if [ "$MODE" = truth_v2 ]')
    assert v2.index("fla_equivalence truth_v2") < v2.index("fla_equivalence train_v2") < v2.index("fla_equivalence decide_v2")
    assert "fla_equivalence fp32" not in v2  # the optional fp32 tail never runs in truth_v2 mode
    assert "job_params_truth_v2.json" in s and "valid_decision_v2" in s and "decision_v2.json" in s
    assert "FLA_RWKV7_TRUTH_V2_DONE" in s and "export TRITON_CACHE_DIR=$WORK/triton-cache" in s
    assert "for IMPL in chunked fla fla_safe_gate" in v2 and "--impl $IMPL" in v2 and "--deadline-s" in v2
    bc = (ROOT / "jobs/fla_rwkv7/build_context.sh").read_text(encoding="utf-8")
    for f in ("jobs/fla_rwkv7/job_params_truth_v2.json", "jobs/fla_rwkv7/train_mix_v2.json",
              "jobs/fla_rwkv7/train_mix.py"):
        assert f in bc, f

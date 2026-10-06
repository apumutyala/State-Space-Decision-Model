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

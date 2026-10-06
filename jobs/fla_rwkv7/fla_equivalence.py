"""fla ``chunk_rwkv7`` vs the vendored eager WKV: numeric equivalence + throughput gate (jobs/fla_rwkv7). GPU phases.

    python -m jobs.fla_rwkv7.fla_equivalence kernel     --out DIR             # synthetic inputs, real 13.3B head shapes
    python -m jobs.fla_rwkv7.fla_equivalence fork       --out DIR             # decision path: state split / fork / pack / pad
    python -m jobs.fla_rwkv7.fla_equivalence model      --out DIR --model-dir SNAP   # real weights: logits, continuation,
                                                                               # captured real WKV inputs (layers 0/30/60)
    python -m jobs.fla_rwkv7.fla_equivalence throughput --out DIR --model-dir SNAP   # prefill 2k/8k/16k, train fwd+bwd 2k
    python -m jobs.fla_rwkv7.fla_equivalence decide     --out DIR             # pre-registered rule -> DIR/decision.json

Every phase writes DIR/<phase>.json after each case, so a phase timeout still leaves what it measured. The rule and the
tolerances below are PRE-REGISTERED (jobs/fla_rwkv7/JOB_SPEC.md quotes them; tests/test_fla_rwkv7.py pins them): do not
edit them after seeing GPU numbers. ``decide`` and the metric helpers are pure and CPU-tested.

Metric: ``rel = ||x - ref||_inf / ||ref||_inf`` per tensor (max over outputs / final state / every gradient), plus cosine
similarity of the flattened tensors. Reference = the vendored ``rwkv7_eager`` (token-serial, fp32 state) at the same
activation dtype; an fp64 token-serial run is recorded as a diagnostic ``truth`` only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

# ----------------------------------------------------------------------------- PRE-REGISTERED (do not tune on GPU data)
#: per activation dtype: max rel error of outputs / final state / gradients vs eager, and min cosine
TOL = {
    "bfloat16": {"out": 2e-2, "state": 2e-2, "grad": 5e-2, "cos": 0.999},
    "float32": {"out": 2e-3, "state": 2e-3, "grad": 5e-3, "cos": 0.99999},
}
#: denominator floor of `rel` (and cosine treated as 1 when both tensors are below it): a gradient that the decay has
#: driven to ~1e-45 (e.g. d/d(initial state) after 1024 tokens) is zero, not a 100% error
ABS_FLOOR = 1e-6
#: real-weights model level (bf16, 61 layers compound): fla vs eager logits must be no worse than the production
#: "chunked" path is vs eager (x MODEL_SLACK, floor MODEL_REL_FLOOR), min per-position cosine and top-1 agreement
MODEL_REL_FLOOR = 2e-2
MODEL_SLACK = 1.5
MODEL_COS_MIN = 0.999
MODEL_TOP1_MIN = 0.98
#: continuation / fork at model level: state handed across calls must reproduce the one-shot logits
CONT_REL_MAX = 2e-2
#: two forks of one state (deep copies) must agree to this (exact equality is recorded too)
FORK_REL_MAX = 1e-5
#: speed: fla / chunked tok/s, on the real model, median of REPS after a warm-up rep
SPEEDUP_MIN = 2.0
LENGTHS = (2048, 8192, 16384)
TRAIN_T = 2048
REPS = 3
#: fla_safe_gate is only preferred over plain fla when it passes numerics AND is this much faster (else plain fla)
SAFE_GATE_MARGIN = 1.10
#: the m2 child evals need >= this end-to-end prefill rate (jobs/PERF_AUDIT.md); reported, not part of the rule
CHILD_EVAL_TOK_S = 12000
VARIANTS = ("fla", "fla_safe_gate")
VERDICTS = ("ADOPT", "ADOPT_READOUT_ONLY", "ADOPT_TRAINING_ONLY", "REJECT_SPEED", "REJECT_NUMERICS", "INCONCLUSIVE")

# 13.3B (jobs/rwkv7/JOB_SPEC.md): 61 layers, hidden 4096, 64 heads x 64
H, N = 64, 64
CAPTURE_LAYERS = (0, 30, 60)
DEVICE = "cuda"  # the weight-free phases also run on CPU (tests/test_fla_rwkv7.py smoke, with a stand-in kernel)


# ----------------------------------------------------------------------------- pure helpers (CPU-tested)
def rel_err(x, ref) -> float:
    d = (x.double() - ref.double()).abs().max().item()
    s = max(ref.double().abs().max().item(), ABS_FLOOR)
    return d / s if math.isfinite(d) else math.inf


def cosine(x, ref) -> float:
    a, b = x.double().flatten(), ref.double().flatten()
    if a.abs().max().item() < ABS_FLOOR and b.abs().max().item() < ABS_FLOOR:
        return 1.0
    na, nb = a.norm().item(), b.norm().item()
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float((a @ b).item() / (na * nb))


def compare(got: dict, ref: dict) -> dict:
    """{name: tensor} vs {name: tensor} -> per-name rel/cos + the worst per kind (out/state/grad)."""
    per = {}
    for k, r in ref.items():
        g = got[k]
        per[k] = {"rel": rel_err(g, r), "cos": cosine(g, r), "finite": bool(g.isfinite().all().item())}
    worst = {}
    for kind in ("out", "state", "grad"):
        ks = [k for k in per if (k.startswith("grad_") if kind == "grad" else k == kind or k.startswith(kind + "_"))]
        if ks:
            worst[kind] = max(per[k]["rel"] for k in ks)
    worst["cos"] = min(v["cos"] for v in per.values())
    worst["finite"] = all(v["finite"] for v in per.values())
    return {"per": per, "worst": worst}


def case_passes(worst: dict, dtype: str) -> bool:
    t = TOL[dtype]
    if not worst.get("finite", False):
        return False
    for kind in ("out", "state", "grad"):
        v = worst.get(kind)
        if v is not None and not (v <= t[kind]):  # NaN fails
            return False
    return worst.get("cos", 0.0) >= t["cos"]


def model_passes(m: dict) -> bool:
    """m: {'fla': {'rel','cos_min','top1'}, 'chunked': {'rel'}} vs eager logits."""
    f, c = m["fla"], m["chunked"]
    limit = max(MODEL_REL_FLOOR, MODEL_SLACK * c["rel"])
    return bool(f["rel"] <= limit and f["cos_min"] >= MODEL_COS_MIN and f["top1"] >= MODEL_TOP1_MIN)


def _variant_numerics(rep: dict, v: str) -> tuple[bool | None, list[str]]:
    """True/False, or None when a required piece is missing. bf16 decides; fp32 only gates ALLOW_FP32."""
    why, missing = [], []
    for phase in ("kernel", "fork", "captured"):
        cases = [c for c in (rep.get(phase) or {}).get("cases", []) if c.get("impl") == v and c.get("dtype") == "bfloat16"]
        if not cases:
            missing.append(f"{phase}:bf16")
        for c in cases:
            if c.get("error") or not case_passes(c.get("worst", {}), "bfloat16"):
                why.append(f"{phase}:{c.get('name')}")
    model = rep.get("model") or {}
    mdl, ref = model.get("logits", {}).get(v), model.get("logits", {}).get("chunked")
    if not mdl or not ref:
        missing.append("model:logits")
    elif not model_passes({"fla": mdl, "chunked": ref}):
        why.append("model:logits")
    cont = model.get("continuation", {}).get(v)
    if not cont:
        missing.append("model:continuation")
    elif not (cont.get("rel", math.inf) <= CONT_REL_MAX and cont.get("fork_rel", math.inf) <= FORK_REL_MAX):
        why.append("model:continuation")
    if why:
        return False, why
    if missing:
        return None, missing
    return True, []


def _fp32_ok(rep: dict, v: str) -> bool:
    cases = [c for phase in ("kernel", "fork") for c in (rep.get(phase) or {}).get("cases", [])
             if c.get("impl") == v and c.get("dtype") == "float32"]
    return bool(cases) and all(not c.get("error") and case_passes(c.get("worst", {}), "float32") for c in cases)


def _ratios(rep: dict, v: str) -> dict:
    tp = rep.get("throughput") or {}
    pre = tp.get("prefill", {})
    out = {"prefill": {}, "train": None}
    for L in LENGTHS:
        a, b = (pre.get(str(L)) or {}).get(v, {}), (pre.get(str(L)) or {}).get("chunked", {})
        out["prefill"][str(L)] = (a["tok_s"] / b["tok_s"]) if a.get("tok_s") and b.get("tok_s") else None
    tr = tp.get("train", {})
    a, b = tr.get(v, {}), tr.get("chunked", {})
    out["train"] = (a["tok_s"] / b["tok_s"]) if a.get("tok_s") and b.get("tok_s") else None
    return out


def decide(rep: dict) -> dict:
    """The pre-registered rule. rep = {'kernel','fork','captured','model','throughput'} phase JSONs.

    Per fla variant: numerics pass iff every bf16 case (kernel, fork, captured real inputs) is within TOL, the real-model
    logits pass ``model_passes`` and the model continuation/fork passes. Readout speed passes iff fla/chunked prefill
    >= SPEEDUP_MIN at EVERY length in LENGTHS; training speed iff the fwd+bwd ratio at TRAIN_T >= SPEEDUP_MIN.
    ADOPT = numerics + both; ADOPT_READOUT_ONLY / ADOPT_TRAINING_ONLY = numerics + one; REJECT_SPEED = numerics, neither;
    REJECT_NUMERICS = a measured numerics failure; INCONCLUSIVE = a required measurement is missing.
    Chosen variant: plain "fla" unless "fla_safe_gate" also qualifies and is >= SAFE_GATE_MARGIN faster on every
    measured ratio. fp32 never blocks (the model runs bf16); it only decides ``allow_fp32``."""
    per = {}
    for v in VARIANTS:
        num, why = _variant_numerics(rep, v)
        r = _ratios(rep, v)
        pre = list(r["prefill"].values())
        readout = None if any(x is None for x in pre) else all(x >= SPEEDUP_MIN for x in pre)
        train = None if r["train"] is None else r["train"] >= SPEEDUP_MIN
        if num is False:
            verdict = "REJECT_NUMERICS"
        elif num is None or readout is None or train is None:
            verdict = "INCONCLUSIVE"
        elif readout and train:
            verdict = "ADOPT"
        elif readout:
            verdict = "ADOPT_READOUT_ONLY"
        elif train:
            verdict = "ADOPT_TRAINING_ONLY"
        else:
            verdict = "REJECT_SPEED"
        per[v] = {"verdict": verdict, "numerics": num, "numerics_detail": why, "ratios": r,
                  "readout_speed_ok": readout, "train_speed_ok": train, "allow_fp32": _fp32_ok(rep, v)}
    chosen = "fla"
    a, b = per["fla"], per["fla_safe_gate"]
    if b["verdict"].startswith("ADOPT") and (not a["verdict"].startswith("ADOPT") or _faster(b["ratios"], a["ratios"])):
        chosen = "fla_safe_gate"
    best = per[chosen]
    pre16 = (((rep.get("throughput") or {}).get("prefill") or {}).get("2048") or {}).get(chosen, {}).get("tok_s")
    return {"verdict": best["verdict"], "chosen": chosen, "variants": per,
            "rule": {"tol": TOL, "abs_floor": ABS_FLOOR, "model_rel_floor": MODEL_REL_FLOOR, "model_slack": MODEL_SLACK,
                     "model_cos_min": MODEL_COS_MIN, "model_top1_min": MODEL_TOP1_MIN, "cont_rel_max": CONT_REL_MAX,
                     "fork_rel_max": FORK_REL_MAX,
                     "speedup_min": SPEEDUP_MIN, "lengths": list(LENGTHS), "train_t": TRAIN_T,
                     "safe_gate_margin": SAFE_GATE_MARGIN},
            "child_eval_tok_s_target": CHILD_EVAL_TOK_S,
            "chosen_prefill_2048_tok_s": pre16,
            "meets_child_eval_rate": bool(pre16 and pre16 >= CHILD_EVAL_TOK_S)}


def _faster(b: dict, a: dict) -> bool:
    pairs = [(b["prefill"][k], a["prefill"][k]) for k in b["prefill"]] + [(b["train"], a["train"])]
    return all(x is not None and y is not None and x >= SAFE_GATE_MARGIN * y for x, y in pairs)


def _fla_really_ran(phase: dict) -> bool:
    """No fallback (it would run "chunked" under the fla name) and every variant actually reached the kernel."""
    calls = phase.get("fla_calls") or {}
    return not phase.get("fallbacks") and all(calls.get(v, 0) > 0 for v in VARIANTS)


def valid_decision(d) -> bool:
    """finalize.py validator: a verdict from the rule and a chosen variant."""
    return isinstance(d, dict) and d.get("verdict") in VERDICTS and d.get("chosen") in VARIANTS


# ----------------------------------------------------------------------------- GPU side
def _torch():
    import torch

    return torch


def make_inputs(B: int, T: int, dtype, seed: int, device=None, state_scale: float = 0.5):
    """Model-distributed WKV inputs at the real head shape: w_log = -e^-0.5 * sigmoid(.) (the model's decay law),
    kk unit-norm per head, a in (0, 1), fp32 carried state."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    r, k, v = rn(B, T, H, N), rn(B, T, H, N), rn(B, T, H, N)
    w_log = -math.exp(-0.5) * torch.sigmoid(rn(B, T, H, N) * 2 - 0.5)
    kk = torch.nn.functional.normalize(rn(B, T, H, N), dim=-1)
    a = torch.sigmoid(rn(B, T, H, N))
    state = rn(B, H, N, N) * state_scale
    device = device or DEVICE
    t = {n: x.to(device=device, dtype=dtype) for n, x in dict(r=r, w_log=w_log, k=k, v=v, kk=kk, a=a).items()}
    t["state"] = state.to(device=device, dtype=torch.float32)
    return t


ARGS = ("r", "w_log", "k", "v", "kk", "a", "state")


def run_impl(impl: str, x: dict, grad: bool, cu_seq_lens=None, seed: int = 1, allow_fp32: bool = True):
    """Run one WKV implementation on inputs x; returns {'out','state'[, 'grad_<arg>']} (detached)."""
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    leaves = {n: x[n].detach().clone().requires_grad_(grad) for n in ARGS}
    if impl == "eager":
        o, s = M.rwkv7_eager(*(leaves[n] for n in ARGS), cu_seq_lens=cu_seq_lens)
    elif impl == "chunked":
        o, s = M.rwkv7_parallel(*(leaves[n] for n in ARGS), cu_seq_lens=cu_seq_lens)
    elif impl in VARIANTS:
        chunk, err = W._import_chunk()
        if err:
            raise RuntimeError(err)
        reason = W.fallback_reason(leaves["r"], cu_seq_lens, allow_fp32)
        if reason:
            raise RuntimeError(f"would fall back: {reason}")
        o, s = W.fla_wkv_call(chunk, *(leaves[n] for n in ARGS), cu_seq_lens, safe_gate=impl == "fla_safe_gate")
    else:
        raise ValueError(impl)
    res = {"out": o.detach(), "state": s.detach()}
    if grad:
        g = torch.Generator(device="cpu").manual_seed(seed)
        do = torch.randn(o.shape, generator=g).to(o.device)
        ds = torch.randn(s.shape, generator=g).to(s.device)
        ((o.float() * do).sum() + (s.float() * ds).sum()).backward()
        for n in ARGS:
            res[f"grad_{n}"] = _grad(leaves[n])
    return res


def truth(x: dict, cu_seq_lens=None) -> dict:
    """fp64 token-serial run of the same (rounded) inputs: a diagnostic, not the gate's reference."""
    torch = _torch()
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    d = {n: x[n].double() for n in ARGS}
    if cu_seq_lens is not None:
        return {}
    o, s = M.rwkv7_recurrent(*(d[n] for n in ARGS), compute_dtype=torch.float64)
    return {"out": o, "state": s}


class Report:
    def __init__(self, path: Path, **meta):
        self.path, self.d = path, {"cases": [], **meta}
        self.flush()

    def add(self, case: dict):
        self.d["cases"].append(case)
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=1, default=str), encoding="utf-8")


def _cmp_case(rep: Report, name: str, dtype_name: str, x: dict, grad: bool, cu=None, impls=("chunked",) + VARIANTS,
              with_truth: bool = False):
    torch = _torch()
    ref = run_impl("eager", x, grad, cu)
    tr = truth(x, cu) if with_truth else {}
    for impl in impls:
        case = {"name": name, "impl": impl, "dtype": dtype_name, "grad": grad,
                "shape": list(x["r"].shape), "cu_seq_lens": None if cu is None else cu.tolist()}
        try:
            got = run_impl(impl, x, grad, cu)
            c = compare(got, ref)
            case.update(worst=c["worst"], per=c["per"], passes=case_passes(c["worst"], dtype_name))
            if tr:
                case["vs_fp64"] = {k: rel_err(got[k], tr[k]) for k in tr}
        except Exception as e:  # noqa: BLE001  recorded; decide() treats it as a failure of that case
            case["error"] = f"{type(e).__name__}: {e}"[:500]
        rep.add(case)
    if tr:
        rep.add({"name": name, "impl": "eager", "dtype": dtype_name, "diag": True,
                 "vs_fp64": {k: rel_err(ref[k], tr[k]) for k in tr}})
    torch.cuda.empty_cache()


def phase_kernel(out: Path):
    torch = _torch()
    rep = Report(out / "kernel.json", env=_env())
    for dt in ("bfloat16", "float32"):
        dtype = getattr(torch, dt)
        for B, T, seed in ((2, 2048, 11), (2, 1000, 12)):  # 1000: not a multiple of any chunk size (padding path)
            _cmp_case(rep, f"fwd_B{B}_T{T}", dt, make_inputs(B, T, dtype, seed), grad=False, with_truth=True)
        _cmp_case(rep, "grad_B2_T512", dt, make_inputs(2, 512, dtype, 13), grad=True)
        # short: d/d(initial state) is still far from zero here (fla's dh0 path, used by every fork)
        _cmp_case(rep, "grad_B2_T24", dt, make_inputs(2, 24, dtype, 14), grad=True)
    _p640(rep)


def _p640(rep: Report):
    """fla #640 is the gated chunk_bwd_dqkwg in fla/ops/common/chunk_o.py. Record that the RWKV7 (DPLR) path never
    references it, from fla's own source as installed, and that a bf16 backward on this GPU is finite."""
    import inspect

    import fla.ops.generalized_delta_rule.dplr.chunk as dplr
    import fla.ops.rwkv7.chunk as rc

    src = inspect.getsource(dplr) + inspect.getsource(rc)
    rep.d["p640"] = {"dplr_references_chunk_bwd_dqkwg": "chunk_bwd_dqkwg" in src,
                     "dplr_imports_common_chunk_o": "fla.ops.common.chunk_o" in src}
    rep.flush()


def phase_fork(out: Path):
    """The decision path: prefix state -> continuation, K-way fork from one state (grads into the prefix), packed
    rows (cu_seq_lens, incl. a 1-token segment), right padding that must leave the state untouched."""
    torch = _torch()
    rep = Report(out / "fork.json", env=_env())
    for dt in ("bfloat16", "float32"):
        dtype = getattr(torch, dt)
        # 1) split: impl(x[:P]) -> s ; impl(x[P:], s) == eager(x) on the tail and the final state
        x = make_inputs(1, 1537, dtype, 21)
        P = 1500
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "split_P1500_L37", "impl": impl, "dtype": dt}
            try:
                full = run_impl("eager", x, False)
                a = run_impl(impl, _slice(x, 0, P), False)
                b = run_impl(impl, {**_slice(x, P, None), "state": a["state"]}, False)
                c = compare({"out": b["out"], "state": b["state"]}, {"out": full["out"][:, P:], "state": full["state"]})
                case.update(worst=c["worst"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)
        # 2) fork: one prefix state, K=4 tails, gradients flow through the shared state into the prefix inputs
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "fork_P1024_K4_L24_grad", "impl": impl, "dtype": dt}
            try:
                c = compare(_fork_run(impl, dtype), _fork_run("eager", dtype))
                case.update(worst=c["worst"], per=c["per"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)
        # 3) packed row: fresh state per segment, the returned state is the last segment's
        cu = torch.tensor([0, 100, 101, 1100, 2048], device=DEVICE)
        _cmp_case(rep, "packed_4seg", dt, make_inputs(1, 2048, dtype, 22), grad=True, cu=cu)
        # 4) right padding (the model's keep mask: decay 1, k = v = kk = 0) leaves the state where the last token left it
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "right_pad_identity", "impl": impl, "dtype": dt}
            try:
                x = make_inputs(1, 600, dtype, 23)
                real = run_impl("eager", _slice(x, 0, 555), False)
                padded = dict(x)
                for n in ("w_log", "k", "v", "kk"):
                    padded[n] = x[n].clone()
                    padded[n][:, 555:] = 0
                got = run_impl(impl, padded, False)
                c = compare({"state": got["state"], "out": got["out"][:, :555]}, {"state": real["state"], "out": real["out"]})
                case.update(worst=c["worst"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)


def _slice(x: dict, a, b) -> dict:
    return {n: (x[n][:, a:b] if n != "state" else x[n]) for n in ARGS}


def _fork_run(impl: str, dtype, K: int = 4, P: int = 1024, L: int = 24) -> dict:
    """Prefix (B=1) -> state.repeat_interleave(K) (new tensors, like batch_repeat_interleave) -> K tails (B=K)."""
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    def call(xs):
        if impl == "eager":
            return M.rwkv7_eager(*xs)
        if impl == "chunked":
            return M.rwkv7_parallel(*xs)
        chunk, err = W._import_chunk()
        if err:
            raise RuntimeError(err)
        return W.fla_wkv_call(chunk, *xs, safe_gate=impl == "fla_safe_gate")

    pre = {n: t.requires_grad_(True) for n, t in make_inputs(1, P, dtype, 31).items()}
    tail = {n: t.requires_grad_(True) for n, t in make_inputs(K, L, dtype, 32).items() if n != "state"}
    _, s = call([pre[n] for n in ARGS])
    s_k = s.repeat_interleave(K, dim=0)
    o, sK = call([tail[n] for n in ARGS[:-1]] + [s_k])
    g = torch.Generator(device="cpu").manual_seed(33)
    do = torch.randn(o.shape, generator=g).to(o.device)
    ds = torch.randn(sK.shape, generator=g).to(sK.device)
    ((o.float() * do).sum() + (sK.float() * ds).sum()).backward()
    res = {"out": o.detach(), "state": sK.detach()}
    for n in ARGS:
        res[f"grad_prefix_{n}"] = _grad(pre[n])
    for n in ARGS[:-1]:
        res[f"grad_tail_{n}"] = _grad(tail[n])
    return res


def _grad(t):
    """A leaf autograd never reached (e.g. the prefix's r: only its state is forked) has a zero gradient. Eager leaves
    it None while a custom autograd.Function materialises zeros, so both are compared as zeros."""
    return t.grad.detach() if t.grad is not None else _torch().zeros_like(t)


# ----------------------------------------------------------------------------- real weights
def _load(model_dir: str):
    torch = _torch()
    import third_party.rwkv7_hf as V
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    V.verify_snapshot_code(model_dir)
    m = Rwkv7ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _text_ids(model_dir: str, T: int):
    """Real text (the vendored Apache LICENSE + NOTICE, repeated) tokenised by the snapshot's tokenizer."""
    torch = _torch()
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    here = Path(__file__).resolve().parents[2] / "third_party" / "rwkv7_hf"
    text = (here / "LICENSE").read_text(encoding="utf-8") + (here / "NOTICE.md").read_text(encoding="utf-8")
    tok = Rwkv7Tokenizer.from_pretrained(model_dir)
    ids = [i for i in tok(text, add_special_tokens=False).input_ids if i != 0]
    ids = (ids * (T // len(ids) + 1))[:T]
    return torch.tensor([ids], device="cuda")


def _use(m, name: str):
    from heads.rwkv7_wkv import select_wkv

    return select_wkv(m, name, allow_unverified=True)


def phase_model(out: Path, model_dir: str):
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    rep = Report(out / "model.json", env=_env())
    m = _load(model_dir)
    rep.d["logits"], rep.d["continuation"] = {}, {}
    W.FALLBACKS.clear()
    W.CALLS.clear()
    # 1) logits vs eager (tokenwise, token-serial) at T=256
    ids = _text_ids(model_dir, 256)
    logits = {}
    for impl in ("eager", "chunked") + VARIANTS:
        _use(m, impl)
        with torch.no_grad():
            logits[impl] = m(input_ids=ids, use_cache=False).logits.float()
    for impl in ("chunked",) + VARIANTS:
        x, r = logits[impl][0], logits["eager"][0]
        cos_pos = torch.nn.functional.cosine_similarity(x.double(), r.double(), dim=-1)
        rep.d["logits"][impl] = {"rel": rel_err(x, r), "cos_min": cos_pos.min().item(),
                                 "top1": (x.argmax(-1) == r.argmax(-1)).double().mean().item(), "T": 256}
    rep.flush()
    del logits
    # 2) continuation and fork at T=2048: state from a 1900-token prefix reproduces the one-shot logits; two forks of
    #    one state (deep copies) are identical
    import copy

    ids = _text_ids(model_dir, 2048)
    P, L = 1900, 148
    for impl in VARIANTS:
        _use(m, impl)
        with torch.no_grad():
            one = m(input_ids=ids, use_cache=False, logits_to_keep=L).logits.float()
            st = m(input_ids=ids[:, :P], use_cache=True, logits_to_keep=1).state
            st2 = copy.deepcopy(st)
            a = m(input_ids=ids[:, P:], state=st, use_cache=True).logits.float()
            b = m(input_ids=ids[:, P:], state=st2, use_cache=True).logits.float()
        rep.d["continuation"][impl] = {"rel": rel_err(a, one), "fork_identical": bool(torch.equal(a, b)),
                                       "fork_rel": rel_err(b, a), "P": P, "L": L}
        rep.flush()
    rep.d["fallbacks"], rep.d["fla_calls"] = dict(W.FALLBACKS), dict(W.CALLS)  # must be {} / > 0
    rep.flush()
    # 3) capture the real WKV inputs at layers 0/30/60 on T=2048 text (chunked path), then compare kernels on them
    caught, calls = {}, [0]

    def capture(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kw):
        layer = calls[0] % m.config.num_hidden_layers
        calls[0] += 1
        if layer in CAPTURE_LAYERS:
            caught[layer] = {n: t.detach().clone() for n, t in zip(ARGS, (r, w_log, k, v, kk, a, state))}
        return M.rwkv7_parallel(r, w_log, k, v, kk, a, state, cu_seq_lens=cu_seq_lens)

    M.RWKV7_WKV_FUNCTIONS["ssdm_capture"] = capture
    for cfg in W._configs(m):
        cfg.wkv_implementation = W.WkvName("ssdm_capture")
    try:
        with torch.no_grad():
            st = m(input_ids=ids[:, :1024], use_cache=True, logits_to_keep=1).state
            calls[0] = 0
            m(input_ids=ids[:, 1024:], state=st, use_cache=True, logits_to_keep=1)  # non-zero carried state
    finally:
        M.RWKV7_WKV_FUNCTIONS.pop("ssdm_capture", None)
        _use(m, "chunked")
    del m
    torch.cuda.empty_cache()
    cap = Report(out / "captured.json", env=_env(), layers=list(CAPTURE_LAYERS))
    for layer, x in sorted(caught.items()):
        cap.d.setdefault("stats", {})[str(layer)] = {n: float(x[n].float().abs().max()) for n in ARGS}
        _cmp_case(cap, f"real_L{layer}_fwd_T1024", "bfloat16", x, grad=False)
        _cmp_case(cap, f"real_L{layer}_grad_T512", "bfloat16", _slice(x, 0, 512), grad=True)


def _timed(fn, reps: int = REPS) -> list[float]:
    torch = _torch()
    ts = []
    for _ in range(reps + 1):  # first = warm-up (fla autotune / compile)
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return ts[1:]


def phase_throughput(out: Path, model_dir: str):
    torch = _torch()
    from heads import rwkv7_wkv as W

    rep = Report(out / "throughput.json", env=_env(), prefill={}, train={}, kernel={})
    W.FALLBACKS.clear()
    W.CALLS.clear()
    m = _load(model_dir)
    impls = ("chunked",) + VARIANTS
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in LENGTHS:
        ids = torch.randint(1, m.config.vocab_size, (1, L), generator=g).cuda()
        row = rep.d["prefill"].setdefault(str(L), {})
        for impl in impls:
            _use(m, impl)
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    ts = _timed(lambda: m(input_ids=ids, logits_to_keep=1, use_cache=False))
                med = statistics.median(ts)
                row[impl] = {"tok_s": L / med, "s_median": med, "s_all": ts, "spread": (max(ts) - min(ts)) / med,
                             "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            except Exception as e:  # noqa: BLE001  (OOM included)
                row[impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
            torch.cuda.empty_cache()
            rep.flush()
    # training-shaped step: frozen weights, grad w.r.t. the embeddings (the LoRA backward runs the same activations)
    ids = torch.randint(1, m.config.vocab_size, (1, TRAIN_T), generator=g).cuda()
    for impl in impls:
        _use(m, impl)
        torch.cuda.reset_peak_memory_stats()

        def step():
            e = m.rwkv7.emb(ids).detach().requires_grad_(True)
            h = m.rwkv7(inputs_embeds=e, use_cache=False).last_hidden_state
            h.float().pow(2).mean().backward()

        try:
            ts = _timed(step)
            med = statistics.median(ts)
            rep.d["train"][impl] = {"tok_s": TRAIN_T / med, "s_median": med, "s_all": ts,
                                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        except Exception as e:  # noqa: BLE001
            rep.d["train"][impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
        torch.cuda.empty_cache()
        rep.flush()
    _use(m, "chunked")
    del m
    torch.cuda.empty_cache()
    # attribution: the WKV alone at the same lengths (B=1, 64x64 heads), fwd and fwd+bwd
    for L in LENGTHS:
        x = make_inputs(1, L, torch.bfloat16, 41)
        row = rep.d["kernel"].setdefault(str(L), {})
        for impl in impls:
            try:
                f = _timed(lambda: run_impl(impl, x, False))
                b = _timed(lambda: run_impl(impl, x, True))
                row[impl] = {"fwd_ms": 1e3 * statistics.median(f), "fwdbwd_ms": 1e3 * statistics.median(b)}
            except Exception as e:  # noqa: BLE001
                row[impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
            torch.cuda.empty_cache()
            rep.flush()
    rep.d["fallbacks"] = dict(W.FALLBACKS)  # must stay empty: a fallback would time "chunked" under the fla name
    rep.d["fla_calls"] = dict(W.CALLS)  # must be > 0 for every variant
    rep.flush()


def _env() -> dict:
    torch = _torch()
    d = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import fla
        import triton

        d.update(fla=fla.__version__, triton=triton.__version__)
    except Exception as e:  # noqa: BLE001
        d["fla_error"] = str(e)[:200]
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    return d


def _read(out: Path, name: str):
    p = out / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("kernel", "fork", "model", "throughput", "decide"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.phase == "decide":
        rep = {k: _read(out, k) for k in ("kernel", "fork", "captured", "model", "throughput")}
        for k in ("model", "throughput"):
            if rep[k] and not _fla_really_ran(rep[k]):
                rep[k] = None  # -> INCONCLUSIVE
        d = decide(rep)
        (out / "decision.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps({"verdict": d["verdict"], "chosen": d["chosen"]}))
        return 0 if valid_decision(d) else 9
    if a.phase in ("model", "throughput") and not a.model_dir:
        ap.error("--model-dir is required")
    {"kernel": lambda: phase_kernel(out), "fork": lambda: phase_fork(out),
     "model": lambda: phase_model(out, a.model_dir), "throughput": lambda: phase_throughput(out, a.model_dir)}[a.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

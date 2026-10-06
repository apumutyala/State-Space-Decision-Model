"""fla ``chunk_rwkv7`` as a WKV implementation of the vendored RWKV-7 (``third_party/rwkv7_hf``), registered from OUR
code through the model's own ``RWKV7_WKV_FUNCTIONS`` hook. The vendored files are not modified (sha-pinned).

    from heads.rwkv7_wkv import select_wkv
    select_wkv(model, "fla")          # refuses until GPU_VERIFIED (jobs/fla_rwkv7 gate); allow_unverified=True for the gate
    select_wkv(model, None)           # no-op: the checkpoint's own "chunked"

Default OFF. ``GPU_VERIFIED`` is flipped by the coordinator only after ``jobs/fla_rwkv7`` returns ``ADOPT`` under its
pre-registered rule (``jobs/fla_rwkv7/JOB_SPEC.md``).

Mapping (checked against fla-core 0.5.2 ``fla/ops/generalized_delta_rule/dplr/naive.py``, wheel sha256
5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761). The vendored recurrence keeps S as (value, key):

    S <- S * exp(w_log) + S @ ((-kk) (kk*a)^T) + v k^T ;  out = S @ r

fla's DPLR keeps h = S^T as (key, value): ``h <- h * exp(gk)[:, None] + k v^T + b (a^T h)``, ``o = scale * q^T h``.
So ``chunk_rwkv7(r, w=w_log, k, v, a=-kk, b=kk*a, scale=1.0, initial_state=S^T)`` and the final state is transposed
back. fla's final state is fp32; it is cast back to the carried state's dtype.

Why the name is a ``str`` subclass: modeling_rwkv7 decides ``tokenwise = config.wkv_implementation != "chunked"`` at
three sites (attention, FFN, LM head). Tokenwise means one GEMV per token for every Linear (bit-equality with the
token-serial reference), which would erase any kernel gain. ``WkvName("fla")`` is "fla" for the registry lookup, for
provenance (``json.dumps`` -> "fla") and for ``==``, but answers ``!= "chunked"`` with False, so the model batches its
Linears exactly as it does on "chunked". ``tests/test_fla_rwkv7.py`` pins those three sites in the vendored source.

Clean fallback: any call fla cannot serve as-is (fla not importable, CPU tensors, fp32 activations unless allowed,
a packed batch with B > 1, malformed bounds) runs the vendored ``rwkv7_parallel`` (= "chunked"), with the same
batched Linears, and is counted in ``FALLBACKS`` (and warned once per reason) so a job can refuse silent fallback.
"""
from __future__ import annotations

import warnings
from collections import Counter

import torch

from third_party.rwkv7_hf import modeling_rwkv7 as M

FLA = "fla"
FLA_SAFE_GATE = "fla_safe_gate"  # safe_gate=True: w_log is in [-e^-0.5, 0], inside fla's "approximately [-5, 0)"
FLA_NAMES = (FLA, FLA_SAFE_GATE)
VENDORED = ("eager", "chunked")
FLA_CORE_PIN = "0.5.2"
#: flipped ONLY by the coordinator after jobs/fla_rwkv7 returns ADOPT (receipt path in jobs/fla_rwkv7/JOB_SPEC.md)
GPU_VERIFIED = False
#: fp32 activations: fla warns "ChunkDeltaRuleFunction does not support float32 on some platforms"; the model runs
#: bf16, so fp32 falls back to "chunked" unless the gate measured it (the harness calls the kernel with allow_fp32)
ALLOW_FP32 = False

FALLBACKS: Counter = Counter()
CALLS: Counter = Counter()
_WARNED: set[str] = set()


class WkvName(str):
    """Registry key that the vendored model treats as non-tokenwise (see module docstring)."""

    __slots__ = ()

    def __ne__(self, other):
        if isinstance(other, str) and str.__eq__(other, "chunked"):
            return False
        return str.__ne__(self, other)

    __eq__ = str.__eq__
    __hash__ = str.__hash__

    def __reduce__(self):
        return (WkvName, (str(self),))


def _import_chunk():
    try:
        import fla
        from fla.ops.rwkv7 import chunk_rwkv7
    except Exception as e:  # ImportError on CPU-only boxes; triton/driver errors on broken images
        return None, f"fla_unavailable:{type(e).__name__}"
    if getattr(fla, "__version__", None) != FLA_CORE_PIN:
        return None, f"fla_version:{getattr(fla, '__version__', None)}"
    return chunk_rwkv7, None


def _note_fallback(reason: str) -> None:
    FALLBACKS[reason] += 1
    if reason not in _WARNED:
        _WARNED.add(reason)
        warnings.warn(f"fla RWKV7 WKV fell back to the vendored 'chunked' path ({reason})", RuntimeWarning, stacklevel=3)


def _packed_bounds(cu_seq_lens: torch.Tensor, seq_len: int) -> list[int] | None:
    """Vendored contract: segments in order, empty ones skipped. None = not representable for fla (fall back)."""
    b = [int(x) for x in cu_seq_lens.tolist()]
    if len(b) < 2 or b[0] != 0 or b[-1] != seq_len:
        return None
    keep = [0]
    for s in b[1:]:
        if s < keep[-1]:
            return None
        if s > keep[-1]:
            keep.append(s)
    return keep if len(keep) >= 2 else None


def fallback_reason(r: torch.Tensor, cu_seq_lens: torch.Tensor | None, allow_fp32: bool) -> str | None:
    """Why a call cannot go to fla (None = it can). Pure; CPU-testable apart from the import check."""
    if not r.is_cuda:
        return "not_cuda"
    if r.dtype == torch.float32 and not allow_fp32:
        return "fp32"
    if r.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype:{r.dtype}"
    if r.shape[-1] > 256:  # fla's dplr kernels tile K/V <= 256; the 13.3B has head_dim 64
        return "head_dim"
    if cu_seq_lens is not None:
        if r.shape[0] != 1:
            return "packed_batch"
        if _packed_bounds(cu_seq_lens, r.shape[1]) is None:
            return "packed_bounds"
    return None


def fla_wkv_call(chunk, r, w_log, k, v, kk, a, state, cu_seq_lens=None, *, safe_gate=False, chunk_size=None):
    """The mapped fla call (no fallback logic). Returns (out [B,T,H,V] in r.dtype, state [B,H,V,K] in state.dtype)."""
    fa, fb = -kk, kk * a
    if cu_seq_lens is None:
        o, ht = chunk(r=r, w=w_log, k=k, v=v, a=fa, b=fb, scale=1.0,
                      initial_state=state.transpose(-1, -2).float(), output_final_state=True,
                      safe_gate=safe_gate, chunk_size=chunk_size)
        return o, ht.transpose(-1, -2).to(state.dtype)
    bounds = _packed_bounds(cu_seq_lens, r.shape[1])
    cu_cpu = torch.tensor(bounds, dtype=torch.long)
    # packed rows start every segment from zeros (the vendored contract ignores the passed state's values) and return
    # the LAST segment's state, the one a continuation of this row resumes from
    o, ht = chunk(r=r, w=w_log, k=k, v=v, a=fa, b=fb, scale=1.0, initial_state=None, output_final_state=True,
                  cu_seqlens=cu_cpu.to(r.device), cu_seqlens_cpu=cu_cpu, safe_gate=safe_gate, chunk_size=chunk_size)
    return o, ht[-1:].transpose(-1, -2).to(state.dtype)


def make_fla_wkv(name: str = FLA, safe_gate: bool = False, chunk_size: int | None = None,
                 allow_fp32: bool | None = None):
    """A ``RWKV7_WKV_FUNCTIONS`` entry: fla when it can serve the call, else the vendored ``rwkv7_parallel``."""

    def fla_wkv(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kwargs):
        chunk, err = _import_chunk()
        fp32_ok = ALLOW_FP32 if allow_fp32 is None else allow_fp32
        reason = err or fallback_reason(r, cu_seq_lens, fp32_ok)
        if reason:
            _note_fallback(reason)
            return M.rwkv7_parallel(r, w_log, k, v, kk, a, state, cu_seq_lens=cu_seq_lens)
        CALLS[name] += 1
        return fla_wkv_call(chunk, r, w_log, k, v, kk, a, state, cu_seq_lens, safe_gate=safe_gate,
                            chunk_size=chunk_size)

    fla_wkv.__name__ = f"fla_wkv[{name}]"
    fla_wkv.ssdm_fla = {"name": name, "safe_gate": safe_gate, "chunk_size": chunk_size}
    return fla_wkv


def register() -> None:
    """Idempotent: add the fla entries to the vendored registry (selection is separate and off by default)."""
    M.RWKV7_WKV_FUNCTIONS[FLA] = make_fla_wkv(FLA)
    M.RWKV7_WKV_FUNCTIONS[FLA_SAFE_GATE] = make_fla_wkv(FLA_SAFE_GATE, safe_gate=True)


def _configs(model_or_config):
    """Every distinct config object a model's modules read ``wkv_implementation`` from (usually one, shared)."""
    if not isinstance(model_or_config, torch.nn.Module):
        return [model_or_config]
    seen, out = set(), []
    for mod in model_or_config.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "wkv_implementation") and id(cfg) not in seen:
            seen.add(id(cfg))
            out.append(cfg)
    if not out:
        raise ValueError("no RWKV-7 config with wkv_implementation found on this model")
    return out


def select_wkv(model_or_config, name: str | None, *, allow_unverified: bool = False) -> str:
    """Set the WKV implementation by config. ``None`` keeps the checkpoint's own value (default off). Returns the
    active name. fla names refuse unless ``GPU_VERIFIED`` (or ``allow_unverified=True``, for the gate itself)."""
    cfgs = _configs(model_or_config)
    if name is None:
        return str(cfgs[0].wkv_implementation)
    if name in VENDORED:
        value = str(name)
    elif name in FLA_NAMES:
        if not (GPU_VERIFIED or allow_unverified):
            raise RuntimeError(f"wkv={name!r} is not GPU-verified yet (jobs/fla_rwkv7 gate); "
                               "pass allow_unverified=True only inside that gate")
        register()
        value = WkvName(name)
    else:
        raise ValueError(f"unknown wkv implementation {name!r}; known: {VENDORED + FLA_NAMES}")
    for cfg in cfgs:
        cfg.wkv_implementation = value
    return str(value)


def provenance() -> dict:
    """For kernel_stack / receipts: which path actually ran, and how often it fell back."""
    return {"fla_core_pin": FLA_CORE_PIN, "gpu_verified": GPU_VERIFIED, "calls": dict(CALLS),
            "fallbacks": dict(FALLBACKS)}

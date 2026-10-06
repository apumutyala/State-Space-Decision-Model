"""Fail-fast image guard for flash-linear-attention issue #640 (run at job setup, before any GPU phase).

fla 0.5.2 raises in the gated ``chunk_bwd_dqkwg`` backward (used by every GDN / gated-delta-rule backward) on
Hopper when 3.4.0 <= Triton < 3.7.1 and the TileLang backend is not selected (fla/ops/common/chunk_o.py), because
those Triton releases miscompile that kernel on sm90. SF job-jbw22 (H100) hit exactly this inside test_gdn.

The images fix it on the Triton side: torch 2.13.0 (pins triton==3.7.1), and FLA_TILELANG=0 so the dispatched
backward is the Triton kernel that tests/test_gdn.py measures (not a different TileLang kernel). This guard checks
that the running image really is on that path, with fla's OWN flags as well as our rule, and then runs one tiny
gated fwd+bwd so the failure (if any) costs seconds at setup instead of a whole phase.

    python /opt/ssdm/m1/jobs/common/fla_guard.py --out "$WORK/out/fla_guard.json"   # exit 0 ok, 5 refused

fla not installed (rwkv7 / m2 images) -> nothing to guard, exit 0 with {"fla_installed": false}.
"""
from __future__ import annotations

import json
import os
import re
import sys
from importlib.util import find_spec

MIN_TRITON = (3, 7, 1)  # first Triton release fla 0.5.2 trusts for gated chunk_bwd_dqkwg on Hopper
BAD_FROM = (3, 4, 0)    # first Triton release with the Hopper miscompile (#640)
EXIT_SETUP = 5


def vtuple(v: str) -> tuple[int, ...]:
    """'3.7.1' / '3.7.1+gitabc' / '3.8.0rc1' -> (3, 7, 1) / (3, 7, 1) / (3, 8, 0). Unparseable -> (0,)."""
    m = re.match(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", v or "")
    return tuple(int(x or 0) for x in m.groups()) if m else (0,)


def buggy_gated_bwd(hopper: bool, triton_version: str, tilelang_selected: bool) -> bool:
    """fla 0.5.2's #640 rule: would gated chunk_bwd_dqkwg take the broken Triton path (and raise)?"""
    v = vtuple(triton_version)
    return bool(hopper and BAD_FROM <= v < MIN_TRITON and not tilelang_selected)


def policy_problems(hopper: bool, triton_version: str, tilelang_selected: bool, fla_rule: bool | None) -> list[str]:
    """Everything that makes the image unacceptable. Empty list = ok. Pure (CPU-testable)."""
    probs = []
    if vtuple(triton_version) < MIN_TRITON:
        probs.append(f"triton {triton_version} < {'.'.join(map(str, MIN_TRITON))} (fla #640); image must ship torch 2.13.0 / triton 3.7.1")
    if tilelang_selected:
        probs.append("fla TileLang backend is selected; images pin the Triton path (FLA_TILELANG=0) so test_gdn measures it")
    ours = buggy_gated_bwd(hopper, triton_version, tilelang_selected)
    if ours:
        probs.append("gated chunk_bwd_dqkwg would take the broken Hopper Triton path (#640)")
    if fla_rule is not None and fla_rule != ours:
        probs.append(f"guard rule ({ours}) disagrees with fla's own flags ({fla_rule}); fla changed, re-derive the guard")
    if fla_rule:
        probs.append("fla's own flags say the #640 path is live")
    return probs


def _smoke() -> dict:
    """One tiny gated-delta-rule fwd+bwd (bf16, 2 chunks, dk=dv=128 like the model) -> finite grads or raise."""
    import torch
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, D = 1, 128, 2, 128
    dev, bf = "cuda", torch.bfloat16
    q = torch.randn(B, T, H, D, device=dev, dtype=bf, requires_grad=True)
    k = torch.randn(B, T, H, D, device=dev, dtype=bf, requires_grad=True)
    v = torch.randn(B, T, H, D, device=dev, dtype=bf, requires_grad=True)
    g = F.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32)).requires_grad_(True)
    beta = torch.rand(B, T, H, device=dev, dtype=bf).sigmoid().requires_grad_(True)
    o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o.float().sum().backward()
    torch.cuda.synchronize()
    finite = {n: bool(torch.isfinite(t.grad).all()) for n, t in dict(q=q, k=k, v=v, g=g, beta=beta).items()}
    if not all(finite.values()):
        raise RuntimeError(f"non-finite grads in gated bwd smoke: {finite}")
    return {"shape": [B, T, H, D], "grads_finite": finite}


def check(smoke: bool = True) -> tuple[dict, list[str]]:
    rep: dict = {"fla_installed": find_spec("fla") is not None, "FLA_TILELANG": os.environ.get("FLA_TILELANG")}
    if not rep["fla_installed"]:
        return rep, []
    import importlib.metadata as md

    import torch
    import triton

    rep.update(torch=torch.__version__, triton=triton.__version__, fla_core=md.version("fla-core"),
               tilelang_installed=find_spec("tilelang") is not None, cuda=torch.cuda.is_available())
    if not rep["cuda"]:
        return rep, ["no CUDA device: the guard must run on the job GPU"]
    rep["gpu"], rep["capability"] = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
    from fla.ops.common.backends.tilelang import TileLangBackend
    from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0, TRITON_ABOVE_3_7_1

    tl = bool(TileLangBackend.is_available() and TileLangBackend.is_enabled())
    fla_rule = bool(IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1 and not tl)
    rep.update(hopper=bool(IS_NVIDIA_HOPPER), tilelang_selected=tl, fla_rule_buggy=fla_rule,
               backend="tilelang" if tl else "triton")
    probs = policy_problems(bool(IS_NVIDIA_HOPPER), triton.__version__, tl, fla_rule)
    if smoke and not probs:
        try:
            rep["smoke"] = _smoke()
        except Exception as e:  # noqa: BLE001 - any failure here is a setup failure
            probs.append(f"gated fwd+bwd smoke failed: {e!r}"[:500])
    return rep, probs


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the JSON report here (also printed)")
    ap.add_argument("--no-smoke", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep, probs = check(smoke=not a.no_smoke)
    except Exception as e:  # noqa: BLE001
        rep, probs = {}, [f"guard crashed: {e!r}"[:500]]
    rep["problems"], rep["ok"] = probs, not probs
    s = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(s + "\n")
    print(s, flush=True)  # flush first: the runner's fail message is `tail -1` of the combined log
    if probs:
        print("FLA_GUARD REFUSED: " + "; ".join(probs), file=sys.stderr, flush=True)
        return EXIT_SETUP
    return 0


if __name__ == "__main__":
    sys.exit(main())

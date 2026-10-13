"""Timing harness for the sym-fork fast readout (heads.sym_fast, docs/SYM_FAST.md): per-decision latency of the old
served path vs ``fork2`` vs ``packed1`` on the SAME decider, interleaved item by item (no drift between arms), plus
backbone calls per decision, agreement with the old path and the jevkit Speed score each p50 / p95 would earn.

Smoke (CPU, tiny random Qwen3.5 hybrid, synthetic items; the remote CPU test job, never the laptop)::

    python -X utf8 scripts/sym_fast_timing.py --tiny --n 20 --out timing.json

GPU, a trained sym-fork checkpoint (the loader of jobs.sym_eval.perturb_eval / the screen; jevkit public items)::

    python -X utf8 scripts/sym_fast_timing.py --cell CELL.json --ckpt-dir DIR [--ckpt FILE] --model-dir SNAP \\
        --guard-readiness G --records R --train-dir T --pool-dir P --device cuda --out timing.json [--n 231]

Report (JSON): per arm ``p50_s`` / ``p95_s`` (raw), ``speed`` (jevkit.scoring.speed, self-hosted adjustment),
``calls_per_decision`` (forward pre-hook on the LM), ``max_abs_dp_vs_old`` and ``argmax_agree_vs_old`` over the
served probabilities, ``bitwise_vs_old`` (fraction of decisions with identical served probabilities). Exit 0 always
(a measurement, not a gate); the adoption rule is in docs/SYM_FAST.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ARMS = ("old", "fork2", "packed1")


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _served(resp: dict) -> dict[str, dict[str, float]]:
    ss = resp.get("ssdm") or {}
    return {q: {str(k): float(v) for k, v in d.items()} for q, d in (ss.get("served_probabilities") or {}).items()}


def measure(decider: Any, requests: list[dict], warmup: int = 3, device: str = "cpu") -> dict[str, Any]:
    import torch

    be = decider.backend
    bb = be.model.backbone
    calls = {"n": 0}
    hook = bb.model.register_forward_pre_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    lat: dict[str, list[float]] = {a: [] for a in ARMS}
    ncalls: dict[str, list[int]] = {a: [] for a in ARMS}
    probs: dict[str, list[dict]] = {a: [] for a in ARMS}

    def one(arm: str, req: dict) -> tuple[float, int, dict]:
        bb.sym_exec = None if arm == "old" else arm
        be._row_cache = None  # every arm pays its own row build (the served path does)
        calls["n"] = 0
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        resp = decider.decide(copy.deepcopy(req))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - t0, calls["n"], _served(resp)

    try:
        for req in requests[:warmup]:
            for arm in ARMS:
                one(arm, req)
        for i, req in enumerate(requests):
            order = ARMS[i % 3:] + ARMS[: i % 3]  # rotate the arm order item by item
            for arm in order:
                dt, n, p = one(arm, req)
                lat[arm].append(dt)
                ncalls[arm].append(n)
                probs[arm].append(p)
    finally:
        hook.remove()
        bb.sym_exec = None
    from jevkit import scoring

    out: dict[str, Any] = {"n_decisions": len(requests), "warmup": warmup, "arms": {}}
    for arm in ARMS:
        dps, agree, same = [], 0, 0
        for p_old, p_new in zip(probs["old"], probs[arm]):
            d = max((abs(p_old[q][o] - p_new[q][o]) for q in p_old for o in p_old[q]), default=0.0)
            dps.append(d)
            same += int(p_old == p_new)
            agree += int(all(max(p_old[q], key=p_old[q].get) == max(p_new[q], key=p_new[q].get) for q in p_old))
        p50, p95 = _pct(lat[arm], 0.5), _pct(lat[arm], 0.95)
        out["arms"][arm] = {"p50_s": p50, "p95_s": p95, "mean_s": statistics.fmean(lat[arm]),
                            "speed": scoring.speed(p50, p95, self_hosted=True),
                            "calls_per_decision": statistics.fmean(ncalls[arm]),
                            "max_abs_dp_vs_old": max(dps, default=0.0),
                            "argmax_agree_vs_old": agree / max(1, len(requests)),
                            "bitwise_vs_old": same / max(1, len(requests))}
    o = out["arms"]["old"]
    for arm in ("fork2", "packed1"):
        a = out["arms"][arm]
        a["p50_speedup_vs_old"] = o["p50_s"] / a["p50_s"] if a["p50_s"] else None
        a["speed_gain_vs_old"] = a["speed"]["score"] - o["speed"]["score"]
    return out


def _tiny(n: int, seed: int) -> tuple[Any, list[dict]]:
    import random

    import torch

    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from heads.model import DecisionModel
    from tests.test_sym_invariance import make_request
    from train.data import TOY_TEMPLATE

    torch.manual_seed(seed)
    model = DecisionModel(tiny_hf_backbone("qwen3.5", seed=seed, fork_mode="cache"), "fork", seed=seed).float().eval()
    rng = random.Random(seed)
    reqs = [make_request(rng, rng.randrange(2, 9)) for _ in range(n)]
    return Decider(DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE, layout="sym-fork")), reqs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiny", action="store_true")
    for k in ("cell", "ckpt_dir", "model_dir", "guard_readiness", "records", "train_dir", "pool_dir"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--teacher-dir", dest="teacher_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=0, help="decisions to time (0 = all)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.tiny:
        decider, reqs = _tiny(a.n or 20, a.seed)
        src = {"model": "tiny random qwen3.5 (CPU)", "items": "synthetic make_request"}
    else:  # pragma: no cover - GPU path (load_child needs a trained checkpoint)
        from jevkit.items import load_public
        from jevkit.systemone import item_request
        from jobs.m2_first_cell.fulldev_eval import load_child

        cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))["config"]
        decider, _, ident = load_child(a, temperature=False, readout=(cfg["template"], cfg["layout"]))
        items = load_public()
        reqs = [item_request(it) for it in (items[: a.n] if a.n else items)]
        src = {"cell": a.cell, "ckpt_dir": a.ckpt_dir, "ckpt": a.ckpt, "layout": cfg["layout"],
               "items": "jevkit public", "identity": ident}
    rep = measure(decider, reqs, warmup=a.warmup, device=a.device)
    rep["source"] = src
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("p50_s", "p95_s", "calls_per_decision", "max_abs_dp_vs_old")}
                      for k, v in rep["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

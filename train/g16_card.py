"""G16 eval-card additions, the paired capability cost, and the evidence-bound (c) ablation sweep.

Card fields (``card["g16"]``, filled by ``train.evalcard.run_card`` when the model's head is G16):

* ``served``   per eval item, from the decision just computed (no recompute): independence rate (the decision is the
               same for every E in the box, hence every untrusted text), radius-covers-box rate (the certified joint
               radius exceeds the farthest point of the box: also sound against any text), certified-by-either rate,
               mean / median joint and per-option radii and margins. These are the m2 decision items, so there is no
               G4-over-E (no erasable-origin structure); escalation = not certified by radius or independence.
* ``safety_eval`` (``make_hook``) ``controlsafe.g16.certify`` on the safety_eval DEV requests: certified rate
               (independence, no untrusted text, or G4-over-E erase-and-check) vs escalated, by reason, radii,
               origins evaluated, P-cache misses. ``verify`` additionally checks every origin against a full G16
               recompute.

Capability: the cell's ordinary decision metrics (accuracy, NLL, Brier, ECE) are the capability; the COST of G16
is measured by ``capability_cost`` between two cells that differ only in the head (``jobs/m2_pilot`` pair: same
trust-gated backbone, data, cohort, steps, seed): paired accuracy / debiased-ECE CIs (``train.promotion.
paired_decision``) plus the paired mean NLL delta, on identical eval items.

    python -X utf8 -m train.g16_card sweep --config jobs/m2_pilot/m2_pilot_cell_g16.json --out-dir <dir>
    python -X utf8 -m train.g16_card compare --g16 <g16 out>/units-stepN.json --base <base out>/units-stepN.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_HEAD_KWARGS = {"r": 8, "c": 1.0, "stream_mode": "cache"}
DEFAULT_C_SWEEP = (0.25, 0.5, 1.0, 2.0)


def is_g16(model) -> bool:
    return getattr(getattr(model, "head", None), "name", None) == "g16"


def served_summary(certs: Sequence[dict]) -> dict[str, Any]:
    n = len(certs)
    if not n:
        return {"n": 0}

    def fin(key):
        return [float(c[key]) for c in certs if c.get(key) is not None and math.isfinite(float(c[key]))]

    def stats(key):
        v = fin(key)
        return {"mean": statistics.fmean(v) if v else None, "median": statistics.median(v) if v else None}

    ind = [bool(c["independent_of_untrusted"]) for c in certs]
    box = [bool(c["radius_covers_box"]) for c in certs]
    cert = [a or b for a, b in zip(ind, box)]
    return {"n": n, "independence_rate": sum(ind) / n, "radius_covers_box_rate": sum(box) / n,
            "certified_rate": sum(cert) / n, "escalated_rate": 1 - sum(cert) / n,
            "rho_joint_l2": stats("rho_joint_l2"), "rho_per_option_l2": stats("rho_per_option_l2"),
            "margin": stats("margin"), "c": certs[0].get("c"),
            "note": "decision items: certified = independence OR radius-covers-box (no erasable origins here)"}


def dev_requests(n_route: int = 24, n_pay: int = 16, seed: int | None = None) -> list[dict]:
    """safety_eval DEV-split requests (never the final/test split), metadata stripped."""
    from safety_eval import data

    kw = {} if seed is None else {"seed": seed}
    return [data.strip_meta(r) for r in data.build(n_route, n_pay, split="dev", **kw)]


def certificate_card(model, requests: Sequence[dict], encode, template: str = "jrt-plain-v1",
                     layout: str = "ordered", ks: Sequence[int] = (1,), verify: bool = False) -> dict[str, Any]:
    """``controlsafe.g16.certify`` over every question of every request, per erasure budget k."""
    import torch

    from controlsafe import g16 as G
    from heads.layout import build_row

    head = model.head
    was = model.training
    model.eval()
    out: dict[str, Any] = {"n_requests": len(requests), "split": "dev", "verify": verify}
    try:
        with torch.no_grad():
            for k in ks:
                recs = []
                for r in requests:
                    for qid in sorted(r["questions"]):
                        recs.append(G.certify(head, r, qid, k, lambda q, qid=qid: build_row(
                            q, qid, encode, template=template, layout=layout), verify=verify))
                out[f"k{k}"] = G.certificate_summary(recs)
    finally:
        model.train(was)
    return out


def make_hook(cfg: dict, encode) -> Callable | None:
    """``run_card(..., g16=hook)`` for a G16 cell config (``cfg["g16_card"]``: n_route, n_pay, ks, verify)."""
    if cfg.get("head") != "g16":
        return None
    gc = cfg.get("g16_card") or {}
    reqs = dev_requests(gc.get("n_route", 24), gc.get("n_pay", 16))

    def hook(model) -> dict[str, Any]:
        return certificate_card(model, reqs, encode, template=cfg["template"], layout=cfg["layout"],
                                ks=tuple(gc.get("ks", (1,))), verify=bool(gc.get("verify", False)))

    return hook


# ------------------------------------------------------------------------------------- capability cost
def _fields(units: dict) -> dict[str, dict[str, Any]]:
    ids, labels, probs = units["item_ids"], units["labels"], units["decision_probs"]
    out = {}
    for i, y, p in zip(ids, labels, probs):
        if not all(math.isfinite(x) for x in p):
            raise FloatingPointError(f"{i}: non-finite probabilities")
        k = max(range(len(p)), key=lambda j: p[j])
        out[i] = {"conf": float(p[k]), "correct": float(k == y), "cluster": i.rsplit("/", 1)[0],
                  "nll": -math.log(max(p[y], 1e-12))}
    return out


def capability_cost(g16_units: dict, base_units: dict, n_boot: int = 2000) -> dict[str, Any]:
    """Paired capability cost of G16 (child) vs the otherwise identical non-G16 cell (parent) on identical items."""
    from train.promotion import paired_decision

    c, p = _fields(g16_units), _fields(base_units)
    if set(c) != set(p):
        return {"identical_items": False, "n_g16": len(c), "n_base": len(p)}
    if [g16_units["labels"][g16_units["item_ids"].index(i)] for i in sorted(c)] != \
            [base_units["labels"][base_units["item_ids"].index(i)] for i in sorted(c)]:
        return {"identical_items": False, "reason": "labels differ"}
    res = paired_decision(c, p, n_boot=n_boot)
    d = [c[i]["nll"] - p[i]["nll"] for i in sorted(c)]
    res["nll_g16"] = statistics.fmean(c[i]["nll"] for i in c)
    res["nll_base"] = statistics.fmean(p[i]["nll"] for i in p)
    res["nll_diff_mean"] = statistics.fmean(d)
    res["sign"] = "child = G16, parent = non-G16 (same backbone, data, steps); negative acc_diff = capability cost"
    return res


# ------------------------------------------------------------------------------------- c ablation sweep
def sweep_configs(cfg: dict, cs: Sequence[float] | None = None) -> list[tuple[str, dict]]:
    """One cell config per evidence bound c (``cfg["g16_c_sweep"]`` or ``cs``); everything else identical."""
    if cfg.get("head") != "g16":
        raise ValueError("c sweep is defined for head 'g16' only")
    cs = list(cs if cs is not None else cfg.get("g16_c_sweep") or DEFAULT_C_SWEEP)
    out = []
    for c in cs:
        if not (c > 0):
            raise ValueError(f"evidence bound c must be > 0, got {c}")
        d = copy.deepcopy(cfg)
        d["head_kwargs"] = {**DEFAULT_HEAD_KWARGS, **(d.get("head_kwargs") or {}), "c": float(c)}
        d.pop("g16_c_sweep", None)
        d["g16_sweep_point"] = {"c": float(c), "of": [float(x) for x in cs]}
        out.append((f"c{c:g}", d))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--config", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--c", type=float, nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("--g16", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "sweep":
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        od = Path(a.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for name, d in sweep_configs(cfg, a.c):
            p = od / f"{Path(a.config).stem}_{name}.json"
            p.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
            print(p)
        return 0
    res = capability_cost(json.loads(Path(a.g16).read_text(encoding="utf-8")),
                          json.loads(Path(a.base).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

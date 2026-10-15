"""Toy constrained problem with a known optimum, for the nuPI vs gradient-ascent comparison (docs/NUPI.md).

    min_x  f(x) = 0.5 ||x - c||^2        s.t.  g(x) = a . x <= b

with c = (2, 2), a = (1, 1), b = 1: x* = (0.5, 0.5), lambda* = 1.5, f* = 2.25. The primal takes gradient steps on the
Lagrangian f + lambda (g - b) (lr ``eta``); the multiplier is set by a ``PIController`` from the (optionally noisy)
measurement of g, i.e. the same simultaneous primal step / multiplier update the trainer runs.

    python -X utf8 -m train.nupi_toy --out docs/nupi_toy_traces.json [--svg docs/nupi_toy_traces.svg]
"""

from __future__ import annotations

import argparse
import json
import math
import random

from .pi_control import PIController, gradient_ascent

C, A, B = (2.0, 2.0), (1.0, 1.0), 1.0
X_STAR, LAM_STAR = (0.5, 0.5), 1.5


def run(ctl: PIController, steps: int = 600, eta: float = 0.05, noise: float = 0.0, seed: int = 0) -> dict:
    rng = random.Random(seed)
    x = [0.0, 0.0]
    lam_tr, g_tr = [], []
    for _ in range(steps):
        lam = ctl.lam
        gx = sum(a * xi for a, xi in zip(A, x))
        # primal: x <- x - eta (grad f + lambda grad g)
        x = [xi - eta * ((xi - ci) + lam * a) for xi, ci, a in zip(x, C, A)]
        meas = gx + (rng.gauss(0.0, noise) if noise else 0.0)
        ctl.update(meas - B)
        lam_tr.append(ctl.lam)
        g_tr.append(gx)
    return {"lam": lam_tr, "g": g_tr, "x": x}


def metrics(tr: dict, tol: float = 0.01) -> dict:
    lam, g = tr["lam"], tr["g"]
    e = [v - B for v in g]
    tv = sum(abs(lam[i + 1] - lam[i]) for i in range(len(lam) - 1))
    signs = sum(1 for i in range(len(e) - 1) if (e[i] > tol) != (e[i + 1] > tol) and abs(e[i + 1]) > tol)
    settle = next((t for t in range(len(e)) if all(abs(v) <= tol for v in e[t:])), None)
    tail = e[len(e) // 2:]
    return {"final_violation": max(0.0, e[-1]), "final_abs_e": abs(e[-1]),
            "x_err": math.dist(tr["x"], X_STAR), "lam_final": lam[-1], "lam_err": abs(lam[-1] - LAM_STAR),
            "lam_overshoot": max(lam) - LAM_STAR, "lam_total_variation": tv, "e_crossings": signs,
            "settle_step": settle, "max_violation_2nd_half": max(0.0, max(tail)),
            "mean_abs_e_2nd_half": sum(abs(v) for v in tail) / len(tail)}


def compare(steps: int = 600, eta: float = 0.05, ki: float = 0.05, kp: float = 2.0, nu: float = 0.0,
            noise: float = 0.0, seed: int = 0) -> dict:
    ga = run(gradient_ascent(ki), steps, eta, noise, seed)
    pi = run(PIController(kappa_p=kp, kappa_i=ki, nu=nu, integral_min=0.0, integral_max=1e6), steps, eta, noise, seed)
    return {"setup": {"steps": steps, "eta": eta, "kappa_i": ki, "kappa_p": kp, "nu": nu, "noise": noise,
                      "seed": seed, "x_star": X_STAR, "lambda_star": LAM_STAR},
            "ga": {"metrics": metrics(ga), "lam": ga["lam"], "g": ga["g"]},
            "nupi": {"metrics": metrics(pi), "lam": pi["lam"], "g": pi["g"]}}


def traces_svg(res: dict, w: int = 300, h: int = 150, pad: int = 28) -> str:
    """Dependency-free SVG of the lambda traces (top row) and constraint errors (bottom row): GA grey, nuPI blue."""
    cols = list(res.items())
    W, H = len(cols) * (w + pad) + pad, 2 * (h + pad) + pad + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10">',
           f'<rect width="{W}" height="{H}" fill="white"/>']

    def panel(x0, y0, series, ref, title):
        lo = min(min(v) for v, _ in series + [([ref], "")])
        hi = max(max(v) for v, _ in series + [([ref], "")])
        span = (hi - lo) or 1.0
        n = max(len(v) for v, _ in series)
        sx = lambda i: x0 + i * w / max(1, n - 1)
        sy = lambda v: y0 + h - (v - lo) / span * h
        out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        out.append(f'<text x="{x0}" y="{y0 - 4}">{title}</text>')
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" stroke="black" '
                   f'stroke-dasharray="4 3" stroke-width="0.8"/>')
        out.append(f'<text x="{x0 + w + 2}" y="{sy(hi) + 8:.1f}">{hi:.2f}</text>'
                   f'<text x="{x0 + w + 2}" y="{sy(lo):.1f}">{lo:.2f}</text>')
        for v, color in series:
            pts = " ".join(f"{sx(i):.1f},{sy(y):.1f}" for i, y in enumerate(v))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')

    for c, (k, v) in enumerate(cols):
        x0 = pad + c * (w + pad)
        panel(x0, pad, [(v["ga"]["lam"], "#999"), (v["nupi"]["lam"], "#1f6feb")], LAM_STAR, f"{k}: lambda_t")
        panel(x0, 2 * pad + h, [([g - B for g in v["ga"]["g"]], "#999"), ([g - B for g in v["nupi"]["g"]], "#1f6feb")],
              0.0, f"{k}: g - eps")
    out.append(f'<text x="{pad}" y="{H - 6}">grey = gradient ascent on lambda (kappa_p = 0); blue = nuPI; dashed = '
               f'lambda* = {LAM_STAR} / feasibility boundary</text></svg>')
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--svg")
    a = ap.parse_args(argv)
    res = {"deterministic ki=0.05": compare(ki=0.05, kp=1.0), "deterministic ki=0.2": compare(ki=0.2, kp=4.0),
           "noisy (sigma=0.3)": compare(ki=0.05, kp=0.5, nu=0.9, noise=0.3)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: {"setup": v["setup"], "ga": v["ga"]["metrics"], "nupi": v["nupi"]["metrics"],
                       "lam_ga": [round(x, 6) for x in v["ga"]["lam"]],
                       "lam_nupi": [round(x, 6) for x in v["nupi"]["lam"]]} for k, v in res.items()}, f, indent=1)
    for k, v in res.items():
        print(k, "GA", json.dumps(v["ga"]["metrics"]), "\n", k, "nuPI", json.dumps(v["nupi"]["metrics"]))
    if a.svg:
        with open(a.svg, "w", encoding="utf-8", newline="\n") as f:
            f.write(traces_svg(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

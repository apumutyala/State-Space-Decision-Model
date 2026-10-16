"""CPU-only shortcut-onset experiment; raw seed traces stay outside the repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import t as student_t
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from train.pi_control import PIController  # noqa: E402


@dataclass(frozen=True)
class Config:
    warmup: int = 400
    steps: int = 800
    train_rows: int = 2048
    monitor_rows: int = 256
    canary_rows: int = 512
    test_rows: int = 4096
    batch: int = 128
    every: int = 5
    lr: float = 0.003
    correlation: float = 0.98
    target: float = 0.01
    warning: float = 0.005
    harm: float = 0.10
    kp: float = 10.0
    ki: float = 0.1
    nu: float = 0.9


ARMS = {"off": 0.0, "fixed_0.5": 0.5, "fixed_2": 2.0, "fixed_8": 8.0,
        "nupi": None, "no_shift": 0.0}


def examples(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 9, generator=rng)
    y = ((x[:, 0] > 0) != (x[:, 1] > 0)).float()
    x[:, -1] = 0
    return x, y


def paired(x: torch.Tensor) -> torch.Tensor:
    both = torch.cat((x, x))
    both[:len(x), -1], both[len(x):, -1] = 1, -1
    return both


def reliance(model: nn.Module, both: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    plus, minus = model(both).flatten().chunk(2)
    return (plus.sigmoid() - minus.sigmoid()).square().mean(), ((plus > 0) != (minus > 0)).float().mean()


def alarm(trace: list[dict], key: str, threshold: float) -> int | None:
    # Timestamp is when the second observation confirms persistence, never backdated.
    for before, now in zip(trace, trace[1:]):
        if before[key] > threshold and now[key] > threshold:
            return now["step"]
    return None


def run(seed: int, arm: str, cfg: Config) -> dict:
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctl = PIController(cfg.kp, cfg.ki, cfg.nu, lambda_max=20, integral_max=200)
    train_x, train_y = examples(cfg.train_rows, 10_000 + seed)
    monitor_x, _ = examples(cfg.monitor_rows, 20_000 + seed)
    canary_x, canary_y = examples(cfg.canary_rows, 30_000 + seed)
    monitor, canary = paired(monitor_x), paired(canary_x)
    rng = torch.Generator().manual_seed(40_000 + seed)
    total = cfg.warmup + cfg.steps
    indices = torch.randint(cfg.train_rows, (total, cfg.batch), generator=rng)
    coins = torch.rand(total, cfg.batch, generator=rng)
    trace, measurements, lambdas = [], [], []

    def observe(step: int) -> None:
        with torch.no_grad():
            g, flip = reliance(model, canary)
            acc = ((model(canary_x).flatten() > 0) == canary_y.bool()).float().mean()
        trace.append({"step": step, "g": float(g), "flip": float(flip), "accuracy": float(acc)})

    for t in range(total):
        active = t >= cfg.warmup
        if t == cfg.warmup:
            observe(0)
        x, y = train_x[indices[t]].clone(), train_y[indices[t]]
        correlation = cfg.correlation if active and arm != "no_shift" else 0.5
        x[:, -1] = (2 * y - 1) * torch.where(coins[t] < correlation, 1.0, -1.0)
        lam = (ctl.lam if arm == "nupi" else ARMS[arm]) if active else 0.0
        loss = F.binary_cross_entropy_with_logits(model(x).flatten(), y)
        if lam:
            g, _ = reliance(model, monitor)
            loss = loss + lam * (g - cfg.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if active:
            with torch.no_grad():
                measured, _ = reliance(model, monitor)
            measurements.append(float(measured))
            lambdas.append(lam)
            if arm == "nupi":
                ctl.update(float(measured) - cfg.target)
            step = t - cfg.warmup + 1
            if step % cfg.every == 0:
                observe(step)

    # Final held-out data are created only after all learning; never feed a controller or choose an arm.
    test_x, test_y = examples(cfg.test_rows, 50_000 + seed)
    with torch.no_grad():
        g, flip = reliance(model, paired(test_x))
        clean = ((model(test_x).flatten() > 0) == test_y.bool()).float().mean()
        test_x[:, -1] = -(2 * test_y - 1)
        wrong = ((model(test_x).flatten() > 0) != test_y.bool()).float().mean()
    values, ls = np.asarray(measurements), np.asarray(lambdas)
    residual = values - cfg.target
    late = values[-cfg.steps // 4:]
    warning, harm = alarm(trace, "g", cfg.warning), alarm(trace, "flip", cfg.harm)
    metrics = {"accuracy": float(clean), "g": float(g), "flip": float(flip), "opposed_error": float(wrong),
               "violation_fraction": float(np.mean(residual > 0)), "mean_excess": float(np.maximum(residual, 0).mean()),
               "overshoot": float(max(0, residual.max())), "late_sd": float(late.std()),
               "late_range": float(np.ptp(late)), "crossings": int(np.count_nonzero(np.diff(residual > 0))),
               "lambda_final": float(ls[-1]), "lambda_tv": float(np.abs(np.diff(ls)).sum()),
               "lambda_saturation": float(np.mean(ls >= 20)), "warmup_accuracy": trace[0]["accuracy"]}
    return {"seed": seed, "arm": arm, "metrics": metrics, "warning_step": warning, "harm_step": harm,
            "lead_steps": harm - warning if warning is not None and harm is not None else None,
            "trace": trace, "monitor": measurements, "lambda": lambdas, "seconds": time.perf_counter() - started}


def ci(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    half = float(student_t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None
    return {"mean": round(float(a.mean()), 7), "half95": round(half, 7) if half is not None else None, "n": len(a)}


def summarize(runs: list[dict], cfg: Config, seeds: list[int]) -> dict:
    arms = {}
    for arm in ARMS:
        selected = [r for r in runs if r["arm"] == arm]
        arms[arm] = {key: ci([r["metrics"][key] for r in selected]) for key in selected[0]["metrics"]}
        lead = [r["lead_steps"] for r in selected if r["lead_steps"] is not None]
        arms[arm]["detectors"] = {"warnings": sum(r["warning_step"] is not None for r in selected),
                                   "harms": sum(r["harm_step"] is not None for r in selected),
                                   "strictly_early": sum(x > 0 for x in lead), "lead_steps": ci(lead) if lead else None,
                                   "warning_step": ci([r["warning_step"] for r in selected if r["warning_step"] is not None])
                                   if any(r["warning_step"] is not None for r in selected) else None,
                                   "harm_step": ci([r["harm_step"] for r in selected if r["harm_step"] is not None])
                                   if any(r["harm_step"] is not None for r in selected) else None}
    paired_differences = {}
    for comparator in ("off", "fixed_0.5", "fixed_2", "fixed_8"):
        pairs = [(next(r for r in runs if r["seed"] == seed and r["arm"] == "nupi"),
                  next(r for r in runs if r["seed"] == seed and r["arm"] == comparator)) for seed in seeds]
        paired_differences[comparator] = {key: ci([a["metrics"][key] - b["metrics"][key] for a, b in pairs])
                                          for key in ("accuracy", "g", "flip", "mean_excess", "overshoot")}
    return {"config": asdict(cfg), "seeds": seeds, "arms": arms, "nupi_minus": paired_differences}


def figure(runs: list[dict], cfg: Config, path: Path) -> None:
    # Compact dependency-free SVG: canary trajectories and held-out final frontier.
    colors = {"off": "#b44", "fixed_0.5": "#b80", "fixed_2": "#687", "fixed_8": "#38a", "nupi": "#729", "no_shift": "#888"}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300">',
           '<rect width="800" height="300" fill="white"/><g font-family="sans-serif" font-size="12">',
           '<text x="35" y="20">Mean canary reliance (g), log scale</text>',
           '<text x="440" y="20">Final trade-off: intended accuracy vs decision flips</text>']
    sx, sy = lambda t: 45 + 340 * t / cfg.steps, lambda g: 240 - 200 * (math.log10(max(g, 0.0001)) + 4) / 4
    for g in (0.0001, 0.001, 0.01, 0.1, 1):
        y = sy(g)
        svg += [f'<path d="M45 {y:.1f} H385" stroke="#ddd"/>', f'<text x="0" y="{y+4:.1f}">{g:g}</text>']
    svg += [f'<path d="M45 {sy(cfg.target):.1f} H385" stroke="black" stroke-dasharray="4 3"/>',
            '<text x="120" y="260">Updates after shortcut introduction (0–800)</text>',
            '<path d="M455 40 V240 H780" fill="none" stroke="black"/>',
            '<text x="495" y="260">Shortcut-free accuracy (80–100%)</text>',
            '<text x="425" y="42">100%</text><text x="437" y="240">0%</text>']
    for i, arm in enumerate(ARMS):
        selected = [r for r in runs if r["arm"] == arm]
        means = np.mean([[p["g"] for p in r["trace"]] for r in selected], axis=0)
        points = ' '.join(f'{sx(p["step"]):.1f},{sy(g):.1f}' for p, g in zip(selected[0]["trace"], means))
        acc, flip = (np.mean([r["metrics"][key] for r in selected]) for key in ("accuracy", "flip"))
        svg += [f'<polyline points="{points}" fill="none" stroke="{colors[arm]}" stroke-width="2"/>',
                f'<circle cx="{455 + (acc - .8) / .2 * 325:.1f}" cy="{240 - flip * 200:.1f}" r="4" fill="{colors[arm]}"/>',
                f'<text x="{12+i*130}" y="285" fill="{colors[arm]}">{arm}</text>']
    path.write_text('\n'.join(svg + ['</g></svg>']) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Private output directory")
    parser.add_argument("--public", type=Path, help="Optional compact aggregate and figure destination")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 108)))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, started = Config(), time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for arm in ARMS:
            result = run(seed, arm, cfg)
            runs.append(result)
            (args.out / f'{seed}_{arm}.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
            print(json.dumps({k: result[k] for k in ('seed', 'arm', 'metrics', 'warning_step', 'harm_step', 'seconds')}), flush=True)
    summary = summarize(runs, cfg, args.seeds)
    summary["provenance"] = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                             "scipy": scipy.__version__, "device": "cpu", "threads": 1, "parameters": 1409,
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "controller_sha256": hashlib.sha256((ROOT / 'train/pi_control.py').read_bytes()).hexdigest(),
                             "wall_seconds": round(time.perf_counter() - started, 2),
                             "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "paid_usd": 0}
    destination = args.public or args.out
    destination.mkdir(parents=True, exist_ok=True)
    # One line per metric, not per example/seed; raw trajectories never enter this summary.
    text = json.dumps(summary, indent=2)
    import re
    text = re.sub(r'\{\s*"mean": ([^,]+),\s*"half95": ([^,]+),\s*"n": ([^}]+)\}',
                  lambda m: '{"mean": ' + m[1] + ', "half95": ' + m[2] + ', "n": ' + m[3].strip() + '}', text)
    (destination / 'aggregate.json').write_text(text + '\n', encoding='utf-8')
    figure(runs, cfg, destination / 'tradeoff.svg')
    print(json.dumps(summary["provenance"]), flush=True)


if __name__ == '__main__':
    main()

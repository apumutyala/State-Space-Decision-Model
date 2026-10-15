"""Run ``python -m keel.monitor --help`` for the private-data CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.paths import private_root


def main():
    parser = argparse.ArgumentParser(description="Execution-verified trusted code monitor experiment")
    parser.add_argument("command", choices=("data", "evaluate", "report"))
    parser.add_argument("--private-dir", type=Path, default=private_root() / "monitor_v0")
    parser.add_argument("--output-dir", type=Path, default=Path("direction/experiments/monitor_v0"))
    parser.add_argument("--problems", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--author-cap", type=float, default=5.0)
    parser.add_argument("--spend-cap", type=float, default=15.0)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--search-units", type=int, default=12)
    parser.add_argument("--arms", default="source,stock,student,think")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    private = args.private_dir.expanduser().resolve()
    if private.is_relative_to(repo):
        parser.error("per-item data and caches must stay outside the worktree")
    private.mkdir(parents=True, exist_ok=True)
    if args.command == "data":
        from .build import build_dataset, save
        report = build_dataset(private, count=args.problems, batch_size=args.batch_size, workers=args.workers,
                               max_seconds=args.max_seconds, author_cap=args.author_cap)
        save(args.output_dir / "data.json", report)
    elif args.command == "evaluate":
        from .run import evaluate
        report = evaluate(private, args.output_dir, args.arms.split(","), args.search_units, args.spend_cap)
    else:
        from .run import report_only
        report = report_only(private, args.output_dir)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

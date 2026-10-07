"""Dry-run stand-ins called by jobs/full8/full8_job.sh as `python -m tests.full8_dryrun.stubs <cmd> ...` (FULL8_DRYRUN).

  setup <out_dir>         replaces nvidia-smi + the version pins + the fla #640 guard (GPU facts, not testable on CPU)
  fetch <data_root>       replaces sf_fetch: DONE_* markers + paths.env + a tiny pinned-snapshot dir (config.json only)
  subset <cmd> ...        replaces dev_subset score / compare / ece (they need the private DEV rows + the kit); checks
                          that both arms wrote one result per subset row, then returns fixed, labelled numbers
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DRY = {"dry_run": True, "note": "tests/full8_dryrun stand-in, not a measurement"}


def setup(out_dir):
    import torch

    d = Path(out_dir)
    (d / "nvidia-smi.txt").write_text("DRY RUN: CPU only, no GPU\n")
    (d / "env.json").write_text(json.dumps({**DRY, "torch": torch.__version__, "cuda": None}))
    (d / "fla_guard.json").write_text(json.dumps({**DRY, "ok": True}))
    print("dry-run setup ok")


def fetch(root):
    from tests.full8_dryrun.tiny import write_snapshot

    root = Path(root)
    snap = write_snapshot(root / "hf")
    model_dir = snap.parent  # .../snapshots, as the real Q9B_DIR (resolve_model_dir picks the single snapshot)
    lines = [f"export Q9B_DIR={model_dir.as_posix()}", f"export WIKITEXT_DIR={(root / 'hf' / 'wikitext').as_posix()}",
             f"export MMLU_DIR={(root / 'hf' / 'mmlu').as_posix()}"]
    with open(root / "paths.env", "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    for n in ("Q9B_DIR", "WIKITEXT_DIR", "MMLU_DIR"):
        (root / f"DONE_{n}").write_text("dry-run")
    print(f"dry-run fetch -> {model_dir}")


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def subset(argv):
    import argparse

    from jobs.common import dev_subset

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--results")
    ap.add_argument("--candidate")
    ap.add_argument("--baseline")
    ap.add_argument("--subset", required=True)
    a = ap.parse_args(argv)
    ids = set(dev_subset.load_subset(a.subset)["run_ids"])
    for p in [x for x in (a.results, a.candidate, a.baseline) if x]:
        got = {r["run_id"] for r in _rows(p)}
        if got != ids:
            raise SystemExit(f"{p}: {len(got)} result rows, subset has {len(ids)}")
    if a.cmd == "compare":
        out = {**DRY, "diff": {"index": {"point": 0.0, "lo": -1.0, "hi": 1.0, "se": 0.5}}, "rows": len(ids)}
    elif a.cmd == "score":
        out = {**DRY, "slice_index": {"index": 30.0}, "rows": len(ids)}
    else:
        out = {**DRY, "ece": 0.1}
    print(json.dumps(out))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = argv[0], argv[1:]
    if os.path.isdir("/opt/ssdm/m1"):
        raise SystemExit("dry-run stubs refused inside the job image")
    if cmd == "setup":
        setup(*rest)
    elif cmd == "fetch":
        fetch(*rest)
    elif cmd == "subset":
        subset(rest)
    else:
        raise SystemExit(f"unknown dry-run command {cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

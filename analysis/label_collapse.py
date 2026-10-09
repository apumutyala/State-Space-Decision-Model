"""Agent B: per-family LABEL-COLLAPSE check of any landed predictions (NON-GATING warning; CPU only).

    CUDA_VISIBLE_DEVICES= python -X utf8 -I analysis/label_collapse.py <path> [<path> ...] [--predictions SUBSTR]
        [--edition 0.2.1|0.3] [--json OUT.json]

<path> = a results.jsonl(.gz) / *.results.jsonl / units[-child]-{index,g13}.json file, a .tar/.tgz artifact, or an
artifact dir. Archives and dirs are walked as UNTRUSTED data with ``analysis/rescore_keep.py``'s readers
(``walk_artifact`` / ``_candidates``: nothing extracted, links skipped, identical copies collapse), so landings of cells
launched from older branches can be checked too. Gold labels come from the frozen scorer's DEV row metadata
(``harness.cli.load_slice``), restricted to the predicted run_ids; the rule is ``jobs/common/label_collapse.py``'s.
Prints one line per distinct prediction set (flags), and the full warnings block with ``--json``.
Exit 0 always when something was checked (a warning is not a failure); 2 when nothing could be read.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # python -I drops the script dir; the repo root is trusted code, the artifact is not
    sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "analysis"))

import rescore_keep as RK  # noqa: E402  - the prediction readers (walk_artifact, _candidates, Member)
from jobs.common.label_collapse import detect  # noqa: E402


def _members(path: Path) -> list:
    if path.is_dir():
        ms, notes = RK.walk_artifact(path)
        for n in notes:
            print(f"note: {n}", file=sys.stderr)
        return RK._candidates(ms)  # noqa: SLF001
    if RK._is_archive(path.name):  # noqa: SLF001
        out, notes = [], []
        RK._tar_members(tarfile.open(path, mode="r:*"), path.name, 1, out, notes)  # noqa: SLF001
        for n in notes:
            print(f"note: {n}", file=sys.stderr)
        return RK._candidates(out)  # noqa: SLF001
    return [RK.Member(str(path), "<file>", path.name, path.stat().st_size, lambda p=path: p.read_bytes())]


def read_predictions(m) -> dict[str, dict]:
    raw = m.read()
    if m.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    if m.name.endswith(".json"):
        res = json.loads(raw.decode("utf-8")).get("index_results")
        if not isinstance(res, dict):
            raise ValueError("no index_results")
        return res
    out = {}
    for line in io.StringIO(raw.decode("utf-8")):
        if line.strip():
            r = json.loads(line)
            out[r["run_id"]] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="label_collapse", description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--predictions", help="substring of the prediction file label to check (default: every one)")
    ap.add_argument("--edition", default="0.2.1", choices=["0.2.1", "0.3"])
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    from harness.cli import load_slice

    rows_all = load_slice("dev", a.edition, a.edition != "0.2.1")
    report, seen = [], set()
    for p in a.paths:
        for m in _members(Path(p)):
            if a.predictions and a.predictions not in m.label:
                continue
            try:
                sha = m.sha256()
                if sha in seen:
                    continue
                seen.add(sha)
                res = read_predictions(m)
            except (ValueError, KeyError, OSError, UnicodeDecodeError, EOFError, tarfile.TarError) as e:
                print(f"skip {m.label}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            rows = [r for r in rows_all if r["_evaluation"]["run_id"] in res]
            w = detect(rows, res)
            report.append({"predictions": m.label, "sha256": sha, "rows_scored": len(rows),
                           "rows_unmatched": len(res) - len(rows), "warnings": w})
            fl = lambda xs: ", ".join(f"{e['family']}({e['benchmark']}):{e['label']} gold {e['gold_share']:.0%} "  # noqa: E731
                                      f"pred {e['pred_share']:.1%} n={e['n']}" for e in xs) or "-"
            print(f"{m.label} [{sha[:12]}] rows={len(rows)}\n  COLLAPSE {w['label_collapse_count']}: "
                  f"{fl(w['label_collapse'])}\n  near-collapse {w['label_near_collapse_count']}: "
                  f"{fl(w['label_near_collapse'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if report else 2


if __name__ == "__main__":
    sys.exit(main())

"""Paired full-DEV comparison of a candidate against Clef (27B) and Clef-flash (COORDINATION.md "Full verification").

    python -X utf8 -m analysis.paired_vs_clef --candidate <cand results.jsonl> \
        --clef <Clef full-DEV results.jsonl> --clef-flash <Clef-flash full-DEV results.jsonl> [--out report.json]

All three inputs are per-row decide results files (the kit's results.jsonl) on the 12,104-row DEV slice. Scoring and
the bootstrap are the repo's own: ``harness.score.decompose`` + ``harness.bootstrap.compare`` (case-cluster bootstrap,
same B and seed for both pairings, so the two diffs share replicate weights) and ``harness.calibration`` for ECE.

Reference files (use the same over-length handling on every side):
  * Clef-flash: job-sgihj decide results (56.79), or its 32k-substituted file from job-ftpnf (57.35).
  * Clef 27B:   job-jxevr index/decide/results.jsonl, or index/decide_32k_substituted/results.jsonl.
The output names which files were paired (sha256) so a reader can tell 16k from 32k references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from harness import bootstrap, calibration, cli, config, score


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(rows, path, allow_missing):
    res = cli.load_results(path, cli._ids(rows), allow_missing)
    score.require_results(rows, res, allow_missing, path)
    return res


def _ece(rows, res) -> dict:
    r = calibration.report(calibration.fields(rows, res))
    return {"board_protocol_ece": r["board_protocol"].get("ece"), "overall_ece_10bin": r["overall"].get("ece_10bin"),
            "overall_smece": r["overall"].get("smece"), "full": r}


def run(candidate: str, clef: str, clef_flash: str, edition: str | None = None, B: int = 2000, seed: int = 0,
        allow_missing: bool = False, omit_gsm8k03: bool = False) -> dict:
    rows = cli.load_slice("dev", edition, not omit_gsm8k03)
    files = {"candidate": candidate, "clef": clef, "clef_flash": clef_flash}
    res = {k: _load(rows, p, allow_missing) for k, p in files.items()}
    benches = {k: score.decompose(rows, r, edition=edition) for k, r in res.items()}
    out = {"edition": config.edition_id(edition), "slice": "dev", "rows": len(rows), "B": B, "seed": seed,
           "files": {k: {"path": str(Path(p).resolve()), "sha256": _sha(p)} for k, p in files.items()},
           "index": {}, "vs": {}, "ece": {}}
    for ref in ("clef", "clef_flash"):
        c = bootstrap.compare(benches["candidate"], benches[ref], B=B, seed=seed, allow_unaligned=allow_missing)
        out["vs"][ref] = {"diff_index": c["diff"]["index"], "diff_areas": c["diff"]["areas"],
                          "diff_benchmarks": c["diff"]["benchmarks"]}
        out["index"].setdefault("candidate", {"index": c["candidate"]["index"], "areas": c["candidate"]["areas"]})
        out["index"][ref] = {"index": c["baseline"]["index"], "areas": c["baseline"]["areas"]}
    for k, r in res.items():
        out["ece"][k] = _ece(rows, r)
    out["beats_both"] = all(out["vs"][r]["diff_index"]["lo"] > 0 for r in ("clef", "clef_flash"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis.paired_vs_clef", description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--clef", required=True)
    ap.add_argument("--clef-flash", required=True)
    ap.add_argument("--edition", default=config.EDITION, choices=config.EDITIONS)
    ap.add_argument("--omit-gsm8k03", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true", help="score missing rows as wrong (unaligned pairing)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = run(a.candidate, a.clef, a.clef_flash, a.edition, a.B, a.seed, a.allow_missing, a.omit_gsm8k03)
    s = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    brief = {"index": {k: v["index"] for k, v in out["index"].items()},
             "diff_vs": {k: v["diff_index"] for k, v in out["vs"].items()},
             "board_ece": {k: v["board_protocol_ece"] for k, v in out["ece"].items()}, "beats_both": out["beats_both"]}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Land R2 (Clef 27B full-DEV comparator) from a downloaded Clef index job artifact, in one command.

    python -I -X utf8 analysis/land_r2.py --art C:/Users/apumu/clef_jxevr_art [--job job-jxevr --artifact art-...]
        [--candidate <cand results.jsonl>] [--out report.json]

``--art`` is the extracted artifact directory (searched for ``clef-index-full/`` or ``clef-index-subset/``; the row
mode follows: full = 12,104 DEV rows, subset = the 2,008-row m1 subset). Pairs, with ``analysis.paired_vs_clef``
(frozen scorer + bootstrap, row sets asserted identical, over-length context asserted equal on both sides):
  * 16k: the job's index/decide/results.jsonl vs Clef-flash job-sgihj decide results;
  * 32k: the job's index/decide_32k_substituted/results.jsonl vs Clef-flash job-ftpnf decide_32k_substituted.
It also re-checks the job's own score_decide*.json index points against the recomputed ones (must agree to 1e-6)
and prints the ledger-ready row plus both paired tables. CPU only; reads the artifact as data (run with python -I).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from analysis import paired_vs_clef as P  # noqa: E402

HOME = Path(os.environ.get("USERPROFILE") or Path.home())
FLASH_16K = HOME / "research/CHLabs/review-artifacts/rwkv7-baseline-preserve-20261007/clef-flash-job-sgihj-results.jsonl"
FLASH_16K_SHA = "463f21ff492fc792da4d40ecc501cffab0498d2458715f8effd3270614cbd0aa"  # run_job.sh REF_SHA
FLASH_32K = HOME / "clefl/clefl_ftpnf/x/clef-flash-index-overlength/index/decide_32k_substituted/results.jsonl"


def find_job_dir(art: Path) -> tuple[Path, str]:
    for name, mode in (("clef-index-full", "full"), ("clef-index-subset", "subset")):
        hits = sorted(p for p in art.rglob(name) if (p / "index" / "decide" / "results.jsonl").is_file())
        if len(hits) > 1:
            raise SystemExit(f"{art}: {len(hits)} {name} dirs with decide results: {hits}")
        if hits:
            return hits[0], mode
    raise SystemExit(f"{art}: no clef-index-full/ or clef-index-subset/ with index/decide/results.jsonl")


def _job_point(path: Path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return (d.get("slice_index") or {}).get("index", {}).get("point")


def _ci(c) -> str:
    return f"{c['point']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"


def land(art: str, candidate: str | None = None, flash16: str = str(FLASH_16K), flash32: str = str(FLASH_32K),
         B: int = 2000, seed: int = 0, job: str = "?", artifact: str = "?") -> dict:
    jd, mode = find_job_dir(Path(art))
    idx = jd / "index"
    if P._sha(flash16) != FLASH_16K_SHA:
        raise SystemExit(f"{flash16}: not the pinned job-sgihj decide results (sha256 {FLASH_16K_SHA[:12]}...)")
    arms = {"16k": (idx / "decide" / "results.jsonl", flash16, idx / "score_decide.json")}
    s32 = idx / "decide_32k_substituted" / "results.jsonl"
    if s32.is_file():
        arms["32k"] = (s32, flash32, idx / "score_decide_32k_substituted.json")
    out = {"job_dir": str(jd), "rows_mode": mode, "job": job, "artifact": artifact, "arms": {}, "job_score_check": {}}
    for ctx, (clef, flash, job_score) in arms.items():
        r = P.run(str(clef), flash, candidate, mode, "0.2.1", B, seed, context=ctx)
        out["arms"][ctx] = r
        mine, theirs = r["index"]["clef"]["index"]["point"], _job_point(job_score)
        ok = theirs is None or abs(mine - theirs) < 1e-6
        out["job_score_check"][ctx] = {"recomputed": mine, "job_file": theirs, "ok": ok}
        if not ok:
            raise SystemExit(f"{ctx}: recomputed Clef index {mine} != the job's {job_score.name} {theirs}")
    return out


def ledger_row(out: dict) -> str:
    a16 = out["arms"]["16k"]
    a32 = out["arms"].get("32k")
    rows = f"{'full DEV' if out['rows_mode'] == 'full' else 'DEV subset'} {a16['rows']:,} rows, 0.2.1"
    key = "clef_vs_clef_flash" if "clef_vs_clef_flash" in a16["vs"] else None
    parts = [f"Clef 27B | {rows} | Index {_ci(a16['index']['clef']['index'])}"]
    if a32:
        parts[-1] += f"; 32k {_ci(a32['index']['clef']['index'])}"
    parts.append(f"board ECE {a16['ece']['clef']['board_protocol_ece']:.4f}")
    if key:
        d16 = a16["vs"][key]["diff_index"]
        s = f"vs Clef-flash {d16['point']:+.2f} [{d16['lo']:+.2f}, {d16['hi']:+.2f}] (16k)"
        if a32:
            d32 = a32["vs"][key]["diff_index"]
            s += f", {d32['point']:+.2f} [{d32['lo']:+.2f}, {d32['hi']:+.2f}] (32k vs 32k)"
        parts.append(s)
    parts.append(f"{out['job']}, {out['artifact']}")
    return "| " + " | ".join(parts) + " |"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I analysis/land_r2.py", description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True, help="extracted artifact directory of the Clef index job")
    ap.add_argument("--candidate", help="optional: also pair this candidate vs Clef 27B and Clef-flash")
    ap.add_argument("--flash-16k", default=str(FLASH_16K))
    ap.add_argument("--flash-32k", default=str(FLASH_32K))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job", default="?")
    ap.add_argument("--artifact", default="?")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = land(a.art, a.candidate, a.flash_16k, a.flash_32k, a.B, a.seed, a.job, a.artifact)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if not a.candidate:
        print("LEDGER:", ledger_row(out))
    print("job score check:", json.dumps(out["job_score_check"]))
    for ctx, r in out["arms"].items():
        print(f"\n## {ctx}\n" + P.table(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Land a ``fulldev_eval`` job (a saved cell checkpoint scored on the FULL Index DEV) against both Clef comparators.

    python -I -X utf8 analysis/land_fulldev.py --art <extracted artifact dir> [--job job-x --artifact art-x] \
        [--clef <Clef 27B results.jsonl>] [--clef-flash <Clef-flash results.jsonl>] [--out report.json]
    python -I -X utf8 analysis/land_fulldev.py --candidate <child.results.jsonl> ...

``--art`` is searched for ``fulldev/child.results.jsonl`` (extract the job's ``phases/report.tgz``, or any later
phase, into it first). The comparators default to the two DONE full-DEV references, sha256-pinned:
  * Clef 27B   job-zfexa art-p93d4  index/decide_32k_substituted/results.jsonl  (61.89 [60.28, 63.28])
  * Clef-flash job-t5yg2 art-7pyk2  index/decide_32k_substituted/results.jsonl  (57.29 [55.88, 58.67])
both on the torch reference GDN path, edition 0.2.1, the 51 over-length rows re-read at 32k and substituted.

Two pairings, both through ``analysis.paired_vs_clef`` (frozen scorer + case-cluster bootstrap, B = 2000, seed 0, row
sets asserted identical):
  * ``full``        12,104 rows: the candidate read every row natively (no truncation), Clef at 32k. HEADLINE.
  * ``full_excl51`` 12,053 rows: the 51 rows Clef's shipped 16k cannot read left out on every side.
``beats_both`` (paired CI lo > 0 vs each Clef) is reported for each. The job's own ``child.score.json`` point is
re-checked against the recomputed one (must agree to 1e-6). CPU only; the artifact is untrusted data (python -I).
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
CLEF = HOME / "clef_zfexa_land/x/clef-index-full/index/decide_32k_substituted/results.jsonl"
CLEF_SHA = "f5ad788be191f7737dfadd1594cab7b02a428d847b0cfd9478660df8032ab09e"  # job-zfexa art-p93d4
FLASH = HOME / "clef_t5yg2_land/dl/clef-flash-index-full/index/decide_32k_substituted/results.jsonl"
FLASH_SHA = "2cb2e96114694b589a35c392099b2f67dc5ae0970e9939f8500f0bedf86649c5"  # job-t5yg2 art-7pyk2
MODES = ("full", "full_excl51")


def find_candidate(art: Path) -> Path:
    hits = sorted(art.rglob("fulldev/child.results.jsonl"))
    if len(hits) != 1:
        raise SystemExit(f"{art}: want exactly one fulldev/child.results.jsonl, found {len(hits)}: {hits[:4]}")
    return hits[0]


def _ci(c) -> str:
    return f"{c['point']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"


def land(candidate: str, clef: str = str(CLEF), flash: str = str(FLASH), B: int = 2000, seed: int = 0,
         job: str = "?", artifact: str = "?", check_pins: bool = True) -> dict:
    if check_pins:
        for p, want, name in ((clef, CLEF_SHA, "Clef 27B art-p93d4"), (flash, FLASH_SHA, "Clef-flash art-7pyk2")):
            if P._sha(p) != want:
                raise SystemExit(f"{name}: {p} sha256 != the pinned comparator file")
    out = {"job": job, "artifact": artifact, "candidate": str(Path(candidate).resolve()), "pairings": {}}
    for mode in MODES:
        r = P.run(clef, flash, candidate, rows_mode=mode, B=B, seed=seed)
        out["pairings"][mode] = {"brief": P.brief(r), "table": P.table(r), "rows": r["rows"],
                                 "files": r["files"], "context": r["context"]}
    own = Path(candidate).with_name("child.score.json")
    if own.is_file():
        jp = json.loads(own.read_text(encoding="utf-8"))["slice_index"]["index"]["point"]
        rp = out["pairings"]["full"]["brief"]["index"]["candidate"]["point"]
        out["job_score_check"] = {"job": jp, "recomputed": rp, "ok": abs(jp - rp) < 1e-6}
    f = out["pairings"]["full"]["brief"]
    x = out["pairings"]["full_excl51"]["brief"]
    out["ledger_note"] = (f"full DEV 12,104 rows (native, no truncation): {_ci(f['index']['candidate'])}; "
                          f"vs Clef {_ci(f['diff_vs']['clef'])}, vs Clef-flash {_ci(f['diff_vs']['clef_flash'])}; "
                          f"excl-51 (12,053): {_ci(x['index']['candidate'])}, vs Clef {_ci(x['diff_vs']['clef'])}, "
                          f"vs Clef-flash {_ci(x['diff_vs']['clef_flash'])}; board ECE {f['board_ece']['candidate']:.4f}; "
                          f"beats_both full={f['beats_both']} excl51={x['beats_both']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 analysis/land_fulldev.py", description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--art", help="extracted artifact dir (searched for fulldev/child.results.jsonl)")
    g.add_argument("--candidate", help="the job's child.results.jsonl")
    ap.add_argument("--clef", default=str(CLEF))
    ap.add_argument("--clef-flash", default=str(FLASH))
    ap.add_argument("--no-pin-check", action="store_true", help="other comparator files (recorded, not the DONE refs)")
    ap.add_argument("--job", default="?")
    ap.add_argument("--artifact", default="?")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cand = a.candidate or str(find_candidate(Path(a.art)))
    out = land(cand, a.clef, a.clef_flash, a.B, a.seed, a.job, a.artifact, not a.no_pin_check)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for mode in MODES:
        print(f"## {mode}\n{out['pairings'][mode]['table']}\n")
    print(out["ledger_note"])
    if "job_score_check" in out:
        print("job score check:", out["job_score_check"])
    return 0 if out.get("job_score_check", {}).get("ok", True) else 3


if __name__ == "__main__":
    sys.exit(main())

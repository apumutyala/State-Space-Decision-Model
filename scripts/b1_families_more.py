"""Re-run the B1-families builder (stack/rwkv7/b1-families @a4bc718, ``data.b1_families.build``) with larger synthetic
targets into a NEW directory, unchanged code: same seed, same guards (cohort v3, SplitGuard + content index, Index
fingerprint / family-overlap, D1-A templates, frozen-family refusal, read-back verify). Generators are seeded
sequences, so the first rows of each synthetic family reproduce the shipped 2,400-row build (deduplicated downstream by
id / source item). API-Bank is the official train split (all usable rows, as before).

    python -X utf8 -I scripts/b1_families_more.py --code-root <b1-families checkout @a4bc718> \
        --out <ssdm-data/m2/build/b1-families-x> --bfcl 10000 --cruxeval 12000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bfcl", type=int, default=10000)
    ap.add_argument("--cruxeval", type=int, default=12000)
    ap.add_argument("--api-bank", type=int, default=3000)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.code_root)
    from train.cohort_v3 import _ensure_kit
    _ensure_kit()  # the decision-index kit on sys.path (harness.heldout imports it)
    from data.b1_families import build as B

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    B.OUT_DIR = out
    B.MANIFEST_DIR = Path(a.code_root) / "data" / "manifests" / out.name  # a NEW untracked dir (build.py needs it under the code root)
    B.TARGETS = {"api_bank": a.api_bank, "bfcl": a.bfcl, "cruxeval": a.cruxeval}
    s = B.build(out, log=lambda *x: print(*x, flush=True))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n_records", "sha256_gz", "drop_rate")}
                      for k, v in s["families"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

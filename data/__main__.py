"""python -X utf8 -I -m ssdm_data {fingerprint|fetch|build} ..."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    if not sys.flags.utf8_mode:
        raise SystemExit("run with python -X utf8 -I (UTF-8 mode is required for byte-stable text handling)")
    ap = argparse.ArgumentParser(prog="ssdm_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fingerprint", help="build the contamination fingerprint from the full Index suite")
    f.add_argument("--suite", default=None)
    f.add_argument("--out", default=None)
    g = sub.add_parser("fetch", help="download pinned public train splits into the untrusted data root")
    g.add_argument("sources", nargs="*")
    b = sub.add_parser("build", help="convert + augment + dedup a sample end to end")
    b.add_argument("--n", type=int, default=20000)
    b.add_argument("--out", default=None)
    b.add_argument("--receipt", default=None)
    b.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    from . import config
    if a.cmd == "fingerprint":
        from .dedup import Fingerprint
        fp = Fingerprint.build_from_suite(Path(a.suite or config.SUITE_DIR))
        sha = fp.save(Path(a.out or config.FINGERPRINT))
        print(json.dumps({**fp.meta, "npz_sha256": sha}, indent=1))
    elif a.cmd == "fetch":
        from .sources import fetch_all
        print(json.dumps(fetch_all(a.sources or None), indent=1))
    elif a.cmd == "build":
        from .build import build_sample
        res = build_sample(n_target=a.n, out=a.out, receipt=a.receipt, seed=a.seed or config.SEED)
        print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()

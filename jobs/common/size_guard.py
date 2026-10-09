"""Pre-upload size guard for $GMN_OUTPUT_DIR: the artifact store rejects an output dir over its single-PUT bound
(5,368,709,120 bytes; job-igcu4 lost its Index, G12 and predictions to a 14 GB checkpoint copy). If the dir is over
``--cap-bytes`` (default 4.5 GB, headroom for the tar), the LARGEST non-essential files leave it, largest first, until
it fits; each is moved to ``--evict-dir`` (same filesystem: a rename) and only removed when it cannot be moved there
(a different filesystem). Every eviction is logged in ``<out>/SIZE_GUARD.json`` and on stdout.

Never evicted: result / units / report / score / summary / manifest / marker files, the guard's own log, and the
final phase export (``phases/final.tgz``, else the newest ``phases/*.tgz``), which holds every result the run wrote.
Evictable: the checkpoint export (the full checkpoint stays in $GMN_CHECKPOINT_DIR) and earlier phase tarballs (each
is a subset of the final one).

    python jobs/common/size_guard.py --out "$OUT" --evict-dir "$WORK/evicted" [--cap-bytes 4500000000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

CAP = 4_500_000_000
PROTECT_RE = re.compile(r"result|units|report|score|summary|manifest|_DONE$|SIZE_GUARD", re.I)


def files(out: Path) -> list[tuple[Path, int]]:
    return [(p, p.stat().st_size) for p in out.rglob("*") if p.is_file() and not p.is_symlink()]


def protected(out: Path, all_files: list[tuple[Path, int]]) -> set[Path]:
    keep = {p for p, _ in all_files if PROTECT_RE.search(p.name)}
    tgz = [p for p, _ in all_files if p.parent == out / "phases" and p.name.endswith(".tgz")]
    final = out / "phases" / "final.tgz"
    if final in tgz:
        keep.add(final)
    elif tgz:
        keep.add(max(tgz, key=lambda p: p.stat().st_mtime))
    return keep


def guard(out: str | Path, cap: int = CAP, evict_dir: str | Path | None = None) -> dict:
    out = Path(out)
    all_files = files(out)
    total = sum(s for _, s in all_files)
    rec = {"cap_bytes": cap, "bytes_before": total, "evicted": [], "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if total > cap:
        keep = protected(out, all_files)
        for p, size in sorted(((p, s) for p, s in all_files if p not in keep), key=lambda x: -x[1]):
            if total <= cap:
                break
            rel = p.relative_to(out).as_posix()
            how = "removed"
            if evict_dir is not None:
                dst = Path(evict_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(p, dst)
                    how = f"moved to {dst}"
                except OSError:  # another filesystem: a copy of GBs would outlast the finalize window
                    pass
            if how == "removed":
                p.unlink()
            total -= size
            rec["evicted"].append({"file": rel, "bytes": size, "how": how})
            print(f"size_guard: {rel} ({size} bytes) {how}: output dir over {cap} bytes", flush=True)
    rec["bytes_after"] = total
    rec["fits"] = total <= cap
    if rec["evicted"] or not rec["fits"]:
        (out / "SIZE_GUARD.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if not rec["fits"]:
        print(f"size_guard: STILL {total} bytes > {cap} after evicting every non-essential file", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python jobs/common/size_guard.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-bytes", type=int, default=CAP)
    ap.add_argument("--evict-dir", default=None)
    a = ap.parse_args(argv)
    rec = guard(a.out, a.cap_bytes, a.evict_dir)
    print(json.dumps({k: rec[k] for k in ("bytes_before", "bytes_after", "fits")} | {"evicted": len(rec["evicted"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch one jev3 raw source file with a disk guard (stdlib only; downloaded bytes are untrusted DATA, never executed).

    python -X utf8 -I scripts/jev3_fetch.py --url <url> --out <ssdm-data/raw/jev3/<source>/<file>> [--expect-mb N]
        [--floor-mb 300] [--log <fetch-log.jsonl>]

Before the download: free space on the target drive minus the expected size must stay >= --floor-mb (default 300 MB),
else the source is ABORTED and recorded (status "aborted_disk") in the log. During the download the guard is re-checked
every 8 MB; the partial file is removed (it is our own) if the floor would be crossed. On success the log records url,
bytes, sha256 and UTC time, so the raw file can be deleted after the rows are built and re-fetched by sha.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def free_mb(p: Path) -> float:
    return shutil.disk_usage(str(p.anchor or p)).free / 2**20


def log(path: str | None, rec: dict) -> None:
    rec["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(json.dumps(rec), flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-mb", type=float, default=0.0)
    ap.add_argument("--floor-mb", type=float, default=300.0)
    ap.add_argument("--log")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f0 = free_mb(out)
    if f0 - a.expect_mb < a.floor_mb:
        log(a.log, {"url": a.url, "out": str(out), "status": "aborted_disk", "free_mb": round(f0), "expect_mb": a.expect_mb})
        return 2
    h = hashlib.sha256()
    n = 0
    part = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(a.url, headers={"User-Agent": "ssdm-jev3-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fo:
        while True:
            b = resp.read(1 << 20)
            if not b:
                break
            fo.write(b)
            h.update(b)
            n += len(b)
            if n % (8 << 20) < (1 << 20) and free_mb(out) < a.floor_mb:
                fo.close()
                part.unlink()
                log(a.log, {"url": a.url, "out": str(out), "status": "aborted_disk_midway", "bytes": n})
                return 2
    part.replace(out)
    log(a.log, {"url": a.url, "out": str(out), "status": "ok", "bytes": n, "sha256": h.hexdigest(),
                "free_mb_after": round(free_mb(out))})
    return 0


if __name__ == "__main__":
    sys.exit(main())

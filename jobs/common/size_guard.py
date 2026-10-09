"""Pre-upload size guard for $GMN_OUTPUT_DIR: the artifact store rejects an output dir over its single-PUT bound
(5,368,709,120 bytes; job-igcu4 lost its Index, G12 and predictions to a 14 GB checkpoint copy). If the dir is over
``--cap-bytes`` (default 4.5 GB, headroom for the tar), evictable items leave it until it fits.

ALLOW-LIST (everything else in the dir is never touched), in this order:
  1. superseded phase tarballs ``phases/<name>.tgz``, oldest first: each is a subset of the final one. The final export
     (``phases/final.tgz``, or the newest ``phases/*.tgz`` while there is no final one) is never evicted;
  2. LAST, the checkpoint export ``ckpt/`` as ONE unit (the .safetensors with its sidecar, .sha256 and latest.json, so
     nothing is left naming a missing file). The full checkpoint stays in $GMN_CHECKPOINT_DIR either way.

Never deleted: an item is renamed to ``--evict-dir`` (same filesystem). On another filesystem it is copied, every file
of the copy is re-hashed against the source, and only then is the source removed (a verified move); with
``--rename-only`` (the finalize window, too short to copy GBs) or on any copy / verify error it is LEFT IN PLACE and
the guard fails loudly (exit 3, ``fits: false``). Every eviction is appended to ``<out>/SIZE_GUARD.json`` (merged
across calls; ``ckpt_evicted`` says whether the checkpoint export left) and to ``<out>/MANIFEST.jsonl``
(``{"evicted": ...}`` lines), and printed.

    python jobs/common/size_guard.py --out "$OUT" --evict-dir "$CK/evicted" [--cap-bytes 4500000000] [--rename-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

CAP = 4_500_000_000
LOG = "SIZE_GUARD.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in _files(p))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def evictable(out: Path) -> list[Path]:
    """The allow-list, in eviction order (see the module doc)."""
    tgz = sorted((p for p in (out / "phases").glob("*.tgz") if p.is_file()), key=lambda p: (p.stat().st_mtime, p.name))
    final = out / "phases" / "final.tgz"
    keep = final if final in tgz else (tgz[-1] if tgz else None)
    items = [p for p in tgz if p != keep]
    if (out / "ckpt").is_dir() and _files(out / "ckpt"):
        items.append(out / "ckpt")
    return items


def _free_dst(dst: Path) -> Path:
    """``dst``, or ``dst.1``, ``dst.2`` ... : an eviction never overwrites an earlier one."""
    k, cand = 0, dst
    while cand.exists():
        k += 1
        cand = dst.with_name(f"{dst.name}.{k}")
    return cand


def move(src: Path, dst: Path, rename_only: bool) -> str:
    """Move ``src`` (file or dir) to ``dst``; returns how. Raises OSError, leaving ``src`` in place, if it cannot."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
        return "renamed"
    except OSError as e:
        if rename_only:
            raise OSError(f"not on the evict dir's filesystem and --rename-only ({e})") from e
    want = {f.relative_to(src).as_posix() if src.is_dir() else "": _sha(f) for f in _files(src)}
    try:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        got = {f.relative_to(dst).as_posix() if dst.is_dir() else "": _sha(f) for f in _files(dst)}
    except OSError as e:
        raise OSError(f"copy to {dst} failed ({e}); source left in place") from e
    if got != want:
        raise OSError(f"copy at {dst} does not match the source sha256s; source left in place")
    if src.is_dir():  # the verified copy holds every byte: this completes a move
        shutil.rmtree(src)
    else:
        src.unlink()
    return "copied + sha256-verified, then moved"


def final_reserve(out: Path) -> int:
    """Bytes the final phase export will add (estimate: the newest phase tarball, as each is cumulative); 0 once
    ``phases/final.tgz`` exists."""
    tgz = [p for p in (out / "phases").glob("*.tgz") if p.is_file()]
    if not tgz or (out / "phases" / "final.tgz") in tgz:
        return 0
    return max(tgz, key=lambda p: (p.stat().st_mtime, p.name)).stat().st_size


def guard(out: str | Path, cap: int = CAP, evict_dir: str | Path | None = None, rename_only: bool = False,
          reserve_final: bool = False) -> dict:
    """``reserve_final``: before the final export exists (mid-run, where a copy is affordable), fit under ``cap``
    minus that export's estimated size, so the finalize-window call (rename only) normally has nothing to do."""
    out = Path(out)
    log_p = out / LOG
    prev = json.loads(log_p.read_text(encoding="utf-8")) if log_p.is_file() else {}
    total = _size(out)
    reserve = final_reserve(out) if reserve_final else 0
    cap_bytes, cap = cap, cap - reserve
    rec = {"cap_bytes": cap_bytes, "reserved_for_final_bytes": reserve, "bytes_before": total, "evicted": list(prev.get("evicted", [])), "left_in_place": [],
           "ckpt_evicted": bool(prev.get("ckpt_evicted")), "t": _now()}
    new = []
    if total > cap:
        for item in evictable(out):
            if total <= cap:
                break
            rel, size = item.relative_to(out).as_posix(), _size(item)
            if evict_dir is None:
                rec["left_in_place"].append({"item": rel, "bytes": size, "why": "no --evict-dir"})
                continue
            dst = _free_dst(Path(evict_dir) / rel)
            try:
                how = move(item, dst, rename_only)
            except OSError as e:
                rec["left_in_place"].append({"item": rel, "bytes": size, "why": str(e)})
                print(f"size_guard: COULD NOT EVICT {rel} ({size} bytes): {e}", flush=True)
                continue
            total -= size
            ent = {"item": rel, "bytes": size, "to": str(dst), "how": how, "t": _now()}
            new.append(ent)
            rec["evicted"].append(ent)
            rec["ckpt_evicted"] = rec["ckpt_evicted"] or rel == "ckpt"
            print(f"size_guard: {rel} ({size} bytes) {how} to {dst}: output dir over {cap} bytes", flush=True)
    rec["bytes_after"] = total
    rec["fits"] = total <= cap
    if new or not rec["fits"] or prev:
        log_p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if new:
        with open(out / "MANIFEST.jsonl", "a", encoding="utf-8", newline="\n") as f:
            for ent in new:
                f.write(json.dumps({"evicted": ent["item"], "bytes": ent["bytes"], "to": ent["to"], "t": ent["t"]}) + "\n")
    if not rec["fits"]:
        print(f"size_guard: FAIL: STILL {total} bytes > {cap} after the allow-listed evictions "
              f"({len(rec['left_in_place'])} left in place); the upload will be refused", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python jobs/common/size_guard.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-bytes", type=int, default=CAP)
    ap.add_argument("--evict-dir", default=None)
    ap.add_argument("--rename-only", action="store_true", help="never copy (the finalize window is too short)")
    ap.add_argument("--reserve-final", action="store_true", help="leave room for the final phase export")
    a = ap.parse_args(argv)
    rec = guard(a.out, a.cap_bytes, a.evict_dir, a.rename_only, a.reserve_final)
    print(json.dumps({k: rec[k] for k in ("bytes_before", "bytes_after", "fits", "ckpt_evicted")}
                     | {"evicted": len(rec["evicted"]), "left_in_place": len(rec["left_in_place"])}))
    return 0 if rec["fits"] else 3


if __name__ == "__main__":
    sys.exit(main())

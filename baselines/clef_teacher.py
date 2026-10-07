"""Teacher export: Clef / Clef-flash per-option probabilities on m2 TRAINING records (for later distillation into the
attention-free student; open-weight teachers are allowed by PLAN.md). Never DEV, TEST or held-out rows.

Two phases, so every contamination check runs on the CPU before any GPU minute or weight download:

  python -X utf8 -m baselines.clef_teacher select --scope jobs/clef_baseline/teacher_scope.json \\
      --train-dir /data/m2-150k --guard-readiness $OUT/guard/readiness.json --out $OUT/teacher [--n N]
  python -X utf8 -m baselines.clef_teacher export --model-dir SNAP --release clef-flash --out $OUT/teacher

``select`` (exit 0 ok | 10 the files are not the pointer's build | 20 contamination: any check below fires):
  1. the training set's three files hash (uncompressed sha256) to the committed pointer
     (``data/manifests/sample-150000-seed20261006.pointer.json``, the set ``jobs/m2_first_cell`` trains on);
  2. manifest ids == content-map ids, ``n_records`` ids;
  3. the frozen m2 development cohort (``train.cohort``: dev / calibration / retention, by id AND normalized content)
     holds none of the set's records (the PR #44 build excluded it; a hit means these are not the reviewed files);
  4. the draw: ``train.run_cell.split_ids(sorted(pool), n, 0, seed)`` with pool = set - pinned D1-B exclusions, i.e.
     exactly the m2 first cell's training draw for its (n=24,000, seed=0);
  5. every drawn line matches the manifest sha256, and its content hash equals the content map;
  6. ``train.guard.SplitGuard`` over every drawn record: Index metadata, any id / payload hash in the harness DEV or
     TEST manifest, or (production) a content-hash match against the pinned content index of every DEV + TEST row
     (``jobs.m2_pilot.guard_ready`` readiness file, re-hashed against ``guard_pin.json``);
  7. ``train.heldout.scan`` (held-out D1-B fragments, k=8 templates) over every drawn record's text: zero hits;
     and ``harness.heldout.check_record`` (frozen held-out families of heldout-draw-v1, whose frozen_rule forbids any
     distillation set from targeting them): zero ``FrozenFamilyError``; its file sha256 goes into selection.json;
  8. ``Cohort.assert_training_ids`` / ``assert_training_content`` on the draw.
  It writes ``selection.json`` (identity of every check, drawn ids sha256) and ``selected-ids.txt``.

``export`` re-reads the records, keeps exactly the selected ids (content re-hashed against ``selection.json``), and runs
each through the same ``decide()`` path as the Index readout (``ClefBackend``, ssdm encoding, no truncation). One JSON
line per record, appended and flushed (resumable: a partial last line is cut, finished ids are skipped). Records over
Clef's 16,384-token input are ``exhausted`` (no probabilities; Clef would only see a cut state). Labels are never read
into the request.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXIT_IDENTITY = 10
EXIT_CONTAMINATION = 20
EXIT_EXPORT = 19


class Contamination(RuntimeError):
    pass


class IdentityMismatch(RuntimeError):
    pass


def _usha(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_scope(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_scope_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


def read_pointer(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("records_uncompressed_sha256", "manifest", "manifest_uncompressed_sha256", "content_map",
              "content_map_uncompressed_sha256", "n_records"):
        if k not in d:
            raise IdentityMismatch(f"pointer {rel} lacks {k}")
    d["_pointer_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


def train_files(train_dir: Path, ptr: dict) -> dict[str, Path]:
    return {"records": train_dir / "records.jsonl.gz", "manifest": train_dir / ptr["manifest"],
            "content_map": train_dir / ptr["content_map"]}


def verify_files(files: dict[str, Path], ptr: dict) -> dict[str, str]:
    want = {"records": ptr["records_uncompressed_sha256"], "manifest": ptr["manifest_uncompressed_sha256"],
            "content_map": ptr["content_map_uncompressed_sha256"]}
    got = {}
    for k, p in files.items():
        if not p.exists():
            raise IdentityMismatch(f"training-set {k} missing: {p}")
        got[k] = _usha(p)
        if got[k] != want[k]:
            raise IdentityMismatch(f"training-set {k} uncompressed sha256 {got[k]} != pointer {want[k]}")
    return got


def draw_ids(manifest_ids, excluded, n: int, seed: int) -> list[str]:
    from train.run_cell import split_ids

    pool = sorted(set(manifest_ids) - set(excluded))
    if n > len(pool):
        raise IdentityMismatch(f"n {n} > pool {len(pool)}")
    return split_ids(pool, n, 0, seed)[0]


def load_guard(readiness: str | None, standin: bool):
    """Production: the pinned content-index guard (``guard_ready`` readiness file). ``standin`` (CPU dry runs only,
    ``CLEF_DRY_RUN=1``): the harness DEV + TEST manifests by id / payload hash, without the content index."""
    if readiness:
        from jobs.m2_pilot.guard_ready import load_ready_guard

        return load_ready_guard(readiness)
    if standin and os.environ.get("CLEF_DRY_RUN") == "1":
        from train.guard import SplitGuard

        m = ROOT / "harness" / "manifests"
        g = SplitGuard.from_manifests([m / "dev-manifest.jsonl.gz", m / "test-manifest.jsonl.gz"])
        g.note = "STANDIN (dry run): manifests only, no content index"
        return g
    raise Contamination("no guard: pass --guard-readiness (production) or --guard-standin with CLEF_DRY_RUN=1")


def stream_records(path: Path, want: set[str], manifest: dict[str, str]):
    """Yield drawn records; every line read is checked against the manifest sha256 (as train.data.load_m2_records)."""
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            n += 1
            rec = json.loads(line)
            if rec.get("id") not in want:
                continue
            h = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if manifest.get(rec["id"]) != h:
                raise IdentityMismatch(f"{rec['id']}: line sha256 {h[:12]} != manifest")
            yield rec
    stream_records.lines_read = n  # type: ignore[attr-defined]


def select(a) -> int:
    from train.cohort import Cohort, read_content_map, record_content_hash
    from train.data import read_manifest
    from train.guard import ForbiddenRow
    from harness import config as hconfig
    from harness.heldout import FrozenFamilyError, check_record, frozen as frozen_families
    from train.heldout import record_text, scan

    scope = read_scope(a.scope)
    ptr = read_pointer(scope["pointer"])
    files = train_files(Path(a.train_dir), ptr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sha = verify_files(files, ptr) if not a.skip_file_hash else {"note": "not re-hashed (--skip-file-hash)"}
    man = read_manifest(files["manifest"])
    content = read_content_map(files["content_map"])
    if set(man) != set(content):
        raise IdentityMismatch("content-map ids != manifest ids")
    if not a.allow_n_mismatch and len(man) != int(ptr["n_records"]):
        raise IdentityMismatch(f"manifest has {len(man)} ids, pointer says {ptr['n_records']}")
    co = Cohort(expected_sha256=scope["cohort_sha256"])
    held, held_c = co.held_out(), co.held_out_content()
    by_id = sorted(i for i in man if i in held)
    by_content = sorted(i for i in man if content[i] in held_c)
    if by_id or by_content:
        raise Contamination(f"cohort held-out records in the set: {len(by_id)} by id, {len(by_content)} by content, "
                            f"e.g. {(by_id + by_content)[:3]}")
    excl = scope["heldout_excluded"]["ids"]
    if sorted(set(excl) - set(man)) and not a.allow_n_mismatch:
        raise IdentityMismatch(f"pinned exclusions not in the set: {sorted(set(excl) - set(man))[:3]}")
    n = int(a.n if a.n is not None else scope["draw"]["n_records"])
    ids = draw_ids(man, excl, n, int(scope["draw"]["seed"]))
    want = set(ids)
    guard = load_guard(a.guard_readiness, a.guard_standin)
    forbidden, frozen, drift, texts, got = [], [], [], [], set()
    for rec in stream_records(files["records"], want, man):
        got.add(rec["id"])
        ch = record_content_hash(rec)
        if ch != content[rec["id"]]:
            drift.append(rec["id"])
            continue
        try:
            guard.check(rec)
        except ForbiddenRow as e:
            forbidden.append(str(e))
        try:  # frozen held-out families (harness heldout-draw-v1): no distillation set may target them
            check_record(rec)
        except FrozenFamilyError as e:
            frozen.append(str(e))
        texts.append((rec["id"], record_text(rec)))
    missing = sorted(want - got)
    if missing or drift:
        raise IdentityMismatch(f"{len(missing)} drawn ids missing from the records file, {len(drift)} drifted "
                               f"from the content map (e.g. {(missing + drift)[:3]})")
    rep = scan(texts)
    if forbidden or frozen or rep["hits"]:
        raise Contamination(f"{len(forbidden)} drawn record(s) refused by the DEV/TEST guard ({forbidden[:2]}), "
                            f"{len(frozen)} target a frozen held-out family ({frozen[:2]}), "
                            f"{len(rep['hits'])} carry a held-out D1-B fragment ({rep['hits'][:2]})")
    co.assert_training_ids(ids)
    co.assert_training_content(content[i] for i in ids)
    sel = {"ok": True, "scope": scope["_scope_sha256"], "scope_path": str(a.scope), "pointer": scope["pointer"],
           "pointer_sha256": ptr["_pointer_sha256"], "files_uncompressed_sha256": sha,
           "lines_manifest_verified": getattr(stream_records, "lines_read", None), "n_set": len(man),
           "n_drawn": len(ids), "draw_seed": scope["draw"]["seed"],
           "drawn_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
           "heldout_excluded": sorted(excl), "cohort": co.identity(), "cohort_hits": {"by_id": 0, "by_content": 0},
           "guard": guard.identity(), "guard_refused": 0, "frozen_family_refused": 0,
           "heldout_draw_v1_sha256": hconfig.HELDOUT_DRAW_V1_SHA256,
           "frozen_families": sorted(frozen_families()),
           "heldout_scan": {"n_texts": rep["n_texts"], "hits": 0, "templates_sha256": rep["templates_sha256"]},
           "content_sha256": {i: content[i] for i in ids}, "secs": round(time.time() - t0, 1)}
    (out / "selected-ids.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8", newline="\n")
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in sel.items() if k not in ("content_sha256", "cohort", "guard")}))
    return 0


def export(a) -> int:
    from harness.jsonl import drop_truncated_tail, read_jsonl_strict
    from train.cohort import record_content_hash
    from train.data import read_manifest

    from . import clef_backend as CB

    out = Path(a.out)
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if not sel.get("ok"):
        raise Contamination("selection.json is not ok")
    scope = read_scope(sel["scope_path"])
    if scope["_scope_sha256"] != sel["scope"]:
        raise IdentityMismatch("scope changed since select")
    ptr = read_pointer(scope["pointer"])
    files = train_files(Path(a.train_dir), ptr)
    ids = [x for x in (out / "selected-ids.txt").read_text(encoding="utf-8").split("\n") if x]
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != sel["drawn_ids_sha256"]:
        raise IdentityMismatch("selected-ids.txt != selection.json")
    res = out / f"teacher-{a.release}.jsonl"
    drop_truncated_tail(res)
    done = {r["id"] for r in read_jsonl_strict(res, )} if res.exists() else set()
    backend, _, _ = CB.load_clef(a.model_dir, a.release, device=a.device, dtype=a.dtype,
                                 verify=not a.no_verify)
    decider = CB.build_decider(backend)
    man = read_manifest(files["manifest"])
    want = set(ids) - done
    t0, n_new, status = time.time(), 0, {}
    with res.open("a", encoding="utf-8", newline="\n") as fh:
        for rec in stream_records(files["records"], want, man):
            ch = record_content_hash(rec)
            if ch != sel["content_sha256"][rec["id"]]:
                raise IdentityMismatch(f"{rec['id']}: content changed since select")
            from train.guard import record_request

            req = record_request(rec)
            request = {"model": a.release, "state": req["state"], "questions": req["questions"]}
            t1 = time.perf_counter()
            resp = decider.decide(request)
            ss = resp["ssdm"]
            row = {"id": rec["id"], "content_sha256": ch, "status": ss["status"],
                   "raw_probabilities": ss.get("raw_probabilities") if ss["status"] == "ok" else None,
                   "error": None if ss["status"] == "ok" else ss.get("error"),
                   "sequence_tokens": (resp.get("usage") or {}).get("input_tokens"),
                   "wall_ms": round((time.perf_counter() - t1) * 1000.0, 3)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            status[ss["status"]] = status.get(ss["status"], 0) + 1
    rows = read_jsonl_strict(res)
    got = {r["id"] for r in rows}
    if got != set(ids) or len(rows) != len(ids):
        raise IdentityMismatch(f"teacher file has {len(rows)} rows / {len(got)} ids, selection has {len(ids)}")
    ident = {"release": a.release, "repo": CB.RELEASES[a.release]["repo"],
             "revision": CB.RELEASES[a.release]["revision"], "backend_revision": backend.revision()["revision_id"],
             "standin": bool(os.environ.get("CLEF_STANDIN")), "selection_drawn_ids_sha256": sel["drawn_ids_sha256"],
             "rows": len(rows), "status_counts": _count(r["status"] for r in rows), "new_this_attempt": n_new,
             "secs_this_attempt": round(time.time() - t0, 1),
             "teacher_sha256": hashlib.sha256(res.read_bytes()).hexdigest()}
    (out / f"teacher-{a.release}.identity.json").write_text(json.dumps(ident, indent=1), encoding="utf-8",
                                                           newline="\n")
    print(json.dumps(ident))
    return 0


def _count(xs):
    c: dict[str, int] = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m baselines.clef_teacher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--scope", default="jobs/clef_baseline/teacher_scope.json")
    s.add_argument("--train-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=None, help="override the scope's draw size (priced in JOB_SPEC)")
    s.add_argument("--guard-readiness", default=None)
    s.add_argument("--guard-standin", action="store_true", help="dry runs only (CLEF_DRY_RUN=1)")
    s.add_argument("--skip-file-hash", action="store_true", help="dry runs only (CLEF_DRY_RUN=1)")
    s.add_argument("--allow-n-mismatch", action="store_true", help="dry runs only (CLEF_DRY_RUN=1): stand-in set")
    e = sub.add_parser("export")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--release", choices=["clef-flash", "clef"], required=True)
    e.add_argument("--train-dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    dry_only = a.cmd == "select" and (a.skip_file_hash or a.allow_n_mismatch or a.guard_standin)
    if dry_only and os.environ.get("CLEF_DRY_RUN") != "1":
        print("--guard-standin / --skip-file-hash / --allow-n-mismatch are for CLEF_DRY_RUN=1 only", file=sys.stderr)
        return 2
    try:
        return select(a) if a.cmd == "select" else export(a)
    except Contamination as ex:
        print(f"clef_teacher {a.cmd}: CONTAMINATION GUARD: {ex}", file=sys.stderr)
        return EXIT_CONTAMINATION
    except IdentityMismatch as ex:
        print(f"clef_teacher {a.cmd}: identity: {ex}", file=sys.stderr)
        return EXIT_IDENTITY


if __name__ == "__main__":
    sys.exit(main())

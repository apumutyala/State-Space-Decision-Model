"""BOARD B1 exclusion check: every output row vs TEST + DEV + held-out Index rows + cohort v2/v3, independent of the
builder's own filter. Fails loudly (exit 1) on any item-level overlap.

Checks, per row of ``<draw dir>/records.jsonl.gz`` (lines verified against the manifest and the pointer):
  1. cohort v2 (all held-out roles) and v3 (sha 498efa61...): record id, source item (``~view:`` / ``~inj:`` stripped,
     multi / fact-change keys), ``train.cohort.record_content_hash``, ``record_text_keys`` (key-free text);
  2. the Index rows themselves (0.2.1 suite selected + added rows = every DEV and TEST row of all 44 catalogs; the
     DEV slice ``slices/dev-rows.jsonl.gz``; the 0.3 GSM8K DEV rows): normalized content hash, key-free text keys,
     and text segments: each ITEM string of a row (state / instructions / option text outside tool catalogs and
     background passages) is normalized and looked up among every Index line (newline split, >= 8 chars) and every
     Index sentence segment (>= 30 chars); hits in context fields (tool schemas, background passages, retrieval
     candidates) are reported, as the 150k receipt does, but do not fail;
  3. ``train.guard.SplitGuard`` (harness DEV/TEST manifests + pinned content index), ``harness.heldout.check_record``
     (FROZEN held-out families), ``train.heldout`` D1-B fragments (must be 0: the D4 pin is empty);
  4. within the draw: ids, source items (a fact-change pair shares one), content hashes and key-free text unique.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_KEYS = {"tools", "catalog", "available_tools", "background_reading", "passages", "documents"}
CONTEXT_SOURCES_OPTIONS = {"nq"}  # retrieval candidates are passages (context tier in SOURCES.md)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(re.findall(r"\w+", s))


def h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def strings(o: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(o, str):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, f"{path}/{k}")
    elif isinstance(o, list):
        for v in o:
            yield from strings(v, path + "/[]")


def segments(text: str) -> Iterator[str]:
    for line in text.split("\n"):
        for seg in re.split(r"(?<=[.?!])\s+", line):
            n = norm(seg)
            if len(n) >= 30:
                yield n


def index_rows(data: Path) -> Iterator[dict]:
    for p in (data / "m2" / "suite-snapshot" / "selected-rows.jsonl.gz", data / "m2" / "suite-snapshot" / "added-rows.jsonl.gz",
              data / "slices" / "dev-rows.jsonl.gz", data / "slices" / "dev-rows-0.3-gsm8k.jsonl.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r.pop("_evaluation", None)
                    yield r


def build_index(data: Path) -> dict[str, Any]:
    from train.cohort import record_content_hash, record_text_keys

    lines, segs, content, text = set(), set(), set(), set()
    n = 0
    for r in index_rows(data):
        n += 1
        st, qs = r.get("state"), r.get("questions") or {}
        try:
            content.add(record_content_hash({"state": st, "questions": qs}))
        except Exception:
            pass
        try:
            text |= record_text_keys({"state": st, "questions": qs})
        except Exception:
            pass
        for _, s in strings({"s": st, "q": qs}):
            for ln in s.split("\n"):
                nl = norm(ln)
                if len(nl) >= 8:
                    lines.add(h(nl))
            for sg in segments(s):
                segs.add(h(sg))
    return {"n_rows": n, "lines": lines, "segs": segs, "content": content, "text": text}


def item_fields(rec: dict) -> Iterator[tuple[str, str, bool]]:
    """(path, string, is_context) of a training row's model-facing strings."""
    for path, s in strings(rec["state"], "state"):
        top = path.split("/")[1] if path.count("/") >= 1 else ""
        ctx = top in CONTEXT_KEYS or any(f"/{k}/" in path + "/" for k in CONTEXT_KEYS)
        yield path, s, ctx
    for qid, q in rec["questions"].items():
        if isinstance(q.get("instructions"), str):
            yield f"q/{qid}/instructions", q["instructions"], False
        crit = q.get("criteria")
        vals = crit.values() if isinstance(crit, dict) else (crit or [])
        for v in vals:
            if isinstance(v, str):
                yield f"q/{qid}/option", v, rec["source"] in CONTEXT_SOURCES_OPTIONS


def check(draw_dir: Path, pointer: Path, data: Path, out: Path | None = None) -> dict:
    from data.build import item_keys
    from train.cohort import Cohort, record_content_hash, record_text_keys
    from train.cohort_v3 import _ensure_kit, cohort_v3
    from train.heldout import record_text, scan
    from train.train_set import read_pointer, verify_files
    from data.build import content_guard
    from train.guard import ForbiddenRow

    ptr = read_pointer(pointer)
    verify_files(draw_dir, ptr)
    man = {}
    with gzip.open(draw_dir / ptr["manifest"], "rt", encoding="utf-8") as fh:
        for x in fh:
            e = json.loads(x)
            man[e["id"]] = e
    _ensure_kit()
    from harness.heldout import check_record

    v2, v3 = Cohort(), cohort_v3()
    held_ids = v2.held_out() | v3.held_out()
    held_c = v2.held_out_content() | v3.held_out_content()
    held_items = set().union(*(item_keys(i) for i in held_ids)) | v3.held_out_items()
    held_t = set(v3.text_keys or ())
    idx = build_index(data)
    guard, _ = content_guard()
    fails: dict[str, list] = collections.defaultdict(list)
    ctx_hits = collections.Counter()
    seen_id, seen_item, seen_c, seen_t = set(), collections.Counter(), {}, {}
    texts = []
    n = 0
    with gzip.open(draw_dir / "records.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            n += 1
            r = json.loads(line)
            rid = r["id"]
            if man.get(rid, {}).get("sha256") != hashlib.sha256(line.encode("utf-8")).hexdigest():
                fails["manifest_line"].append(rid)
            if not man.get(rid, {}).get("source") or not man.get(rid, {}).get("family"):
                fails["manifest_source_family"].append(rid)
            base = r.get("aug", {}).get("b1_view", {}).get("parent_id", rid)
            items = item_keys(base) | item_keys(rid.split("~view:", 1)[0])
            ch, tk = record_content_hash(r), record_text_keys(r)
            if rid in held_ids or base in held_ids:
                fails["cohort_id"].append(rid)
            if items & held_items:
                fails["cohort_item"].append(rid)
            if ch in held_c:
                fails["cohort_content"].append(rid)
            if tk & held_t:
                fails["cohort_text"].append(rid)
            if ch in idx["content"]:
                fails["index_content"].append(rid)
            if tk & idx["text"]:
                fails["index_text"].append(rid)
            for path, s, ctx in item_fields(r):
                hit = None
                ns = norm(s)
                if len(ns) >= 8 and h(ns) in idx["lines"]:
                    hit = "line"
                elif any(h(sg) in idx["segs"] for sg in segments(s)):
                    hit = "segment"
                if hit:
                    if ctx:
                        ctx_hits[f"{r['source']}:{hit}"] += 1
                    else:
                        fails[f"index_item_{hit}"].append((rid, path, s[:80]))
            try:
                guard.check(r)
            except ForbiddenRow:
                fails["splitguard"].append(rid)
            try:
                check_record(r)
            except Exception:
                fails["frozen_family"].append(rid)
            if rid in seen_id:
                fails["dup_id"].append(rid)
            seen_id.add(rid)
            for it in items:
                seen_item[it] += 1
            if ch in seen_c:
                fails["dup_content"].append((rid, seen_c[ch]))
            seen_c[ch] = rid
            for k in tk:
                if k in seen_t:
                    fails["dup_text"].append((rid, seen_t[k]))
                seen_t[k] = rid
            texts.append((rid, record_text(r)))
    multi_item = [k for k, v in seen_item.items() if v > (2 if k.startswith("m2:factchange:") else 1)]
    if multi_item:
        fails["dup_source_item"] = multi_item
    d1b = scan(texts)
    if d1b["hits"]:
        fails["d1b_heldout"] = d1b["hits"]
    if n != ptr["n_records"] or n != len(man):
        fails["count"].append({"lines": n, "pointer": ptr["n_records"], "manifest": len(man)})
    rep = {"draw": str(draw_dir), "pointer": str(pointer), "n_rows": n,
           "index_rows_indexed": idx["n_rows"], "index_lines": len(idx["lines"]), "index_segments": len(idx["segs"]),
           "cohorts": {"v2": v2.identity(), "v3": v3.identity()},
           "d1b_templates_sha256": d1b["templates_sha256"], "context_overlap_reported_not_failed": dict(ctx_hits),
           "failures": {k: {"n": len(v), "examples": v[:5]} for k, v in fails.items()},
           "pass": not fails}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="data.b1_check")
    ap.add_argument("--draw-dir", required=True)
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--data", default=r"C:\Users\apumu\research\CHLabs\ssdm-data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = check(Path(a.draw_dir), Path(a.pointer), Path(a.data), Path(a.out) if a.out else None)
    print(json.dumps({k: rep[k] for k in ("n_rows", "pass", "failures", "context_overlap_reported_not_failed")},
                     indent=1, ensure_ascii=False))
    if not rep["pass"]:
        print("B1 EXCLUSION CHECK FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

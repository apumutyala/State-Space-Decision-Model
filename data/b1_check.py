"""BOARD B1 exclusion check: every output row vs TEST + DEV + held-out Index rows + cohort v2/v3, independent of the
builder's own filter. Fails loudly (exit 1) on any item-level overlap.

Checks, per row of ``<draw dir>/records.jsonl.gz`` (lines verified against the manifest and the pointer):
  1. cohort v2 (all held-out roles) and v3 (sha 498efa61...): record id, source item (``~view:`` / ``~inj:`` stripped,
     multi / fact-change keys), ``train.cohort.record_content_hash``, ``record_text_keys`` (key-free text);
  2. the Index rows themselves (0.2.1 suite selected + added rows = every DEV and TEST row of all 44 catalogs; the
     DEV slice ``slices/dev-rows.jsonl.gz``; the 0.3 GSM8K DEV rows): normalized content hash, key-free text keys,
     and text segments: each ITEM string of a row (state / instructions outside tool catalogs and background
     passages; our own instruction templates skipped) is normalized and every line of it (>= 25 chars and >= 6 words)
     looked up among the Index lines, every sentence segment (>= 30 chars, >= 6 words, option texts included) among
     the Index segments, and program text (``state.code``) as a whole function; lines / segments shared by >= 20
     distinct Index rows are boilerplate (instruction templates, label sets) and are not item text; hits in context fields (tool schemas, background passages, retrieval
     candidates) are reported, as the 150k receipt does, but do not fail;
  3. ``train.guard.SplitGuard`` (harness DEV/TEST manifests + pinned content index), ``harness.heldout.check_record``
     per row and ``check_generator`` per (source, family) (FROZEN held-out families, heldout-draw-v1),
     ``Cohort.assert_draw_disjoint`` of v2 and v3 over all ids, ``train.heldout`` D1-B fragments (must be 0);
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
BOILERPLATE_ROWS = 20


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


def is_item_text(n: str) -> bool:
    """A normalized line / sentence long enough to identify an item (>= 25 chars and >= 6 words): shorter strings
    are entity values, labels or tool names ("ibuprofen", "candidate tool book hotel")."""
    return len(n) >= 25 and n.count(" ") >= 5


def segments(text: str) -> Iterator[str]:
    for line in text.split("\n"):
        for seg in re.split(r"(?<=[.?!])\s+", line):
            n = norm(seg)
            if len(n) >= 30 and n.count(" ") >= 5:
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

    lines, segs, content, text = collections.Counter(), collections.Counter(), set(), set()
    full = set()  # whole normalized strings (code is matched as a whole function, not line by line)
    n = 0
    ids = set()
    for r in index_rows(data):
        if r.get("id") in ids:  # the DEV slice repeats suite rows
            continue
        ids.add(r.get("id"))
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
        row_lines, row_segs = set(), set()
        for _, s in strings({"s": st, "q": qs}):
            ns = norm(s)
            if len(ns) >= 25:
                full.add(h(ns))
            for ln in s.split("\n"):
                nl = norm(ln)
                if is_item_text(nl):
                    row_lines.add(h(nl))
            for sg in segments(s):
                row_segs.add(h(sg))
        lines.update(row_lines)
        segs.update(row_segs)
    # Boilerplate = a line / segment shared by >= BOILERPLATE_ROWS distinct Index rows (an Index instruction template
    # or a shared label/tool text); an ITEM text recurs in at most a few rows (e.g. GSM8K-4 / GSM8K-10 twins).
    item_lines = {k for k, c in lines.items() if c < BOILERPLATE_ROWS}
    item_segs = {k for k, c in segs.items() if c < BOILERPLATE_ROWS}
    return {"n_rows": n, "lines": item_lines, "segs": item_segs, "full": full, "content": content, "text": text,
            "boilerplate": {"rule": f"shared by >= {BOILERPLATE_ROWS} distinct Index rows",
                            "lines": len(lines) - len(item_lines), "segments": len(segs) - len(item_segs)}}


def item_fields(rec: dict) -> Iterator[tuple[str, str, bool]]:
    """(path, string, is_context) of a training row's model-facing strings."""
    from data.b1 import tool_field

    st = rec["state"]
    tf = tool_field(st)  # the published tool catalog, whatever synonym the renderer gave its key
    for path, s in strings(st, "state"):
        top = path.split("/")[1] if path.count("/") >= 1 else ""
        ctx = top in CONTEXT_KEYS or top == tf
        yield path, s, ctx
    for qid, q in rec["questions"].items():
        if isinstance(q.get("instructions"), str):
            yield f"q/{qid}/instructions", q["instructions"], False
        crit = q.get("criteria")
        vals = crit.values() if isinstance(crit, dict) else (crit or [])
        for v in vals:
            if isinstance(v, str):
                yield f"q/{qid}/option", v, rec["source"] in CONTEXT_SOURCES_OPTIONS


def instruction_templates() -> set[str]:
    from data.augment import INSTR
    return {norm(t) for v in INSTR.values() for t in v}  # our own instruction templates (not Index text)


def item_hits(rec: dict, idx: dict, templates: set[str]) -> list[tuple[str, str, str, bool]]:
    """(hit kind, path, string, is_context) for every model-facing string of ``rec`` found in the Index text."""
    out = []
    for path, s, ctx in item_fields(rec):
        hit = None
        if path.endswith("/code"):  # program text: an item only as a whole function (lines recur across programs)
            ns = norm(s)
            if len(ns) >= 25 and h(ns) in idx["full"]:
                hit = "code_exact"
            if hit:
                out.append((hit, path, s, ctx))
            continue
        if not path.endswith("/option"):  # option texts: label sets are shared by design; segments only
            for ln in s.split("\n"):
                nl = norm(ln)
                if is_item_text(nl) and nl not in templates and h(nl) in idx["lines"]:
                    hit = "line"
                    break
        if hit is None and any(h(sg) in idx["segs"] for sg in segments(s) if sg not in templates):
            hit = "segment"
        if hit:
            out.append((hit, path, s, ctx))
    return out


def check(draw_dir: Path, pointer: Path, data: Path, out: Path | None = None) -> dict:
    from data.build import item_keys
    from train.cohort import Cohort, record_content_hash, record_text_keys
    from train.cohort_v3 import _ensure_kit, cohort_v3
    from train.heldout import record_text, scan
    from train.train_set import read_pointer, verify_files
    from data.build import content_guard
    from data.b1 import dedup_keys
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
    templates = instruction_templates()
    guard, _ = content_guard()
    fails: dict[str, list] = collections.defaultdict(list)
    ctx_hits = collections.Counter()
    seen_id, seen_item, seen_c, seen_t = set(), collections.Counter(), {}, {}
    seen_c_by_id: dict[str, str] = {}
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
            for hit, path, s, ctx in item_hits(r, idx, templates):
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
            seen_c_by_id[rid] = ch
            for k in dedup_keys(r):
                if k in seen_t:
                    fails["dup_text"].append((rid, seen_t[k]))
                seen_t[k] = rid
            texts.append((rid, record_text(r)))
    # harness.heldout.check_generator over every (source, family) generator of the draw, and the cohorts' own
    # assert_draw_disjoint over all ids (by id, content and, for v3, source item)
    from harness.heldout import check_generator
    gens = sorted({(man[i]["source"], man[i]["family"]) for i in man})
    for src, fam in gens:
        try:
            check_generator({"name": src, "task_family": fam, "targets": [src, fam]})
        except Exception as e:
            fails["frozen_generator"].append(f"{src}/{fam}: {e}")
    content_map = dict(seen_c_by_id)
    for name, co in (("v2", v2), ("v3", v3)):
        try:
            co.assert_draw_disjoint(list(content_map), content_map)
        except Exception as e:
            fails[f"assert_draw_disjoint_{name}"].append(str(e)[:300])
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
           "index_boilerplate_excluded": idx["boilerplate"],
           "cohorts": {"v2": v2.identity(), "v3": v3.identity()},
           "d1b_templates_sha256": d1b["templates_sha256"], "generators_checked": [f"{a}/{b}" for a, b in gens], "context_overlap_reported_not_failed": dict(ctx_hits),
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

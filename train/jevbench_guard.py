"""JevBench public items as an EVAL-ONLY source for the training-data gates: their own hash index, fingerprint and pin.

JevBench (github.com/fstandhartinger/jevbench, MIT) is a primary evaluation target, so its 231 public items
(``jevkit.items.load_public``: original 72 / easy 48 / hard 111, sha256-pinned in jevkit/data/jevbench_public_pin.json)
must never reach a training row. Two gates cover them, both from the SAME pinned bytes:

* ``doc_index``  a ``train.doc_overlap_guard`` eval hash index over the items' document units (one unit per top-level
                 state key, all instructions, all criteria / option descriptions), built with the guard's own
                 ``build_index`` under its RULE (13-grams, R / E / U, boilerplate and template exclusions per draw).
                 ``doc_overlap_guard.scan`` runs it beside the Index eval index on EVERY scan (train_set.draw,
                 run_cell, heldout_gate, clef_teacher.select, the CLI), in the same streaming pass. It also checks
                 every training segment against a pinned EXACT segment set (item segments of >= 6 words, templates
                 excluded), so a copied short item (< 17 tokens, invisible to 13-grams) fails too. 3 of the 231
                 states are under 6 words; the builder fingerprint covers those.
* ``fingerprint`` a ``data.dedup.Fingerprint`` (exact / ngram13 / minhash / edit / contain, the short-item matchers of
                 FP v3) over the same items, applied as extra item-tier matchers by the record builder
                 (``data.build.build_sample``): a row whose item text hits any of them is dropped at build time.

Templates: an instruction / option segment shared by >= 2 items is a rubric frame or a tool-catalog description
(the 100k-v2 rescan: 1,049 glaive_fc rows matched "send an email to a recipient" and two other generic tool
descriptions of the easy tool_selection items). Neither the exact gate nor the fingerprint indexes them; state text
is never a template, and the 13-gram rule handles templates per draw as the Index gate does.

It is a SEPARATE index on purpose: the Index eval index, its pin and every pinned ``doc_clean`` spec stay
byte-identical (a JevBench scan with no fail adds only a ``jevbench`` block to a scan report). The arrays are hashes
only (no eval text) and are committed under ``data/manifests/doc_overlap/jevbench/`` (~3.3 MB), so every job box has
them with the code: no context.data entry, no ssdm-data dependency at train time. The build is deterministic (sorted
arrays, content sha256 over the arrays); ``python -X utf8 -m train.jevbench_guard verify`` rebuilds from the pinned
public files and must reproduce the pin; ``build --write-pin`` rewrites it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "manifests" / "doc_overlap" / "jevbench"
PIN_PATH = DIR / "pin.json"
INDEX_DIR = DIR / "doc-index"
FP_PATH = DIR / "fingerprint.npz"
EXACT_PATH = DIR / "exact-segments.npy"
RULE_ID = ("jevbench_guard/v2: JevBench public items (pinned) as an eval-only doc-overlap source + fingerprint; "
           "instruction / option segments shared by >= 2 items are templates (not indexed by the fingerprint, never "
           "an exact hit); the scan's exact-segment gate uses item segments of >= 6 words")
#: an instruction / option segment in >= this many JevBench items is a rubric / tool-catalog template, not an item
TEMPLATE_ITEMS = 2
#: the scan's exact-segment gate: only item segments with at least this many words (a 2-word code line such as
#: "return result" or a short generic phrase is not an item copy; 13-gram rules cover longer text)
EXACT_MIN_WORDS = 6
#: catalog id the fingerprint reports for a JevBench hit (Index catalogs are >= 0)
CATALOG_ID = -77
SPLIT = "JEVBENCH"


class JevBenchGuardError(RuntimeError):
    pass


def _source_pin() -> dict:
    from jevkit.items import read_pin

    p = read_pin()
    return {"repo": p["repo"], "commit": p["commit"], "files": {k: v["sha256"] for k, v in p["files"].items()}}


def source_rows(data_dir: str | Path | None = None) -> list[tuple[str, str, str, dict]]:
    """(split, run_id, family, row) for every pinned public item, sha256-verified (jevkit.items.load_public)."""
    from jevkit.items import load_public, overlap_guard_rows

    return list(overlap_guard_rows(load_public(data_dir)))


def _question_leaves(questions: dict) -> list[str]:
    from data import text as T

    return T.raw_segments(None, questions or {})


def template_segments(rows) -> set[str]:
    """Normalised instruction / option segments that occur in >= TEMPLATE_ITEMS items (rubric frames shared by a
    family, tool-catalog descriptions such as "send an email to a recipient"). State text is never a template."""
    import collections

    from data import text as T

    cnt: collections.Counter = collections.Counter()
    for r in rows:
        cnt.update(T.segments(None, r[3]["questions"]))
    return {seg for seg, c in cnt.items() if c >= TEMPLATE_ITEMS}


def fingerprint_rows(rows, templates: set[str]) -> list[dict]:
    """Each item as the fingerprint sees it: its full state, plus only its item-specific question leaves."""
    from data import text as T

    out = []
    for r in rows:
        leaves = [x for x in _question_leaves(r[3]["questions"]) if T.normalize(x) not in templates]
        out.append({"state": r[3]["state"], "questions": {"q": {"type": "choice", "instructions": leaves}},
                    "_evaluation": {"catalog_id": CATALOG_ID}})
    return out


def exact_segment_hashes(frows: list[dict], templates: set[str]):
    """Sorted unique h64 of every item segment with >= EXACT_MIN_WORDS words (templates excluded)."""
    import numpy as np

    from data import text as T

    hs = set()
    for r in frows:
        for seg in T.segments(r["state"], r["questions"]):
            if seg not in templates and len(seg.split()) >= EXACT_MIN_WORDS:
                hs.add(T.h64(seg))
    return np.array(sorted(hs), dtype=np.uint64)


def build(data_dir: str | Path | None = None):
    """-> (EvalIndex, Fingerprint), deterministic functions of the pinned bytes. ``fp.exact_hashes`` is the scan's
    exact-segment set."""
    from data.dedup import Fingerprint
    from train import doc_overlap_guard as DG

    rows = source_rows(data_dir)
    idx = DG.build_index(rows, {"entries": []}, n_parts=1)
    templates = template_segments(rows)
    frows = fingerprint_rows(rows, templates)
    fp = Fingerprint.build(frows, {"source": "jevbench-public", **_source_pin(), "n_template_segments": len(templates),
                                   "template_items": TEMPLATE_ITEMS}, log=lambda *a: None)
    fp.exact_hashes = exact_segment_hashes(frows, templates)
    return idx, fp


def _array_sha(a) -> str:
    import numpy as np

    a = np.ascontiguousarray(a).astype("<u8", copy=False)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def _content_sha(fp) -> str:
    return fp.arrays_sha256()


def write(data_dir: str | Path | None = None) -> dict:
    """Build, save the arrays under DIR and write the pin (``build --write-pin``)."""
    from train import doc_overlap_guard as DG

    idx, fp = build(data_dir)
    ident = {"rule_id": RULE_ID, "source": _source_pin()}
    meta = DG.save_index(idx, INDEX_DIR, ident)
    fp.meta.pop("build_seconds", None)  # keep the committed meta byte-stable across rebuilds
    fp.save(FP_PATH)
    import numpy as np

    np.save(EXACT_PATH, fp.exact_hashes)
    pin = {"rule_id": RULE_ID, "source": _source_pin(), "split": SPLIT, "catalog_id": CATALOG_ID,
           "doc_index": {"rule": DG.RULE, "index_sha256": meta["index_sha256"], "n_eval_rows": idx.n_rows,
                         "n_units": idx.meta["n_units"], "n_grams": idx.meta["n_grams"],
                         "n_postings": idx.meta["n_postings"]},
           "fingerprint": {"version": fp.meta["version"], "arrays_sha256": fp.arrays_sha256(), "rows": fp.meta["rows"],
                           "segments_indexed": fp.meta["segments_indexed"],
                           "ngram13_indexed": fp.meta["ngram13_indexed"],
                           "minhash_segments": fp.meta["minhash_segments"], "short_items": fp.meta["short_items"],
                           "n_template_segments": fp.meta["n_template_segments"], "template_items": TEMPLATE_ITEMS},
           "exact_segments": {"sha256": _array_sha(fp.exact_hashes), "n": int(len(fp.exact_hashes)),
                              "min_words": EXACT_MIN_WORDS}}
    PIN_PATH.write_bytes((json.dumps(pin, indent=1) + "\n").encode("utf-8"))  # LF: pin_sha256 is OS-independent
    return pin


def read_pin() -> dict:
    if not PIN_PATH.exists():
        raise JevBenchGuardError(f"JevBench guard pin missing: {PIN_PATH}")
    return json.loads(PIN_PATH.read_text(encoding="utf-8"))


def pin_sha256() -> str:
    return hashlib.sha256(PIN_PATH.read_bytes()).hexdigest()


_TEST: list[Any] = []


def set_for_tests(pair) -> None:
    """Tests only: install an in-process (EvalIndex, Fingerprint) pair, or a disabled marker ``False``; None resets."""
    _TEST.clear()
    if pair is not None:
        _TEST.append(pair)
    load.cache_clear()


@lru_cache(maxsize=1)
def load():
    """The committed (EvalIndex, Fingerprint), each verified against the pin (fail closed on any drift)."""
    if _TEST:
        return _TEST[0]
    from data.dedup import FP_VERSION, Fingerprint
    from train import doc_overlap_guard as DG

    pin = read_pin()
    if pin.get("rule_id") != RULE_ID:
        raise JevBenchGuardError("JevBench guard pin was made under a different rule")
    if pin["source"] != _source_pin():
        raise JevBenchGuardError("JevBench guard pin does not match jevkit's pinned public items")
    idx = DG.load_index_dir(INDEX_DIR, pin["doc_index"]["index_sha256"])
    DG.check_meta_vs_pin(idx, pin["doc_index"])
    fp = Fingerprint.load(FP_PATH)
    got = fp.arrays_sha256()
    if got != pin["fingerprint"]["arrays_sha256"] or fp.meta.get("version") != FP_VERSION:
        raise JevBenchGuardError(f"JevBench fingerprint arrays {got[:12]} / version {fp.meta.get('version')} "
                                 f"!= pin {pin['fingerprint']['arrays_sha256'][:12]} / {FP_VERSION}")
    import numpy as np

    ex = np.load(EXACT_PATH, allow_pickle=False)
    if _array_sha(ex) != pin["exact_segments"]["sha256"] or len(ex) != pin["exact_segments"]["n"]:
        raise JevBenchGuardError("JevBench exact-segment set does not match the pin")
    fp.exact_hashes = ex
    return idx, fp


def identity() -> dict:
    """What a scan / build report records about this gate."""
    if _TEST:
        pair = _TEST[0]
        if pair is False:
            return {"enabled": False, "note": "disabled in a unit test"}
        return {"enabled": True, "index_sha256": pair[0].meta.get("index_sha256"),
                "fingerprint_arrays_sha256": pair[1].arrays_sha256(), "pin_sha256": "test"}
    pin = read_pin()
    return {"enabled": True, "index_sha256": pin["doc_index"]["index_sha256"],
            "fingerprint_arrays_sha256": pin["fingerprint"]["arrays_sha256"],
            "exact_segments_sha256": pin["exact_segments"]["sha256"], "pin_sha256": pin_sha256(),
            "source_commit": pin["source"]["commit"]}


def exact_segment_hits(fp, state: Any, questions: Any) -> int:
    """Number of the row's normalised segments that equal a JevBench item segment of >= EXACT_MIN_WORDS words
    (templates excluded): exact, cheap, catches copied short items the 13-gram rules cannot reach."""
    import numpy as np

    from data import text as T

    ex = fp.exact_hashes  # set by build() / load(); a fingerprint without it is not a JevBench guard fingerprint
    segs = T.segments(state, questions if isinstance(questions, dict) else {})
    if not segs or not len(ex):
        return 0
    q = np.fromiter((T.h64(s) for s in segs), dtype=np.uint64, count=len(segs))
    return int((fp._lookup(ex, q) >= 0).sum())


def fingerprint_item_hits(fp, state: Any, questions: Any) -> dict[str, list]:
    """All five matchers, keys prefixed ``jevbench_`` (the builder's extra item-tier matchers)."""
    return {f"jevbench_{k}": v for k, v in fp.check(state, questions).items()}


def free_ram_mb() -> int | None:
    """Free physical memory (Windows GlobalMemoryStatusEx; /proc/meminfo elsewhere); None when unknown."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):  # type: ignore[attr-defined]
            return int(m.ullAvailPhys // 2**20)
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def wait_for_ram(min_mb: int, log=print, poll_s: int = 60, max_wait_s: int = 6 * 3600) -> None:
    import time

    waited = 0
    while True:
        free = free_ram_mb()
        if free is None or free >= min_mb or waited >= max_wait_s:
            return
        log(f"free RAM {free} MB < {min_mb} MB: waiting")
        time.sleep(poll_s)
        waited += poll_s


def rescan(paths: Iterable[str | Path], full_fingerprint: bool = True, log=print, min_free_mb: int = 3000) -> dict:
    """Offline JevBench report over pinned draws, ONE draw at a time, streaming (the laptop's RAM is shared):
    pass 1 = the JevBench doc-overlap rule (13-gram R / E / U, this draw's boilerplate) with the tiny JevBench index;
    pass 2 = every JevBench fingerprint matcher (exact / ngram13 / minhash / edit / contain) per record.
    The Index-side scan of these draws is unchanged code with pinned results (data/manifests/doc_clean)."""
    import gc
    import gzip

    from train import doc_overlap_guard as DG

    out = {}
    for p in paths:
        p = Path(p)
        wait_for_ram(min_free_mb, log)
        jidx, fp = load()
        rep = DG.scan(DG.iter_jsonl(p), idx=jidx, keep_ids=True, jevbench=False)
        fp_hits: dict[str, int] = {}
        ex_rows: list[dict] = []
        n = n_hit = 0
        exact_ids: list[str] = []
        if full_fingerprint:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    n += 1
                    if exact_segment_hits(fp, r.get("state"), r.get("questions")):  # the scan's exact gate
                        exact_ids.append(str(r.get("id")))
                    h = {k: v for k, v in fingerprint_item_hits(fp, r.get("state"), r.get("questions")).items() if v}
                    if h:
                        n_hit += 1
                        for k in h:
                            fp_hits[k] = fp_hits.get(k, 0) + 1
                        if len(ex_rows) < 5000:
                            ex_rows.append({"id": r.get("id"), "source": r.get("source"), "matchers": sorted(h)})
                    if len(getattr(fp, "_seg_cache", ())) > 200_000:  # bound the per-segment cache
                        fp._seg_cache.clear()
                        fp._txt_cache.clear()
        doc_ids = rep["fail_ids"]
        gate_ids = sorted(set(doc_ids) | set(exact_ids))
        out[str(p)] = {"n_records": rep["n_records"], "records_sha256": DG._sha_file(p),
                       "jevbench_doc_fail": len(doc_ids), "jevbench_doc_fails": rep["fails"][:5000],
                       "jevbench_exact_segment_rows": len(exact_ids),
                       "jevbench_training_gate_fail": len(gate_ids), "jevbench_training_gate_fail_ids": gate_ids,
                       "jevbench_exact_segment_ids": sorted(exact_ids),
                       "jevbench_fingerprint_rows_hit": n_hit if full_fingerprint else None,
                       "jevbench_fingerprint_hits_by_matcher": fp_hits if full_fingerprint else None,
                       "jevbench_fingerprint_examples": ex_rows, "fingerprint_records_checked": n,
                       "n_boilerplate": rep["n_boilerplate"], "n_template_grams": rep["n_template_grams"]}
        log(json.dumps({str(p): {k: v for k, v in out[str(p)].items()
                                 if k not in ("jevbench_fingerprint_examples", "jevbench_doc_fails")}}))
        del rep
        if hasattr(fp, "_seg_cache"):
            fp._seg_cache.clear()
            fp._txt_cache.clear()
        gc.collect()
    return {"identity": identity(), "draws": out}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.jevbench_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild from the pinned public items")
    b.add_argument("--write-pin", action="store_true")
    b.add_argument("--data-dir")
    v = sub.add_parser("verify", help="rebuild in memory and compare with the committed pin + arrays")
    v.add_argument("--data-dir")
    r = sub.add_parser("rescan", help="report JevBench hits on pinned draws")
    r.add_argument("records", nargs="+")
    r.add_argument("--out")
    r.add_argument("--no-fingerprint", action="store_true")
    r.add_argument("--min-free-mb", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if not a.write_pin:
            idx, fp = build(a.data_dir)
            print(json.dumps({"n_rows": idx.n_rows, "n_grams": idx.meta["n_grams"], "fp": fp.arrays_sha256()}))
            return 0
        print(json.dumps(write(a.data_dir)))
        return 0
    if a.cmd == "verify":
        idx, fp = build(a.data_dir)
        pin = read_pin()
        from train import doc_overlap_guard as DG

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            meta = DG.save_index(idx, Path(td) / "i", {})
        ok = (meta["index_sha256"] == pin["doc_index"]["index_sha256"]
              and fp.arrays_sha256() == pin["fingerprint"]["arrays_sha256"]
              and _array_sha(fp.exact_hashes) == pin["exact_segments"]["sha256"])
        load()  # the committed arrays load and match the pin
        print(json.dumps({"rebuild_matches_pin": ok, "index_sha256": meta["index_sha256"],
                          "fingerprint_arrays_sha256": fp.arrays_sha256(),
                          "exact_segments_sha256": _array_sha(fp.exact_hashes)}))
        return 0 if ok else 1
    rep = rescan(a.records, full_fingerprint=not a.no_fingerprint, min_free_mb=a.min_free_mb)
    if a.out:
        Path(a.out).write_bytes((json.dumps(rep, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

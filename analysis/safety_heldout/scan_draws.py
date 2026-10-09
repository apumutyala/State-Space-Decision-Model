"""Contamination check: does any heldout_v1 template text appear in any training draw on disk? (CPU, streaming)

    python -X utf8 analysis/safety_heldout/scan_draws.py <data root> --out <receipt.json> [files...]

Scans every ``records.jsonl.gz`` / ``training_records.jsonl.gz`` under <data root> (or the given files): every string
in every record (state, questions, targets, aug, ...), after the normalisation of ``baselines.clef_heldout_v1``
(NFKC, format characters such as U+200D removed, the Cyrillic homoglyphs folded, casefolded, word tokens). A hit is
(a) any word 6-gram of a template's static text (text and base64 payload; the design's disjointness criterion), or
(b) an exact contiguous match of a whole static segment of 4-5 words (too short for a 6-gram), or (c) a base64 run
that decodes to text containing (a) or (b). Distinctive markers (template-specific tokens) are counted separately as
information only (natural text may contain them). Receipt: per file sha256, records scanned, hits by template.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.clef_heldout_v1 import _norm_words, ngrams, template_grams  # noqa: E402
from safety_eval import heldout_v1 as HV  # noqa: E402

MARKERS = ("ledgerbot", "policy_lookup", "crm_sync", "hint for automated agents", "<information>",
           "approved_action", "resolved_option", "reassigned by the back office", "worked examples. case 1")
B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
#: templates whose static text has no 4+-word segment: matched structurally on the raw text
STRUCTURAL = {"S02_table": re.compile(r"\|\s*option\s*\|\s*still available\s*\|", re.I)}


def short_segments() -> dict[tuple[str, ...], str]:
    out = {}
    for name, spec in HV.load_templates().items():
        for seg in HV.static_segments(spec):
            w = tuple(_norm_words(seg))
            if 4 <= len(w) < 6:
                out[w] = name
    return out


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def scan_text(text: str, g6: dict, short: dict, short_lens: set[int]) -> set[str]:
    w = _norm_words(text)
    hits = {g6[g] for g in ngrams(w, 6) & g6.keys()}
    for n in short_lens:
        hits |= {short[g] for g in ngrams(w, n) & short.keys()}
    return hits


def scan_file(path: Path, g6, short, short_lens) -> dict:
    h = hashlib.sha256()
    n = 0
    hits: dict[str, list[str]] = {}
    markers = {m: 0 for m in MARKERS}
    with open(path, "rb") as raw:
        for b in iter(lambda: raw.read(1 << 20), b""):
            h.update(b)
    truncated = None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = iter(fh)
        while True:
            try:
                line = next(lines)
            except StopIteration:
                break
            except (EOFError, OSError) as e:  # a partial / interrupted build on disk: scan what is readable, report it
                truncated = f"{type(e).__name__}: {e}"
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                truncated = f"JSONDecodeError at record {n + 1}: {e}"
                break
            n += 1
            text = "\n".join(strings(rec))
            found = scan_text(text, g6, short, short_lens)
            found |= {t for t, rx in STRUCTURAL.items() if rx.search(text)}
            for run in B64.findall(text):
                try:
                    dec = base64.b64decode(run + "=" * (-len(run) % 4), validate=True).decode("utf-8")
                except (binascii.Error, UnicodeDecodeError, ValueError):
                    continue
                found |= scan_text(dec, g6, short, short_lens)
            low = text.casefold()
            for m in MARKERS:
                if m in low:
                    markers[m] += 1
            rid = str(rec.get("id", rec.get("request_id", n)))
            for t in found:
                hits.setdefault(t, []).append(rid)
    return {"path": path.as_posix(), "sha256": h.hexdigest(), "records": n, "truncated": truncated,
            "hits": {t: {"n": len(v), "ids": v[:10]} for t, v in sorted(hits.items())}, "markers_info": markers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    files = [Path(f) for f in a.files] or sorted(
        p for pat in ("**/records.jsonl.gz", "**/training_records.jsonl.gz") for p in root.glob(pat))
    g6 = template_grams(6)
    short = short_segments()
    lens = {len(k) for k in short}
    rep = {"battery": HV.VERSION, "template_set_sha256": HV.TEMPLATE_SET_SHA256, "n_template_6grams": len(g6),
           "short_segments": {" ".join(k): v for k, v in short.items()}, "files": []}
    t0 = time.time()
    for f in files:
        r = scan_file(f, g6, short, lens)
        r["path"] = f.relative_to(root).as_posix() if f.is_relative_to(root) else f.as_posix()
        rep["files"].append(r)
        print(f"{time.time() - t0:7.0f}s {r['records']:8d} recs  hits={ {k: v['n'] for k, v in r['hits'].items()} }  "
              f"{r['path']}", flush=True)
    rep["total_records"] = sum(f["records"] for f in rep["files"])
    rep["files_with_hits"] = [f["path"] for f in rep["files"] if f["hits"]]
    rep["truncated_files"] = {f["path"]: f["truncated"] for f in rep["files"] if f["truncated"]}
    rep["ok"] = not rep["files_with_hits"]
    rep["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"ok": rep["ok"], "files": len(files), "records": rep["total_records"],
                      "files_with_hits": rep["files_with_hits"]}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

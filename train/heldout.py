"""Hash check: the held-out D1-B steer templates (``safety_eval.attacks.D1_TEMPLATES_B``) never appear in training rows.

PREREG §2.2 / amendment v0.4: D1-B is held out from every tuning and training source. A template is a sentence
with placeholders (``{act}``, ``{gold}``, ``{sender}``); its FIXED text is what could leak. For every template we
take the fixed fragments between placeholders, normalise them (NFKC, casefold, non-word runs -> one space) and hash
every word K-gram (sha256, K = ``K_WORDS``; a fragment shorter than K words is hashed whole as one gram). A training
text leaks a template when any of its word K-grams hashes into that template's set, or when a short fragment's hash
equals a hash of the same-length gram in the text. The check is on hashes only, so it can run against texts that are
never committed (``records.jsonl.gz`` on the training box) and the template text itself never has to be
compared verbatim outside this module.

    python -X utf8 -m train.heldout --records <records.jsonl.gz> [--out heldout.json]   # exit 3 on any hit

Fragments of fewer than ``MIN_WORDS`` words (e.g. "; do not use") are too generic to be evidence and are skipped;
the skipped fragments are listed in the report so the coverage is explicit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

K_WORDS = 6
MIN_WORDS = 4
_NONWORD = re.compile(r"[\W_]+", re.UNICODE)


def normalise(text: str) -> list[str]:
    return [w for w in _NONWORD.sub(" ", unicodedata.normalize("NFKC", text).casefold()).split() if w]


def _gram_hash(words: Iterable[str]) -> str:
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()


def fragments(template: str) -> list[str]:
    """The fixed text between ``{placeholder}`` fields."""
    return [lit for lit, _, _, _ in string.Formatter().parse(template) if lit and lit.strip()]


def template_hashes(templates: dict[str, str], k: int = K_WORDS) -> dict[str, Any]:
    """{name: {"grams": {sha256: n_words}, "skipped": [fragment]}}."""
    out: dict[str, Any] = {}
    for name, tpl in sorted(templates.items()):
        grams: dict[str, int] = {}
        skipped = []
        for frag in fragments(tpl):
            w = normalise(frag)
            if len(w) < MIN_WORDS:
                skipped.append(frag)
            elif len(w) < k:
                grams[_gram_hash(w)] = len(w)
            else:
                for i in range(len(w) - k + 1):
                    grams[_gram_hash(w[i: i + k])] = k
        out[name] = {"grams": grams, "skipped": skipped}
    return out


class HeldOutLeak(ValueError):
    pass


def scan(texts: Iterable[tuple[str, str]], templates: dict[str, str] | None = None, k: int = K_WORDS) -> dict[str, Any]:
    """texts: (id, text). Returns {"n_texts", "hits": [{id, template}], "templates", "skipped"}."""
    if templates is None:
        from safety_eval.attacks import D1_TEMPLATES_B as templates  # noqa: N811
    th = template_hashes(templates, k)
    by_len: dict[int, dict[str, str]] = {}
    firsts: dict[int, set[str]] = {}  # prefilter only: a gram is hashed when its first word can start a template gram
    for name, tpl in sorted(templates.items()):
        for frag in fragments(tpl):
            w = normalise(frag)
            if len(w) >= MIN_WORDS:
                size = min(k, len(w))
                firsts.setdefault(size, set()).update(w[i] for i in range(len(w) - size + 1))
    for name, t in th.items():
        for h, n in t["grams"].items():
            by_len.setdefault(n, {})[h] = name
    hits, n = [], 0
    for tid, text in texts:
        n += 1
        w = normalise(text)
        seen = set()
        for size, table in by_len.items():
            fs = firsts[size]
            for i in range(len(w) - size + 1):
                if w[i] not in fs:
                    continue
                name = table.get(_gram_hash(w[i: i + size]))
                if name is not None and name not in seen:
                    seen.add(name)
                    hits.append({"id": tid, "template": name})
    from safety_eval.attacks import template_set_sha256

    return {"n_texts": n, "hits": hits, "k_words": k, "min_words": MIN_WORDS,
            "templates_sha256": template_set_sha256(templates), "n_templates": len(templates),
            "n_gram_hashes": sum(len(t["grams"]) for t in th.values()),
            "skipped_fragments": {nm: t["skipped"] for nm, t in th.items() if t["skipped"]}}


def record_text(rec: dict[str, Any]) -> str:
    """Everything a training row can show the model from a record: state + questions (instructions, options)."""
    from train.guard import record_request

    req = record_request(rec) or rec
    return json.dumps({"state": req.get("state"), "questions": req.get("questions")}, ensure_ascii=False)


def scan_records(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        rep = scan((r["id"], record_text(r)) for r in (json.loads(x) for x in fh if x.strip()))
    rep["records"] = str(p)
    rep["records_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return rep


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.heldout")
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = scan_records(a.records)
    rep["pass"] = not rep["hits"]
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_texts", "pass", "templates_sha256", "n_gram_hashes")}))
    return 0 if rep["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())

"""Training-side exclusion of every row carrying an EVAL ATTACK template (operator rule 2026-10-08: "we can't train on
test", safety test material included). Used by data.b_schema (select stage) and as a stand-alone draw check.

A row is excluded when ANY of:
  1. it is an injected row: ``aug.injection`` present or ``~inj:`` in its id (the 150k pool's D1-A steer rows and its
     training-only hijack rows, all families, dropped conservatively);
  2. its model-facing text (``data.text.raw_segments``: state strings, instructions, option texts) contains, after
     ``data.text.words`` normalisation,
       - a word 6-gram of any template's static text (the text between placeholders), or
       - a template's whole static fragment of 4-5 words verbatim (longer ones are covered by their 6-grams),
     for the template sets
       d1a        safety_eval.attacks.D1_TEMPLATES_A (authority, fake_correction, system_note, urgency)
       d1b        safety_eval.attacks.D1_TEMPLATES_B (v0.2 + v0.4 held-out steer battery)
       hijack     safety_eval.attacks.HIJACK_TEMPLATES (the 5 official)
       train_hijack data.injection.TRAIN_HIJACK (disregard_task, operator_override, predecided, admin_header,
                  grader_note, data_close)
       heldout_v1 stack/clef/safety-heldout @f273618 analysis/safety_heldout/heldout_v1_templates.json (21 templates,
                  read from a sha256-pinned copy outside git; ``$X``-style placeholders and the ``|`` split marker
                  separate fragments; payload strings included).

    python -X utf8 -I -c "...data.attack_filter.main([...])" --records <records.jsonl.gz> --heldout-json <copy> --out <json>
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HELDOUT_V1_SHA256 = "25df1dcf91e7872ff5501b8b9a6ea7a1346594d7213cd03ffe350a41727d8fa5"
HELDOUT_V1_SOURCE = "stack/clef/safety-heldout @f273618 analysis/safety_heldout/heldout_v1_templates.json (blob 1af1d82b)"
N = 6
MIN_EXACT_WORDS = 4


def _frags_braces(t: str) -> list[str]:
    from .injection import static_fragments
    return static_fragments(t)


def _frags_dollar(t: str) -> list[str]:
    return [f for f in re.split(r"\$[A-Z_0-9]+|\|", t) if f.strip()]


def template_sets(heldout_json: str | Path) -> dict[str, dict[str, list[str]]]:
    """{set: {template name: [static fragments]}}"""
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, HIJACK_TEMPLATES
    from .injection import TRAIN_HIJACK

    p = Path(heldout_json)
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != HELDOUT_V1_SHA256:
        raise SystemExit(f"heldout_v1 templates sha256 {got} != pinned {HELDOUT_V1_SHA256}")
    hv1 = json.loads(p.read_text(encoding="utf-8"))
    out = {"d1a": {k: _frags_braces(v) for k, v in D1_TEMPLATES_A.items()},
           "d1b": {k: _frags_braces(v) for k, v in D1_TEMPLATES_B.items()},
           "hijack": {k: _frags_braces(v) for k, v in HIJACK_TEMPLATES.items()},
           "train_hijack": {k: _frags_braces(v) for k, v in TRAIN_HIJACK.items()},
           "heldout_v1": {k: _frags_dollar(v["text"]) + (_frags_dollar(v["payload"]) if v.get("payload") else [])
                          for k, v in hv1.items()}}
    return out


class AttackFilter:
    def __init__(self, heldout_json: str | Path):
        from . import text as T
        self.T = T
        self.sets = template_sets(heldout_json)
        self.grams: dict[str, tuple[str, str]] = {}
        self.exact: list[tuple[str, str, str]] = []
        for s, tpls in self.sets.items():
            for name, frags in tpls.items():
                for f in frags:
                    ws = T.words(f)
                    for i in range(len(ws) - N + 1):
                        self.grams.setdefault(" ".join(ws[i:i + N]), (s, name))
                    if MIN_EXACT_WORDS <= len(ws) < N:  # longer fragments are covered by their 6-grams
                        self.exact.append((" ".join(ws), s, name))
        ident = json.dumps({s: {k: v for k, v in sorted(t.items())} for s, t in sorted(self.sets.items())},
                           sort_keys=True, ensure_ascii=False)
        self.identity = {"template_fragments_sha256": hashlib.sha256(ident.encode("utf-8")).hexdigest(),
                         "n_6grams": len(self.grams), "n_exact_fragments": len(self.exact),
                         "heldout_v1": {"source": HELDOUT_V1_SOURCE, "sha256": HELDOUT_V1_SHA256},
                         "sets": {s: sorted(t) for s, t in self.sets.items()}}

    def hit(self, rec: dict) -> tuple[str, str, str] | None:
        """(reason, template set, template name) or None"""
        aug = rec.get("aug") or {}
        inj = aug.get("injection")
        if inj or "~inj:" in rec.get("id", ""):
            fam = (inj or {}).get("family", "?")
            return ("injected_row", "d1a" if fam == "d1a" else "train_hijack" if fam else "?",
                    (inj or {}).get("template", "?"))
        T = self.T
        for seg in T.raw_segments(rec.get("state"), rec.get("questions") or {}):
            ws = T.words(seg)
            for i in range(len(ws) - N + 1):
                g = self.grams.get(" ".join(ws[i:i + N]))
                if g:
                    return ("6gram", g[0], g[1])
            if len(ws) >= MIN_EXACT_WORDS:
                j = " " + " ".join(ws) + " "
                for frag, s, name in self.exact:
                    if " " + frag + " " in j:
                        return ("exact", s, name)
        return None


def scan(records: str | Path, heldout_json: str | Path) -> dict:
    f = AttackFilter(heldout_json)
    n = 0
    by = collections.Counter()
    ex = []
    with gzip.open(records, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            h = f.hit(r)
            if h:
                by[f"{h[1]}/{h[2]}/{h[0]}"] += 1
                if len(ex) < 10:
                    ex.append((r["id"], *h))
    return {"records": str(records), "n_rows": n, "hits": sum(by.values()), "by_template": dict(by.most_common()),
            "examples": ex, "filter": f.identity, "pass": not by}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="data.attack_filter")
    ap.add_argument("--records", required=True)
    ap.add_argument("--heldout-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = scan(a.records, a.heldout_json)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "hits", "by_template", "pass")}, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

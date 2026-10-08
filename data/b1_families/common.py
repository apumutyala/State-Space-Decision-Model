"""Shared pieces of the B1 family builders: record shape, option shuffling, eval-row reader, family overlap check.

Record shape = plain ``ssdm-train-v0`` (as written by ``data.build``): ``schema, id, family, source, source_id,
source_split, license, state, questions, labels, untrusted_fields, aug`` plus ``provenance`` ("train-split" or
"synthetic"). ``_probe`` (item-tier state/questions for the fingerprint) is stripped before writing.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .. import config
from ..text import canonical_json, normalize

SCHEMA = "ssdm-train-v0"
MAX_OPTIONS = 16          # consumer loader (11fb140) cannot read >16-option sidecars: hard cap per choice question
MAX_QUESTIONS = 16        # per record (BFCL candidate fan-out)
SUITE_FILES = ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")


def make_record(*, family: str, source: str, source_id: str, source_split: str, provenance: str, license: str,
                state: Any, questions: dict, labels: dict, aug: dict, probe: dict) -> dict:
    for qid, q in questions.items():
        if q["type"] == "choice" and len(q["criteria"]) > MAX_OPTIONS:
            raise ValueError(f"{source}:{source_id}/{qid}: {len(q['criteria'])} options > {MAX_OPTIONS}")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"{source}:{source_id}: {len(questions)} questions > {MAX_QUESTIONS}")
    return {"schema": SCHEMA, "id": f"m2:{source}:{source_id}", "family": family, "source": source,
            "source_id": str(source_id), "source_split": source_split, "provenance": provenance, "license": license,
            "state": state, "questions": questions, "labels": labels, "untrusted_fields": [], "aug": aug,
            "_probe": probe}


def shuffled_options(gold: str, distractors: list[str], rng: random.Random, n_opts: int) -> tuple[list[str], int]:
    """Gold plus ``n_opts - 1`` distractors, gold position uniform over 0..n_opts-1 (drawn, not left to sort order)."""
    ds = list(distractors[: n_opts - 1])
    rng.shuffle(ds)
    pos = rng.randrange(len(ds) + 1)
    return ds[:pos] + [gold] + ds[pos:], pos


# ------------------------------------------------------------------------------------------- eval rows
def iter_family_rows(families: set[str], suite_dir: Path | None = None) -> Iterable[dict]:
    """Index rows (both suite files) whose ``family`` or ``benchmark`` is in ``families``. Read-only."""
    d = Path(suite_dir or config.SUITE_DIR)
    for fn in SUITE_FILES:
        with gzip.open(d / fn, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("family") in families or r.get("benchmark") in families:
                    yield r


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class FamilyOverlap:
    """Overlap check of training items against the eval rows of the SAME family (on top of the global
    ``data.dedup.Fingerprint``): normalized exact match of every item text >= ``min_chars`` and any shared word
    ``n``-gram of the item texts. N-grams in more than ``boiler_frac`` of the family's eval rows are template
    boilerplate (instruction prefixes, schema keywords) and are ignored; their count is reported."""

    def __init__(self, family: str, eval_texts: list[list[str]], n: int = 13, min_chars: int = 24,
                 boiler_frac: float = 0.05):
        self.family, self.n, self.min_chars = family, n, min_chars
        df: dict[int, int] = {}
        edf: dict[str, int] = {}
        for texts in eval_texts:
            row_grams: set[int] = set()
            row_exact: set[str] = set()
            for t in texts:
                nt = normalize(t)
                if len(nt) >= min_chars:
                    row_exact.add(nt)
                row_grams |= self._grams(nt)
            for g in row_grams:
                df[g] = df.get(g, 0) + 1
            for e in row_exact:
                edf[e] = edf.get(e, 0) + 1
        cut = max(2, int(boiler_frac * max(1, len(eval_texts))))
        self.grams = {g for g, c in df.items() if c <= cut}
        self.exact = {e for e, c in edf.items() if c <= cut}
        self.n_boiler = len(df) - len(self.grams)
        self.n_boiler_exact = len(edf) - len(self.exact)
        self.n_rows = len(eval_texts)

    def _grams(self, nt: str) -> set[int]:
        ws = nt.split()
        return {hash(" ".join(ws[i:i + self.n])) for i in range(len(ws) - self.n + 1)}

    def hits(self, item_texts: Iterable[str]) -> list[str]:
        out = []
        for t in item_texts:
            nt = normalize(t)
            if len(nt) >= self.min_chars and nt in self.exact:
                out.append(f"exact:{nt[:60]}")
            elif self._grams(nt) & self.grams:
                out.append(f"ngram{self.n}:{nt[:60]}")
        return out

    def identity(self) -> dict:
        return {"family": self.family, "eval_rows": self.n_rows, "n": self.n, "min_chars": self.min_chars,
                "exact_texts": len(self.exact), "ngrams": len(self.grams), "boilerplate_ngrams_ignored": self.n_boiler,
                "boilerplate_exact_ignored": self.n_boiler_exact, "boilerplate_rule": "in > max(2, 5% of eval rows)"}


def probe_texts(probe: dict) -> list[str]:
    return list(_strings(probe.get("state"))) + [s for q in (probe.get("questions") or {}).values()
                                                 for s in _strings([q.get("instructions"), q.get("criteria")])]


_WS = re.compile(r"\s+")


def squash(s: str) -> str:
    return _WS.sub(" ", s).strip()


__all__ = ["make_record", "shuffled_options", "iter_family_rows", "FamilyOverlap", "probe_texts", "canonical_json",
           "MAX_OPTIONS", "MAX_QUESTIONS", "SCHEMA", "squash"]

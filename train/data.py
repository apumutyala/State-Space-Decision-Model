"""Training items: one (request, question, gold option) per row, plus a toy task for tests.

Record format (JSONL): ``{"id": str, "request": <SystemOne request>, "labels": {qid: option_id}}``,
or the PR #16 builder's ``ssdm-train-v0`` shape (``state`` / ``questions`` at the top level; read with
``load_m2_records``, which verifies every line against the committed manifest).
Gold option ids are mapped to the canonical option index of the row. Score labels may be the
level index as a string ("0".."k-1"); noul labels are "true"/"false".

The toy task exists only to exercise the trainer on CPU: a short state names one value of a
field and a choice question asks for it. It is synthetic and has no relation to Index items.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from decide.prompt import Template
from decide.schema import validate_legacy
from heads.layout import EncodeFn, LayoutRow, build_row

from .guard import SplitGuard, record_request


@dataclass
class Item:
    id: str
    row: LayoutRow
    label: int
    axis_target: tuple[float, ...] | None = None  # training-only; never copied into LayoutRow/HeadInput


def items_from_records(records: Iterable[dict[str, Any]], encode: EncodeFn, layout: str = "ordered",
                       template: "str | Template" = "jrt-plain-v1", guard: "SplitGuard | None" = None) -> list[Item]:
    """Rows for training/eval. With a ``guard`` every record is checked against the protected
    Index/test-slice manifests first (``train.guard``)."""
    out = []
    for rec in records:
        if guard is not None:
            guard.check(rec)
        req = record_request(rec)
        if req is None:
            raise ValueError(f"{rec.get('id', '?')}: record has neither 'request' nor top-level state/questions")
        validate_legacy(req)
        from train.cohort import normalized_content_hash

        content_sha = normalized_content_hash(req["state"], req["questions"])
        for qid, gold in sorted(rec["labels"].items()):
            row = build_row(req, qid, encode, template=template, layout=layout)
            row.meta["encoder"] = encoder_identity(encode)
            row.meta["content_sha256"] = content_sha  # normalized record content (train.cohort): disjointness proofs
            gold = str(gold).lower() if row.qtype == "noul" else str(gold)
            if gold not in row.option_ids:
                raise ValueError(f"{rec['id']}/{qid}: gold {gold!r} not among options {row.option_ids}")
            out.append(Item(f"{rec['id']}/{qid}", row, row.option_ids.index(gold)))
    return out


def encoder_identity(encode) -> dict:
    ident = getattr(encode, "identity", None)
    if callable(ident):
        return ident()
    return {"kind": "unidentified", "type": type(encode).__name__}


def split_hash(items: list[Item]) -> str:
    return hashlib.sha256("\n".join(sorted(i.id for i in items)).encode()).hexdigest()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ManifestMismatch(ValueError):
    pass


def read_manifest(path: str | Path) -> dict[str, str]:
    """``{record id: sha256 of its canonical line}`` from a PR #16 ``*.manifest.jsonl.gz``."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return {e["id"]: e["sha256"] for e in (json.loads(x) for x in fh if x.strip())}


def load_m2_records(path: str | Path, manifest: str | Path | dict[str, str] | None = None,
                    keep: Callable[[dict[str, Any]], bool] | None = None, limit: int | None = None,
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream PR #16 ``records.jsonl.gz`` (outside git). Each line's sha256 must equal the committed
    manifest entry for its id (the builder hashes the exact canonical line it writes), so a record
    file that drifted from the reviewed build is refused, not silently trained on. ``keep`` filters
    (e.g. by length for CPU smoke runs) before ``limit``. Returns (records, identity)."""
    want = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out: list[dict[str, Any]] = []
    seen = 0
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            seen += 1
            rec = json.loads(line)
            if want is not None:
                h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                if want.get(rec.get("id")) != h:
                    raise ManifestMismatch(f"{rec.get('id')}: line sha256 {h[:12]} does not match the manifest")
            if keep is not None and not keep(rec):
                continue
            out.append(rec)
            if limit is not None and len(out) >= limit:
                break
    ident = {"path": str(p), "lines_read": seen, "n_selected": len(out), "manifest_verified": want is not None,
             "n_manifest": None if want is None else len(want),
             "selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in out).encode()).hexdigest()}
    return out, ident


COLORS = ["red", "green", "blue", "amber"]


def toy_records(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        color = rng.choice(COLORS)
        filler = rng.choice(["ok", "note", "id 7", "x"])
        size = rng.randrange(3)
        recs.append({
            "id": f"toy{seed}-{i}",
            "request": {
                "model": "toy",
                "state": f"{filler}; color={color}; size={size}",
                "questions": {
                    "c": {"type": "choice", "instructions": "color?", "criteria": {c: None for c in COLORS}},
                    "s": {"type": "score", "instructions": "size?", "criteria": ["0", "1", "2"]},
                },
            },
            "labels": {"c": color, "s": str(size)},
        })
    return recs


#: Compact template for CPU toys (no system prompt), so tiny recurrent models see short rows.
TOY_TEMPLATE = Template(id="toy-v1", head="", tail="\nA:", terminator="\n")

"""Training-data guard: refuse Decision Index rows and the untouched test slice.

GOAL.md integrity rule: no training on, or tuning against, Index test items or the untouched
test slice. The trainer refuses to start without a guard (``TrainConfig.require_guard``), and
``items_from_records(..., guard=)`` checks every record before it becomes a training row.

A record is refused when any of these hold:
* it carries Index evaluation metadata (``_evaluation``), or ``source`` names the Index;
* its id, ``run_id``, ``group_id`` (source case), ``row_sha256`` or ``payload_sha256`` appears in
  a loaded manifest (the m0 harness writes dev/test manifests as gzipped JSONL of exactly these
  keys, ``claude/m0-harness: harness/manifests/*.jsonl.gz``);
* its content matches a protected Index row after laundering (metadata stripped, fresh id, any
  ``model``):
  - **kit payload hash**: exactly the kit's ``payload_sha256``, i.e. sha256 of
    ``json.dumps({"model": "jev-1.13.0", "state", "questions"}, ensure_ascii=False,
    separators=(",", ":"))`` in the row's own key order (``decision_index/suite/build/freeze.py`` and
    ``adapters_added.py``). It is checked against the manifests' ``payload_sha256`` set, so it needs
    no Index content. It is sensitive to key order.
  - **content hash**: ``content_hash(state, questions)`` = ``decide.schema.request_fingerprint`` of
    state + questions only (independent of model, id, key order and option order). The manifests
    carry no such hash, so it is checked against a content index built from the local suite
    (``python -X utf8 -m train.guard build-content-index --out <path>``: sorted hashes only, kept
    outside git; its sha256 is in ``identity()`` and therefore in every receipt). Without a content
    index, ``identity()`` says so.

Neither hash catches paraphrase or partial overlap; that is the PR #16 builder's fingerprint dedup.
Manifests are identified by the sha256 of their uncompressed text, which goes in receipts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from decide.schema import canonical_json, request_fingerprint, sha256_text

KEYS = ("run_id", "group_id", "row_sha256", "payload_sha256")
#: decision_index.constants.REFERENCE_MODEL: the model name inside every 0.2.1 kit payload hash.
KIT_REFERENCE_MODEL = "jev-1.13.0"


class ForbiddenRow(ValueError):
    pass


def kit_payload_hash(state: Any, questions: Any, model: str = KIT_REFERENCE_MODEL) -> str:
    """The kit's ``_evaluation.payload_sha256`` (``decision_index.suite.io.dumps``: insertion order,
    ``ensure_ascii=False``, compact separators)."""
    payload = {"model": model, "state": state, "questions": questions}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def content_hash(state: Any, questions: Any) -> str | None:
    """Laundering-invariant content identity: state + questions, order-insensitive, no model or id."""
    return _safe_fingerprint({"state": state, "questions": questions})


def record_request(record: dict[str, Any]) -> dict[str, Any] | None:
    """The SystemOne request a training record carries.

    Two record shapes exist: ``{"request": {...}}`` (``train.data`` toys) and the PR #16 builder's
    ``ssdm-train-v0`` records, which keep ``state`` and ``questions`` at the top level (no ``request``
    key). Both resolve to ``{"model", "state", "questions"}`` so the guard's hash checks and row
    construction see the same request."""
    req = record.get("request")
    if req is not None:
        return req
    if "questions" in record and "state" in record:
        return {"model": str(record.get("model", "ssdm-train")), "state": record["state"],
                "questions": record["questions"]}
    return None


@dataclass
class SplitGuard:
    ids: set[str] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)
    manifests: dict[str, str] = field(default_factory=dict)  # path -> sha256 of uncompressed text
    note: str = ""
    content_hashes: set[str] = field(default_factory=set)
    content_index: dict[str, Any] | None = None  # {"path", "sha256", "n"} of a loaded content index

    @classmethod
    def from_manifests(cls, paths: Iterable[str | Path], content_index: str | Path | None = None) -> "SplitGuard":
        g = cls()
        for p in paths:
            p = Path(p)
            raw = gzip.open(p, "rt", encoding="utf-8").read() if p.suffix == ".gz" else p.read_text(encoding="utf-8")
            g.manifests[str(p)] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            for line in raw.splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                for k in ("run_id", "group_id"):
                    if e.get(k) is not None:
                        g.ids.add(str(e[k]))
                for k in ("row_sha256", "payload_sha256"):
                    if e.get(k):
                        g.hashes.add(str(e[k]))
        if content_index is not None:
            g.load_content_index(content_index)
        return g

    def load_content_index(self, path: str | Path) -> "SplitGuard":
        p = Path(path)
        raw = gzip.open(p, "rt", encoding="ascii").read() if p.suffix == ".gz" else p.read_text(encoding="ascii")
        hs = {x.strip() for x in raw.splitlines() if x.strip()}
        self.content_hashes |= hs
        self.content_index = {"path": str(p), "sha256": hashlib.sha256(raw.encode()).hexdigest(), "n": len(hs)}
        return self

    def add_index_rows(self, rows: Iterable[dict[str, Any]]) -> "SplitGuard":
        """Protect these Index rows by content (tests; ``build_content_index`` does the real suite)."""
        for r in rows:
            h = content_hash(r["state"], r["questions"])
            if h:
                self.content_hashes.add(h)
        return self

    @classmethod
    def synthetic_only(cls, note: str) -> "SplitGuard":
        """For synthetic data that cannot overlap the Index (toys, unit tests). Still refuses
        anything carrying Index metadata."""
        return cls(note=f"synthetic-only: {note}")

    def identity(self) -> dict[str, Any]:
        if self.content_index is not None:
            ci: Any = self.content_index
        elif self.content_hashes:
            ci = {"n": len(self.content_hashes), "source": "add_index_rows"}
        else:
            ci = "none loaded: laundered rows are caught only by the exact (key-order sensitive) kit payload hash"
        return {"manifests": self.manifests, "n_ids": len(self.ids), "n_hashes": len(self.hashes), "note": self.note,
                "content_index": ci}

    def check(self, record: dict[str, Any]) -> None:
        rid = str(record.get("id", "?"))
        if "_evaluation" in record or str(record.get("source", "")).lower().startswith(("index", "decision-index")):
            raise ForbiddenRow(f"{rid}: carries Decision Index metadata; Index rows are never training data")
        meta = record.get("_evaluation") or {}
        for k in KEYS:
            v = record.get(k, meta.get(k))
            if v is not None and (str(v) in self.ids or str(v) in self.hashes):
                raise ForbiddenRow(f"{rid}: {k}={v} is in a protected manifest")
        if rid in self.ids:
            raise ForbiddenRow(f"{rid}: id is in a protected manifest")
        req = record_request(record)
        if req is None or "state" not in req or "questions" not in req:
            return
        st, qs = req["state"], req["questions"]
        models = sorted({KIT_REFERENCE_MODEL, str(req.get("model"))})
        for h in [kit_payload_hash(st, qs, m) for m in models] + [sha256_text(canonical_json(req))]:
            if h in self.hashes:
                raise ForbiddenRow(f"{rid}: payload hash {h[:12]}... equals a protected Index payload_sha256")
        ch = content_hash(st, qs)
        if ch and ch in self.content_hashes:
            raise ForbiddenRow(f"{rid}: content hash {ch[:12]}... matches a protected Index row (laundered copy)")


def _safe_fingerprint(req: dict[str, Any]) -> str | None:
    try:
        return request_fingerprint(req)
    except Exception:  # noqa: BLE001 - malformed requests are rejected later by validation
        return None


def build_content_index(out: str | Path, manifests: Iterable[str | Path], suite_dir: str | Path | None = None) -> dict:
    """Content hashes of every Index row named in ``manifests`` (dev + test), read from the local 0.2.1
    suite through the kit. Writes sorted hashes only (no content), gzipped; returns its identity."""
    from harness.suite import index_rows, load_suite

    want = SplitGuard.from_manifests(manifests).ids
    hs: set[str] = set()
    n_rows = 0
    for r in index_rows(load_suite(suite_dir)):
        e = r["_evaluation"]
        if str(e["run_id"]) in want or str(e["group_id"]) in want:
            n_rows += 1
            h = content_hash(r["state"], r["questions"])
            if h:
                hs.add(h)
    text = "\n".join(sorted(hs)) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(out, "wb", mtime=0) as f:
        f.write(text.encode("ascii"))
    return {"path": str(out), "sha256": hashlib.sha256(text.encode()).hexdigest(), "n": len(hs), "n_rows": n_rows}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-content-index", help="hash every dev+test Index row's content (local suite)")
    b.add_argument("--out", required=True)
    b.add_argument("--suite", default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "harness" / "manifests"
    print(json.dumps(build_content_index(a.out, [root / "dev-manifest.jsonl.gz", root / "test-manifest.jsonl.gz"],
                                         a.suite), indent=1))

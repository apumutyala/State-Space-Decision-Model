"""Training-data guard: refuse Decision Index rows and the untouched test slice.

GOAL.md integrity rule: no training on, or tuning against, Index test items or the untouched
test slice. The trainer refuses to start without a guard (``TrainConfig.require_guard``), and
``items_from_records(..., guard=)`` checks every record before it becomes a training row.

A record is refused when any of these hold:
* it carries Index evaluation metadata (``_evaluation``), or ``source`` names the Index;
* its id, ``run_id``, ``group_id`` (source case), ``row_sha256`` or ``payload_sha256`` appears in
  a loaded manifest (the m0 harness writes dev/test manifests as gzipped JSONL of exactly these
  keys, ``claude/m0-harness: harness/manifests/*.jsonl.gz``);
* the sha256 of its canonical request JSON, or the m0 ``request_fingerprint``, appears in a
  manifest's hash set. The kit's own payload canonicalisation may differ from ours, so this
  check is defence in depth, not a proof of disjointness; the metadata and id checks are the
  primary ones.

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


class ForbiddenRow(ValueError):
    pass


@dataclass
class SplitGuard:
    ids: set[str] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)
    manifests: dict[str, str] = field(default_factory=dict)  # path -> sha256 of uncompressed text
    note: str = ""

    @classmethod
    def from_manifests(cls, paths: Iterable[str | Path]) -> "SplitGuard":
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
        return g

    @classmethod
    def synthetic_only(cls, note: str) -> "SplitGuard":
        """For synthetic data that cannot overlap the Index (toys, unit tests). Still refuses
        anything carrying Index metadata."""
        return cls(note=f"synthetic-only: {note}")

    def identity(self) -> dict[str, Any]:
        return {"manifests": self.manifests, "n_ids": len(self.ids), "n_hashes": len(self.hashes), "note": self.note}

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
        req = record.get("request")
        if req is not None:
            for h in (sha256_text(canonical_json(req)), _safe_fingerprint(req)):
                if h and h in self.hashes:
                    raise ForbiddenRow(f"{rid}: request hash {h[:12]}... is in a protected manifest")


def _safe_fingerprint(req: dict[str, Any]) -> str | None:
    try:
        return request_fingerprint(req)
    except Exception:  # noqa: BLE001 - malformed requests are rejected later by validation
        return None

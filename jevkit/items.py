"""JevBench item records, the pinned public item loader, and the eval-only fingerprint plan.

Item record (JevBench ``Task`` + ``tier``): ``id, family, state, question{type, instructions, criteria}, labels,
expected, split, group, provenance`` and ``tier`` (easy / standard / judge / hard). ``split`` "public"/"open" counts
as the open side of Intelligence; anything else ("private", "sealed", "heldout") as the sealed side.

Public items: github.com/fstandhartinger/jevbench (MIT) ``datasets/public/{original,easy,hard}.jsonl`` at a pinned
commit, sha256-checked on every load. They live under ``ssdm-data/jevbench/<commit>/``, never in the repo. They are an
EVAL-ONLY set: never train or tune on them (``overlap_guard_rows`` gives the doc-overlap guard rows for its next
index rebuild; see jevkit/README.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .spec import TIERS, TYPES

PIN_PATH = Path(__file__).resolve().parent / "data" / "jevbench_public_pin.json"


class PinMismatch(RuntimeError):
    pass


@dataclass
class Item:
    id: str
    family: str
    state: Any
    question: dict
    labels: list
    expected: Any
    split: str
    tier: str = "standard"
    group: str | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.question["type"]

    @property
    def side(self) -> str:
        return "open" if self.split in ("public", "open") else "sealed"

    def validate(self) -> Item:
        if self.type not in TYPES:
            raise ValueError(f"{self.id}: bad question type {self.type!r}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.id}: bad tier {self.tier!r}")
        if not self.labels or len(set(map(str, self.labels))) != len(self.labels):
            raise ValueError(f"{self.id}: labels must be non-empty and unique")
        if self.type == "noul" and sorted(self.labels) != ["no", "yes"]:
            raise ValueError(f"{self.id}: noul labels must be ['no', 'yes']")
        if self.type == "score" and list(self.labels) != [str(i) for i in range(len(self.labels))]:
            raise ValueError(f"{self.id}: score labels must be '0'..'K-1'")
        if self.type == "choice" and set(self.labels) != set(map(str, self.question.get("criteria") or {})):
            raise ValueError(f"{self.id}: choice labels must equal the criteria keys")
        if self.expected is not None:
            if self.type == "score":
                if isinstance(self.expected, bool) or not isinstance(self.expected, int) or str(self.expected) not in self.labels:
                    raise ValueError(f"{self.id}: score expected must be an int level index in labels")
            elif self.expected not in self.labels:
                raise ValueError(f"{self.id}: expected {self.expected!r} not in labels")
        if isinstance(self.state, dict):
            for banned in ("expected", "label", "ground_truth", "answer_key"):
                if banned in self.state:
                    raise ValueError(f"{self.id}: state contains banned key {banned!r}")
        return self

    @property
    def k(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict, tier: str | None = None) -> Item:
        return Item(id=str(d["id"]), family=d.get("family", "unknown"), state=d.get("state"), question=d["question"],
                    labels=[str(x) for x in d["labels"]], expected=d.get("expected"), split=d.get("split", "public"),
                    tier=d.get("tier") or tier or "standard", group=d.get("group"),
                    provenance=d.get("provenance") or {}).validate()


def load_items(path: str | Path, tier: str | None = None) -> list[Item]:
    """Read an item JSONL. ``tier`` fills items without a ``tier`` field (public files carry none)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Item.from_dict(json.loads(line), tier=tier))
    ids = [it.id for it in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item ids")
    return out


def write_items(items: Iterable[Item], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(it.validate().to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for it in items)


# ------------------------------------------------------------------------------------------- pinned public items
def read_pin(path: str | Path = PIN_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def default_data_dir(pin: dict | None = None) -> Path:
    pin = pin or read_pin()
    root = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))
    return root / "jevbench" / pin["commit"]


def verify_public(data_dir: str | Path | None = None, pin: dict | None = None) -> dict[str, str]:
    """sha256 every pinned file; raise PinMismatch on a missing or changed file. Returns {name: sha256}."""
    pin = pin or read_pin()
    d = Path(data_dir) if data_dir else default_data_dir(pin)
    got = {}
    for name, meta in pin["files"].items():
        p = d / name
        if not p.exists():
            raise PinMismatch(f"{p} missing (run: python -m jevkit fetch)")
        got[name] = sha256_file(p)
        if got[name] != meta["sha256"]:
            raise PinMismatch(f"{p}: sha256 {got[name]} != pinned {meta['sha256']}")
    return got


def load_public(data_dir: str | Path | None = None, pin: dict | None = None) -> list[Item]:
    """The pinned public items (original -> standard, easy -> easy, hard -> hard), verified, eval-only."""
    pin = pin or read_pin()
    d = Path(data_dir) if data_dir else default_data_dir(pin)
    verify_public(d, pin)
    out: list[Item] = []
    for name, meta in pin["files"].items():
        items = load_items(d / name, tier=meta["tier"])
        if len(items) != meta["n"]:
            raise PinMismatch(f"{name}: {len(items)} items, pinned {meta['n']}")
        for it in items:
            it.provenance = {**it.provenance, "jevkit_source": f"{pin['repo']}@{pin['commit']}:{name}",
                             "jevkit_eval_only": True}
        out += items
    return out


def fetch_public(dest: str | Path | None = None, pin: dict | None = None) -> dict[str, str]:
    """Download the pinned files (network; CLI only, never in tests) and verify them."""
    pin = pin or read_pin()
    d = Path(dest) if dest else default_data_dir(pin)
    d.mkdir(parents=True, exist_ok=True)
    for name, meta in pin["files"].items():
        url = pin["raw_url"].format(commit=pin["commit"], path=meta["path"])
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise PinMismatch(f"{url}: downloaded bytes do not match the pin")
        (d / name).write_bytes(data)
    return verify_public(d, pin)


# ------------------------------------------------------------------------------------------- eval-only fingerprints
def overlap_guard_rows(items: Iterable[Item]) -> Iterator[tuple[str, str, str, dict]]:
    """Rows in ``train.doc_overlap_guard.iter_eval_rows`` shape: (split, run_id, family, row).

    For the guard's NEXT index rebuild (not done here): appending these to its eval sources makes every JevBench
    public item an eval document, so no training draw can carry its state / instructions / criteria.
    """
    for it in items:
        yield ("JEVBENCH", f"jevbench:{it.id}", f"jevbench_{it.family}",
               {"id": it.id, "family": it.family, "state": it.state, "questions": {"q": it.question}})


def fingerprint_manifest(items: Iterable[Item]) -> dict:
    """ids + sha256 of each item's canonical text (no text): the eval-only list for the overlap-guard plan."""
    rows = []
    for it in items:
        blob = json.dumps({"state": it.state, "question": it.question}, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        rows.append({"id": it.id, "family": it.family, "tier": it.tier,
                     "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()})
    rows.sort(key=lambda r: r["id"])
    h = hashlib.sha256("".join(r["id"] + r["sha256"] for r in rows).encode()).hexdigest()
    return {"n": len(rows), "manifest_sha256": h, "items": rows}

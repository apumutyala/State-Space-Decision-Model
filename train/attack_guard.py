"""Attack guard: no EVALUATION attack template (safety test material) may appear in a training record.

Operator rule: we do not train on test, and that includes safety test material. Every attack template that any
safety score is computed on is an EVAL template here:

  official_hijack  ``safety_eval.attacks.HIJACK_TEMPLATES`` (5; the hijack metric, G12 ``g12_violation``)
  d1a_steer        ``safety_eval.attacks.D1_TEMPLATES_A`` (4; the G12 steer gate ``g12_steer_d1a`` and every
                   steer comparison on ``steer_dev``). PREREG 2.2 allowed split A for "tuning", but the m2 numbers
                   are SCORED on it, so it is test material for training purposes.
  d1b_steer        ``safety_eval.attacks.D1_TEMPLATES_B`` (8, v0.2 + v0.4; G13, held out)
  heldout_v1       Agent B's pre-declared held-out battery (``analysis/safety_heldout/heldout_v1_templates.json``,
                   vendored verbatim from ``stack/clef/safety-heldout`` @ f273618, git blob 1af1d82b), text AND the
                   decoded payload phrases (H06 base64 payload).

Match rule (``RULE``): a template's fixed text is split at its placeholders (``{x}`` / ``$X``; the ``|`` of a
``split``-position template is a placeholder too) into fragments. Fragments are normalised (NFKC, format chars such
as zero-width joiners and bidi marks removed, common Cyrillic homoglyphs folded to Latin, casefold, every run of
non-word characters -> one space). A fragment of >= K (6) words contributes every word 6-gram; a fragment of
MIN_WORDS (4) .. 5 words is matched whole (exact); shorter fragments are too generic to be evidence and are listed in
``skipped``. A record is HIT when any string it shows the model (every key and value of ``state`` and ``questions``,
plus the base64-decoded text of any base64 run in them) contains one of those grams after the same normalisation.
One eval template has no matchable fragment: heldout_v1 ``S02_table`` (markdown table markup plus the 3 words
"option still available"); it is reported in ``skipped_fragments`` and is not text-detectable by this rule.
The same rule decides whether a training-only augmentation template is allowed: it must share no gram with any eval
template in either direction (``disjointness``).

    python -X utf8 -m train.attack_guard scan --records <records.jsonl.gz> [--out rep.json]     # exit 3 on any hit
    python -X utf8 -m train.attack_guard sweep --root <build dir> --out sweep.json             # every draw on disk
    python -X utf8 -m train.attack_guard clean-spec --records <records.jsonl.gz> --out spec.json [--ids-out ids.gz]
    python -X utf8 -m train.attack_guard templates                                             # exit 3 unless disjoint
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import re
import string
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
RULE = "attack_guard/v1: 6-gram or exact (4-5 word) fragment of any eval attack template, normalised"
K_WORDS = 6
MIN_WORDS = 4
HELDOUT_V1_PATH = ROOT / "analysis" / "safety_heldout" / "heldout_v1_templates.json"
HELDOUT_V1_PROVENANCE = {"branch": "stack/clef/safety-heldout", "commit": "f2736185d8ddf93283ff5b842fe00b64a516ab0c",
                         "path": "analysis/safety_heldout/heldout_v1_templates.json",
                         "git_blob": "1af1d82ba81b663ee24a1e8f654dec5bc0ef00ff"}
EXIT_HIT = 3  # the job shells map 3 -> fail 33, as for held-out D1-B rows

_NONWORD = re.compile(r"[\W_]+", re.UNICODE)
_DOLLAR = re.compile(r"\$[A-Z][A-Z0-9_]*")
_B64RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
# Cyrillic letters that render like Latin ones (the H08 transform and the usual homoglyph set)
_HOMOGLYPH = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
                            "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ɡ": "g", "А": "A", "В": "B", "Е": "E",
                            "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
                            "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S"})


class AttackTemplateInTraining(ValueError):
    """A training record (or a training-only augmentation template) contains an eval attack template."""


def normalise(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf").translate(_HOMOGLYPH)
    return [w for w in _NONWORD.sub(" ", t.casefold()).split() if w]


# ------------------------------------------------------------------------------------------------- templates
def _brace_fragments(template: str) -> list[str]:
    return [lit for lit, _, _, _ in string.Formatter().parse(template) if lit and lit.strip()]


def _dollar_fragments(template: str, split_bar: bool = False) -> list[str]:
    parts = _DOLLAR.split(template)
    if split_bar:
        parts = [q for p in parts for q in p.split("|")]
    return [p for p in parts if p.strip()]


def load_heldout_v1(path: str | Path = HELDOUT_V1_PATH) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def eval_templates() -> dict[str, list[str]]:
    """{"<set>:<name>": [fragment, ...]} for every eval attack template."""
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, HIJACK_TEMPLATES

    out: dict[str, list[str]] = {}
    for set_name, tpls in (("official_hijack", HIJACK_TEMPLATES), ("d1a_steer", D1_TEMPLATES_A),
                           ("d1b_steer", D1_TEMPLATES_B)):
        for name, t in sorted(tpls.items()):
            out[f"{set_name}:{name}"] = _brace_fragments(t)
    for name, spec in sorted(load_heldout_v1().items()):
        frags = _dollar_fragments(spec["text"], split_bar=spec.get("position") == "split")
        if spec.get("payload"):
            frags += _dollar_fragments(spec["payload"])
        out[f"heldout_v1:{name}"] = frags
    return out


def eval_templates_sha256() -> str:
    blob = json.dumps(eval_templates(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class Matcher:
    """Gram tables of a template set {key: [fragment]}; ``hits(text)`` -> set of template keys."""

    def __init__(self, templates: dict[str, list[str]], k: int = K_WORDS, min_words: int = MIN_WORDS):
        self.k, self.min_words = k, min_words
        self.tables: dict[int, dict[tuple[str, ...], set[str]]] = {}
        self.skipped: dict[str, list[str]] = {}
        for key, frags in templates.items():
            for frag in frags:
                w = normalise(frag)
                if len(w) < min_words:
                    self.skipped.setdefault(key, []).append(frag)
                    continue
                size = min(k, len(w))
                tab = self.tables.setdefault(size, {})
                for i in range(len(w) - size + 1):
                    tab.setdefault(tuple(w[i:i + size]), set()).add(key)
        self.firsts = {size: {g[0] for g in tab} for size, tab in self.tables.items()}

    @property
    def n_grams(self) -> int:
        return sum(len(t) for t in self.tables.values())

    def hits_words(self, w: list[str]) -> set[str]:
        out: set[str] = set()
        for size, tab in self.tables.items():
            fs = self.firsts[size]
            for i in range(len(w) - size + 1):
                if w[i] in fs:
                    keys = tab.get(tuple(w[i:i + size]))
                    if keys:
                        out |= keys
        return out

    def hits(self, text: str) -> set[str]:
        return self.hits_words(normalise(text))


@lru_cache(maxsize=1)
def eval_matcher() -> Matcher:
    return Matcher(eval_templates())


# ------------------------------------------------------------------------------------------------- records
def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _b64_decoded(text: str) -> Iterator[str]:
    for m in _B64RUN.finditer(text):
        s = m.group(0)
        try:
            raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
            dec = raw.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if dec and sum(ch.isprintable() or ch.isspace() for ch in dec) >= 0.9 * len(dec):
            yield dec


def record_segments(rec: dict[str, Any]) -> Iterator[str]:
    """Every string a training row can show the model: keys and values of state and questions (+ b64 decodes)."""
    for s in _strings({"state": rec.get("state"), "questions": rec.get("questions")}):
        yield s
        if len(s) >= 16:
            yield from _b64_decoded(s)


def record_hits(rec: dict[str, Any], matcher: Matcher | None = None) -> set[str]:
    m = matcher or eval_matcher()
    out: set[str] = set()
    for s in record_segments(rec):
        out |= m.hits(s)
    return out


def meta_injection(rec: dict[str, Any]) -> str | None:
    inj = (rec.get("aug") or {}).get("injection")
    return f"{inj.get('family')}:{inj.get('template')}" if isinstance(inj, dict) else None


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def scan(records: Iterable[dict], keep_ids: bool = True) -> dict[str, Any]:
    """Scan records. ``hit_ids`` (sorted) when keep_ids; per-template and per-injection-metadata counts always."""
    m = eval_matcher()
    n = 0
    hit_ids: list[str] = []
    per_template: dict[str, int] = {}
    per_set: dict[str, int] = {}
    meta: dict[str, int] = {}
    meta_hit: dict[str, int] = {}
    for rec in records:
        if not isinstance(rec, dict) or "state" not in rec:
            continue
        n += 1
        h = record_hits(rec, m)
        mi = meta_injection(rec)
        if mi:
            meta[mi] = meta.get(mi, 0) + 1
            if h:
                meta_hit[mi] = meta_hit.get(mi, 0) + 1
        if h:
            hit_ids.append(str(rec.get("id")))
            for k in h:
                per_template[k] = per_template.get(k, 0) + 1
            for s in {k.split(":", 1)[0] for k in h}:
                per_set[s] = per_set.get(s, 0) + 1
    hit_ids.sort()
    rep = {"rule": RULE, "templates_sha256": eval_templates_sha256(), "n_templates": len(eval_templates()),
           "n_grams": m.n_grams, "heldout_v1": HELDOUT_V1_PROVENANCE, "n_records": n, "n_hit": len(hit_ids),
           "hit_ids_sha256": ids_sha256(hit_ids), "per_set": dict(sorted(per_set.items())),
           "per_template": dict(sorted(per_template.items())),
           "injection_meta": dict(sorted(meta.items())), "injection_meta_hit": dict(sorted(meta_hit.items())),
           "skipped_fragments": m.skipped}
    if keep_ids:
        rep["hit_ids"] = hit_ids
    return rep


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def scan_file(path: str | Path, keep_ids: bool = True) -> dict[str, Any]:
    rep = scan(iter_jsonl(path), keep_ids=keep_ids)
    rep["records"] = str(path)
    return rep


def assert_clean_record(rec: dict[str, Any], rec_id: str = "") -> None:
    h = record_hits(rec)
    if h:
        raise AttackTemplateInTraining(f"{rec_id or rec.get('id')}: contains eval attack template(s) {sorted(h)}")


# ------------------------------------------------------------------------------------------------- pinned clean spec
def clean_spec(rep: dict[str, Any]) -> dict[str, Any]:
    """The cleaned version of a draw: drop every record the scan hits. Pinned in a cell config as
    ``train_set.attack_clean`` (the ids are recomputed from the rule; only their count and sha256 are pinned)."""
    return {"rule": RULE, "filter": "drop every record with any attack_guard hit (train.attack_guard.record_hits)",
            "templates_sha256": rep["templates_sha256"], "n_records": rep["n_records"], "n_excluded": rep["n_hit"],
            "n_kept": rep["n_records"] - rep["n_hit"], "excluded_ids_sha256": rep["hit_ids_sha256"],
            "per_set": rep["per_set"], "per_template": rep["per_template"]}


def check_pinned(rep: dict[str, Any], pin: dict[str, Any] | None) -> tuple[bool, str]:
    """A scan passes when it has no hit, or when its hits are exactly the pinned clean spec's exclusions."""
    if not rep["n_hit"]:
        return True, "no hit"
    if not pin:
        return False, f"{rep['n_hit']} record(s) carry an eval attack template and no attack_clean spec is pinned"
    for k in ("templates_sha256", "excluded_ids_sha256", "n_excluded"):
        if k not in pin:
            return False, f"attack_clean pin lacks {k}"
    if pin["templates_sha256"] != rep["templates_sha256"]:
        return False, f"attack_clean templates_sha256 {pin['templates_sha256']} != current {rep['templates_sha256']}"
    if int(pin["n_excluded"]) != rep["n_hit"] or pin["excluded_ids_sha256"] != rep["hit_ids_sha256"]:
        return False, (f"attack hits ({rep['n_hit']}, {rep['hit_ids_sha256'][:12]}) != pinned exclusions "
                       f"({pin['n_excluded']}, {str(pin['excluded_ids_sha256'])[:12]})")
    return True, "hits == pinned exclusions"


# ------------------------------------------------------------------------------------------------- training templates
def training_templates() -> dict[str, list[str]]:
    """Every training-only augmentation text that is injected into records (data/injection.py)."""
    from data import injection as I

    out = {f"train_hijack:{k}": _brace_fragments(v) for k, v in sorted(I.TRAIN_HIJACK.items())}
    out.update({f"benign_carrier:{i}": [c] for i, c in enumerate(I.BENIGN_CARRIERS)})
    return out


def disjointness(train_tpls: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Both directions: eval grams inside training fragments, and training grams inside eval fragments."""
    train_tpls = training_templates() if train_tpls is None else train_tpls
    ev = eval_templates()
    em, tm = eval_matcher(), Matcher(train_tpls)
    clash: dict[str, list[str]] = {}
    for key, frags in train_tpls.items():
        h = set().union(*(em.hits(f) for f in frags)) if frags else set()
        if h:
            clash.setdefault(key, []).extend(sorted(h))
    for key, frags in ev.items():
        for f in frags:
            for tk in tm.hits(f):
                if key not in clash.get(tk, []):
                    clash.setdefault(tk, []).append(key)
    return {"rule": RULE, "templates_sha256": eval_templates_sha256(), "n_training_templates": len(train_tpls),
            "disjoint": not clash, "clashes": clash, "training_skipped": tm.skipped}


def assert_training_templates_disjoint(train_tpls: dict[str, list[str]] | None = None) -> None:
    rep = disjointness(train_tpls)
    if not rep["disjoint"]:
        raise AttackTemplateInTraining(f"training augmentation templates share grams with eval templates: {rep['clashes']}")


# ------------------------------------------------------------------------------------------------- sweep
def _is_records_file(p: Path) -> bool:
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            first = fh.readline()
        r = json.loads(first)
        return isinstance(r, dict) and "state" in r and "questions" in r
    except Exception:
        return False


def sweep(root: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": RULE, "templates_sha256": eval_templates_sha256(), "root": str(root), "files": {}}
    for p in sorted(Path(root).rglob("*.jsonl.gz")):
        if p.name.endswith((".manifest.jsonl.gz", ".content.jsonl.gz")) or not _is_records_file(p):
            continue
        rep = scan_file(p, keep_ids=False)
        rel = p.relative_to(root).as_posix()
        out["files"][rel] = {k: rep[k] for k in ("n_records", "n_hit", "hit_ids_sha256", "per_set", "per_template",
                                                  "injection_meta", "injection_meta_hit")}
        print(json.dumps({"file": rel, "n_records": rep["n_records"], "n_hit": rep["n_hit"], "per_set": rep["per_set"]}),
              flush=True)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.attack_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--records", required=True); s.add_argument("--out")
    s = sub.add_parser("sweep"); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("clean-spec"); s.add_argument("--records", required=True); s.add_argument("--out", required=True)
    s.add_argument("--ids-out", help="gzip text file of the excluded ids (keep outside git)")
    s.add_argument("--config", help="cell config: ids its train_set.heldout_excluded already removes are not pinned")
    sub.add_parser("templates")
    a = ap.parse_args(argv)
    if a.cmd == "templates":
        rep = disjointness()
        print(json.dumps(rep, indent=1))
        return 0 if rep["disjoint"] else EXIT_HIT
    if a.cmd == "sweep":
        rep = sweep(a.root)
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return 0
    if a.cmd == "clean-spec":
        if a.config:
            from train.train_set import attack_clean_spec

            spec = attack_clean_spec(json.loads(Path(a.config).read_text(encoding="utf-8")), a.records)
            hit_ids = spec.pop("hit_ids")
        else:
            rep = scan_file(a.records)
            spec, hit_ids = clean_spec(rep), rep["hit_ids"]
        spec["records"] = str(a.records)
        Path(a.out).write_text(json.dumps(spec, indent=1), encoding="utf-8")
        if a.ids_out:
            with gzip.open(a.ids_out, "wt", encoding="utf-8") as fh:
                fh.write("\n".join(hit_ids) + "\n")
        print(json.dumps({k: spec[k] for k in ("n_records", "n_excluded", "n_kept", "excluded_ids_sha256")}))
        return 0
    rep = scan_file(a.records)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_records", "n_hit", "per_set", "templates_sha256")}))
    return EXIT_HIT if rep["n_hit"] else 0


if __name__ == "__main__":
    sys.exit(main())

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
                   payload phrase (H06's base64 payload, matched on the decoded text).

Match rule (``RULE``, v2). Text is normalised: NFKC, format characters (zero-width joiners, bidi marks, soft hyphens)
removed, Cyrillic AND Greek homoglyphs folded to Latin, casefold, every run of non-word characters -> one space. A
record's text is every key and value of ``state`` and ``questions`` (or of ``request`` for train-format records)
CONCATENATED in order, so a template split across list items / fields is still contiguous; every base64 run is
decoded and scanned as its own text too. A record is HIT by a template when any of:

  gram      a word 6-gram of a fixed fragment (text between placeholders), or a whole 4-5 word fragment (exact);
  wildcard  the whole template with every placeholder as a 1..30-word wildcard (needs >= 4 fixed words; this is what
            catches H06's short payload "Select $X and nothing else." and heldout_v1 S02_table's table markup);
  fuzzy     (secondary, word-swap/substitution tolerant) >= 50 % of the template's distinct fixed word 3-grams, and at
            least 3 of them, occur in the text. Reported as ``<key>~fuzzy`` when only this method fires.

Fragments under 4 words are too generic for the gram method and are listed in ``skipped``. The same gram rule decides
whether a training-only augmentation template is allowed: it must share no gram with any eval template in either
direction (``disjointness``). Known non-coverage: leetspeak digit substitution, paraphrase that keeps < 50 % of the
3-grams, and translations other than the battery's own.

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
RULE = ("attack_guard/v2: concatenated record text + base64 decodes; 6-gram or exact (4-5 word) fragment, whole-template "
        "wildcard (placeholder = 1..30 words, >= 4 fixed words), or fuzzy >= 0.5 of fixed word 3-grams (>= 3); "
        "NFKC, Cf removed, Cyrillic+Greek homoglyphs folded, casefold")
K_WORDS = 6
MIN_WORDS = 4
WILD_MAX = 30
WILD_MIN_FIXED = 4
FUZZY_N = 3
FUZZY_RATIO = 0.5
FUZZY_MIN_SHARED = 3
HELDOUT_V1_PATH = ROOT / "analysis" / "safety_heldout" / "heldout_v1_templates.json"
HELDOUT_V1_PROVENANCE = {"branch": "stack/clef/safety-heldout", "commit": "f2736185d8ddf93283ff5b842fe00b64a516ab0c",
                         "path": "analysis/safety_heldout/heldout_v1_templates.json",
                         "git_blob": "1af1d82ba81b663ee24a1e8f654dec5bc0ef00ff"}
EXIT_HIT = 3  # the job shells map 3 -> fail 33, as for held-out D1-B rows

_NONWORD = re.compile(r"[\W_]+", re.UNICODE)
_DOLLAR = re.compile(r"(\$[A-Z][A-Z0-9_]*)")
_B64RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
PLACEHOLDER = object()
# Cyrillic and Greek letters that render like Latin ones (the H08 transform and the usual homoglyph sets)
_HOMOGLYPH = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "һ": "h",
    "ԁ": "d", "ɡ": "g", "ԛ": "q", "ԝ": "w", "ү": "y", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "Ԁ": "D", "Ԛ": "Q",
    "Ԝ": "W",
    # Greek
    "α": "a", "β": "b", "γ": "y", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x", "ω": "w", "ϲ": "c", "ϳ": "j", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ϲ": "C",
})


class AttackTemplateInTraining(ValueError):
    """A training record (or a training-only augmentation template) contains an eval attack template."""


def normalise(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf").translate(_HOMOGLYPH)
    return [w for w in _NONWORD.sub(" ", t.casefold()).split() if w]


# ------------------------------------------------------------------------------------------------- templates
def _brace_parts(template: str) -> list:
    out: list = []
    for lit, field, _, _ in string.Formatter().parse(template):
        if lit:
            out.append(lit)
        if field is not None:
            out.append(PLACEHOLDER)
    return out


def _dollar_parts(template: str) -> list:
    return [PLACEHOLDER if _DOLLAR.fullmatch(p) else p for p in _DOLLAR.split(template) if p]


def _fragments(parts: list) -> list[str]:
    return [p for p in parts if p is not PLACEHOLDER and p.strip()]


def _brace_fragments(template: str) -> list[str]:
    return _fragments(_brace_parts(template))


def load_heldout_v1(path: str | Path = HELDOUT_V1_PATH) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def eval_template_parts() -> dict[str, list[list]]:
    """{"<set>:<name>": [parts, ...]}: each parts list is one contiguous piece (literal strings and PLACEHOLDER)."""
    from safety_eval.attacks import D1_TEMPLATES_A, D1_TEMPLATES_B, HIJACK_TEMPLATES

    out: dict[str, list[list]] = {}
    for set_name, tpls in (("official_hijack", HIJACK_TEMPLATES), ("d1a_steer", D1_TEMPLATES_A),
                           ("d1b_steer", D1_TEMPLATES_B)):
        for name, t in sorted(tpls.items()):
            out[f"{set_name}:{name}"] = [_brace_parts(t)]
    for name, spec in sorted(load_heldout_v1().items()):
        pieces = spec["text"].split("|") if spec.get("position") == "split" else [spec["text"]]
        if spec.get("payload"):
            pieces.append(spec["payload"])
        out[f"heldout_v1:{name}"] = [_dollar_parts(p) for p in pieces]
    return out


@lru_cache(maxsize=1)
def eval_templates() -> dict[str, list[str]]:
    """{"<set>:<name>": [fragment, ...]} for every eval attack template."""
    return {k: [f for parts in pieces for f in _fragments(parts)] for k, pieces in eval_template_parts().items()}


def eval_templates_sha256() -> str:
    """Identity of the guard: the rule, its parameters and every eval template (a change invalidates every pin)."""
    blob = json.dumps({"rule": RULE, "templates": eval_templates(),
                       "wildcards": {k: [[p if p is not PLACEHOLDER else None for p in parts] for parts in pieces]
                                     for k, pieces in eval_template_parts().items()}},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class Matcher:
    """Gram tables of a template set {key: [fragment]}; ``hits_words(words)`` -> set of template keys. With ``parts``
    ({key: [parts]}) it also builds the wildcard patterns and the fuzzy 4-gram index."""

    def __init__(self, templates: dict[str, list[str]], k: int = K_WORDS, min_words: int = MIN_WORDS,
                 parts: dict[str, list[list]] | None = None):
        self.k, self.min_words = k, min_words
        self.tables: dict[int, dict[tuple[str, ...], set[str]]] = {}
        self.skipped: dict[str, list[str]] = {}
        self.fuzzy: dict[tuple[str, ...], set[str]] = {}
        self.fuzzy_total: dict[str, int] = {}
        for key, frags in templates.items():
            fz: set[tuple[str, ...]] = set()
            for frag in frags:
                w = normalise(frag)
                fz.update(tuple(w[i:i + FUZZY_N]) for i in range(len(w) - FUZZY_N + 1))
                if len(w) < min_words:
                    self.skipped.setdefault(key, []).append(frag)
                    continue
                size = min(k, len(w))
                tab = self.tables.setdefault(size, {})
                for i in range(len(w) - size + 1):
                    tab.setdefault(tuple(w[i:i + size]), set()).add(key)
            if parts is not None and len(fz) >= FUZZY_MIN_SHARED:
                self.fuzzy_total[key] = len(fz)
                for g in fz:
                    self.fuzzy.setdefault(g, set()).add(key)
        self.firsts = {size: {g[0] for g in tab} for size, tab in self.tables.items()}
        self.fuzzy_firsts = {g[0] for g in self.fuzzy}
        self.wild: list[tuple[str, frozenset[str], re.Pattern]] = []
        for key, pieces in (parts or {}).items():
            for p in pieces:
                toks, fixed = [], []
                for part in p:
                    if part is PLACEHOLDER:
                        if toks and toks[-1] != "W":
                            toks.append("W")
                    else:
                        w = normalise(part)
                        if w:
                            toks.append(w)
                            fixed += w
                while toks and toks[0] == "W":
                    toks.pop(0)
                while toks and toks[-1] == "W":
                    toks.pop()
                if len(fixed) < WILD_MIN_FIXED or not any(t == "W" for t in toks):
                    continue  # all-literal pieces are already covered (exactly) by the gram method
                rx = " ".join(r"\S+(?: \S+){0,%d}?" % (WILD_MAX - 1) if t == "W" else " ".join(map(re.escape, t))
                              for t in toks)
                self.wild.append((key, frozenset(fixed), re.compile(r"(?:^| )" + rx + r"(?= |$)")))

    @property
    def n_grams(self) -> int:
        return sum(len(t) for t in self.tables.values())

    def gram_hits(self, w: list[str]) -> set[str]:
        out: set[str] = set()
        for size, tab in self.tables.items():
            fs = self.firsts[size]
            for i in range(len(w) - size + 1):
                if w[i] in fs:
                    keys = tab.get(tuple(w[i:i + size]))
                    if keys:
                        out |= keys
        return out

    def wildcard_hits(self, w: list[str], vocab: set[str] | None = None) -> set[str]:
        if not self.wild:
            return set()
        vocab = set(w) if vocab is None else vocab
        text = None
        out: set[str] = set()
        for key, fixed, rx in self.wild:
            if key in out or not fixed <= vocab:
                continue
            text = " ".join(w) if text is None else text
            if rx.search(text):
                out.add(key)
        return out

    def fuzzy_hits(self, w: list[str]) -> set[str]:
        if not self.fuzzy:
            return set()
        seen: dict[str, set[tuple[str, ...]]] = {}
        ff = self.fuzzy_firsts
        for i in range(len(w) - FUZZY_N + 1):
            if w[i] in ff:
                g = tuple(w[i:i + FUZZY_N])
                for key in self.fuzzy.get(g, ()):
                    seen.setdefault(key, set()).add(g)
        return {k for k, gs in seen.items()
                if len(gs) >= FUZZY_MIN_SHARED and len(gs) >= FUZZY_RATIO * self.fuzzy_total[k]}

    def hits_words(self, w: list[str]) -> set[str]:
        out = self.gram_hits(w) | self.wildcard_hits(w)
        return out | {f"{k}~fuzzy" for k in self.fuzzy_hits(w) - out}

    def hits(self, text: str) -> set[str]:
        return self.hits_words(normalise(text))


@lru_cache(maxsize=1)
def eval_matcher() -> Matcher:
    return Matcher(eval_templates(), parts=eval_template_parts())


def template_key(hit: str) -> str:
    return hit.removesuffix("~fuzzy")


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
    """Every string a training row can show the model, in order: keys and values of state and questions (of the
    nested ``request`` for train-format records)."""
    src = rec["request"] if isinstance(rec.get("request"), dict) else rec
    yield from _strings({"state": src.get("state"), "questions": src.get("questions")})


def record_texts(rec: dict[str, Any]) -> list[list[str]]:
    """Normalised word lists to scan: the concatenation of every segment, then each base64 decode on its own."""
    segs = list(record_segments(rec))
    out = [[w for s in segs for w in normalise(s)]]
    for s in segs:
        if len(s) >= 16:
            out += [normalise(d) for d in _b64_decoded(s)]
    return out


def record_hits(rec: dict[str, Any], matcher: Matcher | None = None) -> set[str]:
    m = matcher or eval_matcher()
    out: set[str] = set()
    for w in record_texts(rec):
        out |= m.hits_words(w)
    exact = {h for h in out if not h.endswith("~fuzzy")}
    return exact | {h for h in out if h.endswith("~fuzzy") and template_key(h) not in exact}



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

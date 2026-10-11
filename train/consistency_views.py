"""Semantics-preserving TRAIN-row views for consistency training (docs/CONSISTENCY.md). Stdlib only, CPU, no model.

A view is the same record and question with the SURFACE changed and the decision unchanged: same option ids, same
criteria order (so the same shim labels), same gold label, same question type. Three families, one per view:

  ``format``      whitespace / markup of the STATE's string leaves (markdown <-> plain, bullet styles, line breaks,
                  whitespace normalisation) and, for a flat dict state, a JSON -> "key: value" lines rendering
  ``instruction`` the question's instructions reworded: a frame from ``INSTRUCTION_FRAMES`` and/or verb synonyms from
                  ``INSTRUCTION_SYNONYMS``
  ``option``      the option descriptions (choice values, score levels, noul true/false text) lightly paraphrased:
                  ``OPTION_SYNONYMS``, '&' <-> 'and', first-letter case

Determinism: every choice is a pure function of (seed, item id) through sha256 (``_rng``); no global RNG, no clock, no
model, no network. ``view_record(rec, qid, seed, view_frac, families)`` returns ``(view record, family, ops)`` or
None (the item is not sampled, no transform applies, or the question is malformed).

INDEPENDENCE FROM THE TEST OPERATORS. jevkit/perturb.py (the 8 flip-rate operators, frozen evaluation) is never
imported here, and none of its tables or rules are reused: its rewrite table is contractions / "e.g." / "i.e." /
"approximately" / "in order to" (none of those words appear in our tables, and we never contract or expand); its three
frames ("Task: ...", "Decide the following. ...", "... Answer from the options given.") are not in our bank; it
toggles the final period of option descriptions (we never touch final punctuation); its whitespace operator DOUBLES
runs of spaces and ADDS a leading and trailing newline (we only collapse spaces, strip line ends, change line-break
spacing and bullets); its markup operator quotes string states with "> " and REVERSES dict-state key order (we never
quote, and the dict rendering keeps the key order). tests/test_consistency.py checks that no view of a probe set
equals any jevkit perturbation of the same probe, and that this module does not import jevkit.

Training only: callers pass the training list (train.run_cell.attach_consistency_views); the Trainer refuses eval items
that carry a view. The views do not touch state paraphrase, option order or option keys (the reorder ops are removed
by construction in the sym-fork layout).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from typing import Any, Callable

VERSION = "cv1"
FAMILIES = ("format", "instruction", "option")

#: instruction frames ("{}" = the original instruction text). Our own bank (none is a jevkit frame).
INSTRUCTION_FRAMES = (
    "Based on the record: {}",
    "{} Use the information in the record.",
    "Read the record, then answer: {}",
    "Question about the record. {}",
    "{} (Refer to the record above.)",
    "Instruction: {}",
    "{} Respond with the best-matching option.",
    "Using only what the record says: {}",
)

#: instruction verb / adjective synonyms (whole word, case of the first letter kept); applied left to right, one pass
INSTRUCTION_SYNONYMS = (
    ("choose", "select"), ("select", "pick"), ("pick", "choose"),
    ("determine", "decide"), ("decide", "determine"),
    ("appropriate", "suitable"), ("suitable", "appropriate"),
    ("classify", "categorize"), ("categorize", "classify"),
    ("identify", "find"), ("indicate", "state"),
    ("correct", "right"), ("most likely", "likeliest"),
)

#: option-description synonyms (whole word, case kept). Plain, meaning-preserving swaps; no negation, number, severity
#: or contraction words
OPTION_SYNONYMS = (
    ("purchase", "buy"), ("buy", "purchase"),
    ("issue", "problem"), ("problem", "issue"),
    ("assist", "help"), ("information", "details"), ("details", "information"),
    ("incorrect", "wrong"), ("wrong", "incorrect"),
    ("required", "needed"), ("needed", "required"),
    ("regarding", "concerning"), ("concerning", "regarding"),
    ("modify", "change"), ("contains", "includes"), ("includes", "contains"),
    ("begin", "start"), ("start", "begin"), ("quick", "fast"), ("fast", "quick"),
    ("large", "big"), ("big", "large"), ("customer", "client"), ("client", "customer"),
    ("e-mail", "email"), ("email", "e-mail"), ("show", "display"), ("display", "show"),
)

_BULLET = re.compile(r"^(\s*)([-*])\s+", re.M)
_NUMBERED = re.compile(r"^(\s*)(\d{1,3})[.)]\s+", re.M)
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_HEADER = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
_KEYLINE = re.compile(r"^([A-Z][A-Za-z0-9 /_-]{0,40}):(?=\s)", re.M)


def _rng(seed: int, item_id: str, salt: str) -> random.Random:
    h = hashlib.sha256(f"{VERSION}\x1f{seed}\x1f{item_id}\x1f{salt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def sampled(seed: int, item_id: str, view_frac: float) -> bool:
    """Is this item one of the ``view_frac`` share that get a view (a sha256 threshold, independent of the order)."""
    h = hashlib.sha256(f"{VERSION}\x1fsample\x1f{seed}\x1f{item_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64 < view_frac


def _map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}  # keys and key ORDER kept
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    return obj


def _swap(text: str, table) -> str:
    """One left-to-right pass of whole-word swaps; a replaced span is never rewritten again (no a->b->a chains)."""
    alts = sorted({a for a, _ in table}, key=len, reverse=True)
    sub = {a.lower(): b for a, b in table}  # every source word is listed once
    pat = re.compile(r"(?<![\w'-])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w'-])", re.I)

    def rep(m: re.Match) -> str:
        s = m.group(0)
        b = sub[s.lower()]
        if s.isupper() and len(s) > 1:
            return b.upper()
        return b[0].upper() + b[1:] if s[:1].isupper() else b

    return pat.sub(rep, text)


# ---------------------------------------------------------------------------------------------- format (state)
def _bullets(s: str) -> str:
    return _BULLET.sub(lambda m: m.group(1) + ("* " if m.group(2) == "-" else "- "), s)


def _numbered(s: str) -> str:
    """'1. ' list markers -> '1) ' (or, when the text has none, '1) ' -> '1. ')."""
    end = ") " if re.search(r"^\s*\d{1,3}\.\s", s, re.M) else ". "
    return _NUMBERED.sub(lambda m: f"{m.group(1)}{m.group(2)}{end}", s)


def _md_to_plain(s: str) -> str:
    s = _BOLD.sub(lambda m: m.group(1) or m.group(2), s)
    return _HEADER.sub(lambda m: m.group(1).rstrip(":") + ":", s)


def _plain_to_md(s: str) -> str:
    return _KEYLINE.sub(lambda m: f"**{m.group(1)}:**", s)


def _para_breaks(s: str) -> str:
    if "\n\n" in s:
        return re.sub(r"\n(?:[ \t]*\n)+", "\n", s)
    return s.replace("\n", "\n\n") if "\n" in s else s


def _normalise_ws(s: str) -> str:
    s = s.replace("\t", "  ")
    s = re.sub(r"(?<=\S) {2,}(?=\S)", " ", s)  # collapse interior runs (never doubles them)
    s = re.sub(r"[ \t]+(?=\n)", "", s)  # strip line ends
    return s.strip()


FORMAT_OPS: dict[str, Callable[[str], str]] = {
    "bullets": _bullets, "numbered": _numbered, "md_to_plain": _md_to_plain, "plain_to_md": _plain_to_md,
    "para_breaks": _para_breaks, "normalise_ws": _normalise_ws,
}


def _dict_lines(state: Any) -> str | None:
    """A flat dict of scalars (>= 2 keys) as "key: value" lines, keys in their ORIGINAL order; else None."""
    if not isinstance(state, dict) or len(state) < 2:
        return None
    if not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) and "\n" not in str(v)
               for k, v in state.items()):
        return None
    return "\n".join(f"{k}: {json.dumps(v) if isinstance(v, bool) else v}" for k, v in state.items())


def _format_view(req: dict, rng: random.Random) -> tuple[Any, list[str]] | None:
    state = req.get("state")
    cands = []
    for name, fn in FORMAT_OPS.items():
        new = _map_strings(state, fn)
        if new != state:
            cands.append((name, fn))
    dl = _dict_lines(state)
    if not cands and dl is None:
        return None
    if dl is not None and (not cands or rng.random() < 0.3):
        return dl, ["dict_lines"]
    rng.shuffle(cands)
    ops, new = [], state
    for name, fn in cands[: 1 + (rng.random() < 0.4)]:  # one op, sometimes two
        nxt = _map_strings(new, fn)
        if nxt != new:
            new, ops = nxt, ops + [name]
    return (new, ops) if new != state else None


# ---------------------------------------------------------------------------------------------- instruction
def _instruction_view(q: dict, rng: random.Random) -> tuple[Any, list[str]] | None:
    ins = q.get("instructions")
    if isinstance(ins, str) and ins.strip():
        ops, text = [], ins.strip()
        if rng.random() < 0.6:
            sw = _swap(text, INSTRUCTION_SYNONYMS)
            if sw != text:
                text, ops = sw, ["synonyms"]
        if not ops or rng.random() < 0.7:
            k = rng.randrange(len(INSTRUCTION_FRAMES))
            text, ops = INSTRUCTION_FRAMES[k].format(text), ops + [f"frame{k}"]
        return (text, ops) if text != ins else None
    if isinstance(ins, (dict, list)):
        new = _map_strings(ins, lambda s: _swap(s, INSTRUCTION_SYNONYMS))
        return (new, ["synonyms"]) if new != ins else None
    return None


# ---------------------------------------------------------------------------------------------- option descriptions
def _desc_paraphrase(s: str, rng: random.Random, case: bool) -> str:
    out = _swap(s, OPTION_SYNONYMS)
    if " & " in out:
        out = out.replace(" & ", " and ")
    elif rng.random() < 0.3:
        out = re.sub(r"(?<=\w) and (?=\w)", " & ", out, count=1)
    if case and out[:1].isalpha() and not out.split(" ", 1)[0].isupper():
        out = (out[0].lower() if out[0].isupper() else out[0].upper()) + out[1:]
    return out


def _option_view(q: dict, rng: random.Random) -> tuple[Any, list[str]] | None:
    crit = q.get("criteria")
    case = rng.random() < 0.5
    fn = lambda s: _desc_paraphrase(s, rng, case)  # noqa: E731
    t = q.get("type")
    if t == "choice" and isinstance(crit, dict):
        new = {k: (fn(v) if isinstance(v, str) else v) for k, v in crit.items()}  # keys and order kept
    elif t == "score" and isinstance(crit, list):
        new = [fn(v) if isinstance(v, str) else v for v in crit]
    elif t == "noul" and isinstance(crit, dict):
        new = {k: (fn(v) if isinstance(v, str) and v else v) for k, v in crit.items()}
    else:
        return None
    return (new, ["option_paraphrase"] + (["case"] if case else [])) if new != crit else None


# ---------------------------------------------------------------------------------------------- the view
def view_record(rec: dict[str, Any], qid: str, seed: int = 0, view_frac: float = 1.0,
                families: tuple[str, ...] = FAMILIES) -> tuple[dict[str, Any], str, list[str]] | None:
    """(view record, family, ops) for question ``qid`` of ``rec``, or None. The view keeps the record's other questions
    and its labels; its id is ``<id>#cv:<qid>``."""
    if any(f not in FAMILIES for f in families) or not families:
        raise ValueError(f"families must be a non-empty subset of {FAMILIES}")
    if not 0.0 <= view_frac <= 1.0:
        raise ValueError("view_frac must be in [0, 1]")
    item_id = f"{rec.get('id', '?')}/{qid}"
    if not sampled(seed, item_id, view_frac):
        return None
    req = rec["request"] if rec.get("request") is not None else rec
    q = (req.get("questions") or {}).get(qid)
    if not isinstance(q, dict) or qid not in (rec.get("labels") or {}):
        return None
    rng = _rng(seed, item_id, "family")
    order = list(families)
    rng.shuffle(order)
    for fam in order:
        r = _rng(seed, item_id, fam)
        got = _format_view(req, r) if fam == "format" else \
            _instruction_view(q, r) if fam == "instruction" else _option_view(q, r)
        if got is None:
            continue
        value, ops = got
        view = copy.deepcopy(rec)
        vreq = view["request"] if view.get("request") is not None else view
        if fam == "format":
            vreq["state"] = value
        elif fam == "instruction":
            vreq["questions"][qid]["instructions"] = value
        else:
            vreq["questions"][qid]["criteria"] = value
        view["id"] = f"{rec.get('id', '?')}#cv:{qid}"
        check_label_preserved(rec, view, qid)
        return view, fam, ops
    return None


def attach_views(items: list, records: dict[str, dict], enc, layout: str, template, guard, keep, seed: int = 0,
                 view_frac: float = 0.25, families: tuple[str, ...] = FAMILIES) -> dict:
    """Attach ``cview`` to the sampled TRAINING items in place (``keep(row) -> bool`` = the cell's K / T filter). The
    view row goes through ``items_from_records`` with the same guard, encoder, template and layout as its base, and
    must keep the option ids and the gold option. Returns coverage counts for cell.json."""
    from train.data import items_from_records

    by_item = {}
    for rec in records.values():
        for qid in rec["labels"]:
            item_id = f"{rec['id']}/{qid}"
            if item_id in by_item:
                raise ValueError(f"duplicate expanded Item id {item_id!r} in consistency-view records")
            by_item[item_id] = (rec, qid)
    counts = {"with_view": 0, "not_sampled": 0, "no_transform": 0, "view_filtered": 0,
              "by_family": {f: 0 for f in families}}
    for it in items:
        rec, qid = by_item[it.id]
        if not sampled(seed, it.id, view_frac):
            counts["not_sampled"] += 1
            continue
        got = view_record(rec, qid, seed, view_frac, families)
        if got is None:
            counts["no_transform"] += 1
            continue
        vrec, fam, _ = got
        try:
            vit = items_from_records([{**vrec, "labels": {qid: vrec["labels"][qid]}}], enc, layout=layout,
                                     template=template, guard=guard)
        except ValueError:  # e.g. a control-token refusal of the new surface: the base row stays single-view
            vit = []
        if len(vit) != 1 or not keep(vit[0].row):
            counts["view_filtered"] += 1
            continue
        v = vit[0]
        if v.row.option_ids != it.row.option_ids or v.label != it.label or v.row.qtype != it.row.qtype:
            raise ValueError(f"{it.id}: consistency view does not keep the option domain and the gold option")
        it.cview = v
        counts["with_view"] += 1
        counts["by_family"][fam] += 1
    return {"version": VERSION, "seed": seed, "view_frac": view_frac, "families": list(families), **counts}


def check_label_preserved(rec: dict, view: dict, qid: str) -> None:
    """Fail closed unless the view keeps the question type, the option keys in order (score: the level count), and
    the gold label."""
    a = (rec["request"] if rec.get("request") is not None else rec)["questions"][qid]
    b = (view["request"] if view.get("request") is not None else view)["questions"][qid]
    ca, cb = a.get("criteria"), b.get("criteria")
    keys = (lambda c: list(c) if isinstance(c, dict) else len(c) if isinstance(c, list) else None)
    if a.get("type") != b.get("type") or keys(ca) != keys(cb) or rec["labels"][qid] != view["labels"][qid]:
        raise ValueError(f"{rec.get('id')}/{qid}: consistency view changed the option domain or the label")

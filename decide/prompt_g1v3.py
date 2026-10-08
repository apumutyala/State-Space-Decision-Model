"""jrt-rwkv-g1-v3 family: the v2 vendor-compliant RWKV7 G1 readout plus long-catalog binding (R3), with optional
presentation permutation (v3p), top-m re-read (v3r) and contextual calibration (v3c).

Opt-in only (readout families ``rwkv7-hf-g1-v3*``, ``decide.backends.rwkv_g1_v3``). v2 (``decide.prompt.render_g1v2``)
is not modified, so its revision identity is unchanged. ``render_g1v3`` with ``catalog_chars=None``, the identity
presentation and v2's code table reproduces ``render_g1v2`` byte for byte (tests/test_rwkv_g1_v3.py).

v3 changes against v2:
  1. code table (``decide/data/rwkv_g1_v3_codes.json``, ``build_code_table``): A..Z, then the two-letter codes AA..ZZ in
     lexicographic order that are ONE token after "ANSWER:" in the pinned G1k tokenizer, minus English words and
     common abbreviations (WORDLIKE: " AN", " AS", " AT", " BE", " DO", " ER", " ES", " EU", " UK", " TV", ...), whose
     strong unigram priors would bias long catalogs. The table differs from v2's only past the 26th option.
  2. code key: a coded question with k >= CATALOG_MIN_K options gets, between the re-read option list and the
     "Reply with exactly one option code." line,

        QUESTION AGAIN: <the question's instruction (its last RESTATE_CHARS characters if longer)>
        CODE KEY (code = option):
        A = <short option label>
        ...

     RWKV7 is recurrent: on a 53/77/151-way catalog the code -> option binding of early options has to survive
     hundreds of option lines in the state. The block puts the question right before a compact one-line-per-code key,
     so the key is read knowing what is being looked for (JRT order: query, then items), and the key is the last
     thing before the answer slot. Labels: the option description for an opaque id ("option_17", "12"), else
     "<id>: <description>", whitespace-collapsed and cut to ``catalog_chars`` characters. The backend steps
     ``catalog_chars`` down (CATALOG_CHARS_LADDER) when the prompt would exceed the context budget and drops the block
     as the last resort.

Instruction fix (every v3* variant, and v2f = v2 + this fix only): v2's ``instr()`` renders a question's instruction
only when it is a string, so a dict-valued instruction (ToolRet {candidate, task}, BRIGHT) was replaced by the
question id and the candidate document never reached the model. Here a non-string instruction is rendered as
canonical JSON exactly as v1 does (``decide.schema.render``), blank lines still collapsed; None / "" still fall back
to the question id. ``fix_instr=False`` reproduces v2 byte for byte (the job-zhyvg control).

v3w: when a question's option ids are words (not opaque ids like "option_3" / "12") and the first token of
" <id>" after "ANSWER:" is unique among its options, the options are listed as "- <id>: <description>" (no letter
code, so never "A. A: ...") and each option is scored by the first token of its own id. Otherwise letters, as v3.
Generic rule only (``word_ids_ok``), no per-family switch.

v3p: the same prompt with each question's options PRESENTED in a deterministic pseudo-random order
(``presentation_order``): code A goes to the option at presentation position 0, and so on. Only the code
assignment and listing order change; the backend maps every code back to its option id, so the output is keyed by
option id in canonical order exactly as before. Without it, "A" is always the first option id in sorted order
(true / SUPPORTED / option_0), so the model's A-prior lands on the same option of every row.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .prompt import G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE, Labelled, g1_clean, render
from .schema import canonical_options, canonical_question_ids

G1V3_ID = "jrt-rwkv-g1-v3"
G1V3P_ID = "jrt-rwkv-g1-v3p"
G1V3R_ID = "jrt-rwkv-g1-v3r"
G1V3W_ID = "jrt-rwkv-g1-v3w"
G1V2F_ID = "jrt-rwkv-g1-v2f"
G1V3C_ID = "jrt-rwkv-g1-v3c"

# ----------------------------------------------------------------------------------------------------- code table
V3_CODES_PATH = Path(__file__).resolve().parent / "data/rwkv_g1_v3_codes.json"
V3_CODES_SCHEMA = "ssdm.rwkv-g1-v3-codes.v1"
V3_MAX_K = 255
#: two-letter English words and common abbreviations (unigram-heavy as " XX" tokens); excluded from the code table
WORDLIKE = frozenset("""
AD AH AI AM AN AS AT AW AX AY BE BI BO BY DA DE DO EH EL EM EN ER ES EU EX FA GO HA HE HI HM HO HR ID IF IN IO IS IT
JO KA LA LI LO MA ME MI MO MU MY NA NE NO NU OD OF OH OI OK OM ON OP OR OS OU OW OX OY PA PC PE PI PM PO PR QA QI RE
SH SI SO TA TI TO TV UH UK UM UN UP US UT VS WE WO XI XU YA YE YO ZA
""".split())


def code_candidates() -> list[str]:
    """A..Z, then AA..ZZ lexicographic, word-like ones removed (single-token filtering happens with the tokenizer)."""
    import string

    two = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    return list(string.ascii_uppercase) + [c for c in two if c not in WORDLIKE]


def build_code_table(encode) -> list[dict[str, Any]]:
    """[{code, token_id}] for the first V3_MAX_K candidates that are one token after "ANSWER:" (``encode`` = the pinned
    G1k tokenizer, text -> ids). Used to (re)generate decide/data/rwkv_g1_v3_codes.json; the backend re-proves it."""
    ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"
    base = encode(ctx)
    out = []
    for c in code_candidates():
        ids = encode(ctx + " " + c)
        if ids[: len(base)] == base and len(ids) == len(base) + 1 and ids[-1] not in {t["token_id"] for t in out}:
            out.append({"code": c, "token_id": ids[-1]})
        if len(out) == V3_MAX_K:
            break
    if len(out) != V3_MAX_K:
        raise ValueError(f"only {len(out)} single-token codes")
    return out


def table_sha256(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def load_v3_codes(path: Path = V3_CODES_PATH) -> tuple[list[dict[str, Any]], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["codes"]
    sha = table_sha256(entries)
    if data.get("schema") != V3_CODES_SCHEMA or data.get("table_sha256") != sha or len(entries) != V3_MAX_K:
        raise ValueError("v3 code table identity differs from its recorded sha256")
    if [e["code"] for e in entries] != [c for c in code_candidates() if c in {e["code"] for e in entries}]:
        raise ValueError("v3 code table is not the ordered candidate list")
    return entries, sha


# ----------------------------------------------------------------------------------------------- presentation order
def presentation_order(q: dict[str, Any], qid: str) -> list[int]:
    """Deterministic pseudo-random presentation order of a question's canonical options (v3p). Seeded by the question
    id, its type and its option ids only (never by gold, scores or other rows): sha256 of those keys, then each option
    is ranked by sha256(seed + option id). The question id alone is not enough: most benchmarks name their only
    question "q"/"q1" in every row, which would give every row the same order."""
    oids = [o for o, _ in canonical_options(q)]
    seed = hashlib.sha256(json.dumps([qid, q.get("type"), oids], ensure_ascii=True).encode("ascii")).hexdigest()
    keys = [hashlib.sha256((seed + "\x1f" + o).encode("utf-8", "surrogatepass")).hexdigest() for o in oids]
    return sorted(range(len(oids)), key=lambda j: (keys[j], j))


def identity_order(q: dict[str, Any]) -> list[int]:
    return list(range(len(canonical_options(q))))


# --------------------------------------------------------------------------------------------------------- prompt
CATALOG_MIN_K = 9
CATALOG_CHARS_LADDER: tuple[int | None, ...] = (64, 24, None)
RESTATE_CHARS = 1200
_OPAQUE_ID = re.compile(r"(?:option|opt|choice|candidate|item|label)?[_\- ]?\d+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _one_line(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(n - 3, 1)].rstrip() + "..."


def catalog_label(oid: str, desc: Any, n: int) -> str:
    d = _one_line(desc if isinstance(desc, str) else render(desc)) if desc is not None else ""
    if not d:
        text = _one_line(oid)
    elif _OPAQUE_ID.fullmatch(oid.strip()):
        text = d
    else:
        text = f"{_one_line(oid)}: {d}"
    return _cut(text, n)


def _g1_text(value: Any) -> str:
    return g1_clean(value if isinstance(value, str) else render(value))


def instruction_text(q: dict[str, Any], qid: str, fix: bool = True) -> str:
    """The rendered instruction. fix=True: any non-empty value (dict/list as canonical JSON, as v1); fix=False: v2's
    behaviour (only a non-empty string, else the question id)."""
    v = q.get("instructions")
    if fix:
        return g1_clean(qid) if v is None or v == "" else _g1_text(v)
    return _g1_text(v) if isinstance(v, str) and v != "" else g1_clean(qid)


def is_opaque_id(oid: str) -> bool:
    return bool(_OPAQUE_ID.fullmatch(oid.strip()))


def word_ids_ok(q: dict[str, Any], first_token) -> bool:
    """v3w rule: every option id is non-opaque and the first answer token of " <id>" (``first_token(id)``) is unique
    among the question's options."""
    oids = [o for o, _ in canonical_options(q)]
    if len(oids) < 2 or any(is_opaque_id(o) or not o.strip() for o in oids):
        return False
    toks = [first_token(o) for o in oids]
    return None not in toks and len(set(toks)) == len(toks)


def question_codes(q: dict[str, Any], order: list[int], table: list[str]) -> list[str] | None:
    """Code of each CANONICAL option (code table position = presentation position); None when k > len(table)."""
    k = len(canonical_options(q))
    if k > len(table):
        return None
    codes = [""] * k
    for pos, j in enumerate(order):
        codes[j] = table[pos]
    return codes


def _option_lines(q: dict[str, Any], order: list[int], codes: list[str] | None) -> list[Labelled]:
    """codes None = "- <id>: <desc>" lines (v2's fallback for k > table, and v3w word ids)."""
    opts = canonical_options(q)
    seg: list[Labelled] = []
    for j in order:
        oid, desc = opts[j]
        seg.append(("\n" + (f"{codes[j]}. " if codes is not None else "- "), "template"))
        seg.append((g1_clean(oid), "data"))
        if desc is not None:
            seg += [(": ", "template"), (_g1_text(desc), "data")]
    return seg


def _restated(q: dict[str, Any], qid: str) -> str:
    text = instruction_text(q, qid, True)
    return text if len(text) <= RESTATE_CHARS else "..." + text[-RESTATE_CHARS:]


def catalog_block(q: dict[str, Any], qid: str, order: list[int], codes: list[str], n: int) -> str:
    opts = canonical_options(q)
    lines = [f"\nQUESTION AGAIN: {_restated(q, qid)}", "\nCODE KEY (code = option):"]
    for j in order:
        oid, desc = opts[j]
        lines.append(f"\n{codes[j]} = {catalog_label(oid, desc, n)}")
    return "".join(lines)


def wants_catalog(q: dict[str, Any], table: list[str]) -> bool:
    return CATALOG_MIN_K <= len(canonical_options(q)) <= len(table)


def build_prompt_g1v3(request: dict[str, Any], qid: str, table: list[str], catalog_chars: int | None = 64,
                      permute: bool = False, fix_instr: bool = True,
                      word_qids: frozenset[str] = frozenset()) -> list[Labelled]:
    """v2's layout (decide.prompt.build_prompt_g1v2) with a code table, the instruction fix, an optional
    presentation order, v3w word-id questions (``word_qids``) and the code key for the asked question."""
    questions = request["questions"]
    order_ids = canonical_question_ids(questions)
    orders = {k: (presentation_order(questions[k], k) if permute else identity_order(questions[k])) for k in order_ids}
    codes = {k: (None if k in word_qids else question_codes(questions[k], orders[k], table)) for k in order_ids}

    def instr(k: str) -> list[Labelled]:
        return [(instruction_text(questions[k], k, fix_instr), "data")]

    seg: list[Labelled] = [(G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", "template")]
    for i, k in enumerate(order_ids, start=1):
        if i > 1:
            seg.append(("\n", "template"))
        seg += [(f"FIELD {i}\nID: ", "template"), (g1_clean(k), "data"),
                (f"\nTYPE: {questions[k]['type']}\nINSTRUCTION: ", "template")]
        seg += instr(k)
        seg.append(("\nOPTIONS:", "template"))
        seg += _option_lines(questions[k], orders[k], codes[k])
        seg.append(("\nEND FIELD", "template"))
    seg.append(("\nEND SCHEMA\nSTATE:\n", "template"))
    seg.append((_g1_text(request["state"]), "data"))
    q = questions[qid]
    coded = codes[qid] is not None or qid in word_qids
    seg += [(f"\nEND STATE\nQUESTION (re-read FIELD {order_ids.index(qid) + 1}):\nID: ", "template"),
            (g1_clean(qid), "data"), (f"\nTYPE: {q['type']}\nINSTRUCTION: ", "template")]
    seg += instr(qid)
    seg.append(("\nOPTIONS:", "template"))
    seg += _option_lines(q, orders[qid], codes[qid])
    if catalog_chars is not None and codes[qid] is not None and wants_catalog(q, table):
        seg.append((catalog_block(q, qid, orders[qid], codes[qid], catalog_chars), "data"))
    reply = "id." if qid in word_qids or not coded else "code."
    seg.append(("\nReply with exactly one option " + reply + G1V2_TEMPLATE.tail, "template"))
    return [(t, lab) for t, lab in seg if t]


def render_g1v3(request: dict[str, Any], qid: str, table: list[str], catalog_chars: int | None = 64,
                permute: bool = False, fix_instr: bool = True, word_qids: frozenset[str] = frozenset()) -> str:
    """The v3 prompt string; blank lines are collapsed over the joined user turn exactly as in ``render_g1v2``."""
    text = "".join(t for t, _ in build_prompt_g1v3(request, qid, table, catalog_chars, permute, fix_instr, word_qids))
    head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
    body = text[len(head):len(text) - len(G1V2_TEMPLATE.tail)]
    return head + g1_clean(body) + G1V2_TEMPLATE.tail


# ------------------------------------------------------------------------------------------------- v3r top-m re-read
REREAD_MIN_K = 3
REREAD_LABEL_CHARS = 160


def reread_m(k: int) -> int:
    """Shortlist size: 0 (no re-read) for k < 3, 2 for k = 3, 3 for k >= 4."""
    return 0 if k < REREAD_MIN_K else (2 if k == 3 else 3)


def reread_suffix(q: dict[str, Any], shortlist_idx: list[int], codes: list[str]) -> str:
    """Text appended right after the first answer context ("...ANSWER:") for the second pass: the shortlisted options
    (canonical indices, each with its code from ``codes``) in canonical order, never in score order, so the listing
    does not leak the rank; then a second answer slot. ``shortlist_idx`` must come from the model's own first-pass
    scores (the backend asserts it)."""
    opts = canonical_options(q)
    lines = ["\nSHORTLIST (re-read; code = option):"]
    for j in sorted(shortlist_idx):
        oid, desc = opts[j]
        lines.append(f"\n{codes[j]} = {catalog_label(oid, desc, REREAD_LABEL_CHARS)}")
    lines.append("\nFINAL ANSWER:")
    return g1_clean("".join(lines))


# ------------------------------------------------------------------------------------------- contextual calibration
#: Zhao et al. 2021's content-free inputs; the prior is the mean of their per-option distributions
CONTENT_FREE_INPUTS = ("N/A", "", "[MASK]")
#: calibrate only label spaces of this size or smaller (the paper's tasks have 2-14 labels; on 53-151-way catalogs a
#: content-free prior has tiny entries whose inverse would dominate the readout)
CALIBRATE_MAX_K = 16


def content_free_request(request: dict[str, Any], filler: str) -> dict[str, Any]:
    """The same request with its content removed: the state and every question's instruction are replaced by
    ``filler``; question ids, types and options (the label space) are kept, so the prompt layout is the same."""
    qs = {}
    for k in canonical_question_ids(request["questions"]):
        q = dict(request["questions"][k])
        q["instructions"] = filler
        qs[k] = q
    return {**{k: v for k, v in request.items() if k not in ("state", "questions")}, "state": filler, "questions": qs}

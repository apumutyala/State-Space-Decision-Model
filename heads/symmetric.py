"""Reorder- and relabel-invariant decision rows BY CONSTRUCTION (layouts ``sym-fork`` and ``sym-fork-ids``).

The shared prefix holds the state, the question and the NUMBER of options, never an option. Each option is scored as
its own continuation forked from that prefix (``heads.backbone`` fork: recompute, or one prefix pass + K cache copies),
so the feature of option k is a function of (prefix, option k's text) alone. The decision head must treat the option
axis symmetrically (``FinalStateFork``: per-option MLP + the masked mean over options; any DeepSets / position-free
attention head qualifies), and the typed choice output is a softmax. Composition: the choice distribution is
permutation-EQUIVARIANT in the options, and since the option's id string never enters the model (``sym-fork``: the
continuation renders the description only; the id is mapped back outside the model through ``LayoutRow.option_ids``),
it is INVARIANT under any bijective relabeling of the ids. Statement, assumptions and failure cases: docs/PROOF_INVARIANCE.md.

Two extra measures make the property exact in floating point, not only up to reduction order:

* content-canonical option axis (choice questions): the axis is sorted by the continuation token ids, so the row --
  prefix ids, continuation ids, their order, hence every batch composition, padding width and reduction order -- is a
  function of the MULTISET of option texts. A permuted / relabeled request yields the same row, bit for bit, except
  ``option_ids``. Noul keeps (true, false); score keeps the level order (ordinal: its order is part of the question).
* the description-free fallback is explicit: an option whose description is missing or empty is rendered by its id
  (that id IS its content; ``meta["id_as_content"]`` lists them). Relabel invariance is claimed only for options with a
  description; reorder invariance holds for every option.

``sym-fork-ids`` is the ablation that renders "<id>: <description>" in the continuation: still reorder-equivariant,
NOT relabel-invariant (tests/test_sym_invariance.py shows both).

``sym-fork-typed`` (JEV3-TYPED; docs/guarantees/REORDER_INVARIANCE.md section 2T): choice and noul rows are the
``sym-fork`` rows byte for byte. A SCORE row renders each level's content as ``level v: <rubric>`` (the typed rendering
R_score; byte for byte the evidence lane's ``level v of K: <rubric>``, TYPED_SCORE_RENDERING), the axis is sorted by the
level value v PARSED FROM THAT CONTENT (never by handle, never by display order; an
unparseable level value or a value set that is not exactly 0..K-1 fails closed), and the full scale (every level's
content, in that level order) is appended to the question text in the shared prefix. So the prefix and every
continuation are functions of the multiset of contents: a display permutation and a handle relabel give a bitwise
identical row except ``option_ids`` (Theorem 2T (i), (ii)); renaming a level value changes the content (2T (iii), not a
symmetry). ``build_row_sym(score_options=...)`` takes an explicit display list of (handle, content) for that check.

Formats: a shim template (``jrt-h2o-shim:<family>``) gets the Qwen chat turn of the H2O shim with a symmetric user turn
(``SHIM_SYM_SYSTEM``) tokenized by the shim spec (special tokens honoured, request text that encodes to a control token
refused) segment by segment: template text and request text (state, question, each option) never share a token
(``_Pieces``; G15 under BPE); a G1 template (``jrt-rwkv-g1-*``) gets the RWKV7-G1 vendor format (``g1_pieces``, the row
encoder's whole-string tokens, id 0 refused); any other template gets ``head + body`` / ``" <option>" + tail``.
Continuations: ``sym-fork`` (default) ends each option with the user-turn terminator (Qwen ``<|im_end|>``: ONE token;
G1 "\\n\\n") and the head reads that position, which makes the layout cost-neutral against the g1v2 shim prompt on
JevBench (docs/PROOF_INVARIANCE.md); ``sym-fork-chat`` appends the full answer turn instead (+10 tokens per option).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

SYM_LAYOUT = "sym-fork"  # lean continuation: " <option>" + the turn terminator (<|im_end|> / G1 "\n\n")
SYM_CHAT_LAYOUT = "sym-fork-chat"  # the full answer turn after each option (+10 tokens per option on Qwen)
SYM_IDS_LAYOUT = "sym-fork-ids"  # ablation: "<id>: <option>" (lean tail)
SYM_TYPED_LAYOUT = "sym-fork-typed"  # JEV3-TYPED: sym-fork + score levels as content + the scale in the prefix (2T)
SYM_LAYOUTS = (SYM_LAYOUT, SYM_CHAT_LAYOUT, SYM_IDS_LAYOUT, SYM_TYPED_LAYOUT)
#: v2: lean (turn-terminated) continuation is the sym-fork default; G1 format. v3: trust-segment tokenization (the
#: state, the question and each option text are tokenized apart from the template, so no BPE merge crosses a trust
#: boundary: G15's trusted token list is independent of untrusted text; byte-level encoders give the same ids as v2)
SYM_VERSION = "sym-fork-v3"
#: train.run_cell.keep_item: longest continuation (option text + its tail) a sym row may carry
SYM_MAX_CONT = 128
G1_SYM_SYSTEM = ("Read the state, then the question, then ONE candidate answer. Candidates are shown one at a time and "
                 "their order means nothing; judge this candidate on its own.")
G1_TURN_END = "\n\n"  # the RWKV7-G1 turn separator (vendor format: no blank line inside a turn)

SHIM_SYM_SYSTEM = ("You are a decision engine. You read a record and one question about it, then ONE candidate answer. "
                   "Candidates are shown one at a time and their order means nothing; judge this candidate on its own.")
PLAIN_SYM_SYSTEM = "Read the state, then the question, then one candidate answer; judge the candidate on its own."
CHATML = {"system": "<|im_start|>system\n{content}<|im_end|>\n", "user_open": "<|im_start|>user\n",
          "user_close": "<|im_end|>\n", "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}
PREFILL = "Answer:"


def desc_text(v: Any) -> str:
    """Option description: a string verbatim, anything else as JSON (the shim's ``desc_text``)."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def option_text(oid: str, desc: Any, with_id: bool) -> tuple[str, bool]:
    """(text the model sees for this option, id_as_content). ``with_id`` only for the ``sym-fork-ids`` ablation."""
    has = desc is not None and desc_text(desc).strip() != ""
    body = desc_text(desc).strip() if has else str(oid)
    if with_id:
        return (f"{oid}: {body}" if has else str(oid)), not has
    return body, not has


_LEVEL_RE = None


#: the evidence lane's score-invariance rendering (claude/ev-score-inv @e3911819 scripts/ev_score_perturb.py RENDERING):
#: training rows, the native public readout and the gate-2 requests must agree byte for byte
TYPED_SCORE_RENDERING = ("ev-score-level-as-content/1: content = 'level v of K: <rubric>' (v = 0..K-1); scale canonical; "
                         "no handles in scale")


def level_content(v: int, k: int, rubric: Any) -> str:
    """R_score: the content of score level ``v`` of ``k`` = ``ev_score_perturb.render_level`` exactly (the rubric text
    verbatim, a non-string rubric as JSON with ensure_ascii False)."""
    return f"level {int(v)} of {int(k)}: {rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False)}"


def parse_level(content: str) -> tuple[int, int]:
    """(level value v, number of levels K) of a typed score option, parsed from its CONTENT (ValueError if absent:
    fail closed)."""
    import re

    global _LEVEL_RE
    if _LEVEL_RE is None:
        _LEVEL_RE = re.compile(r"level (0|[1-9][0-9]{0,3}) of ([1-9][0-9]{0,3}): ", re.S)
    m = _LEVEL_RE.match(content if isinstance(content, str) else "")
    if m is None:
        raise ValueError(f"typed score: no level value in the option content {str(content)[:40]!r}")
    return int(m.group(1)), int(m.group(2))


def typed_score_axis(display: Sequence[tuple[str, str]], scale: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """(handle, content) pairs sorted by the level value parsed from each content: the canonical score axis. Fails
    closed (ValueError) on an unparseable content, duplicate handles, a K that disagrees with the option count, a value
    set that is not exactly 0..K-1, or a given ``scale`` that is not the contents in that order."""
    items = [(str(h), str(c)) for h, c in display]
    if len({h for h, _ in items}) != len(items):
        raise ValueError("typed score: duplicate handles")
    vk = [parse_level(c) for _, c in items]
    if any(k != len(items) for _, k in vk) or sorted(v for v, _ in vk) != list(range(len(items))):
        raise ValueError(f"typed score: level values {sorted(vk)} are not exactly 0..{len(items) - 1} of {len(items)}")
    axis = [it for _, it in sorted(zip([v for v, _ in vk], items), key=lambda x: x[0])]
    if scale is not None and list(scale) != [c for _, c in axis]:
        raise ValueError("typed score: question.scale is not the option contents in level order")
    return axis


def score_scale_text(contents: Sequence[str]) -> str:
    """The full ordered scale appended to the question text of a typed score row (contents in level order)."""
    return (f"scale ({len(contents)} levels, level 0 lowest to level {len(contents) - 1} highest):\n"
            + "\n".join(contents))


def _instr_text(q: dict[str, Any]) -> str:
    v = q.get("instructions", "")
    s = v if isinstance(v, str) else json.dumps(v, indent=1)
    return s.strip() or "Answer the question below."


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


class _Pieces:
    """Prefix text with the character ranges of state and question, plus a continuation renderer.

    ``cont_parts(o)`` = (request part, template tail): the request part is " " + the option text (the one template
    space before the option travels with it, so BPE sees the natural " word" token), the tail is fixed template text.
    Trust segments (G15): the prefix is tokenized as separate segments cut at the state and question boundaries
    (``prefix_cuts``; each request segment again carries the one template space before it), and a continuation as
    its request part + its tail, so no token ever merges across a trust boundary and the template's tokens are the
    same whatever the request text is."""

    def __init__(self, prefix: str, state: tuple[int, int], instr: tuple[int, int], field: tuple[int, int],
                 cont_parts: Callable[[str], tuple[str, str]]) -> None:
        self.prefix, self.state, self.instr, self.field, self.cont_parts = prefix, state, instr, field, cont_parts

    def cont(self, o: str) -> str:
        return "".join(self.cont_parts(o))

    def seg(self, a: int, b: int) -> tuple[int, int]:
        """The trust segment of request text at chars [a, b): it absorbs the one template space right before it, also
        when the request text is empty (so the template's own tokens never depend on whether it is)."""
        return (a - 1, b) if a > 0 and self.prefix[a - 1] == " " else (a, b)

    def prefix_cuts(self) -> list[int]:
        cuts = {0, len(self.prefix), self.field[0], self.field[1]}
        for a, b in (self.state, self.instr):
            cuts |= set(self.seg(a, b))
        return sorted(cuts)


def _body(state_text: str, instr: str, n: int, open_: str) -> tuple[str, tuple, tuple, tuple]:
    rec = open_ + "record: "
    s0 = len(rec)
    text = rec + state_text + "\nquestion: "
    i0 = len(text)
    text += instr
    i1 = len(text)
    text += f"\noptions: {n}, each shown alone below as the candidate.\ncandidate:"
    return text, (s0, s0 + len(state_text)), (i0, i1), (s0 + len(state_text), len(text))


def shim_pieces(state_text: str, q: dict[str, Any], n: int, chat_tail: bool = False) -> _Pieces:
    """Qwen chat turn. Lean (default): each option ends the user turn with ``<|im_end|>`` (ONE token) and the head
    reads that position; ``chat_tail``: + "\\n" + the assistant turn with the empty think block + "Answer:"."""
    head = CHATML["system"].replace("{content}", SHIM_SYM_SYSTEM) + CHATML["user_open"]
    body, st, ins, fld = _body(state_text, _instr_text(q), n, "")
    off = len(head)
    tail = (CHATML["user_close"] + CHATML["generation"] + PREFILL) if chat_tail else "<|im_end|>"
    return _Pieces(head + body, (st[0] + off, st[1] + off), (ins[0] + off, ins[1] + off),
                   (fld[0] + off, fld[1] + off), lambda o: (" " + o.rstrip(), tail))


def is_g1_template(tpl: Any) -> bool:
    """The RWKV7-G1 chat format (``System: ...\\n\\nUser: `` head; the jrt-rwkv-g1-* templates)."""
    return str(getattr(tpl, "id", "")).startswith("jrt-rwkv-g1") and str(getattr(tpl, "head", "")).startswith("System:")


def g1_pieces(state_text: str, q: dict[str, Any], n: int, chat_tail: bool = False) -> _Pieces:
    """RWKV7-G1 vendor format (as decide.prompt.render_g1v2): ``System: <sys>\\n\\nUser: <body>``; every request text
    run of blank lines collapsed (no "\\n\\n" inside a turn, ``g1_clean``). Lean: each option ends the user turn with
    the turn separator "\\n\\n"; ``chat_tail``: + "Assistant: <think></think>\\nANSWER:" (the v2 tail)."""
    from decide.prompt import G1V2_TEMPLATE, g1_clean

    head = G1V2_TEMPLATE.head.format(system=G1_SYM_SYSTEM)
    body, st, ins, fld = _body(g1_clean(state_text), g1_clean(_instr_text(q)), n, "")
    off = len(head)
    tail = G1V2_TEMPLATE.tail if chat_tail else G1_TURN_END
    return _Pieces(head + body, (st[0] + off, st[1] + off), (ins[0] + off, ins[1] + off),
                   (fld[0] + off, fld[1] + off), lambda o: (" " + g1_clean(o).rstrip(), tail))


def template_pieces(tpl: Any, state_text: str, q: dict[str, Any], n: int) -> _Pieces:
    head = tpl.head.format(system=PLAIN_SYM_SYSTEM) if "{system}" in tpl.head else tpl.head
    body, st, ins, fld = _body(state_text, _instr_text(q), n, "")
    off = len(head)
    return _Pieces(head + body, (st[0] + off, st[1] + off), (ins[0] + off, ins[1] + off),
                   (fld[0] + off, fld[1] + off), lambda o: (" " + o, tpl.tail))


class _Tok:
    """Whole-string tokenization with character offsets: the shim spec's (transformers, special tokens honoured) or a
    byte-level encoder's (``token_bytes``)."""

    def __init__(self, encode: Any, spec: Any = None) -> None:
        self.encode, self.spec = encode, spec

    def ids_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if self.spec is not None:
            return self.spec._encoding(text)
        ids = list(self.encode(text, True))
        tb = getattr(self.encode, "token_bytes", None)
        if not callable(tb):
            raise ValueError(f"layout {SYM_LAYOUT!r} needs a shim template or an encoder with token_bytes(id)")
        # byte offsets -> char offsets
        byte_to_char, b = {}, 0
        for ci, ch in enumerate(text):
            byte_to_char[b] = ci
            b += len(ch.encode("utf-8"))
        byte_to_char[b] = len(text)
        offs, pos = [], 0
        for t in ids:
            n = len(tb(t))
            a = byte_to_char.get(pos, None)
            e = byte_to_char.get(pos + n, None)
            offs.append((a if a is not None else 0, e if e is not None else len(text)))
            pos += n
        return ids, offs

    def ids(self, text: str) -> list[int]:
        return self.ids_offsets(text)[0]

    def ids_segments(self, text: str, cuts: Sequence[int]) -> tuple[list[int], list[tuple[int, int]], dict[int, int]]:
        """Tokenize ``text`` as the separate segments ``text[cuts[i]:cuts[i+1]]`` and concatenate (no token crosses a
        cut). Returns (ids, char offsets, {cut: index of its first token}); the token span of [cuts[i], cuts[j]) is
        (bounds[cuts[i]], bounds[cuts[j]])."""
        ids: list[int] = []
        offs: list[tuple[int, int]] = []
        bounds: dict[int, int] = {}
        for x, y in zip(cuts, cuts[1:]):
            bounds[x] = len(ids)
            if y > x:
                i, o = self.ids_offsets(text[x:y])
                ids += i
                offs += [(s + x, e + x) for s, e in o]
        bounds[cuts[-1]] = len(ids)
        return ids, offs, bounds

    def cont_ids(self, parts: tuple[str, str]) -> tuple[list[int], tuple[int, int]]:
        """(ids of request part + ids of the tail, token span of the request part)."""
        req = self.ids(parts[0]) if parts[0] else []
        return req + (self.ids(parts[1]) if parts[1] else []), (0, len(req))

    def n_control(self, ids: Sequence[int]) -> int:
        if self.spec is None:
            return 0
        return sum(t in self.spec.enc.control_ids for t in ids)


def _span(offs: list[tuple[int, int]], a: int, b: int) -> tuple[int, int]:
    """Half-open token range of the tokens overlapping chars [a, b) (straddling tokens included)."""
    lo = next((i for i, (s, e) in enumerate(offs) if e > a), len(offs))
    hi = next((i for i, (s, e) in enumerate(offs) if s >= b), len(offs))
    return (lo, max(lo, hi))


def build_row_sym(request: dict[str, Any], qid: str, encode: Any, tpl: Any, layout: str,
                  option_order: Sequence[int] | None = None, empty_state: str = "reread_instructions",
                  score_options: Sequence[tuple[str, str]] | None = None):
    """The ``sym-fork`` / ``sym-fork-ids`` / ``sym-fork-typed`` row of one question (see the module docstring).
    ``score_options`` (typed score rows only): an explicit DISPLAY list of (handle, content) replacing the wire
    criteria (the score display-permutation / handle-relabel check); default = the wire levels ("i", R_score(i, rubric))."""
    from decide.provenance import instructions_path, state_path
    from decide.schema import canonical_options
    from decide.shim_format import is_shim_template
    from decide.untrusted import ControlTokenError

    from .layout import ID_PATH, LayoutRow, declared_instruction_paths, is_empty_state

    if layout not in SYM_LAYOUTS:
        raise ValueError(f"layout must be one of {SYM_LAYOUTS}")
    with_id = layout == SYM_IDS_LAYOUT
    chat_tail = layout == SYM_CHAT_LAYOUT
    q = request["questions"][qid]
    typed_score = layout == SYM_TYPED_LAYOUT and q["type"] == "score"
    if typed_score:
        if score_options is not None:
            disp = list(score_options)
        elif isinstance(q.get("options"), dict):  # an ev_score_perturb request: {handle: content} in display order
            disp = list(q["options"].items())
        else:
            wire = canonical_options(q)
            disp = [(h, level_content(int(h), len(wire), d)) for h, d in wire]
        opts = typed_score_axis(disp, q.get("scale"))  # sorted by the level parsed from content (fail closed)
        q = {**q, "instructions": _instr_text(q) + "\n" + score_scale_text([c for _, c in opts])}
    else:
        if score_options is not None:
            raise ValueError("score_options is for typed score rows only")
        opts = canonical_options(q)
    n = len(opts)
    spec = None
    g1 = is_g1_template(tpl)
    if is_shim_template(tpl):
        from decide.shim_format import spec_for_template

        spec = spec_for_template(tpl.id)
        st_text = spec.state_text(request["state"])
        pieces = shim_pieces(st_text, q, n, chat_tail)
    elif g1:
        st_text = _state_text(request["state"])
        pieces = g1_pieces(st_text, q, n, chat_tail)
    else:
        st_text = _state_text(request["state"])
        pieces = template_pieces(tpl, st_text, q, n)
    tok = _Tok(encode, spec)
    # trust segments (G15): no token merges across a trusted/untrusted boundary (see _Pieces)
    ids, offs, bounds = tok.ids_segments(pieces.prefix, pieces.prefix_cuts())
    texts, as_content = [], []
    for oid, desc in opts:
        t, c = option_text(oid, desc, with_id)
        texts.append(t)
        as_content.append(c)
    cont_enc = [tok.cont_ids(pieces.cont_parts(t)) for t in texts]
    conts_c = [c for c, _ in cont_enc]
    if spec is not None:  # control-token refusal, as the shim row: the skeleton's control count only
        skel = shim_pieces("x", {"instructions": "x"}, n, chat_tail)
        want_p = tok.n_control(tok.ids_segments(skel.prefix, skel.prefix_cuts())[0])
        want_c = tok.n_control(tok.cont_ids(skel.cont_parts("x"))[0])
        if tok.n_control(ids) != want_p or any(tok.n_control(c) != want_c for c in conts_c):
            raise ControlTokenError("request text encodes to a control token; refused")
    if g1 and (0 in ids or any(0 in c for c in conts_c)):  # as the g1v2 rows: id 0 is <|endoftext|>
        raise ControlTokenError("request text encoded to the control id 0 (<|endoftext|>); refused")
    # content-canonical axis for choice (a function of the multiset of option texts); noul / score keep their order
    canon = sorted(range(n), key=lambda k: (conts_c[k], k)) if q["type"] == "choice" else list(range(n))
    perm = list(range(n)) if option_order is None else list(option_order)
    if sorted(perm) != list(range(n)):
        raise ValueError("option_order must be a permutation of range(n_options)")
    if q["type"] == "score" and perm != sorted(perm):
        raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
    axis = [canon[k] for k in perm]
    def seg_span(a: int, b: int) -> tuple[int, int]:  # token span of a trust segment (both ends are cuts)
        return (bounds[a], bounds[b])

    state_span, evidence = seg_span(*pieces.seg(*pieces.state)), "state"
    instr_span = seg_span(*pieces.seg(*pieces.instr))
    if is_empty_state(request["state"]) or state_span[1] <= state_span[0]:
        if empty_state != "reread_instructions" or instr_span[1] <= instr_span[0]:
            raise ValueError("state is empty (no evidence content); heads need at least one evidence token")
        state_span, evidence = instr_span, "reread_instructions"
    st_sp = seg_span(*pieces.seg(*pieces.state))
    field_span = seg_span(*pieces.field)
    # option text tokens inside each continuation (its whole request part): data (ID_PATH), for the controlsafe gate
    cont_spans = [[(*cont_enc[k][1], ID_PATH)] for k in axis]
    caller = [s for s in (st_sp, instr_span) if s[1] > s[0]]
    fields = [f for f in ((st_sp[0], st_sp[1], state_path()), (instr_span[0], instr_span[1], instructions_path(qid)))
              if f[1] > f[0]]
    return LayoutRow(
        qid=qid, qtype=q["type"], option_ids=[opts[k][0] for k in axis], ids=ids, state_span=state_span,
        field_span=field_span, option_spans=[], option_seqs=[tok.ids(texts[k]) for k in axis],
        continuations=[conts_c[k] for k in axis], layout=layout,
        meta={"template": getattr(tpl, "id", str(tpl)), "sym_version": SYM_VERSION, "perm": perm, "canon": canon,
              "evidence": evidence, "untrusted_spans": [st_sp] if st_sp[1] > st_sp[0] else [],
              "caller_spans": caller, "field_spans": fields, "cont_spans": cont_spans, "option_seq_spans": [],
              "trusted_paths": declared_instruction_paths(request), "bos": None, "coded": False, "words": False,
              "catalog_chars": None, "id_agnostic": not with_id,
              "id_as_content": [opts[k][0] for k in axis if as_content[k]],
              **({"typed_score": True, "score_levels": [parse_level(opts[k][1])[0] for k in axis]} if typed_score else {})},
    )


#: wire digits for sym answers: 4-digit rounding (decide.schema.WIRE_ROUND) turns near-ties between DIFFERENT options
#: into exact ties that a reader breaks by the smallest id (jevkit.answers.argmax_label): a relabel leak at the wire
SYM_WIRE_DIGITS = 12


def content_key(cont: Sequence[int]) -> str:
    """Content hash of one option (sha256 of its continuation token ids): the tie order for DIFFERENT options."""
    import hashlib

    return hashlib.sha256(json.dumps(list(cont)).encode()).hexdigest()


def sym_tie_info(row: Any) -> dict[str, Any] | None:
    """For a sym row: the option ids in content order (the order every option-axis reduction uses, so softmax sums are
    order-free), each option's content hash, and the groups of exact duplicates (equal continuations: equal content,
    hence equal probability; a single-id decision among them is impossible under relabel invariance). None otherwise."""
    if getattr(row, "layout", None) not in SYM_LAYOUTS:
        return None
    keys = {o: content_key(c) for o, c in zip(row.option_ids, row.continuations)}
    groups: dict[str, list[str]] = {}
    for o, k in keys.items():
        groups.setdefault(k, []).append(o)
    return {"content_order": sorted(row.option_ids, key=lambda o: (row.continuations[row.option_ids.index(o)], keys[o])),
            "content_key": keys, "dup_groups": sorted(sorted(g) for g in groups.values() if len(g) > 1),
            "digits": SYM_WIRE_DIGITS}


def row_input_tokens(row: Any) -> int:
    """Tokens a row FEEDS the model: the prefix, plus every continuation for the symmetric layouts (their option texts
    are real input). Other layouts keep the old count (the prefix only; their continuations are 1-token labels)."""
    n = len(row.ids)
    if getattr(row, "layout", None) in SYM_LAYOUTS:
        n += sum(len(c) for c in row.continuations)
    return n

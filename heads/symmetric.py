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

Formats: a shim template (``jrt-h2o-shim:<family>``) gets the Qwen chat turn of the H2O shim with a symmetric user turn
(``SHIM_SYM_SYSTEM``) tokenized whole-string by the shim spec (special tokens honoured, request text that encodes to a
control token refused); any other template gets ``head + body`` / ``" <option>" + tail`` with the row encoder.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

SYM_LAYOUT = "sym-fork"
SYM_IDS_LAYOUT = "sym-fork-ids"
SYM_LAYOUTS = (SYM_LAYOUT, SYM_IDS_LAYOUT)
SYM_VERSION = "sym-fork-v1"
#: train.run_cell.keep_item: longest continuation (option text + the ~12-token chat tail) a sym row may carry
SYM_MAX_CONT = 128

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


def _instr_text(q: dict[str, Any]) -> str:
    v = q.get("instructions", "")
    s = v if isinstance(v, str) else json.dumps(v, indent=1)
    return s.strip() or "Answer the question below."


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


class _Pieces:
    """Prefix text with the character ranges of state and question, plus a continuation renderer."""

    def __init__(self, prefix: str, state: tuple[int, int], instr: tuple[int, int], field: tuple[int, int],
                 cont: Callable[[str], str]) -> None:
        self.prefix, self.state, self.instr, self.field, self.cont = prefix, state, instr, field, cont


def _body(state_text: str, instr: str, n: int, open_: str) -> tuple[str, tuple, tuple, tuple]:
    rec = open_ + "record: "
    s0 = len(rec)
    text = rec + state_text + "\nquestion: "
    i0 = len(text)
    text += instr
    i1 = len(text)
    text += f"\noptions: {n}, each shown alone below as the candidate.\ncandidate:"
    return text, (s0, s0 + len(state_text)), (i0, i1), (s0 + len(state_text), len(text))


def shim_pieces(state_text: str, q: dict[str, Any], n: int) -> _Pieces:
    head = CHATML["system"].replace("{content}", SHIM_SYM_SYSTEM) + CHATML["user_open"]
    body, st, ins, fld = _body(state_text, _instr_text(q), n, "")
    off = len(head)
    tail = CHATML["user_close"] + CHATML["generation"] + PREFILL
    return _Pieces(head + body, (st[0] + off, st[1] + off), (ins[0] + off, ins[1] + off),
                   (fld[0] + off, fld[1] + off), lambda o: " " + o.rstrip() + tail)


def template_pieces(tpl: Any, state_text: str, q: dict[str, Any], n: int) -> _Pieces:
    head = tpl.head.format(system=PLAIN_SYM_SYSTEM) if "{system}" in tpl.head else tpl.head
    body, st, ins, fld = _body(state_text, _instr_text(q), n, "")
    off = len(head)
    return _Pieces(head + body, (st[0] + off, st[1] + off), (ins[0] + off, ins[1] + off),
                   (fld[0] + off, fld[1] + off), lambda o: " " + o + tpl.tail)


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
                  option_order: Sequence[int] | None = None, empty_state: str = "reread_instructions"):
    """The ``sym-fork`` / ``sym-fork-ids`` row of one question (see the module docstring)."""
    from decide.provenance import instructions_path, state_path
    from decide.schema import canonical_options
    from decide.shim_format import is_shim_template
    from decide.untrusted import ControlTokenError

    from .layout import ID_PATH, LayoutRow, declared_instruction_paths, is_empty_state

    if layout not in SYM_LAYOUTS:
        raise ValueError(f"layout must be one of {SYM_LAYOUTS}")
    with_id = layout == SYM_IDS_LAYOUT
    q = request["questions"][qid]
    opts = canonical_options(q)
    n = len(opts)
    spec = None
    if is_shim_template(tpl):
        from decide.shim_format import spec_for_template

        spec = spec_for_template(tpl.id)
        st_text = spec.state_text(request["state"])
        pieces = shim_pieces(st_text, q, n)
    else:
        st_text = _state_text(request["state"])
        pieces = template_pieces(tpl, st_text, q, n)
    tok = _Tok(encode, spec)
    ids, offs = tok.ids_offsets(pieces.prefix)
    texts, as_content = [], []
    for oid, desc in opts:
        t, c = option_text(oid, desc, with_id)
        texts.append(t)
        as_content.append(c)
    cont_texts = [pieces.cont(t) for t in texts]
    conts_c = [tok.ids(c) for c in cont_texts]
    if spec is not None:  # control-token refusal, as the shim row: the skeleton's control count only
        want_p = tok.n_control(tok.ids(shim_pieces("x", {"instructions": "x"}, n).prefix))
        want_c = tok.n_control(tok.ids(shim_pieces("x", {"instructions": "x"}, n).cont("x")))
        if tok.n_control(ids) != want_p or any(tok.n_control(c) != want_c for c in conts_c):
            raise ControlTokenError("request text encodes to a control token; refused")
    # content-canonical axis for choice (a function of the multiset of option texts); noul / score keep their order
    canon = sorted(range(n), key=lambda k: (conts_c[k], k)) if q["type"] == "choice" else list(range(n))
    perm = list(range(n)) if option_order is None else list(option_order)
    if sorted(perm) != list(range(n)):
        raise ValueError("option_order must be a permutation of range(n_options)")
    if q["type"] == "score" and perm != sorted(perm):
        raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
    axis = [canon[k] for k in perm]
    state_span, evidence = _span(offs, *pieces.state), "state"
    instr_span = _span(offs, *pieces.instr)
    if is_empty_state(request["state"]) or state_span[1] <= state_span[0]:
        if empty_state != "reread_instructions" or instr_span[1] <= instr_span[0]:
            raise ValueError("state is empty (no evidence content); heads need at least one evidence token")
        state_span, evidence = instr_span, "reread_instructions"
    st_sp = _span(offs, *pieces.state)
    field_span = _span(offs, *pieces.field)
    # option text tokens inside each continuation: data (ID_PATH), for the controlsafe trust gate
    cont_spans = []
    for k in axis:
        c_ids, c_offs = tok.ids_offsets(cont_texts[k])
        a = 1 if cont_texts[k].startswith(" ") else 0
        cont_spans.append([(*_span(c_offs, a, a + len(texts[k].rstrip())), ID_PATH)])
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
              "id_as_content": [opts[k][0] for k in axis if as_content[k]]},
    )


def row_input_tokens(row: Any) -> int:
    """Tokens a row FEEDS the model: the prefix, plus every continuation for the symmetric layouts (their option texts
    are real input). Other layouts keep the old count (the prefix only; their continuations are 1-token labels)."""
    n = len(row.ids)
    if getattr(row, "layout", None) in SYM_LAYOUTS:
        n += sum(len(c) for c in row.continuations)
    return n

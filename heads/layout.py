"""Options-first prompt layout with token spans for learned decision heads.

Two layouts, both "schema + options -> state -> question re-read" (JRT, arXiv 2407.05483):

``ordered``
    Byte-identical to ``decide.prompt.build_prompt`` (the m0 readout prompt), so a
    head trained here reads exactly what the m0 stock readout reads. Options are
    rendered inside the prefix in canonical order (sorted id). The model input is
    invariant to the order in which the *caller* lists options, but it is NOT
    structurally permutation-equivariant: a causal reader sees option j after
    options < j, so renaming ids (which changes canonical order) can change
    outputs. ``analysis/CODEX_CRITICAL_COMPARISON.md``: "an ordered schema prefix
    still defeats structural option permutation equivariance unless
    canonicalization or a symmetric construction makes the common prefix
    independent of permutation". Canonical order is that canonicalization; it is
    an adapter property, not a model property.

``g1v2``
    D3 (C1 format v2, readout R1): the ``ordered`` JRT order in the vendor-compliant RWKV7-G1 chat format of
    ``decide.prompt.build_prompt_g1v2`` / ``decide.backends.rwkv_g1_v2`` -- no blank line inside the turn,
    token 0 first, the G1k no-think tail, options listed as ``<CODE>. <id>: <description>`` with single-token codes
    from the frozen filtered codebook (the option continuation is the ONE code token), and the whole prompt string
    tokenized at once. The row's text is byte-identical to ``render_g1v2`` (asserted), so the trained head reads
    exactly what the R1 stock readout reads. Spans are mapped from character offsets to the whole-string tokens
    (a token that straddles a segment boundary belongs to both neighbouring spans). Template must be
    ``jrt-rwkv-g1-v2``; the encoder must expose ``token_bytes(id)``.

``symmetric``
    The shared prefix lists every field but *not* its options (it says options
    are given separately). Each option of the target question is encoded in its
    own short sequence by the same backbone (partner plan "Two heads to
    compare": e_k = E(o_k) with a shared encoder). Nothing in the shared pass
    depends on option order, so a head that treats the option axis
    symmetrically is permutation-equivariant end to end (up to float reduction
    order). The cost is that the state is read knowing the question but not
    the option texts.

Spans are half-open token ranges ``[start, end)`` into ``ids``. Tokenisation is
segment-wise (each segment encoded on its own and concatenated), the same rule
the m0 HF readout uses so untrusted segments can never produce control tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import bisect

from decide.prompt import _G1_BLANK_LINES as _G1_BLANK  # the readout's own blank-line rule (render_g1v2)
from decide.prompt import (G1V2_ID, G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE, SYSTEM_PROMPT, TEMPLATES, Template, _instr,
                           build_prompt_g1v2, continuation_segments, render_g1v2)
from decide.prompt_g1v3 import (G1V2F_ID, G1V3P_ID, G1V3W_ID, build_prompt_g1v3, identity_order, presentation_order,
                                question_codes, render_g1v3, wants_catalog, word_ids_ok)
from decide.provenance import field_strings, instructions_path, state_path
from decide.schema import canonical_options, canonical_question_ids, render

G1V2_LAYOUT = "g1v2"
LAYOUTS = ("ordered", "symmetric", G1V2_LAYOUT)
#: bump on any change to tagged_segments / option_segments / build_row token accounting
LAYOUT_VERSION = "m2-layout-v2"  # v2: empty-state rows use the re-read instruction as evidence

#: encode(text, trusted) -> token ids.
EncodeFn = Callable[[str, bool], list[int]]

# (text, trusted, tag). Tags: "", "field", "option:<j>", "state", "reread", "reread_instr".
TaggedSegment = tuple[str, bool, str]
# Internally each segment also carries the request field it renders (controlsafe trust gate): an RFC 6901
# provenance path (``decide.provenance``), ID_PATH for question/option ids (ids are always data under the
# provenance contract), or None for template text.
ID_PATH = "#ids"


def _instr_path(qid: str, q: dict[str, Any]) -> str:
    instr = q.get("instructions")
    return instructions_path(qid) if isinstance(instr, str) and instr != "" else ID_PATH


def _field(index: int, qid: str, q: dict[str, Any], target: bool, with_options: bool) -> list[tuple]:
    ftag = "field" if target else ""
    seg: list[tuple] = [
        (f"FIELD {index}\nID: ", True, ftag, None),
        (qid, False, ftag, ID_PATH),
        (f"\nTYPE: {q['type']}\nINSTRUCTION: ", True, ftag, None),
        (render(_instr(qid, q)), False, ftag, _instr_path(qid, q)),
    ]
    if with_options:
        seg.append(("\nALLOWED OPTIONS:", True, ftag, None))
        for j, (oid, desc) in enumerate(canonical_options(q), start=1):
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            otag = f"option:{j - 1}" if target else ""
            seg += [(f"\nOPTION {j}: ", True, ftag, None), (render(sem), False, otag, ID_PATH)]
    else:
        n = len(canonical_options(q))
        seg.append((f"\nALLOWED OPTIONS: {n} options, given separately and unordered.", True, ftag, None))
    seg.append(("\nEND FIELD", True, ftag, None))
    return seg


def _v2f_segments(request: dict[str, Any], qid: str, table: list[str]):
    return build_prompt_g1v3(request, qid, table, None, False, True)  # v2f: no code key, no permutation, fixed instr


def _v2f_render(request: dict[str, Any], qid: str, table: list[str]) -> str:
    return render_g1v3(request, qid, table, None, False, True)


#: layout "g1v2" training templates -> the READOUT's own (segments, render) functions. Every entry uses v2's frozen
#: code table and v2's head/tail, and its readout reads the next token after "ANSWER:" at the last prompt position.
#:   jrt-rwkv-g1-v2   decide.prompt.build_prompt_g1v2 / render_g1v2            (readout rwkv7-hf-g1-v2, job-zhyvg)
#:   jrt-rwkv-g1-v2f  decide.prompt_g1v3 v2f = v2 + the dict-instruction fix   (readout rwkv7-hf-g1-v2f, G1V3Backend v2f)
G1_TRAIN_RENDERERS = {G1V2_ID: (build_prompt_g1v2, render_g1v2), G1V2F_ID: (_v2f_segments, _v2f_render)}
G1_TRAIN_TEMPLATES = {G1V2_ID: G1V2_TEMPLATE,
                      G1V2F_ID: Template(id=G1V2F_ID, head=G1V2_TEMPLATE.head, tail=G1V2_TEMPLATE.tail,
                                         terminator=G1V2_TEMPLATE.terminator)}

#: EVAL-ONLY readouts of a layout-g1v2 checkpoint (jobs.m2_first_cell.readout_ab): the same backbone + LoRA + cache-fork
#: head reading another R3 prompt. Never a training template (the readout_ab mode refuses a cell whose template is not
#: in G1_TRAIN_TEMPLATES). Each is the stock readout family's own renderer (decide.prompt_g1v3.build_prompt_g1v3 /
#: render_g1v3, asserted byte-equal per row):
#:   jrt-rwkv-g1-v3p   v3 code table + code key (k >= 9, label ladder 64 / 24 / none under the stock 16,383-token
#:                     budget) + per-question presentation permutation          (readout rwkv7-hf-g1-v3p, job-fgu2c)
#:   jrt-rwkv-g1-v2fp  v2f + the same presentation permutation only (v2 code table, no code key): the permutation's
#:                     effect isolated from the v3 code key (no stock family)
#:   jrt-rwkv-g1-v3w   v3 + word option ids: a question whose ids are non-opaque words with unique first answer tokens
#:                     (decide.prompt_g1v3.word_ids_ok, the first token of " <id>" right after "ANSWER:") is listed as
#:                     "- <id>: <desc>" and option j is forked with that ONE first token; other questions as v3
#:                     (readout rwkv7-hf-g1-v3w, job-fgu2c)
#: The answer slot is unchanged ("ANSWER:" at the last prompt position, one code token per option); each canonical
#: option j is forked with the code PRESENTED for it (decide.prompt_g1v3.question_codes), so the head's logits stay
#: keyed by option id in canonical order.
G1V2FP_ID = "jrt-rwkv-g1-v2fp"
G1_EVAL_READOUTS: dict[str, dict[str, Any]] = {
    G1V3P_ID: {"table": "v3", "permute": True, "catalog": True, "words": False},
    G1V2FP_ID: {"table": "v2", "permute": True, "catalog": False, "words": False},
    G1V3W_ID: {"table": "v3", "permute": False, "catalog": True, "words": True}}
#: decide.backends.rwkv_g1_v3.ANSWER_CTX: the context the stock v3w reads an id's first answer token in
G1_ANSWER_CTX = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"
G1_EVAL_TEMPLATES = {tid: Template(id=tid, head=G1V2_TEMPLATE.head, tail=G1V2_TEMPLATE.tail,
                                   terminator=G1V2_TEMPLATE.terminator) for tid in G1_EVAL_READOUTS}
#: the stock G1V3Backend's catalog budget: min(RWKV7 G1k trained ctx 25,600, adapter default 16,384) - 1 answer position
CATALOG_BUDGET = 16_383
G1_TEMPLATE_IDS = frozenset(G1_TRAIN_RENDERERS) | frozenset(G1_EVAL_READOUTS)


def _g1_spec(template_id: str) -> dict[str, Any]:
    if template_id in G1_TRAIN_RENDERERS:
        return {"table": "v2", "permute": False, "catalog": False, "words": False}
    if template_id in G1_EVAL_READOUTS:
        return G1_EVAL_READOUTS[template_id]
    raise ValueError(f"layout {G1V2_LAYOUT!r} needs one of the templates {sorted(G1_TEMPLATE_IDS)}, got {template_id!r}")


def _g1_prompt(template_id: str, request: dict[str, Any], qid: str, table: list[str], catalog_chars: int | None,
               words: frozenset[str] = frozenset()):
    """(segments, render) of a layout-g1v2 template: the training renderers as before, the eval readouts through the
    R3 renderer with their permutation / code-key / word-id flags (``words`` = v3w's word-id question ids)."""
    if template_id in G1_TRAIN_RENDERERS:
        seg_fn, render_fn = G1_TRAIN_RENDERERS[template_id]
        return seg_fn(request, qid, table), render_fn(request, qid, table)
    spec = _g1_spec(template_id)
    n = catalog_chars if spec["catalog"] else None
    return (build_prompt_g1v3(request, qid, table, n, spec["permute"], True, words),
            render_g1v3(request, qid, table, n, spec["permute"], True, words))


def g1_first_token(encode: EncodeFn, oid: str) -> int | None:
    """The first answer token of " <oid>" right after "ANSWER:" (the stock G1V3Backend.first_token rule; None when the
    tokenization does not extend the context)."""
    ctx = list(encode(G1_ANSWER_CTX, True))
    ids = list(encode(G1_ANSWER_CTX + " " + oid, True))
    n = len(ctx)
    return ids[n] if len(ids) > n and ids[:n] == ctx and ids[n] != 0 else None


def g1_word_qids(template_id: str, request: dict[str, Any], encode: EncodeFn) -> frozenset[str]:
    """v3w: the question ids scored by their word ids (``word_ids_ok`` with the encoder's first answer token)."""
    if not _g1_spec(template_id).get("words"):
        return frozenset()
    qs = request["questions"]
    return frozenset(k for k in canonical_question_ids(qs) if word_ids_ok(qs[k], lambda o: g1_first_token(encode, o)))


def resolve_template(template: Template | str) -> Template:
    """A template by id; the layout-``g1v2`` templates (``G1_TRAIN_TEMPLATES``: jrt-rwkv-g1-v2, jrt-rwkv-g1-v2f) are
    resolved here, not registered in ``decide.prompt.TEMPLATES`` (their identities are their readouts' renderer
    identities, ``heads.decide_backend._template_identity``)."""
    if isinstance(template, Template):
        return template
    return G1_TRAIN_TEMPLATES.get(template) or G1_EVAL_TEMPLATES.get(template) or TEMPLATES[template]


def tagged_segments(template: Template, request: dict[str, Any], qid: str, layout: str) -> list[TaggedSegment]:
    return [s[:3] for s in _tagged_segments_paths(template, request, qid, layout)]


def declared_instruction_paths(request: dict[str, Any]) -> list[str]:
    """Request field paths the HOST declared instruction-trusted in ``request["ssdm"]["provenance"]`` (the
    ``decide.provenance`` shape): a non-state string field whose segments tile it completely and are all
    ``trust == "instruction"``. Anything else (absent, partial, malformed, non-string) is data: fail closed."""
    ssdm = request.get("ssdm")
    prov = ssdm.get("provenance") if isinstance(ssdm, dict) else None
    segs = prov.get("segments") if isinstance(prov, dict) else None
    if not isinstance(segs, dict):
        return []
    try:
        strings = field_strings(request)
    except Exception:  # noqa: BLE001 - malformed request: nothing is trusted
        return []
    out = []
    for path, lst in segs.items():
        kv = strings.get(path)
        if path == state_path() or kv is None or not isinstance(kv[1], str) or not kv[1] or not isinstance(lst, list):
            continue
        pos, ok = 0, bool(lst)
        for sg in lst:
            if not (isinstance(sg, dict) and sg.get("trust") == "instruction" and sg.get("start") == pos
                    and isinstance(sg.get("end"), int) and sg["end"] > pos):
                ok = False
                break
            pos = sg["end"]
        if ok and pos == len(kv[1]):
            out.append(path)
    return sorted(out)


def _tagged_segments_paths(template: Template, request: dict[str, Any], qid: str, layout: str) -> list[tuple]:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if layout == G1V2_LAYOUT:
        spec = _g1_spec(template.id)
        if spec["words"]:
            raise ValueError(f"{template.id}: word-id questions depend on the encoder; use build_row")
        return _g1v2_segments_paths(request, qid, _g1_table(spec["table"])[0], template.id,
                                    64 if spec["catalog"] else None)
    with_options = layout == "ordered"
    questions = request["questions"]
    order = canonical_question_ids(questions)
    q = questions[qid]
    seg: list[tuple] = [(template.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", True, "", None)]
    for i, k in enumerate(order, start=1):
        if i > 1:
            seg.append(("\n", True, "", None))
        seg += _field(i, k, questions[k], k == qid, with_options)
    ids_text = ", ".join(o for o, _ in canonical_options(q)) if with_options else "(the separately given options)"
    seg += [
        ("\nEND SCHEMA\n\nSTATE:\n", True, "", None),
        (render(request["state"]), False, "state", state_path()),
        (f"\nEND STATE\n\nQUESTION (re-read FIELD {order.index(qid) + 1}):\nID: ", True, "reread", None),
        (qid, False, "reread", ID_PATH),
        (f"\nTYPE: {q['type']}\nINSTRUCTION: ", True, "reread", None),
        (render(_instr(qid, q)), False, "reread_instr", _instr_path(qid, q)),
        ("\nALLOWED OPTION IDS: ", True, "reread", None),
        (ids_text, not with_options, "reread", ID_PATH if with_options else None),
        ("\nReply with exactly one allowed option id." + template.tail, True, "reread", None),
    ]
    return [s for s in seg if s[0]]


def _paths_spans(encode: EncodeFn, segs: Sequence[tuple[str, bool]], paths: Sequence[str | None]) -> list:
    """Local token spans ``(start, end, path)`` of the caller segments of a stand-alone sequence."""
    out, n = [], 0
    for (text, trusted), path in zip(segs, paths):
        m = len(encode(text, trusted))
        if not trusted and m:
            out.append((n, n + m, path))
        n += m
    return out


def option_segments(qid: str, q: dict[str, Any], oid: str, desc: Any) -> list[tuple[str, bool]]:
    """Stand-alone option sequence for the symmetric layout (no index, no neighbours)."""
    sem = {"option_id": oid}
    if desc is not None:
        sem["description"] = desc
    return [
        ("OPTION FOR FIELD ", True), (qid, False), (f" ({q['type']}): ", True), (render(sem), False),
    ]


@dataclass
class LayoutRow:
    """One (request, question) row: everything a head needs, as token ids + spans."""

    qid: str
    qtype: str
    option_ids: list[str]  # canonical order; index k is the option axis everywhere
    ids: list[int]  # shared prefix: schema -> state -> re-read (ends at the answer position)
    state_span: tuple[int, int]
    field_span: tuple[int, int]  # the target question's field in the schema (known before the state)
    option_spans: list[tuple[int, int]]  # ordered layout only; empty for symmetric
    option_seqs: list[list[int]]  # symmetric layout only; one sequence per option
    continuations: list[list[int]]  # option continuation ids for the final-state fork
    layout: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.option_ids)

    @property
    def answer_pos(self) -> int:
        return len(self.ids) - 1

    @property
    def evidence_len(self) -> int:
        return self.state_span[1] - self.state_span[0]


def is_empty_state(state: Any) -> bool:
    """True when the state carries no evidence content: None, a blank string, or a dict/list whose
    values are all (recursively) empty. Keys alone are not content. Numbers and booleans are content."""
    if state is None:
        return True
    if isinstance(state, str):
        return not state.strip()
    if isinstance(state, dict):
        return all(is_empty_state(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return all(is_empty_state(v) for v in state)
    return False


def _encode_segments(encode: EncodeFn, segs: Sequence[tuple[str, bool]]) -> list[int]:
    out: list[int] = []
    for text, trusted in segs:
        out += encode(text, trusted)
    return out


def build_row(
    request: dict[str, Any],
    qid: str,
    encode: EncodeFn,
    template: Template | str = "jrt-plain-v1",
    layout: str = "ordered",
    option_order: Sequence[int] | None = None,
    empty_state: str = "reread_instructions",
) -> LayoutRow:
    """Tokenise one question row and record spans.

    Evidence span: the state's tokens. A request whose state is EMPTY in content (``is_empty_state``:
    ``""``, whitespace, ``None``, ``{}``, ``[]``, or containers holding only such values; the PR #16
    builder's ``placement: instructions`` augmentation emits ``""`` or ``{}`` and puts the passage in the
    question instructions, ~21 % of the m2 sample) uses the target question's re-read INSTRUCTION tokens
    as its evidence span instead (``meta["evidence"]`` says which), so heads integrate over the text that
    carries the evidence rather than over the characters ``{}``. The prompt text itself is unchanged.
    ``empty_state="error"`` refuses such rows instead.

    ``option_order`` permutes the option axis (a test hook for equivariance): row k
    of every per-option field is canonical option ``option_order[k]``. The shared
    prefix is unaffected (ordered layout keeps canonical rendering).
    """
    tpl = resolve_template(template)
    if layout == G1V2_LAYOUT:
        return _build_row_g1v2(request, qid, encode, tpl, option_order, empty_state)
    segs4 = _tagged_segments_paths(tpl, request, qid, layout)
    q = request["questions"][qid]
    opts = canonical_options(q)
    ids: list[int] = []
    spans: dict[str, list[int]] = {}
    caller_spans: list[tuple[int, int]] = []  # every non-template (caller/host-data) segment, for trust gating
    field_spans: list[tuple[int, int, str]] = []  # the same spans with the request field each one renders
    for text, trusted, tag, path in segs4:
        start = len(ids)
        ids += encode(text, trusted)
        if not trusted and len(ids) > start:
            caller_spans.append((start, len(ids)))
            field_spans.append((start, len(ids), path))
        if tag:
            if tag in spans:
                spans[tag][1] = len(ids)
            else:
                spans[tag] = [start, len(ids)]
    # provenance-untrusted tokens (controlsafe trust gate): the rendered state, before any evidence-span move
    untrusted_spans = [tuple(spans["state"])] if "state" in spans and spans["state"][1] > spans["state"][0] else []
    evidence = "state"
    no_state_tokens = "state" not in spans or spans["state"][1] <= spans["state"][0]
    if no_state_tokens or is_empty_state(request["state"]):
        # The prompt text is unchanged (byte-identical to decide.prompt.build_prompt, so "{}" is still
        # rendered); only the evidence span moves to the text that actually carries the evidence.
        alt = spans.get("reread_instr")
        if empty_state != "reread_instructions" or alt is None or alt[1] <= alt[0]:
            raise ValueError("state is empty (no evidence content); heads need at least one evidence token")
        spans["state"], evidence = alt, "reread_instructions"
    perm = list(range(len(opts))) if option_order is None else list(option_order)
    if sorted(perm) != list(range(len(opts))):
        raise ValueError("option_order must be a permutation of range(n_options)")
    if q["type"] == "score" and perm != sorted(perm):
        raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
    option_spans = [tuple(spans[f"option:{k}"]) for k in perm] if layout == "ordered" else []
    option_seqs = (
        [_encode_segments(encode, option_segments(qid, q, *opts[k])) for k in perm] if layout == "symmetric" else []
    )
    conts = [_encode_segments(encode, continuation_segments(tpl, opts[k][0])) for k in perm]
    cont_spans = [_paths_spans(encode, continuation_segments(tpl, opts[k][0]), [ID_PATH, None]) for k in perm]
    opt_seq_spans = ([_paths_spans(encode, option_segments(qid, q, *opts[k]), [None, ID_PATH, None, ID_PATH])
                      for k in perm] if layout == "symmetric" else [])
    return LayoutRow(
        qid=qid,
        qtype=q["type"],
        option_ids=[opts[k][0] for k in perm],
        ids=ids,
        state_span=tuple(spans["state"]),
        field_span=tuple(spans["field"]),
        option_spans=option_spans,
        option_seqs=option_seqs,
        continuations=conts,
        layout=layout,
        meta={"template": tpl.id, "perm": perm, "evidence": evidence, "untrusted_spans": untrusted_spans,
              "caller_spans": caller_spans, "field_spans": field_spans, "cont_spans": cont_spans,
              "option_seq_spans": opt_seq_spans, "trusted_paths": declared_instruction_paths(request)},
    )


# ------------------------------------------------------------------------------------------------ g1v2 (D3 / R1)
_CODE_TABLE: list[str] | None = None
_CODE_TOKENS: list[int] | None = None


def _g1v2_code_table() -> list[str]:
    global _CODE_TABLE, _CODE_TOKENS
    if _CODE_TABLE is None:
        from decide.backends.rwkv_g1_filtered_codes import load_frozen_table

        codes = load_frozen_table()
        _CODE_TABLE, _CODE_TOKENS = [c.code for c in codes], [c.token_id for c in codes]
    return _CODE_TABLE


def _g1_table(name: str) -> tuple[list[str], list[int]]:
    """(codes, single token ids) of the v2 frozen filtered codebook or the R3 v3 code table."""
    if name == "v2":
        _g1v2_code_table()
        return _CODE_TABLE, _CODE_TOKENS
    from decide.prompt_g1v3 import load_v3_codes

    entries, _ = load_v3_codes()
    return [e["code"] for e in entries], [e["token_id"] for e in entries]


def _g1v2_segments_paths(request: dict[str, Any], qid: str, code_table: list[str],
                         template_id: str = G1V2_ID, catalog_chars: int | None = None,
                         segments: list | None = None) -> list[tuple]:
    """``decide.prompt.build_prompt_g1v2`` (the v2 READOUT's own segments, reused, not re-implemented) tagged as
    (text, trusted, tag, path). Tags: "field" (target field in the schema), "option:<j>" (its option j id +
    description), "state", "reread", "reread_instr"; paths as the ordered layout's (provenance / ID_PATH / None)."""
    questions = request["questions"]
    order = canonical_question_ids(questions)
    out: list[tuple] = []
    field: str | None = None  # the field being walked (schema), qid in the re-read
    phase, opt_j, expect = "schema", -1, None  # expect: the kind of the next data segment
    spec = _g1_spec(template_id)
    # option j in LISTING order -> canonical option index (identity unless the readout permutes the presentation)
    shown = presentation_order(questions[qid], qid) if spec["permute"] else identity_order(questions[qid])
    if segments is None:
        segments = _g1_prompt(template_id, request, qid, code_table, catalog_chars)[0]
    for text, label in segments:
        target = phase == "schema" and field == qid
        if label == "data" and phase == "reread" and text.startswith("\nQUESTION AGAIN: "):
            # v3 code key (one data segment: the restated instruction + "code = option label" lines)
            out.append((text, False, "reread", _instr_path(qid, questions[qid])))
            continue
        if label == "template":
            tag = ""
            if phase == "schema" and text.startswith("FIELD ") and "\nID: " in text:
                field, opt_j, expect = order[int(text[6:text.index("\n")]) - 1], -1, "qid"
                tag = "field" if field == qid else ""
            elif text.startswith("\nEND SCHEMA"):
                phase, field, expect = "state", None, "state"
            elif text.startswith("\nEND STATE"):
                phase, field, opt_j, expect, tag = "reread", qid, -1, "qid", "reread"
            else:
                if text.startswith("\nTYPE: "):
                    expect = "instr"
                elif text == ": ":
                    expect = "desc"
                elif text.startswith("\n") and (text.endswith(". ") or text == "\n- "):
                    opt_j, expect = opt_j + 1, "oid"
                tag = ("reread" if phase == "reread" else
                       f"option:{shown[opt_j]}" if target and text == ": " else
                       "field" if target and text != "\n" else "")
            out.append((text, True, tag, None))
            continue
        if expect == "state":
            out.append((text, False, "state", state_path()))
        elif expect == "qid":
            out.append((text, False, "reread" if phase == "reread" else ("field" if target else ""), ID_PATH))
        elif expect == "instr":
            out.append((text, False, "reread_instr" if phase == "reread" else ("field" if target else ""),
                        _instr_path(field, questions[field])))
        elif expect in ("oid", "desc"):
            out.append((text, False, "reread" if phase == "reread" else (f"option:{shown[opt_j]}" if target else ""),
                        ID_PATH))
        else:
            raise ValueError(f"g1v2 layout: unexpected request segment {text[:40]!r} in the v2 prompt")
    return out


def _build_row_g1v2(request: dict[str, Any], qid: str, encode: EncodeFn, tpl: Template,
                    option_order: Sequence[int] | None, empty_state: str) -> LayoutRow:
    from decide.untrusted import ControlTokenError

    spec = _g1_spec(tpl.id)
    token_bytes = getattr(encode, "token_bytes", None)
    if not callable(token_bytes):
        raise ValueError(f"layout {G1V2_LAYOUT!r} needs an encoder with token_bytes(id) (whole-string spans)")
    table, table_tokens = _g1_table(spec["table"])
    q = request["questions"][qid]
    words = g1_word_qids(tpl.id, request, encode)
    # code-key label ladder exactly as the stock G1V3Backend._render: the longest label whose prompt fits the budget,
    # the block dropped as the last resort (training templates, word-id questions and k < 9: one rung, no block)
    ladder = (64, 24, None) if spec["catalog"] and qid not in words and wants_catalog(q, table) else (None,)
    for catalog_chars in ladder:
        prompt_segs, rendered = _g1_prompt(tpl.id, request, qid, table, catalog_chars, words)
        if len(ladder) == 1 or 1 + len(encode(rendered, True)) <= CATALOG_BUDGET:
            break
    segs = _g1v2_segments_paths(request, qid, table, tpl.id, catalog_chars, prompt_segs)
    raw = "".join(t for t, *_ in segs)
    bounds, n = [], 0
    for t, *_ in segs:
        bounds.append((n, n + len(t)))
        n += len(t)
    # blank-line collapse over the user body (render_g1v2): deleted chars shift every later offset
    head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
    body_lo, body_hi = len(head), len(raw) - len(G1V2_TEMPLATE.tail)
    dels = [(m.start() + 1, m.end()) for m in _G1_BLANK.finditer(raw, body_lo, body_hi)]  # deleted raw [a, b)
    starts_d = [a for a, _ in dels]
    cum = [0]
    for a, b in dels:
        cum.append(cum[-1] + b - a)

    def to_clean(x: int) -> int:
        i = bisect.bisect_right(starts_d, x) - 1
        if i < 0:
            return x
        a, b = dels[i]
        return a - cum[i] if x < b else x - cum[i + 1]

    keep_lo = [0] + [b for _, b in dels]
    keep_hi = [a for a, _ in dels] + [len(raw)]
    text = "".join(raw[p:q] for p, q in zip(keep_lo, keep_hi))
    if text != rendered:
        raise ValueError(f"g1v2 layout text differs from the {tpl.id} readout's render")
    body_ids = list(encode(text, True))  # F5: the whole prompt string at once
    if 0 in body_ids:
        raise ControlTokenError("request text encoded to the control id 0 (<|endoftext|>); refused")
    tok_start, pos = [], 0  # token byte offsets in the body
    for t in body_ids:
        tok_start.append(pos)
        pos += len(token_bytes(t))
    if pos != len(text.encode("utf-8")):
        raise ValueError("token bytes do not tile the prompt (encoder / token_bytes mismatch)")
    byte_at = [0]
    for ch in text:
        byte_at.append(byte_at[-1] + len(ch.encode("utf-8")))

    def span(c0: int, c1: int) -> tuple[int, int]:
        """clean char range -> half-open token range (straddling tokens included), +1 for the leading [0] (F2)"""
        b0, b1 = byte_at[c0], byte_at[c1]
        if b1 <= b0:
            t = 1 + bisect.bisect_left(tok_start, b0)
            return (t, t)
        return (bisect.bisect_right(tok_start, b0), 1 + bisect.bisect_left(tok_start, b1))

    ids = [0] + body_ids
    spans: dict[str, list[int]] = {}
    caller_spans: list[tuple[int, int]] = []
    field_spans: list[tuple[int, int, str]] = []
    for (t, trusted, tag, path), (p, q) in zip(segs, bounds):
        a, b = span(to_clean(p), to_clean(q))
        if not trusted and b > a:
            caller_spans.append((a, b))
            field_spans.append((a, b, path))
        if tag and b > a:
            if tag in spans:
                spans[tag][1] = b
            else:
                spans[tag] = [a, b]
    untrusted_spans = [tuple(spans["state"])] if "state" in spans else []
    evidence = "state"
    if "state" not in spans or is_empty_state(request["state"]):
        alt = spans.get("reread_instr")
        if empty_state != "reread_instructions" or alt is None or alt[1] <= alt[0]:
            raise ValueError("state is empty (no evidence content); heads need at least one evidence token")
        spans["state"], evidence = alt, "reread_instructions"
    q = request["questions"][qid]  # (``q`` is reused as a char bound above)
    opts = canonical_options(q)
    perm = list(range(len(opts))) if option_order is None else list(option_order)
    if sorted(perm) != list(range(len(opts))):
        raise ValueError("option_order must be a permutation of range(n_options)")
    if q["type"] == "score" and perm != sorted(perm):
        raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
    coded = len(opts) <= len(table) or qid in words
    codes: list[str] | None = None
    if qid in words:  # v3w: option j forked with the ONE first answer token of its own id (unique by word_ids_ok)
        conts = [[g1_first_token(encode, opts[k][0])] for k in perm]
        cont_spans = [[(0, 1, ID_PATH)] for _ in perm]
    elif coded:  # F4: the answer is ONE code token after "ANSWER:" (" A", " B", ...)
        # the code SHOWN for canonical option j (identity order: table[j]; permuted readouts: its presentation slot)
        codes = question_codes(q, presentation_order(q, qid) if spec["permute"] else identity_order(q), table)
        conts = [list(encode(" " + codes[k], True)) for k in perm]
        tok = dict(zip(table, table_tokens))
        if getattr(encode, "is_rwkv7_g1", False) and conts != [[tok[codes[k]]] for k in perm]:
            raise ValueError("g1v2 code continuations are not the frozen single code tokens")
        cont_spans: list = [[] for _ in perm]
    else:  # fallback (k > codebook): multi-token " <option id>\n", as decide.backends.rwkv_g1_v2
        conts = [list(encode(" " + opts[k][0] + G1V2_TEMPLATE.terminator, True)) for k in perm]
        cont_spans = [[(0, len(c), ID_PATH)] for c in conts]
    return LayoutRow(
        qid=qid, qtype=q["type"], option_ids=[opts[k][0] for k in perm], ids=ids, state_span=tuple(spans["state"]),
        field_span=tuple(spans["field"]), option_spans=[tuple(spans[f"option:{k}"]) for k in perm], option_seqs=[],
        continuations=conts, layout=G1V2_LAYOUT,
        meta={"template": tpl.id, "perm": perm, "evidence": evidence, "untrusted_spans": untrusted_spans,
              "caller_spans": caller_spans, "field_spans": field_spans, "cont_spans": cont_spans,
              "option_seq_spans": [], "trusted_paths": declared_instruction_paths(request), "bos": 0, "coded": coded,
              **({"codes": [codes[k] for k in perm] if codes else None, "catalog_chars": catalog_chars,
                  "word_ids": qid in words} if tpl.id in G1_EVAL_READOUTS else {})},
    )


class ByteEncoder:
    """Test tokenizer: UTF-8 bytes shifted by 4 (0..3 reserved: pad, bos, eos, unk). Vocab 260."""

    vocab_size = 260
    pad_id = 0

    def identity(self) -> dict:
        return {"kind": "byte+4", "vocab_size": self.vocab_size}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        return [b + 4 for b in text.encode("utf-8")]

    def token_bytes(self, i: int) -> bytes:
        return bytes([i - 4])


def hf_segment_encoder(tokenizer: Any) -> EncodeFn:
    """Segment encoder for an HF tokenizer: trusted text verbatim, untrusted text via the
    fail-closed untrusted encoder (no control/special tokens can be produced)."""
    from decide.untrusted import UntrustedEncoder

    untrusted = UntrustedEncoder(tokenizer)

    def encode(text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return tokenizer.encode(text, add_special_tokens=False)
        return untrusted.encode(text)

    import hashlib
    import json

    vocab_sha = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items())).encode()).hexdigest()
    ident = {"kind": "hf", "name_or_path": getattr(tokenizer, "name_or_path", None), "vocab_sha256": vocab_sha,
             "untrusted": untrusted.identity()}
    encode.identity = lambda: ident  # type: ignore[attr-defined]
    return encode

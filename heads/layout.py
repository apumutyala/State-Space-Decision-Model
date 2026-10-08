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

from decide.prompt import SYSTEM_PROMPT, TEMPLATES, Template, _instr, continuation_segments
from decide.provenance import field_strings, instructions_path, state_path
from decide.schema import canonical_options, canonical_question_ids, render

LAYOUTS = ("ordered", "symmetric")
#: bump on default tagged_segments / option_segments / build_row token-accounting changes.
#: The opt-in intent16 rendering has its own version in the training lineage sidecar.
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


def _field(index: int, qid: str, q: dict[str, Any], target: bool, with_options: bool,
           rendered_option_ids: Sequence[str] | None = None) -> list[tuple]:
    ftag = "field" if target else ""
    seg: list[tuple] = [
        (f"FIELD {index}\nID: ", True, ftag, None),
        (qid, False, ftag, ID_PATH),
        (f"\nTYPE: {q['type']}\nINSTRUCTION: ", True, ftag, None),
        (render(_instr(qid, q)), False, ftag, _instr_path(qid, q)),
    ]
    if with_options:
        seg.append(("\nALLOWED OPTIONS:", True, ftag, None))
        opts = canonical_options(q)
        order = list(range(len(opts))) if rendered_option_ids is None else [
            [oid for oid, _ in opts].index(oid) for oid in rendered_option_ids]
        for j, option_index in enumerate(order, start=1):
            oid, desc = opts[option_index]
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            otag = f"option:{option_index}" if target else ""
            seg += [(f"\nOPTION {j}: ", True, ftag, None), (render(sem), False, otag, ID_PATH)]
    else:
        n = len(canonical_options(q))
        seg.append((f"\nALLOWED OPTIONS: {n} options, given separately and unordered.", True, ftag, None))
    seg.append(("\nEND FIELD", True, ftag, None))
    return seg


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


def _tagged_segments_paths(template: Template, request: dict[str, Any], qid: str, layout: str,
                           rendered_option_ids: Sequence[str] | None = None) -> list[tuple]:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    with_options = layout == "ordered"
    questions = request["questions"]
    order = canonical_question_ids(questions)
    q = questions[qid]
    if rendered_option_ids is not None:
        ids = [oid for oid, _ in canonical_options(q)]
        if q["type"] != "choice" or sorted(rendered_option_ids) != sorted(ids):
            raise ValueError("training rendered_option_ids must permute the full choice domain")
    seg: list[tuple] = [(template.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", True, "", None)]
    for i, k in enumerate(order, start=1):
        if i > 1:
            seg.append(("\n", True, "", None))
        seg += _field(i, k, questions[k], k == qid, with_options,
                      rendered_option_ids if k == qid else None)
    rendered_ids = rendered_option_ids if rendered_option_ids is not None else [o for o, _ in canonical_options(q)]
    ids_text = ", ".join(rendered_ids) if with_options else "(the separately given options)"
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
    rendered_option_ids: Sequence[str] | None = None,
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

    ``rendered_option_ids`` is an explicit training-only permutation of the target
    choice's displayed options. IDs, the canonical option axis and their span
    alignment remain unchanged. Default serving and evaluation rendering is canonical.
    """
    tpl = TEMPLATES[template] if isinstance(template, str) else template
    # Training-only intent views may permute rendering while preserving option IDs
    # and the canonical option axis (including literal injection references).
    segs4 = _tagged_segments_paths(tpl, request, qid, layout, rendered_option_ids)
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


class ByteEncoder:
    """Test tokenizer: UTF-8 bytes shifted by 4 (0..3 reserved: pad, bos, eos, unk). Vocab 260."""

    vocab_size = 260
    pad_id = 0

    def identity(self) -> dict:
        return {"kind": "byte+4", "vocab_size": self.vocab_size}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        return [b + 4 for b in text.encode("utf-8")]


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

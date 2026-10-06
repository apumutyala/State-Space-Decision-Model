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
from decide.schema import canonical_options, canonical_question_ids, render

LAYOUTS = ("ordered", "symmetric")
#: bump on any change to tagged_segments / option_segments / build_row token accounting
LAYOUT_VERSION = "m2-layout-v2"  # v2: empty-state rows use the re-read instruction as evidence

#: encode(text, trusted) -> token ids.
EncodeFn = Callable[[str, bool], list[int]]

# (text, trusted, tag). Tags: "", "field", "option:<j>", "state", "reread", "reread_instr".
TaggedSegment = tuple[str, bool, str]


def _field(index: int, qid: str, q: dict[str, Any], target: bool, with_options: bool) -> list[TaggedSegment]:
    ftag = "field" if target else ""
    seg: list[TaggedSegment] = [
        (f"FIELD {index}\nID: ", True, ftag),
        (qid, False, ftag),
        (f"\nTYPE: {q['type']}\nINSTRUCTION: ", True, ftag),
        (render(_instr(qid, q)), False, ftag),
    ]
    if with_options:
        seg.append(("\nALLOWED OPTIONS:", True, ftag))
        for j, (oid, desc) in enumerate(canonical_options(q), start=1):
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            otag = f"option:{j - 1}" if target else ""
            seg += [(f"\nOPTION {j}: ", True, ftag), (render(sem), False, otag)]
    else:
        n = len(canonical_options(q))
        seg.append((f"\nALLOWED OPTIONS: {n} options, given separately and unordered.", True, ftag))
    seg.append(("\nEND FIELD", True, ftag))
    return seg


def tagged_segments(template: Template, request: dict[str, Any], qid: str, layout: str) -> list[TaggedSegment]:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    with_options = layout == "ordered"
    questions = request["questions"]
    order = canonical_question_ids(questions)
    q = questions[qid]
    seg: list[TaggedSegment] = [(template.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", True, "")]
    for i, k in enumerate(order, start=1):
        if i > 1:
            seg.append(("\n", True, ""))
        seg += _field(i, k, questions[k], k == qid, with_options)
    ids_text = ", ".join(o for o, _ in canonical_options(q)) if with_options else "(the separately given options)"
    seg += [
        ("\nEND SCHEMA\n\nSTATE:\n", True, ""),
        (render(request["state"]), False, "state"),
        (f"\nEND STATE\n\nQUESTION (re-read FIELD {order.index(qid) + 1}):\nID: ", True, "reread"),
        (qid, False, "reread"),
        (f"\nTYPE: {q['type']}\nINSTRUCTION: ", True, "reread"),
        (render(_instr(qid, q)), False, "reread_instr"),
        ("\nALLOWED OPTION IDS: ", True, "reread"),
        (ids_text, not with_options, "reread"),
        ("\nReply with exactly one allowed option id." + template.tail, True, "reread"),
    ]
    return [s for s in seg if s[0]]


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
    tpl = TEMPLATES[template] if isinstance(template, str) else template
    segs = tagged_segments(tpl, request, qid, layout)
    q = request["questions"][qid]
    opts = canonical_options(q)
    ids: list[int] = []
    spans: dict[str, list[int]] = {}
    for text, trusted, tag in segs:
        start = len(ids)
        ids += encode(text, trusted)
        if tag:
            if tag in spans:
                spans[tag][1] = len(ids)
            else:
                spans[tag] = [start, len(ids)]
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
        meta={"template": tpl.id, "perm": perm, "evidence": evidence},
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

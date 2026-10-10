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

``sym-fork`` / ``sym-fork-ids`` (``heads.symmetric``)
    Reorder AND relabel invariant by construction: the prefix holds state + question + the option COUNT, each option
    is its own forked continuation rendering only its description (``sym-fork``; the id never enters the model), the
    choice axis is content-canonical. Proof and caveats: docs/PROOF_INVARIANCE.md. ``sym-fork-ids`` is the ablation that
    also renders the id (reorder-equivariant, not relabel-invariant).

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
from decide.prompt_g1v3 import (CATALOG_CHARS_LADDER, G1V2F_ID, G1V3W_ID, build_prompt_g1v3, load_v3_codes,
                                 render_g1v3, wants_catalog, word_ids_ok)
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


def _v2_segments(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None):
    return build_prompt_g1v2(request, qid, table)


def _v2_render(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None) -> str:
    return render_g1v2(request, qid, table)


def _v2f_segments(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None):
    return build_prompt_g1v3(request, qid, table, None, False, True)  # v2f: no code key, no permutation, fixed instr


def _v2f_render(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None) -> str:
    return render_g1v3(request, qid, table, None, False, True)


def _v3w_segments(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None):
    """v3w = the readout G1V3Backend(mode="words"): v3 code table, code key at the label width the readout's budget
    ladder picked (``ctx["catalog_chars"]``), word-id questions (``ctx["words"]``) listed as "- id: desc"."""
    return build_prompt_g1v3(request, qid, table, ctx["catalog_chars"], False, True, ctx["words"])


def _v3w_render(request: dict[str, Any], qid: str, table: list[str], ctx: dict | None = None) -> str:
    return render_g1v3(request, qid, table, ctx["catalog_chars"], False, True, ctx["words"])


#: layout "g1v2" training templates -> the READOUT's own (segments, render) functions. Every entry uses v2's head/tail,
#: and its readout reads the next token after "ANSWER:" at the last prompt position.
#:   jrt-rwkv-g1-v2   decide.prompt.build_prompt_g1v2 / render_g1v2            (readout rwkv7-hf-g1-v2, job-zhyvg)
#:   jrt-rwkv-g1-v2f  decide.prompt_g1v3 v2f = v2 + the dict-instruction fix   (readout rwkv7-hf-g1-v2f, G1V3Backend v2f)
#:   jrt-rwkv-g1-v3w  decide.prompt_g1v3 v3w = v3 code table + code key + word ids (readout rwkv7-hf-g1-v3w,
#:                    G1V3Backend mode "words"); the target of a word-id question is the first token of " <id>"
G1_TRAIN_RENDERERS = {G1V2_ID: (_v2_segments, _v2_render), G1V2F_ID: (_v2f_segments, _v2f_render),
                      G1V3W_ID: (_v3w_segments, _v3w_render)}
G1_TRAIN_TEMPLATES = {G1V2_ID: G1V2_TEMPLATE,
                      **{t: Template(id=t, head=G1V2_TEMPLATE.head, tail=G1V2_TEMPLATE.tail,
                                     terminator=G1V2_TEMPLATE.terminator) for t in (G1V2F_ID, G1V3W_ID)}}
#: the v3w readout's "ANSWER:" context (decide.backends.rwkv_g1_v3.ANSWER_CTX; first-token and code proofs)
V3W_ANSWER_CTX = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"


def v3w_budget() -> int:
    """The v3w readout's token budget for the catalog-label ladder: G1V3Backend's default
    min(plugin.max_context, DEFAULT_LIMITS max_input_tokens) - 1 with the RWKV7-G1k plugin's max_context."""
    from decide.adapter import DEFAULT_LIMITS
    from decide.backends.hf_readout import RWKV7_G1K_TRAINED_CTX

    return min(RWKV7_G1K_TRAINED_CTX, DEFAULT_LIMITS["max_input_tokens"]) - 1


def resolve_template(template: Template | str) -> Template:
    """A template by id; the layout-``g1v2`` templates (``G1_TRAIN_TEMPLATES``: jrt-rwkv-g1-v2, jrt-rwkv-g1-v2f) are
    resolved here, not registered in ``decide.prompt.TEMPLATES`` (their identities are their readouts' renderer
    identities, ``heads.decide_backend._template_identity``)."""
    if isinstance(template, Template):
        return template
    from decide.shim_format import is_shim_template
    from decide.v3w_generic import is_generic_template

    if is_shim_template(template):  # the H2O shim format (decide.shim_format): its rows carry the whole prompt
        from decide.shim_format import template_obj

        return template_obj(template)
    if is_generic_template(template):  # generic v3w (decide.v3w_generic): the bound spec's chat format
        from decide.backends.v3w_generic import spec_template
        from decide.v3w_generic import spec_for_template

        return spec_template(spec_for_template(template))
    return G1_TRAIN_TEMPLATES.get(template) or TEMPLATES[template]


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
        if template.id == G1V3W_ID:
            raise ValueError(f"template {G1V3W_ID!r} needs the tokenizer (word ids, budget ladder): use build_row")
        return _g1v2_segments_paths(request, qid, _g1v2_code_table(), template.id)
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
    from decide.shim_format import is_shim_template
    from decide.v3w_generic import is_generic_template

    from .symmetric import SYM_LAYOUTS

    if layout in SYM_LAYOUTS:  # reorder + relabel invariant by construction (docs/PROOF_INVARIANCE.md)
        from .symmetric import build_row_sym

        return build_row_sym(request, qid, encode, tpl, layout, option_order, empty_state)
    if is_shim_template(tpl):  # the trainable H2O shim format (decide.shim_format): ids == the shim's prompt ids
        if layout != G1V2_LAYOUT:
            raise ValueError(f"template {tpl.id} needs layout {G1V2_LAYOUT!r}, got {layout!r}")
        from decide.shim_format import spec_for_template as shim_spec

        return shim_spec(tpl.id).row(request, qid, option_order, empty_state)
    if is_generic_template(tpl):  # a trained generic-v3w cell: the spec's rows (ids == its readout's prompt ids)
        if layout != G1V2_LAYOUT:
            raise ValueError(f"template {tpl.id} needs layout {G1V2_LAYOUT!r}, got {layout!r}")
        from decide.v3w_generic import spec_for_template

        return build_row_v3w_generic(request, qid, spec_for_template(tpl.id), option_order, empty_state)
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


_V3_TABLE: list[str] | None = None
_V3_TOKENS: list[int] | None = None


def _v3_code_table() -> list[str]:
    """The v3 code table (decide/data/rwkv_g1_v3_codes.json, sha-checked), as the v3w readout uses it."""
    global _V3_TABLE, _V3_TOKENS
    if _V3_TABLE is None:
        entries, _ = load_v3_codes()
        _V3_TABLE, _V3_TOKENS = [e["code"] for e in entries], [e["token_id"] for e in entries]
    return _V3_TABLE


def v3w_first_token(encode: EncodeFn, oid: str) -> int | None:
    """G1V3Backend.first_token: the token right after "ANSWER:" in the whole-string tokenization of
    ANSWER_CTX + " " + oid (None if that tokenization does not extend the context's, or the token is id 0)."""
    base = list(encode(V3W_ANSWER_CTX, True))
    ids = list(encode(V3W_ANSWER_CTX + " " + oid, True))
    n = len(base)
    return ids[n] if len(ids) > n and ids[:n] == base and ids[n] != 0 else None


def v3w_context(request: dict[str, Any], qid: str, encode: EncodeFn, table: list[str],
                budget: int | None = None) -> dict:
    """The readout's per-request choices for template v3w (G1V3Backend.word_qids + ._render): the word-id questions
    (non-opaque ids whose first answer tokens are unique) and, for a coded catalog question, the longest label width
    on CATALOG_CHARS_LADDER whose [0] + whole-string prompt fits the budget (None = no code key)."""
    memo: dict[str, int | None] = {}

    def first(o: str) -> int | None:
        if o not in memo:
            memo[o] = v3w_first_token(encode, o)
        return memo[o]

    qs = request["questions"]
    words = frozenset(k for k in canonical_question_ids(qs) if word_ids_ok(qs[k], first))
    budget = v3w_budget() if budget is None else budget
    n_used = None
    if qid not in words and wants_catalog(qs[qid], table):
        for n in CATALOG_CHARS_LADDER:
            n_used = n
            if 1 + len(encode(render_g1v3(request, qid, table, n, False, True, words), True)) <= budget:
                break
    return {"words": words, "catalog_chars": n_used, "first_token": first}


def _g1v2_segments_paths(request: dict[str, Any], qid: str, code_table: list[str],
                         template_id: str = G1V2_ID, ctx: dict | None = None) -> list[tuple]:
    """``decide.prompt.build_prompt_g1v2`` (the v2 READOUT's own segments, reused, not re-implemented) tagged as
    (text, trusted, tag, path). Tags: "field" (target field in the schema), "option:<j>" (its option j id +
    description), "state", "reread", "reread_instr"; paths as the ordered layout's (provenance / ID_PATH / None)."""
    questions = request["questions"]
    order = canonical_question_ids(questions)
    out: list[tuple] = []
    field: str | None = None  # the field being walked (schema), qid in the re-read
    phase, opt_j, expect = "schema", -1, None  # expect: the kind of the next data segment
    if template_id not in G1_TRAIN_RENDERERS:
        raise ValueError(f"layout {G1V2_LAYOUT!r} needs one of the templates {sorted(G1_TRAIN_RENDERERS)}, got {template_id!r}")
    for text, label in G1_TRAIN_RENDERERS[template_id][0](request, qid, code_table, ctx):
        target = phase == "schema" and field == qid
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
                       f"option:{opt_j}" if target and text == ": " else
                       "field" if target and text != "\n" else "")
            out.append((text, True, tag, None))
            continue
        if phase == "reread" and text.startswith("\nQUESTION AGAIN: "):  # v3* code key (one data segment)
            # restated instruction + one "<code> = <label>" line per option: request text, gated as data (ID_PATH)
            out.append((text, False, "reread", ID_PATH))
        elif expect == "state":
            out.append((text, False, "state", state_path()))
        elif expect == "qid":
            out.append((text, False, "reread" if phase == "reread" else ("field" if target else ""), ID_PATH))
        elif expect == "instr":
            out.append((text, False, "reread_instr" if phase == "reread" else ("field" if target else ""),
                        _instr_path(field, questions[field])))
        elif expect in ("oid", "desc"):
            out.append((text, False, "reread" if phase == "reread" else (f"option:{opt_j}" if target else ""),
                        ID_PATH))
        else:
            raise ValueError(f"g1v2 layout: unexpected request segment {text[:40]!r} in the v2 prompt")
    return out


def build_row_v3w_generic(request: dict[str, Any], qid: str, spec: Any, option_order: Sequence[int] | None = None,
                          empty_state: str = "reread_instructions") -> LayoutRow:
    """Layout-g1v2 v3w row for any backbone (``spec`` = decide.v3w_generic.V3W): the v3w segments with the spec's
    chat-format head / tail, ``spec.fmt.bos`` first, the spec's whole-string tokenizer, code table and word-id
    first tokens; ids == spec.questions(...)'s prompt ids (scripts/check_v3w_generic.py)."""
    return _build_row_g1v2(request, qid, spec.encode, resolve_template(G1V3W_ID), option_order, empty_state, spec)


#: lean v3w (A-PREREG-SIZECLIFF lean template): the schema block opens with this text right after the chat head
LEAN_SCHEMA_OPEN = "SCHEMA FIELDS:\n"
LEAN_REREAD_PREFIX = "\nEND STATE\nQUESTION (re-read FIELD "


def _opens_option(text: str) -> bool:
    """A template segment that starts option line j (the same rule ``_g1v2_segments_paths`` walks the schema with)."""
    return text.startswith("\n") and (text.endswith(". ") or text == "\n- ")


def lean_segments(segs: list[tuple]) -> list[tuple]:
    """The LEAN v3w layout from the full v3w segments (after the chat-format swap): the SCHEMA block is dropped (every
    field; the asked field is the re-read below), the state comes first, then the asked question ONCE ("QUESTION:",
    no re-read field number, no question id), its options and the reply line, unchanged. The re-read block takes over
    the spans the schema field carried: "field" (the question block), "option:<j>" (option j's id + description),
    and "reread_instr" (the instruction; empty-state evidence fallback). Same text pieces, same trust and paths."""
    t0 = segs[0][0]
    if not t0.endswith(LEAN_SCHEMA_OPEN):
        raise ValueError("lean v3w: the first segment does not open the schema block")
    try:
        i_end = next(i for i, s in enumerate(segs) if s[1] and s[0] == "\nEND SCHEMA\nSTATE:\n")
        rest = segs[i_end + 1:]
        j = next(i for i, s in enumerate(rest) if s[1] and s[0].startswith(LEAN_REREAD_PREFIX))
    except StopIteration:
        raise ValueError("lean v3w: no schema end / re-read header in the v3w segments") from None
    if not rest[j][0].endswith("\nID: ") or j + 1 >= len(rest) or rest[j + 1][1] or rest[j + 1][3] != ID_PATH:
        raise ValueError("lean v3w: the re-read header is not followed by the question id")
    out = [(t0[: len(t0) - len(LEAN_SCHEMA_OPEN)] + "STATE:\n",) + tuple(segs[0][1:])]
    out += rest[:j]  # the state (data, its own path)
    out.append(("\nEND STATE\nQUESTION:", True, "field", None))
    opt = -1
    for text, trusted, tag, path in rest[j + 2:]:
        if trusted:
            if _opens_option(text):
                opt, new = opt + 1, "field"
            elif text.startswith("\nReply with exactly one option"):
                new = "reread"
            elif opt >= 0 and text == ": ":
                new = f"option:{opt}"
            else:
                new = "field"
        elif tag == "reread_instr" or text.startswith("\nQUESTION AGAIN: "):
            new = tag
        else:
            new = f"option:{opt}" if opt >= 0 else "field"
        out.append((text, trusted, new, path))
    return out


def lean_v3w_render(request: dict[str, Any], qid: str, spec: Any, catalog_chars: int | None,
                    words: frozenset[str]) -> str:
    """The lean v3w prompt text for ``spec`` (``decide.v3w_generic.V3W`` with ``lean``): the lean segments joined, the
    user body's blank lines collapsed as every v3w render (``g1_clean``)."""
    from decide.prompt import g1_clean

    ctx = {"words": words, "catalog_chars": catalog_chars, "first_token": spec.first_token}
    g1_head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
    head, tail = spec.fmt.head_text(), spec.fmt.tail
    segs = lean_segments(_swap_format(_g1v2_segments_paths(request, qid, spec.table, G1V3W_ID, ctx), g1_head,
                                      G1V2_TEMPLATE.tail, head, tail))
    raw = "".join(t for t, *_ in segs)
    return head + g1_clean(raw[len(head):len(raw) - len(tail)]) + tail


def _swap_format(segs: list[tuple], head_old: str, tail_old: str, head_new: str, tail_new: str) -> list[tuple]:
    """Replace the G1 head (first segment's prefix) and tail (last segment's suffix) by another chat format's."""
    t0, t1 = segs[0][0], segs[-1][0]
    if not t0.startswith(head_old) or not t1.endswith(tail_old) or len(segs) < 2:
        raise ValueError("v3w segments do not start with the G1 head and end with the G1 tail")
    out = list(segs)
    out[0] = (head_new + t0[len(head_old):],) + tuple(segs[0][1:])
    out[-1] = (t1[: len(t1) - len(tail_old)] + tail_new,) + tuple(segs[-1][1:])
    return out


def _build_row_g1v2(request: dict[str, Any], qid: str, encode: EncodeFn, tpl: Template,
                    option_order: Sequence[int] | None, empty_state: str, spec: Any = None) -> LayoutRow:
    from decide.untrusted import ControlTokenError

    if tpl.id not in G1_TRAIN_RENDERERS:
        raise ValueError(f"layout {G1V2_LAYOUT!r} needs one of the templates {sorted(G1_TRAIN_RENDERERS)}, got {tpl.id!r}")
    token_bytes = getattr(encode, "token_bytes", None)
    byte_starts = getattr(spec, "byte_starts", None) if spec is not None else None
    if not callable(token_bytes) and not callable(byte_starts):
        raise ValueError(f"layout {G1V2_LAYOUT!r} needs an encoder with token_bytes(id) (whole-string spans)")
    v3w = tpl.id == G1V3W_ID
    g1_head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
    if spec is not None:  # generic v3w (decide.v3w_generic): the spec's table, word ids, budget ladder and format
        table = spec.table
        words = spec.word_qids(request)
        _, _, n_used = spec.choose(request, qid, words)
        ctx = {"words": words, "catalog_chars": n_used, "first_token": spec.first_token}
        head, tail, bos = spec.fmt.head_text(), spec.fmt.tail, list(spec.fmt.bos)
        segs = _swap_format(_g1v2_segments_paths(request, qid, table, tpl.id, ctx), g1_head, G1V2_TEMPLATE.tail,
                            head, tail)
        if getattr(spec, "lean", False):  # lean v3w: schema dropped, the asked question once (lean_segments)
            segs = lean_segments(segs)
    else:
        table = _v3_code_table() if v3w else _g1v2_code_table()
        ctx = v3w_context(request, qid, encode, table) if v3w else None
        head, tail, bos = g1_head, G1V2_TEMPLATE.tail, [0]
        segs = _g1v2_segments_paths(request, qid, table, tpl.id, ctx)
    raw = "".join(t for t, *_ in segs)
    bounds, n = [], 0
    for t, *_ in segs:
        bounds.append((n, n + len(t)))
        n += len(t)
    # blank-line collapse over the user body (render_g1v2): deleted chars shift every later offset
    body_lo, body_hi = len(head), len(raw) - len(tail)
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
    want = (spec.render(request, qid, ctx["catalog_chars"], ctx["words"]) if spec is not None
            else G1_TRAIN_RENDERERS[tpl.id][1](request, qid, table, ctx))
    if text != want:
        raise ValueError(f"g1v2 layout text differs from the {tpl.id} readout's render")
    if spec is not None:
        from decide.backends.base import BackendUnsupported

        try:
            body_ids = spec.body_ids(text)  # whole string; control ids only where the chat format puts them
        except BackendUnsupported as e:
            raise ControlTokenError(str(e)) from e
    else:
        body_ids = list(encode(text, True))  # F5: the whole prompt string at once
        if 0 in body_ids:
            raise ControlTokenError("request text encoded to the control id 0 (<|endoftext|>); refused")
    if callable(byte_starts):
        tok_start = byte_starts(text, body_ids)
    else:
        tok_start, pos = [], 0  # token byte offsets in the body
        for t in body_ids:
            tok_start.append(pos)
            pos += len(token_bytes(t))
        if pos != len(text.encode("utf-8")):
            raise ValueError("token bytes do not tile the prompt (encoder / token_bytes mismatch)")
    byte_at = [0]
    for ch in text:
        byte_at.append(byte_at[-1] + len(ch.encode("utf-8")))
    nb = len(bos)

    def span(c0: int, c1: int) -> tuple[int, int]:
        """clean char range -> half-open token range (straddling tokens included), + nb for the leading ids (F2)"""
        b0, b1 = byte_at[c0], byte_at[c1]
        if b1 <= b0:
            t = nb + bisect.bisect_left(tok_start, b0)
            return (t, t)
        return (nb - 1 + bisect.bisect_right(tok_start, b0), nb + bisect.bisect_left(tok_start, b1))

    ids = bos + body_ids
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
    q = request["questions"][qid]
    opts = canonical_options(q)
    perm = list(range(len(opts))) if option_order is None else list(option_order)
    if sorted(perm) != list(range(len(opts))):
        raise ValueError("option_order must be a permutation of range(n_options)")
    if q["type"] == "score" and perm != sorted(perm):
        raise ValueError("score levels are ordinal; their order is part of the question and cannot be permuted")
    words = v3w and qid in ctx["words"]
    coded = words or len(opts) <= len(table)
    if words:  # v3w word ids: the answer is the (unique) first token of " <id>" after "ANSWER:", the readout's read
        conts = [[ctx["first_token"](opts[k][0])] for k in perm]
        cont_spans: list = [[(0, 1, ID_PATH)] for _ in perm]
    elif coded:  # F4: the answer is ONE code token after "ANSWER:" (" A", " B", ...)
        if spec is not None:  # the code token proved one token after the format's answer prefix
            conts = [[spec.code_token[table[k]]] for k in perm]
        else:
            conts = [list(encode(" " + table[k], True)) for k in perm]
        frozen = _V3_TOKENS if v3w else _CODE_TOKENS
        if spec is None and getattr(encode, "is_rwkv7_g1", False) and conts != [[frozen[k]] for k in perm]:
            raise ValueError("g1v2 code continuations are not the frozen single code tokens")
        cont_spans = [[] for _ in perm]
    else:  # fallback (k > codebook): multi-token " <option id>\n", as decide.backends.rwkv_g1_v2
        conts = [list(encode(" " + opts[k][0] + G1V2_TEMPLATE.terminator, True)) for k in perm]
        cont_spans = [[(0, len(c), ID_PATH)] for c in conts]
    return LayoutRow(
        qid=qid, qtype=q["type"], option_ids=[opts[k][0] for k in perm], ids=ids, state_span=tuple(spans["state"]),
        field_span=tuple(spans["field"]), option_spans=[tuple(spans[f"option:{k}"]) for k in perm], option_seqs=[],
        continuations=conts, layout=G1V2_LAYOUT,
        meta={"template": tpl.id if spec is None else spec.template_id, "perm": perm, "evidence": evidence,
              "untrusted_spans": untrusted_spans,
              "caller_spans": caller_spans, "field_spans": field_spans, "cont_spans": cont_spans,
              "option_seq_spans": [], "trusted_paths": declared_instruction_paths(request),
              "bos": bos[0] if len(bos) == 1 else (list(bos) or None), "coded": coded,
              "words": words, "catalog_chars": ctx["catalog_chars"] if v3w else None},
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


class HFSegmentEncoder:
    """Segment encoder for an HF tokenizer: trusted text verbatim (``tokenizer.encode``, no specials added), untrusted
    text via the fail-closed untrusted encoder (no control/special tokens can be produced).

    PICKLABLE (job-rtg9e: the closure this replaced could not cross ``train.run_cell.cell_items``'s spawn
    ProcessPoolExecutor): the pickled state is the tokenizer's source directory (``name_or_path``, the snapshot the
    parent loaded it from) plus the encoder identity; a worker reloads the SAME files with
    ``transformers.AutoTokenizer.from_pretrained`` and refuses (ValueError) unless its identity (vocab sha256, untrusted
    encoder identity, source) equals the parent's. Same tokenizer class, same calls: outputs are identical."""

    def __init__(self, tokenizer: Any, source: str | None = None) -> None:
        self._setup(tokenizer, source if source is not None else getattr(tokenizer, "name_or_path", None))

    def _setup(self, tokenizer: Any, source: str | None) -> None:
        import hashlib
        import json

        from decide.untrusted import UntrustedEncoder

        self.tokenizer, self.source = tokenizer, source
        self.untrusted = UntrustedEncoder(tokenizer)
        vocab_sha = hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items())).encode()).hexdigest()
        self._ident = {"kind": "hf", "name_or_path": getattr(tokenizer, "name_or_path", None),
                       "vocab_sha256": vocab_sha, "untrusted": self.untrusted.identity()}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        if trusted:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.untrusted.encode(text)

    def identity(self) -> dict[str, Any]:
        return self._ident

    def __getstate__(self) -> dict[str, Any]:
        if not self.source:
            raise TypeError("HFSegmentEncoder: no tokenizer source directory to reload from in a worker")
        return {"source": str(self.source), "ident": self._ident}

    def __setstate__(self, state: dict[str, Any]) -> None:
        from transformers import AutoTokenizer

        self._setup(AutoTokenizer.from_pretrained(state["source"]), state["source"])
        if self._ident != state["ident"]:
            raise ValueError(f"HFSegmentEncoder: the tokenizer reloaded from {state['source']} has another identity "
                             "than the parent's (files changed?)")


def hf_segment_encoder(tokenizer: Any) -> EncodeFn:
    """Segment encoder for an HF tokenizer (``HFSegmentEncoder``: picklable, so it crosses process pools)."""
    return HFSegmentEncoder(tokenizer)

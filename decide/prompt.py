"""JRT readout prompt: schema + options -> state -> question re-read.

Order (JRT, arXiv 2407.05483): a causal/recurrent reader must see the schema
and options *before* the state, so the state is read knowing what will be
asked, then the target question is re-read right before the answer. This is
the opposite of Clef's state-first order, which is fine for Clef only because
its attention head can look back over the whole sequence.

Questions and options are rendered in canonical order (questions sorted by id;
choice options sorted by id; noul true,false; score by level), so the model
input is byte-identical under any reordering of keys in the request. That is
an adapter property; it says nothing about the model's own order robustness.

Prompts are built as (text, trusted) segments. Trusted = template and fixed
scaffolding. Untrusted = everything from the request (state, question ids,
instructions, option ids and descriptions); backends must encode untrusted
segments so they cannot produce control/special tokens (role-token injection,
interface fixture 7). Plain-text spoofing (a state containing ``END STATE`` or
``Assistant:``) cannot be prevented by any text template; only control-token
spoofing is prevented.
"""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass
from typing import Any

from .provenance import criteria_path, instructions_path, split, state_path
from .schema import canonical_options, canonical_question_ids, render, sha256_text

SYSTEM_PROMPT = (
    "Read the schema, then the state, then the question. Answer with exactly one of "
    "the question's allowed option ids."
)
BODY_VERSION = "jrt-body-v2-segmented"

Segment = tuple[str, bool]  # (text, trusted)


@dataclass(frozen=True)
class Template:
    id: str
    head: str  # before the user body; may contain {system}
    tail: str  # after the body, ends right before the answer continuation
    terminator: str  # appended to every option continuation

    def identity(self) -> dict[str, str]:
        # A rendered canary is hashed in, so any edit to the body builder changes the identity.
        text = "\x1f".join(
            [self.id, self.head, self.tail, self.terminator, SYSTEM_PROMPT, BODY_VERSION, canary_sha256(self)]
        )
        return {"id": self.id, "sha256": sha256_text(text)}


TEMPLATES = {
    # Qwen3.5 chat format, thinking disabled (empty think block), as Clef's suffix.
    "jrt-chatml-v1": Template(
        id="jrt-chatml-v1",
        head="<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
        tail="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nANSWER:",
        terminator="<|im_end|>",
    ),
    # RWKV7 G1 chat format.
    "jrt-rwkv-g1-v1": Template(
        id="jrt-rwkv-g1-v1",
        head="System: {system}\n\nUser: ",
        tail="\n\nAssistant: <think>\n</think>\nANSWER:",
        terminator="\n\n",
    ),
    # Base (non-chat) LMs, e.g. Mamba-3 SISO pretrained on FineWeb-Edu.
    "jrt-plain-v1": Template(
        id="jrt-plain-v1",
        head="{system}\n\n",
        tail="\nANSWER:",
        terminator="\n",
    ),
}


def _instr(qid: str, q: dict[str, Any]) -> str:
    instr = q.get("instructions")
    return qid if instr is None or instr == "" else instr


Labelled = tuple[str, str]  # (text, label); label in {"template", "data", "instruction"}


def _escaped(piece: str) -> str:
    # json.dumps escapes per code point, so escaping pieces and concatenating equals
    # escaping the whole string (ensure_ascii=False, as decide.schema.render).
    return json.dumps(piece, ensure_ascii=False)[1:-1]


def _text_field(value: str, name: str, prov: dict[str, Any] | None) -> list[Labelled]:
    """A request string field; split by provenance when supplied, else one data segment."""
    if prov is None or name not in prov["segments"]:
        return [(value, "data")]
    return [(t, trust) for t, trust in split(value, prov["segments"][name])]


def _option_segments(qid: str, oid: str, desc: Any, prov: dict[str, Any] | None) -> list[Labelled]:
    sem = {"option_id": oid}
    if desc is not None:
        sem["description"] = desc
    name = criteria_path(qid, oid)
    if prov is None or not isinstance(desc, str) or name not in prov["segments"]:
        return [(render(sem), "data")]
    # render(sem) == '{"description":"<escaped desc>","option_id":<json id>}' (sorted keys)
    out: list[Labelled] = [('{"description":"', "template")]
    out += [(_escaped(t), trust) for t, trust in split(desc, prov["segments"][name])]
    out += [('","option_id":', "template"), (json.dumps(oid, ensure_ascii=False), "data"), ("}", "template")]
    return out


def _instr_segments(qid: str, q: dict[str, Any], prov: dict[str, Any] | None) -> list[Labelled]:
    instr = q.get("instructions")
    if isinstance(instr, str) and instr != "":
        return _text_field(instr, instructions_path(qid), prov)
    return [(render(_instr(qid, q)), "data")]


def _field_labelled(index: int, qid: str, q: dict[str, Any], prov: dict[str, Any] | None) -> list[Labelled]:
    seg: list[Labelled] = [(f"FIELD {index}\nID: ", "template"), (qid, "data"),
                           (f"\nTYPE: {q['type']}\nINSTRUCTION: ", "template")]
    seg += _instr_segments(qid, q, prov)
    seg.append(("\nALLOWED OPTIONS:", "template"))
    for j, (oid, desc) in enumerate(canonical_options(q), start=1):
        seg.append((f"\nOPTION {j}: ", "template"))
        seg += _option_segments(qid, oid, desc, prov)
    seg.append(("\nEND FIELD", "template"))
    return seg


def build_prompt_labelled(template: Template, request: dict[str, Any], qid: str,
                          prov: dict[str, Any] | None = None) -> list[Labelled]:
    """Prompt for one question (each question is its own row, as Kev does for hybrids).

    ``prov`` is the *validated* provenance (``decide.provenance.validate_provenance``);
    without it every request-derived span is labelled ``data``, exactly as before."""
    questions = request["questions"]
    order = canonical_question_ids(questions)
    q = questions[qid]
    option_ids = ", ".join(o for o, _ in canonical_options(q))
    seg: list[Labelled] = [(template.head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", "template")]
    for i, k in enumerate(order, start=1):
        if i > 1:
            seg.append(("\n", "template"))
        seg += _field_labelled(i, k, questions[k], prov)
    seg.append(("\nEND SCHEMA\n\nSTATE:\n", "template"))
    if isinstance(request["state"], str):
        seg += _text_field(request["state"], state_path(), prov)
    else:
        seg.append((render(request["state"]), "data"))
    seg += [(f"\nEND STATE\n\nQUESTION (re-read FIELD {order.index(qid) + 1}):\nID: ", "template"), (qid, "data"),
            (f"\nTYPE: {q['type']}\nINSTRUCTION: ", "template")]
    seg += _instr_segments(qid, q, prov)
    seg += [("\nALLOWED OPTION IDS: ", "template"), (option_ids, "data"),
            ("\nReply with exactly one allowed option id." + template.tail, "template")]
    return [(t, lab) for t, lab in seg if t]


def build_prompt_segments(template: Template, request: dict[str, Any], qid: str) -> list[Segment]:
    """(text, trusted) view without provenance; trusted == adapter template text."""
    return [(t, lab == "template") for t, lab in build_prompt_labelled(template, request, qid)]


def build_prompt(template: Template, request: dict[str, Any], qid: str,
                 prov: dict[str, Any] | None = None) -> str:
    return "".join(text for text, _ in build_prompt_labelled(template, request, qid, prov))


def continuation_segments(template: Template, option_id: str) -> list[Segment]:
    return [(" " + option_id, False), (template.terminator, True)]


def continuation(template: Template, option_id: str) -> str:
    return " " + option_id + template.terminator


CANARY_REQUEST = {
    "model": "canary",
    "state": {"k": "v"},
    "questions": {
        "b": {"type": "choice", "instructions": "pick", "criteria": {"y": "Y", "x": None}},
        "a": {"type": "score", "criteria": ["lo", "hi"]},
        "c": {"type": "noul"},
    },
}


def canary_sha256(template: Template) -> str:
    parts = []
    for qid in ("a", "b", "c"):
        parts += [f"{t}\x1e{int(tr)}" for t, tr in build_prompt_segments(template, CANARY_REQUEST, qid)]
    parts += [f"{t}\x1e{int(tr)}" for t, tr in continuation_segments(template, "x")]
    return sha256_text("\x1f".join(parts))


# ----------------------------------------------------------------------------------------------------------------
# jrt-rwkv-g1-v2: the vendor-compliant RWKV7 G1 readout (C1, research-notes/RWKV7_GAP_LEVERS.md F1-F5). Opt-in only
# (decide.backends.rwkv_g1_v2 / readout family ``rwkv7-hf-g1-v2``); jrt-rwkv-g1-v1 stays the default everywhere.
#   F1  no "\n\n" inside the system or user turn: template text uses "\n", request text has \n{2,} collapsed to "\n"
#       (BlinkDL: "\n\n" is the chat-round separator in G1 pretraining)
#   F2  token 0 first (BlinkDL rwkv_mmlu_eval.py: [0] + encode(prompt)); added by the backend, not here
#   F3  G1k no-think generation prefix "Assistant: <think></think>\n" (chat_template.jinja), then "ANSWER:"
#   F4  every option is listed as "<CODE>. <option id>: <description>" with a single-token code (A..Z, AA.. from the
#       frozen filtered codebook); the answer is the one code token. Questions with more options than the codebook
#       fall back to "- <option id>: <description>" lines and multi-token option-id scoring
#   F5  the whole prompt string is tokenized at once (the backend), not segment by segment
# The JRT order (schema + options -> state -> question re-read) is kept; the re-read now repeats the coded options.
# ----------------------------------------------------------------------------------------------------------------

G1V2_ID = "jrt-rwkv-g1-v2"
G1V2_SYSTEM_PROMPT = (
    "Read the schema, then the state, then the question. Answer with the code of exactly one of the "
    "question's options."
)
G1V2_TEMPLATE = Template(
    id=G1V2_ID,
    head="System: {system}\n\nUser: ",
    tail="\n\nAssistant: <think></think>\nANSWER:",
    terminator="\n",  # fallback (multi-token id) continuations only
)
_G1_BLANK_LINES = _re.compile(r"\n(?:[ \t\r\f\v]*\n)+")


def g1_clean(text: str) -> str:
    """Collapse every run of blank lines to a single newline (vendor rule: no "\n\n" inside a turn)."""
    return _G1_BLANK_LINES.sub("\n", text)


def _g1_text(value: Any) -> str:
    return g1_clean(value if isinstance(value, str) else render(value))


def g1v2_option_lines(q: dict[str, Any], codes: list[str] | None) -> list[Labelled]:
    seg: list[Labelled] = []
    for j, (oid, desc) in enumerate(canonical_options(q)):
        seg.append(("\n" + (f"{codes[j]}. " if codes is not None else "- "), "template"))
        seg.append((g1_clean(oid), "data"))
        if desc is not None:
            seg += [(": ", "template"), (_g1_text(desc), "data")]
    return seg


def build_prompt_g1v2(request: dict[str, Any], qid: str, code_table: list[str]) -> list[Labelled]:
    """jrt-rwkv-g1-v2 prompt for one question as (text, label) segments (labels are informational: the backend
    tokenizes the joined string once and fails closed on control ids). A field whose option count exceeds the code
    table is rendered without codes (fallback)."""
    questions = request["questions"]
    order = canonical_question_ids(questions)

    def codes_for(k: str) -> list[str] | None:
        n = len(canonical_options(questions[k]))
        return code_table[:n] if n <= len(code_table) else None

    def instr(k: str) -> list[Labelled]:
        v = questions[k].get("instructions")
        return [(_g1_text(v) if isinstance(v, str) and v != "" else g1_clean(k), "data")]

    seg: list[Labelled] = [(G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT) + "SCHEMA FIELDS:\n", "template")]
    for i, k in enumerate(order, start=1):
        if i > 1:
            seg.append(("\n", "template"))
        seg += [(f"FIELD {i}\nID: ", "template"), (g1_clean(k), "data"),
                (f"\nTYPE: {questions[k]['type']}\nINSTRUCTION: ", "template")]
        seg += instr(k)
        seg.append(("\nOPTIONS:", "template"))
        seg += g1v2_option_lines(questions[k], codes_for(k))
        seg.append(("\nEND FIELD", "template"))
    seg.append(("\nEND SCHEMA\nSTATE:\n", "template"))
    state = request["state"]
    seg.append((_g1_text(state), "data"))
    q = questions[qid]
    coded = codes_for(qid) is not None
    seg += [(f"\nEND STATE\nQUESTION (re-read FIELD {order.index(qid) + 1}):\nID: ", "template"), (g1_clean(qid), "data"),
            (f"\nTYPE: {q['type']}\nINSTRUCTION: ", "template")]
    seg += instr(qid)
    seg.append(("\nOPTIONS:", "template"))
    seg += g1v2_option_lines(q, codes_for(qid))
    seg.append(("\nReply with exactly one option " + ("code." if coded else "id.") + G1V2_TEMPLATE.tail, "template"))
    return [(t, lab) for t, lab in seg if t]


def render_g1v2(request: dict[str, Any], qid: str, code_table: list[str]) -> str:
    """The jrt-rwkv-g1-v2 prompt string. Blank lines are collapsed once more over the joined user turn, so a request
    string that ends or starts with a newline cannot form "\n\n" with the adjacent template newline."""
    text = "".join(t for t, _ in build_prompt_g1v2(request, qid, code_table))
    head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT)
    body = text[len(head):len(text) - len(G1V2_TEMPLATE.tail)]
    return head + g1_clean(body) + G1V2_TEMPLATE.tail

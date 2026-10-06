"""JRT readout prompt: schema + options -> state -> question re-read.

Order (JRT, arXiv 2407.05483): a causal/recurrent reader must see the schema
and options *before* the state, so the state is read knowing what will be
asked, then the target question is re-read right before the answer. This is
the opposite of Clef's state-first order, which is fine for Clef only because
its attention head can look back over the whole sequence.

Questions and options are rendered in canonical order (questions sorted by id;
choice options sorted by id; noul true,false; score by level), so the prompt,
and therefore the readout, is invariant to the order of keys in the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import canonical_options, canonical_question_ids, render, sha256_text

SYSTEM_PROMPT = (
    "Read the schema, then the state, then the question. Answer with exactly one of "
    "the question's allowed option ids."
)


@dataclass(frozen=True)
class Template:
    id: str
    head: str  # before the user body; may contain {system}
    tail: str  # after the body, ends right before the answer continuation
    terminator: str  # appended to every option continuation

    def identity(self) -> dict[str, str]:
        text = "\x1f".join([self.id, self.head, self.tail, self.terminator, SYSTEM_PROMPT, BODY_VERSION])
        return {"id": self.id, "sha256": sha256_text(text)}


BODY_VERSION = "jrt-body-v1"

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


def _field_block(index: int, qid: str, q: dict[str, Any]) -> str:
    instr = q.get("instructions")
    if instr is None or instr == "":
        instr = qid
    lines = [f"FIELD {index}", f"ID: {qid}", f"TYPE: {q['type']}", f"INSTRUCTION: {render(instr)}", "ALLOWED OPTIONS:"]
    for j, (oid, desc) in enumerate(canonical_options(q), start=1):
        sem = {"option_id": oid}
        if desc is not None:
            sem["description"] = desc
        lines.append(f"OPTION {j}: {render(sem)}")
    lines.append("END FIELD")
    return "\n".join(lines)


def build_prompt(template: Template, request: dict[str, Any], qid: str) -> str:
    """Prompt for one question (each question is its own row, as Kev does for hybrids)."""
    questions = request["questions"]
    order = canonical_question_ids(questions)
    schema = "\n".join(_field_block(i, k, questions[k]) for i, k in enumerate(order, start=1))
    q = questions[qid]
    instr = q.get("instructions")
    if instr is None or instr == "":
        instr = qid
    option_ids = ", ".join(o for o, _ in canonical_options(q))
    body = (
        f"SCHEMA FIELDS:\n{schema}\nEND SCHEMA\n\n"
        f"STATE:\n{render(request['state'])}\nEND STATE\n\n"
        f"QUESTION (re-read FIELD {order.index(qid) + 1}):\n"
        f"ID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: {render(instr)}\n"
        f"ALLOWED OPTION IDS: {option_ids}\n"
        "Reply with exactly one allowed option id."
    )
    return template.head.format(system=SYSTEM_PROMPT) + body + template.tail


def continuation(template: Template, option_id: str) -> str:
    return " " + option_id + template.terminator

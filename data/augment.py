"""Schema augmentation: base item -> SystemOne training record, with the variation the Clef recipe names
("permutating field orders, prompts, and schema structures") and the shapes the Index uses.

Per record (all draws from a per-record seeded RNG, so a build is reproducible):
  * option order permuted (always), option ids relabelled: letters | digits | option_k | opaque random ids |
    the loader's semantic ids (only when they do not reveal the gold: ``opaque_ids``);
  * instruction paraphrase (>= 5 per family), question key name varied;
  * item placement: in ``state`` (dict with renamed, reordered keys, or a bare string) or inside the
    instructions (as the Index does for BANKING77/CLINC/MMLU-style rows), state then empty;
  * score scales sometimes reversed (label index mapped), score levels never permuted (ordinal);
  * retrieval items rendered either as one choice over up to 255 passages or as per-candidate yes/no
    questions (BRIGHT/ToolRet shape);
  * optional background padding (unrelated passages) for length variation; labels unchanged.
Multi-question records merge several rendered items into one state with one question per item.

Record (``RECORD_SCHEMA``)::

  {"id", "schema", "source", "source_split", "source_id", "license", "family",
   "state": str | dict, "questions": {qid: {"type", "instructions", "criteria"?}},
   "labels": {qid: option id | "true"/"false" | level index as str},
   "untrusted_fields": [state keys the host marks untrusted],
   "aug": {...what was applied...}}

An engine receives exactly ``state`` and ``questions``; ``labels`` and everything else are training-side only.
``_probe`` (the item-tier texts for the contamination guard, see dedup.py) is dropped before writing.
"""
from __future__ import annotations

import random
import string
from typing import Any

from . import RECORD_SCHEMA

INSTR = {
    "intent_banking": ["Classify the banking intent of this customer message.",
                       "Which support intent does the customer's message express?",
                       "Route this banking request to the matching intent.",
                       "Pick the intent label that best fits the customer's request.",
                       "What is the customer asking the bank about? Choose one intent.",
                       "Select the category this banking query belongs to."],
    "intent_general": ["Classify the intent of this user request; use the out-of-scope option if no intent applies.",
                       "Which intent does the user's utterance express (or is it out of scope)?",
                       "Map the request to one supported intent, or mark it out of scope.",
                       "Choose the assistant intent this command belongs to.",
                       "Identify the user's intent from the list; pick out of scope when nothing fits.",
                       "What does the user want? Choose the matching intent label."],
    "nli": ["Classify the relationship between the premise and the hypothesis.",
            "Does the premise entail, contradict, or say nothing decisive about the hypothesis?",
            "Given the premise, how does the hypothesis stand?",
            "Decide the inference relation from premise to hypothesis.",
            "Judge whether the hypothesis follows from, conflicts with, or is unrelated to the premise.",
            "Natural language inference: pick the relation."],
    "continuation": ["Which continuation is most plausible?", "Choose the most likely next part of the description.",
                     "Pick the ending that best continues the context.", "What most plausibly happens next?",
                     "Select the sentence ending that fits the scene.", "Which option completes the context best?"],
    "fill_blank": ["Which option correctly fills the blank (_)?", "Choose the name or phrase that belongs in the gap.",
                   "Resolve the blank in the sentence.", "Who or what does the _ refer to?",
                   "Fill in the missing word using common sense.", "Pick the option that makes the sentence sensible."],
    "math_mcq": ["Choose the numeric answer to the problem. Do not show work.", "What is the final answer?",
                 "Solve the word problem and select its result.", "Which number answers the question?",
                 "Pick the correct value for the problem.", "Select the answer to this arithmetic word problem."],
    "mcq": ["Answer the multiple-choice question.", "Choose the best answer.", "Which option is correct?",
            "Select the correct answer to the question.", "Pick the right choice.",
            "Read the question and choose the most accurate option."],
    "sentiment_score": ["How positive is the sentiment of this review?", "Rate the reviewer's sentiment.",
                        "Place the review on the sentiment scale.", "What overall sentiment does the text convey?",
                        "Score the opinion expressed, from the scale given.", "Judge how favourable the review is."],
    "yesno_passage": ["Using only the passage, is the answer to the question yes?",
                      "Based on the passage, answer the yes/no question.",
                      "Does the passage support answering yes?", "Read the passage: is the statement in the question true?",
                      "Answer from the passage: yes or no?", "According to the text, is the answer yes?"],
    "response_select": ["Which response should the assistant give, given the available tools?",
                        "Choose the most appropriate assistant reply to the user's question.",
                        "Given the tool list, pick the right way to respond.",
                        "Which reply correctly handles this request with the tools on hand?",
                        "Select the assistant's best next message.", "How should the assistant respond?"],
    "tool_select": ["Which tool should be called next for this dialogue?",
                    "Pick the single function to invoke for the user's latest request.",
                    "Choose the tool that serves the request (or none).",
                    "Select the API call the assistant should make now.",
                    "Given the published tools, which one should be invoked?",
                    "Which listed function answers the latest user turn?"],
    "tool_multi": ["Should this tool be invoked to answer the latest user request?",
                   "Is this function needed for the user's latest turn?",
                   "Mark yes if the assistant should call this tool now.",
                   "Does serving the request require calling this tool?",
                   "Tool selection: is this tool part of the correct call set?"],
    "retrieval": ["Which passage answers the query?", "Select the most relevant passage for the query.",
                  "Rank-1 retrieval: pick the passage that best answers the search query.",
                  "Which candidate document is relevant to the question?",
                  "Choose the passage containing the answer.", "Find the passage that answers the query."],
    "retrieval_yesno": ["Is this candidate passage relevant to the query?", "Does this passage answer the query?",
                        "Relevance judgement: is the candidate useful for the query?",
                        "Mark yes if the passage answers the search query.", "Is the document relevant?"],
    # added for the 150k build (arts area: Humicroedit)
    "humor_score": ["How funny is the edited headline?", "Rate the humour of the one-word headline edit.",
                    "Place the edited headline on the funniness scale.",
                    "Judge how funny the edit makes the headline.",
                    "Score the edited headline for humour, using the scale given.",
                    "How amusing is the headline after the edit?"],
    "humor_pair": ["Which edited version of the headline is funnier?",
                   "Pick the funnier one-word edit of the headline.",
                   "Two edits of the same headline: which one is more humorous?",
                   "Choose the edited headline most readers would find funnier.",
                   "Which headline edit gets the bigger laugh?", "Select the funnier edit."],
}

FIELD_SYNONYMS = {
    "customer_message": ["customer_message", "message", "query", "utterance", "request_text"],
    "user_request": ["user_request", "utterance", "command", "text", "input"],
    "premise": ["premise", "context", "evidence", "statement_a"],
    "hypothesis": ["hypothesis", "claim", "statement_b", "candidate_statement"],
    "activity": ["activity", "topic", "activity_label"],
    "context": ["context", "scene", "story_so_far", "prefix"],
    "sentence": ["sentence", "text", "item"],
    "problem": ["problem", "question", "word_problem"],
    "question": ["question", "prompt", "item"],
    "review": ["review", "text", "comment"],
    "passage": ["passage", "document", "evidence", "source_text"],
    "available_tools": ["available_tools", "tools", "functions"],
    "user_question": ["user_question", "question", "user_turn"],
    "tools": ["tools", "functions", "catalog"],
    "dialogue": ["dialogue", "conversation", "messages"],
    "query": ["query", "search_query", "question"],
    "original_headline": ["original_headline", "headline", "original"],
    "edited_headline": ["edited_headline", "edited", "new_headline"],
    "headline": ["headline", "original_headline", "title"],
}

QKEYS = ["q1", "answer", "label", "decision", "choice", "output", "field_1", "result"]
ID_STYLES = ["letters", "digits", "option_k", "opaque", "semantic"]


def _letters(k: int) -> str:
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = string.ascii_uppercase[r] + s
    return s


def relabel(ids: list[str], style: str, rng: random.Random) -> list[str]:
    n = len(ids)
    if style == "letters":
        return [_letters(i) for i in range(n)]
    if style == "digits":
        start = rng.choice([0, 1])
        return [str(i + start) for i in range(n)]
    if style == "option_k":
        return [f"option_{i}" for i in range(n)]
    if style == "opaque":
        out, seen = [], set()
        while len(out) < n:
            t = "k_" + "".join(rng.choice("0123456789abcdef") for _ in range(8))
            if t not in seen:
                seen.add(t); out.append(t)
        return out
    return list(ids)


def _render_fields(fields: dict, rng: random.Random) -> tuple[dict, dict]:
    """Rename keys (synonyms) and shuffle their order; returns (state dict, rename map)."""
    keys = list(fields)
    if rng.random() < 0.5:
        rng.shuffle(keys)
    out, ren, used = {}, {}, set()
    for k in keys:
        syn = [s for s in FIELD_SYNONYMS.get(k, [k]) if s not in used] or [k]
        nk = rng.choice(syn)
        used.add(nk)
        out[nk] = fields[k]
        ren[k] = nk
    return out, ren


def _fields_as_text(fields: dict) -> str:
    from .text import canonical_json
    lines = []
    for k, v in fields.items():
        v = v if isinstance(v, str) else canonical_json(v)
        lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")
    return "\n".join(lines)


def _choice_question(base: dict, rng: random.Random, instr: str) -> tuple[dict, str, dict]:
    ids = list(base["options"])
    rng.shuffle(ids)
    semantic_ok = not base.get("opaque_ids") and all(0 < len(i) <= 64 for i in ids)
    styles = [s for s in ID_STYLES if s != "semantic" or semantic_ok]
    style = rng.choice(styles)
    new = relabel(ids, style, rng)
    crit = {n: base["options"][o] for n, o in zip(new, ids)}
    gold = new[ids.index(base["gold"])]
    return {"type": "choice", "instructions": instr, "criteria": crit}, gold, {"id_style": style}


def _score_question(base: dict, rng: random.Random, instr: str) -> tuple[dict, str, dict]:
    levels = list(base["options"])
    g = int(base["gold"])
    rev = rng.random() < 0.3
    if rev:
        levels = levels[::-1]
        g = len(levels) - 1 - g
    return {"type": "score", "instructions": instr, "criteria": levels}, str(g), {"reversed_scale": rev}


def render(base: dict, rng: random.Random, pad_pool: list[str] | None = None, p_pad: float = 0.1,
           placement: str | None = None) -> dict:
    fam = base["family"]
    aug: dict[str, Any] = {}
    instr = rng.choice(INSTR[fam]) if fam in INSTR else "Answer the question."
    aug["instr_variant"] = INSTR.get(fam, []).index(instr) if instr in INSTR.get(fam, []) else -1
    long_item = fam in ("retrieval", "tool_select", "tool_multi", "response_select") or \
        sum(len(str(v)) for v in base["fields"].values()) > 3000
    placement = placement or ("state" if long_item or rng.random() < 0.6 else "instructions")
    aug["placement"] = placement
    if placement == "state":
        st, ren = _render_fields(base["fields"], rng)
        if len(st) == 1 and isinstance(next(iter(st.values())), str) and rng.random() < 0.3:
            state: Any = next(iter(st.values()))
            aug["state_shape"] = "string"
        else:
            state = st
            aug["state_shape"] = "dict"
    else:
        st, ren = _render_fields(base["fields"], rng)
        instr = instr + "\n" + _fields_as_text(st)
        state = rng.choice([{}, ""])
        aug["state_shape"] = "empty"
    questions, labels = {}, {}
    if base["qtype"] == "choice" and fam == "retrieval" and len(base["options"]) <= 32 and rng.random() < 0.3:
        # BRIGHT/ToolRet shape: one yes/no question per candidate
        yn = rng.choice([("yes", "no"), ("relevant", "not_relevant"), ("1", "0")])
        y_instr = rng.choice(INSTR["retrieval_yesno"])
        ids = list(base["options"])
        rng.shuffle(ids)
        for k, o in enumerate(ids):
            qid = f"doc_{k}"
            questions[qid] = {"type": "choice", "instructions": {"candidate": base["options"][o], "task": y_instr},
                              "criteria": {yn[0]: "Relevant to the query.", yn[1]: "Not relevant to the query."}}
            labels[qid] = yn[0] if o == base["gold"] else yn[1]
        aug["shape"] = "per_candidate_yesno"
    elif base["qtype"] == "multi_yesno":
        yn = rng.choice([("yes", "no"), ("call", "skip"), ("true", "false")])
        names = list(base["options"])
        rng.shuffle(names)
        for k, n in enumerate(names):
            qid = f"tool_{k}"  # positional, collision-free (names can share long prefixes); the name is in the text
            questions[qid] = {"type": "choice", "instructions": f"{instr}\nTool: {n}",
                              "criteria": {yn[0]: "Invoke this tool.", yn[1]: "Do not invoke this tool."}}
            labels[qid] = yn[0] if n in base["gold"] else yn[1]
        if len(questions) != len(names):
            raise AssertionError(f"{base['id']}: {len(names)} tools but {len(questions)} questions")
        aug["shape"] = "per_tool_yesno"
    else:
        qid = rng.choice(QKEYS)
        if base["qtype"] == "choice":
            q, gold, a = _choice_question(base, rng, instr)
        elif base["qtype"] == "score":
            q, gold, a = _score_question(base, rng, instr)
        elif base["qtype"] == "noul":
            q, gold, a = {"type": "noul", "instructions": instr}, base["gold"], {}
        else:
            raise ValueError(base["qtype"])
        aug.update(a)
        questions[qid], labels[qid] = q, gold
        aug["shape"] = "single"
    if pad_pool and rng.random() < p_pad and isinstance(state, (dict, str)):
        k = rng.choice([1, 2, 4, 8, 16])
        pads = [rng.choice(pad_pool) for _ in range(k)]
        state = {**(state if isinstance(state, dict) else {"content": state} if state else {}),
                 "background_reading": pads}
        aug["padding_passages"] = k
    from .text import strings_in
    probe = [t for f in base.get("item_fields") or list(base["fields"]) for t in strings_in(base["fields"].get(f))]
    io = base.get("item_options", "all")
    if base["qtype"] in ("choice", "multi_yesno") and isinstance(base["options"], dict):
        if io == "all":
            probe += [t for v in base["options"].values() for t in strings_in(v)]
        elif io == "gold":
            probe += list(strings_in(base["options"][base["gold"]]))
    rec = {"id": base["id"], "schema": RECORD_SCHEMA, "source": base["source"], "source_split": base["source_split"],
           "source_id": base["source_id"], "license": base["license"], "family": fam, "state": state,
           "questions": questions, "labels": labels, "untrusted_fields": list(base.get("untrusted_fields") or []),
           "aug": aug, "_probe": probe}
    return rec


def reshuffle_options(rec: dict, rng: random.Random) -> dict:
    """Permute and relabel the options of every choice question of an already-rendered record."""
    questions, labels, styles = {}, {}, {}
    for qid, q in rec["questions"].items():
        if q["type"] != "choice":
            questions[qid], labels[qid] = q, rec["labels"][qid]
            continue
        base = {"options": q["criteria"], "gold": rec["labels"][qid]}
        nq, gold, a = _choice_question(base, rng, q["instructions"])
        questions[qid], labels[qid], styles[qid] = nq, gold, a["id_style"]
    return {**rec, "questions": questions, "labels": labels, "aug": {**rec.get("aug", {}), "id_styles": styles}}


def merge(recs: list[dict], rng: random.Random) -> dict:
    """Several single-question records -> one multi-question record (one state entry per item)."""
    prefix = rng.choice(["item", "case", "record", "ticket"])
    state, questions, labels, parts = {}, {}, {}, []
    for k, r in enumerate(recs, 1):
        key = f"{prefix}_{k}"
        state[key] = r["state"]
        for qid, q in r["questions"].items():
            nq = dict(q)
            if isinstance(q.get("instructions"), str):
                nq["instructions"] = f"For {key}: {q['instructions']}"
            questions[f"{key}_{qid}"] = nq
            labels[f"{key}_{qid}"] = r["labels"][qid]
        parts.append(r["id"])
    srcs = sorted({r["source"] for r in recs})
    return {"id": "m2:multi:" + "+".join(parts), "schema": RECORD_SCHEMA, "source": srcs[0] if len(srcs) == 1 else "mixed",
            "source_split": "train", "source_id": "+".join(r["source_id"] for r in recs),
            "license": "; ".join(sorted({r["license"] for r in recs})), "family": "multi",
            "state": state, "questions": questions, "labels": labels,
            "untrusted_fields": [], "aug": {"multi": len(recs), "parts": parts, "sources": srcs},
            "_probe": [t for r in recs for t in r.get("_probe", [])]}


def validate(rec: dict) -> None:
    """Structural checks mirroring decide/schema.py: types, 2..255 options, labels present and valid."""
    assert rec["questions"], rec["id"]
    assert set(rec["labels"]) == set(rec["questions"]), rec["id"]
    for qid, q in rec["questions"].items():
        t = q["type"]
        lab = rec["labels"][qid]
        if t == "choice":
            assert isinstance(q["criteria"], dict) and 2 <= len(q["criteria"]) <= 255, (rec["id"], qid)
            assert lab in q["criteria"], (rec["id"], qid)
            for oid in q["criteria"]:
                assert oid and "<|" not in oid and "|>" not in oid and len(oid) <= 64, (rec["id"], oid)
        elif t == "score":
            assert isinstance(q["criteria"], list) and len(q["criteria"]) >= 2, (rec["id"], qid)
            assert 0 <= int(lab) < len(q["criteria"]), (rec["id"], qid)
        elif t == "noul":
            assert lab in ("true", "false"), (rec["id"], qid)
        else:
            raise AssertionError(f"{rec['id']}: unknown type {t}")
    if rec.get("untrusted_fields"):
        assert isinstance(rec["state"], dict) and all(f in rec["state"] for f in rec["untrusted_fields"]), rec["id"]

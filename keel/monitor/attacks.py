"""Evaluation-only model-authored comments with AST-preserving insertion.

The attack prose comes from the monitored generator, not distillation templates.
AST identity is an initial screen only: final variants still need execution.
"""
from __future__ import annotations

import ast
import re
from dataclasses import replace

from keel.card.search import Candidate, wrap_attack

from .data import FAMILIES, Submission


def comments_disjoint(comments: dict[str, str]) -> bool:
    from keel.distill.triplets import TEMPLATES
    def grams(text):
        words = re.findall(r"\w+", text.lower())
        return {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
    forbidden = set().union(*(grams(text) for values in TEMPLATES.values() for text in values))
    return not any(grams(text) & forbidden for text in comments.values())


def insert_comment(code: str, text: str, position: str) -> str:
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    payload = ["# " + line + "\n" for line in text.splitlines()]
    if not payload:
        payload = ["#\n"]
    if position == "start":
        output = payload + lines
    elif position == "end":
        output = lines + payload
    elif position == "mid":
        boundary = tree.body[len(tree.body) // 2].lineno - 1 if tree.body else 0
        # A decorator and the decorated statement are one indivisible unit.
        node = tree.body[len(tree.body) // 2] if tree.body else None
        if node is not None and getattr(node, "decorator_list", ()):
            boundary = min(x.lineno for x in node.decorator_list) - 1
        output = lines[:boundary] + payload + lines[boundary:]
    elif position == "split":
        middle = (len(payload) + 1) // 2
        output = payload[:middle] + lines + payload[middle:]
    else:
        raise ValueError(f"unknown position: {position}")
    result = "".join(output)
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(result), include_attributes=False):
        raise ValueError("comment insertion changed executable syntax")
    return result


def attacked(submission: Submission, candidate: Candidate) -> Submission:
    if candidate.template not in FAMILIES or candidate.target_phrasing != "literal":
        raise ValueError("candidate outside frozen model-authored comment space")
    if not comments_disjoint(submission.comments):
        raise ValueError("evaluation comments overlap distillation training templates")
    text = wrap_attack(submission.comments[candidate.template], candidate.wrapper)
    return replace(submission, code=insert_comment(submission.code, text, candidate.position))


def static_candidates() -> list[Candidate]:
    return [Candidate(name, "literal", "start", "plain") for name in FAMILIES]
